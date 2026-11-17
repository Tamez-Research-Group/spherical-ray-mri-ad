"""
evaluate_registration_fidelity.py — post hoc, exploratory: does the advantage of
the registration-free spherical CNN3D over registered-input CNN3D depend on how
well each brain was registered?

Registration fidelity = correlation of the SyN-warped brain with the MNI152
template within the template mask (registered_cache/registration_qc.csv,
Methods 2.9). It is lower in Dementia than in CN, so two analyses separate
fidelity from diagnosis:

1. Strata: test subjects in tertiles of SyN fidelity. Within each tertile, the
   seed-mean test AUC of each CNN3D input and the paired difference
   spherical - other (B = 2,000 resamples within the tertile, seed 47), plus the
   interaction: difference in the lowest minus the highest tertile.
2. Per subject: Spearman correlation between the relative advantage of the
   spherical input, m = (2y - 1) * (percentile rank of the spherical score -
   percentile rank of the other score), and the subject's fidelity expressed
   relative to its diagnosis x protocol group (residual), with a bootstrap CI.
   Negative = the spherical input gains where registration was worse than usual.

The native-volume CNN3D involves no registration and serves as a negative
control. Writes results/registration_fidelity.csv (aggregate results only).

    python evaluate_registration_fidelity.py
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from evaluate import CLASSES, LABEL_MAP, bootstrap_auc, deep_scores, percentile_ci, save_csv

REPO = Path(__file__).resolve().parent
SPH = "CNN3D (best epoch)"
OTHERS = {"SyN volume": "CNN3D, MNI SyN (best epoch)", "affine volume": "CNN3D, MNI affine (best epoch)",
          "native volume (control)": "CNN3D, native volume (best epoch)"}
N_BOOT, SEED = 2000, 47


def strat_indices(y, n_boot=N_BOOT, seed=SEED):
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(y), size=(n_boot, len(y)))
    return idx[[len(np.unique(y[i])) == 2 for i in idx]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=REPO / "results")
    args = ap.parse_args()

    ref = pd.read_csv(REPO / "cohort" / "manifest.csv")
    ref = ref[ref["Diagnosis"].isin(CLASSES)].copy()
    ref["label"] = ref["Diagnosis"].map(LABEL_MAP)
    ref = ref[[(REPO / "spherical_cache" / f"{p}.npy").exists() for p in ref["PTID"]]].reset_index(drop=True)
    val_df = ref[ref["split"] == "validation"].reset_index(drop=True)
    test_df = ref[ref["split"] == "test"].reset_index(drop=True)
    qc = pd.read_csv(REPO / "registered_cache" / "registration_qc.csv").set_index("PTID")
    test_df["fidelity"] = qc.loc[test_df["PTID"], "syn"].to_numpy()
    y = test_df["label"].to_numpy()
    scores = deep_scores(REPO / "runs", val_df, test_df)
    per_seed = {c: [s_test for _, s_test in scores[c].values()] for c in [SPH, *OTHERS.values()]}

    rows = []
    # 1. tertiles of SyN fidelity
    edges = np.quantile(test_df["fidelity"], [1 / 3, 2 / 3])
    tert = np.digitize(test_df["fidelity"], edges)            # 0 = lowest fidelity
    deltas = {}
    for t, name in enumerate(["lowest", "middle", "highest"]):
        sel = np.where(tert == t)[0]
        ys = y[sel]
        idx = strat_indices(ys, seed=SEED + t)
        point = {c: np.mean([bootstrap_auc(ys, s[sel], np.arange(len(sel))[None, :])[0] for s in ss])
                 for c, ss in per_seed.items()}
        boot = {c: np.mean([bootstrap_auc(ys, s[sel], idx) for s in ss], axis=0) for c, ss in per_seed.items()}
        row = {"analysis": "SyN-fidelity tertile", "stratum": name, "n": len(sel), "n_dementia": int(ys.sum()),
               "fidelity_min": float(test_df["fidelity"].iloc[sel].min()),
               "fidelity_max": float(test_df["fidelity"].iloc[sel].max()), "AUC_spherical": point[SPH]}
        for lab, c in OTHERS.items():
            d = boot[SPH] - boot[c]
            lo, hi = percentile_ci(d)
            row[f"AUC_{lab}"] = point[c]
            row[f"delta_vs_{lab}"], row[f"lo_vs_{lab}"], row[f"hi_vs_{lab}"] = point[SPH] - point[c], float(lo), float(hi)
            deltas[(name, lab)] = (point[SPH] - point[c], d)
        rows.append(row)
    # interaction: lowest minus highest tertile (independent resamples)
    row = {"analysis": "interaction", "stratum": "lowest minus highest tertile"}
    for lab in OTHERS:
        (p_lo, d_lo), (p_hi, d_hi) = deltas[("lowest", lab)], deltas[("highest", lab)]
        k = min(len(d_lo), len(d_hi))
        diff = d_lo[:k] - d_hi[:k]
        lo, hi = percentile_ci(diff)
        row[f"delta_vs_{lab}"], row[f"lo_vs_{lab}"], row[f"hi_vs_{lab}"] = p_lo - p_hi, float(lo), float(hi)
        row[f"p_vs_{lab}"] = float(min(1.0, 2 * min((diff <= 0).mean(), (diff >= 0).mean())))
    rows.append(row)

    # 2. per-subject relative advantage vs fidelity residual (within diagnosis x protocol)
    resid = test_df["fidelity"] - test_df.groupby(["Diagnosis", "protocol"])["fidelity"].transform("mean")
    pct = {c: rankdata(np.mean(ss, axis=0)) / len(y) for c, ss in per_seed.items()}
    rng = np.random.RandomState(SEED + 10)
    row = {"analysis": "per-subject Spearman", "stratum": "advantage vs fidelity residual", "n": len(y)}
    for lab, c in OTHERS.items():
        m = (2 * y - 1) * (pct[SPH] - pct[c])
        rho = spearmanr(m, resid).statistic
        bs = []
        for _ in range(N_BOOT):
            i = rng.randint(0, len(y), len(y))
            bs.append(spearmanr(m[i], resid.to_numpy()[i]).statistic)
        lo, hi = percentile_ci(np.array(bs))
        row[f"delta_vs_{lab}"], row[f"lo_vs_{lab}"], row[f"hi_vs_{lab}"] = float(rho), float(lo), float(hi)
        row[f"p_vs_{lab}"] = float(spearmanr(m, resid).pvalue)
    rows.append(row)

    out = pd.DataFrame(rows)
    save_csv(out, args.out_dir / "registration_fidelity.csv")
    pd.set_option("display.width", 250)
    print(out.round(3).T.to_string())


if __name__ == "__main__":
    main()
