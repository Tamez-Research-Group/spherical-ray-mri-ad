"""
evaluate_demographics.py — test AUC within demographic subgroups (sex, age
tertile) and the ADNI phase of each acquisition protocol (Methods Sections
2.6 and 2.8; Results Section 3.3; CLAIM items 13 and 39).

Sex, age and ADNI phase come from the ADNI IDA search export (one row per
image); for each subject the earliest record is used, as for Table 1.
Deep-network scores are read from runs/<Model>/seed<k>/predictions.csv
(best-epoch weights, test partition); the AUC is the seed mean, and its
95% CI comes from the bootstrap of the seed-mean within each subgroup
(B = 2000, random seed 45). Writes results/subgroup_demographics.csv and
results/protocol_by_phase.csv (counts only).

    python evaluate_demographics.py --ida /path/to/idaSearch_export.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate import N_BOOT, _manual_auc, bootstrap_auc, bootstrap_indices, percentile_ci, save_csv

REPO = Path(__file__).resolve().parent
MODELS = {"CNN3D": "CNN3D", "Spherical Transformer": "Transformer", "CNN3D, MNI SyN": "CNN3D_mni-syn",
          "CNN3D, native volume": "CNN3D_native-volume"}
SEEDS = [0, 1, 2, 3, 4]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ida", type=Path, required=True,
                    help="ADNI IDA image-search export (CSV with Subject ID, Sex, Age, Phase, Study Date)")
    ap.add_argument("--out-dir", type=Path, default=REPO / "results")
    args = ap.parse_args()

    man = pd.read_csv(REPO / "cohort" / "manifest.csv")
    ida = pd.read_csv(args.ida, dtype=str)
    ida["date"] = pd.to_datetime(ida["Study Date"], errors="coerce")
    first = ida.sort_values("date").groupby("Subject ID").first()
    age = ida.assign(Age=ida["Age"].astype(float)).groupby("Subject ID")["Age"].min()   # as in Table 1
    man = man.join(first[["Phase", "Sex"]], on="PTID").join(age, on="PTID")

    # protocol x ADNI phase x diagnosis (counts; whole cohort)
    tab = man.groupby(["protocol", "Phase", "Diagnosis"]).size().unstack(fill_value=0).reset_index()
    save_csv(tab, args.out_dir / "protocol_by_phase.csv")
    print(tab.to_string(index=False))

    test = man[man["split"] == "test"].reset_index(drop=True)
    test["label"] = (test["Diagnosis"] == "Dementia").astype(int)
    edges = np.quantile(test["Age"], [1 / 3, 2 / 3])
    test["age_group"] = pd.cut(test["Age"], [-np.inf, *edges, np.inf],
                               labels=[f"< {edges[0]:.1f} y", f"{edges[0]:.1f}-{edges[1]:.1f} y", f"> {edges[1]:.1f} y"])
    groups = [("Sex", "F"), ("Sex", "M")] + [("Age tertile", g) for g in test["age_group"].cat.categories]

    rows = []
    for name, run in MODELS.items():
        scores = []
        for s in SEEDS:
            p = pd.read_csv(REPO / "runs" / run / f"seed{s}" / "predictions.csv").set_index("PTID")
            scores.append(p.loc[test["PTID"], "prob_best"].to_numpy())
        for var, level in groups:
            col = "Sex" if var == "Sex" else "age_group"
            k = np.flatnonzero((test[col] == level).to_numpy())
            yk = test["label"].to_numpy()[k]
            idx = bootstrap_indices(yk, n_boot=N_BOOT, seed=45)
            aucs = [_manual_auc(yk, sc[k]) for sc in scores]
            lo, hi = percentile_ci(np.mean([bootstrap_auc(yk, sc[k], idx) for sc in scores], axis=0))
            rows.append({"model": name, "variable": var, "level": str(level), "n": len(k),
                         "n_dementia": int(yk.sum()), "AUC_mean": float(np.mean(aucs)),
                         "AUC_sd": float(np.std(aucs, ddof=1)), "AUC_lo": float(lo), "AUC_hi": float(hi)})
    out = pd.DataFrame(rows)
    save_csv(out, args.out_dir / "subgroup_demographics.csv")
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
