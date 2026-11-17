"""
evaluate_equivalence.py — post hoc equivalence tests (TOST) for the deep-network
comparisons that the paper describes as comparable (Methods 2.8, Results 3.4).

No equivalence margin was pre-specified, so these tests are post hoc and are
reported as such. For each pair (a, b), delta = seed-mean test AUC(a) - AUC(b) is
resampled with exactly the paired bootstrap of evaluate.py (same test order,
same B = 2,000 subject-level resamples, seed 42), so the 95% intervals reproduce
results/registration_comparison.csv and registration_spherical_comparison.csv.

Two one-sided tests at alpha = 0.05 against a margin m: equivalence when the
90% bootstrap interval of delta lies inside (-m, +m); the TOST p-value is the
larger of the two one-sided bootstrap p-values, P*(delta <= -m) and
P*(delta >= +m). Holm adjustment over the pairs, per margin. The smallest margin
at which a pair is equivalent is max(|90% lower bound|, |90% upper bound|).

    python evaluate_equivalence.py [--margins 0.05 0.03]
Writes results/equivalence.csv (aggregate results only).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate import (CLASSES, LABEL_MAP, bootstrap_auc, bootstrap_indices, deep_scores, holm,
                      percentile_ci, save_csv)

REPO = Path(__file__).resolve().parent

# deep-network pairs the paper describes as comparable: (registration-free, reference)
PAIRS = [
    ("family (i)", "CNN3D (best epoch)", "CNN3D, MNI SyN (best epoch)"),
    ("family (i)", "CNN3D (best epoch)", "CNN3D, MNI affine (best epoch)"),
    ("family (ii)", "CNN3D (best epoch)", "CNN3D, spherical of MNI SyN (best epoch)"),
    ("family (ii)", "Spherical Transformer (best epoch)",
     "Spherical Transformer, spherical of MNI SyN (best epoch)"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=REPO / "cohort" / "manifest.csv")
    ap.add_argument("--cache-dir", type=Path, default=REPO / "spherical_cache")
    ap.add_argument("--runs-dir", type=Path, default=REPO / "runs")
    ap.add_argument("--out-dir", type=Path, default=REPO / "results")
    ap.add_argument("--margins", type=float, nargs="+", default=[0.05, 0.03])
    args = ap.parse_args()

    # the same subjects, in the same order, as evaluate.py
    ref = pd.read_csv(args.manifest)
    ref = ref[ref["Diagnosis"].isin(CLASSES)].copy()
    ref["label"] = ref["Diagnosis"].map(LABEL_MAP)
    ref = ref[[(args.cache_dir / f"{p}.npy").exists() for p in ref["PTID"]]].reset_index(drop=True)
    val_df = ref[ref["split"] == "validation"].reset_index(drop=True)
    test_df = ref[ref["split"] == "test"].reset_index(drop=True)
    y = test_df["label"].to_numpy()
    idx = bootstrap_indices(y)
    scores = deep_scores(args.runs_dir, val_df, test_df)

    def seed_mean(config):
        tests = [s_test for _, s_test in scores[config].values()]
        point = np.mean([bootstrap_auc(y, s, np.arange(len(y))[None, :])[0] for s in tests])
        return point, np.mean([bootstrap_auc(y, s, idx) for s in tests], axis=0)

    rows = []
    for family, a, b in PAIRS:
        if a not in scores or b not in scores:
            print(f"skipped (no runs): {a} vs {b}")
            continue
        (pa, ba), (pb, bb) = seed_mean(a), seed_mean(b)
        d = ba - bb
        lo95, hi95 = percentile_ci(d, 0.95)
        lo90, hi90 = percentile_ci(d, 0.90)
        row = {"family": family, "config_a": a, "config_b": b, "delta_AUC": pa - pb,
               "lo95": float(lo95), "hi95": float(hi95), "lo90": float(lo90), "hi90": float(hi90),
               "smallest_equivalence_margin": float(max(abs(lo90), abs(hi90))), "n_boot": len(d)}
        for m in args.margins:
            p = max((d <= -m).mean(), (d >= m).mean(), 1 / len(d))
            row[f"p_tost_{m:g}"] = float(p)
            row[f"equivalent_{m:g}"] = bool(lo90 > -m and hi90 < m)
        rows.append(row)
        print(f"{a} vs {b}: delta {pa - pb:+.4f}, 95% [{lo95:+.4f}, {hi95:+.4f}], "
              f"90% [{lo90:+.4f}, {hi90:+.4f}], smallest margin {row['smallest_equivalence_margin']:.4f}")

    out = pd.DataFrame(rows)
    for m in args.margins:
        out[f"p_tost_{m:g}_holm"] = holm(out[f"p_tost_{m:g}"])
    save_csv(out, args.out_dir / "equivalence.csv")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
