"""Test-set evaluation of the toolkit's similarity-to-dementia score.

The score (pipeline.Reference.pattern_score) correlates a scan's z-map against the
CN mean of its protocol with the protocol-adjusted Cohen's d map (CN - Dementia),
over cells with |d| >= 0.15 and |phi| <= 75 deg. The reference statistics come from
the training partition only (toolkit/build_reference.py), so the test partition is
untouched by it. Reports test AUC with the study's bootstrap (B = 2,000, seed 42),
AUC within each protocol, and the Spearman correlation with the CNN3D seed-mean
probability. Writes results/similarity_score.csv (aggregate results only).

    python toolkit/evaluate_similarity.py
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]

import numpy as np          # noqa: E402
import pandas as pd         # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

import pipeline as P        # noqa: E402
from evaluate import (CLASSES, LABEL_MAP, _manual_auc, bootstrap_auc, bootstrap_indices,  # noqa: E402
                      deep_scores, percentile_ci, save_csv)

REPO = HERE.parent


def main():
    ref_stats = P.Reference.load()
    man = pd.read_csv(REPO / "cohort" / "manifest.csv")
    man = man[man["Diagnosis"].isin(CLASSES)].copy()
    man["label"] = man["Diagnosis"].map(LABEL_MAP)
    man = man[[(REPO / "spherical_cache" / f"{p}.npy").exists() for p in man["PTID"]]].reset_index(drop=True)
    val_df = man[man["split"] == "validation"].reset_index(drop=True)
    test = man[man["split"] == "test"].reset_index(drop=True)
    y = test["label"].to_numpy()
    score = np.array([ref_stats.pattern_score(np.load(REPO / "spherical_cache" / f"{p}.npy"), pr)
                      for p, pr in zip(test["PTID"], test["protocol"])])
    rows = []
    idx = bootstrap_indices(y)
    lo, hi = percentile_ci(bootstrap_auc(y, score, idx))
    rows.append({"set": "test, all", "n": len(y), "n_dementia": int(y.sum()), "AUC": _manual_auc(y, score),
                 "AUC_lo": float(lo), "AUC_hi": float(hi),
                 "median_score_CN": float(np.median(score[y == 0])),
                 "median_score_Dementia": float(np.median(score[y == 1]))})
    for k, proto in enumerate(("slice1.2", "iso1.0")):
        sel = (test["protocol"] == proto).to_numpy()
        ii = bootstrap_indices(y[sel], seed=43 + 0 * k)
        lo, hi = percentile_ci(bootstrap_auc(y[sel], score[sel], ii))
        rows.append({"set": f"test, {proto}", "n": int(sel.sum()), "n_dementia": int(y[sel].sum()),
                     "AUC": _manual_auc(y[sel], score[sel]), "AUC_lo": float(lo), "AUC_hi": float(hi)})
    cnn = np.mean([s for _, s in deep_scores(REPO / "runs", val_df, test)["CNN3D (best epoch)"].values()], axis=0)
    rows.append({"set": "test, Spearman with CNN3D seed-mean probability", "n": len(y),
                 "AUC": float(spearmanr(score, cnn).statistic)})
    out = pd.DataFrame(rows)
    save_csv(out, REPO / "results" / "similarity_score.csv")
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
