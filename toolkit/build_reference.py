"""Build toolkit/reference/adni_reference.npz: aggregate ADNI reference
statistics that the toolkit uses to compare a scan with the CN and Dementia
groups. Run once on the machine that holds the spherical-tensor cache:

    python toolkit/build_reference.py

Contents (aggregate statistics only, no participant-level data):
  mean_CN_<p>, sd_CN_<p>, mean_Dementia_<p>   training-partition mean and SD
                                               tensors per acquisition protocol p
  n_<class>_<p>                               group sizes
  d                                           protocol-adjusted Cohen's d
                                               (CN - Dementia), as in Fig. 4
  score_grid, score_density_CN, score_density_Dementia, score_auc
      the pattern-similarity score (pipeline.pattern_score) of the validation
      partition, summarised as smooth density curves (Gaussian KDE) and its AUC.
The test partition is not used.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]

import pipeline as P                                        # noqa: E402
from figures_spherical import cohen_d_protocol_adjusted     # noqa: E402
from evaluate import _manual_auc                            # noqa: E402

CACHE = P.REPO / "spherical_cache"


def main():
    man = pd.read_csv(P.REPO / "cohort" / "manifest.csv")
    out = {"d": cohen_d_protocol_adjusted(man).astype(np.float32)}
    tr = man[man["split"] == "train"]
    for (proto, dx), g in tr.groupby(["protocol", "Diagnosis"]):
        X = np.stack([np.load(CACHE / f"{p}.npy") for p in g["PTID"]]).astype(np.float64)
        out[f"mean_{dx}_{proto}"] = X.mean(0).astype(np.float32)
        out[f"n_{dx}_{proto}"] = np.int32(len(g))
        if dx == "CN":
            out[f"sd_CN_{proto}"] = X.std(0, ddof=1).astype(np.float32)
        print(f"train {proto:8s} {dx:8s} n={len(g)}")
    ref = P.Reference(out)
    va = man[man["split"] == "validation"]
    scores = np.array([ref.pattern_score(np.load(CACHE / f"{p}.npy"), pr)
                       for p, pr in zip(va["PTID"], va["protocol"])])
    y = (va["Diagnosis"] == "Dementia").to_numpy()
    grid = np.linspace(-0.6, 0.9, 301)
    out.update(score_grid=grid.astype(np.float32),
               score_density_CN=gaussian_kde(scores[~y])(grid).astype(np.float32),
               score_density_Dementia=gaussian_kde(scores[y])(grid).astype(np.float32),
               score_auc=np.float32(_manual_auc(y.astype(int), scores)),
               n_validation_CN=np.int32((~y).sum()), n_validation_Dementia=np.int32(y.sum()))
    print(f"validation: score AUC {out['score_auc']:.3f} (CN {(~y).sum()}, Dementia {y.sum()})")
    dst = HERE / "reference" / "adni_reference.npz"
    dst.parent.mkdir(exist_ok=True)
    tmp = dst.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **out)
    tmp.replace(dst)
    print(f"-> {dst} ({dst.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
