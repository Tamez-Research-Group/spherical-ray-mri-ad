"""Validate the research toolkit against the study pipeline on the whole test partition.

For every test subject, the toolkit processes the skull-stripped NIfTI from scratch
(harmonization + spherical transform, toolkit/pipeline.py) and is compared with
  - the spherical tensor of the study cache (spherical_cache/<PTID>.npy), and
  - the test probabilities of every training run (runs/<Model>/seed<k>/predictions.csv),
    for both networks and both weight sets, using the weights published on Hugging Face.
It also checks that the per-seed calls at the validation thresholds are identical.

Per-subject progress is kept in results/_cache/ (gitignored, holds PTIDs) so a crash
resumes; only aggregate numbers are written to results/toolkit_validation.csv.

    python toolkit/validate_toolkit.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]

import numpy as np          # noqa: E402
import pandas as pd         # noqa: E402

import pipeline as P        # noqa: E402
from evaluate import save_csv  # noqa: E402

REPO = HERE.parent
RUN_DIRS = {"cnn": "CNN3D", "transformer": "Transformer"}


def main():
    man = pd.read_csv(REPO / "cohort" / "manifest.csv")
    test = man[man["split"] == "test"].reset_index(drop=True)
    preds = {(m, s): pd.read_csv(REPO / "runs" / d / f"seed{s}" / "predictions.csv").set_index("PTID")
             for m, d in RUN_DIRS.items() for s in P.SEEDS}
    part = REPO / "results" / "_cache" / "toolkit_validation_partial.json"
    part.parent.mkdir(parents=True, exist_ok=True)
    done = json.loads(part.read_text()) if part.exists() else {}

    for i, r in enumerate(test.itertuples()):
        if r.PTID in done:
            continue
        t0 = time.perf_counter()
        for attempt in range(5):           # N4 occasionally raises a spurious ITK error (see pipeline._n4)
            try:
                subj = P.process(r.brain_path)
                break
            except RuntimeError:
                if attempt == 4:
                    raise
        cached = np.load(REPO / "spherical_cache" / f"{r.PTID}.npy")
        rec = {"tensor_max_abs_diff": float(np.abs(cached - subj.tensor).max()),
               "seconds": time.perf_counter() - t0}
        for m in RUN_DIRS:
            for w in ("best", "avg"):
                df = P.predict(subj.tensor, m, w)
                ref = np.array([preds[(m, s)].loc[r.PTID, f"prob_{w}"] for s in P.SEEDS])
                thr = df["threshold"].to_numpy()
                rec[f"{m}_{w}_max_abs_dP"] = float(np.abs(df["P(Dementia)"].to_numpy() - ref).max())
                rec[f"{m}_{w}_calls_equal"] = bool(np.array_equal(df["P(Dementia)"].to_numpy() >= thr, ref >= thr))
        done[r.PTID] = rec
        tmp = part.with_suffix(".tmp")
        tmp.write_text(json.dumps(done))
        tmp.replace(part)
        print(f"{i + 1}/{len(test)}  tensor diff {rec['tensor_max_abs_diff']:.2e}  "
              f"max dP {max(v for k, v in rec.items() if k.endswith('dP')):.1e}  {rec['seconds']:.1f}s", flush=True)

    d = pd.DataFrame(done.values())
    rows = [{"check": "spherical tensor vs study cache", "n_subjects": len(d),
             "identical": int((d["tensor_max_abs_diff"] == 0).sum()),
             "max_abs_diff": d["tensor_max_abs_diff"].max(), "median_abs_diff": d["tensor_max_abs_diff"].median()}]
    for m in RUN_DIRS:
        for w in ("best", "avg"):
            rows.append({"check": f"P(Dementia) {m} {w}, 5 seeds vs training runs", "n_subjects": len(d),
                         "identical": int(d[f"{m}_{w}_calls_equal"].sum()),
                         "max_abs_diff": d[f"{m}_{w}_max_abs_dP"].max(),
                         "median_abs_diff": d[f"{m}_{w}_max_abs_dP"].median()})
    rows.append({"check": "processing time per subject (s), median", "n_subjects": len(d),
                 "identical": np.nan, "max_abs_diff": d["seconds"].max(), "median_abs_diff": d["seconds"].median()})
    out = pd.DataFrame(rows)
    save_csv(out, REPO / "results" / "toolkit_validation.csv")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
