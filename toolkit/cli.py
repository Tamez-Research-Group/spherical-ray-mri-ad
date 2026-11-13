"""Batch use of the toolkit without the web interface.

    python toolkit/cli.py brain1.nii.gz brain2.nii.gz --model cnn --weights best \
        --out predictions.csv [--save-tensors DIR]

Writes one row per input and seed: P(Dementia), the seed's validation
threshold and the call. Research use only; the output is not a diagnosis.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np          # noqa: E402
import pandas as pd         # noqa: E402

import pipeline as P        # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", type=Path, help="skull-stripped T1 NIfTI files")
    ap.add_argument("--model", choices=list(P.MODELS), default="cnn")
    ap.add_argument("--weights", choices=list(P.WEIGHTS), default="best")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(P.SEEDS))
    ap.add_argument("--out", type=Path, default=Path("predictions.csv"))
    ap.add_argument("--save-tensors", type=Path, help="also save each spherical tensor (.npy) here")
    a = ap.parse_args()

    rows = []
    for f in a.inputs:
        try:
            subj = P.process(f)
        except Exception as e:
            print(f"{f}: FAILED ({e})")
            continue
        df = P.predict(subj.tensor, a.model, a.weights, a.seeds)
        s = P.summarize(df)
        print(f"{subj.name}: {s['call']} ({s['agree']}/{s['n']} seeds, mean P = {s['mean_p']:.3f})"
              + "".join(f"\n  warning: {w}" for w in subj.qc))
        rows.append(df.assign(input=subj.name, model=a.model, weights=a.weights, protocol=subj.protocol))
        if a.save_tensors:
            a.save_tensors.mkdir(parents=True, exist_ok=True)
            np.save(a.save_tensors / f"{subj.name}.npy", subj.tensor)
    if rows:
        out = pd.concat(rows)[["input", "model", "weights", "seed", "P(Dementia)", "threshold", "call", "protocol"]]
        out.to_csv(a.out, index=False)
        print(f"-> {a.out}")


if __name__ == "__main__":
    main()
