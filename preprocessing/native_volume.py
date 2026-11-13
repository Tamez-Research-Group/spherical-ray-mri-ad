"""
native_volume.py — registration-free Cartesian reference input: each
subject's brain on a fixed Cartesian grid, without any registration
(Methods, Section 2.9).

It separates the two things the spherical pipeline does differently from
the registration-based reference: it skips registration, and it resamples
the brain onto subject-centred spherical coordinates. This input skips
registration but keeps Cartesian coordinates, so

    spherical tensor  vs  native volume   isolates the representation,
    native volume     vs  MNI SyN volume  isolates registration.

For each subject listed in cohort/manifest.csv:
  1. Exactly the preprocessing of the spherical pipeline: RAS reorientation,
     N4 bias-field correction and resampling to 1.2 mm isotropic
     (preprocessing/harmonize.py), brain mask = vol > 0.
  2. Trilinear sampling onto a 2 mm grid of the same shape as the
     MNI-registered volumes (99, 117, 95), axis-aligned in RAS and centred on
     the subject's own centre of mass (Methods Eq. 2). No rotation, scaling
     or deformation; only the translation that puts the centre of mass in
     the middle of the grid.
  3. Intensity normalization as for the registered volumes: median/MAD
     within the (resampled) subject brain mask, clipped to +-8, 0 outside.
Saved as float32 (99, 117, 95) to <out>/<PTID>.npy. The fraction of brain
volume falling outside the grid is written per subject to <out>/_qc.csv.

Resumable (finished subjects are skipped) and crash-safe (atomic writes).

    python preprocessing/native_volume.py --workers 16
"""

from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from preprocessing.register import _atomic_bytes, _atomic_npy, _normalize   # noqa: E402

GRID = (99, 117, 95)     # shape of the MNI152 2 mm grid used for the registered volumes
GRID_MM = 2.0


def native_volume(vol: np.ndarray, spacing):
    """(GRID-shaped volume, fraction of brain volume outside the grid) for a
    harmonized RAS volume `vol` with voxel spacing `spacing` (mm)."""
    from scipy.ndimage import map_coordinates
    h = np.asarray(spacing, dtype=np.float64)
    mask = vol > 0
    com = np.argwhere(mask).mean(axis=0) * h                       # Eq. 2, mm
    axes = [(np.arange(n) - (n - 1) / 2) * GRID_MM for n in GRID]  # mm offsets from the centre
    pts = np.meshgrid(*axes, indexing="ij")
    coords = np.stack([(com[i] + pts[i]) / h[i] for i in range(3)])
    out = map_coordinates(vol, coords, order=1, mode="constant", cval=0.0)
    m_out = map_coordinates(mask.astype(np.float32), coords, order=1, mode="constant", cval=0.0) >= 0.5
    inside = m_out.sum() * GRID_MM ** 3
    total = mask.sum() * np.prod(h)
    return _normalize(out, m_out), float(max(0.0, 1.0 - inside / total))


def _one(job):
    ptid, brain_path, dst = job
    if dst.exists():
        return ptid, None
    from spherical_transform import load_nifti_as_ras
    from preprocessing.harmonize import n4_correct, resample_common
    vol, nii = load_nifti_as_ras(Path(brain_path))
    if not np.any(vol > 0):
        raise RuntimeError(f"empty volume for {ptid}")
    sp = tuple(float(z) for z in nii.header.get_zooms()[:3])
    vol, sp = resample_common(n4_correct(vol, sp), sp)
    V, outside = native_volume(vol, sp)
    _atomic_npy(dst, V)
    return ptid, outside


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=REPO / "cohort" / "manifest.csv")
    ap.add_argument("--out", type=Path, default=REPO / "native_cache")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="Only the first N subjects (testing)")
    args = ap.parse_args()

    man = pd.read_csv(args.manifest)
    args.out.mkdir(parents=True, exist_ok=True)
    _atomic_bytes(args.out / "_volume.json", json.dumps({
        "registration": "none", "harmonize": "n4+resample1.2mm", "grid": list(GRID),
        "grid_mm": GRID_MM, "centre": "subject centre of mass", "interpolation": "trilinear",
        "normalization": "median/MAD within subject mask, clip +-8, 0 outside"}).encode())
    jobs = [(r.PTID, r.brain_path, args.out / f"{r.PTID}.npy") for r in man.itertuples()]
    if args.limit:
        jobs = jobs[: args.limit]

    qc_path = args.out / "_qc.csv"
    qc = pd.read_csv(qc_path).set_index("PTID")["outside_fraction"].to_dict() if qc_path.exists() else {}
    failed = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_one, j): j[0] for j in jobs}
        for n, f in enumerate(as_completed(futs), 1):
            try:
                ptid, outside = f.result()
                if outside is not None:
                    qc[ptid] = outside
            except Exception as e:
                failed.append(futs[f])
                print(f"FAILED {futs[f]}: {e}", flush=True)
            if n % 100 == 0 or n == len(jobs):
                print(f"native volumes {n}/{len(jobs)}", flush=True)
                _atomic_bytes(qc_path, pd.DataFrame({"PTID": list(qc), "outside_fraction": list(qc.values())})
                              .to_csv(index=False, lineterminator="\n").encode())
    if qc:
        v = np.array(list(qc.values()))
        print(f"brain volume outside the grid: median {np.median(v):.4%}, max {v.max():.4%}")
    print(f"DONE ({len(failed)} failed: {failed})", flush=True)


if __name__ == "__main__":
    main()
