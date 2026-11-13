"""
build_manifest.py — writes cohort/manifest.csv: one row per cohort subject
with its skull-stripped volume path and RAS-canonical image geometry.

Replaces the older experiment's pca_axial_dataset.csv (which pointed at 2-D
PCA images no longer used). Only header information is read, so this is
fast. Voxel spacing is recorded because the spherical transform
(spherical_transform.py) needs it to work in millimetres (anisotropic or
scanner-dependent spacing would otherwise distort its angles), and because
spacing tracks acquisition protocol, which is associated with diagnosis in
this cohort. `protocol` is a two-level proxy derived from the left-right
spacing: "slice1.2" (1.2 mm, the ADNI-1-style sagittal acquisition) or
"iso1.0" (<= 1.0 mm). It is used to balance training batches (train.py) and
for protocol-stratified evaluation (evaluate.py).

    python cohort/build_manifest.py --nifti-root /path/to/MRIPrep_outputs
"""

import argparse
from pathlib import Path

import nibabel as nib
import pandas as pd


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--nifti-root", type=Path, required=True,
                        help="Directory of <PTID>/brain.nii.gz volumes")
    parser.add_argument("--override-root", type=Path, default=here.parent / "MRIPrep_outputs",
                        help="Optional; checked first, for replacement copies of individual volumes")
    parser.add_argument("--cohort-csv", type=Path, default=here / "cohort.csv")
    parser.add_argument("--out", type=Path, default=here / "manifest.csv")
    args = parser.parse_args()

    cohort = pd.read_csv(args.cohort_csv)
    rows, missing, unreadable = [], [], []
    for row in cohort.itertuples():
        path = (args.override_root / row.PTID / "brain.nii.gz").resolve()
        source = "rebuilt"
        if not path.exists():
            path = (args.nifti_root / row.PTID / "brain.nii.gz").resolve()
            source = "hd-bet"
        if not path.exists():
            missing.append(row.PTID)
            continue
        try:
            img = nib.as_closest_canonical(nib.load(str(path)))
        except Exception as e:
            unreadable.append((row.PTID, f"{type(e).__name__}: {e}"))
            continue
        sx, sy, sz = (float(z) for z in img.header.get_zooms()[:3])
        nx, ny, nz = img.shape[:3]
        rows.append({
            "PTID": row.PTID, "Diagnosis": row.Diagnosis, "split": row.split,
            "brain_path": str(path), "source": source,
            "spacing_0": round(sx, 4), "spacing_1": round(sy, 4), "spacing_2": round(sz, 4),
            "shape_0": nx, "shape_1": ny, "shape_2": nz,
            "protocol": "slice1.2" if round(sx, 1) == 1.2 else "iso1.0",
        })

    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"Wrote {len(rows)} rows to {args.out}")
    if missing:
        print(f"WARNING: {len(missing)} cohort subjects have no brain.nii.gz: {missing}")
    for ptid, err in unreadable:
        print(f"WARNING: unreadable volume for {ptid}: {err}")


if __name__ == "__main__":
    main()
