"""
register.py — registration-based reference pipeline, used to compare the
registration-free spherical representation against conventional
template registration on the same subjects (Methods, Section 2.9).

For each subject listed in cohort/manifest.csv:
  1. RAS reorientation and N4 bias-field correction (the same
     preprocessing/harmonize.n4_correct used by the spherical pipeline).
  2. ANTs registration of the skull-stripped brain to the MNI152 (ICBM 2009)
     2 mm brain template, twice: Affine (12 dof) and SyN (affine + deformable).
  3. Each warped brain is intensity-normalized like the spherical pipeline
     (median/MAD within the template brain mask, clipped to +-8) and set to 0
     outside the mask; saved as float32 (99, 117, 95) to
     <out>/{affine,syn}/<PTID>.npy.
  4. The Harvard-Oxford subcortical atlas (thr25, 2 mm) is warped into native
     space with the inverse SyN transform; regional volumes (mm^3) and total
     brain volume are saved to <out>/roi/<PTID>.json.

ANTs runs in deterministic mode (fixed seed, one thread per process) so that
results are reproducible; parallelism comes from running many subjects at
once. Resumable (finished subjects are skipped) and crash-safe (atomic writes).

    python preprocessing/register.py --workers 24
    python preprocessing/register.py --spherical      # then: spherical tensors of the SyN volumes

--spherical applies the spherical transform (spherical_transform.py) to each
SyN-registered volume, with the template brain mask as the brain mask, and
writes the (36, 72, 48) tensors to <out>/spherical_syn/<PTID>.npy. Because all
registered brains share the template's mask, centre of mass and R_edge are
the same for every subject; only the intensities differ. This isolates what
registration adds to the spherical representation (Methods 2.9).
"""

from __future__ import annotations

import os
import sys
import json
import time
import argparse
import tempfile
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import nibabel as nib

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

SEED = 42
CLIP = 8.0
TRANSFORMS = {"affine": "Affine", "syn": "SyN"}


def _atomic_bytes(path: Path, data: bytes):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _atomic_npy(path: Path, arr: np.ndarray):
    import io
    buf = io.BytesIO()
    np.save(buf, arr)
    _atomic_bytes(path, buf.getvalue())


def prepare_templates(out: Path):
    """Write the MNI152 2 mm template, its brain mask and the Harvard-Oxford
    subcortical atlas once; return their paths and the atlas labels."""
    from nilearn import datasets
    from nilearn.datasets import load_mni152_template, load_mni152_brain_mask
    out.mkdir(parents=True, exist_ok=True)
    paths = {"template": out / "mni152_2009_2mm.nii.gz", "mask": out / "mni152_2009_2mm_mask.nii.gz",
             "atlas": out / "harvard_oxford_sub_thr25_2mm.nii.gz"}
    if not paths["template"].exists():
        nib.save(load_mni152_template(resolution=2), str(paths["template"]))
    if not paths["mask"].exists():
        nib.save(load_mni152_brain_mask(resolution=2), str(paths["mask"]))
    atlas = datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm", verbose=0)
    if not paths["atlas"].exists():
        img = atlas.maps if not isinstance(atlas.maps, str) else nib.load(atlas.maps)
        nib.save(img, str(paths["atlas"]))
    return paths, list(atlas.labels)


def _normalize(vol: np.ndarray, mask: np.ndarray) -> np.ndarray:
    vals = vol[mask]
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med))) + 1e-6
    out = np.clip((vol - med) / (1.4826 * mad), -CLIP, CLIP)
    out[~mask] = 0.0
    return out.astype(np.float32)


def _worker_init():
    # one ITK thread per process + fixed seed (deterministic registration)
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "1"


def register_one(job):
    ptid, brain_path, out, paths, labels = job
    outs = {k: out / k / f"{ptid}.npy" for k in TRANSFORMS}
    roi_out = out / "roi" / f"{ptid}.json"
    if all(p.exists() for p in outs.values()) and roi_out.exists():
        return ptid, "skip", 0.0
    t0 = time.time()
    import ants
    ants.config.set_ants_deterministic(True, seed_value=SEED)
    from spherical_transform import load_nifti_as_ras
    from preprocessing.harmonize import n4_correct

    vol, nii = load_nifti_as_ras(Path(brain_path))
    sp = tuple(float(z) for z in nii.header.get_zooms()[:3])
    vol = n4_correct(vol, sp)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "n4.nii.gz"
        nib.save(nib.Nifti1Image(vol, nii.affine), str(tmp))
        moving = ants.image_read(str(tmp))
        fixed = ants.image_read(str(paths["template"]))
        tmask = ants.image_read(str(paths["mask"])).numpy() > 0.5
        reg = {}
        for key, ttype in TRANSFORMS.items():
            reg[key] = ants.registration(fixed=fixed, moving=moving, type_of_transform=ttype,
                                         outprefix=str(Path(td) / f"{key}_"))
            warped = reg[key]["warpedmovout"].numpy()
            _atomic_npy(outs[key], _normalize(warped, tmask))
        # atlas -> native space with the inverse SyN transform
        atlas = ants.image_read(str(paths["atlas"]))
        native = ants.apply_transforms(fixed=moving, moving=atlas, transformlist=reg["syn"]["invtransforms"],
                                       interpolator="genericLabel").numpy().astype(int)
    voxvol = float(np.prod(sp))
    rois = {labels[i]: float((native == i).sum() * voxvol) for i in range(1, len(labels))}
    rois["Brain volume"] = float((vol > 0).sum() * voxvol)
    _atomic_bytes(roi_out, json.dumps({"PTID": ptid, "voxel_mm3": voxvol, "volumes_mm3": rois}).encode())
    return ptid, "ok", time.time() - t0


def _spherical_one(job):
    ptid, src, dst, mask_path = job
    if dst.exists():
        return ptid, "skip"
    from spherical_transform import spherical_tensor_from_volume
    mask = nib.load(str(mask_path)).get_fdata() > 0.5
    T = spherical_tensor_from_volume(np.load(src), (2.0, 2.0, 2.0), mask=mask)
    if T is None:
        raise RuntimeError(f"empty registered volume for {ptid}")
    _atomic_npy(dst, T)
    return ptid, "ok"


def build_spherical(args, man):
    src_dir, dst_dir = args.out / "syn", args.out / "spherical_syn"
    dst_dir.mkdir(parents=True, exist_ok=True)
    _atomic_bytes(dst_dir / "_transform.json", json.dumps({
        "geometry": "mm", "source": "SyN-registered volumes (MNI152 2009, 2 mm)",
        "mask": "template brain mask", "grid": [36, 72, 48]}).encode())
    jobs = [(p, src_dir / f"{p}.npy", dst_dir / f"{p}.npy", args.out / "mni152_2009_2mm_mask.nii.gz")
            for p in man["PTID"]]
    if args.limit:
        jobs = jobs[: args.limit]
    failed = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_spherical_one, j): j[0] for j in jobs}
        for n, f in enumerate(as_completed(futs), 1):
            try:
                f.result()
            except Exception as e:
                failed.append(futs[f])
                print(f"FAILED {futs[f]}: {e}", flush=True)
            if n % 200 == 0 or n == len(jobs):
                print(f"spherical {n}/{len(jobs)}", flush=True)
    print(f"DONE spherical ({len(failed)} failed: {failed})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=REPO / "cohort" / "manifest.csv")
    ap.add_argument("--out", type=Path, default=REPO / "registered_cache")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0, help="Only the first N subjects (testing)")
    ap.add_argument("--spherical", action="store_true",
                    help="Build spherical tensors of the (already computed) SyN volumes instead of registering")
    args = ap.parse_args()
    if args.spherical:
        build_spherical(args, pd.read_csv(args.manifest))
        return

    paths, labels = prepare_templates(args.out)
    for k in list(TRANSFORMS) + ["roi"]:
        (args.out / k).mkdir(parents=True, exist_ok=True)
    _atomic_bytes(args.out / "_registration.json", json.dumps({
        "template": "MNI152 ICBM2009 2mm brain (nilearn)", "transforms": TRANSFORMS, "seed": SEED,
        "deterministic": True, "n4": "preprocessing.harmonize.n4_correct",
        "normalization": "median/MAD within template mask, clip +-8, 0 outside",
        "atlas": "Harvard-Oxford subcortical thr25 2mm, inverse SyN, genericLabel"}, indent=2).encode())

    man = pd.read_csv(args.manifest)
    jobs = [(r.PTID, r.brain_path, args.out, paths, labels) for r in man.itertuples()]
    if args.limit:
        jobs = jobs[: args.limit]
    done, failed, t0 = 0, [], time.time()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_worker_init) as pool:
        futs = {pool.submit(register_one, j): j[0] for j in jobs}
        for f in as_completed(futs):
            try:
                ptid, status, dt = f.result()
            except Exception as e:
                failed.append(futs[f])
                print(f"FAILED {futs[f]}: {type(e).__name__}: {e}", flush=True)
                continue
            done += 1
            if done % 25 == 0 or done == len(jobs) or status == "ok" and done <= 3:
                print(f"{done}/{len(jobs)}  elapsed {time.time() - t0:.0f}s  last {ptid} {status} {dt:.1f}s", flush=True)
    print(f"DONE ({len(failed)} failed: {failed})", flush=True)


if __name__ == "__main__":
    main()
