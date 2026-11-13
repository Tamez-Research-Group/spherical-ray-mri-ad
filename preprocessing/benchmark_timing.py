"""
benchmark_timing.py — per-subject processing time of the registration-free
and registration-based pipelines on the same workstation (Methods, Section
2.10; Discussion, limitations).

For a random sample of manifest subjects, every stage is timed in one
process, one subject at a time:
  - load: NIfTI read + RAS reorientation (shared by all pipelines)
  - n4: N4 bias-field correction (preprocessing/harmonize.n4_correct; shared)
  - resample: blur + 1.2 mm resampling (spherical and native-volume pipelines)
  - spherical: the spherical transform (Methods Eqs. 1-8)
  - native_volume: Cartesian sampling of preprocessing/native_volume.py
  - ants_affine, ants_syn: ANTs registration to the MNI152 2 mm template as in
    preprocessing/register.py (deterministic mode, seed 42); SyN includes
    its affine stage

--threads 1 reproduces how both pipelines were run in the study (one thread
per subject, parallelism across subjects); --threads 0 lets ITK use every
core, for the ANTs stages only (N4 is always single-threaded, as in the study).
The ITK thread count must be fixed before ANTs is imported, so each setting
is a separate run. The preprocessing stages are timed for every subject
before ANTsPy is imported, as in the spherical pipeline, which never loads it. Results are appended to <out> (one row per subject and
stage) and summarized as median and interquartile range.

    python preprocessing/benchmark_timing.py --n 20 --threads 1
    python preprocessing/benchmark_timing.py --n 20 --threads 0
"""

from __future__ import annotations

import os
import sys
import time
import argparse
import platform
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=REPO / "cohort" / "manifest.csv")
    ap.add_argument("--template-dir", type=Path, default=REPO / "registered_cache")
    ap.add_argument("--out", type=Path, default=REPO / "results" / "timing.csv")
    ap.add_argument("--n", type=int, default=20, help="Number of randomly sampled subjects")
    ap.add_argument("--seed", type=int, default=0, help="Seed of the subject sample")
    ap.add_argument("--threads", type=int, default=1, help="ITK threads for ANTs (0: all cores)")
    args = ap.parse_args()

    if args.threads > 0:
        os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(args.threads)
    import numpy as np
    import pandas as pd
    import nibabel as nib
    import tempfile
    from spherical_transform import load_nifti_as_ras, spherical_tensor_from_volume
    from preprocessing.harmonize import n4_correct, resample_common
    from preprocessing.native_volume import native_volume
    from preprocessing.register import SEED, _atomic_bytes

    man = pd.read_csv(args.manifest)
    sample = man.sample(n=args.n, random_state=args.seed)
    threads = args.threads if args.threads > 0 else os.cpu_count()
    cpu = platform.processor() or platform.machine()
    times, retries = {r.PTID: {} for r in sample.itertuples()}, {}

    def timed(ptid, name, fn, *a, **kw):
        # A step that raises is retried (up to 3 attempts) and only the
        # successful attempt is timed.
        for attempt in range(3):
            t0 = time.perf_counter()
            try:
                out = fn(*a, **kw)
            except RuntimeError as e:
                print(f"  {name} failed (attempt {attempt + 1}): {str(e).splitlines()[-1][:120]}", flush=True)
                retries[name] = retries.get(name, 0) + 1
                continue
            times[ptid][name] = time.perf_counter() - t0
            return out
        raise RuntimeError(f"{name} failed 3 times for {ptid}")

    with tempfile.TemporaryDirectory() as td:
        # Phase 1, ANTsPy not loaded (as in the spherical pipeline): ANTsPy and
        # SimpleITK each ship their own ITK, and with both in one process N4
        # fails with spurious BSplineControlPointImageFilter errors.
        for k, r in enumerate(sample.itertuples(), 1):
            vol, nii = timed(r.PTID, "load", load_nifti_as_ras, Path(r.brain_path))
            sp = tuple(float(z) for z in nii.header.get_zooms()[:3])
            v4 = timed(r.PTID, "n4", n4_correct, vol, sp)
            vr, spr = timed(r.PTID, "resample", resample_common, v4, sp)
            timed(r.PTID, "spherical", spherical_tensor_from_volume, vr, spr)
            timed(r.PTID, "native_volume", native_volume, vr, spr)
            nib.save(nib.Nifti1Image(v4, nii.affine), str(Path(td) / f"{r.PTID}.nii.gz"))
            print(f"[prep {k}/{args.n}] {r.PTID} "
                  + " ".join(f"{s}={v:.1f}s" for s, v in times[r.PTID].items()), flush=True)

        # Phase 2: ANTs registration of the N4-corrected volumes (preprocessing/register.py).
        import ants
        fixed = ants.image_read(str(args.template_dir / "mni152_2009_2mm.nii.gz"))
        for k, r in enumerate(sample.itertuples(), 1):
            moving = ants.image_read(str(Path(td) / f"{r.PTID}.nii.gz"))
            for key, ttype in (("ants_affine", "Affine"), ("ants_syn", "SyN")):
                ants.config.set_ants_deterministic(True, seed_value=SEED)
                timed(r.PTID, key, ants.registration, fixed=fixed, moving=moving, type_of_transform=ttype,
                      outprefix=str(Path(td) / f"{r.PTID}_{key}_"))
            print(f"[ants {k}/{args.n}] {r.PTID} affine={times[r.PTID]['ants_affine']:.1f}s "
                  f"syn={times[r.PTID]['ants_syn']:.1f}s", flush=True)

    rows = [{"PTID": p, "protocol": proto, "stage": s, "seconds": v, "ants_threads": threads, "cpu": cpu}
            for p, proto in zip(sample.PTID, sample.protocol) for s, v in times[p].items()]

    df = pd.DataFrame(rows)
    if args.out.exists():
        old = pd.read_csv(args.out)
        df = pd.concat([old[old["ants_threads"] != threads], df])
    _atomic_bytes(args.out, df.to_csv(index=False, lineterminator="\n").encode())

    cur = df[df["ants_threads"] == threads]
    q = cur.groupby("stage")["seconds"].quantile([0.25, 0.5, 0.75]).unstack()
    print(f"\nANTs threads = {threads}; median [IQR] seconds per subject (n = {cur.PTID.nunique()}):")
    for s, v in q.iterrows():
        print(f"  {s:14s} {v[0.5]:6.1f} [{v[0.25]:.1f}-{v[0.75]:.1f}]")
    per = cur.pivot_table(index="PTID", columns="stage", values="seconds")
    sph = per[["n4", "resample", "spherical"]].sum(1)
    syn = per[["n4", "ants_syn"]].sum(1)
    print(f"  pipeline totals excl. loading: spherical {sph.median():.1f} s, "
          f"SyN-registered {syn.median():.1f} s (median)")
    print(f"  retried steps: {retries or 'none'}")


if __name__ == "__main__":
    main()
