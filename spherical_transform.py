"""
spherical_transform.py — the registration-free spherical intensity
representation (Methods, Section 2.3; Eqs. 1-8).

Resamples a skull-stripped, intensity-normalized brain volume onto a
per-subject spherical grid (N_PHI x N_THETA x N_RADII), centered on that
subject's own center of mass, with the radial axis scaled to that subject's
own per-direction brain boundary (R_edge). No deformable registration to a
template or atlas is used anywhere in this module.

Cleaned up from deep_learning_experiment.py's compute_spherical_tensor():
the in-mask voxel-index components, originally named zi/yi/xi (misleading --
they are RAS-canonical array-axis indices 0/1/2, not literal Cartesian z/y/x),
are renamed v0/v1/v2 throughout to match the paper's notation exactly.
Geometry is now computed in millimetres by default (physical=True); the
original voxel-index geometry is kept as physical=False.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import nibabel as nib
from scipy.ndimage import gaussian_filter, map_coordinates

CLIP_SIGMA = 8.0          # Eq. 1 (normalize.py) clip bound

N_SPH_PHI   = 36          # N_phi,   Methods 2.3
N_SPH_THETA = 72          # N_theta, Methods 2.3
N_SPH_RADII = 48          # N_r,     Methods Eq. 7


def load_nifti_as_ras(path: Path):
    """Load a NIfTI volume and reorient to RAS canonical axis order (Methods 2.2)."""
    nii = nib.load(str(path))
    nii_ras = nib.as_closest_canonical(nii)
    return nii_ras.get_fdata().astype(np.float32), nii_ras


def _make_mask(vol: np.ndarray) -> np.ndarray:
    """In-mask (brain) voxel set M (Methods 2.2): positive-intensity voxels
    after skull stripping."""
    return vol > 0


def _bbox_slices(mask: np.ndarray, margin: int = 20):
    """Axis-aligned bounding box of the mask, with a fixed voxel margin
    (Methods 2.2: 20-voxel margin)."""
    slices = []
    for ax_coords, size in zip(np.where(mask), mask.shape):
        lo = max(0, int(ax_coords.min()) - margin)
        hi = min(size, int(ax_coords.max()) + margin + 1)
        slices.append(slice(lo, hi))
    return tuple(slices)


def normalize(vol: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Robust median/MAD intensity normalization, clipped to +-CLIP_SIGMA
    (Methods Eq. 1)."""
    vals = vol[mask]
    if vals.size == 0:
        return vol.copy()
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med))) + 1e-6
    out = (vol - med) / (1.4826 * mad)
    return np.clip(out, -CLIP_SIGMA, CLIP_SIGMA).astype(np.float32)


def compute_spherical_tensor(
    brain_path: Path,
    n_phi: int = N_SPH_PHI,
    n_theta: int = N_SPH_THETA,
    n_radii: int = N_SPH_RADII,
    physical: bool = True,
    harmonize: bool = True,
) -> "np.ndarray | None":
    """
    Load a skull-stripped volume (brain_path) and produce a
    (n_phi, n_theta, n_radii) spherical intensity tensor T (Methods Eq. 8).

    physical=True (default) measures the center of mass, directions and
    radii in millimetres, using the volume's voxel spacing, so that
    anisotropic or scanner-dependent voxel sizes do not distort the angular
    grid; sample points are converted back to voxel indices only for
    interpolation. physical=False reproduces the original voxel-index
    geometry (every voxel treated as a unit cube).

    harmonize=True (default, requires physical) first applies N4 bias-field
    correction and resampling to a common 1.2 mm isotropic resolution
    (preprocessing/harmonize.py, Methods 2.2).

    Returns None if the file is missing or corrupt.
    """
    if not brain_path.exists():
        return None
    try:
        vol, nii_ras = load_nifti_as_ras(brain_path)
    except Exception:
        return None
    if not np.any(vol > 0):
        return None  # empty / zero-filled volume (e.g. a truncated write)
    if harmonize and not physical:
        raise ValueError("harmonize=True requires physical=True")
    spacing = tuple(float(z) for z in nii_ras.header.get_zooms()[:3]) if physical else (1.0, 1.0, 1.0)
    if harmonize:
        from preprocessing.harmonize import n4_correct, resample_common
        vol, spacing = resample_common(n4_correct(vol, spacing), spacing)
    return spherical_tensor_from_volume(vol, spacing, n_phi, n_theta, n_radii)


def spherical_tensor_from_volume(
    vol: np.ndarray,
    spacing,
    n_phi: int = N_SPH_PHI,
    n_theta: int = N_SPH_THETA,
    n_radii: int = N_SPH_RADII,
    return_geometry: bool = False,
    mask: "np.ndarray | None" = None,
):
    """Spherical tensor of an RAS-canonical skull-stripped volume already in
    memory, with voxel spacing `spacing` (mm per voxel along array axes 0-2;
    pass ones for voxel-index geometry). Returns None for an empty volume.

    `mask` overrides the brain mask (default: vol > 0). It is needed for
    volumes that are already intensity-normalized, such as the MNI-registered
    volumes of preprocessing/register.py, where brain values can be negative;
    for those the template brain mask is passed (Methods 2.9).

    With return_geometry=True, returns (tensor, geometry) where geometry holds
    what is needed to map tensor cells back into the brain (visualization):
    the normalized cropped volume, crop slices, spacing h, centre of mass c
    (in units of h, cropped frame), R_edge, and the sample points (Eq. 7) in
    the same frame."""
    if mask is None:
        if not np.any(vol > 0):
            return None  # empty / zero-filled volume (e.g. a truncated write)
        mask = _make_mask(vol)
    elif not np.any(vol[mask] != 0):
        return None
    crop  = _bbox_slices(mask, margin=20)
    vol_c = vol[crop]
    msk_c = mask[crop]
    vol_n = normalize(vol_c, msk_c)

    # Voxel spacing h along the three RAS-canonical array axes (0: left->right,
    # 1: posterior->anterior, 2: inferior->superior). All geometry below is in
    # units of h: millimetres, or voxels when spacing is all ones.
    h = np.asarray(spacing, dtype=np.float64)

    # In-mask coordinates, voxel index scaled by spacing.
    v0, v1, v2 = np.where(msk_c)
    vox   = np.stack([v0, v1, v2], axis=1).astype(np.float64) * h
    com   = vox.mean(axis=0)                                # Eq. 2 (center of mass, c)
    disp  = vox - com
    r_all = np.linalg.norm(disp, axis=1)
    valid = r_all > 0.5
    disp_v, r_v = disp[valid], r_all[valid]
    udir = disp_v / r_v[:, None]

    # Spherical angles (Eqs. 3-4): elevation phi from the inferior-superior
    # axis (array axis 2), azimuth theta in the left-right/posterior-anterior
    # plane (array axes 0,1).
    phi_v   = np.arcsin(np.clip(udir[:, 2], -1.0, 1.0))
    theta_v = np.arctan2(udir[:, 1], udir[:, 0]) % (2 * np.pi)

    phi_idx   = np.clip(((phi_v + np.pi / 2) / np.pi * n_phi).astype(int),   0, n_phi - 1)
    theta_idx = np.clip((theta_v / (2 * np.pi) * n_theta).astype(int),      0, n_theta - 1)

    # Per-direction edge radius R_edge (Eqs. 5-6): scatter-max, then
    # anisotropic Gaussian smoothing (periodic in theta, reflecting in phi),
    # then floored at a minimum of 4 (mm, or voxels when not physical).
    R_edge = np.zeros((n_phi, n_theta), dtype=np.float64)
    np.maximum.at(R_edge, (phi_idx, theta_idx), r_v)
    R_edge_raw = R_edge.copy()                      # Eq. 5, before smoothing (for visualization)
    R_edge = gaussian_filter(R_edge, sigma=1.2, mode=["reflect", "wrap"])
    R_edge = np.maximum(R_edge, 4.0)

    # Fixed angular bin centers and fractional radii (Methods 2.3).
    phis_g   = (np.arange(n_phi)   + 0.5) / n_phi   * np.pi - np.pi / 2
    thetas_g = (np.arange(n_theta) + 0.5) / n_theta * 2 * np.pi
    k_fracs  = np.arange(1, n_radii + 1) / n_radii

    PHI, THETA = np.meshgrid(phis_g, thetas_g, indexing="ij")
    d0 = np.cos(PHI) * np.cos(THETA)
    d1 = np.cos(PHI) * np.sin(THETA)
    d2 = np.sin(PHI)

    # Ray-sample points p(phi_i, theta_j, r_k) (Eq. 7), converted from
    # physical coordinates back to voxel indices for interpolation.
    R  = R_edge[:, :, None] * k_fracs[None, None, :]
    s0 = (com[0] + R * d0[:, :, None]) / h[0]
    s1 = (com[1] + R * d1[:, :, None]) / h[1]
    s2 = (com[2] + R * d2[:, :, None]) / h[2]

    # Trilinear interpolation, zero-padded outside the volume (Eq. 8).
    vals = map_coordinates(
        vol_n.astype(np.float64),
        np.array([s0.ravel(), s1.ravel(), s2.ravel()]),
        order=1, mode="constant", cval=0.0,
    )
    tensor = vals.reshape(n_phi, n_theta, n_radii).astype(np.float32)
    if not return_geometry:
        return tensor
    return tensor, {
        "vol_n": vol_n, "crop": crop, "h": h, "com": com, "R_edge": R_edge, "R_edge_raw": R_edge_raw,
        "points": np.stack([com[0] + R * d0[:, :, None], com[1] + R * d1[:, :, None],
                            com[2] + R * d2[:, :, None]], axis=-1),   # (phi, theta, r, 3), units of h
        "phis": phis_g, "thetas": thetas_g, "k_fracs": k_fracs,
    }


def _atomic_write_bytes(path: Path, data: bytes):
    """Write via a temp file, fsync and rename, so a crash never leaves a
    truncated or zero-filled file in place."""
    import os
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _cache_one(job):
    import io
    ptid, brain_path, out, physical, harmonize = job
    tensor = compute_spherical_tensor(Path(brain_path), physical=physical, harmonize=harmonize)
    if tensor is None:
        return ptid
    buf = io.BytesIO()
    np.save(buf, tensor)
    _atomic_write_bytes(out, buf.getvalue())
    return None


def build_cache(manifest, cache_dir: Path, force: bool = False, workers: int = 8,
                physical: bool = True, harmonize: bool = True):
    """Compute and save spherical tensors to cache_dir/<PTID>.npy for every
    row of `manifest` (a DataFrame with PTID and brain_path columns, see
    cohort/manifest.csv). Returns the list of PTIDs that failed.

    cache_dir/_transform.json records the geometry the cache was built with;
    reusing a cache built with a different geometry is refused unless
    force=True (which rebuilds every tensor)."""
    import json
    from concurrent.futures import ProcessPoolExecutor
    from tqdm import tqdm

    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path = cache_dir / "_transform.json"
    meta = {"geometry": "mm" if physical else "voxel",
            "harmonize": "n4+resample1.2mm" if harmonize else "none",
            "grid": [N_SPH_PHI, N_SPH_THETA, N_SPH_RADII]}
    raw = meta_path.read_bytes() if meta_path.exists() else None
    if raw is not None and not raw.strip(b"\x00 \r\n\t"):
        raise RuntimeError(f"{meta_path} is empty or zero-filled (a crash during a write); "
                           f"restore it or rebuild with --force-cache")
    existing = json.loads(raw) if raw is not None else None
    if existing != meta and not force:
        if existing is not None or any(cache_dir.glob("*.npy")):
            raise RuntimeError(
                f"{cache_dir} was built with {existing or 'an unrecorded transform'}, "
                f"requested {meta}; rebuild with --force-cache")
    if existing != meta:      # rewrite only on change, atomically (crash-safe)
        _atomic_write_bytes(meta_path, json.dumps(meta).encode())

    jobs = [(r.PTID, r.brain_path, cache_dir / f"{r.PTID}.npy", physical, harmonize)
            for r in manifest.itertuples()
            if force or not (cache_dir / f"{r.PTID}.npy").exists()]
    failed = []
    if jobs:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for res in tqdm(pool.map(_cache_one, jobs, chunksize=4), total=len(jobs),
                            desc="Building spherical-tensor cache"):
                if res is not None:
                    failed.append(res)
    if failed:
        print(f"WARNING: {len(failed)} subjects could not be transformed: {failed}")
    return failed
