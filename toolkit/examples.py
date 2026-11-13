"""Example inputs for trying the toolkit without an MRI of your own.

  phantom   A synthetic, T1-like, skull-stripped brain built from simple shapes:
            folded cortex with sulci, white matter, lateral ventricles, deep
            grey matter, cerebellum and brainstem, plus a smooth bias field and
            noise. It is not anyone's brain, so it can be shared freely.
            `atrophy` (0-1) enlarges the ventricles and sulci and thins the
            cortex, to show how the rays and shells respond.
  mni152    The MNI152 2009 template (an average of 152 adult brains, shipped
            with nilearn) multiplied by its brain mask: a realistic, but
            averaged and therefore unusually smooth, skull-stripped brain.
            Copyright (C) 1993-2009 Louis Collins, McConnell Brain Imaging
            Centre, Montreal Neurological Institute, McGill University
            (free to use and redistribute with this notice).

The models were trained on real ADNI scans: their output on either example is
out of distribution and means nothing. The examples demonstrate the pipeline.

    python toolkit/examples.py phantom --out phantom.nii.gz [--atrophy 0.6]
    python toolkit/examples.py mni152  --out mni152_brain.nii.gz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter

T1 = {"wm": 1.0, "gm": 0.62, "deep_gm": 0.75, "csf": 0.22}   # relative T1-weighted intensities


def _ellipsoid(X, Y, Z, c, axes):
    return ((X - c[0]) / axes[0]) ** 2 + ((Y - c[1]) / axes[1]) ** 2 + ((Z - c[2]) / axes[2]) ** 2


def phantom(atrophy: float = 0.0, seed: int = 0, shape=(160, 196, 160)) -> nib.Nifti1Image:
    """Synthetic skull-stripped T1 brain, 1 mm isotropic, RAS orientation."""
    a = float(np.clip(atrophy, 0, 1))
    rng = np.random.default_rng(seed)
    centre = (np.asarray(shape) - 1) / 2
    X, Y, Z = np.meshgrid(*(np.arange(n) - c for n, c in zip(shape, centre)), indexing="ij")  # mm; x L->R, y P->A, z I->S
    Z = Z - 8                                     # cerebrum sits above the volume centre

    def field(sigma):                             # smooth random field, unit SD
        f = gaussian_filter(rng.standard_normal(shape), sigma)
        return f / f.std()

    gyri, sulc = field(3.0), field(4.0)
    shrink = 1 - 0.04 * a
    # cerebrum: ellipsoid ~135 x 165 x 112 mm with a gently undulating surface
    rho = np.sqrt(_ellipsoid(X, Y, Z, (0, 2, 4), (67 * shrink, 82 * shrink, 56 * shrink))) + 0.02 * gyri
    cerebrum = rho < 1.0
    # interhemispheric fissure: full depth at the poles, down to the corpus callosum elsewhere
    floor = 24 - 34 * np.clip((np.abs(Y - 2) - 30) / 20, 0, 1)        # deepens smoothly towards the poles
    fissure = (np.abs(X) < 1.0 + 1.5 * a) & (Z > floor)
    cerebrum &= ~fissure
    # cerebellum and brainstem
    cereb_r = np.sqrt(_ellipsoid(X, Y, Z, (0, -46, -40), (48 * shrink, 26 * shrink, 20 * shrink))) + 0.03 * gyri
    cerebellum = (cereb_r < 1.0) & ~((np.abs(X) < 2) & (Z > -30))
    stem = (np.sqrt(X ** 2 + (Y + 18) ** 2) < 11) & (Z < -10) & (Z > -62)
    brain = cerebrum | cerebellum | stem

    vol = np.zeros(shape, np.float32)
    vol[brain] = T1["wm"]
    # sulci: thin CSF sheets (zero-crossings of a smooth field) reaching ~14 mm
    # into the cerebrum; wider and deeper with atrophy
    depth = distance_transform_edt(cerebrum)
    sulci = cerebrum & (np.abs(sulc) < 0.10 + 0.12 * a) & (depth < 14 * (1 + 0.3 * a))
    # cortex: a ~2.6 mm grey-matter ribbon lining the outer surface and every sulcus
    # (thinner with atrophy)
    from_csf = distance_transform_edt(cerebrum & ~sulci)
    vol[cerebrum & (from_csf <= 2.6 * (1 - 0.25 * a))] = T1["gm"]
    vol[sulci] = T1["csf"]
    vol[cerebellum & ((cereb_r > 0.82) | (np.sin(Z * 1.3 + gyri) > 0.55))] = T1["gm"]   # folia
    # deep grey matter (thalami, basal ganglia) and lateral ventricles
    for side in (-1, 1):
        vol[_ellipsoid(X, Y, Z, (side * 11, -6, 2), (8, 14, 9)) < 1] = T1["deep_gm"]
        vol[_ellipsoid(X, Y, Z, (side * 19, 12, 0), (6, 11, 9)) < 1] = T1["deep_gm"]
        vent = 1 + 1.4 * a
        body = _ellipsoid(X, Y, Z, (side * (6 + 2 * a), 0, 16), (4 * vent, 26, 6 * vent)) < 1
        horn = _ellipsoid(X, Y, Z, (side * (16 + 2 * a), -18, -6), (3 * vent, 12, 4 * vent)) < 1   # temporal horn
        vol[(body | horn) & cerebrum] = T1["csf"]
    # partial volume, bias field and noise, as in a real acquisition
    vol = gaussian_filter(vol, 0.7)
    bias = 1 + 0.08 * (X / 80) - 0.06 * (Z / 60) + 0.04 * field(25.0)
    vol = vol * bias + rng.normal(0, 0.02, shape)
    mask = gaussian_filter(brain.astype(np.float32), 0.7) > 0.5
    vol = np.where(mask, np.clip(vol, 0.05, None), 0) * 800             # scanner-like range, zero background
    affine = np.eye(4)
    affine[:3, 3] = -centre
    img = nib.Nifti1Image(vol.astype(np.float32), affine)
    img.header.set_xyzt_units("mm")
    return img


def mni152() -> nib.Nifti1Image:
    """MNI152 2009 1 mm template, skull-stripped with its brain mask (bundled with nilearn)."""
    from nilearn.datasets import load_mni152_brain_mask, load_mni152_template
    t1, mask = load_mni152_template(resolution=1), load_mni152_brain_mask(resolution=1)
    data = np.asarray(t1.dataobj, np.float32) * (np.asarray(mask.dataobj) > 0)
    return nib.Nifti1Image(data, t1.affine)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=["phantom", "mni152"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--atrophy", type=float, default=0.0, help="phantom only, 0-1")
    ap.add_argument("--seed", type=int, default=0, help="phantom only")
    a = ap.parse_args()
    img = phantom(a.atrophy, a.seed) if a.kind == "phantom" else mni152()
    nib.save(img, str(a.out))
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
