"""
harmonize.py — image-level harmonization applied to each skull-stripped
volume before the spherical transform (Methods, Section 2.2).

Acquisition protocol differs across ADNI phases and is associated with
diagnosis in this cohort (Methods, Section 2.6). Two standard steps reduce
protocol-dependent image differences without altering disease signal:

- n4_correct: N4 bias-field correction (Tustison et al., 2010), estimated
  within the brain mask on a 4x-shrunk copy and applied at full
  resolution. Removes smooth, scanner/coil-dependent intensity gradients.
- resample_common: blurs every axis finer than the target to a common
  effective resolution (Gaussian FWHM matching) and resamples onto an
  isotropic grid, so that scans acquired at different resolutions have
  comparable sharpness.

On this cohort, N4 + 1.2 mm resampling lowered how well a linear probe can
recognize the protocol among CN subjects (test AUC 0.907 -> 0.887) with
no change in CN-vs-Dementia AUC (0.857 -> 0.856).
"""

from __future__ import annotations

import numpy as np

TARGET_MM = 1.2   # coarsest acquired axis in the cohort (ADNI-1 sagittal slices)


def n4_correct(vol: np.ndarray, spacing, shrink: int = 4,
               iterations=(50, 50, 50, 50)) -> np.ndarray:
    """N4 bias-field correction of a skull-stripped volume (mask = vol > 0).
    Returns a float32 volume, zero outside the mask."""
    import SimpleITK as sitk
    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1)   # deterministic
    # SimpleITK arrays are (z, y, x): reverse the axes and the spacing.
    img = sitk.GetImageFromArray(np.ascontiguousarray(vol.transpose(2, 1, 0)).astype(np.float32))
    img.SetSpacing(tuple(float(s) for s in spacing))
    mask = sitk.GetImageFromArray(np.ascontiguousarray((vol > 0).transpose(2, 1, 0)).astype(np.uint8))
    mask.CopyInformation(img)
    img_s = sitk.Shrink(img, [shrink] * 3)
    mask_s = sitk.Shrink(mask, [shrink] * 3)
    corr = sitk.N4BiasFieldCorrectionImageFilter()
    corr.SetMaximumNumberOfIterations(list(iterations))
    corr.Execute(img_s, mask_s)
    log_bias = corr.GetLogBiasFieldAsImage(img)
    out = img / sitk.Exp(log_bias)
    arr = sitk.GetArrayFromImage(out).transpose(2, 1, 0)
    return np.where(vol > 0, arr, 0).astype(np.float32)


def resample_common(vol: np.ndarray, spacing, target: float = TARGET_MM):
    """Blur each axis finer than `target` mm so its effective FWHM equals
    `target`, then interpolate (trilinear) onto a `target`-mm isotropic grid.
    The brain mask is resampled separately and re-applied. Returns
    (volume, new_spacing)."""
    from scipy.ndimage import gaussian_filter, zoom
    h = np.asarray(spacing, dtype=np.float64)
    sigma_vox = np.sqrt(np.maximum(target ** 2 - h ** 2, 0.0)) / 2.3548 / h
    mask = (vol > 0).astype(np.float32)
    blurred = gaussian_filter(vol, sigma_vox) if np.any(sigma_vox > 0) else vol
    factors = h / target
    vol_r = zoom(blurred, factors, order=1)
    mask_r = zoom(mask, factors, order=1) >= 0.5
    return np.where(mask_r, vol_r, 0).astype(np.float32), (target,) * 3
