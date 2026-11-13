"""
convert.py — DICOM to NIfTI conversion (Methods, Section 2.2).

Converts a raw DICOM series directory into a single compressed NIfTI file
using dicom2nifti, with automatic reorientation to a standard anatomical
plane. Source: adapted verbatim from the research pipeline's
MRIPrep/pipeline/steps/convert.py; logic unchanged.
"""

from __future__ import annotations

import gzip
import logging
from pathlib import Path

import dicom2nifti
import nibabel as nib

log = logging.getLogger(__name__)


def _is_valid_nifti(path: Path) -> bool:
    """
    Return True if the NIfTI file at `path` is readable and not truncated.
    Loads only the header + a single data slice to keep it fast.
    """
    try:
        img = nib.load(str(path))
        # Trigger actual decompression by reading one slice
        img.dataobj[0]
        return True
    except (EOFError, gzip.BadGzipFile, Exception):
        return False


def run(
    dicom_dir: Path,
    output_path: Path,
    force: bool = False,
) -> Path:
    """
    Convert a DICOM series to NIfTI.

    Parameters
    ----------
    dicom_dir   : directory that contains the *.dcm files
    output_path : desired output path, e.g. .../subject/original.nii.gz
    force       : re-run even if output_path already exists

    Returns
    -------
    Path to the created NIfTI file.
    """
    if output_path.exists() and not force:
        if _is_valid_nifti(output_path):
            log.info("DICOM->NIfTI: output exists and is valid, skipping (%s)", output_path.name)
            return output_path
        else:
            log.warning(
                "DICOM->NIfTI: existing file is corrupted, reconverting (%s)",
                output_path.name,
            )
            output_path.unlink()

    log.info("DICOM->NIfTI: converting %s", dicom_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dicom2nifti.dicom_series_to_nifti(
        str(dicom_dir),
        str(output_path),
        reorient_nifti=True,
    )

    if not output_path.exists():
        raise RuntimeError(
            f"dicom2nifti finished but output not found: {output_path}"
        )

    if not _is_valid_nifti(output_path):
        output_path.unlink()
        raise RuntimeError(
            f"dicom2nifti produced a corrupted NIfTI: {output_path}"
        )

    log.info("DICOM->NIfTI: saved %s", output_path)
    return output_path
