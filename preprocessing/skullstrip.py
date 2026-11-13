"""
skullstrip.py — Skull stripping with HD-BET (Methods, Section 2.2).

HD-BET (Isensee et al., 2019) is a deep-learning brain extraction tool.
Install: pip install hd-bet
Docs:    https://github.com/MIC-DKFZ/HD-BET

Source: adapted verbatim from the research pipeline's
MRIPrep/pipeline/steps/skullstrip.py; logic unchanged.

Confirmed output naming (HD-BET >= 2.0, passing -o <path>.nii.gz):
    <path>.nii.gz           -- skull-stripped brain volume  (the -o path itself)
    <stem>_bet.nii.gz       -- binary brain mask            (requires --save_bet_mask)

Device values accepted by -device flag:
    cpu     -- CPU inference
    cuda    -- CUDA GPU (default in HD-BET when a GPU is available)
    mps     -- Apple Silicon GPU

Windows note: HD-BET's nnU-Net backend occasionally exits with a non-zero
return code (0xC0000409, stack-buffer-overrun in cleanup) even when outputs
are written correctly. We therefore check for the output file rather than
trusting the return code alone.

HD-BET also writes intermediate nnU-Net files into the output directory
(dataset.json, plans.json, predict_from_raw_data_args.json). These are
cleaned up regardless of whether the run succeeds or fails.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)

# Temp files written by nnU-Net / HD-BET into the output directory
_HDBET_TEMP_FILES = {
    "dataset.json",
    "plans.json",
    "predict_from_raw_data_args.json",
}


def _cleanup_temp_files(directory: Path) -> None:
    """Remove nnU-Net intermediate files left by HD-BET."""
    for name in _HDBET_TEMP_FILES:
        f = directory / name
        if f.exists():
            f.unlink()
            log.debug("Skull strip: removed temp file %s", name)


def run(
    nifti_path: Path,
    brain_path: Path,
    device: str = "cpu",
    disable_tta: bool = True,
    force: bool = False,
    timeout: int = 300,
) -> tuple[Path, Path | None]:
    """
    Run HD-BET skull stripping on a NIfTI volume.

    Parameters
    ----------
    nifti_path  : path to the input whole-head NIfTI (original.nii.gz)
    brain_path  : desired path for the skull-stripped output (brain.nii.gz).
                  HD-BET writes the brain directly to this path.
    device      : "cpu", "cuda", or "mps"
    disable_tta : disable test-time augmentation (faster, small accuracy cost;
                  recommended for CPU, less important on GPU)
    force       : re-run even if outputs already exist
    timeout     : seconds before the HD-BET subprocess is killed (default 300).
                  Prevents a CUDA deadlock from blocking the entire pipeline.

    Returns
    -------
    (brain_path, mask_path)
        brain_path -- skull-stripped volume
        mask_path  -- binary brain mask (.nii.gz), or None if not produced
    """
    # HD-BET writes the mask to <stem>_bet.nii.gz when --save_bet_mask is set
    stem      = brain_path.name.replace(".nii.gz", "").replace(".nii", "")
    mask_path = brain_path.parent / f"{stem}_bet.nii.gz"

    if brain_path.exists() and not force:
        log.info("Skull strip: output exists, skipping (%s)", brain_path.name)
        return brain_path, mask_path if mask_path.exists() else None

    brain_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "hd-bet",
        "-i", str(nifti_path),
        "-o", str(brain_path),
        "-device", device,
        "--save_bet_mask",
    ]
    if disable_tta:
        cmd.append("--disable_tta")

    log.info(
        "Skull strip: running HD-BET (device=%s, timeout=%ds) on %s",
        device, timeout, nifti_path.name,
    )
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        _cleanup_temp_files(brain_path.parent)
        raise RuntimeError(
            f"HD-BET timed out after {timeout}s on {nifti_path.name} -- "
            f"possible CUDA deadlock. Re-run with --force to retry."
        )

    # Always clean up nnU-Net temp files regardless of success or failure
    _cleanup_temp_files(brain_path.parent)

    # HD-BET on Windows sometimes exits with a non-zero code due to an
    # nnU-Net cleanup crash (STATUS_STACK_BUFFER_OVERRUN) even when the
    # brain image was written successfully. We accept the run if the
    # expected output exists.
    if not brain_path.exists():
        raise RuntimeError(
            f"HD-BET failed (exit {result.returncode}) -- "
            f"output not found: {brain_path}\n"
            f"STDOUT: {result.stdout[-800:]}\n"
            f"STDERR: {result.stderr[-400:]}"
        )

    if result.returncode != 0:
        log.warning(
            "Skull strip: HD-BET exited with code %d but output exists -- "
            "treating as success (common on Windows)",
            result.returncode,
        )

    log.info("Skull strip: brain -> %s", brain_path.name)
    return brain_path, mask_path if mask_path.exists() else None
