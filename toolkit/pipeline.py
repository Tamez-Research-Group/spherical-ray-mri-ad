"""Toolkit core: one skull-stripped T1 volume -> spherical tensor -> model outputs.

The steps are the paper's own code, not re-implementations:
  preprocessing/harmonize.py   N4 bias-field correction, 1.2 mm isotropic (Methods 2.2)
  spherical_transform.py       spherical ray representation, 36 x 72 x 48 (Methods 2.3)
  models.py                    CNN3D and Spherical Transformer (Methods 2.4-2.5)
  gradcam.py                   Grad-CAM of the Dementia logit (supplementary)
Trained weights come from the Hugging Face repository of the paper; decision
thresholds are the per-seed validation thresholds (Youden's J, Methods 2.7)
stored in results/metrics_per_seed.csv.

Research use only. This is not a medical device and its output is not a diagnosis.
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from models import CNN3D, SphericalTransformer                          # noqa: E402
from preprocessing.harmonize import n4_correct, resample_common         # noqa: E402
from spherical_transform import load_nifti_as_ras, spherical_tensor_from_volume  # noqa: E402

HF_REPO = "lalodeavilaarm/adni-spherical-classification"
SEEDS = (0, 1, 2, 3, 4)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# key: (label, Hugging Face family, results config prefix, constructor)
MODELS = {
    "cnn":         ("CNN3D", "CNN3D", "CNN3D", lambda: CNN3D(circular_theta=True)),
    "transformer": ("Spherical Transformer", "Transformer", "Spherical Transformer", SphericalTransformer),
}
WEIGHTS = {"best": ("best epoch", "best epoch"), "avg": ("top-5 average", "top-K avg")}


# ── preprocessing and transform ──────────────────────────────────────────────

@dataclass
class Subject:
    name: str
    tensor: np.ndarray                      # (36, 72, 48) float32
    geometry: dict                          # see spherical_tensor_from_volume(return_geometry=True)
    spacing: tuple                          # native voxel spacing (mm), RAS order
    protocol: str                           # acquisition-protocol proxy used in the paper
    qc: list = field(default_factory=list)  # warnings for the user
    seconds: float = 0.0


def qc_checks(vol: np.ndarray, spacing) -> list[str]:
    """Checks that the input looks like what the models were trained on:
    a skull-stripped T1-weighted brain (zero background)."""
    warn = []
    nonzero = float(np.mean(vol > 0))
    if np.any(vol < 0):
        warn.append("The volume has negative intensities; the pipeline expects a skull-stripped "
                    "T1 with zero background (values <= 0 are treated as background).")
    if nonzero > 0.6:
        warn.append(f"{nonzero:.0%} of the voxels are non-zero. The input may not be skull-stripped; "
                    "the models were trained on HD-BET skull-stripped volumes.")
    brain_ml = float(np.sum(vol > 0) * np.prod(spacing)) / 1000
    if not 700 <= brain_ml <= 2200:
        warn.append(f"Brain-mask volume is {brain_ml:.0f} mL, outside the usual 700-2200 mL; "
                    "check the skull stripping and the voxel size in the header.")
    if max(spacing) > 2.0:
        warn.append(f"Coarse voxels ({' x '.join(f'{s:.2f}' for s in spacing)} mm); the models were "
                    "trained on 1.0-1.2 mm MPRAGE.")
    return warn


def _n4(vol, spacing, attempts: int = 3):
    """N4 with retries: SimpleITK's N4 occasionally raises a spurious
    'collapse point ... outside the parametric domain' error that does not
    recur on a second call (the study's timing benchmark retries likewise)."""
    for i in range(attempts):
        try:
            return n4_correct(vol, spacing)
        except RuntimeError:
            if i == attempts - 1:
                raise


def process(path: str | Path, harmonize: bool = True) -> Subject:
    """Load a skull-stripped NIfTI, harmonize it and compute its spherical tensor."""
    t0 = time.perf_counter()
    path = Path(path)
    vol, nii = load_nifti_as_ras(path)
    spacing = tuple(float(z) for z in nii.header.get_zooms()[:3])
    if not np.any(vol > 0):
        raise ValueError("The volume is empty (no voxel above zero).")
    qc = qc_checks(vol, spacing)
    protocol = "slice1.2" if round(spacing[0], 1) == 1.2 else "iso1.0"
    v, sp = (resample_common(_n4(vol, spacing), spacing) if harmonize else (vol, spacing))
    out = spherical_tensor_from_volume(v, sp, return_geometry=True)
    if out is None:
        raise ValueError("The spherical transform failed (empty brain mask).")
    T, geo = out
    name = path.name.removesuffix(".gz").removesuffix(".nii")
    return Subject(name, T, geo, spacing, protocol, qc, time.perf_counter() - t0)


# ── models ───────────────────────────────────────────────────────────────────

def _weights_file(family: str, seed: int, weights: str) -> Path:
    """Local copy if SPHERICAL_WEIGHTS_DIR is set (same layout as the Hugging
    Face repository: weights/<family>/seed<k>/<best|avg>.safetensors),
    otherwise downloaded once from Hugging Face into its local cache."""
    rel = f"weights/{family}/seed{seed}/{weights}.safetensors"
    local = os.environ.get("SPHERICAL_WEIGHTS_DIR")
    if local:
        p = Path(local) / rel
        if not p.exists():
            raise FileNotFoundError(p)
        return p
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(HF_REPO, rel))


@lru_cache(maxsize=32)
def load_model(model: str, seed: int, weights: str) -> torch.nn.Module:
    from safetensors.torch import load_file
    _, family, _, build = MODELS[model]
    net = build()
    net.load_state_dict(load_file(str(_weights_file(family, seed, weights))))
    return net.to(DEVICE).eval()


@lru_cache(maxsize=1)
def _threshold_table() -> pd.DataFrame:
    return pd.read_csv(REPO / "results" / "metrics_per_seed.csv", encoding="utf-8-sig")


def threshold(model: str, seed: int, weights: str) -> float:
    cfg = f"{MODELS[model][2]} ({WEIGHTS[weights][1]})"
    t = _threshold_table()
    row = t[(t["config"] == cfg) & (t["seed"] == seed)]
    if len(row) != 1:
        raise KeyError(f"no validation threshold for {cfg}, seed {seed}")
    return float(row["threshold"].iloc[0])


def predict(T: np.ndarray, model: str, weights: str = "best", seeds=SEEDS) -> pd.DataFrame:
    """Probability of Dementia per seed and the call at that seed's validation threshold."""
    x = torch.from_numpy(T[None]).float().to(DEVICE)
    rows = []
    for s in seeds:
        with torch.no_grad():
            p = float(torch.sigmoid(load_model(model, s, weights)(x)).item())
        thr = threshold(model, s, weights)
        rows.append({"seed": s, "P(Dementia)": p, "threshold": thr,
                     "call": "Dementia" if p >= thr else "CN"})
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> dict:
    n_dem = int((df["call"] == "Dementia").sum())
    n = len(df)
    call = "Dementia" if n_dem > n / 2 else "CN"
    agree = n_dem if call == "Dementia" else n - n_dem
    return {"call": call, "agree": agree, "n": n, "mean_p": float(df["P(Dementia)"].mean())}


def scaled_cam(cam: np.ndarray) -> np.ndarray:
    """Grad-CAM divided by its 99th percentile and clipped to [0, 1], the
    scaling of the paper's Grad-CAM figures."""
    q = float(np.percentile(cam, 99))
    return np.clip(cam / q, 0, 1) if q > 0 else cam


POLAR_DEG, TOP_RAYS = 75, 0.95            # as in gradcam.py (supplementary ray analysis)


def ray_importance(cam: np.ndarray, phis: np.ndarray):
    """Per-ray importance (mean over r, scaled to the 99th percentile of the
    non-polar rays), the mask of non-polar rays, and the depth profile
    (mean, SD over the top 5% of rays) of the 99th-percentile-scaled map."""
    ok = np.broadcast_to((np.abs(np.degrees(phis)) <= POLAR_DEG)[:, None], cam.shape[:2])
    ray = cam.mean(2)
    q = np.percentile(ray[ok], 99)
    ray = ray / q if q > 0 else ray
    top = ok & (ray >= np.quantile(ray[ok], TOP_RAYS))
    prof = (cam / max(float(np.percentile(cam, 99)), 1e-12))[top]
    return ray, ok, prof.mean(0), prof.std(0)


# ── comparison with the ADNI groups ──────────────────────────────────────────

REFERENCE_FILE = Path(__file__).resolve().parent / "reference" / "adni_reference.npz"
D_MIN = 0.15                              # cells with |d| below this are ignored (Fig. 4 caption)


class Reference:
    """Aggregate ADNI training statistics (toolkit/build_reference.py)."""

    def __init__(self, data: dict):
        self.data = data
        self.d = np.asarray(data["d"], np.float64)
        phis = (np.arange(self.d.shape[0]) + 0.5) / self.d.shape[0] * 180 - 90
        polar_ok = (np.abs(phis) <= POLAR_DEG)[:, None, None]
        self.mask = (np.abs(self.d) >= D_MIN) & polar_ok

    @classmethod
    def load(cls, path: Path = REFERENCE_FILE) -> "Reference":
        with np.load(path) as z:
            return cls({k: z[k] for k in z.files})

    def mean(self, dx: str, protocol: str) -> np.ndarray:
        return np.asarray(self.data[f"mean_{dx}_{protocol}"], np.float64)

    def n(self, dx: str, protocol: str) -> int:
        return int(self.data[f"n_{dx}_{protocol}"])

    def deviation(self, T: np.ndarray, protocol: str) -> np.ndarray:
        """z = (CN mean - scan) / CN SD, within the scan's protocol group.
        Positive (red) = darker than CN, the direction of the Dementia effect."""
        sd = np.maximum(np.asarray(self.data[f"sd_CN_{protocol}"], np.float64), 1e-3)
        return (self.mean("CN", protocol) - T) / sd

    def pattern_score(self, T: np.ndarray, protocol: str) -> float:
        """Correlation, over the cells where the groups differ (|d| >= 0.15,
        non-polar), between the scan's deviation from CN and the group
        effect map d (CN - Dementia). Near 0: no resemblance; positive: the
        scan departs from CN in the same places and direction as Dementia."""
        z, m = self.deviation(T, protocol)[self.mask], self.d[self.mask]
        return float(np.corrcoef(z, m)[0, 1])


def gradcam(T: np.ndarray, model: str, weights: str = "best", seeds=SEEDS) -> np.ndarray:
    """Grad-CAM of the Dementia logit on the spherical grid, each seed's map
    scaled to max 1, then averaged over seeds (as in the paper's group maps)."""
    from gradcam import cam_cnn, cam_transformer
    x = torch.from_numpy(T[None]).float().to(DEVICE)
    sign = torch.ones(1, device=DEVICE)
    maps = []
    for s in seeds:
        net = load_model(model, s, weights)
        with torch.enable_grad():
            cam = (cam_cnn(net, x, sign, spherical=True) if model == "cnn"
                   else cam_transformer(net, x, sign))[0].cpu().numpy()
        net.zero_grad(set_to_none=True)
        maps.append(cam / cam.max() if cam.max() > 0 else cam)
    return np.mean(maps, axis=0).astype(np.float32)
