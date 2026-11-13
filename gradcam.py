"""
gradcam.py — Grad-CAM saliency of the trained networks (Supplementary analysis).

For every test subject and seed, the Grad-CAM map of the Dementia logit
(Dementia subjects) and of the negated logit (CN subjects) is computed with
the best-epoch weights:
  - CNN3D: gradient-weighted activations of the last convolutional block
    (after ReLU), upsampled to the input grid (circularly in theta).
  - Spherical Transformer: the token form of Grad-CAM on the input of the last
    encoder layer; one value per (phi, theta) token, so the map has no radial
    resolution and is broadcast along r.
Each subject's map is scaled to a maximum of 1 and averaged over subjects of
a class, per seed.

Maps are compared in MNI152 template space: Cartesian maps of the
registered-volume CNN3D are native there; spherical maps are placed with the
spherical geometry of the template brain mask (centre, R_edge), which is
exact for the spherical tensors of registered brains and an approximation
(same direction and fractional radius) for native-brain tensors.

Checks: agreement between seeds and between models (Pearson r over brain
voxels), a model-randomization test (randomly initialized CNN3D), and the
mean map value per Harvard-Oxford subcortical structure relative to the
brain mean. Only group-averaged maps are saved; the figure is drawn on the
public MNI152 template.

    python gradcam.py                     # all configurations with finished runs
    python gradcam.py --seeds 0 --limit 8 # quick test
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib

from models import CNN3D, SphericalTransformer
from spherical_transform import spherical_tensor_from_volume
from train import DEVICE, _atomic_write, save_text

REPO = Path(__file__).resolve().parent

# name: (run directory, model, input directory, input kind)
CONFIGS = {
    "CNN3D":                       ("CNN3D",                   "cnn",         "spherical_cache",                "spherical"),
    "CNN3D, spherical of MNI SyN": ("CNN3D_mni-syn-spherical", "cnn",         "registered_cache/spherical_syn", "spherical"),
    "CNN3D, MNI SyN":              ("CNN3D_mni-syn",           "cnn",         "registered_cache/syn",           "volume"),
    "Spherical Transformer":       ("Transformer",             "transformer", "spherical_cache",                "spherical"),
}


# ── Grad-CAM ─────────────────────────────────────────────────────────────────

def _capture(store):
    """Forward hook: keep the layer output and, on backward, its gradient."""
    def hook(_m, _i, out):
        store["a"] = out
        out.register_hook(lambda g: store.__setitem__("g", g))
    return hook


def cam_cnn(model: CNN3D, x: torch.Tensor, sign: torch.Tensor, spherical: bool) -> torch.Tensor:
    layer = model.encoder[-2]
    assert isinstance(layer, nn.ReLU), "expected the ReLU of the last conv block"
    store = {}
    h = layer.register_forward_hook(_capture(store))
    try:
        model.zero_grad(set_to_none=True)
        (sign * model(x)).sum().backward()
    finally:
        h.remove()
    a, g = store["a"], store["g"]                                    # (B, 64, d0, d1, d2)
    cam = F.relu((g.mean(dim=(2, 3, 4), keepdim=True) * a).sum(1, keepdim=True)).detach()
    size = x.shape[1:]
    if not spherical:
        return F.interpolate(cam, size=size, mode="trilinear", align_corners=False)[:, 0]
    # theta (axis 3 here) is periodic: pad one low-resolution cell on each side, upsample, crop
    f = size[1] // cam.shape[3]
    cam = torch.cat([cam[:, :, :, -1:], cam, cam[:, :, :, :1]], dim=3)
    cam = F.interpolate(cam, size=(size[0], size[1] + 2 * f, size[2]), mode="trilinear", align_corners=False)
    return cam[:, 0, :, f:f + size[1]]


def cam_transformer(model: SphericalTransformer, x: torch.Tensor, sign: torch.Tensor) -> torch.Tensor:
    store = {}

    def pre_hook(_m, args):
        store["a"] = args[0]
        args[0].register_hook(lambda g: store.__setitem__("g", g))

    h = model.transformer.layers[-1].register_forward_pre_hook(pre_hook)
    try:
        model.zero_grad(set_to_none=True)
        (sign * model(x)).sum().backward()
    finally:
        h.remove()
    a, g = store["a"][:, 1:], store["g"][:, 1:]                      # drop the class token
    cam = F.relu((g.mean(1, keepdim=True) * a).sum(-1)).detach()     # (B, n_tok)
    cam = cam.view(-1, model.n_phi, model.n_theta)
    cam = cam.repeat_interleave(model.DS, 1).repeat_interleave(model.DS, 2)
    return cam[..., None].expand(-1, -1, -1, x.shape[-1])


def group_maps(model, kind, spherical, paths, labels, batch):
    """Mean max-normalized Grad-CAM per class: {'Dementia': map, 'CN': map}."""
    sums, counts = {}, {}
    for s in range(0, len(paths), batch):
        x = torch.stack([torch.from_numpy(np.load(p)) for p in paths[s:s + batch]]).to(DEVICE)
        y = torch.as_tensor(labels[s:s + batch], device=DEVICE)
        sign = torch.where(y == 1, 1.0, -1.0)                       # evidence for the subject's own class
        cam = cam_cnn(model, x, sign, spherical) if kind == "cnn" else cam_transformer(model, x, sign)
        cam = cam / cam.flatten(1).amax(1).clamp_min(1e-12).view(-1, *[1] * (cam.dim() - 1))
        for cls, name in ((1, "Dementia"), (0, "CN")):
            k = y == cls
            if k.any():
                sums[name] = sums.get(name, 0) + cam[k].sum(0).double().cpu().numpy()
                counts[name] = counts.get(name, 0) + int(k.sum())
    return {n: (sums[n] / counts[n]).astype(np.float32) for n in sums}


def build_model(kind, spherical):
    if kind == "cnn":
        return CNN3D(circular_theta=spherical)
    return SphericalTransformer()


# ── template space ───────────────────────────────────────────────────────────

class TemplateSpace:
    """MNI152 template grid and the spherical geometry of its brain mask."""

    def __init__(self, reg_dir: Path):
        t = nib.load(str(reg_dir / "mni152_2009_2mm.nii.gz"))
        self.img, self.affine = np.asanyarray(t.dataobj).astype(np.float32), t.affine
        self.mask = np.asanyarray(nib.load(str(reg_dir / "mni152_2009_2mm_mask.nii.gz")).dataobj) > 0.5
        _, geo = spherical_tensor_from_volume(self.img, (2.0, 2.0, 2.0), mask=self.mask, return_geometry=True)
        n_phi, n_theta = geo["R_edge"].shape
        n_r = len(geo["k_fracs"])
        # Cell of every brain voxel: same binning as the transform (Eqs. 3-6)
        idx = np.argwhere(self.mask)
        start = np.array([sl.start or 0 for sl in geo["crop"]])
        disp = (idx - start) * geo["h"] - geo["com"]
        r = np.linalg.norm(disp, axis=1).clip(1e-6)
        phi = np.arcsin(np.clip(disp[:, 2] / r, -1, 1))
        theta = np.arctan2(disp[:, 1], disp[:, 0]) % (2 * np.pi)
        i = np.clip(((phi + np.pi / 2) / np.pi * n_phi).astype(int), 0, n_phi - 1)
        j = np.clip((theta / (2 * np.pi) * n_theta).astype(int), 0, n_theta - 1)
        k = np.clip(np.round(r / geo["R_edge"][i, j] * n_r).astype(int) - 1, 0, n_r - 1)
        self.cells = (i, j, k)
        # Sample points of every tensor cell (Eq. 7), as template voxel indices
        self.points_vox = geo["points"] / geo["h"] + start                 # (phi, theta, r, 3)
        self.centre_mni = (geo["com"] / geo["h"] + start) @ self.affine[:3, :3].T + self.affine[:3, 3]
        self.phis, self.thetas = geo["phis"], geo["thetas"]

    def shell_mni(self, k: int) -> np.ndarray:
        """MNI coordinates (mm) of tensor shell k: (phi, theta, 3)."""
        v = self.points_vox[:, :, k]
        return v @ self.affine[:3, :3].T + self.affine[:3, 3]

    def volume_on_shells(self, V: np.ndarray) -> np.ndarray:
        """Template-space map sampled at the tensor cells: (phi, theta, r)."""
        from scipy.ndimage import map_coordinates
        p = self.points_vox.reshape(-1, 3).T
        return map_coordinates(np.nan_to_num(V), p, order=1, mode="constant").reshape(self.points_vox.shape[:3])

    def from_spherical(self, T: np.ndarray) -> np.ndarray:
        out = np.full(self.mask.shape, np.nan, dtype=np.float32)
        out[self.mask] = T[self.cells]
        return out

    def from_volume(self, V: np.ndarray) -> np.ndarray:
        out = np.where(self.mask, V, np.nan).astype(np.float32)
        return out

    def voxel(self, mni_xyz):
        return np.round(np.linalg.solve(self.affine, [*mni_xyz, 1])[:3]).astype(int)


def nanmean_maps(maps) -> np.ndarray:
    """Voxel-wise mean of template-space maps (NaN outside the brain mask)."""
    stack = np.stack(maps)
    out = np.full(stack.shape[1:], np.nan, dtype=np.float32)
    k = np.isfinite(stack).all(0)
    out[k] = stack[:, k].mean(0)
    return out


def corr(a, b):
    k = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[k], b[k])[0, 1])


def region_table(maps: dict, ts: TemplateSpace, reg_dir: Path) -> pd.DataFrame:
    """Mean map value per Harvard-Oxford subcortical structure (left and right
    pooled) divided by the mean over the brain mask."""
    from nilearn import datasets, image
    atlas = datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm", verbose=0)
    img = nib.load(str(reg_dir / "harvard_oxford_sub_thr25_2mm.nii.gz"))
    tmpl = nib.Nifti1Image(ts.img, ts.affine)
    if img.shape != ts.img.shape or not np.allclose(img.affine, ts.affine):
        img = image.resample_to_img(img, tmpl, interpolation="nearest", force_resample=True, copy_header=True)
    lab = np.asanyarray(img.dataobj).astype(int)
    names = [str(n).replace("Left ", "").replace("Right ", "") for n in atlas.labels]
    rows = []
    for region in dict.fromkeys(names[1:]):
        sel = np.isin(lab, [i for i, n in enumerate(names) if n == region and i > 0]) & ts.mask
        rows.append({"region": region, "voxels": int(sel.sum()),
                     **{cfg: float(np.nanmean(m[sel]) / np.nanmean(m)) for cfg, m in maps.items()}})
    return pd.DataFrame(rows).sort_values(next(iter(maps)), ascending=False)


# ── figure ───────────────────────────────────────────────────────────────────

def _style():
    """Elsevier artwork style shared by every figure (Arial 7 pt, 600 dpi)."""
    import matplotlib as mpl
    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 9,
                         "axes.titlesize": 9, "savefig.dpi": 600, "savefig.bbox": "tight",
                         "savefig.pad_inches": 0.02, "pdf.fonttype": 42})


_style()


def save_figure(fig, out: Path):
    """PDF, 600-dpi PNG and LZW TIFF, each written atomically (a crash must not
    leave a truncated figure behind)."""
    import io
    for ext, kw in (("pdf", {}), ("png", {}), ("tif", {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}})):
        buf = io.BytesIO()
        fig.savefig(buf, format="tiff" if ext == "tif" else ext, **kw)
        _atomic_write(out.with_suffix(f".{ext}"), lambda f: f.write(buf.getvalue()))


SLICES = {"Axial": ("z", -14), "Coronal": ("y", -22), "Sagittal": ("x", -26)}   # MNI mm, through the hippocampus


def plot(maps: dict, ts: TemplateSpace, out: Path):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    MM = 1 / 25.4
    cmap = mpl.colormaps["inferno"]
    fig, axes = plt.subplots(len(maps), 3, figsize=(190 * MM, 46 * MM * len(maps)),
                             gridspec_kw={"wspace": 0.04, "hspace": 0.12}, squeeze=False)
    bg_lo, bg_hi = np.percentile(ts.img[ts.mask], [1, 99])
    for row, (cfg, m) in enumerate(maps.items()):
        m = m / np.nanpercentile(m, 99)
        for col, (plane, (axis, mm)) in enumerate(SLICES.items()):
            ax = axes[row, col]
            v = ts.voxel([mm if axis == "x" else 0, mm if axis == "y" else 0, mm if axis == "z" else 0])
            a = "xyz".index(axis)
            take = lambda vol: np.take(vol, v[a], axis=a).T
            ax.imshow(take(ts.img), cmap="gray", origin="lower", vmin=bg_lo, vmax=bg_hi, interpolation="nearest")
            val = np.nan_to_num(np.clip(take(m), 0, 1))
            rgba = cmap(val)
            rgba[..., 3] = np.clip((val - 0.2) / 0.4, 0, 1) * 0.85          # hide the lowest 20 %
            ax.imshow(rgba, origin="lower", interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if row == 0:
                ax.set_title(f"{plane} ({axis} = {mm} mm)")
            if col == 0:
                ax.set_ylabel(cfg.replace(", ", ",\n"), fontsize=9)
            ax.text(0.02, 0.97, f"({chr(97 + 3 * row + col)})", transform=ax.transAxes, fontsize=10,
                    fontweight="bold", color="white", va="top")
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 1), cmap=cmap),
                      ax=axes, fraction=0.015, pad=0.01)
    cb.set_label("Grad-CAM (Dementia subjects, mean;\nscaled to the 99th percentile)")
    save_figure(fig, out)
    plt.close(fig)


# ── 3-D shells ───────────────────────────────────────────────────────────────

SHELLS = (0.3, 0.5, 0.7, 0.9)                         # fractional radii r / R_edge
SHELL_MODELS = ("CNN3D", "CNN3D, spherical of MNI SyN", "CNN3D, MNI SyN")


def cell_maps(arrays, ts: TemplateSpace, seeds) -> dict:
    """Seed-mean Dementia Grad-CAM on the (phi, theta, r) cells of the template
    geometry, divided by its 99th percentile. Spherical maps are already on
    that grid; the Cartesian map is sampled at the cell positions."""
    out = {}
    for cfg, (_run, _kind, _inp, inkind) in CONFIGS.items():
        if f"{cfg}|template|Dementia" not in arrays:
            continue
        if inkind == "spherical":
            S = np.mean([arrays[f"{cfg}|seed{s}|Dementia"] for s in seeds], axis=0)
        else:
            S = ts.volume_on_shells(arrays[f"{cfg}|template|Dementia"])
        out[cfg] = S / np.percentile(S, 99)
    return out


def shell_maps(arrays, ts: TemplateSpace, seeds) -> dict:
    return {cfg: np.clip(S, 0, 1) for cfg, S in cell_maps(arrays, ts, seeds).items()}


def _closed(a):
    """Repeat the theta = 0 column so a surface closes around the azimuth."""
    return np.concatenate([a, a[:, :1]], axis=1)


def plot_shells(shells: dict, ts: TemplateSpace, out: Path):
    """Static figure: one row per model, one column per shell, left-anterior view."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    MM = 1 / 25.4
    cmap = mpl.colormaps["inferno"]
    models = [m for m in SHELL_MODELS if m in shells]
    fig = plt.figure(figsize=(190 * MM, 50 * MM * len(models)))
    for row, cfg in enumerate(models):
        for col, f in enumerate(SHELLS):
            k = int(round(f * shells[cfg].shape[2])) - 1
            xyz = _closed(ts.shell_mni(k))
            ax = fig.add_subplot(len(models), len(SHELLS), row * len(SHELLS) + col + 1, projection="3d")
            ax.plot_surface(xyz[..., 0], xyz[..., 1], xyz[..., 2], facecolors=cmap(_closed(shells[cfg][:, :, k])),
                            rstride=1, cstride=1, lw=0, antialiased=False, shade=False)
            half = 80
            ax.set_xlim(-half, half); ax.set_ylim(-half - 18, half - 18); ax.set_zlim(-half + 15, half + 15)
            ax.set_box_aspect((1, 1, 1), zoom=1.3)
            ax.view_init(elev=12, azim=155)              # from the left, slightly anterior and superior
            ax.set_axis_off()
            if row == 0:
                ax.set_title(f"r = {f:g} R$_{{edge}}$", pad=-4)
            ax.text2D(0.02, 0.92, f"({chr(97 + row * len(SHELLS) + col)})", transform=ax.transAxes,
                      fontsize=10, fontweight="bold")
            if col == 0:
                ax.text2D(-0.08, 0.5, cfg.replace(", ", ",\n"), transform=ax.transAxes, rotation=90,
                          ha="center", va="center", fontsize=9)
    fig.subplots_adjust(wspace=-0.05, hspace=0.0, left=0.05, right=0.9)
    cax = fig.add_axes([0.92, 0.25, 0.012, 0.5])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 1), cmap=cmap), cax=cax)
    cb.set_label("Grad-CAM (Dementia subjects, mean;\nscaled to the 99th percentile)")
    save_figure(fig, out)
    plt.close(fig)


def shells_html(shells: dict, summary: dict, ts: TemplateSpace, out: Path, rays: dict | None = None):
    """Interactive 3-D view: a model menu, shells and the most important rays
    toggled from the legend, and the template brain surface for orientation
    (template and group means only)."""
    import plotly.graph_objects as go
    from skimage.measure import marching_cubes
    from scipy.ndimage import gaussian_filter
    verts, faces, _, _ = marching_cubes(gaussian_filter(ts.mask.astype(np.float32), 1.0), level=0.5, step_size=2)
    verts = verts @ ts.affine[:3, :3].T + ts.affine[:3, 3]
    traces = [go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2], i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                        color="lightgray", opacity=0.08, name="Brain surface (MNI152)", showlegend=True,
                        hoverinfo="skip")]
    owner = [None]
    for cfg, S in shells.items():
        for f in SHELLS:
            k = int(round(f * S.shape[2])) - 1
            xyz = _closed(ts.shell_mni(k))
            traces.append(go.Surface(
                x=xyz[..., 0], y=xyz[..., 1], z=xyz[..., 2], surfacecolor=_closed(S[:, :, k]),
                colorscale="Inferno", cmin=0, cmax=1, showscale=(f == SHELLS[0]), showlegend=True,
                name=f"r = {f:g} R_edge", colorbar=dict(title="Grad-CAM", len=0.6),
                hovertemplate="Grad-CAM %{surfacecolor:.2f}<extra></extra>"))
            owner.append((cfg, f))
        if rays is not None and cfg in rays["ray"]:
            _, _, top = rays["profile"][cfg]
            ends = ts.shell_mni(ts.points_vox.shape[2] - 1)
            xs, ys, zs, col = [], [], [], []
            for i, j in zip(*np.nonzero(top)):
                v = float(np.clip(rays["ray"][cfg][i, j], 0, 1))
                for p in (ts.centre_mni, ends[i, j]):
                    xs.append(p[0]); ys.append(p[1]); zs.append(p[2]); col.append(v)
                xs.append(None); ys.append(None); zs.append(None); col.append(v)
            traces.append(go.Scatter3d(
                x=xs, y=ys, z=zs, mode="lines", name=f"Top {100 * (1 - TOP_RAYS):.0f}% rays", showlegend=True,
                line=dict(color=col, colorscale="Inferno", cmin=0, cmax=1, width=4), hoverinfo="skip"))
            owner.append((cfg, "rays"))

    def visibility(cfg):
        return [True if o is None else (o[0] == cfg) and (True if o[1] == 0.5 else "legendonly") for o in owner]

    def title(cfg):
        r = (summary.get(cfg) or {}).get("seed_agreement_r_mean")
        note = f" (seed agreement r = {r:.2f}{'; unstable, not interpretable' if r is not None and r < 0.3 else ''})" \
            if r is not None else ""
        return f"Grad-CAM, Dementia test subjects: {cfg}{note}"

    first = next(iter(shells))
    for tr, v in zip(traces, visibility(first)):
        tr.visible = v
    fig = go.Figure(traces)
    fig.update_layout(
        title=title(first),
        updatemenus=[dict(type="dropdown", x=0, y=1.08, xanchor="left", buttons=[
            dict(label=cfg, method="update", args=[{"visible": visibility(cfg)}, {"title": title(cfg)}])
            for cfg in shells])],
        scene=dict(xaxis_title="x: L → R (mm)", yaxis_title="y: P → A (mm)", zaxis_title="z: I → S (mm)",
                   aspectmode="data"),
        legend=dict(x=0, y=0.9, title="Click to show / hide"), margin=dict(l=0, r=0, t=60, b=0))
    html = fig.to_html(include_plotlyjs=True, full_html=True)
    _atomic_write(out, lambda fh: fh.write(html.encode("utf-8")))


# ── rays ─────────────────────────────────────────────────────────────────────

POLAR_DEG = 75          # rays with |phi| above this have unreliable R_edge (empty bins); not ranked
TOP_RAYS = 0.95         # quantile defining the most important rays for the depth profile
RAY_MODELS = SHELL_MODELS + ("Spherical Transformer",)


def ray_analysis(cells: dict, ts: TemplateSpace) -> dict:
    """Per-ray importance (mean over r), the depth profile along the top rays,
    and agreement between models on per-ray importance."""
    polar = np.abs(np.degrees(ts.phis)) > POLAR_DEG                  # (phi,)
    ok = np.broadcast_to(~polar[:, None], (len(ts.phis), len(ts.thetas)))
    res = {"ray": {}, "profile": {}, "summary": {}}
    for cfg, S in cells.items():
        ray = S.mean(2)
        ray = ray / np.percentile(ray[ok], 99)
        top = ok & (ray >= np.quantile(ray[ok], TOP_RAYS))
        prof = S[top]                                                 # (n_top, r)
        res["ray"][cfg], res["profile"][cfg] = ray, (prof.mean(0), prof.std(0), top)
        i, j = np.unravel_index(np.argmax(np.where(ok, ray, -np.inf)), ray.shape)
        res["summary"][cfg] = {
            "top_rays": int(top.sum()),
            "peak_depth_r_over_Redge": float((np.argmax(prof.mean(0)) + 1) / S.shape[2]),
            "max_ray_phi_deg": round(float(np.degrees(ts.phis[i])), 1),
            "max_ray_theta_deg": round(float(np.degrees(ts.thetas[j])), 1)}
    names = list(cells)
    res["summary"]["between_models_ray_r"] = {
        f"{a} | {b}": float(np.corrcoef(res["ray"][a][ok], res["ray"][b][ok])[0, 1])
        for n, a in enumerate(names) for b in names[n + 1:]}
    return res


def plot_rays(rays: dict, ts: TemplateSpace, summary: dict, out: Path):
    """Per model: (i) per-ray importance as an elevation x azimuth map, (ii) the
    rays in 3-D coloured by importance, (iii) depth profile along the top rays."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    MM = 1 / 25.4
    cmap = mpl.colormaps["inferno"]
    models = [m for m in RAY_MODELS if m in rays["ray"]]
    fig = plt.figure(figsize=(190 * MM, 47 * MM * len(models)))
    gs = fig.add_gridspec(len(models), 3, width_ratios=[1.25, 1, 1], wspace=0.25, hspace=0.45)
    ends = ts.shell_mni(ts.points_vox.shape[2] - 1)                  # ray end points (r = R_edge)
    c = ts.centre_mni
    sub = [(i, j) for i in range(1, len(ts.phis), 2) for j in range(0, len(ts.thetas), 2)]
    letters = iter("abcdefghijklmnop")
    for row, cfg in enumerate(models):
        ray = rays["ray"][cfg]
        unstable = (summary.get(cfg) or {}).get("seed_agreement_r_mean", 1) < 0.3

        ax = fig.add_subplot(gs[row, 0])
        ax.imshow(np.clip(ray, 0, 1), origin="lower", cmap=cmap, vmin=0, vmax=1, extent=[0, 360, -90, 90],
                  aspect="auto", interpolation="nearest")
        for y0 in (POLAR_DEG, -90):
            ax.axhspan(y0, y0 + 90 - POLAR_DEG, color="#9a9a9a", alpha=0.85, lw=0)
        ax.set_xticks([0, 90, 180, 270], ["R", "A", "L", "P"])
        ax.set_yticks([-90, 0, 90], ["I", "0", "S"])
        ax.set_ylabel(cfg.replace(", ", ",\n") + ("\n(unstable across seeds)" if unstable else ""), fontsize=9)
        if row == len(models) - 1:
            ax.set_xlabel("Azimuth θ")
        ax.text(0.0, 1.03, f"({next(letters)})", transform=ax.transAxes, fontsize=10, fontweight="bold",
                ha="left", va="bottom")
        heads = [ax]

        ax = fig.add_subplot(gs[row, 1], projection="3d")
        order = sorted(sub, key=lambda ij: ray[ij])                       # draw the strongest last
        for i, j in order:
            v = float(np.clip(ray[i, j], 0, 1))
            polar = abs(np.degrees(ts.phis[i])) > POLAR_DEG
            col = "#bdbdbd" if polar else cmap(v)
            ax.plot(*zip(c, ends[i, j]), color=col, lw=0.3 + 1.2 * v, alpha=0.15 + 0.85 * v)
        ax.set_xlim(-75, 75); ax.set_ylim(-93, 57); ax.set_zlim(-60, 90)
        ax.set_box_aspect((1, 1, 1), zoom=1.35)
        ax.view_init(elev=12, azim=155)
        ax.set_axis_off()
        ax.text2D(0.0, 1.03, f"({next(letters)})", transform=ax.transAxes, fontsize=10, fontweight="bold",
                  ha="left", va="bottom")
        heads.append(ax)

        ax = fig.add_subplot(gs[row, 2])
        m, s, top = rays["profile"][cfg]
        r = np.arange(1, len(m) + 1) / len(m)
        if cfg == "Spherical Transformer":
            ax.text(0.5, 0.5, "No radial resolution\n(one token per ray)", ha="center", va="center",
                    transform=ax.transAxes, color="#555555")
            ax.set_yticks([])
        else:
            ax.fill_between(r, m - s, m + s, color=cmap(0.6), alpha=0.25, lw=0)
            ax.plot(r, m, color=cmap(0.6), lw=1.2, label=f"top {100 * (1 - TOP_RAYS):.0f}% of rays")
            ax.set_ylim(bottom=0)
            ax.set_ylabel("Grad-CAM (scaled)")
            ax.axvline(r[np.argmax(m)], color="#555555", lw=0.6, ls="--")
        ax.set_xlim(0, 1)
        if row == len(models) - 1:
            ax.set_xlabel("r / R$_{edge}$")
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(0.0, 1.03, f"({next(letters)})", transform=ax.transAxes, fontsize=10, fontweight="bold",
                ha="left", va="bottom")
        heads.append(ax)
        if row == 0:
            top_row = heads
    fig.subplots_adjust(bottom=0.13, top=0.94)
    # column headings above the first row, clear of the panel letters
    for ax, title in zip(top_row, ("Importance per ray (mean over r)", "Rays coloured by importance",
                                   f"Depth profile, top {100 * (1 - TOP_RAYS):.0f}% of rays")):
        b = ax.get_position()
        fig.text((b.x0 + b.x1) / 2, b.y1 + 0.022, title, ha="center", va="bottom", fontsize=9)
    cax = fig.add_axes([0.13, 0.035, 0.25, 0.008])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 1), cmap=cmap), cax=cax,
                      orientation="horizontal")
    cb.set_label(f"Ray importance (scaled to the 99th percentile)\ngrey: |φ| > {POLAR_DEG}°, not ranked")
    save_figure(fig, out)
    plt.close(fig)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=REPO / "cohort" / "manifest.csv")
    ap.add_argument("--runs-dir", type=Path, default=REPO / "runs")
    ap.add_argument("--registered-dir", type=Path, default=REPO / "registered_cache")
    ap.add_argument("--out-dir", type=Path, default=REPO / "results" / "gradcam")
    ap.add_argument("--fig", type=Path, default=REPO / "figures" / "figS_gradcam")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="Only the first N test subjects (testing)")
    ap.add_argument("--plot-only", action="store_true", help="Redraw the figures from the saved maps")
    ap.add_argument("--figures", nargs="+", choices=["slices", "shells", "rays", "html"],
                    default=["slices", "shells", "rays", "html"],
                    help="Figures to draw (one per process keeps each process short on an unstable machine)")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig.parent.mkdir(parents=True, exist_ok=True)
    shells_fig = args.fig.with_name(args.fig.name + "_shells")

    def draw(arrays, summary, ts):
        cells = cell_maps(arrays, ts, summary["seeds"])
        rays = ray_analysis(cells, ts)
        save_text(json.dumps(rays["summary"], indent=2), args.out_dir / "gradcam_rays.json")
        print(json.dumps(rays["summary"], indent=2))
        clipped = {c: np.clip(S, 0, 1) for c, S in cells.items()}
        if "slices" in args.figures:
            plot({k.split("|")[0]: v for k, v in arrays.items() if k.endswith("|template|Dementia")}, ts, args.fig)
        if "shells" in args.figures:
            plot_shells(clipped, ts, shells_fig)
        if "rays" in args.figures:
            plot_rays(rays, ts, summary, args.fig.with_name(args.fig.name + "_rays"))
        if "html" in args.figures:
            shells_html(clipped, summary, ts, shells_fig.with_suffix(".html"), rays)

    if args.plot_only:
        ts = TemplateSpace(args.registered_dir)
        with np.load(args.out_dir / "gradcam_maps.npz") as z:
            arrays = dict(z)
        draw(arrays, json.loads((args.out_dir / "gradcam_summary.json").read_text()), ts)
        return

    test = pd.read_csv(args.manifest).query("split == 'test'").reset_index(drop=True)
    if args.limit:
        test = test.iloc[: args.limit]
    labels = test["Diagnosis"].eq("Dementia").astype(int).to_numpy()
    ts = TemplateSpace(args.registered_dir)
    to_template = lambda kind, M: ts.from_spherical(M) if kind == "spherical" else ts.from_volume(M)

    per_seed, mean_maps, summary = {}, {}, {"n_test": len(test), "n_dementia": int(labels.sum()), "seeds": args.seeds}
    for cfg, (run, kind, inp, inkind) in CONFIGS.items():
        ckpts = [args.runs_dir / run / f"seed{s}" / "best.pt" for s in args.seeds]
        paths = [REPO / inp / f"{p}.npy" for p in test["PTID"]]
        if not all(c.exists() for c in ckpts) or not all(p.exists() for p in paths):
            print(f"{cfg}: runs or inputs missing -- skipped")
            continue
        print(f"{cfg} ...", flush=True)
        maps = []
        for c in ckpts:
            model = build_model(kind, inkind == "spherical").to(DEVICE)
            model.load_state_dict(torch.load(c, map_location=DEVICE, weights_only=True))
            model.eval()
            maps.append(group_maps(model, kind, inkind == "spherical", paths, labels, args.batch))
        per_seed[cfg] = maps
        tm = [to_template(inkind, m["Dementia"]) for m in maps]
        mean_maps[cfg] = nanmean_maps(tm)
        rs = [corr(tm[a], tm[b]) for a in range(len(tm)) for b in range(a + 1, len(tm))]
        summary[cfg] = {"seed_agreement_r_mean": float(np.mean(rs)) if rs else None,
                        "seed_agreement_r_min": float(np.min(rs)) if rs else None}

        if cfg == "CNN3D":   # model-randomization test: random weights should not give the same map
            rand = []
            for s in args.seeds:
                torch.manual_seed(1000 + s)
                model = build_model(kind, True).to(DEVICE).eval()
                rand.append(to_template(inkind, group_maps(model, kind, True, paths, labels, args.batch)["Dementia"]))
            summary[cfg]["randomized_weights_r"] = corr(mean_maps[cfg], nanmean_maps(rand))

    names = list(mean_maps)
    summary["between_models_r"] = {f"{a} | {b}": corr(mean_maps[a], mean_maps[b])
                                   for i, a in enumerate(names) for b in names[i + 1:]}
    regions = region_table(mean_maps, ts, args.registered_dir)

    # Group averages only (no per-subject maps): native-input grid, per seed and class
    arrays = {f"{cfg}|seed{s}|{cls}": m[cls] for cfg, ms in per_seed.items()
              for s, m in zip(args.seeds, ms) for cls in m}
    arrays.update({f"{cfg}|template|Dementia": m for cfg, m in mean_maps.items()})
    import io
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    _atomic_write(args.out_dir / "gradcam_maps.npz", lambda f: f.write(buf.getvalue()))
    save_text(json.dumps(summary, indent=2), args.out_dir / "gradcam_summary.json")
    save_text(regions.to_csv(index=False, float_format="%.3f"), args.out_dir / "gradcam_regions.csv")
    draw(arrays, summary, ts)

    print(json.dumps(summary, indent=2))
    print(regions.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
