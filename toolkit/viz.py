"""Toolkit figures. They follow the paper's figure conventions:
- spherical-tensor maps: elevation x azimuth, axes I/0/S and R/A/L/P, grey,
  fixed range -3 to 1.2 (Fig. S cn_vs_ad);
- differences: RdBu, red = darker / lower than the normal (CN) average
  (Fig. 4, Fig. S cn_vs_ad);
- Grad-CAM: inferno, scaled to the 99th percentile, grey polar caps
  |phi| > 75 deg (Figs. S gradcam, S gradcam_rays, S gradcam_shells);
- slices: centre-of-mass planes in standard display (figures_spherical.slice_panel).
Every function takes `dark` so the figures match the app's theme."""
from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl                              # noqa: E402
import matplotlib.pyplot as plt                       # noqa: E402
import plotly.graph_objects as go                     # noqa: E402
from scipy.ndimage import map_coordinates            # noqa: E402

import pipeline as P                                  # noqa: E402  (puts the repo on sys.path)
from figures_spherical import VIEWS, _in_plane, slice_panel   # noqa: E402

ORANGE, BLUE = "#e2622b", "#3b82f6"
INFERNO = mpl.colormaps["inferno"]
T_LIM = (-3.0, 1.2)            # tensor display range (Fig. S cn_vs_ad)
D_LIM, D_HIDE = 1.0, 0.15      # Cohen's d range and hidden |d| (Fig. 4)
Z_LIM, Z_HIDE = 3.0, 0.5       # z-score range and hidden |z| for a single scan
FLAT = dict(ambient=0.9, diffuse=0.25, specular=0.05, roughness=0.9)
THEMES = {False: dict(bg="#ffffff", fg="#1f2937", muted="#6b7280", grid="#e5e7eb", poly="#9a9a9a"),
          True: dict(bg="#0e1117", fg="#e5e7eb", muted="#9ca3af", grid="#262b36", poly="#4b5563")}
CAM_LABEL = "Model attention (Grad-CAM)"


def _style(dark: bool) -> dict:
    t = THEMES[dark]
    plt.rcParams.update({
        "figure.facecolor": t["bg"], "axes.facecolor": t["bg"], "savefig.facecolor": t["bg"],
        "text.color": t["fg"], "axes.labelcolor": t["fg"], "axes.edgecolor": t["muted"],
        "xtick.color": t["muted"], "ytick.color": t["muted"], "font.size": 9,
        "axes.titlesize": 10, "axes.linewidth": 0.6, "figure.dpi": 130,
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"]})
    return t


def _k(f: float, n_r: int) -> int:
    return min(max(int(round(f * n_r)) - 1, 0), n_r - 1)


def _angles(ax, left: bool = True, bottom: bool = True):
    ax.set_xticks([0, 90, 180, 270], ["R", "A", "L", "P"] if bottom else [])
    ax.set_yticks([-90, 0, 90], ["I", "0", "S"] if left else [])


def _map(ax, M, cmap, vmin, vmax):
    return ax.imshow(M, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, extent=[0, 360, -90, 90],
                     aspect="auto", interpolation="nearest")


def _fix_labels(ax, t):
    ax.xaxis.label.set_color(t["fg"]), ax.yaxis.label.set_color(t["fg"])


# ── slices: the three standard views ─────────────────────────────────────────

def _on_plane(geo, values, view, polar_nan=False):
    """Sample a (phi, theta, r) map at every voxel of the centre-of-mass slice
    of `view` (trilinear in the map, periodic in theta). Returns the slice in
    (horizontal, vertical) array order; NaN outside the brain."""
    vol, h, c = geo["vol_n"], geo["h"], geo["com"]
    axis, (hx, vy) = VIEWS[view][0], VIEWS[view][1]
    n_phi, n_theta, n_r = values.shape
    I, J = np.meshgrid(np.arange(vol.shape[hx]), np.arange(vol.shape[vy]), indexing="ij")
    vox = np.zeros(I.shape + (3,))
    vox[..., hx], vox[..., vy] = I, J
    vox[..., axis] = int(round(c[axis] / h[axis]))
    p = vox * h - c
    r = np.linalg.norm(p, axis=-1) + 1e-9
    phi = np.arcsin(np.clip(p[..., 2] / r, -1, 1))
    theta = np.arctan2(p[..., 1], p[..., 0]) % (2 * np.pi)
    fi = (phi + np.pi / 2) / np.pi * n_phi - 0.5
    ti = theta / (2 * np.pi) * n_theta - 0.5 + 1                       # +1: one padded column
    Re = np.concatenate([geo["R_edge"][:, -1:], geo["R_edge"], geo["R_edge"][:, :1]], axis=1)
    R_edge = map_coordinates(Re, [fi.ravel(), ti.ravel()], order=1, mode="nearest").reshape(r.shape)
    ki = r / R_edge * n_r - 1
    V = np.concatenate([values[:, -1:], values, values[:, :1]], axis=1)
    out = map_coordinates(V, [fi.ravel(), ti.ravel(), np.clip(ki, 0, n_r - 1).ravel()],
                          order=1, mode="nearest").reshape(r.shape)
    sl = [slice(None)] * 3
    sl[axis] = int(round(c[axis] / h[axis]))
    keep = (vol[tuple(sl)] != vol[0, 0, 0]) & (ki <= n_r - 0.5)
    if polar_nan:
        keep &= np.abs(np.degrees(phi)) <= P.POLAR_DEG
    return np.where(keep, out, np.nan)


def _overlay(ax, geo, view, M, cmap, vmin, vmax, alpha):
    axis, (hx, vy) = VIEWS[view][0], VIEWS[view][1]
    h, c = geo["h"], geo["com"]
    ext_x = np.arange(M.shape[0]) * h[hx] - c[hx]
    ext_y = np.arange(M.shape[1]) * h[vy] - c[vy]
    rgba = mpl.colormaps[cmap](mpl.colors.Normalize(vmin, vmax)(np.nan_to_num(M.T)))
    rgba[..., 3] = np.where(np.isnan(M.T), 0, alpha(np.nan_to_num(M.T)))
    ax.imshow(rgba, origin="lower", extent=[ext_x[0], ext_x[-1], ext_y[0], ext_y[-1]],
              interpolation="bilinear")


def _dots(ax, geo, view, M, lim, hide, cmap="RdBu_r"):
    axis, (hx, vy) = VIEWS[view][0], VIEWS[view][1]
    Pt, c = geo["points"], geo["com"]
    show = _in_plane(geo, axis) & (np.abs(M) >= hide)
    pts, val = Pt[show], M[show]
    o = np.argsort(np.abs(val))
    return ax.scatter(pts[o, hx] - c[hx], pts[o, vy] - c[vy], c=val[o], cmap=cmap, vmin=-lim, vmax=lim,
                      s=3, linewidths=0)


THREE_VIEW_MODES = {"anatomy": "Brain", "pattern": "Typical dementia pattern",
                    "scan": "This scan vs normal", "tensor": "Spherical sampling"}


def three_views(geo, mode="anatomy", dark=False, d=None, z=None, T=None, cam=None, rays=False):
    """Axial, coronal and sagittal slices through the brain's centre of mass.
    mode: 'anatomy' (smoothed brain edge and centre; with rays=True also the rays),
    'pattern' (where the ADNI groups differ, Cohen's d, as in Fig. 4),
    'scan' (where this scan differs from the normal average, z-score),
    'tensor' (the values the networks see, drawn back at their positions).
    cam (scaled Grad-CAM) adds the model-attention overlay to 'anatomy'."""
    t = _style(dark)
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.9))
    sc = None
    for ax, view in zip(axes, VIEWS):
        if mode == "tensor":
            slice_panel(ax, geo, view, mode=None, anatomy=False)
            axis, (hx, vy) = VIEWS[view][0], VIEWS[view][1]
            near, Pt, c = _in_plane(geo, axis), geo["points"], geo["com"]
            sc = ax.scatter(Pt[near][:, hx] - c[hx], Pt[near][:, vy] - c[vy], c=T[near], cmap="gray",
                            vmin=T_LIM[0], vmax=T_LIM[1], s=4, linewidths=0)
        elif mode in ("pattern", "scan"):
            slice_panel(ax, geo, view, mode=None)
            sc = _dots(ax, geo, view, *((d, D_LIM, D_HIDE) if mode == "pattern" else (z, Z_LIM, Z_HIDE)))
        elif cam is not None:
            slice_panel(ax, geo, view, mode=None)
            _overlay(ax, geo, view, _on_plane(geo, cam, view, polar_nan=True), "inferno", 0, 1,
                     lambda v: np.where(v < 0.15, 0, 0.25 + 0.6 * v))
            slice_panel(ax, geo, view, mode="rays" if rays else "ring", anatomy=False)
        else:
            slice_panel(ax, geo, view, mode="rays" if rays else "ring")
        ax.set_title(VIEWS[view][3])
        _fix_labels(ax, t)
    label = {"tensor": "Normalized intensity", "pattern": "Group difference (Cohen's d)\nred: lower in dementia",
             "scan": "Difference from normal (z)\nred: darker than normal"}.get(mode)
    if cam is not None and mode == "anatomy":
        sc, label = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 1), cmap=INFERNO), CAM_LABEL
    if label:
        fig.colorbar(sc, ax=list(axes), fraction=0.015, pad=0.015).set_label(label)
    else:
        fig.tight_layout()
    return fig


# ── interactive 3-D ──────────────────────────────────────────────────────────

def _scene(fig, dark, height=600, eye=(-1.6, 1.1, 0.6)):
    t = THEMES[dark]
    ax = dict(color=t["muted"], gridcolor=t["grid"], backgroundcolor=t["bg"], zerolinecolor=t["grid"],
              showbackground=True)
    fig.update_layout(
        paper_bgcolor=t["bg"], plot_bgcolor=t["bg"], font=dict(color=t["fg"], size=12),
        scene=dict(xaxis=dict(title="L → R (mm)", **ax), yaxis=dict(title="P → A (mm)", **ax),
                   zaxis=dict(title="I → S (mm)", **ax), aspectmode="data",
                   camera=dict(eye=dict(x=eye[0], y=eye[1], z=eye[2]))),
        margin=dict(l=0, r=0, t=10, b=0), height=height,
        legend=dict(orientation="h", y=1.0, x=0, bgcolor="rgba(0,0,0,0)"))
    return fig


def _slice_surface(geo, axis, step=2):
    vol, h, c = geo["vol_n"], geo["h"], geo["com"]
    idx = [slice(None, None, step)] * 3
    idx[axis] = int(round(c[axis] / h[axis]))
    img = vol[tuple(idx)].astype(float)
    others = [a for a in range(3) if a != axis]
    g = [np.arange(0, vol.shape[a], step) * h[a] - c[a] for a in others]
    A, B = np.meshgrid(g[0], g[1], indexing="ij")
    xyz = [None] * 3
    xyz[others[0]], xyz[others[1]], xyz[axis] = A, B, np.zeros_like(A)
    keep = img != vol[0, 0, 0]
    xyz = [np.where(keep, q, np.nan) for q in xyz]
    return go.Surface(x=xyz[0], y=xyz[1], z=xyz[2], surfacecolor=img, cmin=T_LIM[0], cmax=T_LIM[1],
                      colorscale="Gray", showscale=False, hoverinfo="skip", name="slice",
                      lighting=dict(ambient=1.0, diffuse=0, specular=0))


def _segments(E, sel):
    """x, y, z arrays of line segments from the centre to each selected ray end."""
    pts = E[sel]
    return [np.column_stack([np.zeros(len(pts)), pts[:, a], np.full(len(pts), np.nan)]).ravel()
            for a in range(3)]


def rays_3d(geo, dark=False, step=3, planes=("axial", "sagittal"), edge=True, cam=None):
    """Rays from the centre of mass to the smoothed edge, the edge surface and
    slices. With `cam` (scaled Grad-CAM) the rays are coloured by importance
    (mean over each ray), as in Fig. S gradcam_rays; polar rays are grey."""
    t = THEMES[dark]
    Pt, c = geo["points"], geo["com"]
    E = Pt[:, :, -1] - c
    fig = go.Figure()
    for v in planes:
        fig.add_trace(_slice_surface(geo, VIEWS[v][0]))
    if edge:
        X, Y, Z = (np.concatenate([E[:, :, a], E[:, :1, a]], axis=1) for a in range(3))
        fig.add_trace(go.Surface(x=X, y=Y, z=Z, surfacecolor=np.concatenate(
            [geo["R_edge"], geo["R_edge"][:, :1]], axis=1), colorscale="Oranges", opacity=0.12,
            lighting=FLAT, showscale=cam is None, colorbar=dict(title="Edge distance (mm)", len=0.5),
            name="Brain edge", showlegend=True, hovertemplate="%{surfacecolor:.0f} mm<extra></extra>"))
    grid = np.zeros(E.shape[:2], bool)
    grid[::step, ::step] = True
    if cam is None:
        xs, ys, zs = _segments(E, grid)
        fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color=BLUE, width=2),
                                   name="Rays", hoverinfo="skip"))
    else:
        ray, ok, _, _ = P.ray_importance(cam, geo["phis"])
        ray = np.clip(ray, 0, 1)
        xs, ys, zs = _segments(E, grid & ~ok)
        fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color=t["poly"], width=1.5),
                                   name="Rays near the poles (not ranked)", hoverinfo="skip"))
        sel = grid & ok
        xs, ys, zs = _segments(E, sel)
        col = np.repeat(ray[sel], 3)
        fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", name="Rays, coloured by attention",
                                   line=dict(color=col, colorscale="Inferno", cmin=0, cmax=1, width=4,
                                             colorbar=dict(title="Attention", len=0.5)), hoverinfo="skip"))
    ends = E[grid]
    fig.add_trace(go.Scatter3d(x=ends[:, 0], y=ends[:, 1], z=ends[:, 2], mode="markers",
                               marker=dict(size=2, color=ORANGE), name="Ray ends", hoverinfo="skip"))
    fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode="markers", name="Centre of mass",
                               marker=dict(size=6, color=ORANGE, line=dict(color="white", width=1))))
    return _scene(fig, dark, 620)


def shells_3d(values, geo, dark=False, radii=(0.5, 0.9), kind="intensity", cutaway=True):
    """Nested shells at fractions of R_edge. kind: 'intensity' (grey, -3..1.2),
    'gradcam' (inferno, 0..1, already scaled) or 'deviation' (RdBu, z in ±3).
    Without the cut-away, outer shells are translucent so inner ones stay visible."""
    Pt, c = geo["points"], geo["com"]
    n_r = values.shape[2]
    th = geo["thetas"]
    hide = (np.cos(th) > 0) & (np.sin(th) > 0)                     # right-anterior quadrant
    scale = {"intensity": ("Gray", *T_LIM, "Intensity"),
             "gradcam": ("Inferno", 0.0, 1.0, "Attention"),
             "deviation": ("RdBu_r", -Z_LIM, Z_LIM, "Difference from normal (z)")}[kind]
    radii = sorted(radii)
    fig = go.Figure()
    for i, f in enumerate(radii):
        k = _k(f, n_r)
        X, Y, Z = (Pt[:, :, k, a] - c[a] for a in range(3))
        C = values[:, :, k].astype(float)
        if cutaway:
            X, Y, Z = (np.where(hide[None, :], np.nan, A) for A in (X, Y, Z))
        X, Y, Z, C = (np.concatenate([A, A[:, :1]], axis=1) for A in (X, Y, Z, C))
        opacity = 1.0 if (cutaway or i == 0) else max(0.12, 0.3 - 0.06 * (i - 1))
        fig.add_trace(go.Surface(
            x=X, y=Y, z=Z, surfacecolor=C, cmin=scale[1], cmax=scale[2], colorscale=scale[0],
            showscale=(i == len(radii) - 1), opacity=opacity, lighting=FLAT,
            colorbar=dict(title=scale[3], len=0.6), name=f"{int(f * 100)}% of the way to the edge",
            showlegend=True, hovertemplate="%{surfacecolor:.2f}<extra></extra>"))
    return _scene(fig, dark, 600, eye=(1.5, 1.3, 0.7))


# ── Grad-CAM summaries (paper Fig. S gradcam_rays) ───────────────────────────

def attention_summary(cam_s, geo, dark=False, transformer=False):
    """Left: attention per ray direction (mean over depth), grey polar caps.
    Right: attention along the 5% most important rays, from centre to edge."""
    t = _style(dark)
    ray, ok, m, s = P.ray_importance(cam_s, geo["phis"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 3.2), gridspec_kw={"width_ratios": [1.35, 1], "wspace": 0.3})
    im = _map(a1, np.clip(ray, 0, 1), INFERNO, 0, 1)
    for y0 in (P.POLAR_DEG, -90):
        a1.axhspan(y0, y0 + 90 - P.POLAR_DEG, color=t["poly"], alpha=0.85, lw=0)
    _angles(a1)
    a1.set_xlabel("Direction around the head"), a1.set_ylabel("Inferior – superior")
    a1.set_title("Attention by direction")
    fig.colorbar(im, ax=a1, fraction=0.04, pad=0.02).set_label("Attention")
    r = np.arange(1, len(m) + 1) / len(m)
    if transformer:
        a2.text(0.5, 0.5, "Not available for this model\n(no depth resolution)", ha="center", va="center",
                transform=a2.transAxes, color=t["muted"])
        a2.set_yticks([])
    else:
        a2.fill_between(r, m - s, m + s, color=INFERNO(0.6), alpha=0.25, lw=0)
        a2.plot(r, m, color=INFERNO(0.6), lw=1.6)
        a2.axvline(r[np.argmax(m)], color=t["muted"], lw=0.7, ls="--")
        a2.set_ylim(bottom=0)
        a2.set_ylabel("Attention")
    a2.set_xlim(0, 1)
    a2.set_xticks([0, 0.25, 0.5, 0.75, 1], ["centre", "", "half-way", "", "edge"])
    a2.set_xlabel("Position along the ray")
    a2.spines[["top", "right"]].set_visible(False)
    a2.set_title("Attention by depth (most important rays)")
    return fig


# ── comparison with the ADNI groups (Fig. 4, Fig. S cn_vs_ad) ────────────────

def group_maps(T, ref, protocol, f, dark=False):
    """At r = f R_edge. Top: normal average, dementia average and this scan
    (grey, as in Fig. S cn_vs_ad). Bottom: the typical dementia pattern
    (Cohen's d, Fig. 4) and this scan's difference from the normal average (z)."""
    t = _style(dark)
    k = _k(f, T.shape[2])
    fig = plt.figure(figsize=(11, 5.8))
    sf_top, sf_bot = fig.subfigures(2, 1, hspace=0.04)
    top_axes = sf_top.subplots(1, 3, gridspec_kw={"wspace": 0.08})
    bot_axes = sf_bot.subplots(1, 3, gridspec_kw={"wspace": 0.45})
    top = [(ref.mean("CN", protocol)[:, :, k], f"Normal average (n = {ref.n('CN', protocol)})"),
           (ref.mean("Dementia", protocol)[:, :, k], f"Dementia average (n = {ref.n('Dementia', protocol)})"),
           (T[:, :, k], "This scan")]
    for j, (M, title) in enumerate(top):
        im = _map(top_axes[j], M, "gray", *T_LIM)
        _angles(top_axes[j], left=j == 0)
        top_axes[j].set_title(title)
    sf_top.colorbar(im, ax=list(top_axes), fraction=0.015, pad=0.015).set_label("Intensity")
    z = ref.deviation(T, protocol)
    for j, (M, lim, title, label) in enumerate((
            (ref.d[:, :, k], D_LIM, "Typical dementia pattern", "Group difference (d)"),
            (z[:, :, k], Z_LIM, "This scan vs normal average", "Difference (z)"))):
        imd = _map(bot_axes[j], M, "RdBu_r", -lim, lim)
        _angles(bot_axes[j], left=j == 0)
        bot_axes[j].set_title(title)
        bot_axes[j].set_xlabel("Direction around the head")
        sf_bot.colorbar(imd, ax=bot_axes[j], fraction=0.05, pad=0.03).set_label(label)
    bot_axes[2].axis("off")
    bot_axes[2].text(0.05, 0.5, "Red: darker than normal.\nBlue: brighter than normal.",
                     transform=bot_axes[2].transAxes, va="center", color=t["muted"])
    top_axes[0].set_ylabel("Inferior – superior"), bot_axes[0].set_ylabel("Inferior – superior")
    return fig


def score_plot(ref, score, dark=False):
    """Validation-set distributions of the similarity score, with this scan."""
    t = _style(dark)
    g = ref.data["score_grid"]
    fig, ax = plt.subplots(figsize=(6.2, 2.8))
    for dx, col, name in (("CN", "#2b8cbe", "Normal"), ("Dementia", "#d7301f", "Dementia")):
        y = ref.data[f"score_density_{dx}"]
        ax.fill_between(g, y, color=col, alpha=0.18, lw=0)
        ax.plot(g, y, color=col, lw=1.4, label=f"{name} (ADNI, n = {int(ref.data[f'n_validation_{dx}'])})")
    ax.axvline(score, color=t["fg"], lw=1.6)
    ax.text(score, ax.get_ylim()[1] * 0.95, " this scan", color=t["fg"], va="top")
    ax.set_xlabel("Similarity to the dementia pattern")
    ax.set_yticks([])
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=8)
    fig.tight_layout()
    return fig
