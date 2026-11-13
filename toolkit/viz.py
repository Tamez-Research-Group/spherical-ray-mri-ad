"""Toolkit figures: rays on the subject's own slices, the unwrapped spherical
tensor, and interactive 3-D shells. Slices reuse figures_spherical.slice_panel
(the paper's Fig. 1 and 4 display conventions)."""
from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
import plotly.graph_objects as go                    # noqa: E402

import pipeline                                      # noqa: E402,F401  (puts the repo on sys.path)
from figures_spherical import slice_panel, VIEWS, _in_plane     # noqa: E402

plt.rcParams.update({"savefig.dpi": 150, "figure.dpi": 110, "font.size": 10})
ORANGE, BLUE = "#e2622b", "#3b82f6"
FLAT = dict(ambient=0.9, diffuse=0.25, specular=0.05, roughness=0.9)   # even, readable shading


def slices(geo: dict, T: np.ndarray | None = None, mode: str = "rays"):
    """Axial, coronal and sagittal slices through the centre of mass.
    mode 'rays': rays and the smoothed edge R_edge; 'tensor': the tensor's
    sample values drawn at their positions in the brain."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.0), facecolor="white")
    for ax, view in zip(axes, ("axial", "coronal", "sagittal")):
        if mode == "tensor":            # anatomy-free, with screen-sized markers
            slice_panel(ax, geo, view, mode=None, anatomy=False)
            axis, (hx, vy) = VIEWS[view][0], VIEWS[view][1]
            near, Pt, c = _in_plane(geo, axis), geo["points"], geo["com"]
            lo, hi = np.percentile(T, [1, 99])
            ax.scatter(Pt[near][:, hx] - c[hx], Pt[near][:, vy] - c[vy], c=T[near], cmap="gray",
                       vmin=lo, vmax=hi, s=4, linewidths=0)
        else:
            slice_panel(ax, geo, view, mode=mode)
        ax.set_title(VIEWS[view][3])
    if mode == "rays":
        axes[0].plot([], [], color=BLUE, lw=1, label="rays from the centre of mass")
        axes[0].plot([], [], color=ORANGE, lw=1.5, label="smoothed edge $R_{edge}$")
        fig.legend(loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.07 if mode == "rays" else 0, 1, 1))
    return fig


def unwrapped(T: np.ndarray, k: int, cam: np.ndarray | None = None):
    """The tensor at radius index k as a (phi x theta) map; optional Grad-CAM contours."""
    n_phi, n_theta, n_r = T.shape
    ext = [0, 360, -90, 90]
    fig, ax = plt.subplots(figsize=(9, 4.9 if cam is not None else 4.4), facecolor="white")
    img = T[:, :, k]
    lo, hi = np.percentile(T, [1, 99])
    im = ax.imshow(img, origin="lower", cmap="gray", vmin=lo, vmax=hi, extent=ext, aspect="auto")
    fig.colorbar(im, ax=ax, label="normalized intensity", fraction=0.04, pad=0.02)
    if cam is not None:
        # scaled to the maximum over the whole tensor, so shells are comparable;
        # weak attributions (< 0.2) are left transparent
        c = cam[:, :, k] / max(float(cam.max()), 1e-8)
        ov = ax.imshow(np.ma.masked_less(c, 0.2), origin="lower", cmap="hot", vmin=0, vmax=1,
                       alpha=0.6, extent=ext, aspect="auto", interpolation="bilinear")
        fig.colorbar(ov, ax=ax, orientation="horizontal", fraction=0.05, pad=0.2,
                     label="Grad-CAM, Dementia logit (relative to the subject's maximum)")
    ax.set_xlabel("azimuth θ (°)   0 = right, 90 = anterior, 180 = left, 270 = posterior")
    ax.set_ylabel("elevation φ (°)")
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_yticks([-90, -45, 0, 45, 90])
    ax.set_title(f"Shell r = {(k + 1) / n_r:.2f} R_edge  (radius index {k + 1} of {n_r})")
    fig.tight_layout()
    return fig


def radial_profiles(T: np.ndarray, cam: np.ndarray | None = None):
    """Mean intensity along the rays (and mean Grad-CAM), as a function of r / R_edge."""
    r = np.arange(1, T.shape[2] + 1) / T.shape[2]
    fig, ax = plt.subplots(figsize=(6, 3.4), facecolor="white")
    m, sd = T.mean((0, 1)), T.std((0, 1))
    ax.plot(r, m, color="#1a1a1a", label="mean over rays")
    ax.fill_between(r, m - sd, m + sd, color="#999999", alpha=0.3, lw=0, label="± SD")
    ax.set_xlabel("r / R_edge")
    ax.set_ylabel("normalized intensity")
    if cam is not None:
        ax2 = ax.twinx()
        ax2.plot(r, cam.mean((0, 1)), color=ORANGE, label="Grad-CAM (mean)")
        ax2.set_ylabel("Grad-CAM", color=ORANGE)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    return fig


def _slice_surface(geo, axis, step=2):
    """The harmonized volume's slice through the centre of mass as a flat
    plotly surface (background left out), in mm from the centre of mass."""
    vol, h, c = geo["vol_n"], geo["h"], geo["com"]
    idx = [slice(None, None, step)] * 3
    idx[axis] = int(round(c[axis] / h[axis]))
    img = vol[tuple(idx)].astype(float)
    bg = vol[0, 0, 0]
    others = [a for a in range(3) if a != axis]
    g = [np.arange(0, vol.shape[a], step) * h[a] - c[a] for a in others]
    A, B = np.meshgrid(g[0], g[1], indexing="ij")
    coords = [None] * 3
    coords[others[0]], coords[others[1]] = A, B
    coords[axis] = np.zeros_like(A)
    keep = img != bg
    coords = [np.where(keep, q, np.nan) for q in coords]
    lo, hi = np.percentile(img[keep], [1, 99.5])
    return go.Surface(x=coords[0], y=coords[1], z=coords[2], surfacecolor=img, cmin=lo, cmax=hi,
                      colorscale="Gray", showscale=False, hoverinfo="skip", name="slice",
                      lighting=dict(ambient=1.0, diffuse=0, specular=0))


def rays_3d(geo: dict, step: int = 3, slices_on=("axial", "sagittal"), edge: bool = True):
    """Interactive 3-D view of the sampling: rays from the centre of mass to
    the smoothed edge R_edge (every `step`-th bin in elevation and azimuth),
    the edge surface itself, and orthogonal slices of the harmonized brain."""
    P, c = geo["points"], geo["com"]
    E = P[:, :, -1] - c                                   # ray end points (r = R_edge)
    fig = go.Figure()
    for v in slices_on:
        fig.add_trace(_slice_surface(geo, VIEWS[v][0]))
    if edge:
        X, Y, Z = (np.concatenate([E[:, :, a], E[:, :1, a]], axis=1) for a in range(3))
        fig.add_trace(go.Surface(x=X, y=Y, z=Z, surfacecolor=np.concatenate(
            [geo["R_edge"], geo["R_edge"][:, :1]], axis=1), colorscale="Oranges", opacity=0.12, lighting=FLAT,
            showscale=True, colorbar=dict(title="R_edge (mm)", len=0.5, x=1.0),
            name="smoothed edge R_edge", showlegend=True,
            hovertemplate="R_edge %{surfacecolor:.0f} mm<extra></extra>"))
    sel = E[::step, ::step].reshape(-1, 3)
    xs, ys, zs = (np.column_stack([np.zeros(len(sel)), sel[:, a], np.full(len(sel), np.nan)]).ravel()
                  for a in range(3))
    fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color=BLUE, width=2),
                               name=f"rays (every {step}rd bin)" if step == 3 else f"rays (every {step}th bin)",
                               hoverinfo="skip"))
    fig.add_trace(go.Scatter3d(x=sel[:, 0], y=sel[:, 1], z=sel[:, 2], mode="markers",
                               marker=dict(size=2, color=ORANGE), name="ray ends", hoverinfo="skip"))
    fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode="markers", name="centre of mass",
                               marker=dict(size=6, color=ORANGE, line=dict(color="white", width=1))))
    fig.update_layout(
        scene=dict(xaxis_title="L → R (mm)", yaxis_title="P → A (mm)", zaxis_title="I → S (mm)",
                   aspectmode="data", camera=dict(eye=dict(x=-1.6, y=1.1, z=0.6))),
        margin=dict(l=0, r=0, t=10, b=0), height=620,
        legend=dict(orientation="h", y=1.0, x=0, bgcolor="rgba(255,255,255,0.6)"))
    return fig


def shells(T: np.ndarray, geo: dict, radii=(0.5, 0.9), cam: np.ndarray | None = None,
           cutaway: bool = True):
    """Interactive 3-D shells: the sample points at fixed fractions of R_edge,
    coloured by tensor intensity (or Grad-CAM). Coordinates are millimetres
    from the centre of mass (x: left->right, y: posterior->anterior,
    z: inferior->superior). cutaway hides the anterior-right quadrant."""
    P, c = geo["points"], geo["com"]
    n_phi, n_theta, n_r = T.shape
    thetas = geo["thetas"]
    hide = (np.cos(thetas) > 0) & (np.sin(thetas) > 0)          # right & anterior
    vals = cam if cam is not None else T
    lo, hi = (0.0, float(cam.max()) or 1.0) if cam is not None else tuple(np.percentile(T, [1, 99]))
    fig = go.Figure()
    for i, f in enumerate(sorted(radii)):
        k = int(round(f * n_r)) - 1
        k = min(max(k, 0), n_r - 1)
        X, Y, Z = (P[:, :, k, a] - c[a] for a in range(3))
        C = vals[:, :, k].astype(float).copy()
        if cutaway:
            X, Y, Z = (np.where(hide[None, :], np.nan, A) for A in (X, Y, Z))
        X, Y, Z, C = (np.concatenate([A, A[:, :1]], axis=1) for A in (X, Y, Z, C))  # close the θ seam
        fig.add_trace(go.Surface(
            x=X, y=Y, z=Z, surfacecolor=C, cmin=lo, cmax=hi,
            colorscale="YlOrRd" if cam is not None else "Gray",
            showscale=(i == 0), opacity=1.0 if i == 0 else 0.55, lighting=FLAT,
            colorbar=dict(title="Grad-CAM" if cam is not None else "intensity", len=0.6),
            name=f"r = {f:.2f} R_edge", showlegend=True,
            hovertemplate=f"r = {f:.2f} R_edge<br>value %{{surfacecolor:.2f}}<extra></extra>"))
    fig.update_layout(
        scene=dict(xaxis_title="L → R (mm)", yaxis_title="P → A (mm)", zaxis_title="I → S (mm)",
                   aspectmode="data", camera=dict(eye=dict(x=1.5, y=1.3, z=0.7))),
        margin=dict(l=0, r=0, t=30, b=0), height=560,
        legend=dict(orientation="h", y=1.02, x=0))
    return fig
