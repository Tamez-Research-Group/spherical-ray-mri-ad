"""
figures_spherical.py — the two figures that explain the spherical
representation (Methods Section 2.3, Fig. 1; Results, Fig. 4).

  rays_3d           (a) rays from the centre of mass to R_edge inside the brain
                    surface, (b) the R_edge surface, (c) one tensor shell
                    (Methods Eqs. 2-8), for one training subject.
  group_effect_map  Cohen's d (CN - Dementia) of every tensor cell on the
                    training partition, computed within each acquisition
                    protocol and averaged; (a-d) elevation x azimuth maps at
                    four fractional radii, (e-g) the same effect sizes placed
                    back at their sample positions in the axial, coronal and
                    sagittal planes through the example subject's centre,
                    (h-i) on three nested shells in 3-D (r = 0.3, 0.6, 0.9),
                    with the quadrant facing the viewer cut away.
  backprojection    (Supplementary) MRI with rays and R_edge, and the tensor
                    values at their sample positions, in the three planes.
  edge_diagnostics  (Supplementary) accuracy of R_edge: true brain edge, R_edge
                    before and after smoothing, in the three planes.
  cn_vs_ad          (Supplementary) the age-, sex- and protocol-matched test
                    pair: slices with R_edge, tensor maps and their difference.

Slices use the standard anatomical display: axial and coronal seen from the
front with superior / anterior up, sagittal with anterior on the left.
All three slice panels share one field of view (mm around the centre), so
the panels and their titles line up. Red: lower intensity in Dementia.

Figures follow the Elsevier artwork rules of notebooks/results.ipynb
(190 mm width, Arial 9 pt; vector PDF, 600-dpi LZW TIFF and PNG) and are
written to figures/ (ADNI_FIGURES). The subjects shown are read from the
gitignored cohort/figure_subjects.json; no identifier or clinical value is
drawn on any figure. The example subject is derived from ADNI imaging:
check the Data Use Agreement before using it outside the paper.

    python figures_spherical.py [--subject PTID]
"""

from __future__ import annotations

import os
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from spherical_transform import load_nifti_as_ras, spherical_tensor_from_volume
from preprocessing.harmonize import n4_correct, resample_common

REPO = Path(__file__).resolve().parent
FIG_DIR = Path(os.environ.get("ADNI_FIGURES", REPO / "figures"))
MM = 1 / 25.4
COL2 = 190 * MM
INK, INK_2 = "#1a1a1a", "#555555"
mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5, "axes.linewidth": 0.6, "axes.edgecolor": INK_2, "text.color": INK,
    "xtick.color": INK_2, "ytick.color": INK_2, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "savefig.dpi": 600, "savefig.pad_inches": 0.02, "pdf.fonttype": 42, "ps.fonttype": 42,
})
D_LIM, D_HIDE = 1.0, 0.15          # colour range and hidden |d| (Fig. 4 caption)
RADII = (0.3, 0.5, 0.7, 0.9)       # fractional radii of panels (a-d)
FOV = 95.0                         # half-width (mm) of every slice panel
# The subjects shown in the figures are read from a local, gitignored file:
# under the ADNI Data Use Agreement, subject identifiers are never published.
# Keys: "example" (training CN subject whose native pose is closest to MNI152;
# Figs. 1, 2 and 4), "cn" and "ad" (age-, sex- and protocol-matched test pair;
# Fig. S-cnvsad).
FIGURE_SUBJECTS = REPO / "cohort" / "figure_subjects.json"


def figure_subjects():
    if not FIGURE_SUBJECTS.exists():
        raise SystemExit(f"{FIGURE_SUBJECTS} not found: create it locally with the keys "
                         '"example", "cn" and "ad" (ADNI PTIDs; never commit it)')
    return json.loads(FIGURE_SUBJECTS.read_text())


def save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    kw = {"bbox_inches": "tight"}
    fig.savefig(FIG_DIR / f"{name}.pdf", **kw)
    fig.savefig(FIG_DIR / f"{name}.png", dpi=600, **kw)
    fig.savefig(FIG_DIR / f"{name}.tif", dpi=600, pil_kwargs={"compression": "tiff_lzw"}, **kw)
    print("wrote", FIG_DIR / name, "{.pdf,.png,.tif}")


def panel(fig, ax, letter, title):
    """Panel letter and title at a fixed offset above the axes, so that
    panels of equal size have aligned labels."""
    b = ax.get_position()
    label(fig, b.x0, b.y1 + 0.012, letter, title)


def label(fig, x, y, letter, title):
    """Bold '(letter)' followed by the title, at figure coordinates (x, y);
    the title starts after the measured width of the letter."""
    t = fig.text(x, y, f"({letter})", fontweight="bold", ha="left", va="bottom")
    w = t.get_window_extent(renderer=fig.canvas.get_renderer()).width / fig.bbox.width
    fig.text(x + w + 0.008, y, title, ha="left", va="bottom")


# ── data ─────────────────────────────────────────────────────────────────────

def cohen_d_protocol_adjusted(man: pd.DataFrame) -> np.ndarray:
    """Cohen's d (CN - Dementia) per tensor cell, training partition only,
    within each protocol (pooled SD), averaged over the protocols."""
    tr = man[man["split"] == "train"]
    ds = []
    for proto, g in tr.groupby("protocol"):
        X = np.stack([np.load(REPO / "spherical_cache" / f"{p}.npy") for p in g["PTID"]]).astype(np.float64)
        y = (g["Diagnosis"] == "Dementia").to_numpy()
        a, b = X[~y], X[y]
        sp = np.sqrt(((len(a) - 1) * a.var(0, ddof=1) + (len(b) - 1) * b.var(0, ddof=1)) / (len(a) + len(b) - 2))
        ds.append((a.mean(0) - b.mean(0)) / np.maximum(sp, 1e-6))
        print(f"  {proto}: {(~y).sum()} CN, {y.sum()} Dementia")
    return np.mean(ds, axis=0)


def example_geometry(path: Path):
    vol, nii = load_nifti_as_ras(path)
    sp = tuple(float(z) for z in nii.header.get_zooms()[:3])
    vol, sp = resample_common(n4_correct(vol, sp), sp)
    T, geo = spherical_tensor_from_volume(vol, sp, return_geometry=True)
    return T, geo


# ── Fig. 4: group effect map ─────────────────────────────────────────────────

VIEWS = {   # plane axis, (horizontal axis, vertical axis), flip horizontal, title, axis labels
    "axial":    (2, (0, 1), False, "Axial", ("L", "R"), ("P", "A")),
    "coronal":  (1, (0, 2), False, "Coronal", ("L", "R"), ("I", "S")),
    "sagittal": (0, (1, 2), True,  "Sagittal", ("A", "P"), ("I", "S")),
}


def _in_plane(geo, axis, one_side=False):
    """Cells whose angular bin is adjacent to the plane through the centre
    (within one 5-degree bin of it, in the angle that crosses the plane:
    elevation for the axial plane, azimuth for the coronal and sagittal
    planes); their samples are projected onto the plane. No bin centre lies
    exactly in a cardinal plane, so exact-plane samples alone would be sparse.
    one_side keeps a single bin per direction, which gives a closed ring."""
    phis, thetas = geo["phis"], geo["thetas"]
    lim = np.sin(np.radians(5.0))
    comp = {2: np.sin(phis)[:, None] + 0 * thetas[None, :],          # axial: elevation
            1: np.sin(thetas)[None, :] + 0 * phis[:, None],          # coronal: azimuth near 0 / 180
            0: np.cos(thetas)[None, :] + 0 * phis[:, None]}[axis]    # sagittal: azimuth near 90 / 270
    sel = (comp >= 0) & (comp < lim) if one_side else np.abs(comp) < lim
    return np.broadcast_to(sel[:, :, None], geo["points"].shape[:3])


def slice_panel(ax, geo, view, mode="effect", d=None, T=None, anatomy=True):
    """One slice through the centre of mass in standard display. mode:
    'effect' (Cohen's d at the sample positions), 'rays' (rays and R_edge),
    'ring' (R_edge only), 'tensor' (tensor values at the sample positions)
    or None (anatomy only)."""
    axis, (hx, vy), flip, _, (l0, l1), (b0, b1) = VIEWS[view]
    vol, h, c = geo["vol_n"], geo["h"], geo["com"]
    if anatomy:
        idx = [slice(None)] * 3
        idx[axis] = int(round(c[axis] / h[axis]))
        img = vol[tuple(idx)]                              # axes (hx, vy) in array order
        ext_x = (np.arange(img.shape[0]) * h[hx] - c[hx])
        ext_y = (np.arange(img.shape[1]) * h[vy] - c[vy])
        lo, hi = np.percentile(img[img != vol[0, 0, 0]], [1, 99.5])
        ax.imshow(img.T, origin="lower", cmap="gray", vmin=lo, vmax=hi, interpolation="bilinear",
                  extent=[ext_x[0], ext_x[-1], ext_y[0], ext_y[-1]])
    P, near = geo["points"], _in_plane(geo, axis)
    if mode == "effect":
        show = near & (np.abs(d) >= D_HIDE)
        pts, val = P[show], d[show]
        order = np.argsort(np.abs(val))                    # strongest effects drawn last
        ax.scatter(pts[order, hx] - c[hx], pts[order, vy] - c[vy], c=val[order], cmap="RdBu_r",
                   vmin=-D_LIM, vmax=D_LIM, s=1.2, linewidths=0, rasterized=True)
    elif mode == "tensor":
        lo, hi = np.percentile(T, [1, 99])
        ax.scatter(P[near][:, hx] - c[hx], P[near][:, vy] - c[vy], c=T[near], cmap="gray",
                   vmin=lo, vmax=hi, s=1.2, linewidths=0, rasterized=True)
    elif mode in ("rays", "ring"):
        E = P[:, :, -1]                                    # r = 1: the boundary R_edge
        ring_cells = _in_plane(geo, axis, one_side=True)[:, :, -1]
        cells = np.argwhere(ring_cells)
        ang = np.arctan2(E[ring_cells][:, vy] - c[vy], E[ring_cells][:, hx] - c[hx])
        o = np.argsort(ang)
        ring = E[cells[o, 0], cells[o, 1]]
        ax.plot(np.r_[ring[:, hx], ring[0, hx]] - c[hx], np.r_[ring[:, vy], ring[0, vy]] - c[vy],
                color="#e2622b", lw=1.0)
        for e in (ring[::3] if mode == "rays" else []):    # every third in-plane ray
            ax.plot([0, e[hx] - c[hx]], [0, e[vy] - c[vy]], color="#6fa8e8", lw=0.4)
        ax.scatter([0], [0], color="#e2622b", s=14, edgecolor="white", linewidth=0.6, zorder=5)
    # centre the view on the brain's bounding box (the centre of mass is not
    # the middle of the brain, e.g. along anterior-posterior)
    nz = np.argwhere(vol != vol[0, 0, 0])
    mid = (nz.min(0) + nz.max(0)) / 2 * h - c
    mx, my = mid[hx], mid[vy]
    ax.set_xlim((mx + FOV, mx - FOV) if flip else (mx - FOV, mx + FOV))
    ax.set_ylim(my - FOV, my + FOV)
    ax.set_aspect("equal")
    ax.set_facecolor("black")
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel(f"{l0} ← → {l1}", labelpad=2)
    ax.set_ylabel(f"{b0} ← → {b1}", labelpad=2)


SHELL_RADII = (0.3, 0.6, 0.9)      # nested shells of panels (h-i)
SHELL_VIEWS = ((45, "right-anterior"), (225, "left-posterior"))   # (azimuth of the viewer, deg)


def effect_shells_3d(ax, d, geo, view_azim, radii=SHELL_RADII, elev=22):
    """Cohen's d on nested tensor shells in 3-D, at their sample positions in
    the example subject. The quadrant of directions facing the viewer is cut
    away so the inner shells are visible. All quads go into one collection,
    so they are depth-sorted together (matplotlib sorts per collection)."""
    P, c = geo["points"], geo["com"]
    thetas = geo["thetas"]
    a = np.radians(view_azim)
    facing = np.cos(thetas - a) > np.cos(np.radians(45))            # azimuths within 45 deg of the viewer
    n_r = d.shape[2]
    cmap, norm = mpl.cm.RdBu_r, mpl.colors.Normalize(-D_LIM, D_LIM)
    quads, cols = [], []
    for f in radii:
        k = int(round(f * n_r)) - 1
        S = np.concatenate([P[:, :, k], P[:, :, k][:, :1]], axis=1)  # theta is periodic: close the shell
        C = d[:, :, k]
        for i in range(S.shape[0] - 1):
            for j in range(S.shape[1] - 1):
                if facing[j % len(thetas)]:
                    continue
                quads.append([S[i, j], S[i + 1, j], S[i + 1, j + 1], S[i, j + 1]])
                cols.append(C[i, j % len(thetas)])
    pc = Poly3DCollection(np.array(quads), facecolors=cmap(norm(np.array(cols))),
                          edgecolor="none", linewidth=0, antialiased=False)
    ax.add_collection3d(pc)
    q = np.array(quads).reshape(-1, 3) - c
    half = np.abs(q).max()
    ax.set_xlim(c[0] - half, c[0] + half), ax.set_ylim(c[1] - half, c[1] + half), ax.set_zlim(c[2] - half, c[2] + half)
    ax.set_box_aspect((1, 1, 1), zoom=1.75)
    ax.set_axis_off()
    ax.view_init(elev=elev, azim=view_azim)
    # direction letters just outside the shells, on the side facing the viewer
    far = 0.98 * half
    side = (("R", (1, 0, 0)), ("L", (-1, 0, 0)), ("A", (0, 1, 0)), ("P", (0, -1, 0)))
    for lab, v in side:
        if np.cos(np.arctan2(v[1], v[0]) - a) > 0:                  # visible side only
            ax.text(*(c + far * np.array(v)), lab, color=INK_2, ha="center", va="center", fontweight="bold")


def group_effect_map(d, T, geo, out="group_effect_map"):
    n_phi, n_theta, n_r = T.shape
    H0, H3 = 128 * MM, 66 * MM                           # panels (a-g), then the 3-D row (h-i)
    H = H0 + H3
    yy = lambda y: (y * H0 + H3) / H                       # (a-g) keep their layout above the new row
    hh = lambda h: h * H0 / H
    fig = plt.figure(figsize=(COL2, H))
    w, gap, x0 = 0.19, 0.03, 0.06
    top = [fig.add_axes([x0 + i * (w + gap), yy(0.62), w, hh(0.33)]) for i in range(4)]
    for i, (ax, r) in enumerate(zip(top, RADII)):
        k = int(round(r * n_r)) - 1
        im = ax.imshow(d[:, :, k], origin="lower", cmap="RdBu_r", vmin=-D_LIM, vmax=D_LIM,
                       extent=[0, 360, -90, 90], aspect="auto", interpolation="nearest")
        ax.set_xticks([0, 90, 180, 270], ["R", "A", "L", "P"])
        ax.set_yticks([-90, 0, 90], ["I", "0", "S"] if i == 0 else [])
        ax.set_xlabel("Azimuth θ")
        if i == 0:
            ax.set_ylabel("Elevation φ")
        panel(fig, ax, "abcd"[i], f"r = {r} R$_{{edge}}$")
    cax = fig.add_axes([x0 + 4 * (w + gap) - gap + 0.012, yy(0.62), 0.012, hh(0.33)])
    fig.colorbar(im, cax=cax).set_label("Cohen's d (CN − Dementia)\nred: lower in Dementia")

    s = 0.265                                              # square slice panels
    bottom = [fig.add_axes([x0 + i * (s + 0.035), yy(0.04), s, s * COL2 / H]) for i in range(3)]
    for ax, view, letter in zip(bottom, VIEWS, "efg"):
        slice_panel(ax, geo, view, "effect", d=d)
        panel(fig, ax, letter, f"{VIEWS[view][3]}, effect size on anatomy")
    b = bottom[-1].get_position()
    cax2 = fig.add_axes([b.x1 + 0.015, b.y0, 0.012, b.height])
    sm = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(-D_LIM, D_LIM), cmap="RdBu_r")
    cb = fig.colorbar(sm, cax=cax2)
    cb.set_label("Cohen's d (CN − Dementia)")
    cb.set_ticks([-1, -0.5, 0, 0.5, 1])                    # same ticks as the colour bar of (a-d)

    w3, h3 = 0.42, (H3 - 6 * MM) / H                       # (h-i): nested shells in 3-D
    shells = [fig.add_axes([x0 + i * (w3 + 0.06), 0.015, w3, h3], projection="3d") for i in range(2)]
    rr = ", ".join(str(r) for r in SHELL_RADII)
    for i, (ax, (azim, name)) in enumerate(zip(shells, SHELL_VIEWS)):
        effect_shells_3d(ax, d, geo, azim)
        label(fig, x0 + i * (w3 + 0.06), h3 + 0.004, "hi"[i], f"Shells at r = {rr} R$_{{edge}}$, {name} view")
    save(fig, out)
    plt.close(fig)


# ── Fig. 1: rays in 3-D ──────────────────────────────────────────────────────

def _style3d(ax, c, half=FOV):
    ax.set_xlim(c[0] - half, c[0] + half), ax.set_ylim(c[1] - half, c[1] + half), ax.set_zlim(c[2] - half, c[2] + half)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=12, azim=-60)
    ax.set_axis_off()
    # orientation tripod in the lower-left corner (S, A, R)
    o = np.array([c[0] - half, c[1] - half, c[2] - half])
    L = 0.35 * half
    for vec, lab in (((0, 0, L), "S"), ((0, L, 0), "A"), ((L, 0, 0), "R")):
        ax.plot(*[[o[i], o[i] + vec[i]] for i in range(3)], color=INK_2, lw=0.8)
        ax.text(*(o + 1.25 * np.array(vec)), lab, color=INK_2, fontsize=9, ha="center", va="center")


def rays_3d(T, geo, out="rays_3d"):
    from skimage.measure import marching_cubes
    h, c, R = geo["h"], geo["com"], geo["R_edge"]
    mask = geo["vol_n"] != geo["vol_n"][0, 0, 0]          # crop margin = background value
    verts, faces, _, _ = marching_cubes(mask[::2, ::2, ::2].astype(np.float32), 0.5)
    verts = verts * 2 * h
    phis, thetas = geo["phis"], geo["thetas"]
    PH, TH = np.meshgrid(phis, thetas, indexing="ij")
    D = np.stack([np.cos(PH) * np.cos(TH), np.cos(PH) * np.sin(TH), np.sin(PH)], -1)
    wrap = lambda a: np.concatenate([a, a[:, :1]], axis=1)   # theta is periodic: close the surface

    fig = plt.figure(figsize=(COL2, 62 * MM))
    w = 0.28
    axs = [fig.add_axes([0.0 + i * 0.335, -0.02, w, 0.92], projection="3d") for i in range(3)]

    ax = axs[0]
    ax.add_collection3d(Poly3DCollection(verts[faces], facecolor="#bbbbbb", edgecolor="none", alpha=0.08))
    sel = (slice(2, None, 5), slice(0, None, 6))           # a subset of the 36 x 72 directions
    E = c + R[..., None] * D
    for e in E[sel].reshape(-1, 3):
        ax.plot([c[0], e[0]], [c[1], e[1]], [c[2], e[2]], color="#2a78d6", lw=0.5)
    ax.scatter(*E[sel].reshape(-1, 3).T, color="#e2622b", s=3, depthshade=False)
    ax.scatter(*c, color="#e2622b", s=14, edgecolor="white", linewidth=0.6, depthshade=False)
    _style3d(ax, c, 75)

    norm_r = mpl.colors.Normalize(R.min(), R.max())
    S1 = wrap(c + R[..., None] * D)
    ax = axs[1]
    ax.plot_surface(S1[..., 0], S1[..., 1], S1[..., 2], facecolors=mpl.cm.viridis(norm_r(wrap(R))),
                    rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)
    _style3d(ax, c, 75)

    k = int(round(0.6 * T.shape[2])) - 1
    S2 = wrap(c + 0.6 * R[..., None] * D)
    shell = T[:, :, k]
    norm_t = mpl.colors.Normalize(np.percentile(shell, 1), np.percentile(shell, 99))
    ax = axs[2]
    ax.plot_surface(S2[..., 0], S2[..., 1], S2[..., 2], facecolors=mpl.cm.gray(norm_t(wrap(shell))),
                    rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)
    _style3d(ax, c, 48)                                    # shell at 0.6 R_edge: zoomed

    titles = ("Rays from the centre of mass", "R$_{edge}$ surface (per-direction boundary)",
              "Tensor shell at r = 0.6 R$_{edge}$")
    for i, (ax, t) in enumerate(zip(axs, titles)):     # one baseline for all three titles
        b = ax.get_position()
        label(fig, b.x0 + 0.01, 0.93, "abc"[i], t)
    for i, (sm_norm, cmap, lab) in enumerate(((norm_r, "viridis", "R$_{edge}$ (mm)"),
                                              (norm_t, "gray", "Normalized intensity"))):
        b = axs[i + 1].get_position()
        cax = fig.add_axes([b.x1 - 0.005, 0.18, 0.010, 0.56])
        fig.colorbar(mpl.cm.ScalarMappable(norm=sm_norm, cmap=cmap), cax=cax).set_label(lab)
    save(fig, out)
    plt.close(fig)


# ── Supplementary: back-projection and the matched CN / Dementia pair ────────

def _slice_axes(fig, x0, y0, s, H, n=3, dx=0.035):
    return [fig.add_axes([x0 + i * (s + dx), y0, s, s * COL2 / H]) for i in range(n)]


def backprojection(T, geo, out="backprojection"):
    """Top (a, c, e): MRI with rays and R_edge; bottom (b, d, f): tensor
    values at their sample positions, in the three planes through the centre."""
    H = 135 * MM
    fig = plt.figure(figsize=(COL2, H))
    s = 0.28
    top = _slice_axes(fig, 0.06, 0.54, s, H)
    bot = _slice_axes(fig, 0.06, 0.05, s, H)
    for i, view in enumerate(VIEWS):
        name = VIEWS[view][3]
        slice_panel(top[i], geo, view, "rays")
        panel(fig, top[i], "ace"[i], f"{name}, rays and R$_{{edge}}$")
        slice_panel(bot[i], geo, view, "tensor", T=T, anatomy=False)
        panel(fig, bot[i], "bdf"[i], f"{name}, tensor values")
    save(fig, out)
    plt.close(fig)


def _true_edge(geo, D):
    """Distance (units of h) from the centre of mass to the outermost brain
    voxel along each unit direction in D (n, 3), marched in 0.25 mm steps."""
    vol, h, c = geo["vol_n"], geo["h"], geo["com"]
    mask = vol != vol[0, 0, 0]
    t = np.arange(0.0, 150.0, 0.25)
    P = (c[None, None, :] + t[None, :, None] * D[:, None, :]) / h
    idx = np.rint(P).astype(int)
    ok = np.all((idx >= 0) & (idx < np.array(mask.shape)), axis=-1)
    inside = np.zeros(ok.shape, bool)
    inside[ok] = mask[idx[ok][:, 0], idx[ok][:, 1], idx[ok][:, 2]]
    last = np.where(inside.any(1), inside.shape[1] - 1 - np.argmax(inside[:, ::-1], axis=1), 0)
    return t[last]


def edge_stats(geo):
    """Numbers quoted in the Fig. S-edge caption and Discussion: share of
    empty angular bins near the poles (|phi| > 75 deg) and the offset of the
    smoothed R_edge from the true brain edge elsewhere, per in-plane ray."""
    phis, thetas = geo["phis"], geo["thetas"]
    polar = np.abs(np.degrees(phis)) > 75
    empty = geo["R_edge_raw"] == 0
    out = {"empty_polar": float(empty[polar].mean()), "empty_nonpolar": float(empty[~polar].mean())}
    PH, TH = np.meshgrid(phis, thetas, indexing="ij")
    D = np.stack([np.cos(PH) * np.cos(TH), np.cos(PH) * np.sin(TH), np.sin(PH)], -1)
    off = geo["R_edge"] - _true_edge(geo, D.reshape(-1, 3)).reshape(PH.shape)
    out["median_offset_nonpolar_mm"] = float(np.median(off[~polar]))
    out["iqr_offset_nonpolar_mm"] = [float(q) for q in np.percentile(off[~polar], [25, 75])]
    return out


def edge_diagnostics(geo, out="edge_diagnostics"):
    """(Supplementary, Fig. S-edge) Accuracy of R_edge in the three planes
    through the centre: true brain edge in the slice (green), R_edge before
    smoothing (blue, Eq. 5) and the smoothed R_edge used (orange, Eq. 6)."""
    H = 84 * MM
    fig = plt.figure(figsize=(COL2, H))
    s = 0.28
    axs = _slice_axes(fig, 0.06, 0.20, s, H)
    P, c = geo["points"], geo["com"]
    E = P[:, :, -1] - c                                   # smoothed boundary vectors
    R = geo["R_edge"]
    Eraw = E / R[..., None] * geo["R_edge_raw"][..., None]
    for ax, view, letter in zip(axs, VIEWS, "abc"):
        axis, (hx, vy) = VIEWS[view][0], VIEWS[view][1]
        slice_panel(ax, geo, view, None)
        vol, h = geo["vol_n"], geo["h"]
        idx = [slice(None)] * 3
        idx[axis] = int(round(c[axis] / h[axis]))
        m = (vol[tuple(idx)] != vol[0, 0, 0]).astype(float)
        xs = np.arange(m.shape[0]) * h[hx] - c[hx]
        ys = np.arange(m.shape[1]) * h[vy] - c[vy]
        ax.contour(xs, ys, m.T, levels=[0.5], colors="#3fbf5f", linewidths=0.9)
        ring_cells = _in_plane(geo, axis, one_side=True)[:, :, -1]
        cells = np.argwhere(ring_cells)
        o = np.argsort(np.arctan2(E[ring_cells][:, vy], E[ring_cells][:, hx]))
        for V, col in ((Eraw, "#6fa8e8"), (E, "#e2622b")):
            ring = V[cells[o, 0], cells[o, 1]]
            ax.plot(np.r_[ring[:, hx], ring[0, hx]], np.r_[ring[:, vy], ring[0, vy]], color=col, lw=0.9)
        ax.scatter([0], [0], color="#e2622b", s=14, edgecolor="white", linewidth=0.6, zorder=5)
        panel(fig, ax, letter, VIEWS[view][3])
    handles = [mpl.lines.Line2D([], [], color=col, lw=1.2) for col in ("#3fbf5f", "#6fa8e8", "#e2622b")]
    fig.legend(handles, ["True brain edge in the slice", "R$_{edge}$ before smoothing",
                         "R$_{edge}$ used (smoothed)"], loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    save(fig, out)
    plt.close(fig)


def cn_vs_ad(man, cn, ad, out="cn_vs_ad"):
    """Matched CN/Dementia pair. No identifiers or clinical values are drawn
    (ADNI Data Use Agreement: no participant-level data in published figures)."""
    H = 140 * MM
    fig = plt.figure(figsize=(COL2, H))
    s, x0, dx = 0.14, 0.06, 0.035
    hgt = s * COL2 / H                                     # square slice panels: same height as the maps
    RT = (0.3, 0.6)
    tens, rows = {}, ((cn, "CN", 0.72), (ad, "Dementia", 0.45))
    for row, (ptid, lab, y0) in enumerate(rows):
        _, geo = example_geometry(Path(man.set_index("PTID").loc[ptid, "brain_path"]))
        tens[lab] = np.load(REPO / "spherical_cache" / f"{ptid}.npy")
        axs = _slice_axes(fig, x0, y0, s, H, dx=dx)
        for ax, view in zip(axs, VIEWS):
            slice_panel(ax, geo, view, "ring")
            if row == 0:
                b = ax.get_position()
                fig.text(b.x0 + s / 2, b.y1 + 0.012, VIEWS[view][3], ha="center", va="bottom")
        fig.text(0.008, y0 + hgt / 2, lab, rotation=90, ha="center", va="center", fontweight="bold")
        for j, r in enumerate(RT):
            ax = fig.add_axes([0.63 + j * 0.155, y0, 0.135, hgt])
            k = int(round(r * tens[lab].shape[2])) - 1
            im = ax.imshow(tens[lab][:, :, k], origin="lower", cmap="gray", vmin=-3, vmax=1.2,
                           extent=[0, 360, -90, 90], aspect="auto", interpolation="nearest")
            ax.set_xticks([0, 90, 180, 270], ["R", "A", "L", "P"])
            ax.set_yticks([-90, 0, 90], ["I", "0", "S"] if j == 0 else [])
            if row == 0:
                ax.set_title(f"r = {r} R$_{{edge}}$", pad=4)
    cax = fig.add_axes([0.94, rows[1][2], 0.010, rows[0][2] + hgt - rows[1][2]])
    fig.colorbar(im, cax=cax).set_label("Normalized intensity")
    y0 = 0.07
    for j, r in enumerate(RT):
        ax = fig.add_axes([0.63 + j * 0.155, y0, 0.135, hgt])
        k = int(round(r * tens["CN"].shape[2])) - 1
        imd = ax.imshow(tens["CN"][:, :, k] - tens["Dementia"][:, :, k], origin="lower", cmap="RdBu_r", vmin=-2,
                        vmax=2, extent=[0, 360, -90, 90], aspect="auto", interpolation="nearest")
        ax.set_xticks([0, 90, 180, 270], ["R", "A", "L", "P"])
        ax.set_yticks([-90, 0, 90], ["I", "0", "S"] if j == 0 else [])
        ax.set_xlabel("Azimuth θ")
        ax.set_title(f"r = {r} R$_{{edge}}$", pad=4)
    fig.text(0.63 + 0.145, y0 + hgt + 0.045, "Difference, CN − Dementia", ha="center", va="bottom",
             fontweight="bold")
    cax = fig.add_axes([0.94, y0, 0.010, hgt])
    fig.colorbar(imd, cax=cax).set_label("CN − Dementia\nred: darker in Dementia")
    # tensor-map block label above the two columns of maps
    fig.text(0.63 + 0.145, rows[0][2] + hgt + 0.045, "Spherical tensor", ha="center", va="bottom", fontweight="bold")
    text = ("Matched test pair (same protocol, sex and age)\n"
            "Orange: R$_{edge}$ (smoothed boundary); dot: centre of mass.")
    fig.text(x0, 0.02, text, ha="left", va="bottom", linespacing=1.45)
    save(fig, out)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default=None,
                    help='Example training subject (default: "example" in cohort/figure_subjects.json)')
    ap.add_argument("--only", choices=["rays", "groupmap", "backprojection", "pair", "edge"], default=None)
    args = ap.parse_args()
    man = pd.read_csv(REPO / "cohort" / "manifest.csv")
    subj = figure_subjects()
    if args.subject is None:
        args.subject = subj["example"]
    row = man.set_index("PTID").loc[args.subject]
    print("example subject:", args.subject, row.protocol, row.Diagnosis, row.split)
    T, geo = example_geometry(Path(row.brain_path))
    if args.only in (None, "rays"):
        rays_3d(T, geo)
    if args.only in (None, "backprojection"):
        backprojection(T, geo)
    if args.only in (None, "groupmap"):
        print("Cohen's d on the training partition:")
        group_effect_map(cohen_d_protocol_adjusted(man), T, geo)
    if args.only in (None, "edge"):
        print("R_edge diagnostics:", json.dumps(edge_stats(geo)))
        edge_diagnostics(geo)
    if args.only in (None, "pair"):
        cn_vs_ad(man, subj["cn"], subj["ad"])


if __name__ == "__main__":
    main()
