"""Paper figure of the research toolkit, on the MNI152 example (no participant data).

(a) the web app (screenshot of the Rays-and-attention section, passed as --screenshot);
(b) attention by direction and (c) by depth along the most important rays;
(d) centre-of-mass slices with rays and the model-attention (Grad-CAM) overlay;
(e) the typical dementia pattern (protocol-adjusted Cohen's d, ADNI training set),
(f) this scan's difference from the ADNI CN mean of its protocol group (z), and
(g) its similarity score against the ADNI validation distributions.
Panels are drawn with toolkit/viz.py conventions, as the app draws them.

    python toolkit/figure_toolkit.py --screenshot app.png [--out figures/toolkit_overview]
"""
from __future__ import annotations

import argparse
import io
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]

import matplotlib                       # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt         # noqa: E402
import nibabel as nib                   # noqa: E402
import numpy as np                      # noqa: E402
from PIL import Image                   # noqa: E402

import examples                         # noqa: E402
import pipeline as P                    # noqa: E402
import viz                              # noqa: E402

MM = 1 / 25.4
DPI = 600


def render(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def crop_white(im, pad=10):
    a = np.asarray(im.convert("L")) < 250
    ys, xs = np.where(a)
    return im.crop((max(xs.min() - pad, 0), max(ys.min() - pad, 0),
                    min(xs.max() + pad, im.width), min(ys.max() + pad, im.height)))


def small(fig_w, fig_h):
    viz._style(False)
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9})
    return plt.subplots(figsize=(fig_w, fig_h))


def panel_ray_map(cam, geo):
    fig, ax = small(3.6, 2.2)
    ray, ok, _, _ = P.ray_importance(cam, geo["phis"])
    im = viz._map(ax, np.clip(ray, 0, 1), viz.INFERNO, 0, 1)
    for y0 in (P.POLAR_DEG, -90):
        ax.axhspan(y0, y0 + 90 - P.POLAR_DEG, color="#9a9a9a", alpha=0.85, lw=0)
    viz._angles(ax)
    ax.set_xlabel("Direction around the head"), ax.set_ylabel("Inferior – superior")
    ax.set_title("Attention by direction")
    fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02).set_label("Attention")
    return render(fig)


def panel_depth(cam, geo):
    fig, ax = small(3.6, 1.9)
    _, _, m, s = P.ray_importance(cam, geo["phis"])
    r = np.arange(1, len(m) + 1) / len(m)
    ax.fill_between(r, m - s, m + s, color=viz.INFERNO(0.6), alpha=0.25, lw=0)
    ax.plot(r, m, color=viz.INFERNO(0.6), lw=1.4)
    ax.axvline(r[np.argmax(m)], color="#6b7280", lw=0.7, ls="--")
    ax.set_xlim(0, 1), ax.set_ylim(bottom=0)
    ax.set_xticks([0, 0.5, 1], ["centre", "half-way", "edge"])
    ax.set_ylabel("Attention"), ax.set_title("Attention by depth (top 5% of rays)")
    ax.spines[["top", "right"]].set_visible(False)
    return render(fig)


def panel_map(M, lim, title, label):
    fig, ax = small(2.6, 1.9)
    im = viz._map(ax, M, "RdBu_r", -lim, lim)
    viz._angles(ax)
    ax.set_xlabel("Direction around the head"), ax.set_ylabel("Inferior – superior")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03).set_label(label)
    return render(fig)


def panel_score(ref, score):
    viz._style(False)
    plt.rcParams.update({"font.size": 8})
    fig = viz.score_plot(ref, score, False)
    fig.set_size_inches(3.0, 1.9)
    return render(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--screenshot", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=HERE.parent / "figures" / "toolkit_overview")
    a = ap.parse_args()

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "mni152.nii.gz"
        nib.save(examples.mni152(), str(p))
        subj = P.process(p)
    ref = P.Reference.load()
    geo, T = subj.geometry, subj.tensor
    cam = P.scaled_cam(P.gradcam(T, "cnn"))
    k = viz._k(0.3, T.shape[2])
    z = ref.deviation(T, subj.protocol)

    imgs = {"a": crop_white(Image.open(a.screenshot).convert("RGB")),
            "b": panel_ray_map(cam, geo), "c": panel_depth(cam, geo),
            "d": render(viz.three_views(geo, "anatomy", False, cam=cam, rays=True)),
            "e": panel_map(ref.d[:, :, k], viz.D_LIM, "Typical dementia pattern", "Cohen's d"),
            "f": panel_map(z[:, :, k], viz.Z_LIM, "This scan vs normal mean", "z"),
            "g": panel_score(ref, ref.pattern_score(T, subj.protocol))}
    W, gap = 190.0, 3.0                                          # mm

    def hw(im, w):
        return w * im.height / im.width

    wa = 0.60 * W                                                # row 1: (a) | (b) over (c)
    wr = W - wa - gap
    h1 = max(hw(imgs["a"], wa), hw(imgs["b"], wr) + gap + hw(imgs["c"], wr))
    h2 = hw(imgs["d"], W)                                        # row 2: (d)
    w3 = (W - 2 * gap) / 3                                       # row 3: (e) (f) (g)
    h3 = max(hw(imgs[x], w3) for x in "efg")
    H = h1 + h2 + h3 + 2 * gap
    y3 = h1 + h2 + 2 * gap
    boxes = {"a": (0, 0, wa), "b": (wa + gap, 0, wr), "c": (wa + gap, hw(imgs["b"], wr) + gap, wr),
             "d": (0, h1 + gap, W), "e": (0, y3, w3), "f": (w3 + gap, y3, w3), "g": (2 * (w3 + gap), y3, w3)}
    fig = plt.figure(figsize=(W * MM, H * MM))
    for key, (x, y, w) in boxes.items():
        h = hw(imgs[key], w)
        ax = fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H])
        ax.imshow(imgs[key], interpolation="lanczos")
        ax.set_axis_off()
        ax.text(0.0, 1.0, f"({key})", transform=ax.transAxes, fontsize=10, fontweight="bold",
                va="top", ha="left", family="sans-serif",
                bbox=dict(facecolor="white", edgecolor="none", pad=0.5, alpha=0.8))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{a.out}.pdf", dpi=DPI)
    fig.savefig(f"{a.out}.png", dpi=300)
    print(f"-> {a.out}.pdf/.png: {W:.0f} x {H:.0f} mm")


if __name__ == "__main__":
    main()
