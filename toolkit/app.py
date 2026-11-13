"""Spherical-ray MRI research toolkit: local web app.

    python toolkit/app.py            # then open http://127.0.0.1:7860

Upload a skull-stripped T1-weighted NIfTI, pick a model, and see the
harmonized spherical representation (rays, shells, unwrapped tensor), the
model output of each training seed and, optionally, Grad-CAM. Everything runs
on this machine; uploaded scans are not sent anywhere. Weights are downloaded
once from Hugging Face (or read from SPHERICAL_WEIGHTS_DIR).

Research use only. Not a medical device; the output is not a diagnosis.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gradio as gr                     # noqa: E402
import numpy as np                      # noqa: E402

import pipeline as P                    # noqa: E402
import viz                              # noqa: E402

DISCLAIMER = (
    "**Research use only.** Not a medical device; the output is **not a diagnosis**. "
    "The models were trained to separate cognitively normal (CN) from Dementia participants of ADNI "
    "(baseline 1.0–1.2 mm MPRAGE, HD-BET skull stripping). They have never seen MCI or other "
    "conditions, and their behaviour on other scanners, protocols or populations is unknown."
)
MODEL_CHOICES = [("CNN3D", "cnn"), ("Spherical Transformer", "transformer")]
WEIGHT_CHOICES = [("Best epoch (primary analysis)", "best"), ("Top-5 checkpoint average", "avg")]
SEED_CHOICES = [("All 5 seeds (majority vote)", "all")] + [(f"Seed {s}", str(s)) for s in P.SEEDS]
SHELL_CHOICES = [0.3, 0.5, 0.7, 0.9, 1.0]


def _seeds(choice):
    return P.SEEDS if choice == "all" else (int(choice),)


def _result_md(subj, model, weights, df):
    s = P.summarize(df)
    label = dict((v, k) for k, v in MODEL_CHOICES)[model]
    wlabel = P.WEIGHTS[weights][0]
    vote = (f"{s['agree']} of {s['n']} seeds" if s["n"] > 1 else "single seed")
    lines = [
        f"## Model output: **{s['call']}**",
        f"{label}, {wlabel} weights · {vote} · mean P(Dementia) = {s['mean_p']:.3f}",
        "",
        "Each seed's probability is compared with that seed's own threshold, chosen on the "
        "validation set (Youden's J), as in the paper.",
        "",
        f"**Input** `{subj.name}` · voxel size {' × '.join(f'{v:.2f}' for v in subj.spacing)} mm · "
        f"protocol group `{subj.protocol}` · processed in {subj.seconds:.1f} s on "
        f"{'GPU' if P.DEVICE.type == 'cuda' else 'CPU'}",
    ]
    if subj.protocol == "slice1.2":
        lines.append("\n> In ADNI the 1.2 mm protocol coincides with the ADNI-1 phase and a higher "
                     "Dementia prevalence; the paper controls for this, but treat single outputs with care.")
    for w in subj.qc:
        lines.append(f"\n> ⚠️ {w}")
    lines += ["", DISCLAIMER]
    return "\n".join(lines)


def run(file, model, weights, seed_choice, want_cam, progress=gr.Progress()):
    if file is None:
        raise gr.Error("Upload a skull-stripped NIfTI (.nii or .nii.gz) first.")
    progress(0.05, desc="N4 bias correction, 1.2 mm resampling, spherical transform")
    try:
        subj = P.process(file)
    except Exception as e:
        raise gr.Error(f"Could not process the volume: {e}")
    seeds = _seeds(seed_choice)
    progress(0.6, desc="Loading weights and predicting")
    df = P.predict(subj.tensor, model, weights, seeds)
    cam = None
    if want_cam:
        progress(0.8, desc="Grad-CAM")
        cam = P.gradcam(subj.tensor, model, weights, seeds)
    progress(0.9, desc="Drawing")
    out = Path(tempfile.mkdtemp()) / f"{subj.name}_spherical_36x72x48.npy"
    np.save(out, subj.tensor)
    table = df.assign(**{"P(Dementia)": df["P(Dementia)"].round(4), "threshold": df["threshold"].round(4)})
    k = len(subj.tensor[0, 0]) // 2
    note = ("" if cam is None or model == "cnn" else
            "Transformer Grad-CAM has no radial resolution and was unstable across seeds in the paper "
            "(r ≈ 0.12); do not interpret it.")
    state = {"T": subj.tensor, "geo": subj.geometry, "cam": cam}
    return (
        _result_md(subj, model, weights, df), table,
        viz.slices(subj.geometry, mode="rays"), viz.slices(subj.geometry, subj.tensor, mode="tensor"),
        viz.unwrapped(subj.tensor, k, cam), viz.radial_profiles(subj.tensor, cam),
        viz.shells(subj.tensor, subj.geometry, (0.5, 0.9), None, True),
        gr.update(interactive=cam is not None, value="Intensity"), note, str(out), state,
    )


def redraw_map(state, k, show_cam):
    if not state:
        return None
    cam = state["cam"] if show_cam else None
    return viz.unwrapped(state["T"], int(k) - 1, cam)


def redraw_shells(state, radii, color, cutaway):
    if not state:
        return None
    cam = state["cam"] if color == "Grad-CAM" and state["cam"] is not None else None
    return viz.shells(state["T"], state["geo"], [float(r) for r in radii] or [0.9], cam, cutaway)


def build():
    with gr.Blocks(title="Spherical-ray MRI toolkit") as demo:
        gr.Markdown("# Spherical-ray MRI toolkit\n"
                    "Registration-free spherical representation of structural MRI and the CN-vs-Dementia "
                    "models of *Is template registration necessary?* "
                    "Scans are processed locally and never leave this computer.")
        state = gr.State(None)
        with gr.Row():
            with gr.Column(scale=1, min_width=300):
                file = gr.File(label="Skull-stripped T1 NIfTI (.nii / .nii.gz)",
                               file_types=[".nii", ".gz"], type="filepath")
                model = gr.Dropdown(MODEL_CHOICES, value="cnn", label="Model")
                weights = gr.Radio(WEIGHT_CHOICES, value="best", label="Weights")
                seed = gr.Dropdown(SEED_CHOICES, value="all", label="Training seeds")
                want_cam = gr.Checkbox(value=True, label="Compute Grad-CAM")
                go_btn = gr.Button("Run", variant="primary")
                gr.Markdown(DISCLAIMER)
            with gr.Column(scale=3):
                with gr.Tabs():
                    with gr.Tab("Result"):
                        result = gr.Markdown()
                        table = gr.Dataframe(label="Per-seed output", interactive=False)
                        tensor_file = gr.File(label="Spherical tensor (.npy, φ × θ × r)")
                    with gr.Tab("Rays"):
                        gr.Markdown("Rays cast from the subject's own centre of mass to the smoothed, "
                                    "direction-specific brain edge (orange), on the harmonized volume. "
                                    "Every third in-plane ray is drawn.")
                        rays_plot = gr.Plot()
                        gr.Markdown("The tensor's sample values drawn back at their positions in the brain:")
                        back_plot = gr.Plot()
                    with gr.Tab("Spherical tensor"):
                        with gr.Row():
                            k = gr.Slider(1, 48, value=24, step=1, label="Radius index (1 = centre, 48 = edge)")
                            show_cam = gr.Checkbox(value=False, label="Overlay Grad-CAM")
                        map_plot = gr.Plot()
                        profile_plot = gr.Plot()
                    with gr.Tab("3-D shells"):
                        with gr.Row():
                            radii = gr.CheckboxGroup(SHELL_CHOICES, value=[0.5, 0.9],
                                                     label="Shells (fraction of R_edge)")
                            color = gr.Radio(["Intensity", "Grad-CAM"], value="Intensity",
                                             label="Colour", interactive=False)
                            cutaway = gr.Checkbox(value=True, label="Cut away anterior-right quadrant")
                        shells_plot = gr.Plot()
                        cam_note = gr.Markdown()

        go_btn.click(run, [file, model, weights, seed, want_cam],
                     [result, table, rays_plot, back_plot, map_plot, profile_plot, shells_plot,
                      color, cam_note, tensor_file, state])
        for ctl in (k, show_cam):
            ctl.change(redraw_map, [state, k, show_cam], map_plot)
        for ctl in (radii, color, cutaway):
            ctl.change(redraw_shells, [state, radii, color, cutaway], shells_plot)
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--host", default="127.0.0.1", help="127.0.0.1 keeps the app private to this computer")
    a = ap.parse_args()
    build().queue().launch(server_name=a.host, server_port=a.port, share=False)
