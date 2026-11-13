"""Spherical-ray MRI toolkit: local web app (Streamlit).

    python toolkit/app.py            # opens http://localhost:8501

Choose a skull-stripped T1-weighted NIfTI (or an example), a model, and press
Analyze. The app shows the subject-centred spherical representation, the model
output per training seed, the scan compared with the ADNI CN and Dementia
groups, and Grad-CAM. It runs on this computer only (localhost, Streamlit
usage statistics off); weights come from Hugging Face or SPHERICAL_WEIGHTS_DIR.

Research use only. Not a medical device; the output is not a diagnosis.
"""
from __future__ import annotations

import hashlib
import io
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _launch():
    """`python toolkit/app.py`: start Streamlit with private, local-only settings."""
    from streamlit.web import cli
    sys.argv = ["streamlit", "run", str(Path(__file__).resolve()),
                "--server.address", "localhost", "--browser.gatherUsageStats", "false",
                "--server.maxUploadSize", "500", "--client.toolbarMode", "minimal", "--theme.base", "light", "--theme.primaryColor", "#2563eb",
                *sys.argv[1:]]
    sys.exit(cli.main())


if __name__ == "__main__":
    from streamlit.runtime import exists
    if not exists():
        _launch()

sys.path.insert(0, str(HERE))

import matplotlib.pyplot as plt     # noqa: E402
import numpy as np                  # noqa: E402
import streamlit as st              # noqa: E402

import examples                     # noqa: E402
import pipeline as P                # noqa: E402
import viz                          # noqa: E402

st.set_page_config(page_title="Spherical-ray MRI", layout="wide")

MODELS = {"cnn": "CNN3D", "transformer": "Spherical Transformer"}
WEIGHTS = {"best": "Best epoch", "avg": "Top-5 average"}
SOURCES = {"upload": "Upload a scan", "phantom": "Synthetic phantom", "mni152": "MNI152 template"}
SHELLS = [0.3, 0.5, 0.7, 0.9]
DISCLAIMER = "Research use only. Not a medical device; the output is not a diagnosis."


# ── theme ────────────────────────────────────────────────────────────────────

# View settings live in session state so they survive the rerun that applies a theme change
# (Streamlit forgets the state of widgets that were not drawn in the interrupted run).
VIEW_DEFAULTS = {"dark": st.query_params.get("theme") == "dark", "section_ctl": "Summary",
                 "views_mode": "anatomy", "rays_colour": "Plain", "rays_density": "Medium",
                 "rays_planes": ["axial", "sagittal"], "rays_edge": True, "shell_kind": "intensity",
                 "shell_radii": [0.3, 0.5, 0.9], "shell_cut": False, "cmp_r": 0.3}
for _k, _v in VIEW_DEFAULTS.items():
    st.session_state[_k] = st.session_state.get(_k, _v)


def _set_theme():
    """Called by the Dark mode switch before the page is redrawn, so the new
    theme applies in the same run. (Streamlit's theme option is server-wide;
    this is a single-user local app.)"""
    want = "dark" if st.session_state.dark else "light"
    st._config.set_option("theme.base", want)
    st.query_params["theme"] = want


def _apply_theme():
    """On first load (e.g. ?theme=dark) the theme may differ from the server's;
    one rerun applies it before any widget has been used."""
    want = "dark" if st.session_state.dark else "light"
    st.query_params["theme"] = want
    if st._config.get_option("theme.base") != want:
        st._config.set_option("theme.base", want)
        st.rerun()


_apply_theme()
DARK = st.session_state.dark
T = viz.THEMES[DARK]
st.markdown(f"""
<style>
  .block-container {{padding-top: 1.6rem; max-width: 1320px;}}
  h1 {{font-size: 1.55rem !important; font-weight: 600 !important; margin-bottom: 0 !important;}}
  .sub {{color: {T['muted']}; margin: 0 0 1rem;}}
  .card {{border: 1px solid {T['grid']}; border-left: 5px solid var(--accent); border-radius: 8px;
          padding: .9rem 1.2rem; margin: .2rem 0 1rem;}}
  .card .label {{color: {T['muted']}; font-size: .8rem; text-transform: uppercase; letter-spacing: .04em;}}
  .card .value {{font-size: 1.5rem; font-weight: 600; margin: .1rem 0;}}
  .card .detail {{color: {T['muted']}; font-size: .9rem;}}
  .note {{color: {T['muted']}; font-size: .85rem;}}
  [data-testid="stMetricValue"] {{font-size: 1.25rem;}}
</style>""", unsafe_allow_html=True)


# ── cached computation ───────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def reference():
    return P.Reference.load()


@st.cache_data(show_spinner=False, max_entries=4)
def process_bytes(data: bytes, name: str):
    import tempfile
    suffix = ".nii.gz" if data[:2] == b"\x1f\x8b" else ".nii"          # gzip magic bytes
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / f"scan{suffix}"
        p.write_bytes(data)
        return P.process(p)


@st.cache_data(show_spinner=False, max_entries=12)
def example_bytes(kind: str, atrophy: float) -> bytes:
    return (examples.phantom(atrophy) if kind == "phantom" else examples.mni152()).to_bytes()


@st.cache_data(show_spinner=False, max_entries=64)
def png(key: str, name: str, dark: bool, params: tuple, _fn, _args: tuple) -> bytes:
    """Matplotlib figure rendered once per analysis, theme and parameter set."""
    fig = _fn(*_args, dark=dark, **dict(params))
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


VIZ_VERSION = hashlib.md5(Path(viz.__file__).read_bytes()).hexdigest()[:8]   # new figure code -> new images


def show(name, fn, *args, **params):
    """Display a cached figure; `params` must be small hashable keyword arguments."""
    r = st.session_state.result
    st.image(png(r["key"] + VIZ_VERSION, name, DARK, tuple(sorted(params.items())), fn, args),
             use_container_width=True)


def figure(name, fn, *args, caption=None, info=None, widths=(1, 10, 1), **params):
    """A centred figure with a one-line caption; the caption's info icon explains it on hover."""
    with st.columns(widths)[1]:
        show(name, fn, *args, **params)
        if caption:
            st.caption(caption, help=info)


# ── sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("**Spherical-ray MRI**")
    st.toggle("Dark mode", key="dark", on_change=_set_theme)
    source = st.radio("Input", list(SOURCES), format_func=SOURCES.get,
                      help="The phantom and the MNI152 template are examples with no participant data.")
    with st.form("setup", border=False):
        up, atrophy = None, 0.0
        if source == "upload":
            up = st.file_uploader("Skull-stripped T1 (NIfTI)", type=["nii", "gz"],
                                  help="T1-weighted MRI with the skull removed (zero background), e.g. HD-BET output.")
        elif source == "phantom":
            atrophy = st.slider("Phantom atrophy", 0.0, 1.0, 0.0, 0.1,
                                help="Enlarges the ventricles and sulci and thins the cortex.")
        model = st.selectbox("Model", list(MODELS), format_func=MODELS.get,
                             help="CNN3D gave the best results in the paper.")
        weights = st.selectbox("Weights", list(WEIGHTS), format_func=WEIGHTS.get,
                               help="Best epoch is the paper's main analysis; top-5 average is an alternative.")
        seed_opt = st.selectbox("Training runs", ["all"] + [str(s) for s in P.SEEDS],
                                format_func=lambda s: "All 5 (majority vote)" if s == "all" else f"Run {s}",
                                help="Each network was trained five times with different random starts.")
        run = st.form_submit_button("Analyze", type="primary", use_container_width=True)
    st.caption(DISCLAIMER)
    st.caption("Scans are processed on this computer only.")

seeds = P.SEEDS if seed_opt == "all" else (int(seed_opt),)
settings = (source, up.name if up else None, atrophy, model, weights, seeds)

st.title("Spherical-ray MRI")
st.markdown("<p class='sub'>Brain MRI analysis without template alignment · research models trained on "
            "ADNI (cognitively normal vs Dementia)</p>", unsafe_allow_html=True)


# ── analysis (only when Analyze is pressed) ──────────────────────────────────

def analyze():
    if source == "upload":
        preload = os.environ.get("SPHERICAL_TOOLKIT_INPUT")        # optional, for testing
        if up is not None:
            data = up.getvalue()
        elif preload and Path(preload).exists():
            data = Path(preload).read_bytes()
        else:
            st.error("Choose a NIfTI file, or select an example input.")
            return
    else:
        data = example_bytes(source, atrophy if source == "phantom" else 0.0)
    try:
        with st.spinner("Harmonizing and computing the spherical tensor"):
            subj = process_bytes(data, up.name if (source == "upload" and up) else "scan.nii")
    except Exception as e:
        st.error(f"The volume could not be processed: {e}")
        return
    with st.spinner("Running the models and Grad-CAM"):
        df = P.predict(subj.tensor, model, weights, seeds)
        cam = P.scaled_cam(P.gradcam(subj.tensor, model, weights, seeds))
        ref = reference()
    st.session_state.result = dict(
        key=hashlib.sha1(data).hexdigest() + model + weights + str(seeds), subj=subj, df=df, cam=cam,
        z=ref.deviation(subj.tensor, subj.protocol), score=ref.pattern_score(subj.tensor, subj.protocol),
        model=model, weights=weights, seeds=seeds, example=source != "upload", settings=settings)


if run or (os.environ.get("SPHERICAL_TOOLKIT_INPUT") and "result" not in st.session_state):
    analyze()                      # SPHERICAL_TOOLKIT_INPUT: analyze that file at start (testing, demos)

res = st.session_state.get("result")
if res is None:
    c1, c2, c3, c4 = st.columns(4)
    for col, (head, body) in zip((c1, c2, c3, c4), (
            ("1. Prepare", "Corrects scanner brightness variations and resamples to 1.2 mm."),
            ("2. Cast rays", "Lines from the centre of the brain to its outer edge, in 2,592 directions."),
            ("3. Sample", "48 points along every ray summarise the brain."),
            ("4. Classify", "Two networks trained on ADNI compare the scan with CN and Dementia."))):
        col.markdown(f"**{head}**  \n<span class='note'>{body}</span>", unsafe_allow_html=True)
    st.info("Choose an input in the sidebar and press Analyze.")
    st.stop()

if res["settings"] != settings:
    st.info("The settings have changed. Press Analyze to update the results.")

subj, df, cam, ref = res["subj"], res["df"], res["cam"], reference()
Tn, geo = subj.tensor, subj.geometry
s = P.summarize(df)
is_tf = res["model"] == "transformer"

INFO = {
    "output": """
**What the model concluded.** Each of the five training runs ("seeds") of the selected network gives a
probability that the scan belongs to the Dementia group. That probability is compared with a cut-off chosen
on separate ADNI participants. The result shown is the majority of the five decisions.

The models only know cognitively normal (CN) and Dementia participants from ADNI. They were not trained on
MCI or other conditions. This is a research tool, not a diagnosis.""",
    "seeds": """
**One row per training run.** *P(Dementia)* is the model's probability. *Threshold* is the cut-off for that
run, chosen on ADNI validation participants. *Call* is Dementia when the probability is at or above the
threshold.""",
    "groups": """
**This scan next to the ADNI groups.** Each map shows one "shell", the set of points at the same relative
depth along every ray (for example 30% of the way from the centre to the edge). The map is unrolled like a
world map: left to right goes around the head (Right, Anterior, Left, Posterior), bottom to top goes from
inferior to superior.

*Top row*: the average of ADNI cognitively normal (CN) participants, the average of ADNI Dementia
participants, and this scan. All three come from the same scanner protocol as the scan.

*Bottom row, left*: where Dementia brains typically differ from CN brains. *Bottom row, right*: where this
scan differs from the CN average. Red means darker than normal, blue brighter than normal. Areas that are
red in both maps are where this scan departs from normal in the way dementia typically does.""",
    "score": """
**Similarity to the dementia pattern.** Does this scan differ from a normal brain in the same places, and in
the same direction, as dementia brains typically do? Near 0 means no resemblance. A positive value means it
resembles the dementia pattern; a negative value means the opposite.

The curves show where this number falls for ADNI participants who were not used to build the averages:
blue for CN, red for Dementia. On those participants it separates the two groups with an AUC of
{auc:.2f}. It is a visual aid, not the model's decision.""",
    "views": """
**The three standard views** through the centre of the brain:

- *Brain*: the scan, with its outer edge (orange) and centre (dot) as used by the method.
- *Typical dementia pattern*: where ADNI Dementia brains differ from CN brains on average, drawn on this
  brain. Red means darker in dementia; the strongest signal is around the enlarged ventricles.
- *This scan vs normal*: where this scan is darker (red) or brighter (blue) than the CN average. Small
  differences are hidden. Scattered stripes in the cortex are mostly individual anatomy, not disease.
- *Spherical sampling*: the values the networks actually see, drawn back where they were sampled.""",
    "rays": """
**Rays.** The method needs no alignment to a template brain. Instead it finds the centre of this brain and
casts rays outward to the brain's smoothed outer edge, in 36 × 72 directions. It reads 48 values along
each ray. Drag to rotate and scroll to zoom.

With *Model attention* selected, each ray is coloured by how strongly the regions along it pushed the
model towards Dementia: yellow is strong, dark purple weak. Rays pointing almost straight up or down are
grey because the brain edge is unreliable there.""",
    "slices": """
**Slices through the centre of the brain**, with the rays that lie in each plane. With *Model attention*,
the colour shows the regions that most increased the Dementia output. Attention marks where the model
looked, not where tissue is abnormal. Compare it with *This scan vs normal* in the ADNI comparison.""",
    "attention": """
**Where the model looked.**

*Attention by direction*: each pixel is one ray direction, unrolled like a world map. Grey bands are the
near-vertical rays, which are left out.

*Attention by depth*: for the 5% most important rays, how attention changes from the centre of the brain
(left) to its edge (right). Attention concentrated in the outer few percent would suggest the model relies
on the brain outline rather than on internal structure.

The map comes from the network's last layer, so it marks broad regions rather than specific structures.""",
    "shells": """
**Shells.** All points at the same relative depth along every ray form a closed surface shaped like this
brain: 30% of the way out is a small inner shell, 90% is close to the cortex. Colour the shells by image
intensity, by model attention, or by difference from the CN average (red: darker than normal). Without the
cut-away, the outer shells are see-through so the inner ones stay visible.""",
}


def heading(title, info=None):
    st.markdown(f"##### {title}", help=info)


SECTIONS = ["Summary", "Comparison with ADNI groups", "Rays and attention", "3-D shells", "About"]
sec = st.segmented_control("Section", SECTIONS, key="section_ctl",
                           label_visibility="collapsed")
if sec is None:                                   # clicking the active option deselects it
    sec = st.session_state.get("last_section", SECTIONS[0])
st.session_state.last_section = sec

if sec == "Summary":
    accent = "#d97706" if s["call"] == "Dementia" else "#059669"
    vote = f"{s['agree']} of {s['n']} training runs agree" if s["n"] > 1 else "single training run"
    st.markdown(f"<div class='card' style='--accent:{accent}'><div class='label'>Model output</div>"
                f"<div class='value'>{s['call']}</div><div class='detail'>{MODELS[res['model']]} · "
                f"{WEIGHTS[res['weights']].lower()} · {vote}</div></div>", unsafe_allow_html=True)
    st.caption("How the result is obtained", help=INFO["output"])
    if res["example"]:
        st.warning("Example input. The model output is not meaningful for synthetic or template images.")
    for w in subj.qc:
        st.warning(w)
    m = st.columns(4)
    m[0].metric("Mean P(Dementia)", f"{s['mean_p']:.3f}", help="Average of the five training runs.")
    m[1].metric("Similarity to dementia pattern", f"{res['score']:+.2f}",
                help="How much this scan's differences from a normal brain look like the typical dementia "
                     "pattern (0: not at all; positive: similar). See Comparison with ADNI groups.")
    m[2].metric("Voxel size (mm)", " × ".join(f"{v:.2f}" for v in subj.spacing))
    m[3].metric("Protocol group", subj.protocol,
                help="slice1.2: 1.2 mm slices (ADNI-1 style); iso1.0: 1.0 mm or finer. Comparisons use "
                     "ADNI participants of the same group.")
    left, right = st.columns([3, 2], gap="large")
    with left:
        heading("Output per training run", INFO["seeds"])
        st.dataframe(df, hide_index=True, use_container_width=True, column_config={
            "seed": st.column_config.NumberColumn("Run", format="%d"),
            "P(Dementia)": st.column_config.ProgressColumn("P(Dementia)", min_value=0, max_value=1, format="%.3f"),
            "threshold": st.column_config.NumberColumn("Threshold", format="%.3f"),
            "call": st.column_config.TextColumn("Call")})
    with right:
        st.markdown("##### Downloads")
        buf = io.BytesIO()
        np.save(buf, Tn)
        st.download_button("Spherical representation (.npy)", buf.getvalue(), "spherical_tensor_36x72x48.npy",
                           use_container_width=True)
        st.download_button("Model output (.csv)", df.to_csv(index=False), "model_output.csv",
                           use_container_width=True)
        st.caption(f"Processed in {subj.seconds:.1f} s on {P.DEVICE.type.upper()}.")

elif sec == "Comparison with ADNI groups":
    heading(f"This scan next to the ADNI groups (protocol {subj.protocol})")
    f = st.columns([1, 3, 1])[1].select_slider("Depth of the shell (fraction of the way to the edge)", SHELLS,
                                               key="cmp_r")
    figure("group", viz.group_maps, Tn, ref, subj.protocol, f=f,
           caption="Top: normal and dementia averages beside this scan. Bottom: the typical dementia pattern "
                   "beside this scan's difference from normal (red: darker than normal).",
           info=INFO["groups"])
    heading(f"Similarity to the dementia pattern: {res['score']:+.2f}")
    figure("score", viz.score_plot, ref, res["score"], widths=(1, 3, 1),
           caption="Where this scan falls among ADNI normal (blue) and dementia (red) participants.",
           info=INFO["score"].format(auc=float(ref.data["score_auc"])))
    heading("On this brain")
    with st.columns([1, 10, 1])[1]:
        mode = st.segmented_control("Show", list(viz.THREE_VIEW_MODES),
                                    format_func=viz.THREE_VIEW_MODES.get, key="views_mode",
                                    label_visibility="collapsed") or "anatomy"
    figure(f"views-{mode}", lambda dark: viz.three_views(geo, mode, dark, d=ref.d, z=res["z"], T=Tn),
           caption={"anatomy": "The scan with its outer edge (orange) and centre (dot).",
                    "pattern": "Where dementia brains typically differ from normal, drawn on this brain.",
                    "scan": "Where this scan is darker (red) or brighter (blue) than the normal average.",
                    "tensor": "The values the networks see, drawn where they were sampled."}[mode],
           info=INFO["views"])

elif sec == "Rays and attention":
    heading("Rays from the centre of the brain", INFO["rays"])
    c0, c1, c2, c3 = st.columns([2.2, 1.8, 2.6, 1.3], vertical_alignment="bottom")
    colour = c0.segmented_control("Colour", ["Plain", "Model attention"],
                                  key="rays_colour") or "Plain"
    attention = colour == "Model attention"
    density = {"Sparse": 6, "Medium": 3, "Dense": 2}
    step = density[c1.select_slider("Number of rays", list(density), key="rays_density")]
    planes = c2.multiselect("Slices", ["axial", "coronal", "sagittal"],
                            key="rays_planes")
    edge = c3.checkbox("Brain edge", key="rays_edge")
    if attention and is_tf:
        st.caption("Spherical Transformer attention was unstable across training runs in the paper; "
                   "interpret with caution.")
    st.plotly_chart(viz.rays_3d(geo, DARK, step, planes, edge, cam if attention else None),
                    use_container_width=True, config={"displaylogo": False})
    heading("Slices through the centre of the brain")
    figure(f"rayslices-{attention}",
           lambda dark: viz.three_views(geo, "anatomy", dark, cam=cam if attention else None, rays=True),
           caption=("Colour: regions that most increased the dementia output." if attention else
                    "Rays in each plane, ending at the brain's outer edge (orange)."),
           info=INFO["slices"])
    if attention:
        heading("Where the model looked")
        figure("attention", viz.attention_summary, cam, geo, transformer=is_tf,
               caption="Left: attention for each direction. Right: attention from the centre to the edge "
                       "along the most important rays.",
               info=INFO["attention"])

elif sec == "3-D shells":
    heading("Shells at fixed depths", INFO["shells"])
    c1, c2, c3 = st.columns([2, 3, 1.6], vertical_alignment="bottom")
    kind = c1.selectbox("Colour", ["intensity", "gradcam", "deviation"], key="shell_kind",
                        format_func={"intensity": "Image intensity", "gradcam": "Model attention",
                                     "deviation": "Difference from normal"}.get)
    radii = c2.multiselect("Shells (fraction of the way to the edge)", SHELLS + [1.0],
                           key="shell_radii")
    cut = c3.checkbox("Cut away front-right", key="shell_cut")
    vals = {"intensity": Tn, "gradcam": cam, "deviation": res["z"]}[kind]
    st.plotly_chart(viz.shells_3d(vals, geo, DARK, radii or [0.9], kind, cut), use_container_width=True,
                    config={"displaylogo": False})

else:
    st.markdown(f"""
**Method.** The skull-stripped T1 scan is corrected for scanner brightness variations (N4) and resampled to
1.2 mm. Rays from the brain's centre of mass in 36 × 72 directions reach its smoothed outer edge and are
sampled at 48 points each, giving a 36 × 72 × 48 spherical representation. No alignment to a template brain
is used.

**Models.** CNN3D (72,001 parameters) and Spherical Transformer (244,801), five training runs each, trained
on ADNI cognitively normal and Dementia participants. Weights: huggingface.co/{P.HF_REPO}.

**ADNI comparison.** Averages and spreads of the ADNI training participants per scanner protocol; the
typical dementia pattern is the protocol-adjusted effect size of the paper's Fig. 4. Only aggregate
statistics are distributed.

**Model attention** is Grad-CAM of the Dementia output, scaled to its 99th percentile, as in the paper.

**Reproducibility.** The toolkit runs the repository's own preprocessing, transform and model code. On ADNI
test participants it reproduces the paper's representation exactly and its model probabilities to within
1e-7.

**Citation.** *Is template registration necessary? A subject-centred spherical representation of structural
MRI for Alzheimer's disease classification* (citation to be added on publication). Data used in developing
the models were obtained from the Alzheimer's Disease Neuroimaging Initiative (ADNI) database.

{DISCLAIMER}
""")
