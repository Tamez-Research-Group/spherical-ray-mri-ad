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

if "dark" not in st.session_state:
    st.session_state.dark = st.query_params.get("theme") == "dark"


def _apply_theme():
    """The theme is sent to the browser when a run starts, so a change takes
    effect after one rerun. (Streamlit's theme option is server-wide; this is a
    single-user local app.)"""
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


def show(name, fn, *args, **params):
    """Display a cached figure; `params` must be small hashable keyword arguments."""
    r = st.session_state.result
    st.image(png(r["key"], name, DARK, tuple(sorted(params.items())), fn, args), use_container_width=True)


# ── sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("**Spherical-ray MRI**")
    st.toggle("Dark mode", key="dark")
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
        model = st.selectbox("Model", list(MODELS), format_func=MODELS.get)
        weights = st.selectbox("Weights", list(WEIGHTS), format_func=WEIGHTS.get,
                               help="Best epoch is the paper's primary analysis.")
        seed_opt = st.selectbox("Training seeds", ["all"] + [str(s) for s in P.SEEDS],
                                format_func=lambda s: "All 5 (majority vote)" if s == "all" else f"Seed {s}")
        run = st.form_submit_button("Analyze", type="primary", use_container_width=True)
    st.caption(DISCLAIMER)
    st.caption("Scans are processed on this computer only.")

seeds = P.SEEDS if seed_opt == "all" else (int(seed_opt),)
settings = (source, up.name if up else None, atrophy, model, weights, seeds)

st.title("Spherical-ray MRI")
st.markdown("<p class='sub'>Registration-free spherical representation of structural MRI · "
            "CN vs Dementia models trained on ADNI</p>", unsafe_allow_html=True)


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
            ("1. Harmonize", "N4 bias-field correction and 1.2 mm resampling."),
            ("2. Cast rays", "From the brain's centre of mass to its smoothed edge, in 36 × 72 directions."),
            ("3. Sample", "48 points per ray give a 36 × 72 × 48 spherical tensor."),
            ("4. Classify", "CNN3D or Spherical Transformer, five training seeds."))):
        col.markdown(f"**{head}**  \n<span class='note'>{body}</span>", unsafe_allow_html=True)
    st.info("Choose an input in the sidebar and press Analyze.")
    st.stop()

if res["settings"] != settings:
    st.info("The settings have changed. Press Analyze to update the results.")

subj, df, cam, ref = res["subj"], res["df"], res["cam"], reference()
Tn, geo = subj.tensor, subj.geometry
s = P.summarize(df)
is_tf = res["model"] == "transformer"

tabs = st.tabs(["Summary", "Group comparison", "Grad-CAM", "Rays", "Spherical tensor", "3-D shells", "About"])

# Summary
with tabs[0]:
    accent = "#d97706" if s["call"] == "Dementia" else "#059669"
    vote = f"{s['agree']} of {s['n']} seeds" if s["n"] > 1 else "single seed"
    st.markdown(f"<div class='card' style='--accent:{accent}'><div class='label'>Model output</div>"
                f"<div class='value'>{s['call']}</div><div class='detail'>{MODELS[res['model']]} · "
                f"{WEIGHTS[res['weights']].lower()} · {vote}</div></div>", unsafe_allow_html=True)
    if res["example"]:
        st.warning("Example input. The representation is computed normally, but the model output is not "
                   "meaningful for synthetic or template images, which are outside the training data.")
    for w in subj.qc:
        st.warning(w)
    m = st.columns(4)
    m[0].metric("Mean P(Dementia)", f"{s['mean_p']:.3f}")
    m[1].metric("Similarity to Dementia pattern", f"{res['score']:+.2f}",
                help="Correlation between this scan's deviation from the CN mean and the CN − Dementia "
                     "group difference. See Group comparison.")
    m[2].metric("Voxel size (mm)", " × ".join(f"{v:.2f}" for v in subj.spacing))
    m[3].metric("Protocol group", subj.protocol,
                help="slice1.2: 1.2 mm left-right spacing (ADNI-1 style); iso1.0: 1.0 mm or finer.")
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.dataframe(df, hide_index=True, use_container_width=True, column_config={
            "seed": st.column_config.NumberColumn("Seed", format="%d"),
            "P(Dementia)": st.column_config.ProgressColumn("P(Dementia)", min_value=0, max_value=1, format="%.3f"),
            "threshold": st.column_config.NumberColumn("Threshold", format="%.3f",
                                                       help="Chosen on the ADNI validation set (Youden's J)."),
            "call": st.column_config.TextColumn("Call")})
        st.markdown("<p class='note'>Each seed is compared with its own validation threshold; the output "
                    "is the majority vote.</p>", unsafe_allow_html=True)
    with right:
        buf = io.BytesIO()
        np.save(buf, Tn)
        st.download_button("Download spherical tensor (.npy)", buf.getvalue(), "spherical_tensor_36x72x48.npy",
                           use_container_width=True)
        st.download_button("Download model output (.csv)", df.to_csv(index=False), "model_output.csv",
                           use_container_width=True)
        st.markdown(f"<p class='note'>Processed in {subj.seconds:.1f} s on {P.DEVICE.type.upper()}.</p>",
                    unsafe_allow_html=True)

# Group comparison
with tabs[1]:
    st.markdown(f"This scan against the ADNI training groups of the same protocol ({subj.protocol}).")
    f = st.columns([2, 5])[0].select_slider("Shell (r / R_edge)", SHELLS, value=0.3, key="cmp_r")
    show("group", viz.group_maps, Tn, ref, subj.protocol, f=f)
    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        show("score", viz.score_plot, ref, res["score"])
    with c2:
        st.markdown(f"**Similarity to the Dementia pattern: {res['score']:+.2f}**")
        st.markdown(f"<p class='note'>Correlation between this scan's z-score map and the group effect map, "
                    f"over the cells where the groups differ (|d| ≥ {P.D_MIN}). Curves: ADNI validation "
                    f"participants (not used for the group means); on its own this score separates them "
                    f"with AUC {float(ref.data['score_auc']):.2f}. Descriptive only.</p>", unsafe_allow_html=True)
    st.markdown("**On this brain**")
    show("anatomy", viz.anatomy_effects, geo, ref.d, res["z"])

# Grad-CAM
with tabs[2]:
    if is_tf:
        st.warning("Spherical Transformer Grad-CAM has no radial resolution and was unstable across seeds in "
                   "the paper; it is shown for completeness and should not be interpreted.")
    st.markdown("Regions that increase the Dementia output, averaged over the selected seeds.")
    show("cam_rays", viz.cam_rays, cam, geo, transformer=is_tf)
    show("cam_slices", viz.cam_slices, cam, geo)
    show("cam_shells", viz.cam_shells, cam, geo)

# Rays
with tabs[3]:
    c1, c2, c3 = st.columns([2, 3, 1.4])
    density = {"Sparse": 6, "Medium": 3, "Dense": 2}
    step = density[c1.select_slider("Rays shown", list(density), value="Medium")]
    planes = c2.multiselect("Slices", ["axial", "coronal", "sagittal"], default=["axial", "sagittal"])
    edge = c3.checkbox("Edge surface", value=True)
    st.plotly_chart(viz.rays_3d(geo, DARK, step, planes, edge), use_container_width=True,
                    config={"displaylogo": False})
    show("rays2d", viz.slices, geo)

# Spherical tensor
with tabs[4]:
    show("tensor", viz.tensor_maps, Tn)
    st.markdown("<p class='note'>Each map is one shell: elevation (inferior to superior) against azimuth "
                "(right, anterior, left, posterior).</p>", unsafe_allow_html=True)
    show("samples", lambda g, t, dark: viz.slices(g, dark, "tensor", t), geo, Tn)

# 3-D shells
with tabs[5]:
    c1, c2, c3 = st.columns([2, 3, 1.6])
    kind = c1.selectbox("Colour", ["intensity", "gradcam", "deviation"],
                        format_func={"intensity": "Intensity", "gradcam": "Grad-CAM",
                                     "deviation": "Deviation from CN (z)"}.get)
    radii = c2.multiselect("Shells (r / R_edge)", SHELLS + [1.0], default=[0.3, 0.5, 0.9])
    cut = c3.checkbox("Cut away front-right", value=False)
    vals = {"intensity": Tn, "gradcam": cam, "deviation": res["z"]}[kind]
    st.plotly_chart(viz.shells_3d(vals, geo, DARK, radii or [0.9], kind, cut), use_container_width=True,
                    config={"displaylogo": False})

# About
with tabs[6]:
    st.markdown(f"""
**Method.** The skull-stripped T1 volume is bias-corrected (N4) and resampled to 1.2 mm. Rays from the
brain's centre of mass in 36 × 72 directions reach a smoothed, direction-specific brain edge and are sampled
at 48 fractions of their length, giving a 36 × 72 × 48 tensor. No registration to a template is used.

**Models.** CNN3D (72,001 parameters) and Spherical Transformer (244,801), five training seeds each,
trained on ADNI CN and Dementia participants. Weights: huggingface.co/{P.HF_REPO}.

**Group comparison.** Means and standard deviations of the ADNI training partition per acquisition protocol;
group effect as protocol-adjusted Cohen's d (paper, Fig. 4). Only aggregate statistics are distributed.

**Reproducibility.** The toolkit runs the repository's own preprocessing, transform and model code. On ADNI
test subjects it reproduces the paper's tensors exactly and its model probabilities to within 1e-7.

**Citation.** *Is template registration necessary? A subject-centred spherical representation of structural
MRI for Alzheimer's disease classification* (citation to be added on publication). Data used in developing
the models were obtained from the Alzheimer's Disease Neuroimaging Initiative (ADNI) database.

{DISCLAIMER}
""")
