"""Spherical-ray MRI toolkit: local web app (Streamlit).

    python toolkit/app.py            # opens http://localhost:8501

Upload a skull-stripped T1-weighted NIfTI, pick a model, and explore the
harmonized spherical representation (3-D rays, slices, shells, unwrapped
tensor), the model output of each training seed and, optionally, Grad-CAM.
Everything runs on this machine: the server listens on localhost only and
Streamlit's usage statistics are switched off. Weights are downloaded once
from Hugging Face (or read from SPHERICAL_WEIGHTS_DIR).

Research use only. Not a medical device; the output is not a diagnosis.
"""
from __future__ import annotations

import hashlib
import io
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _launch():
    """`python toolkit/app.py`: start Streamlit with private, local-only settings."""
    from streamlit.web import cli
    sys.argv = ["streamlit", "run", str(Path(__file__).resolve()),
                "--server.address", "localhost", "--browser.gatherUsageStats", "false",
                "--server.maxUploadSize", "500", "--theme.base", "light",
                "--theme.primaryColor", "#2563eb", *sys.argv[1:]]
    sys.exit(cli.main())


if __name__ == "__main__":
    from streamlit.runtime import exists
    if not exists():
        _launch()

sys.path.insert(0, str(HERE))

import numpy as np          # noqa: E402
import pandas as pd         # noqa: E402
import streamlit as st      # noqa: E402

import examples             # noqa: E402
import pipeline as P        # noqa: E402
import viz                  # noqa: E402

st.set_page_config(page_title="Spherical-ray MRI toolkit", page_icon="🧠", layout="wide")
st.markdown("""
<style>
  .block-container {padding-top: 2rem; max-width: 1400px;}
  [data-testid="stMetricValue"] {font-size: 1.6rem;}
  .verdict {border-radius: 12px; padding: 1.1rem 1.4rem; margin-bottom: .6rem;
            border: 1px solid rgba(0,0,0,.08);}
  .verdict.cn {background: #ecfdf5;} .verdict.dem {background: #fff7ed;}
  .verdict h2 {margin: 0; font-size: 1.5rem;} .verdict p {margin: .2rem 0 0; color: #475569;}
  .note {font-size: .85rem; color: #64748b;}
</style>""", unsafe_allow_html=True)

DISCLAIMER = (
    "**Research use only.** Not a medical device; the output is **not a diagnosis**. The models were "
    "trained to separate cognitively normal (CN) from Dementia participants of ADNI (baseline "
    "1.0–1.2 mm MPRAGE, HD-BET skull stripping). They have never seen MCI or other conditions, and "
    "their behaviour on other scanners, protocols or populations is unknown.")
MODEL_LABEL = {"cnn": "CNN3D", "transformer": "Spherical Transformer"}
EXAMPLES = {"none": "—", "phantom": "Synthetic phantom", "mni152": "MNI152 template (average brain)"}
EXAMPLE_NOTE = (
    "**Example input, not a real scan.** The pipeline (rays, shells, tensor) works on it, but the "
    "**model output is meaningless**: the models only know real ADNI scans, and these examples are far "
    "outside that distribution. In our tests the CNN3D called the phantom *Dementia* and the template *CN* "
    "with near certainty, which shows why outputs outside the training distribution cannot be trusted.")
WEIGHT_LABEL = {"best": "Best epoch (primary)", "avg": "Top-5 checkpoint average"}


# ── cached computation ───────────────────────────────────────────────────────

@st.cache_data(show_spinner=False, max_entries=4)
def process_bytes(data: bytes, name: str):
    suffix = ".nii.gz" if name.endswith(".gz") else ".nii"
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / f"{name.removesuffix('.gz').removesuffix('.nii')}{suffix}"
        p.write_bytes(data)
        return P.process(p)


@st.cache_data(show_spinner=False, max_entries=12)
def example_bytes(kind: str, atrophy: float) -> bytes:
    img = examples.phantom(atrophy) if kind == "phantom" else examples.mni152()
    return img.to_bytes()


@st.cache_data(show_spinner=False, max_entries=16)
def predict_cached(key: str, _T, model, weights, seeds):
    return P.predict(_T, model, weights, seeds)


@st.cache_data(show_spinner=False, max_entries=8)
def gradcam_cached(key: str, _T, model, weights, seeds):
    return P.gradcam(_T, model, weights, seeds)


# ── sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🧠 Spherical-ray MRI")
    st.caption("Registration-free spherical representation of structural MRI")
    up = st.file_uploader("Skull-stripped T1 (NIfTI)", type=["nii", "gz"],
                          help="A skull-stripped T1-weighted volume with zero background, e.g. HD-BET output.")
    example = None
    if up is None:
        example = st.selectbox("…or try an example", list(EXAMPLES), format_func=EXAMPLES.get,
                               help="No real participant data: a synthetic phantom or the MNI152 average template.")
        atrophy = (st.slider("Phantom atrophy", 0.0, 1.0, 0.0, 0.1,
                             help="Enlarges ventricles and sulci and thins the cortex")
                   if example == "phantom" else 0.0)
    st.divider()
    model = st.selectbox("Model", list(MODEL_LABEL), format_func=MODEL_LABEL.get)
    weights = st.radio("Weights", list(WEIGHT_LABEL), format_func=WEIGHT_LABEL.get)
    seed_opt = st.selectbox("Training seeds", ["all"] + [str(s) for s in P.SEEDS],
                            format_func=lambda s: "All 5 (majority vote)" if s == "all" else f"Seed {s}")
    want_cam = st.toggle("Grad-CAM", value=False, help="Gradient-weighted class activation of the Dementia logit.")
    st.divider()
    st.caption(DISCLAIMER)
    st.caption("Scans are processed on this computer and never uploaded anywhere.")

seeds = P.SEEDS if seed_opt == "all" else (int(seed_opt),)

st.title("Spherical-ray MRI toolkit")

preload = os.environ.get("SPHERICAL_TOOLKIT_INPUT")      # optional: open this file at start
if up is not None:
    data, up_name = up.getvalue(), up.name
elif example in ("phantom", "mni152"):
    data, up_name = example_bytes(example, atrophy), f"example_{example}.nii"
elif preload and Path(preload).exists():
    data, up_name = Path(preload).read_bytes(), Path(preload).name
else:
    st.markdown(
        "Upload a **skull-stripped T1-weighted NIfTI** in the sidebar, or pick an example. The toolkit runs "
        "the paper's pipeline (no registration to a template):")
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown("**1 · Harmonize**  \nN4 bias-field correction, 1.2 mm isotropic resampling")
    c2.markdown("**2 · Cast rays**  \nfrom the subject's centre of mass to the smoothed brain edge")
    c3.markdown("**3 · Sample**  \n36 elevations × 72 azimuths × 48 radii = one spherical tensor")
    c4.markdown("**4 · Classify**  \nCNN3D or Spherical Transformer, 5 training seeds")
    st.info(DISCLAIMER)
    st.stop()

key = hashlib.sha1(data).hexdigest()
with st.spinner("N4 bias correction, resampling and spherical transform…"):
    try:
        subj = process_bytes(data, up_name)
    except Exception as e:
        st.error(f"Could not process the volume: {e}")
        st.stop()
T, geo = subj.tensor, subj.geometry
with st.spinner("Loading weights and predicting…"):
    df = predict_cached(f"{key}{model}{weights}{seeds}", T, model, weights, seeds)
cam = None
if want_cam:
    with st.spinner("Grad-CAM…"):
        cam = gradcam_cached(f"{key}{model}{weights}{seeds}", T, model, weights, seeds)
s = P.summarize(df)

# ── headline ─────────────────────────────────────────────────────────────────

cls = "dem" if s["call"] == "Dementia" else "cn"
vote = f"{s['agree']} of {s['n']} seeds agree" if s["n"] > 1 else "single seed"
st.markdown(f"<div class='verdict {cls}'><h2>Model output: {s['call']}</h2>"
            f"<p>{MODEL_LABEL[model]} · {WEIGHT_LABEL[weights].lower()} · {vote}</p></div>",
            unsafe_allow_html=True)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Mean P(Dementia)", f"{s['mean_p']:.3f}")
m2.metric("Seeds voting Dementia", f"{int((df['call'] == 'Dementia').sum())} / {len(df)}")
m3.metric("Voxel size (mm)", " × ".join(f"{v:.2f}" for v in subj.spacing))
m4.metric("Processing", f"{subj.seconds:.1f} s", P.DEVICE.type.upper(), delta_color="off")
if up is None and example in ("phantom", "mni152"):
    st.warning(EXAMPLE_NOTE, icon="🧪")
for w in subj.qc:
    st.warning(w, icon="⚠️")
if subj.protocol == "slice1.2":
    st.info("1.2 mm acquisition: in ADNI this protocol coincides with the ADNI-1 phase and a higher "
            "Dementia prevalence. The paper controls for it, but treat single outputs with care.", icon="ℹ️")

tab_res, tab_rays, tab_shells, tab_map, tab_about = st.tabs(
    ["Result", "Rays", "3-D shells", "Spherical tensor", "About"])

with tab_res:
    left, right = st.columns([3, 2])
    with left:
        st.markdown("##### Output per training seed")
        st.dataframe(
            df, hide_index=True, width="stretch",
            column_config={
                "seed": st.column_config.NumberColumn("Seed", format="%d"),
                "P(Dementia)": st.column_config.ProgressColumn("P(Dementia)", min_value=0, max_value=1, format="%.3f"),
                "threshold": st.column_config.NumberColumn("Threshold", format="%.3f",
                                                           help="Chosen on the validation set (Youden's J)"),
                "call": st.column_config.TextColumn("Call")})
        st.markdown("<p class='note'>Each seed's probability is compared with that seed's own threshold, "
                    "chosen on the validation set (Youden's J), as in the paper. The headline is the "
                    "majority vote.</p>", unsafe_allow_html=True)
    with right:
        st.markdown("##### Spherical tensor")
        st.markdown(f"Shape **{' × '.join(map(str, T.shape))}** (elevation φ × azimuth θ × radius r)")
        buf = io.BytesIO()
        np.save(buf, T)
        st.download_button("Download tensor (.npy)", buf.getvalue(),
                           file_name=f"{subj.name}_spherical_36x72x48.npy", width="stretch")
        st.download_button("Download per-seed output (.csv)",
                           df.assign(model=model, weights=weights).to_csv(index=False),
                           file_name=f"{subj.name}_{model}_{weights}.csv", width="stretch")

with tab_rays:
    c1, c2, c3 = st.columns([2, 2, 1])
    density = {"sparse": 6, "medium": 3, "dense": 2}
    step = density[c1.select_slider("Ray density", options=list(density), value="medium")]
    planes = c2.multiselect("Slices", ["axial", "coronal", "sagittal"], default=["axial", "sagittal"])
    edge = c3.toggle("Edge surface", value=True)
    st.plotly_chart(viz.rays_3d(geo, step, planes, edge), use_container_width=True,
                    config={"displaylogo": False})
    st.markdown("<p class='note'>Drag to rotate, scroll to zoom. Rays start at the subject's own centre "
                "of mass and end at the smoothed, direction-specific brain edge R<sub>edge</sub>; "
                "each ray is sampled at 48 fractions of its length.</p>", unsafe_allow_html=True)
    st.markdown("##### On the subject's slices")
    st.pyplot(viz.slices(geo, mode="rays"), width="stretch")
    with st.expander("What the network sees: the tensor's samples drawn back into the brain"):
        st.pyplot(viz.slices(geo, T, mode="tensor"), width="stretch")

with tab_shells:
    c1, c2, c3 = st.columns([3, 2, 1])
    radii = c1.multiselect("Shells (fraction of R_edge)", [0.3, 0.5, 0.7, 0.9, 1.0], default=[0.5, 0.9])
    color = c2.radio("Colour", ["Intensity", "Grad-CAM"], horizontal=True, disabled=cam is None,
                     help=None if cam is not None else "Switch on Grad-CAM in the sidebar")
    cut = c3.toggle("Cut-away", value=True)
    st.plotly_chart(viz.shells(T, geo, radii or [0.9], cam if color == "Grad-CAM" else None, cut),
                    use_container_width=True, config={"displaylogo": False})
    if cam is not None and model == "transformer":
        st.caption("Transformer Grad-CAM has no radial resolution and was unstable across seeds in the "
                   "paper (r ≈ 0.12); do not interpret it.")

with tab_map:
    c1, c2 = st.columns([4, 1])
    k = c1.slider("Radius index (1 = centre, 48 = edge)", 1, T.shape[2], T.shape[2] // 2)
    show_cam = c2.toggle("Grad-CAM overlay", value=cam is not None, disabled=cam is None)
    st.pyplot(viz.unwrapped(T, k - 1, cam if show_cam else None), width="stretch")
    st.pyplot(viz.radial_profiles(T, cam), width="content")

with tab_about:
    st.markdown("""
**Method.** Each skull-stripped T1 volume is bias-corrected (N4) and resampled to 1.2 mm. Rays are cast
from the subject's own centre of mass in 36 × 72 directions to a smoothed, direction-specific brain edge
and sampled at 48 fractions of their length, giving a 36 × 72 × 48 tensor. The representation replaces
template registration: no subject is warped to an atlas. Two networks are trained on it: a 3-D CNN with
circular padding in azimuth (72,001 parameters) and a Spherical Transformer over 648 radial-profile tokens
(244,801 parameters), each with 5 training seeds.

**Reproducibility.** The toolkit runs the repository's own preprocessing, transform and model code with
the published weights. On ADNI test subjects it reproduces the paper's tensors bit for bit and its model
probabilities to within 1e-7.

**Citation.** *Is template registration necessary? A subject-centred spherical representation of
structural MRI for Alzheimer's disease classification* (citation to be added on publication). Data used in
developing the models were obtained from the Alzheimer's Disease Neuroimaging Initiative (ADNI) database.
""")
    st.info(DISCLAIMER)
