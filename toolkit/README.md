# Spherical-ray MRI toolkit

A local web app and command-line tool for trying the paper's pipeline on your own scans. Upload a skull-stripped T1-weighted MRI and choose a model. The toolkit then:

1. runs the paper's harmonization (N4 bias-field correction, 1.2 mm isotropic resampling) and spherical ray transform (36 × 72 × 48 tensor);
2. shows the rays cast from the subject's centre of mass to the smoothed brain edge, as an interactive 3-D view and on the subject's own axial, coronal and sagittal slices;
3. shows the spherical tensor as unwrapped shells (one per radius, with a slider) and as interactive 3-D shells;
4. returns the output of the CNN3D or the Spherical Transformer for each of the 5 training seeds, each compared with that seed's validation threshold, plus a majority vote;
5. optionally shows Grad-CAM of the Dementia logit on the shells and the unwrapped maps.

The toolkit runs the repository's own code (`preprocessing/harmonize.py`, `spherical_transform.py`, `models.py`, `gradcam.py`) with the published weights. On ADNI test subjects it reproduces the paper's spherical tensors bit for bit and its model probabilities to within 1e-7.

> **Research use only.** This is not a medical device and its output is not a diagnosis. The models were trained only on ADNI participants (cognitively normal vs Dementia; baseline 1.0–1.2 mm MPRAGE; HD-BET skull stripping). They have not seen MCI or other conditions, and their behaviour on other scanners, protocols or populations is unknown.

## Install

Starting from the repository's environment (`requirements.txt`, Python 3.12):

```bash
pip install -r toolkit/requirements.txt
```

The weights (about 60 MB in total) are downloaded from [Hugging Face](https://huggingface.co/lalodeavilaarm/adni-spherical-classification) the first time each one is used, then reused from the local Hugging Face cache. To work offline, download the repository once and point `SPHERICAL_WEIGHTS_DIR` at the folder that contains `weights/`.

A GPU is optional. On a CPU, processing one scan takes a few seconds and a 5-seed prediction under a second.

## Web app

```bash
python toolkit/app.py            # opens http://localhost:8501
```

The app is built with [Streamlit](https://streamlit.io). `python toolkit/app.py` starts it so that it listens only on this computer, with Streamlit's usage statistics switched off; uploaded scans never leave the machine. (`streamlit run toolkit/app.py` also works but uses Streamlit's default settings.)

| Control | Options |
|---|---|
| Model | CNN3D, Spherical Transformer |
| Weights | Best epoch (the paper's primary analysis) or the top-5 checkpoint average |
| Training seeds | All 5 (majority vote) or a single seed |
| Grad-CAM | On or off. Transformer Grad-CAM has no radial resolution and was unstable across seeds in the paper; it is shown but should not be interpreted |

| Tab | Shows |
|---|---|
| Result | Model output, per-seed probabilities and thresholds, input checks; download the spherical tensor (`.npy`) and the per-seed output (`.csv`) |
| Rays | Interactive 3-D rays, ray ends and the edge surface R_edge with orthogonal slices (adjustable density); the same on 2-D slices; the tensor's samples drawn back into the brain |
| Spherical tensor | One shell r / R_edge at a time as an elevation × azimuth map (with optional Grad-CAM), and the mean radial profile |
| About | Method summary and citation |
| 3-D shells | Interactive shells at chosen fractions of R_edge, coloured by intensity or Grad-CAM, with an optional cut-away |

## No scan at hand? Examples

The sidebar offers two examples that contain no participant data:

- **Synthetic phantom.** A T1-like, skull-stripped brain generated from simple shapes (`toolkit/examples.py`): folded cortex with sulci, white matter, ventricles, deep grey matter, cerebellum and brainstem, with a bias field and noise. An *atrophy* slider enlarges the ventricles and sulci and thins the cortex, so you can watch the rays and shells respond.
- **MNI152 template.** The MNI152 2009 average of 152 adult brains, which ships with nilearn, skull-stripped with its own mask.

Both are useful for exploring the representation, but **the model output on them is meaningless**: they are far from the real ADNI scans the models were trained on. In our tests the CNN3D called the phantom Dementia and the template CN, both with near certainty. That is a reminder that out-of-distribution outputs can't be trusted.

To write either example to a file, for instance to try the command-line tool:

```bash
python toolkit/examples.py phantom --out phantom.nii.gz --atrophy 0.6
python toolkit/examples.py mni152 --out mni152_brain.nii.gz
```

## Command line

```bash
python toolkit/cli.py scans/*.nii.gz --model cnn --weights best --out predictions.csv --save-tensors tensors/
```

The output has one row per scan and seed: `P(Dementia)`, the seed's threshold, the call and the acquisition-protocol group.

## Input requirements

- A NIfTI file (`.nii` or `.nii.gz`) of a **skull-stripped** T1-weighted volume, with background exactly 0. The paper used [HD-BET](https://github.com/MIC-DKFZ/HD-BET) (`preprocessing/skullstrip.py`).
- Any orientation; the volume is reoriented to RAS. Any voxel size; the geometry is computed in millimetres from the header.
- No registration to a template is needed. That is the point of the method.

The app warns when an input does not look skull-stripped, has a brain volume outside the usual range, or has coarse voxels. It also flags the 1.2 mm acquisition protocol, which in ADNI coincides with the ADNI-1 phase and a higher Dementia prevalence (see the paper's protocol analysis).
