# Spherical-ray MRI toolkit

A local web application and command-line tool that apply the paper's pipeline to one skull-stripped T1-weighted MRI:

- the subject-centred spherical representation (rays, 36 × 72 × 48 tensor, shells);
- the CNN3D or Spherical Transformer output for each training seed;
- the scan compared with the ADNI CN and Dementia groups;
- Grad-CAM of the Dementia output.

The toolkit runs the repository's own code (`preprocessing/harmonize.py`, `spherical_transform.py`, `models.py`, `gradcam.py`) with the published weights. On ADNI test subjects it reproduces the paper's tensors exactly and its model probabilities to within 1e-7. All figures follow the paper's conventions: the same colour maps, scales and axes.

> **Research use only.** Not a medical device; the output is not a diagnosis. The models were trained on ADNI CN and Dementia participants (baseline 1.0–1.2 mm MPRAGE, HD-BET skull stripping). They have not seen MCI or other conditions, and their behaviour on other scanners, protocols or populations is unknown.

## Install and run

Starting from the repository's environment (`requirements.txt`, Python 3.12):

```bash
pip install -r toolkit/requirements.txt
python toolkit/app.py            # opens http://localhost:8501
```

The app listens on this computer only, and Streamlit's usage statistics are switched off, so scans never leave the machine.

The weights (about 60 MB) are downloaded from [Hugging Face](https://huggingface.co/lalodeavilaarm/adni-spherical-classification) the first time each one is used. To work offline, set `SPHERICAL_WEIGHTS_DIR` to a local copy of that repository. A GPU is optional: on a CPU, one analysis takes a few seconds.

## Using the app

1. In the sidebar, choose the input: upload a scan, or one of two examples with no participant data (a synthetic phantom with adjustable atrophy, or the MNI152 template).
2. Choose the model, the weights (best epoch, the paper's main analysis, or the top-5 average) and the training runs (all five with a majority vote, or one).
3. Press **Analyze**. Changing a setting afterwards does not recompute anything until you press Analyze again.

The **Dark mode** switch changes the theme of the app and of every figure. Every section has an info button (ⓘ) that explains, in plain language, how to read it.

| Section | Contents |
|---|---|
| Summary | Model output (majority vote of the training runs), mean probability, similarity to the dementia pattern, input checks, per-run probabilities and thresholds; downloads of the spherical representation (`.npy`) and the output (`.csv`) |
| Comparison with ADNI groups | The scan beside the ADNI normal and dementia averages of the same scanner protocol, at a chosen depth; the typical dementia pattern (Cohen's d, as in the paper's Fig. 4) beside the scan's difference from normal; the similarity score against the ADNI validation distributions; the three standard views with a selector: brain, typical dementia pattern, this scan vs normal, or the spherical sampling |
| Rays and attention | Interactive 3-D rays from the centre of the brain to its edge, with slices, and the same rays on the three views. Selecting **Model attention** colours the rays by Grad-CAM (as in Fig. S gradcam_rays), overlays Grad-CAM on the slices (Fig. S gradcam), and adds attention by direction and by depth |
| 3-D shells | Interactive nested shells coloured by image intensity, model attention or difference from normal. Outer shells are see-through unless the front-right quadrant is cut away |

**Similarity to the dementia pattern.** This is the correlation between the scan's z-score map against the CN mean and the CN − Dementia effect map, over the cells where the groups differ (|d| ≥ 0.15, polar caps excluded). It is descriptive: on ADNI validation participants, who were not used to compute the group statistics, it separates CN from Dementia with an AUC of 0.85. It is not one of the paper's classifiers.

**Reference data.** `toolkit/reference/adni_reference.npz` holds aggregate statistics of the ADNI training partition: the mean and SD tensors per group and acquisition protocol (n = 92 to 496 per cell), the protocol-adjusted Cohen's d map, and smoothed density curves of the similarity score on the validation partition. It contains no participant-level data. `toolkit/build_reference.py` rebuilds it from the spherical-tensor cache.

## Examples

Both examples contain no participant data:

- **Synthetic phantom** (`toolkit/examples.py`): a T1-like brain generated from simple shapes. It has a folded cortex, white matter, ventricles, deep grey matter, cerebellum and brainstem, plus a bias field and noise.
- **MNI152 template**: the 2009 average of 152 adult brains, which ships with nilearn.

The representation is computed normally on them, but **the model output is meaningless**, because they are far from real ADNI scans. In our tests the CNN3D called the phantom Dementia and the template CN, both with near certainty.

```bash
python toolkit/examples.py phantom --out phantom.nii.gz --atrophy 0.6
```

## Command line

```bash
python toolkit/cli.py scans/*.nii.gz --model cnn --weights best --out predictions.csv --save-tensors tensors/
```

The CSV has one row per scan and seed, with `P(Dementia)`, the seed's threshold, the call and the protocol group.

## Input requirements

- A NIfTI file (`.nii` or `.nii.gz`) of a **skull-stripped** T1-weighted volume with background exactly 0. The paper used [HD-BET](https://github.com/MIC-DKFZ/HD-BET) (`preprocessing/skullstrip.py`).
- Any orientation or voxel size. The volume is reoriented to RAS, and the geometry is computed in millimetres.
- No registration to a template.

The app warns when an input does not look skull-stripped, has an unusual brain volume or has coarse voxels.
