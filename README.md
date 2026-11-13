# Is template registration necessary? A subject-centred spherical representation of structural MRI for Alzheimer's disease classification

Code accompanying the manuscript submitted to *Computerized Medical Imaging and Graphics* (citation to be added on publication).

Each skull-stripped T1-weighted MRI is resampled along rays cast from the subject's own centre of mass. The rays end at a smoothed, direction-specific brain edge and give a 36 × 72 × 48 tensor (polar angle × azimuth × radius). This tensor stands in for template registration. Two deep networks are trained on it to separate cognitively normal (CN) from Dementia participants in ADNI: a 3-D CNN with circular padding in azimuth, and a Spherical Transformer over radial-profile tokens. They are compared with PCA-based classifiers, with an ANTs affine/SyN registration reference, and with a native-space Cartesian volume.

## Contents

| Path | Purpose | Manuscript |
|---|---|---|
| `cohort/assign_split.py` | Stratified 70/15/15 split (seed 42) | §2.1 |
| `cohort/build_manifest.py` | Writes `cohort/manifest.csv`: volume paths, voxel spacing, acquisition-protocol proxy | §2.1 |
| `preprocessing/convert.py`, `skullstrip.py` | DICOM → NIfTI, HD-BET skull stripping | §2.2 |
| `preprocessing/harmonize.py` | N4 bias-field correction, 1.2 mm isotropic resampling | §2.2 |
| `spherical_transform.py` | Spherical ray representation (Eqs. 1–8) and the tensor cache | §2.3 |
| `models.py` | CNN3D (72,001 parameters) and Spherical Transformer (244,801) | §2.4–2.5 |
| `train.py` | Training: class × protocol balanced sampling, augmentation, early stopping on validation AUC, top-5 checkpoint averaging | §2.6 |
| `evaluate.py` | PCA(50) + cosine / SVM / XGBoost; all metrics, paired bootstrap (B = 2000), Holm-adjusted comparisons, protocol subgroups, cross-protocol analysis | §2.5, §2.7–2.8 |
| `evaluate_demographics.py` | Test AUC by sex and age tertile; protocol by ADNI phase | §2.8 |
| `preprocessing/register.py` | ANTs affine and SyN to MNI152 2009 (registration reference), Harvard–Oxford regional volumes | §2.9 |
| `preprocessing/native_volume.py` | Native-space Cartesian volume baseline (no registration) | §2.9 |
| `preprocessing/benchmark_timing.py` | Per-subject preprocessing time | §2.10 |
| `gradcam.py` | Grad-CAM group maps (supplementary) | Suppl. |
| `figures_spherical.py` | Method figures (rays, group-effect map, edge diagnostics) | Figs. 1, 4, S |
| `run_experiments.ps1` | Runs every training configuration and seed, then `evaluate.py` | — |
| `notebooks/results.ipynb`, `notebooks/registration_comparison.ipynb` | Figures and tables of the paper, computed from `results/` and the training runs | §3 |
| `results/` | Aggregate results reported in the paper (no per-subject data) | §3 |

Trained weights for every configuration and seed (safetensors, best-epoch and top-5 averaged) are at
<https://huggingface.co/lalodeavilaarm/adni-spherical-classification>.

## Installation

Python 3.12 and a CUDA GPU (the results were produced on an RTX 4090).

```bash
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

The commands below use the Windows launcher `py -3.12`; on Linux or macOS use `python`.

## Reproducing the experiments

ADNI data cannot be redistributed. You need your own ADNI access ([adni.loni.usc.edu](https://adni.loni.usc.edu)).

1. **Cohort.** Create `cohort/cohort.csv` with columns `PTID`, `Diagnosis` (`CN` or `Dementia`) and `split`, one row per participant, **sorted by PTID**. Inclusion criteria (manuscript §2.1):
   - The diagnosis comes from the ADNI Diagnostic Summary (DXSUM) record closest in time to the participant's screening MMSE.
   - The participant has a baseline MPRAGE.
   - Skull stripping succeeds.

   Fill the `split` column with `assign_split.assign_split`. Running `py -3.12 cohort/assign_split.py` then checks the stored split, and on the same subject set it reproduces the paper's partition exactly.
2. **Preprocessing.** For each subject, `preprocessing/convert.py` then `preprocessing/skullstrip.py` produce `<root>/<PTID>/brain.nii.gz`.
3. **Manifest and spherical tensors.**
   ```bash
   py -3.12 cohort/build_manifest.py --nifti-root <root>
   py -3.12 train.py --cache-only        # N4 + resampling + spherical transform, ~17 min on 16 workers
   ```
4. **Training and evaluation** (5 seeds per model):
   ```powershell
   .\run_experiments.ps1 -CrossProtocol               # spherical CNN3D and Transformer, incl. cross-protocol runs

   py -3.12 preprocessing/register.py                   # ANTs affine + SyN to MNI152, regional volumes (~80 min)
   .\run_experiments.ps1 -Models cnn -Registration    # registration reference

   py -3.12 preprocessing/register.py --spherical       # spherical tensors of the SyN-registered brains
   .\run_experiments.ps1 -RegisteredSpherical

   py -3.12 preprocessing/native_volume.py              # native-space Cartesian volumes (~17 min)
   .\run_experiments.ps1 -Models cnn -NativeVolume    # native-volume baseline
   ```
   You can also run a single model with `py -3.12 train.py --model cnn --seeds 0`, and the evaluation with `py -3.12 evaluate.py`. Runs are written to `runs/` and tables to `results/`.
5. **Figures and tables.**
   ```bash
   py -3.12 -m nbconvert --to notebook --execute --inplace notebooks/results.ipynb
   py -3.12 -m nbconvert --to notebook --execute --inplace notebooks/registration_comparison.ipynb
   ```

A quick check that needs no MRI data:

```bash
py -3.12 -c "import torch; from models import CNN3D, SphericalTransformer; x = torch.randn(2, 36, 72, 48); print(CNN3D()(x).shape, SphericalTransformer()(x).shape)"
```

## Data availability

The ADNI Data Use Agreement prohibits sharing participant-level data. This repository therefore contains no images, image-derived tensors, subject identifiers, diagnoses, partition assignments or per-subject outputs. Only code, aggregate results and trained weights are published.

## Acknowledgement

Data used in this work were obtained from the Alzheimer's Disease Neuroimaging Initiative (ADNI) database (adni.loni.usc.edu). If you use ADNI data, follow ADNI's data use and acknowledgement requirements.

## License

MIT, see [LICENSE](LICENSE).
