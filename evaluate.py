"""
evaluate.py — the five-classifier benchmark and statistical evaluation
(Methods, Sections 2.4, 2.8 and 2.9).

Benchmarks Cosine Similarity, SVM (RBF), XGBoost, CNN3D and the Spherical
Transformer on the same held-out test partition. CNN3D and the Transformer
are read from train.py's runs/<Model>/seed<k>/predictions.csv, in two
variants: best-epoch weights and top-K averaged weights.

For every (configuration, seed):
  - the decision threshold is chosen on the validation partition by
    maximizing Youden's J (sensitivity + specificity - 1) and then applied,
    unchanged, to the test partition;
  - AUC (closed-form rank-sum estimator) and threshold metrics are computed
    on the test partition, with 95% percentile-bootstrap CIs
    (B = 2,000 subject-level resamples).

The same bootstrap resamples are used for every configuration, so
comparisons are paired. Per configuration, the seed-level results are
summarized as mean +- SD across seeds (training variability) and a 95%
bootstrap CI of the seed-mean (test-sampling variability). Every pair of
configurations is compared by a paired-bootstrap test on the difference in
seed-mean AUC, with Holm adjustment across all pairs.

Two confound controls address acquisition protocol, which is associated
with diagnosis in this cohort (cohort/build_manifest.py): a "voxel spacing
only" classifier (training-partition Dementia rate per voxel-spacing
triple; excluded from the pairwise tests) and protocol-stratified test AUCs
for every configuration.

Outputs (results/):
    predictions.csv      long format: config, seed, PTID, split, label, score
    metrics_per_seed.csv one row per (config, seed), with per-seed CIs
    summary.csv          one row per config
    pairwise_auc.csv     paired ΔAUC with CI, bootstrap p and Holm p
    subgroup_auc.csv     test AUC within each acquisition protocol
    cross_protocol.csv   train on one protocol, test on the other
    registration_comparison.csv
                         registration-free vs registration-based pairs (own Holm family)
    registration_spherical_comparison.csv
                         spherical of native vs of SyN-registered brain (own Holm family)
    summary.json         run metadata (n, PCA variance, seeds, B)
"""

import os
import json
import pickle
import hashlib
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from sklearn.decomposition import IncrementalPCA
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import xgboost as xgb

CLASSES   = ["CN", "Dementia"]
LABEL_MAP = {"CN": 0, "Dementia": 1}
N_PCA     = 50      # Methods Eq. 9
N_BOOT    = 2000
BOOT_SEED = 42
SEEDS     = [0, 1, 2, 3, 4]

METRICS = ["AUC", "Acc", "Sens", "Spec", "PPV", "NPV", "F1"]

SPACING_CONTROL = "Voxel spacing only (control)"

DEEP_MODELS = [("CNN3D", "CNN3D"), ("Transformer", "Spherical Transformer"),
               ("CNN3D_mni-syn", "CNN3D, MNI SyN"), ("CNN3D_mni-affine", "CNN3D, MNI affine"),
               ("CNN3D_mni-syn-spherical", "CNN3D, spherical of MNI SyN"),
               ("Transformer_mni-syn-spherical", "Spherical Transformer, spherical of MNI SyN"),
               ("CNN3D_native-volume", "CNN3D, native volume")]
VARIANTS    = [("best", "best epoch"), ("avg", "top-K avg")]


# ── statistics (Methods Section 2.8) ─────────────────────────────────────────

def _confusion_counts(y_true, y_pred):
    """Plain-numpy equivalent of sklearn's confusion_matrix([0,1] labels)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tn, fp, fn, tp


def _manual_auc(y_true, scores):
    """
    Rank-sum (Mann-Whitney U) AUC (Methods Section 2.8) -- numerically verified
    identical to sklearn's roc_auc_score (incl. tie handling) across 300
    randomized trials, max diff ~1e-16. Used instead of sklearn's version
    because repeated calls to sklearn.metrics functions reliably crashed the
    original development environment once called a few hundred to a few
    thousand times in the same process (the bootstrap below).
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=np.float64)
    n1 = int(np.sum(y_true == 1))
    n0 = len(y_true) - n1
    if n0 == 0 or n1 == 0:
        return 0.5
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sum_ranks_per_value = np.zeros(len(counts))
    np.add.at(sum_ranks_per_value, inv, ranks)
    ranks = (sum_ranks_per_value / counts)[inv]
    sum_ranks_pos = ranks[y_true == 1].sum()
    return float((sum_ranks_pos - n1 * (n1 + 1) / 2) / (n1 * n0))


def youden_threshold(y_true, scores) -> float:
    """Threshold maximizing sensitivity + specificity - 1 (predict positive
    when score >= threshold). Ties are broken toward the smallest threshold."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=np.float64)
    n1, n0 = (y_true == 1).sum(), (y_true == 0).sum()
    best_j, best_t = -np.inf, None
    for t in np.unique(scores):
        pred = scores >= t
        j = (pred & (y_true == 1)).sum() / n1 + (~pred & (y_true == 0)).sum() / n0 - 1
        if j > best_j:
            best_j, best_t = j, float(t)
    return best_t


def compute_metrics(y_true, scores, threshold):
    auc  = _manual_auc(y_true, scores)
    pred = (np.asarray(scores) >= threshold).astype(int)
    tn, fp, fn, tp = _confusion_counts(y_true, pred)
    acc  = (tp + tn) / (tp + tn + fp + fn)
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv  = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    f1   = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0   # harmonic mean of PPV and sensitivity
    return {
        "AUC": auc, "Acc": float(acc), "Sens": float(sens),
        "Spec": float(spec), "PPV": float(ppv), "NPV": float(npv), "F1": float(f1),
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
    }


def bootstrap_indices(y_true, n_boot=N_BOOT, seed=BOOT_SEED):
    """Subject-level resamples of the test partition, shared by every
    configuration. Resamples containing a single class are discarded."""
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true)
    n = len(y_true)
    idx = rng.randint(0, n, size=(n_boot, n))
    keep = np.array([len(np.unique(y_true[i])) == 2 for i in idx])
    return idx[keep]


def bootstrap_auc(y_true, scores, idx):
    """Rank-sum AUC (Methods 2.8, midranks for ties) for every resample in `idx`
    at once; equal to _manual_auc applied row by row."""
    Y = np.asarray(y_true)[idx]
    R = rankdata(np.asarray(scores, dtype=np.float64)[idx], method="average", axis=1)
    n1 = Y.sum(axis=1)
    n0 = Y.shape[1] - n1
    return ((R * Y).sum(axis=1) - n1 * (n1 + 1) / 2) / (n1 * n0)


def bootstrap_metrics(y_true, scores, threshold, idx):
    """(n_resamples, len(METRICS)) array of metrics over shared resamples,
    computed for all resamples at once (same definitions as compute_metrics)."""
    Y = np.asarray(y_true)[idx] == 1
    P = np.asarray(scores)[idx] >= threshold
    tp = (P & Y).sum(axis=1)
    tn = (~P & ~Y).sum(axis=1)
    fp = (P & ~Y).sum(axis=1)
    fn = (~P & Y).sum(axis=1)

    def ratio(a, b):
        return np.divide(a, b, out=np.zeros(len(a)), where=b > 0)

    return np.column_stack([
        bootstrap_auc(y_true, scores, idx),
        (tp + tn) / Y.shape[1],
        ratio(tp, tp + fn), ratio(tn, tn + fp),
        ratio(tp, tp + fp), ratio(tn, tn + fn),
        ratio(2 * tp, 2 * tp + fp + fn),
    ])


def percentile_ci(samples, ci=0.95):
    lo, hi = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    return np.percentile(samples, lo, axis=0), np.percentile(samples, hi, axis=0)


def holm(pvals):
    """Holm step-down adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


# ── classifiers on the shared PCA space (Methods Section 2.4) ────────────────

def _atomic_write(path: Path, data: bytes):
    """Write to a temp file, fsync, then rename: a crash leaves either the
    previous complete file or the new one, never a partial/zero-filled one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def save_csv(df: pd.DataFrame, path: Path):
    _atomic_write(path, df.to_csv(index=False).encode("utf-8"))


def cached_classical_scores(train_df, val_df, test_df, cache_dir: Path, seeds, store_dir, reuse=True,
                            mask=None, suffix=""):
    """classical_scores, stored in store_dir and reused when the same subjects,
    seeds and input cache are requested again (so a crashed evaluation
    resumes without refitting PCA/SVM/XGBoost). Pass reuse=False (--fresh)
    after changing the classifier code."""
    meta = cache_dir / "_transform.json"
    if not meta.exists():
        meta = cache_dir.parent / "_registration.json"
    key = [list(train_df["PTID"]), list(val_df["PTID"]), list(test_df["PTID"]),
           meta.read_text() if meta.exists() else "", list(seeds), N_PCA]
    if suffix:
        key.append(suffix)
    key_src = json.dumps(key)
    path = store_dir / f"classical_{hashlib.sha1(key_src.encode()).hexdigest()[:16]}.pkl"
    if reuse and path.exists():
        try:
            out = pickle.loads(path.read_bytes())
            print(f"Reusing stored classical-model scores ({path.name})")
            return out
        except Exception as e:
            print(f"WARNING: could not read {path.name} ({e}); recomputing")
    out = classical_scores(train_df, val_df, test_df, cache_dir, seeds, mask=mask, suffix=suffix)
    _atomic_write(path, pickle.dumps(out))
    return out


def classical_scores(train_df, val_df, test_df, cache_dir: Path, seeds, mask=None, suffix=""):
    """Returns ({config: {seed: (val_scores, test_scores)}}, pca_variance).
    `mask` selects voxels (registered volumes: the template brain mask);
    `suffix` is appended to every configuration name."""

    def load_flat(df):
        if mask is None:
            X = np.stack([np.load(cache_dir / f"{p}.npy").ravel() for p in df["PTID"]])
        else:
            X = np.stack([np.load(cache_dir / f"{p}.npy")[mask] for p in df["PTID"]])
        return X.astype(np.float64), df["label"].to_numpy()

    print("Loading flattened tensors ...")
    X_tr, y_tr = load_flat(train_df)
    X_va, _    = load_flat(val_df)
    X_te, _    = load_flat(test_df)

    print(f"Fitting IncrementalPCA(n_components={N_PCA}) ...")
    pca = IncrementalPCA(n_components=N_PCA, batch_size=64).fit(X_tr)
    Z_tr, Z_va, Z_te = (pca.transform(X).astype(np.float32) for X in (X_tr, X_va, X_te))
    var = float(pca.explained_variance_ratio_.sum())
    print(f"Variance explained: {var:.1%}")

    out = {}

    # Cosine Similarity (Methods 2.4, Eq. 10): the raw prototype margin is the
    # score; the validation-chosen threshold replaces min-max rescaling.
    mu_cn, mu_dem = Z_tr[y_tr == 0].mean(axis=0), Z_tr[y_tr == 1].mean(axis=0)

    def margin(Z):
        def cos(mu):
            return Z @ mu / (np.linalg.norm(Z, axis=1) * np.linalg.norm(mu) + 1e-10)
        return cos(mu_dem) - cos(mu_cn)

    out["Cosine Similarity"] = {None: (margin(Z_va), margin(Z_te))}

    # SVM (Methods 2.4): decision_function scores (no Platt scaling).
    print("SVM (RBF) -- grid search ...")
    gs = GridSearchCV(
        SVC(kernel="rbf", class_weight="balanced"),
        {"C": [0.1, 1, 10, 100], "gamma": ["scale", "auto"]},
        scoring="roc_auc", cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), n_jobs=-1,
    ).fit(Z_tr, y_tr)
    print(f"  best params: {gs.best_params_}  CV AUC: {gs.best_score_:.3f}")
    svm = gs.best_estimator_
    out["SVM (RBF)"] = {None: (svm.decision_function(Z_va), svm.decision_function(Z_te))}

    # XGBoost (Methods 2.4): seed-dependent through row/column subsampling.
    n_neg, n_pos = int((y_tr == 0).sum()), int((y_tr == 1).sum())
    out["XGBoost"] = {}
    for seed in seeds:
        print(f"XGBoost seed {seed} ...")
        model = xgb.XGBClassifier(
            n_estimators=500, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=n_neg / n_pos,
            early_stopping_rounds=30, eval_metric="auc",
            random_state=seed, verbosity=0,
        )
        model.fit(Z_tr, y_tr, eval_set=[(Z_va, val_df["label"].to_numpy())], verbose=False)
        out["XGBoost"][seed] = (model.predict_proba(Z_va)[:, 1], model.predict_proba(Z_te)[:, 1])

    if suffix:
        out = {f"{k}{suffix}": v for k, v in out.items()}
    return out, var


# ── registration-based reference (Methods 2.9) ───────────────────────────────

REG_INPUTS = [("syn", ", MNI SyN"), ("affine", ", MNI affine")]
ROI_CONFIG = "Regional volumes (MNI SyN atlas)"


def registered_classical_scores(train_df, val_df, test_df, reg_dir: Path, seeds, store_dir, reuse=True):
    """PCA-based classifiers on the template-masked voxels of the affine and
    SyN registered volumes (preprocessing/register.py)."""
    import nibabel as nib
    mask_path = reg_dir / "mni152_2009_2mm_mask.nii.gz"
    if not mask_path.exists():
        return {}
    mask = np.asanyarray(nib.load(str(mask_path)).dataobj) > 0.5
    out = {}
    for sub, suffix in REG_INPUTS:
        d = reg_dir / sub
        if not all((d / f"{p}.npy").exists() for df in (train_df, val_df, test_df) for p in df["PTID"]):
            print(f"Registered inputs in {d} incomplete -- skipped")
            continue
        print(f"\nClassical models on registered voxels ({sub}) ...")
        sc, var = cached_classical_scores(train_df, val_df, test_df, d, seeds, store_dir, reuse,
                                          mask=mask, suffix=suffix)
        print(f"  PCA variance explained ({sub}): {var:.1%}")
        out.update(sc)
    return out


def roi_scores(train_df, val_df, test_df, reg_dir: Path):
    """Regional-volume reference: Harvard-Oxford subcortical volumes warped
    into native space (inverse SyN), each divided by total brain volume, plus
    brain volume; standardized and fed to an L2 logistic regression whose C
    is chosen by 5-fold cross-validated AUC on the training partition."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    roi_dir = reg_dir / "roi"

    def feats(df):
        rows = []
        for p in df["PTID"]:
            f = roi_dir / f"{p}.json"
            if not f.exists():
                return None
            v = json.loads(f.read_text())["volumes_mm3"]
            bv = v["Brain volume"]
            rows.append([v[k] / bv for k in sorted(v) if k != "Brain volume"] + [bv])
        return np.array(rows)

    X_tr, X_va, X_te = feats(train_df), feats(val_df), feats(test_df)
    if X_tr is None or X_va is None or X_te is None:
        print("Regional volumes incomplete -- skipped")
        return {}
    y_tr = train_df["label"].to_numpy()
    gs = GridSearchCV(make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced")),
                      {"logisticregression__C": [0.01, 0.1, 1, 10, 100]}, scoring="roc_auc",
                      cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)).fit(X_tr, y_tr)
    print(f"Regional volumes: best C {gs.best_params_}, CV AUC {gs.best_score_:.3f}")
    m = gs.best_estimator_
    return {ROI_CONFIG: {None: (m.decision_function(X_va), m.decision_function(X_te))}}


# Pre-specified comparisons for the registration question (own Holm family):
# each pair is (registration-free, registration-based).
REG_PAIRS = [
    ("CNN3D (best epoch)", "CNN3D, MNI SyN (best epoch)"),
    ("CNN3D (best epoch)", "CNN3D, MNI affine (best epoch)"),
    ("SVM (RBF)", "SVM (RBF), MNI SyN"),
    ("SVM (RBF)", "SVM (RBF), MNI affine"),
    ("CNN3D (best epoch)", ROI_CONFIG),
]

# Second pre-specified family (own Holm): does registering first improve the
# spherical representation? Same transform and model, applied to the native
# brain vs. the SyN-registered brain (Methods 2.9).
REG_SPH_PAIRS = [
    ("CNN3D (best epoch)", "CNN3D, spherical of MNI SyN (best epoch)"),
    ("Spherical Transformer (best epoch)", "Spherical Transformer, spherical of MNI SyN (best epoch)"),
]

# Third family (own Holm), added after the two families above had been
# analysed, so it is reported as such (Methods 2.9): CNN3D on the unregistered
# brain in Cartesian coordinates (preprocessing/native_volume.py) separates
# the representation (spherical vs native volume) from registration
# (native volume vs SyN volume).
NATIVE_PAIRS = [
    ("CNN3D (best epoch)", "CNN3D, native volume (best epoch)"),
    ("CNN3D, native volume (best epoch)", "CNN3D, MNI SyN (best epoch)"),
]


def spacing_scores(train_df, val_df, test_df):
    """Confound control: score = training-partition Dementia rate among
    subjects with the same voxel-spacing triple (overall rate if unseen)."""
    def key(df):
        return df[["spacing_0", "spacing_1", "spacing_2"]].round(2).astype(str).agg("x".join, axis=1)
    rate = train_df.groupby(key(train_df))["label"].mean()
    prior = train_df["label"].mean()
    def score(df):
        return key(df).map(rate).fillna(prior).to_numpy(dtype=float)
    return {SPACING_CONTROL: {None: (score(val_df), score(test_df))}}


def deep_scores(runs_dir: Path, val_df, test_df):
    """Reads runs/<Model>/seed<k>/predictions.csv, aligned to val_df/test_df order."""
    out = {}
    for model_dir, display in DEEP_MODELS:
        for seed_dir in sorted((runs_dir / model_dir).glob("seed*")):
            pred_path = seed_dir / "predictions.csv"
            if not pred_path.exists():
                print(f"WARNING: {pred_path} missing -- run train.py first; skipped")
                continue
            seed = int(seed_dir.name[len("seed"):])
            pred = pd.read_csv(pred_path).set_index(["split", "PTID"])
            for variant, vlabel in VARIANTS:
                col = f"prob_{variant}"
                val = pred.loc["validation"].loc[val_df["PTID"], col].to_numpy()
                tst = pred.loc["test"].loc[test_df["PTID"], col].to_numpy()
                out.setdefault(f"{display} ({vlabel})", {})[seed] = (val, tst)
    return out


PROTOCOLS = ["slice1.2", "iso1.0"]


def cross_protocol(ref, cache_dir: Path, runs_dir: Path, seeds, store_dir: Path, reuse: bool = True):
    """Cross-protocol generalization (Methods 2.8). For each source protocol:
    classical models are refit on source-protocol training subjects; deep
    models come from runs/<Model>_train-<source>/ (train.py --train-protocol).
    Thresholds are chosen on source-protocol validation subjects. Each model
    is scored on (a) the source-protocol test subjects (in-protocol
    reference) and (b) every target-protocol subject, none of whom was used
    for training, validation or thresholding."""
    rows = []
    for src in PROTOCOLS:
        tgt = next(p for p in PROTOCOLS if p != src)
        s_train = ref[(ref["protocol"] == src) & (ref["split"] == "train")].reset_index(drop=True)
        s_val   = ref[(ref["protocol"] == src) & (ref["split"] == "validation")].reset_index(drop=True)
        # "cross-protocol test" scores the same subjects as the target
        # protocol's own "in-protocol test" (the other direction), so the two
        # are directly comparable; "cross-protocol" uses every target subject.
        sets = {"in-protocol test": ref[(ref["protocol"] == src) & (ref["split"] == "test")],
                "cross-protocol test": ref[(ref["protocol"] == tgt) & (ref["split"] == "test")],
                "cross-protocol": ref[ref["protocol"] == tgt]}
        eval_df = pd.concat(sets.values()).drop_duplicates("PTID").reset_index(drop=True)
        print(f"\nCross-protocol: train on {src} (n={len(s_train)}), test on {tgt} (n={len(sets['cross-protocol'])})")

        scores, _ = cached_classical_scores(s_train, s_val, eval_df, cache_dir, seeds, store_dir, reuse)
        for model_dir, display in DEEP_MODELS:
            for seed_dir in sorted((runs_dir / f"{model_dir}_train-{src}").glob("seed*")):
                pred_path = seed_dir / "predictions.csv"
                if not pred_path.exists():
                    continue
                pred = pd.read_csv(pred_path).set_index("PTID")
                seed = int(seed_dir.name[len("seed"):])
                for variant, vlabel in VARIANTS:
                    col = f"prob_{variant}"
                    scores.setdefault(f"{display} ({vlabel})", {})[seed] = (
                        pred.loc[s_val["PTID"], col].to_numpy(), pred.loc[eval_df["PTID"], col].to_numpy())

        y_val = s_val["label"].to_numpy()
        for set_name, sdf in sets.items():
            k = eval_df.index[eval_df["PTID"].isin(sdf["PTID"])].to_numpy()
            yk = eval_df.loc[k, "label"].to_numpy()
            idx = bootstrap_indices(yk, seed=BOOT_SEED + 2)
            for config, by_seed in scores.items():
                aucs, sens, spec, boots = [], [], [], []
                for s_val_scores, s_eval in by_seed.values():
                    sk = np.asarray(s_eval)[k]
                    m = compute_metrics(yk, sk, youden_threshold(y_val, s_val_scores))
                    aucs.append(m["AUC"]); sens.append(m["Sens"]); spec.append(m["Spec"])
                    boots.append(bootstrap_auc(yk, sk, idx))
                lo, hi = percentile_ci(np.mean(boots, axis=0))
                rows.append({"train_protocol": src, "test_protocol": src if set_name.startswith("in") else tgt,
                             "eval_set": set_name, "config": config, "n": len(k),
                             "n_dementia": int(yk.sum()), "n_seeds": len(aucs),
                             "AUC_mean": float(np.mean(aucs)),
                             "AUC_sd": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0,
                             "AUC_lo": float(lo), "AUC_hi": float(hi),
                             "Sens_mean": float(np.mean(sens)), "Spec_mean": float(np.mean(spec))})
    return pd.DataFrame(rows)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    repo_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=repo_root / "cohort" / "manifest.csv")
    parser.add_argument("--cache-dir", type=Path, default=repo_root / "spherical_cache",
                        help="Spherical-tensor cache directory (not included in this repo)")
    parser.add_argument("--runs-dir", type=Path, default=repo_root / "runs")
    parser.add_argument("--out-dir", type=Path, default=repo_root / "results")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS, help="XGBoost seeds")
    parser.add_argument("--fresh", action="store_true",
                        help="Recompute classical-model scores instead of reusing <out-dir>/_cache")
    parser.add_argument("--no-cross-protocol", action="store_true",
                        help="Skip the cross-protocol generalization analysis")
    parser.add_argument("--registered-dir", type=Path, default=repo_root / "registered_cache",
                        help="Outputs of preprocessing/register.py (registration-based reference)")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. data ────────────────────────────────────────────────────────────
    ref = pd.read_csv(args.manifest)
    ref = ref[ref["Diagnosis"].isin(CLASSES)].copy()
    ref["label"] = ref["Diagnosis"].map(LABEL_MAP)
    missing = [p for p in ref["PTID"] if not (args.cache_dir / f"{p}.npy").exists()]
    if missing:
        print(f"WARNING: {len(missing)} subjects have no cached tensor and are excluded: {missing}")
    ref = ref[~ref["PTID"].isin(missing)].reset_index(drop=True)

    train_df = ref[ref["split"] == "train"].reset_index(drop=True)
    val_df   = ref[ref["split"] == "validation"].reset_index(drop=True)
    test_df  = ref[ref["split"] == "test"].reset_index(drop=True)
    print(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")
    y_val, y_test = val_df["label"].to_numpy(), test_df["label"].to_numpy()

    # ── 2. scores from every configuration ────────────────────────────────
    store = args.out_dir / "_cache"
    scores, pca_var = cached_classical_scores(train_df, val_df, test_df, args.cache_dir, args.seeds,
                                              store, reuse=not args.fresh)
    scores.update(spacing_scores(train_df, val_df, test_df))
    scores.update(deep_scores(args.runs_dir, val_df, test_df))
    # registration-based reference (Methods 2.9), when its inputs exist
    scores.update(registered_classical_scores(train_df, val_df, test_df, args.registered_dir,
                                              args.seeds, store, reuse=not args.fresh))
    scores.update(roi_scores(train_df, val_df, test_df, args.registered_dir))

    def is_registration(c):
        # reference configurations, kept out of the primary pairwise family
        return "MNI" in c or c == ROI_CONFIG or "native volume" in c

    # ── 3. per-seed metrics, thresholds chosen on validation ──────────────
    idx = bootstrap_indices(y_test)
    print(f"Bootstrap: {len(idx)} valid resamples of {N_BOOT}")

    per_seed_rows, pred_rows, boot = [], [], {}
    for config, by_seed in scores.items():
        boot[config] = []
        for seed, (s_val, s_test) in by_seed.items():
            thr = youden_threshold(y_val, s_val)
            m = compute_metrics(y_test, s_test, thr)
            b = bootstrap_metrics(y_test, s_test, thr, idx)
            boot[config].append(b)
            lo, hi = percentile_ci(b)
            row = {"config": config, "seed": seed, "threshold": thr,
                   "val_AUC": _manual_auc(y_val, s_val)}
            for j, k in enumerate(METRICS):
                row[k], row[f"{k}_lo"], row[f"{k}_hi"] = m[k], lo[j], hi[j]
            row.update({k: m[k] for k in ("TN", "FP", "FN", "TP")})
            per_seed_rows.append(row)
            for split, df, s in (("validation", val_df, s_val), ("test", test_df, s_test)):
                pred_rows.append(pd.DataFrame({"config": config, "seed": seed, "PTID": df["PTID"],
                                               "split": split, "label": df["label"], "score": s,
                                               "threshold": thr}))
            print(f"  {config:40s} seed={seed!s:4s} AUC={m['AUC']:.3f} "
                  f"[{lo[0]:.3f}, {hi[0]:.3f}]  thr={thr:.4g}  Sens={m['Sens']:.3f} Spec={m['Spec']:.3f}")

    per_seed = pd.DataFrame(per_seed_rows)

    # ── 4. per-configuration summary ──────────────────────────────────────
    seed_mean_boot = {c: np.mean(np.stack(bs), axis=0) for c, bs in boot.items()}
    summary_rows = []
    for config in scores:
        g = per_seed[per_seed["config"] == config]
        lo, hi = percentile_ci(seed_mean_boot[config])
        row = {"config": config, "n_seeds": len(g)}
        for j, k in enumerate(METRICS):
            row[f"{k}_mean"] = g[k].mean()
            row[f"{k}_sd"]   = g[k].std(ddof=1) if len(g) > 1 else 0.0
            row[f"{k}_lo"], row[f"{k}_hi"] = lo[j], hi[j]
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)

    # ── 5. pairwise paired-bootstrap ΔAUC (seed-mean), Holm-adjusted ──────
    # Primary family: the pre-declared registration-free configurations only
    # (the registration-based reference has its own family below).
    point = summary.set_index("config")["AUC_mean"]

    def paired(a, b):
        d = seed_mean_boot[a][:, 0] - seed_mean_boot[b][:, 0]
        lo, hi = percentile_ci(d)
        p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
        return {"config_a": a, "config_b": b, "delta_AUC": point[a] - point[b],
                "lo": float(lo), "hi": float(hi), "p_boot": max(p, 1 / len(d))}

    configs = [c for c in scores if c != SPACING_CONTROL and not is_registration(c)]
    pairwise = pd.DataFrame([paired(a, b) for a, b in combinations(configs, 2)])
    pairwise["p_holm"] = holm(pairwise["p_boot"])

    # Registration family: pre-specified (registration-free, registration-based)
    # pairs, delta = free - registered, Holm-adjusted within the family.
    reg_rows = [paired(a, b) for a, b in REG_PAIRS if a in scores and b in scores]
    regcomp = pd.DataFrame(reg_rows)
    if len(regcomp):
        regcomp["p_holm"] = holm(regcomp["p_boot"])

    # Second family: spherical of the native brain vs spherical of the
    # SyN-registered brain (delta = native - registered), own Holm.
    regsph = pd.DataFrame([paired(a, b) for a, b in REG_SPH_PAIRS if a in scores and b in scores])
    if len(regsph):
        regsph["p_holm"] = holm(regsph["p_boot"])

    # Third family: spherical vs native volume vs SyN volume (CNN3D), own Holm.
    native = pd.DataFrame([paired(a, b) for a, b in NATIVE_PAIRS if a in scores and b in scores])
    if len(native):
        native["p_holm"] = holm(native["p_boot"])

    # ── 6. protocol-stratified test AUC ───────────────────────────────────
    sub_rows = []
    protocols = test_df["protocol"].to_numpy()
    for proto in sorted(set(protocols)):
        k = np.flatnonzero(protocols == proto)
        yk = y_test[k]
        sub_idx = bootstrap_indices(yk, seed=BOOT_SEED + 1)
        for config, by_seed in scores.items():
            aucs, boots = [], []
            for _, s_test in by_seed.values():
                sk = np.asarray(s_test)[k]
                aucs.append(_manual_auc(yk, sk))
                boots.append(bootstrap_auc(yk, sk, sub_idx))
            lo, hi = percentile_ci(np.mean(boots, axis=0))
            sub_rows.append({"config": config, "protocol": proto, "n": len(k),
                             "n_dementia": int(yk.sum()), "n_seeds": len(aucs),
                             "AUC_mean": float(np.mean(aucs)),
                             "AUC_sd": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0,
                             "AUC_lo": float(lo), "AUC_hi": float(hi)})
    subgroup = pd.DataFrame(sub_rows)

    # ── 7. cross-protocol generalization ──────────────────────────────────
    cross = None if args.no_cross_protocol else cross_protocol(ref, args.cache_dir, args.runs_dir, args.seeds,
                                                                  store, reuse=not args.fresh)

    # ── 8. write out ───────────────────────────────────────────────────────
    save_csv(pd.concat(pred_rows), args.out_dir / "predictions.csv")
    save_csv(per_seed, args.out_dir / "metrics_per_seed.csv")
    save_csv(summary, args.out_dir / "summary.csv")
    save_csv(pairwise, args.out_dir / "pairwise_auc.csv")
    save_csv(subgroup, args.out_dir / "subgroup_auc.csv")
    if len(regcomp):
        save_csv(regcomp, args.out_dir / "registration_comparison.csv")
    if len(regsph):
        save_csv(regsph, args.out_dir / "registration_spherical_comparison.csv")
    if len(native):
        save_csv(native, args.out_dir / "native_volume_comparison.csv")
    if cross is not None:
        save_csv(cross, args.out_dir / "cross_protocol.csv")
    _atomic_write(args.out_dir / "summary.json", json.dumps({
        "n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df),
        "n_test_dementia": int(y_test.sum()), "classes": CLASSES,
        "pca_components": N_PCA, "pca_variance_explained": pca_var,
        "threshold_rule": "Youden J on validation partition",
        "n_boot": N_BOOT, "n_boot_valid": int(len(idx)), "boot_seed": BOOT_SEED,
        "xgboost_seeds": args.seeds, "excluded_no_cache": missing,
        "test_protocol_counts": {f"{p}|{d}": int(n) for (p, d), n in
                                 test_df.groupby(["protocol", "Diagnosis"]).size().items()},
    }, indent=2).encode("utf-8"))

    print("\nSummary (mean across seeds, 95% bootstrap CI of the seed-mean):")
    for r in summary_rows:
        print(f"  {r['config']:40s} AUC={r['AUC_mean']:.3f} +/-{r['AUC_sd']:.3f} "
              f"[{r['AUC_lo']:.3f}, {r['AUC_hi']:.3f}]  n_seeds={r['n_seeds']}")
    print("\nDone. Exported to:", args.out_dir)


if __name__ == "__main__":
    main()
