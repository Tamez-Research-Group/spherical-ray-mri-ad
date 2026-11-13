"""
train.py — training for CNN3D and the Spherical Transformer
(Methods, Section 2.7).

Usage
-----
    # Step 1 (one-time): build the spherical-tensor cache from the volumes
    # listed in cohort/manifest.csv (see cohort/build_manifest.py)
    python train.py --cache-only

    # Step 2: train both models for seeds 0-4
    python train.py

    # One model, chosen seeds, from scratch
    python train.py --model cnn --seeds 0 1 --reset

Each (model, seed) run writes to runs/<Model>/seed<k>/:
    best.pt          state_dict of the epoch with the best validation AUC
    avg.pt           state_dict averaged over the top-K validation-AUC epochs
                     (snapshot / checkpoint averaging), BatchNorm re-estimated
    latest.pt        full resume state (deleted with --reset)
    training.csv     one row per epoch
    predictions.csv  validation + test probabilities for both best and avg
    summary.json     best epoch, top-K epochs, validation AUC of best and avg

Test-set metrics are deliberately not computed here; evaluate.py derives
all metrics, validation-chosen thresholds and confidence intervals from
predictions.csv.

Adapted from the research repository's deep_learning_experiment.py.
Changes from the first version of the pipeline: class imbalance is
handled by the weighted sampler only (the extra pos_weight loss reweighting
was removed), and the sampler balances class x acquisition protocol so
that protocol is not a label shortcut; the spherical transform works in
millimetres; runs are seeded; training-time augmentation was added; and
top-K checkpoint averaging was added alongside best-epoch selection.
"""

import os
import csv
import json
import random
import shutil
import argparse
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.swa_utils import update_bn

from spherical_transform import build_cache
from models import CNN3D, SphericalTransformer

CLASSES    = ["CN", "Dementia"]
LABEL_MAP  = {"CN": 0, "Dementia": 1}

# Training hyperparameters (Methods Section 2.7)
LR           = 1e-4
WEIGHT_DECAY = 1e-2
MAX_EPOCHS   = 50
PATIENCE     = 10
GRAD_CLIP    = 1.0
TOP_K        = 5    # epochs averaged into avg.pt

BATCH_CNN = 8
BATCH_TRF = 4   # smaller due to attention memory

SEEDS = [0, 1, 2, 3, 4]

MODELS = {
    "cnn":         ("CNN3D",       CNN3D,                BATCH_CNN),
    "transformer": ("Transformer", SphericalTransformer, BATCH_TRF),
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int):
    """Seed every RNG the training run touches. cuDNN is put in
    deterministic mode; a few CUDA kernels (e.g. max-pool backward) have no
    deterministic implementation, so reruns are close but not guaranteed
    bit-identical on GPU."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


# ── augmentation ──────────────────────────────────────────────────────────────

class SphericalAugment:
    """Training-time perturbation of a (phi, theta, r) spherical tensor.

    - theta roll: circular shift of up to `max_theta` azimuth bins (5 deg
      each), i.e. a small rotation about the inferior-superior axis; exact,
      because theta is periodic.
    - phi shift: shift of up to `max_phi` elevation bins with edge
      replication (phi is not periodic).
    - additive Gaussian noise of std `noise_std`.

    No left-right mirroring: it would swap hemispheres.
    """

    def __init__(self, max_theta: int = 2, max_phi: int = 1, noise_std: float = 0.05,
                 p_shift: float = 0.5, p_noise: float = 0.5):
        self.max_theta, self.max_phi = max_theta, max_phi
        self.noise_std, self.p_shift, self.p_noise = noise_std, p_shift, p_noise

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(()) < self.p_shift:
            dt = int(torch.randint(-self.max_theta, self.max_theta + 1, ()))
            if dt:
                x = torch.roll(x, shifts=dt, dims=1)
            dp = int(torch.randint(-self.max_phi, self.max_phi + 1, ()))
            if dp > 0:
                x = torch.cat([x[:1].expand(dp, -1, -1), x[:-dp]], dim=0)
            elif dp < 0:
                x = torch.cat([x[-dp:], x[-1:].expand(-dp, -1, -1)], dim=0)
        if torch.rand(()) < self.p_noise:
            x = x + torch.randn_like(x) * self.noise_std
        return x


class VolumeAugment:
    """Training-time perturbation of a Cartesian 2 mm volume, MNI-registered
    or native (Methods 2.9), the counterpart of SphericalAugment: with
    probability p_shift a random
    translation of up to `max_shift` voxels (2 mm each) along every axis, and
    with probability p_noise additive Gaussian noise. The volume is zero
    outside the brain, so the wrap-around of torch.roll only moves background.
    No left-right mirroring."""

    def __init__(self, max_shift: int = 2, noise_std: float = 0.05,
                 p_shift: float = 0.5, p_noise: float = 0.5):
        self.max_shift, self.noise_std = max_shift, noise_std
        self.p_shift, self.p_noise = p_shift, p_noise

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(()) < self.p_shift:
            shifts = tuple(int(s) for s in torch.randint(-self.max_shift, self.max_shift + 1, (3,)))
            if any(shifts):
                x = torch.roll(x, shifts=shifts, dims=(0, 1, 2))
        if torch.rand(()) < self.p_noise:
            x = x + torch.randn_like(x) * self.noise_std
        return x


# input name -> (sub-directory of --registered-dir, kind). "volume" inputs are
# Cartesian 2 mm volumes (CNN3D only, zero padding, VolumeAugment);
# "spherical" inputs are (36, 72, 48) spherical tensors (both models).
# mni-syn-spherical = the spherical transform applied to the SyN-registered
# brains (preprocessing/register.py --spherical; Methods 2.9).
# native-volume = the unregistered brain on a Cartesian grid centred on its
# centre of mass (preprocessing/native_volume.py, read from --native-dir).
INPUTS = {"spherical": (None, "spherical"), "mni-affine": ("affine", "volume"),
          "mni-syn": ("syn", "volume"), "mni-syn-spherical": ("spherical_syn", "spherical"),
          "native-volume": (None, "volume")}


# ── PyTorch Dataset ───────────────────────────────────────────────────────────

class SphericalDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cache_dir: Path, augment=None):
        self.augment = augment
        self.rows = [
            (row.PTID, cache_dir / f"{row.PTID}.npy", int(row.label), row.protocol)
            for row in df.itertuples()
            if (cache_dir / f"{row.PTID}.npy").exists()
        ]
        self.ptids     = [r[0] for r in self.rows]
        self.labels    = np.array([r[2] for r in self.rows], dtype=int)
        self.protocols = np.array([r[3] for r in self.rows])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        _, path, label, _ = self.rows[idx]
        x = torch.from_numpy(np.load(path))          # (36, 72, 48)
        if self.augment is not None:
            x = self.augment(x)
        y = torch.tensor(label, dtype=torch.float32)
        return x, y


def sampling_weights(labels: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Inverse cell-frequency weights over (class, group) cells (Methods 2.7).
    Every cell is drawn equally often, so classes are balanced within each
    acquisition protocol and protocol carries no information about the
    label in the training batches. With a single group this reduces to
    inverse class-frequency weighting."""
    cells = np.char.add(labels.astype(str), np.char.add("|", groups.astype(str)))
    _, inv, counts = np.unique(cells, return_inverse=True, return_counts=True)
    return 1.0 / counts[inv]


def make_loader(ds: SphericalDataset, batch_size: int, training: bool, seed: int = 0,
                balance: str = "class_protocol") -> DataLoader:
    pin = DEVICE.type == "cuda"
    if training:
        groups = ds.protocols if balance == "class_protocol" else np.zeros(len(ds), dtype=int)
        w = sampling_weights(ds.labels, groups)
        gen = torch.Generator().manual_seed(seed)
        sampler = WeightedRandomSampler(w, len(w), replacement=True, generator=gen)
        return DataLoader(ds, batch_size=batch_size, sampler=sampler,
                          num_workers=0, pin_memory=pin)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=0, pin_memory=pin)


# ── training logger ───────────────────────────────────────────────────────────

class TrainingLogger:
    """Appends one CSV row per epoch to <run_dir>/training.csv."""

    FIELDS = ["epoch", "train_loss", "val_AUC", "best_AUC", "patience", "lr", "timestamp"]

    def __init__(self, path: Path):
        self.path = path
        self._write_header = not self.path.exists()

    def log(self, epoch: int, loss: float, val_auc: float, best_auc: float,
            patience_ctr: int, lr: float):
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDS)
            if self._write_header:
                writer.writeheader()
                self._write_header = False
            writer.writerow({
                "epoch":      epoch,
                "train_loss": f"{loss:.6f}",
                "val_AUC":    f"{val_auc:.4f}",
                "best_AUC":   f"{best_auc:.4f}",
                "patience":   patience_ctr,
                "lr":         f"{lr:.2e}",
                "timestamp":  datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })
            f.flush()
            os.fsync(f.fileno())


# ── crash-safe file writes ────────────────────────────────────────────────────
# A system crash can leave a just-written file at full size but zero-filled
# (data still in the OS write cache). Every output is therefore written to a
# temporary file, forced to disk, and only then renamed over the target, so
# a crash leaves either the previous complete file or the new one.

def _atomic_write(path: Path, write_fn, mode: str = "wb"):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, mode) as f:
        write_fn(f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def save_torch(obj, path: Path):
    _atomic_write(path, lambda f: torch.save(obj, f))


def save_text(text: str, path: Path):
    _atomic_write(path, lambda f: f.write(text), mode="w")


def load_checkpoint(path: Path, name: str):
    """Returns the resume dict, or None if there is none or it is unreadable.
    An unreadable checkpoint is renamed to *.corrupt so the run restarts."""
    if not path.exists():
        return None
    try:
        # map to CPU: RNG states must stay CPU ByteTensors; model and
        # optimizer state are copied to DEVICE by load_state_dict.
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        bad = path.with_name(path.name + ".corrupt")
        os.replace(path, bad)
        print(f"[{name}] WARNING: {path.name} is unreadable ({type(e).__name__}: {e}); "
              f"moved to {bad.name}, restarting this run from epoch 1")
        return None


# ── inference ─────────────────────────────────────────────────────────────────

def predict(model: nn.Module, loader: DataLoader) -> np.ndarray:
    model.eval()
    logits = []
    with torch.no_grad():
        for x, _ in loader:
            logits.append(model(x.to(DEVICE)).cpu())
    return torch.sigmoid(torch.cat(logits)).numpy()


def val_auc(model: nn.Module, loader: DataLoader) -> float:
    return float(roc_auc_score(loader.dataset.labels, predict(model, loader)))


def _clone_state(model: nn.Module) -> dict:
    # A deep copy: model.state_dict() tensors alias the live parameters.
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def average_states(states: list) -> dict:
    """Element-wise mean of floating-point tensors; integer buffers (e.g.
    BatchNorm's num_batches_tracked) are taken from the first state."""
    out = {}
    for k, v in states[0].items():
        if v.is_floating_point():
            out[k] = torch.stack([s[k] for s in states]).mean(dim=0)
        else:
            out[k] = v.clone()
    return out


# ── training ──────────────────────────────────────────────────────────────────

def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                bn_loader: DataLoader, run_dir: Path, name: str, top_k: int = TOP_K,
                max_epochs: int = MAX_EPOCHS) -> dict:
    """Train with early stopping on validation AUC. Returns a summary dict;
    writes best.pt and avg.pt to run_dir."""
    ckpt_path = run_dir / "latest.pt"

    criterion = nn.BCEWithLogitsLoss()   # imbalance handled by the sampler only (Methods 2.7)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=5, factor=0.5)

    best_auc, best_epoch, patience_ctr, start_epoch = 0.0, 0, 0, 1
    best_state, top = None, []          # top: [(val_auc, epoch, state)], best first
    log_path = run_dir / "training.csv"
    had_checkpoint = ckpt_path.exists()
    saved = load_checkpoint(ckpt_path, name)
    if saved is None and had_checkpoint and log_path.exists():
        # restarting after a corrupt checkpoint: keep the old log aside
        os.replace(log_path, log_path.with_name(log_path.name + ".corrupt"))
    logger = TrainingLogger(log_path)

    if saved is not None:
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        scheduler.load_state_dict(saved["scheduler_state"])
        torch.set_rng_state(saved["torch_rng"])
        if torch.cuda.is_available() and saved.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        train_loader.sampler.generator.set_state(saved["sampler_rng"])
        best_auc, best_epoch = saved["best_auc"], saved["best_epoch"]
        best_state, top = saved["best_state"], saved["top"]
        patience_ctr = saved["patience_ctr"]
        start_epoch  = saved["epoch"] + 1
        if saved["finished"]:
            print(f"[{name}] already finished (epoch {saved['epoch']}), skipping training")
            start_epoch = max_epochs + 1
        else:
            print(f"[{name}] resuming at epoch {start_epoch}  best_AUC={best_auc:.3f}")

    epoch = start_epoch - 1
    for epoch in range(start_epoch, max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            epoch_loss += loss.item()

        auc = val_auc(model, val_loader)
        scheduler.step(auc)
        lr = optimizer.param_groups[0]["lr"]

        state = None
        if auc > best_auc:
            best_auc, best_epoch, patience_ctr = auc, epoch, 0
            state = best_state = _clone_state(model)
            tag = " [best]"
        else:
            patience_ctr += 1
            tag = ""
        if len(top) < top_k or auc > top[-1][0]:
            top.append((auc, epoch, state if state is not None else _clone_state(model)))
            top = sorted(top, key=lambda t: -t[0])[:top_k]

        finished = patience_ctr >= PATIENCE or epoch == max_epochs
        save_torch({
            "epoch":           epoch,
            "finished":        finished,
            "model_state":     _clone_state(model),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "torch_rng":       torch.get_rng_state(),
            "cuda_rng":        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "sampler_rng":     train_loader.sampler.generator.get_state(),
            "best_auc":        best_auc,
            "best_epoch":      best_epoch,
            "best_state":      best_state,
            "top":             top,
            "patience_ctr":    patience_ctr,
        }, ckpt_path)

        avg_loss = epoch_loss / len(train_loader)
        logger.log(epoch, avg_loss, auc, best_auc, patience_ctr, lr)
        print(f"[{name}] {epoch:03d}  loss={avg_loss:.4f}  val_AUC={auc:.3f}  best={best_auc:.3f}"
              f"  lr={lr:.1e}  pat={patience_ctr}/{PATIENCE}{tag}")

        if patience_ctr >= PATIENCE:
            print(f"[{name}] early stop at epoch {epoch}")
            break

    save_torch(best_state, run_dir / "best.pt")

    # Top-K checkpoint averaging. Averaged weights invalidate BatchNorm
    # running statistics, so they are re-estimated with one pass over the
    # (un-augmented) training set; LayerNorm-only models are unaffected.
    model.load_state_dict(average_states([s for _, _, s in top]))
    update_bn(bn_loader, model, device=DEVICE)
    avg_state = _clone_state(model)
    save_torch(avg_state, run_dir / "avg.pt")
    avg_auc = val_auc(model, val_loader)

    summary = {
        "best_epoch": best_epoch, "best_val_AUC": best_auc,
        "top_k_epochs": [e for _, e, _ in top], "top_k_val_AUC": [a for a, _, _ in top],
        "avg_val_AUC": avg_auc, "last_epoch": epoch,
    }
    print(f"[{name}] best epoch {best_epoch} (val AUC {best_auc:.3f}); "
          f"top-{len(top)} average val AUC {avg_auc:.3f}")
    return summary


def run_one(key: str, seed: int, train_df, val_df, all_df, args):
    name, cls, batch = MODELS[key]
    registered = INPUTS[args.input][1] == "volume"          # Cartesian volume input (MNI or native)
    run_name = name if args.input == "spherical" else f"{name}_{args.input}"
    if args.train_protocol is not None:
        run_name += f"_train-{args.train_protocol}"
    run_dir = args.runs_dir / run_name / f"seed{seed}"
    if args.reset and run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # A complete run (finished checkpoint + summary + predictions) is skipped
    # outright, so relaunching after a crash does not redo averaging/inference.
    if (run_dir / "predictions.csv").exists() and (run_dir / "summary.json").exists():
        try:
            finished = torch.load(run_dir / "latest.pt", map_location="cpu", weights_only=False)["finished"]
        except Exception:
            finished = False
        if finished:
            print(f"[{run_name}/s{seed}] complete, skipping")
            return

    set_seed(seed)
    augment = None if args.no_augment else (VolumeAugment() if registered else SphericalAugment())
    train_ds = SphericalDataset(train_df, args.data_dir, augment=augment)
    clean_train_ds = SphericalDataset(train_df, args.data_dir)
    val_ds   = SphericalDataset(val_df, args.data_dir)
    all_ds   = SphericalDataset(all_df, args.data_dir)

    print("\n" + "=" * 60)
    print(f"  {run_name}  seed={seed}  augment={augment is not None}  "
          f"train/val={len(train_ds)}/{len(val_ds)}")
    print("=" * 60)

    model = (cls(circular_theta=False) if registered else cls()).to(DEVICE)
    summary = train_model(
        model,
        make_loader(train_ds, batch, training=True, seed=seed, balance=args.balance),
        make_loader(val_ds, batch, training=False),
        make_loader(clean_train_ds, batch, training=False),
        run_dir, f"{run_name}/s{seed}", top_k=args.top_k, max_epochs=args.max_epochs,
    )
    summary.update({"model": name, "seed": seed, "augment": augment is not None,
                    "balance": args.balance, "geometry": "voxel" if args.voxel_geometry else "mm",
                    "harmonize": not (args.no_harmonize or args.voxel_geometry),
                    "train_protocol": args.train_protocol or "all", "input": args.input,
                    "n_train": len(train_ds), "n_val": len(val_ds)})
    save_text(json.dumps(summary, indent=2), run_dir / "summary.json")

    # Probabilities for every subject (all partitions, both protocols) and
    # both weight variants; evaluate.py selects the rows each analysis needs.
    loader = make_loader(all_ds, batch, training=False)
    probs = {}
    for variant in ("best", "avg"):
        model.load_state_dict(torch.load(run_dir / f"{variant}.pt", map_location=DEVICE))
        probs[variant] = predict(model, loader)
    split_of = all_df.set_index("PTID")["split"]
    preds = pd.DataFrame({"PTID": all_ds.ptids, "split": split_of.loc[all_ds.ptids].to_numpy(),
                          "protocol": all_ds.protocols, "label": all_ds.labels,
                          "prob_best": probs["best"], "prob_avg": probs["avg"]})
    save_text(preds.to_csv(index=False, lineterminator="\n"), run_dir / "predictions.csv")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    repo_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=repo_root / "cohort" / "manifest.csv",
                        help="PTID, Diagnosis, split, brain_path (cohort/build_manifest.py)")
    parser.add_argument("--cache-dir", type=Path, default=repo_root / "spherical_cache",
                        help="Spherical-tensor cache directory (not included in this repo -- derived from patient data)")
    parser.add_argument("--runs-dir", type=Path, default=repo_root / "runs")
    parser.add_argument("--cache-only",  action="store_true", help="Build tensor cache then exit")
    parser.add_argument("--force-cache", action="store_true", help="Recompute cached tensors even if they exist")
    parser.add_argument("--workers", type=int, default=8, help="Processes for building the cache")
    parser.add_argument("--voxel-geometry", action="store_true",
                        help="Build the cache with the original voxel-index geometry instead of millimetres")
    parser.add_argument("--no-harmonize", action="store_true",
                        help="Build the cache without N4 + 1.2 mm resampling (preprocessing/harmonize.py)")
    parser.add_argument("--balance", choices=["class_protocol", "class"], default="class_protocol",
                        help="Sampler cells: class x acquisition protocol (default) or class only")
    parser.add_argument("--model", choices=["cnn", "transformer", "both"], default="both")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--no-augment", action="store_true", help="Disable training-time augmentation")
    parser.add_argument("--train-protocol", choices=["slice1.2", "iso1.0"], default=None,
                        help="Cross-protocol analysis: train and validate on this protocol only "
                             "(run directory <Model>_train-<protocol>); evaluate.py tests on the other")
    parser.add_argument("--reset", action="store_true",
                        help="Delete each selected run directory (checkpoints, log, predictions) first")
    parser.add_argument("--input", choices=list(INPUTS), default="spherical",
                        help="spherical tensors (default), MNI-registered volumes from "
                             "preprocessing/register.py (registration-based reference; CNN3D only), or "
                             "unregistered Cartesian volumes from preprocessing/native_volume.py (CNN3D only)")
    parser.add_argument("--registered-dir", type=Path, default=repo_root / "registered_cache")
    parser.add_argument("--native-dir", type=Path, default=repo_root / "native_cache",
                        help="Outputs of preprocessing/native_volume.py (--input native-volume)")
    parser.add_argument("--allow-missing", action="store_true",
                        help="Train on the subjects available even if some inputs are missing (testing only)")
    args = parser.parse_args()

    print(f"Device : {DEVICE}")
    print(f"PyTorch: {torch.__version__}")

    ref = pd.read_csv(args.manifest)
    ref = ref[ref["Diagnosis"].isin(CLASSES)].copy()
    ref["label"] = ref["Diagnosis"].map(LABEL_MAP)

    if args.input == "spherical":
        build_cache(ref, args.cache_dir, force=args.force_cache, workers=args.workers,
                    physical=not args.voxel_geometry,
                    harmonize=not (args.no_harmonize or args.voxel_geometry))
        args.data_dir = args.cache_dir
    else:
        args.data_dir = (args.native_dir if args.input == "native-volume"
                         else args.registered_dir / INPUTS[args.input][0])
        if INPUTS[args.input][1] == "volume":
            if args.model == "transformer":
                raise SystemExit("The Spherical Transformer needs spherical tensors; use --model cnn with MNI volumes")
            args.model = "cnn"
    if args.cache_only:
        print("Cache built. Exiting.")
        return

    missing = [p for p in ref["PTID"] if not (args.data_dir / f"{p}.npy").exists()]
    if missing:
        print(f"WARNING: {len(missing)} manifest subjects have no input in {args.data_dir}")
        if args.input != "spherical" and not args.allow_missing:
            raise SystemExit(f"Inputs are incomplete in {args.data_dir}; finish preprocessing/"
                             f"{'native_volume' if args.input == 'native-volume' else 'register'}.py first "
                             "(or pass --allow-missing for a smoke test)")
    ref = ref[~ref["PTID"].isin(missing)]

    source = ref if args.train_protocol is None else ref[ref["protocol"] == args.train_protocol]
    train_df = source[source["split"] == "train"]
    val_df   = source[source["split"] == "validation"]

    keys = ["cnn", "transformer"] if args.model == "both" else [args.model]
    for key in keys:
        for seed in args.seeds:
            run_one(key, seed, train_df, val_df, ref, args)


if __name__ == "__main__":
    main()
