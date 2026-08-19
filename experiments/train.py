from __future__ import annotations
"""
train.py -- main training entrypoint

Reads a config from experiments/configs/, trains any model (baseline or ours) on the
sepsis-onset cohort at a specified lead time, saves results to results/.

TODO:
    [ ] Argparse: --config path, --seed
    [ ] Load cohort + labels + chosen modality combo
    [ ] Instantiate model from config (baseline or models/ours/backbone.py + sdca/sarl)
    [ ] Standard train loop with early stopping
    [ ] Save metrics to results/<run_name>.json (never overwrite -- append run metadata)
"""

# TODO: implementation goes here
"""
train.py -- main training entrypoint.

Reads a YAML config from experiments/configs/, builds the dataset/dataloaders (dataset.py),
instantiates a model, trains with early stopping on val AUPRC, evaluates through
evaluate.py under BOTH protocols (PROJECT_CONTEXT.md rule #7), and saves one timestamped
JSON per run under results/ (never overwrite, per PROJECT_CONTEXT.md).

*** WHAT'S REAL vs PLACEHOLDER RIGHT NOW ***
The data loop, training loop, checkpointing, early stopping, and eval wiring are real
and runnable today -- against synthetic data (make_synthetic_data.py) while CXR is still
downloading, and against real TS+notes data as soon as ehr_extraction.py /
notes_extraction.py output lands, with modalities: [ts, notes] in the config.

The MODEL is a placeholder (`SanityBaseline` below): mean-pool TS values, mean-pool a
bag-of-words note embedding, concat, small MLP. It exists ONLY to validate the pipeline
end-to-end (does loss go down, do metrics compute, does checkpointing work) -- it is
NOT one of the paper baselines and should never appear in Table 2/3. Swap in real models
by adding an entry to MODEL_REGISTRY once models/baselines/*.py and
models/ours/backbone.py are implemented; nothing else in this file needs to change.

Usage:
    python train.py --config configs/sanity_check.yaml
    python train.py --config configs/sanity_check.yaml --smoke_test   # uses synthetic data
"""
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="utde")

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"
# models/baselines/*.py and models/ours/*.py import as `models.baselines.xyz` /
# `models.ours.xyz` (repo-root-relative), so the repo root needs to be on sys.path.
# Running `python experiments/train.py` from the repo root already puts experiments/ on
# sys.path (for `dataset`/`evaluate`) but NOT the repo root itself -- add it explicitly
# so both import styles work regardless of your cwd.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset import SepsisDataset, collate_sepsis_batch, VARIABLE_VOCAB
from evaluate import evaluate_both_protocols, compute_metrics
from models.baselines.utde import UTDEBaseline


# --------------------------------------------------------------------------------------
# Placeholder model -- see module docstring. Delete/replace once real models exist.
# --------------------------------------------------------------------------------------
class SanityBaseline(nn.Module):
    """Mean-pooled TS features -> MLP. Ignores notes/CXR content, just checks whether a
    modality was present at all (a crude missingness signal), so the pipeline exercises
    every batch key without needing a real text/image encoder yet."""

    def __init__(self, n_ts_vars: int = len(VARIABLE_VOCAB), hidden: int = 64):
        super().__init__()
        self.n_ts_vars = n_ts_vars
        in_dim = n_ts_vars + 2  # + notes-present flag + cxr-present flag
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, batch: dict) -> torch.Tensor:
        ts = batch["ts"]
        B = ts["value"].shape[0]
        device = ts["value"].device
        var_idx = ts["var_idx"].clamp(min=0)  # padding uses -1, clamp to a dummy valid bin
        valid = ts["mask"]

        # Vectorized replacement for the per-event Python loop this used to have --
        # identical result (scatter_add_ == "sum values into their variable's bucket",
        # same as the loop did), ~300x faster. Zero out padded entries first so their
        # clamped var_idx=0 doesn't add spurious mass into channel 0; valid.float()
        # being 0 at pad positions means counts isn't corrupted either.
        masked_value = ts["value"] * valid.float()
        pooled = torch.zeros(B, self.n_ts_vars, device=device)
        counts = torch.zeros(B, self.n_ts_vars, device=device)
        pooled.scatter_add_(1, var_idx, masked_value)
        counts.scatter_add_(1, var_idx, valid.float())
        pooled = pooled / counts.clamp(min=1)

        notes_present = batch["notes"]["mask"].any(dim=1, keepdim=True).float()
        cxr_present = batch["cxr"]["mask"].any(dim=1, keepdim=True).float()
        x = torch.cat([pooled, notes_present, cxr_present], dim=1)
        return self.mlp(x).squeeze(-1)  # logits


def _build_sanity_baseline(config: dict, device: str):
    return SanityBaseline()


def _build_mult_cross_ts(config: dict, device: str):
    return MultCrossTSBaseline(config, device)


# Every entry is a (config, device) -> nn.Module builder, so each model can take
# whatever constructor args it actually needs (a bare nn.Module vs. one that wants the
# full config dict + device, like MultCrossTSBaseline) without train.py caring which.
def _build_utde(config: dict, device: str):
    return UTDEBaseline(config, device)


MODEL_REGISTRY = {
    "sanity_baseline": _build_sanity_baseline,
    "mult_cross_ts": _build_mult_cross_ts,
    "utde": _build_utde,
    # "medpatch": ...       # TODO once models/baselines/medpatch.py is implemented
    # "fusemoe": ...
    # "drfuse": ...
    # "medfuse": ...
    # "ours": ...           # models/ours/backbone.py + sdca.py + sarl.py
}


def load_tables(data_dir: Path):
    labels = pd.read_parquet(data_dir / "sepsis_labels.parquet")
    ehr = pd.read_parquet(data_dir / "ehr_timeseries.parquet")
    notes = pd.read_parquet(data_dir / "notes.parquet")
    cxr = pd.read_parquet(data_dir / "cxr_metadata.parquet")
    return labels, ehr, notes, cxr


def move_batch_to_device(batch: dict, device: str) -> dict:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        elif isinstance(v, dict):
            out[k] = move_batch_to_device(v, device)
        else:
            out[k] = v  # ragged text/path lists, hours_to_onset list -- stay on CPU
    return out


def run_epoch(model, dl, device, optimizer=None) -> tuple[float, list, list, list, list]:
    train_mode = optimizer is not None
    model.train(train_mode)
    loss_fn = nn.BCEWithLogitsLoss()
    total_loss, n_batches = 0.0, 0
    all_labels, all_probs, all_hours, all_pre_susp = [], [], [], []

    epoch_t0 = time.time()
    for i, batch in enumerate(dl):
        if i % 50 == 0 and i > 0:
            elapsed = time.time() - epoch_t0
            rate = i / elapsed  # batches/sec
            remaining = (len(dl) - i) / rate
            print(f"    batch {i}/{len(dl)} | {elapsed:.0f}s elapsed | "
                  f"ETA {remaining/60:.1f} min", flush=True)
        batch = move_batch_to_device(batch, device)
        with torch.set_grad_enabled(train_mode):
            logits = model(batch)
            loss = loss_fn(logits, batch["label"])
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += loss.item()
        n_batches += 1
        all_labels.extend(batch["label"].detach().cpu().tolist())
        all_probs.extend(torch.sigmoid(logits).detach().cpu().tolist())
        all_hours.extend(batch["hours_to_onset"])
        all_pre_susp.extend(batch["used_only_pre_suspicion"].detach().cpu().tolist())

    return total_loss / max(n_batches, 1), all_labels, all_probs, all_hours, all_pre_susp


def main(config: dict, smoke_test: bool = False):
    seed = config.get("seed", 1002)
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if smoke_test:
        from make_synthetic_data import make_synthetic_tables
        labels, ehr, notes, cxr = make_synthetic_tables(n_admissions=config.get("smoke_n_admissions", 300))
    else:
        labels, ehr, notes, cxr = load_tables(Path(config["data_dir"]))

    modalities = tuple(config.get("modalities", ["ts", "notes"]))
    susp = labels.set_index("hadm_id")["suspicion_time_hours"].dropna() \
        if "suspicion_time_hours" in labels.columns else None

    common = dict(labels_df=labels, ehr_df=ehr, notes_df=notes, cxr_df=cxr,
                  modalities=modalities, obs_buffer_hours=config.get("obs_buffer_hours", 5.0),
                  suspicion_times=susp, seed=seed)
    train_ds = SepsisDataset(split="train", neg_subsample_ratio=config.get("neg_subsample_ratio", 5.0), **common)
    val_ds = SepsisDataset(split="val", neg_subsample_ratio=None, **common)
    test_ds = SepsisDataset(split="test", neg_subsample_ratio=None, **common)

    print(f"train/val/test PATIENTS: {len(train_ds.labels)}/{len(val_ds.labels)}/{len(test_ds.labels)}")
    print(f"train/val/test timepoints: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")
    print(f"train label balance (timepoints): {Counter(r[2] for r in train_ds.index)}")
    print(f"train/val/test patient-level positive rate: "
          f"{train_ds.labels['label'].mean():.2%}/{val_ds.labels['label'].mean():.2%}/"
          f"{test_ds.labels['label'].mean():.2%}")

    bs = config.get("batch_size", 32)
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, collate_fn=collate_sepsis_batch, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=bs, shuffle=False, collate_fn=collate_sepsis_batch, num_workers=0)
    test_dl = DataLoader(test_ds, batch_size=bs, shuffle=False, collate_fn=collate_sepsis_batch, num_workers=0)

    model_name = config.get("model", "sanity_baseline")
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Available: {list(MODEL_REGISTRY)}. "
                          f"Real baselines aren't wired in yet -- see MODEL_REGISTRY comment.")
    model = MODEL_REGISTRY[model_name](config, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get("lr", 1e-3))

    epochs = config.get("epochs", 20)
    patience = config.get("patience", 5)
    best_val_auprc, best_state, epochs_no_improve = -1.0, None, 0

    for epoch in range(1, epochs + 1):
        epoch_t0 = time.time()
        train_loss, *_ = run_epoch(model, train_dl, device, optimizer)
        val_loss, val_labels, val_probs, val_hours, val_pre_susp = run_epoch(model, val_dl, device)
        epoch_duration = time.time() - epoch_t0
        epochs_remaining = epochs - epoch
        print(f"  epoch took {epoch_duration/60:.1f} min | "
              f"~{epochs_remaining * epoch_duration / 60:.1f} min remaining "
              f"for the other {epochs_remaining} epoch(s) (assumes similar cost, "
              f"ignores early stopping)")
        # Fast point estimates only -- no bootstrap CI per epoch, that's what was slow.
        # Full bootstrap (evaluate_both_protocols, config's eval_bootstrap_n) runs once,
        # on the test set, after training -- see save_results() below.
        val_metrics = compute_metrics(val_labels, val_probs, n_boot=0, seed=seed)
        val_auprc = val_metrics.auprc
        print(f"epoch {epoch:3d} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} "
              f"| val_auroc {val_metrics.auroc:.3f} | val_auprc {val_auprc:.3f}")

        if val_auprc > best_val_auprc:
            best_val_auprc, best_state, epochs_no_improve = val_auprc, model.state_dict(), 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"early stopping at epoch {epoch} (no val AUPRC improvement for {patience} epochs)")
                break

    model.load_state_dict(best_state)
    test_loss, test_labels, test_probs, test_hours, test_pre_susp = run_epoch(model, test_dl, device)
    test_metrics = evaluate_both_protocols(test_labels, test_probs, test_pre_susp, test_hours,
                                            n_boot=config.get("eval_bootstrap_n", 1000), seed=seed)

    save_results(config, model_name, best_val_auprc, test_metrics, smoke_test)


def save_results(config: dict, model_name: str, best_val_auprc: float, test_metrics: dict, smoke_test: bool):
    results_dir = Path(config.get("results_dir", DEFAULT_RESULTS_DIR))
    results_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"{model_name}_{config.get('run_tag', 'run')}_{int(time.time())}"
    out = {
        "run_name": run_name, "model": model_name, "config": config,
        "smoke_test": smoke_test, "best_val_auprc": best_val_auprc,
        "test_metrics": test_metrics,
    }
    path = results_dir / f"{run_name}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"saved results to {path}")
    print(f"test standard AUROC {test_metrics['standard']['auroc']:.3f} "
          f"AUPRC {test_metrics['standard']['auprc']:.3f}")
    if test_metrics.get("strict_pre_suspicion"):
        print(f"test strict-pre-suspicion AUROC {test_metrics['strict_pre_suspicion']['auroc']:.3f} "
              f"AUPRC {test_metrics['strict_pre_suspicion']['auprc']:.3f}")
    else:
        print(f"strict pre-suspicion protocol skipped: {test_metrics.get('strict_pre_suspicion_note')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="path to a YAML config under experiments/configs/")
    ap.add_argument("--smoke_test", action="store_true",
                     help="ignore config data_dir, generate synthetic data in-memory instead")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.seed is not None:
        cfg["seed"] = args.seed

    main(cfg, smoke_test=args.smoke_test)