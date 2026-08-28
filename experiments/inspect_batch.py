"""One-off diagnostic: print the real batch schema collate_sepsis_batch produces,
so drfuse.py's adapter can be fixed against ground truth instead of guesses.
Run from the repo root: python experiments/inspect_batch.py
"""
import sys
from pathlib import Path

import pandas as pd
import yaml
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for `dataset`/`evaluate`

from dataset import SepsisDataset, collate_sepsis_batch  # noqa: E402

CONFIG_PATH = REPO_ROOT / "experiments" / "configs" / "drfuse.yaml"

with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

data_dir = Path(cfg["data_dir"])
labels = pd.read_parquet(data_dir / "sepsis_labels.parquet")
ehr = pd.read_parquet(data_dir / "ehr_timeseries.parquet")
notes = pd.read_parquet(data_dir / "notes.parquet")
cxr = pd.read_parquet(data_dir / "cxr_metadata.parquet")

ds = SepsisDataset(
    split="train", labels_df=labels, ehr_df=ehr, notes_df=notes, cxr_df=cxr,
    modalities=("ts", "cxr"), obs_buffer_hours=cfg.get("obs_buffer_hours", 5.0),
    neg_subsample_ratio=cfg.get("neg_subsample_ratio", 5.0), suspicion_times=None, seed=1,
)
dl = DataLoader(ds, batch_size=4, collate_fn=collate_sepsis_batch)
batch = next(iter(dl))


def describe(d, prefix=""):
    for k, v in d.items():
        if isinstance(v, dict):
            print(f"{prefix}{k}: (nested dict)")
            describe(v, prefix=prefix + "  ")
        else:
            shape = getattr(v, "shape", None)
            dtype = getattr(v, "dtype", None)
            if shape is not None:
                print(f"{prefix}{k}: shape={tuple(shape)} dtype={dtype}")
            else:
                sample = v[:3] if hasattr(v, "__getitem__") else v
                print(f"{prefix}{k}: type={type(v).__name__} sample={sample}")


print("=== batch schema ===")
describe(batch)
