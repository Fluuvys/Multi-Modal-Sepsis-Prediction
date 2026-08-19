from __future__ import annotations
"""
dataset.py -- shared PyTorch Dataset for the rolling hourly sepsis-onset task.

Every baseline and our own model reads through THIS file. Per PROJECT_CONTEXT.md rule #5
and docs/data_schema.md, this stays a thin, lossless pass-through over the master Parquet
tables -- it does NOT bin, truncate, or reduce anything. Model-specific reduction (e.g.
MedPatch's hourly-binning + most-recent-only behavior) belongs in each model's own adapter
in models/baselines/*.py or models/ours/*.py, not here.

WHAT THIS FILE DOES:
    1. Builds the hourly prediction-timepoint index per admission (Section 7 of
       PROJECT_CONTEXT.md: one timepoint/hour, starting after an observation buffer,
       stopping at onset or discharge, label = 1 if onset is within the next W=4h).
    2. For a given (hadm_id, t) sample, returns every observation from every modality
       with timestamp <= t (causal -- nothing after the prediction time leaks in),
       as raw irregular event streams, not binned.
    3. Provides a collate_fn that pads the irregular per-sample sequences into a batch,
       plus a boolean mask per modality per token, and leaves free text / image paths as
       ragged Python lists (tokenization / JPEG loading is model-specific and belongs in
       each model's forward pass or adapter, not here).

*** SCHEMA GAP -- READ BEFORE USING ***
docs/data_schema.md's `hours_before_onset` column is computed relative to
`sepsis_labels.sepsis_onset_time`, which is NULL for every negative admission. That makes
it unusable as the time axis for the rolling task, since negatives need hourly timepoints
across their whole stay too (Section 7: "Negative patients are label = 0 at every valid
hourly timepoint across their stay"). This file assumes two small additions your
preprocessing outputs need to carry (they're almost certainly already computed
internally in ehr_extraction.py / label_sepsis3.py since LOS filtering depends on them --
they just need to be written out):

    sepsis_labels.parquet   needs: icu_intime (datetime), icu_los_hours (float)
    ehr_timeseries.parquet  needs: hours_since_admission (float)  [= timestamp - icu_intime]
    notes.parquet           needs: hours_since_admission (float)
    cxr_metadata.parquet    needs: hours_since_admission (float)

`hours_before_onset` is still useful (and kept) for the lead-time stratification (Table 2)
and the strict pre-suspicion protocol (Section 4) -- it's just not the indexing axis.
If you'd rather not touch the schema, pass `icu_intime_col=None` and supply
`hours_since_admission` yourself upstream; the loader will use it as-is if present.
"""


import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# TODO: confirm this matches the exact variable set / naming used in ehr_extraction.py.
# Fixed, ordered vocab so variable indices are stable across train/val/test and across
# every model that consumes this dataset.
# Exact strings from ehr_extraction.py's VARIABLE_ITEMID_MAP / ALL_17_VARIABLES --
# NOT a naming convention of our own choosing. Confirmed by direct inspection of
# ehr_extraction.py's actual output column, not assumed. Note "Glascow" (not
# "Glasgow") -- that's the real spelling preserved from the MIMIC-IV source data.
ALL_17_VARIABLES = [
    "Capillary refill rate", "Diastolic blood pressure", "Fraction inspired oxygen",
    "Glascow coma scale eye opening", "Glascow coma scale motor response",
    "Glascow coma scale total", "Glascow coma scale verbal response", "Glucose",
    "Heart Rate", "Height", "Mean blood pressure", "Oxygen saturation",
    "Respiratory rate", "Systolic blood pressure", "Temperature", "Weight", "pH",
]
VARIABLE_VOCAB = {name: i for i, name in enumerate(ALL_17_VARIABLES)}
NOTE_TYPE_VOCAB = {"RR": 0, "DN": 1}

HORIZON_HOURS = 4.0  # locked, PROJECT_CONTEXT.md Section 7 -- do not change per-run


@dataclass
class SepsisSample:
    hadm_id: int
    t_hours: float
    label: int
    hours_to_onset: Optional[float]  # true distance-to-onset, for lead-time stratification; None if negative
    used_only_pre_suspicion: bool    # True if every observation in this sample predates suspicion_time
    ts_hours: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    ts_var_idx: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    ts_value: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    notes_hours: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    notes_type_idx: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    notes_text: list = field(default_factory=list)
    cxr_hours: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    cxr_path: list = field(default_factory=list)


class SepsisDataset(Dataset):
    """One sample = one (admission, hourly prediction timepoint) pair.

    Args:
        labels_df: sepsis_labels.parquet, already filtered to columns described in
            docs/data_schema.md plus icu_intime/icu_los_hours (see module docstring).
        ehr_df, notes_df, cxr_df: master tables, each with hours_since_admission.
        split: "train" / "val" / "test" -- filters labels_df.
        modalities: subset of {"ts", "notes", "cxr"} to include. Drop "cxr" here while
            CXR preprocessing is still running -- everything else works unmodified.
        obs_buffer_hours: don't generate a prediction timepoint before this many hours
            of admission (need *something* to condition on).
        neg_subsample_ratio: if set (e.g. 5.0), randomly keep at most this many negative
            timepoints per positive timepoint, GLOBALLY. Keep None for val/test (no
            subsampling at eval time -- only use it for the train split).
        suspicion_times: optional Series indexed by hadm_id giving suspicion_time in
            hours-since-admission, for the Section 4 strict pre-suspicion protocol. If
            provided, each sample records whether every observation used predates it.
    """

    def __init__(
        self,
        labels_df: pd.DataFrame,
        ehr_df: pd.DataFrame,
        notes_df: pd.DataFrame,
        cxr_df: pd.DataFrame,
        split: str,
        modalities: Sequence[str] = ("ts", "notes", "cxr"),
        obs_buffer_hours: float = 5.0,
        neg_subsample_ratio: Optional[float] = None,
        suspicion_times: Optional[pd.Series] = None,
        seed: int = 1002,
    ):
        assert set(modalities) <= {"ts", "notes", "cxr"}
        self.modalities = set(modalities)
        self.obs_buffer_hours = obs_buffer_hours
        self.suspicion_times = suspicion_times
        self.rng = np.random.default_rng(seed)

        labels = labels_df[labels_df["split"] == split].copy()
        labels = labels[labels["excluded_reason"].isna()]
        for col in ("icu_intime", "icu_los_hours"):
            if col not in labels.columns:
                raise ValueError(
                    f"sepsis_labels is missing '{col}'. See the module docstring under "
                    f"'SCHEMA GAP' -- the rolling task needs an admission-time reference "
                    f"independent of sepsis_onset_time (which is null for negatives)."
                )
        self.labels = labels.set_index("hadm_id")

        # group once, O(1) per-admission lookup in __getitem__ instead of re-filtering
        # the whole table every sample
        self._ehr_by_hadm = self._group(ehr_df, "hours_since_admission") if "ts" in self.modalities else {}
        self._notes_by_hadm = self._group(notes_df, "hours_since_admission") if "notes" in self.modalities else {}
        self._cxr_by_hadm = self._group(cxr_df, "hours_since_admission") if "cxr" in self.modalities else {}

        self.index = self._build_timepoint_index()
        if neg_subsample_ratio is not None:
            self.index = self._subsample_negatives(self.index, neg_subsample_ratio)

    @staticmethod
    def _group(df: pd.DataFrame, sort_col: str) -> dict:
        if "hours_since_admission" not in df.columns:
            raise ValueError(
                "Expected a 'hours_since_admission' column -- see module docstring "
                "under 'SCHEMA GAP'."
            )
        out = {}
        for hadm_id, g in df.sort_values(sort_col).groupby("hadm_id"):
            out[hadm_id] = g.reset_index(drop=True)
        return out

    def _build_timepoint_index(self) -> list[tuple]:
        """One entry per (hadm_id, t_hours, label, hours_to_onset)."""
        rows = []
        for hadm_id, row in self.labels.iterrows():
            los = row["icu_los_hours"]
            onset = row.get("sepsis_onset_time_hours", None)  # see note below
            # Prefer an explicit onset-in-hours-since-admission column if you have one;
            # falling back to None means this admission is treated as negative for
            # indexing purposes even if label==1, which would be wrong -- so require it
            # whenever label==1.
            if row["label"] == 1 and (onset is None or (isinstance(onset, float) and math.isnan(onset))):
                raise ValueError(
                    f"hadm_id={hadm_id} is labeled positive but has no "
                    f"'sepsis_onset_time_hours' (onset expressed in hours-since-"
                    f"admission, not a datetime). Add this column alongside "
                    f"icu_intime/icu_los_hours -- it's just "
                    f"(sepsis_onset_time - icu_intime) in hours."
                )
            last_hour = onset if row["label"] == 1 else los
            if last_hour is None or (isinstance(last_hour, float) and math.isnan(last_hour)) \
                    or last_hour <= self.obs_buffer_hours:
                continue  # nothing usable (including NaN icu_los_hours -- some ICU stays have
                          # a null outtime in icustays.csv) -- exclude, matches the
                          # 4h-of-admission exclusion logic
            for t in np.arange(self.obs_buffer_hours, last_hour, 1.0):
                if row["label"] == 1:
                    label = 1 if (onset - t) <= HORIZON_HOURS else 0
                    hours_to_onset = onset - t
                else:
                    label = 0
                    hours_to_onset = None
                rows.append((hadm_id, float(t), int(label), hours_to_onset))
        return rows

    def _subsample_negatives(self, index: list[tuple], ratio: float) -> list[tuple]:
        pos = [r for r in index if r[2] == 1]
        neg = [r for r in index if r[2] == 0]
        keep_n = min(len(neg), int(len(pos) * ratio))
        if keep_n < len(neg):
            idxs = self.rng.choice(len(neg), size=keep_n, replace=False)
            neg = [neg[i] for i in idxs]
        combined = pos + neg
        self.rng.shuffle(combined)
        return combined

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx: int) -> SepsisSample:
        hadm_id, t, label, hours_to_onset = self.index[idx]
        sample = SepsisSample(
            hadm_id=hadm_id, t_hours=t, label=label, hours_to_onset=hours_to_onset,
            used_only_pre_suspicion=self._check_pre_suspicion(hadm_id, t),
        )

        if "ts" in self.modalities and hadm_id in self._ehr_by_hadm:
            g = self._ehr_by_hadm[hadm_id]
            g = g[g["hours_since_admission"] <= t]
            sample.ts_hours = g["hours_since_admission"].to_numpy(dtype=np.float32)
            sample.ts_var_idx = g["variable_name"].map(VARIABLE_VOCAB).to_numpy(dtype=np.int64)
            sample.ts_value = g["value"].to_numpy(dtype=np.float32)

        if "notes" in self.modalities and hadm_id in self._notes_by_hadm:
            g = self._notes_by_hadm[hadm_id]
            g = g[g["hours_since_admission"] <= t]
            sample.notes_hours = g["hours_since_admission"].to_numpy(dtype=np.float32)
            sample.notes_type_idx = g["note_type"].map(NOTE_TYPE_VOCAB).to_numpy(dtype=np.int64)
            sample.notes_text = g["raw_text"].tolist()

        if "cxr" in self.modalities and hadm_id in self._cxr_by_hadm:
            g = self._cxr_by_hadm[hadm_id]
            g = g[g["hours_since_admission"] <= t]
            sample.cxr_hours = g["hours_since_admission"].to_numpy(dtype=np.float32)
            sample.cxr_path = g["image_path"].tolist()

        return sample

    def _check_pre_suspicion(self, hadm_id: int, t: float) -> bool:
        if self.suspicion_times is None or hadm_id not in self.suspicion_times.index:
            return False
        return t <= self.suspicion_times.loc[hadm_id]


def _pad_stack(seqs: list[np.ndarray], dtype, pad_value=0.0):
    lengths = [len(s) for s in seqs]
    max_len = max(lengths) if lengths else 0
    max_len = max(max_len, 1)  # avoid zero-width tensors when a whole batch has no obs
    out = np.full((len(seqs), max_len), pad_value, dtype=dtype)
    mask = np.zeros((len(seqs), max_len), dtype=bool)
    for i, s in enumerate(seqs):
        n = len(s)
        if n:
            out[i, :n] = s
            mask[i, :n] = True
    return torch.as_tensor(out), torch.as_tensor(mask)


def collate_sepsis_batch(batch: list[SepsisSample]) -> dict:
    """Pads each modality's ragged sequences to the batch max length + boolean mask.
    Text and image paths are kept as ragged Python lists (list-of-lists, one per sample)
    since tokenization / JPEG loading is model-specific -- zip against the notes/cxr mask
    in your model's adapter to know which slots are real vs padding.
    """
    hadm_ids = torch.tensor([b.hadm_id for b in batch], dtype=torch.long)
    t_hours = torch.tensor([b.t_hours for b in batch], dtype=torch.float32)
    labels = torch.tensor([b.label for b in batch], dtype=torch.float32)
    hours_to_onset = [b.hours_to_onset for b in batch]  # keep as python list, has Nones
    pre_suspicion = torch.tensor([b.used_only_pre_suspicion for b in batch], dtype=torch.bool)

    ts_hours, ts_mask = _pad_stack([b.ts_hours for b in batch], np.float32)
    _, _ = ts_mask, ts_hours  # reuse mask below
    ts_var_idx, _ = _pad_stack([b.ts_var_idx for b in batch], np.int64, pad_value=-1)
    ts_value, _ = _pad_stack([b.ts_value for b in batch], np.float32)

    notes_hours, notes_mask = _pad_stack([b.notes_hours for b in batch], np.float32)
    notes_type_idx, _ = _pad_stack([b.notes_type_idx for b in batch], np.int64, pad_value=-1)
    notes_text = [b.notes_text for b in batch]

    cxr_hours, cxr_mask = _pad_stack([b.cxr_hours for b in batch], np.float32)
    cxr_path = [b.cxr_path for b in batch]

    return {
        "hadm_id": hadm_ids,
        "t_hours": t_hours,
        "label": labels,
        "hours_to_onset": hours_to_onset,
        "used_only_pre_suspicion": pre_suspicion,
        "ts": {"hours": ts_hours, "var_idx": ts_var_idx, "value": ts_value, "mask": ts_mask},
        "notes": {"hours": notes_hours, "type_idx": notes_type_idx, "text": notes_text, "mask": notes_mask},
        "cxr": {"hours": cxr_hours, "path": cxr_path, "mask": cxr_mask},
    }