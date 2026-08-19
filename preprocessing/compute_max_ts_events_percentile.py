"""
compute_max_ts_events_percentile.py -- STraTS (Tipirneni & Reddy, 2022) sets their
max-observations cap using the 99th percentile of observations-per-ICU-stay from their
own cohort, explicitly to avoid memory overflow with batch gradient descent (Section 4.4
of the paper) -- the same problem utde.py's max_ts_events config knob exists to solve.

This script computes the analogous statistic for THIS project's setup: not observations
per whole stay (STraTS's target task uses a single fixed 24h window, so "per stay" and
"per window" coincide for them), but observations per LOOKBACK WINDOW ending at each
real hourly prediction timepoint t -- since that's the actual quantity max_ts_events
bounds here, given the rolling task's t varies within a stay.

Usage: python compute_max_ts_events_percentile.py --data_dir data/cohort --lookback_hours 48
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--data_dir", type=str, default="data/cohort")
ap.add_argument("--lookback_hours", type=float, default=48.0)
ap.add_argument("--obs_buffer_hours", type=float, default=5.0,
                 help="Matches dataset.py's obs_buffer_hours -- first prediction "
                      "timepoint per admission, same as your training config.")
ap.add_argument("--sample_admissions", type=int, default=2000,
                 help="Compute over a random sample of admissions rather than the "
                      "full cohort, for speed. None-equivalent: pass a number >= "
                      "cohort size to use everything.")
ap.add_argument("--percentile", type=float, default=99.0)
args = ap.parse_args()

data_dir = Path(args.data_dir)
labels = pd.read_parquet(data_dir / "sepsis_labels.parquet")
ehr = pd.read_parquet(data_dir / "ehr_timeseries.parquet")

eligible = labels[labels["excluded_reason"].isna()]
hadm_ids = eligible["hadm_id"].unique()
if len(hadm_ids) > args.sample_admissions:
    rng = np.random.default_rng(1002)
    hadm_ids = rng.choice(hadm_ids, size=args.sample_admissions, replace=False)

labels_by_hadm = eligible.set_index("hadm_id")
ehr_by_hadm = {h: g for h, g in ehr[ehr["hadm_id"].isin(hadm_ids)].groupby("hadm_id")}

counts = []
for hadm_id in hadm_ids:
    row = labels_by_hadm.loc[hadm_id]
    los = row["icu_los_hours"]
    onset = row.get("sepsis_onset_time_hours", None)
    last_hour = onset if row["label"] == 1 else los
    if last_hour is None or (isinstance(last_hour, float) and np.isnan(last_hour)) or last_hour <= args.obs_buffer_hours:
        continue
    g = ehr_by_hadm.get(hadm_id)
    if g is None:
        continue
    hours = g["hours_since_admission"].to_numpy()
    # same hourly grid dataset.py builds -- one prediction timepoint per hour
    for t in np.arange(args.obs_buffer_hours, last_hour, 1.0):
        window_count = int(((hours <= t) & (hours >= t - args.lookback_hours)).sum())
        counts.append(window_count)

counts = np.array(counts)
print(f"computed over {len(counts):,} (admission, timepoint) samples from "
      f"{len(hadm_ids)} admissions")
print(f"mean events per {args.lookback_hours:.0f}h window: {counts.mean():.1f}")
print(f"median: {np.median(counts):.0f}")
for p in [90, 95, 99, 99.5, 99.9]:
    print(f"  {p}th percentile: {int(np.percentile(counts, p))}")
print(f"\nSuggested max_ts_events (matching STraTS's {args.percentile:.0f}th-percentile "
      f"methodology): {int(np.percentile(counts, args.percentile))}")





