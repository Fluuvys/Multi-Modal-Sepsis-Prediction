"""
classify_extreme_ehr_events.py -- distinguishes "legitimate pre-ICU floor observation"
(event timestamp falls within [admittime, dischtime] of the SAME admission, just before
icu_intime) from "genuine misattribution/corruption" (event timestamp falls entirely
outside even the admission's own admittime-dischtime window -- physically impossible for
a real single hospitalization). This is the precise version of the cruder +/-7-day
threshold used earlier, which couldn't tell these two cases apart.

Usage:
    python classify_extreme_ehr_events.py \\
        --data_dir data/cohort --mimic_hosp_dir /path/to/hosp
"""
import argparse
from pathlib import Path

import duckdb
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--data_dir", type=str, default="data/cohort")
ap.add_argument("--mimic_hosp_dir", type=str, required=True)
ap.add_argument("--window_days", type=float, default=7,
                 help="Flag events with |hours_since_admission| beyond this many days "
                      "for classification (same threshold as the original ehr diagnostic).")
args = ap.parse_args()
data_dir = Path(args.data_dir)
window_hours = args.window_days * 24

ehr = pd.read_parquet(data_dir / "ehr_timeseries.parquet")
extreme = ehr[(ehr["hours_since_admission"] < -window_hours) | (ehr["hours_since_admission"] > window_hours)]
print(f"{len(ehr)} total ehr rows, {len(extreme)} ({100*len(extreme)/len(ehr):.3f}%) "
      f"outside +/-{args.window_days:.0f} days")

if extreme.empty:
    print("nothing to classify.")
    raise SystemExit

affected_hadm = extreme["hadm_id"].unique()
print(f"affects {len(affected_hadm)} distinct hadm_id(s)")

hosp_dir = Path(args.mimic_hosp_dir)
adm_path = hosp_dir / "admissions.csv"
if not adm_path.exists():
    adm_path = adm_path.with_suffix(".csv.gz")

con = duckdb.connect()
ids_sql = ",".join(str(int(h)) for h in affected_hadm)
admittimes = con.execute(f"""
    SELECT hadm_id, CAST(admittime AS TIMESTAMP) AS admittime,
           CAST(dischtime AS TIMESTAMP) AS dischtime
    FROM read_csv('{adm_path.as_posix()}', parallel=false, ignore_errors=false)
    WHERE hadm_id IN ({ids_sql})
""").df()

merged = extreme.merge(admittimes, on="hadm_id", how="left")
missing_adm = merged["admittime"].isna()
print(f"{missing_adm.sum()} extreme row(s) have a hadm_id not found in admissions.csv "
      f"at all -- these can't be classified, investigate separately.")

within_admission = (merged["timestamp"] >= merged["admittime"]) & (merged["timestamp"] <= merged["dischtime"])
legitimate = merged[within_admission & ~missing_adm]
still_unexplained = merged[~within_admission & ~missing_adm]

print(f"\n=== RESULTS ===")
print(f"{len(legitimate)} row(s) ({legitimate['hadm_id'].nunique()} hadm_id(s)): "
      f"LEGITIMATE -- timestamp falls within the admission's own [admittime, dischtime], "
      f"just before icu_intime (real pre-ICU floor/ED observation, not a bug)")
print(f"{len(still_unexplained)} row(s) ({still_unexplained['hadm_id'].nunique() if len(still_unexplained) else 0} "
      f"hadm_id(s)): STILL UNEXPLAINED -- timestamp falls OUTSIDE even the admission's own "
      f"admittime-dischtime window -- not explainable by a long pre-ICU stay, needs further "
      f"investigation")

if len(still_unexplained):
    print(f"\nWorst still-unexplained rows:")
    worst = still_unexplained.reindex(
        still_unexplained["hours_since_admission"].abs().sort_values(ascending=False).index
    )
    print(worst[["hadm_id", "timestamp", "admittime", "dischtime", "hours_since_admission"]]
          .head(10).to_string(index=False))