"""
diagnose_full_cohort_intime.py -- checks icu_intime vs admissions.csv's own admittime
across the ENTIRE cohort (not just the 328 hadm_ids flagged by downstream ehr/notes/cxr
extremes) to find the TRUE prevalence of the icustays/admissions/patients CSV read bug
in label_sepsis3.py's connect_duckdb(). This determines whether the fix is "patch the
reader + cheaply re-derive icu_intime for a handful of rows" or "this touched enough
rows that sepsis_onset_time/label themselves need re-verification for the affected
admissions, since Stage B's hourly SOFA grid is built directly from intime."

Usage:
    python diagnose_full_cohort_intime.py --data_dir "/home/fluuvys-main/Research/Multi modal sepsis prediction/Multi-Modal-Sepsis-Prediction/data/cohort" \\
        --mimic_hosp_dir "/home/fluuvys-main/Research/Multi modal sepsis prediction/Data/mimic-iv-3.1/hosp"
"""
import argparse
from pathlib import Path

import duckdb
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--data_dir", type=str, default="data/cohort")
ap.add_argument("--mimic_hosp_dir", type=str, required=True)
args = ap.parse_args()
data_dir = Path(args.data_dir)

labels = pd.read_parquet(data_dir / "sepsis_labels.parquet")
print(f"checking icu_intime vs admittime for all {len(labels)} admissions in "
      f"sepsis_labels.parquet...")

hosp_dir = Path(args.mimic_hosp_dir)
adm_path = hosp_dir / "admissions.csv"
if not adm_path.exists():
    adm_path = adm_path.with_suffix(".csv.gz")

con = duckdb.connect()
# SAFE reader (parallel=false, ignore_errors=false) -- this check must not be
# vulnerable to the same bug it's trying to diagnose
admittimes = con.execute(f"""
    SELECT hadm_id, CAST(admittime AS TIMESTAMP) AS admittime
    FROM read_csv('{adm_path.as_posix()}', parallel=false, ignore_errors=false)
""").df()
print(f"read {len(admittimes)} admittime rows from admissions.csv with the safe reader")

merged = labels[["hadm_id", "icu_intime", "label", "split"]].merge(admittimes, on="hadm_id", how="left")
missing_admittime = merged["admittime"].isna().sum()
if missing_admittime:
    print(f"WARNING: {missing_admittime} hadm_id(s) in sepsis_labels.parquet not found "
          f"in admissions.csv at all -- separate issue, investigate if this is nonzero.")

merged["icu_intime"] = pd.to_datetime(merged["icu_intime"])
merged["gap_hours"] = (merged["icu_intime"] - merged["admittime"]).dt.total_seconds() / 3600.0

suspicious = merged[(merged["gap_hours"] < 0) | (merged["gap_hours"] > 24 * 14)].dropna(subset=["gap_hours"])
print(f"\n{len(suspicious)} of {len(merged)} admissions ({100*len(suspicious)/len(merged):.3f}%) "
      f"have a suspicious icu_intime (negative gap, or > 14 days from admittime)")

if len(suspicious):
    n_positive_affected = int((suspicious["label"] == 1).sum())
    print(f"  -> {n_positive_affected} of these are LABEL=1 (sepsis-positive) admissions -- "
          f"for these specifically, sepsis_onset_time/label were computed against a "
          f"corrupted intime in Stage B and should be treated as unverified until "
          f"re-checked, not just their icu_intime/icu_los_hours/hours_since_admission.")
    print(f"\n  split breakdown of affected admissions:")
    print(suspicious["split"].value_counts().to_string())
    print(f"\n  gap_hours distribution among affected:")
    print(suspicious["gap_hours"].describe().to_string())