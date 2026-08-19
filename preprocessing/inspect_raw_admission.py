"""
inspect_raw_admission.py -- ground-truth check, bypassing DuckDB entirely. Pulls the
raw CSV lines for one specific hadm_id directly from admissions.csv and icustays.csv
using Python's csv module, so we can see with our own eyes whether the huge intime-
admittime gap is really sitting in the source file, or whether something upstream of
this point is still introducing it.

Usage: python inspect_raw_admission.py --mimic_hosp_dir /path/to/hosp \
    --mimic_icu_dir /path/to/icu --hadm_id 20478888
"""
import argparse
import csv
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--mimic_hosp_dir", required=True)
ap.add_argument("--mimic_icu_dir", required=True)
ap.add_argument("--hadm_id", type=int, required=True)
args = ap.parse_args()

hosp_dir = Path(args.mimic_hosp_dir)
icu_dir = Path(args.mimic_icu_dir)
target = str(args.hadm_id)


def find_rows(csv_path: Path, hadm_id_col_candidates=("hadm_id",)):
    if not csv_path.exists():
        gz = csv_path.with_suffix(csv_path.suffix + ".gz")
        if gz.exists():
            import gzip
            f = gzip.open(gz, "rt", newline="")
        else:
            print(f"  NOT FOUND: {csv_path}")
            return
    else:
        f = open(csv_path, "r", newline="")
    with f:
        reader = csv.DictReader(f)
        hadm_col = next((c for c in reader.fieldnames if c in hadm_id_col_candidates), None)
        if hadm_col is None:
            print(f"  no hadm_id-like column found in {csv_path.name}, columns: {reader.fieldnames}")
            return
        n_matches = 0
        for row in reader:
            if row.get(hadm_col) == target:
                n_matches += 1
                print(f"  MATCH {n_matches}: {dict(row)}")
        print(f"  -> {n_matches} raw row(s) found for hadm_id={target} in {csv_path.name}")


print(f"=== admissions.csv, hadm_id={target} ===")
find_rows(hosp_dir / "admissions.csv")

print(f"\n=== icustays.csv, hadm_id={target} ===")
find_rows(icu_dir / "icustays.csv")