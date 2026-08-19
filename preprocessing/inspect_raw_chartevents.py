"""
inspect_raw_chartevents.py -- ground-truth check for one hadm_id directly in
chartevents.csv (too large to scan row-by-row in Python at ~300M+ rows). Uses grep
for a fast first pass, then verifies each candidate line with a real CSV parse
(matched against the header's actual hadm_id column position) so a coincidental
substring match elsewhere in the row doesn't fool us.

Usage:
    python inspect_raw_chartevents.py --mimic_icu_dir /path/to/icu --hadm_id 25934959
"""
import argparse
import csv
import io
import subprocess
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--mimic_icu_dir", required=True)
ap.add_argument("--hadm_id", type=int, required=True)
ap.add_argument("--max_candidates", type=int, default=200,
                 help="Safety cap on how many grep candidate lines to CSV-verify.")
args = ap.parse_args()

chartevents_path = Path(args.mimic_icu_dir) / "chartevents.csv"
if not chartevents_path.exists():
    gz = chartevents_path.with_suffix(chartevents_path.suffix + ".gz")
    if gz.exists():
        chartevents_path = gz
    else:
        raise FileNotFoundError(f"chartevents.csv not found under {args.mimic_icu_dir}")

target = str(args.hadm_id)

# header, to find hadm_id's real column index
with open(chartevents_path, "r", newline="") if chartevents_path.suffix == ".csv" else \
     __import__("gzip").open(chartevents_path, "rt", newline="") as f:
    header = next(csv.reader(f))
hadm_col_idx = header.index("hadm_id")
print(f"chartevents.csv columns: {header}")
print(f"hadm_id is column index {hadm_col_idx}")

# fast candidate search via grep (fixed-string, so it's a literal substring search --
# candidates may include false positives from other columns, filtered below)
print(f"\ngrepping for '{target}' (fast candidate pass, may take a minute on a large file)...")
grep_cmd = ["grep", "-F", f",{target},"] if chartevents_path.suffix == ".csv" else \
           ["zgrep", "-F", f",{target},"]
result = subprocess.run(grep_cmd + [str(chartevents_path)], capture_output=True, text=True)
candidate_lines = result.stdout.splitlines()
print(f"grep found {len(candidate_lines)} candidate line(s)")

confirmed = []
for line in candidate_lines[:args.max_candidates]:
    row = next(csv.reader(io.StringIO(line)))
    if len(row) > hadm_col_idx and row[hadm_col_idx] == target:
        confirmed.append(dict(zip(header, row)))

print(f"\n{len(confirmed)} confirmed row(s) with hadm_id == {target} (verified against "
      f"the real column position, not just a substring match):")
for r in confirmed:
    print(f"  {r}")