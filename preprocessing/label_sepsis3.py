from __future__ import annotations
"""
label_sepsis3.py

PURPOSE:
    Single source of truth for BOTH cohort construction (filtering) AND Sepsis-3
    labeling on MIMIC-IV. Every downstream experiment depends on this file's output --
    do NOT reimplement this logic anywhere else in the repo. If the definition needs
    to change, discuss with the team first (see PROJECT_CONTEXT.md rule #2).

TWO STAGES, KEEP AS SEPARATE FUNCTIONS INTERNALLY:

    Stage A -- build_cohort(): filtering applied BEFORE labeling
        - One ICU stay per patient (first eligible stay only)
        - Adults only (age >= 18)
        - Minimum ICU length of stay (exclude very short stays)
        - Exclude admissions where sepsis criteria are already met at/before
          ICU admission (no "early prediction" possible -- exclude, don't label)
        - Exclude admissions with insufficient chart/lab density for reliable SOFA
        - Split assignment at subject_id level, NOT hadm_id level -- if a patient
          contributes more than one admission, ALL go in the same split

    Stage B -- assign_labels(cohort): Sepsis-3 labeling on the filtered cohort
        - Suspicion of infection: earlier timestamp of IV antibiotics + blood
          cultures, within 24h (antibiotics -> cultures) or 72h (cultures ->
          antibiotics)
        - Sepsis onset: suspicion of infection AND SOFA score increase >= 2 points,
          within a -48h/+24h window around suspicion

See PROJECT_CONTEXT.md section 5 for the full locked definition of both stages.

TODO (assign as a GitHub issue, milestone 0):
    [ ] Implement build_cohort() with all filtering criteria above
    [ ] Implement assign_labels() -- suspicion-of-infection + SOFA-over-time +
        sepsis onset timestamp
    [ ] Output: one row per admission with columns
        [subject_id, hadm_id, sepsis_onset_time, sofa_at_onset, label,
        excluded_reason, split]
    [ ] Sanity check cohort size/positive rate against published Sepsis-3-on-MIMIC
        benchmarks before moving on -- do not proceed to modeling until this passes

INPUT:  raw MIMIC-IV tables (see data/raw_links/)
OUTPUT: data/cohort/sepsis_labels.parquet -- see docs/data_schema.md for exact schema
"""

# TODO: implementation goes here
"""
preprocessing/label_sepsis3.py
================================================================================
SINGLE SOURCE OF TRUTH for Sepsis-3 cohort construction and labeling.
See PROJECT_CONTEXT.md rule #2: if you need a different definition, discuss
with the team first — do not fork this logic elsewhere.

Locked definition (PROJECT_CONTEXT.md §5):
  - Suspicion of infection: earlier of (first antibiotic, first culture),
    counted only if the other event follows within antibiotics->cultures 24h,
    or cultures->antibiotics 72h.
  - Sepsis onset: first time total SOFA increases by >=2 relative to baseline,
    within a -48h/+24h window around the suspicion-of-infection time.
  - One row per subject_id (first eligible ICU stay only).
  - Split assigned once at the subject_id level, persisted, never regenerated.

Reference implementations this file was cross-checked against (per the task):
  - MIT-LCP/mimic-code, mimic-iv/concepts_duckdb/{sepsis,score,medication,measurement}/*.sql
  - alistairewj/sepsis3-mimic, query/tbls/suspicion-of-infection.sql (MIMIC-III;
    ported table/column names to MIMIC-IV)
  - yinchangchang/SepsisCalc, code/preprocessing/ (hours-since-admission alignment
    pattern; raw unbinned timestamps used here per instructions, NOT their 3h bins)

TODO checklist (kept in sync with the GitHub issue / Milestone 0 - Cohort lock):
  [x] build_cohort()
  [x] assign_labels()
  [x] cohort_stats summary (console + JSON)
  [ ] Team sign-off on the ASSUMPTION items below before this is truly "locked"

================================================================================
ASSUMPTIONS / DEVIATIONS FLAGGED FOR TEAM REVIEW — read before trusting output
================================================================================
Search "ASSUMPTION" in this file for the code sites. Summary:

A1. "Blood cultures" (task spec) vs "any culture" (both mimic-code's
    suspicion_of_infection.sql and alistairewj/sepsis3-mimic's suspicion-of-
    infection.sql). NEITHER reference implementation restricts to
    spec_type_desc containing 'BLOOD' — they use ALL microbiologyevents rows
    regardless of specimen type. I implemented ANY-culture to match both
    references (rule: cross-check the three and don't invent our own
    definition). A `--blood_cultures_only` flag is provided to restrict to
    blood specimens if the team decides the task spec's literal wording should
    win instead.

A2. Antibiotic route: the task spec says "first IV antibiotic time". The
    mimic-code reference does NOT restrict to IV — it includes all
    non-topical/non-ophthalmic/non-otic routes (PO, IV, IM, etc.), matching
    the broader Seymour et al. Sepsis-3 "new antimicrobial" definition. I
    implemented the mimic-code (broader) behavior with a `--iv_antibiotics_only`
    flag to restrict to IV/IV push/IV drip routes if the team wants the
    literal "IV antibiotic" reading instead.

A3. "First eligible ICU stay per patient" is implemented as: take each
    subject's chronologically FIRST ICU stay only, and apply the age/LOS/
    density filters to that stay. If it fails, the subject is excluded
    entirely (excluded_reason populated) rather than falling back to a later
    stay. This is the more literal reading of "first eligible stay only, not
    all admissions" but the team should confirm this is the intended behavior
    versus "first stay that happens to satisfy eligibility, skipping earlier
    ineligible stays for the same subject."

A4. Baseline SOFA (needed for the ">= 2 point increase" test) is not directly
    defined anywhere in the task spec or the three references for the MIMIC
    setting (patients arrive without a known premorbid SOFA). Per the common
    Sepsis-3-on-MIMIC convention (Johnson et al., Seymour et al.), I use the
    SOFA value computed at ICU-admission hour (hr=0) as baseline. This is an
    assumption the team should confirm.

A5. [SUPERSEDED -- see PB2 in _compute_hourly_sofa's docstring] Ventilation
    status was originally approximated as "a Ventilator Mode chartevent
    within +/-6h", not the real mimic-code state machine. This has been
    replaced with a faithful port of treatment/ventilation.sql +
    measurement/ventilator_setting.sql + measurement/oxygen_delivery.sql
    (device/mode classification, then episode-building with the 14h
    continuity-gap rule), verified against a synthetic ventilated patient to
    confirm the classification and episode logic actually work, not just
    compile. Left here only so the history of the fix is visible; the
    current, accurate description of what this file does lives in PB2.

A6. [SUPERSEDED -- see PB1 in _compute_hourly_sofa's docstring] PaO2/FiO2
    pairing originally used chartevents FiO2 only, with no fraction/percent
    normalization. This has been replaced with a faithful port of bg.sql's
    actual dual-source logic: prefer a directly-drawn labevents FiO2 (itemid
    50816, with bg.sql's exact 0.2-1.0-as-fraction / 20-100-as-percent
    normalization), falling back to the nearest preceding chartevents FiO2
    (itemid 223835, same normalization, 4h lookback) only if no lab FiO2
    exists. Left here only so the history of the fix is visible; the
    current, accurate description lives in PB1.

A7. "Missing SOFA component -> assume healthy (0 points)" (task spec, item 2
    under Stage B) is operationalized using the SAME mechanism mimic-code
    itself uses to encode exactly this principle: each component is taken as
    the WORST (max severity) value in the trailing 24h window, and only
    coalesced to 0 if there is truly no value anywhere in that 24h window.
    This avoids a raw hour-to-hour flicker of the score that a literal
    "missing-this-instant -> 0" reading would otherwise produce, while
    satisfying the same non-dropping intent as the spec.

A8. [Re-corrected after direct repo inspection -- see A8.1 below] The task
    spec's ">= 5 valid hourly observation timepoints" was described as
    "adapted from SepsisCalc's own exclusion threshold." Confirmed by
    directly cloning yinchangchang/SepsisCalc and inspecting
    code/preprocessing/generate_sepsis_variables.py: the actual line is
    `if len(file_data) < 5 or sorted(vector)[2] < 1`, where `file_data` is
    built from `delta_hour = int((now_time - hadm_time) / 3600)` -- i.e.
    RAW, UNBINNED integer hours since admission, one row per distinct hour
    with at least one new observation across their merged lab+vital+SOFA
    feature set. There is no 3-hour binning anywhere in their preprocessing
    code (searched the full directory for it). So SepsisCalc's "5" IS a
    raw-hour count, at the same granularity MIN_VALID_OBS_HOURS already uses
    here -- "5" is a faithful match to their precedent, not a looser
    approximation of it.

A8.1. [Correction of a correction] An earlier version of this docstring
    claimed SepsisCalc's "5" corresponds to ~15 raw hours because their data
    was supposedly 3-hour-binned, and suggested --min_valid_obs_hours 15 as
    the "faithful" reading. That claim was wrong and should not have been
    taken on faith without checking the actual repository -- there is no 3h
    binning step in SepsisCalc's code. --min_valid_obs_hours remains
    available as a CLI override (default 5) since the threshold is still a
    reasonable thing for the team to tune, but "match SepsisCalc's
    precedent" no longer motivates changing it away from 5.
A9. Sepsis is only detectable from data recorded during the ICU stay itself
    (chartevents/labevents/inputevents/outputevents are ICU-scoped tables).
    If suspicion of infection or the necessary SOFA lookback window falls
    mostly before ICU admission (e.g., antibiotics started in the ED), this
    pipeline cannot see that pre-ICU data and may under-detect onset in that
    window. This is a separate, unresolved visibility gap from the
    sepsis_within_4h_of_admission exclusion described in A10 below.

A10. [Team decision -- see A4 for the full baseline-convention history this
    depends on] excluded_reason = "sepsis_within_4h_of_admission" (renamed
    from the earlier "sepsis_at_admission", and re-derived, not just
    renamed -- the condition itself changed). A stay is excluded if
    sepsis_onset_time IS NOT NULL AND sepsis_onset_time <= intime + 4h. This
    fully REPLACES two earlier versions of this exclusion:
      (1) The original "sepsis_onset_time <= intime" check, which was
          structurally unsatisfiable (onset_time is always >= intime + 1h
          from the hourly grid) and was silently dead code -- confirmed via
          cohort_stats.json showing zero admissions with this reason.
      (2) A subsequent "suspicion_time <= intime AND hr=0 SOFA >= 2" check,
          built specifically to work around the fixed-0 baseline that was
          in place at the time (ASSUMPTION A4's mimic-code-aligned version).
    Both of those became moot once A4 was reverted to a measured hr=0
    baseline (team decision, matching Moor et al. / AI Gone Astray): with a
    real baseline, sepsis_onset_time is a well-defined, non-degenerate
    quantity even near admission, so the exclusion can test it directly
    instead of routing around baseline=0's degeneracy at hr=0.
    4h (rather than 6h) was chosen for internal consistency with this
    project's own locked 4-hour rolling prediction horizon
    (PROJECT_CONTEXT.md sec 5, SepsisCalc/SepsisLab). 6h precedent exists
    in both arXiv:2210.15056 (UnfoldML, MIMIC-III: "88.1% of sepsis onsets
    happened within the first 6 hours after ICU admission and are excluded
    from our study cohort") and arXiv:2511.08986 (a direct replication of
    the same Moor et al. / AI Gone Astray task setup this project cites,
    which also uses a 6h buffer on top of the same hr=0 baseline). Recorded
    here so a reviewer sees 4h was a considered choice against a real 6h
    alternative, not an arbitrary number.

A11. [SUPERSEDED -- see PB4 in _compute_hourly_sofa's docstring] Renal
    urine-output validity originally used a fixed `hr >= 18` proxy instead of
    mimic-code's actual 22-30h collection-window validity check. This has
    been replaced with a faithful port of urine_output_rate.sql: urine
    output is only trusted when the ACTUAL time covered by the trailing-24h
    window (uo_tm_24hr, summed inter-measurement gaps) is itself >= 24h, and
    the extrapolated volume matches sofa.sql's own
    `urineoutput_24hr / uo_tm_24hr * 24` formula exactly. Left here only so
    the history of the fix is visible; the current, accurate description
    lives in PB4.

A12. [Originally an inline code comment, now folded into PB3] GCS's
    previous-row carry-forward (the `b2` self-join in gcs.sql, looking back
    up to 6h when the current row's component is still missing) was
    initially skipped; this has been added as a faithful port. See PB3 in
    _compute_hourly_sofa's docstring.
================================================================================
"""
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import duckdb
import numpy as np
import pandas as pd

# ==============================================================================
# CONFIG / CONSTANTS
# ==============================================================================

# --- Cohort construction thresholds (PROJECT_CONTEXT.md sec 5 / task spec) ---
MIN_AGE_YEARS = 18
MIN_ICU_LOS_HOURS = 12.0
MIN_VALID_OBS_HOURS = 5

# --- Suspicion-of-infection windows (Seymour et al. Sepsis-3 convention,
#     matches mimic-code's suspicion_of_infection.sql and the task spec) ---
ANTIBIOTIC_TO_CULTURE_HOURS = 24  # culture must follow antibiotic within 24h
CULTURE_TO_ANTIBIOTIC_HOURS = 72  # antibiotic must follow culture within 72h

# --- Sepsis onset window relative to suspicion-of-infection time ---
SOFA_WINDOW_BEFORE_HOURS = 48
SOFA_WINDOW_AFTER_HOURS = 24
SOFA_DELTA_THRESHOLD = 2

# --- Antibiotic name substrings, mirrors mimic-code
#     mimic-iv/concepts/medication/antibiotic.sql (extracted programmatically
#     from that file, 154 entries) ---
ANTIBIOTIC_NAME_SUBSTRINGS = [
    "adoxa", "ala-tet", "alodox", "amikacin", "amikin", "amoxicill",
    "amphotericin", "anidulafungin", "ancef", "clavulanate", "ampicillin",
    "augmentin", "avelox", "avidoxy", "azactam", "azithromycin", "aztreonam",
    "axetil", "bactocill", "bactrim", "bactroban", "bethkis", "biaxin",
    "bicillin l-a", "cayston", "cefazolin", "cedax", "cefoxitin",
    "ceftazidime", "cefaclor", "cefadroxil", "cefdinir", "cefditoren",
    "cefepime", "cefotan", "cefotetan", "cefotaxime", "ceftaroline",
    "cefpodoxime", "cefpirome", "cefprozil", "ceftibuten", "ceftin",
    "ceftriaxone", "cefuroxime", "cephalexin", "cephalothin", "cephapririn",
    "chloramphenicol", "cipro", "ciprofloxacin", "claforan", "clarithromycin",
    "cleocin", "clindamycin", "cubicin", "dicloxacillin", "dirithromycin",
    "doryx", "doxycy", "duricef", "dynacin", "ery-tab", "eryped", "eryc",
    "erythrocin", "erythromycin", "factive", "flagyl", "fortaz", "furadantin",
    "garamycin", "gentamicin", "kanamycin", "keflex", "kefzol", "ketek",
    "levaquin", "levofloxacin", "lincocin", "linezolid", "macrobid",
    "macrodantin", "maxipime", "mefoxin", "metronidazole", "meropenem",
    "methicillin", "minocin", "minocycline", "monodox", "monurol", "morgidox",
    "moxatag", "moxifloxacin", "mupirocin", "myrac", "nafcillin", "neomycin",
    "nicazel doxy 30", "nitrofurantoin", "norfloxacin", "noroxin", "ocudox",
    "ofloxacin", "omnicef", "oracea", "oraxyl", "oxacillin", "pc pen vk",
    "pce dispertab", "panixine", "pediazole", "penicillin", "periostat",
    "pfizerpen", "piperacillin", "tazobactam", "primsol", "proquin",
    "raniclor", "rifadin", "rifampin", "rocephin", "smz-tmp", "septra",
    "septra ds", "solodyn", "spectracef", "streptomycin", "sulfadiazine",
    "sulfamethoxazole", "trimethoprim", "sulfatrim", "sulfisoxazole",
    "suprax", "synercid", "tazicef", "tetracycline", "timentin",
    "tobramycin", "unasyn", "vancocin", "vancomycin", "vantin", "vibativ",
    "vibra-tabs", "vibramycin", "zinacef", "zithromax", "zosyn", "zyvox",
]
# route/drug exclusions, mirrors mimic-code antibiotic.sql exactly
ANTIBIOTIC_ROUTE_EXCLUDE = ("OU", "OS", "OD", "AU", "AS", "AD", "TP")
ANTIBIOTIC_ROUTE_EXCLUDE_SUBSTR = ("ear", "eye")
ANTIBIOTIC_DRUG_EXCLUDE_SUBSTR = ("cream", "desensitization", "ophth oint", "gel")
# used only if --iv_antibiotics_only is passed (see ASSUMPTION A2)
IV_ROUTE_SUBSTR = ("iv", "intravenous")

# --- itemids (MIMIC-IV chartevents/labevents/inputevents/outputevents) ---
# labevents
ITEMID_PLATELET = 51265
ITEMID_BILIRUBIN_TOTAL = 50885
ITEMID_CREATININE = 50912
ITEMID_PO2 = 50821
ITEMID_SPECIMEN_TYPE = 52033  # value 'ART.' identifies arterial blood gas
ITEMID_FIO2_LABEVENTS = 50816  # bg.sql: FiO2 drawn as part of the ABG panel itself
# chartevents
ITEMID_FIO2_CHARTEVENTS = 223835
ITEMID_MBP = (220052, 220181, 225312)
ITEMID_GCS_MOTOR = 223901
ITEMID_GCS_VERBAL = 223900
ITEMID_GCS_EYES = 220739
ITEMID_VENT_MODE = 223849  # ventilator_setting.sql: ventilator_mode
ITEMID_VENT_MODE_HAMILTON = 229314  # ventilator_setting.sql: ventilator_mode_hamilton
ITEMID_O2_DEVICE = 226732  # oxygen_delivery.sql: o2 delivery device
# inputevents (vasopressor rate, uom already rate per mimic-code convention
# except norepinephrine, which has its own unit-bug correction -- see PB5)
ITEMID_NOREPINEPHRINE = 221906
ITEMID_EPINEPHRINE = 221289
ITEMID_DOPAMINE = 221662
ITEMID_DOBUTAMINE = 221653
# outputevents (urine output components, mirrors mimic-code urine_output.sql)
ITEMID_URINE_OUTPUT = (
    226559, 226560, 226561, 226584, 226563, 226564, 226565, 226567,
    226557, 226558, 227488, 227489,
)
ITEMID_URINE_OUTPUT_NEGATE = 227488  # GU irrigant out, value gets negated

FIO2_LOOKBACK_HOURS = 4  # bg.sql's stg_fio2 lookback window, unchanged
SOFA_ROLLING_WINDOW_HOURS = 24  # matches mimic-code

DEFAULT_TRAIN_FRAC = 0.70
DEFAULT_VAL_FRAC = 0.15
# test frac is the remainder


# ==============================================================================
# DATA LOADING
# ==============================================================================

# The four big MIMIC-IV event tables. CSV has no index, so every query that
# touches these re-scans the ENTIRE file from disk regardless of how many
# patients you filtered to -- --sample_size does NOT reduce this cost, since
# filtering only happens after the scan. Converting to Parquet once (cached
# to disk, reused on every future run) gives DuckDB per-column min/max stats
# it can use to skip whole row groups, and Parquet itself is far cheaper to
# re-parse than CSV text. This is normally the single biggest speedup
# available here -- see the performance note in main()'s --cache_dir help.
LARGE_EVENT_TABLES = ("chartevents", "labevents", "outputevents", "inputevents")


def _ensure_parquet_cache(view_name: str, csv_path: Path, cache_dir: Path) -> Path:
    """
    One-time conversion of a large CSV event table to Parquet, cached under
    cache_dir. Safe to call every run -- if the cached file already exists
    (and is newer than the source CSV) we skip straight to using it.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = cache_dir / f"{view_name}.parquet"
    if parquet_path.exists() and parquet_path.stat().st_mtime >= csv_path.stat().st_mtime:
        return parquet_path

    print(f"[label_sepsis3]   Building Parquet cache for {view_name} "
          f"(one-time cost, reused on every future run)...", file=sys.stderr)
    t0 = time.time()
    tmp_con = duckdb.connect(database=":memory:")
    tmp_con.execute("PRAGMA threads=4;")
    tmp_con.execute(f"""
        COPY (
            SELECT * FROM read_csv_auto('{csv_path.as_posix()}',
                ALL_VARCHAR=FALSE, IGNORE_ERRORS=TRUE)
        ) TO '{parquet_path.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD);
    """)
    tmp_con.close()
    print(f"[label_sepsis3]   -> {view_name}.parquet written in "
          f"{time.time() - t0:.0f}s", file=sys.stderr)
    return parquet_path


def connect_duckdb(hosp_dir: Path, icu_dir: Path, sample_size: Optional[int],
                    cache_dir: Optional[Path] = None) -> duckdb.DuckDBPyConnection:
    """
    Register the MIMIC-IV tables as DuckDB views so we can query them with
    SQL directly (per user environment note: prefer duckdb over loading full
    tables into pandas memory).

    If cache_dir is set (recommended -- this is the default in main()), the
    four large event tables (chartevents/labevents/outputevents/inputevents)
    are transparently converted to Parquet on first use and read from that
    cache on every subsequent run. This does NOT change any query results --
    it's purely a read-performance layer. Small dimension tables
    (admissions/patients/icustays/prescriptions/microbiologyevents) are read
    directly from CSV every time since they're cheap to scan.

    If sample_size is set, we still register full views for the small
    "dimension" tables (patients/admissions/icustays) but the actual event
    tables are still queried with hadm_id/stay_id filters at query time (see
    build_cohort/assign_labels) so that --sample_size gives a coherent
    subset -- it does NOT limit how much of the underlying file is scanned
    for CSV; the Parquet cache is what makes that scan cheap.
    """
    con = duckdb.connect(database=":memory:")
    con.execute("PRAGMA threads=4;")

    # datetime-like columns per table -- cast explicitly at view-registration
    # time so downstream SQL never hits a VARCHAR/TIMESTAMP mismatch (this can
    # happen on small/sparse real subsets or empty tables where DuckDB's type
    # sniffer falls back to VARCHAR for an all-null or all-empty column).
    DATETIME_COLS = {
        "admissions": ["admittime", "dischtime", "deathtime", "edregtime", "edouttime"],
        "patients": ["dod"],
        "prescriptions": ["starttime", "stoptime"],
        "microbiologyevents": ["chartdate", "charttime", "storedate", "storetime"],
        "labevents": ["charttime", "storetime"],
        "icustays": ["intime", "outtime"],
        "chartevents": ["charttime", "storetime"],
        "inputevents": ["starttime", "endtime", "storetime"],
        "outputevents": ["charttime", "storetime"],
    }

    def reg(view_name: str, path: Path):
        # ASSUMPTION A13 (added after team review): the user's environment
        # note describes plain, uncompressed .csv files, which is what this
        # function expects by default. If a plain .csv is missing but a
        # same-named .csv.gz exists, we transparently use that instead --
        # DuckDB's read_csv_auto handles gzip natively, no extra flags
        # needed. If neither exists, we fail loudly rather than silently
        # skipping a table.
        if not path.exists():
            gz_path = path.with_suffix(path.suffix + ".gz")
            if gz_path.exists():
                path = gz_path
            else:
                raise FileNotFoundError(
                    f"Expected MIMIC-IV file not found (checked both "
                    f"{path} and {gz_path})"
                )

        dt_cols = DATETIME_COLS.get(view_name, [])

        if cache_dir is not None and view_name in LARGE_EVENT_TABLES:
            parquet_path = _ensure_parquet_cache(view_name, path, cache_dir)
            replace_clause = ""
            if dt_cols:
                casts = ", ".join(f"CAST({c} AS TIMESTAMP) AS {c}" for c in dt_cols)
                replace_clause = f" REPLACE ({casts})"
            con.execute(
                f"CREATE OR REPLACE VIEW {view_name} AS "
                f"SELECT *{replace_clause} FROM read_parquet('{parquet_path.as_posix()}');"
            )
            return

        replace_clause = ""
        if dt_cols:
            casts = ", ".join(f"CAST({c} AS TIMESTAMP) AS {c}" for c in dt_cols)
            replace_clause = f" REPLACE ({casts})"
        con.execute(
            f"CREATE OR REPLACE VIEW {view_name} AS "
            f"SELECT *{replace_clause} FROM read_csv_auto('{path.as_posix()}', "
            f"ALL_VARCHAR=FALSE, IGNORE_ERRORS=TRUE);"
        )

    # hosp
    reg("admissions", hosp_dir / "admissions.csv")
    reg("patients", hosp_dir / "patients.csv")
    reg("prescriptions", hosp_dir / "prescriptions.csv")
    reg("microbiologyevents", hosp_dir / "microbiologyevents.csv")
    reg("labevents", hosp_dir / "labevents.csv")
    # icu
    reg("icustays", icu_dir / "icustays.csv")
    reg("chartevents", icu_dir / "chartevents.csv")
    reg("inputevents", icu_dir / "inputevents.csv")
    reg("outputevents", icu_dir / "outputevents.csv")

    return con


def _get_or_create_splits(con: duckdb.DuckDBPyConnection, subject_ids: pd.Series,
                           splits_path: Path, seed: int = 20240115) -> pd.DataFrame:
    """
    Split assignment at the subject_id level, generated ONCE and persisted.
    Per PROJECT_CONTEXT.md sec 5 / task Definition of Done: "generated once,
    never regenerated on reruns". If splits_path already exists, we load it
    and simply add any brand-new subject_ids (rare -- only if new subjects
    appear in a later MIMIC-IV refresh); we never touch existing assignments.
    """
    subject_ids = pd.Series(sorted(pd.unique(subject_ids)), name="subject_id")

    if splits_path.exists():
        existing = pd.read_parquet(splits_path)
        new_subjects = subject_ids[~subject_ids.isin(existing["subject_id"])]
        if len(new_subjects) == 0:
            return existing
        new_splits = _assign_splits_deterministic(new_subjects, seed=seed)
        combined = pd.concat([existing, new_splits], ignore_index=True)
        combined.to_parquet(splits_path, index=False)
        return combined

    splits_path.parent.mkdir(parents=True, exist_ok=True)
    new_splits = _assign_splits_deterministic(subject_ids, seed=seed)
    new_splits.to_parquet(splits_path, index=False)
    return new_splits


def _assign_splits_deterministic(subject_ids: pd.Series, seed: int) -> pd.DataFrame:
    """
    Deterministic subject-level split: hash each subject_id with a fixed seed
    so the assignment is reproducible independent of row order, and is stable
    even if this function is later re-run with a superset of subject_ids
    (each subject's split depends only on its own id + the seed, never on the
    rest of the batch). Uses hashlib (not Python's built-in hash()) so the
    assignment is stable across machines/processes, not just within one.
    """
    import hashlib

    def _bucket(sid: int) -> float:
        digest = hashlib.md5(f"{int(sid)}-{seed}".encode()).hexdigest()
        return (int(digest[:8], 16) % 10_000) / 10_000.0

    rng_vals = subject_ids.apply(_bucket)
    split = np.where(
        rng_vals < DEFAULT_TRAIN_FRAC, "train",
        np.where(rng_vals < DEFAULT_TRAIN_FRAC + DEFAULT_VAL_FRAC, "val", "test"),
    )
    return pd.DataFrame({"subject_id": subject_ids.values, "split": split})


# ==============================================================================
# STAGE A -- build_cohort()
# ==============================================================================

def build_cohort(con: duckdb.DuckDBPyConnection, splits_path: Path,
                  sample_size: Optional[int] = None,
                  min_valid_obs_hours: int = MIN_VALID_OBS_HOURS) -> pd.DataFrame:
    """
    Structural cohort filtering (age, first-ICU-stay, LOS, observation
    density). Does NOT yet know about sepsis timing -- the
    "sepsis_within_4h_of_admission" exclusion (which needs suspicion-of-
    infection + SOFA computed first) is applied in assign_labels(), see
    ASSUMPTION A3 in the module docstring for why these two exclusion
    families are split this way.

    Returns one row per subject_id (their chronologically first ICU stay),
    with excluded_reason populated for every subject that fails a structural
    filter -- nothing is silently dropped from the returned frame.
    """
    # ---- first ICU stay per subject, joined to admission/patient demographics ----
    first_stay = con.execute("""
        WITH ranked AS (
            SELECT
                ie.subject_id,
                ie.hadm_id,
                ie.stay_id,
                ie.intime,
                ie.outtime,
                a.admittime,
                a.dischtime,
                a.deathtime,
                p.anchor_age,
                p.anchor_year,
                ROW_NUMBER() OVER (
                    PARTITION BY ie.subject_id ORDER BY ie.intime ASC
                ) AS stay_rank
            FROM icustays AS ie
            INNER JOIN admissions AS a
                ON ie.hadm_id = a.hadm_id
            INNER JOIN patients AS p
                ON ie.subject_id = p.subject_id
        )
        SELECT *
        FROM ranked
        WHERE stay_rank = 1
        ORDER BY subject_id
    """).df()

    if sample_size is not None:
        # BUG FIX (found after review -- real, confirmed reproducibility bug):
        # the query above previously had no ORDER BY. DuckDB does not
        # guarantee row order for an unordered result set, especially with
        # PRAGMA threads=4 running the scan/join in parallel -- so the same
        # query, on the same unchanged data, could return rows in a
        # different order on every run. pandas' DataFrame.sample(random_state
        # =0) samples by ROW POSITION, not by row content/identity, so a
        # different incoming row order silently produces a DIFFERENT set of
        # 500 sampled patients each time, even with a fixed seed -- defeating
        # the entire point of random_state=0. Confirmed as the explanation
        # for two --sample_size 500 runs (same command, same data) producing
        # different exclusion counts, positive rates, and split composition.
        # Fix: explicit `ORDER BY subject_id` above makes the row order (and
        # therefore the sample) deterministic and reproducible across runs.
        first_stay = first_stay.sample(
            n=min(sample_size, len(first_stay)), random_state=0
        ).reset_index(drop=True)

    # ---- age at admission (MIMIC-IV convention: anchor_age applies in anchor_year) ----
    first_stay["intime"] = pd.to_datetime(first_stay["intime"])
    first_stay["outtime"] = pd.to_datetime(first_stay["outtime"])
    first_stay["admittime"] = pd.to_datetime(first_stay["admittime"])
    first_stay["dischtime"] = pd.to_datetime(first_stay["dischtime"])
    first_stay["age_at_admission"] = first_stay["anchor_age"] + (
        first_stay["admittime"].dt.year - first_stay["anchor_year"]
    )
    first_stay["los_hours"] = (
        first_stay["outtime"] - first_stay["intime"]
    ).dt.total_seconds() / 3600.0

    # ---- valid hourly observation density (chartevents OR labevents) ----
    stay_ids = first_stay["stay_id"].dropna().unique().tolist()
    hadm_ids = first_stay["hadm_id"].dropna().unique().tolist()
    obs_density = _compute_observation_density(con, stay_ids, hadm_ids)
    first_stay = first_stay.merge(obs_density, on="stay_id", how="left")
    first_stay["n_valid_obs_hours"] = first_stay["n_valid_obs_hours"].fillna(0).astype(int)

    # ---- structural exclusions (order matters only for the reported reason;
    #      a subject can only carry ONE excluded_reason) ----
    def reason(row) -> Optional[str]:
        if row["age_at_admission"] < MIN_AGE_YEARS:
            return "age_under_18"
        if row["los_hours"] < MIN_ICU_LOS_HOURS:
            return "los_under_12h"
        if row["n_valid_obs_hours"] < min_valid_obs_hours:
            return "insufficient_observation_density"
        return None

    first_stay["excluded_reason"] = first_stay.apply(reason, axis=1)

    # ---- split assignment, subject_id level, persisted once ----
    splits = _get_or_create_splits(con, first_stay["subject_id"], splits_path)
    first_stay = first_stay.merge(splits, on="subject_id", how="left")

    cols = [
        "subject_id", "hadm_id", "stay_id", "intime", "outtime",
        "admittime", "dischtime", "deathtime", "age_at_admission",
        "los_hours", "n_valid_obs_hours", "excluded_reason", "split",
    ]
    return first_stay[cols].reset_index(drop=True)


def _compute_observation_density(con: duckdb.DuckDBPyConnection, stay_ids: list,
                                  hadm_ids: list) -> pd.DataFrame:
    """
    Count distinct hours-since-ICU-admission that have >=1 chartevents OR
    labevents observation, per stay. Used for the ">=5 valid hourly
    observation timepoints" exclusion (ASSUMPTION A8).
    """
    if not stay_ids:
        return pd.DataFrame({"stay_id": [], "n_valid_obs_hours": []})

    stay_ids_sql = ",".join(str(int(s)) for s in stay_ids)
    hadm_ids_sql = ",".join(str(int(h)) for h in hadm_ids) if hadm_ids else "NULL"

    df = con.execute(f"""
        WITH ce_hours AS (
            SELECT stay_id,
                   DATE_TRUNC('hour', CAST(charttime AS TIMESTAMP)) AS obs_hour
            FROM chartevents
            WHERE stay_id IN ({stay_ids_sql})
        ), le_hours AS (
            SELECT ie.stay_id,
                   DATE_TRUNC('hour', CAST(le.charttime AS TIMESTAMP)) AS obs_hour
            FROM labevents AS le
            INNER JOIN icustays AS ie
                ON le.hadm_id = ie.hadm_id
                AND le.charttime >= ie.intime
                AND le.charttime <= ie.outtime
            WHERE ie.stay_id IN ({stay_ids_sql})
                AND le.hadm_id IN ({hadm_ids_sql})
        ), unioned AS (
            SELECT stay_id, obs_hour FROM ce_hours
            UNION
            SELECT stay_id, obs_hour FROM le_hours
        )
        SELECT stay_id, COUNT(DISTINCT obs_hour) AS n_valid_obs_hours
        FROM unioned
        GROUP BY stay_id
    """).df()
    return df


# ==============================================================================
# STAGE B -- assign_labels()
# ==============================================================================

def _antibiotic_name_predicate(column: str = "drug") -> str:
    clauses = " OR ".join(
        f"LOWER({column}) LIKE '%{name}%'" for name in ANTIBIOTIC_NAME_SUBSTRINGS
    )
    return f"({clauses})"


def _compute_suspicion_of_infection(con: duckdb.DuckDBPyConnection,
                                     hadm_ids: list,
                                     iv_only: bool, blood_only: bool) -> pd.DataFrame:
    """
    Mirrors mimic-code's suspicion_of_infection.sql / alistairewj's
    suspicion-of-infection.sql: for every antibiotic administration, look for
    a qualifying culture either up to 72h BEFORE it or up to 24h AFTER it;
    the earlier of (culture time, antibiotic time) is the suspicion time for
    that antibiotic event. We then take the EARLIEST qualifying
    suspicion_infection_time per admission as "first suspicion of infection".

    See ASSUMPTION A1 (blood vs any culture) and A2 (IV vs any route) in the
    module docstring.
    """
    if not hadm_ids:
        return pd.DataFrame({
            "hadm_id": [], "suspicion_time": [], "antibiotic_time": [],
            "culture_time": [],
        })
    hadm_ids_sql = ",".join(str(int(h)) for h in hadm_ids)

    abx_predicate = _antibiotic_name_predicate("drug")
    route_exclude = ", ".join(f"'{r}'" for r in ANTIBIOTIC_ROUTE_EXCLUDE)
    route_substr_exclude = " AND ".join(
        f"LOWER(route) NOT LIKE '%{s}%'" for s in ANTIBIOTIC_ROUTE_EXCLUDE_SUBSTR
    )
    drug_substr_exclude = " AND ".join(
        f"LOWER(drug) NOT LIKE '%{s}%'" for s in ANTIBIOTIC_DRUG_EXCLUDE_SUBSTR
    )
    iv_filter = ""
    if iv_only:
        iv_clause = " OR ".join(f"LOWER(route) LIKE '%{s}%'" for s in IV_ROUTE_SUBSTR)
        iv_filter = f"AND ({iv_clause})"

    query = f"""
        WITH abx AS (
            SELECT
                subject_id, hadm_id,
                CAST(starttime AS TIMESTAMP) AS antibiotic_time,
                ROW_NUMBER() OVER (
                    PARTITION BY hadm_id
                    ORDER BY starttime NULLS LAST
                ) AS ab_id
            FROM prescriptions
            WHERE hadm_id IN ({hadm_ids_sql})
                AND {abx_predicate}
                AND NOT route IN ({route_exclude})
                AND {route_substr_exclude}
                AND {drug_substr_exclude}
                {iv_filter}
        ), micro AS (
            SELECT
                hadm_id,
                micro_specimen_id,
                MAX(spec_type_desc) AS spec_type_desc,
                COALESCE(MAX(charttime), CAST(MAX(chartdate) AS TIMESTAMP)) AS culture_time
            FROM microbiologyevents
            WHERE hadm_id IN ({hadm_ids_sql})
            GROUP BY hadm_id, micro_specimen_id
            {("HAVING UPPER(MAX(spec_type_desc)) LIKE '%BLOOD%'" if blood_only else "")}
        ), me_before_ab AS (
            -- culture up to 72h BEFORE antibiotic
            SELECT
                abx.hadm_id, abx.ab_id,
                MIN(micro.culture_time) AS culture_time
            FROM abx
            INNER JOIN micro
                ON abx.hadm_id = micro.hadm_id
                AND micro.culture_time <= abx.antibiotic_time
                AND micro.culture_time > abx.antibiotic_time - INTERVAL '{CULTURE_TO_ANTIBIOTIC_HOURS}' HOUR
            GROUP BY abx.hadm_id, abx.ab_id
        ), me_after_ab AS (
            -- culture up to 24h AFTER antibiotic
            SELECT
                abx.hadm_id, abx.ab_id,
                MIN(micro.culture_time) AS culture_time
            FROM abx
            INNER JOIN micro
                ON abx.hadm_id = micro.hadm_id
                AND micro.culture_time > abx.antibiotic_time
                AND micro.culture_time <= abx.antibiotic_time + INTERVAL '{ANTIBIOTIC_TO_CULTURE_HOURS}' HOUR
            GROUP BY abx.hadm_id, abx.ab_id
        )
        SELECT
            abx.hadm_id,
            abx.antibiotic_time,
            COALESCE(before.culture_time, after.culture_time) AS culture_time,
            CASE
                WHEN before.culture_time IS NOT NULL THEN before.culture_time
                WHEN after.culture_time IS NOT NULL THEN abx.antibiotic_time
                ELSE NULL
            END AS suspicion_time
        FROM abx
        LEFT JOIN me_before_ab AS before
            ON abx.hadm_id = before.hadm_id AND abx.ab_id = before.ab_id
        LEFT JOIN me_after_ab AS after
            ON abx.hadm_id = after.hadm_id AND abx.ab_id = after.ab_id
    """
    df = con.execute(query).df()
    df = df.dropna(subset=["suspicion_time"])
    if df.empty:
        return pd.DataFrame({
            "hadm_id": [], "suspicion_time": [], "antibiotic_time": [],
            "culture_time": [],
        })
    # first qualifying suspicion of infection per admission
    df["suspicion_time"] = pd.to_datetime(df["suspicion_time"])
    first_susp = (
        df.sort_values("suspicion_time")
        .groupby("hadm_id", as_index=False)
        .first()
    )
    return first_susp[["hadm_id", "suspicion_time", "antibiotic_time", "culture_time"]]


def _compute_hourly_sofa(con: duckdb.DuckDBPyConnection, stay_map: pd.DataFrame) -> pd.DataFrame:
    """
    Compute an hourly total-SOFA trajectory per ICU stay.

    stay_map: dataframe with columns [stay_id, hadm_id, subject_id, intime, outtime]
    for the structurally-eligible candidates only (keeps this query scoped and
    fast rather than running over the entire ICU database).

    Component thresholds are copied verbatim from mimic-code's
    concepts_duckdb/score/sofa.sql (respiration/coagulation/liver/
    cardiovascular/cns/renal CASE WHEN blocks). Each component is rolled up
    over a trailing 24h window (MAX = worst value), then coalesced to 0 if
    nothing was ever observed in that window -- this is how "missing
    component -> assume healthy" (task spec) is operationalized; see
    ASSUMPTION A7.

    NEW ASSUMPTIONS SECTION (this function was rewritten as a faithful port
    of mimic-code's actual SQL, replacing the earlier reconstruction-from-
    description approach -- per team review after the A4 baseline bug).
    Source files ported (from MIT-LCP/mimic-code, concepts_duckdb, the
    DuckDB-native variant of the current repo -- NOT the deprecated
    mimic-iv legacy path, and NOT the pivoted/concepts_postgres pre-binned
    tables, which we deliberately do not use):
      - measurement/bg.sql            (PaO2/FiO2 pairing)
      - measurement/ventilator_setting.sql, measurement/oxygen_delivery.sql,
        treatment/ventilation.sql     (ventilation status state machine)
      - measurement/gcs.sql           (GCS with carry-forward)
      - measurement/urine_output.sql, measurement/urine_output_rate.sql
                                       (24h urine-output validity window)
      - medication/norepinephrine.sql (rate unit-bug correction)
      - score/sofa.sql                (component thresholds -- unchanged
                                        from the original port, already
                                        verbatim)

    PB1. [Faithful port] FiO2 for the PaO2/FiO2 ratio now matches bg.sql
        exactly: prefer a directly-drawn labevents FiO2 (itemid 50816, with
        bg.sql's exact fraction-vs-percent normalization: 0.2-1.0 treated as
        a fraction and multiplied by 100, 20-100 treated as already a
        percent, everything else discarded), falling back to the nearest
        PRECEDING chartevents FiO2 (itemid 223835, same normalization,
        4h lookback) only if no lab FiO2 exists. This replaces the earlier
        chartevents-only, no-normalization version (formerly ASSUMPTION A6,
        now resolved).

    PB2. [Faithful port] Ventilation status is now the actual mimic-code
        state machine (ventilator_setting + oxygen_delivery -> per-charttime
        status classification -> episode building with the 14h continuity
        gap rule), not the +/-6h "any Ventilator Mode chartevent nearby"
        proxy used previously (formerly ASSUMPTION A5, now resolved). A
        PaO2/FiO2 reading counts as "on InvasiveVent" only if it falls
        within an actual InvasiveVent episode's [starttime, endtime].
        RESIDUAL SIMPLIFICATION: mimic-code's oxygen_delivery.sql resolves
        same-timestamp duplicate device readings via
        `ROW_NUMBER() ... ORDER BY storetime DESC, value DESC` (deterministic
        tie-break on storetime then device-name string) to pick
        o2_delivery_device_1..4; this port uses DuckDB's
        `ROW_NUMBER() ... ORDER BY storetime DESC NULLS LAST, value DESC`
        which is logically the same ordering but NULL storetime handling in
        DuckDB vs Postgres can differ at the margin for the rare row with no
        storetime. Not expected to materially change results.

    PB3. [Faithful port] GCS now includes the previous-row carry-forward
        (the `b2` self-join in gcs.sql, looking back up to 6h for a still-
        missing component) in addition to the current-row "verbal=0 -> 15"
        rule already ported correctly before (formerly ASSUMPTION A12's
        noted gap, now resolved).

    PB4. [Faithful port] Urine output validity now matches
        urine_output_rate.sql: a trailing 24h window's urineoutput_24hr is
        only used if the actual TIME COVERED by that window (uo_tm_24hr,
        summed inter-measurement gaps, in hours) is >= 24h -- not the
        earlier `hr >= 18` fixed-offset proxy (formerly ASSUMPTION A11, now
        resolved). Extrapolated volume is urineoutput_24hr / uo_tm_24hr * 24,
        matching sofa.sql's own `uo.urineoutput_24hr / uo.uo_tm_24hr * 24`
        exactly. NOTE: mimic-code's uo_mlkghr_* (per-kg-weight rate) columns
        are NOT ported -- sofa.sql itself never uses them (it uses the raw
        extrapolated volume, not a weight-normalized rate), so
        weight_durations was never a real dependency for OUR purposes and is
        correctly omitted.

    PB5. [Faithful port] Norepinephrine rate now includes the exact unit-bug
        correction from medication/norepinephrine.sql: if rateuom is
        'mg/kg/min', the rate is treated as already being in the intended
        unit if patientweight = 1 (a data-entry sentinel), otherwise
        multiplied by 1000. Epinephrine/dopamine/dobutamine have NO such
        conversion in mimic-code (verified directly -- their .sql files use
        `rate AS vaso_rate` with no CASE WHEN at all), so those three are
        unchanged from the original port.

    PB6. [Deliberately NOT ported -- explicit instruction] The hourly grid
        itself (icustay_hourly.sql / icustay_times.sql, which round to
        whole-hour boundaries anchored on heart-rate charting and use a
        [-24, N] hour range) is intentionally NOT ported. This file
        continues to use its own GENERATE_SERIES(0, N) grid anchored on raw
        ICU intime, per explicit team instruction to keep this part as-is.
        This means hour boundaries here will not line up exactly with
        mimic-code's own hour boundaries at the margins (e.g. a component
        value falling just before/after an hour edge could be bucketed
        differently). Not expected to matter for the >=2-point SOFA delta
        test at the resolution we care about, but noted for completeness.

    PB7. [Minor, not separately fixed] mimic-code's vitalsign.sql AVERAGES
        multiple simultaneous MBP readings at the exact same charttime
        before sofa.sql takes MIN across the hour; this port takes MIN
        directly across all raw MBP readings in the hour without first
        averaging same-timestamp duplicates. Affects only the rare case of
        two MBP source devices (e.g. arterial line + non-invasive cuff)
        charted at the identical timestamp; expected impact is negligible
        and this was not considered worth the added complexity to fix.
    """
    if stay_map.empty:
        return pd.DataFrame(columns=["stay_id", "hr", "hour_start", "sofa_total"])

    stay_ids = stay_map["stay_id"].astype(int).tolist()
    hadm_ids = stay_map["hadm_id"].astype(int).tolist()
    stay_ids_sql = ",".join(str(s) for s in stay_ids)
    hadm_ids_sql = ",".join(str(h) for h in hadm_ids)
    mbp_ids_sql = ",".join(str(i) for i in ITEMID_MBP)
    urine_ids_sql = ",".join(str(i) for i in ITEMID_URINE_OUTPUT)

    # register the stay_map (with intime/outtime) as a duckdb view for the hourly grid
    con.register("stay_map_tbl", stay_map[["stay_id", "hadm_id", "subject_id", "intime", "outtime"]])

    query = f"""
        WITH hourly_grid AS (
            -- one row per (stay_id, hr) from intime to outtime -- OUR OWN
            -- grid mechanism, deliberately kept as-is (see PB6)
            SELECT
                s.stay_id, s.hadm_id,
                gs.hr,
                s.intime + (gs.hr * INTERVAL '1' HOUR) AS hour_start,
                s.intime + ((gs.hr + 1) * INTERVAL '1' HOUR) AS hour_end
            FROM stay_map_tbl AS s,
                 LATERAL (
                    SELECT UNNEST(GENERATE_SERIES(
                        0, CAST(GREATEST(DATEDIFF('hour', s.intime, s.outtime), 0) AS INTEGER)
                    )) AS hr
                 ) AS gs
        ),
        -- ================= RESPIRATION: PaO2/FiO2 (PB1, PB2) =================
        abg_art AS (
            -- arterial PaO2 readings, mirrors bg.sql's specimen='ART.' filter
            SELECT
                le.hadm_id, le.specimen_id, le.charttime,
                MAX(CASE WHEN le.itemid = {ITEMID_PO2} THEN le.valuenum END) AS po2,
                MAX(CASE WHEN le.itemid = {ITEMID_SPECIMEN_TYPE} THEN le.value END) AS specimen
            FROM labevents AS le
            WHERE le.hadm_id IN ({hadm_ids_sql})
                AND le.itemid IN ({ITEMID_PO2}, {ITEMID_SPECIMEN_TYPE}, {ITEMID_FIO2_LABEVENTS})
            GROUP BY le.hadm_id, le.specimen_id, le.charttime
            HAVING MAX(CASE WHEN le.itemid = {ITEMID_SPECIMEN_TYPE} THEN le.value END) = 'ART.'
                AND MAX(CASE WHEN le.itemid = {ITEMID_PO2} THEN le.valuenum END) IS NOT NULL
        ),
        fio2_lab AS (
            -- FiO2 drawn directly as part of the same blood-gas panel (bg.sql
            -- itemid 50816), with bg.sql's exact fraction/percent normalization
            SELECT
                le.hadm_id, le.specimen_id,
                CASE
                    WHEN le.valuenum > 20 AND le.valuenum <= 100 THEN le.valuenum
                    WHEN le.valuenum > 0.2 AND le.valuenum <= 1.0 THEN le.valuenum * 100.0
                    ELSE NULL
                END AS fio2
            FROM labevents AS le
            WHERE le.hadm_id IN ({hadm_ids_sql}) AND le.itemid = {ITEMID_FIO2_LABEVENTS}
        ),
        fio2_ce AS (
            -- fallback: nearest chartevents FiO2 (itemid 223835), same
            -- normalization as bg.sql's stg_fio2 CTE
            SELECT
                ce.stay_id, ce.charttime,
                CASE
                    WHEN ce.valuenum >= 20 AND ce.valuenum <= 100 THEN ce.valuenum
                    WHEN ce.valuenum > 0.2 AND ce.valuenum <= 1 THEN ce.valuenum * 100
                    ELSE NULL
                END AS fio2_chartevents
            FROM chartevents AS ce
            WHERE ce.stay_id IN ({stay_ids_sql})
                AND ce.itemid = {ITEMID_FIO2_CHARTEVENTS}
                AND ce.valuenum > 0 AND ce.valuenum <= 100
        ),
        -- ---- ventilation status state machine (PB2) ----
        vent_setting_raw AS (
            SELECT
                ce.stay_id, ce.charttime,
                MAX(CASE WHEN ce.itemid = {ITEMID_VENT_MODE} THEN CAST(ce.value AS VARCHAR) END) AS ventilator_mode,
                MAX(CASE WHEN ce.itemid = {ITEMID_VENT_MODE_HAMILTON} THEN CAST(ce.value AS VARCHAR) END) AS ventilator_mode_hamilton
            FROM chartevents AS ce
            WHERE ce.stay_id IN ({stay_ids_sql})
                AND ce.itemid IN ({ITEMID_VENT_MODE}, {ITEMID_VENT_MODE_HAMILTON})
                AND ce.value IS NOT NULL
            GROUP BY ce.stay_id, ce.charttime
        ),
        o2_device_raw AS (
            SELECT
                ce.stay_id, ce.charttime, CAST(ce.value AS VARCHAR) AS o2_device,
                ROW_NUMBER() OVER (
                    PARTITION BY ce.stay_id, ce.charttime
                    ORDER BY ce.storetime DESC NULLS LAST, CAST(ce.value AS VARCHAR) DESC
                ) AS rn
            FROM chartevents AS ce
            WHERE ce.stay_id IN ({stay_ids_sql})
                AND ce.itemid = {ITEMID_O2_DEVICE}
                AND ce.value IS NOT NULL
        ),
        o2_device_pivot AS (
            SELECT
                stay_id, charttime,
                MAX(CASE WHEN rn = 1 THEN o2_device END) AS o2_delivery_device_1
            FROM o2_device_raw
            GROUP BY stay_id, charttime
        ),
        vent_status_raw AS (
            SELECT
                tm.stay_id, tm.charttime,
                o2.o2_delivery_device_1,
                CASE
                    WHEN o2.o2_delivery_device_1 IN ('Tracheostomy tube', 'Trach mask ')
                        THEN 'Tracheostomy'
                    WHEN o2.o2_delivery_device_1 IN ('Endotracheal tube')
                        OR vs.ventilator_mode IN (
                            '(S) CMV','APRV','APRV/Biphasic+ApnPress','APRV/Biphasic+ApnVol',
                            'APV (cmv)','Ambient','Apnea Ventilation','CMV','CMV/ASSIST',
                            'CMV/ASSIST/AutoFlow','CMV/AutoFlow','CPAP/PPS','CPAP/PSV',
                            'CPAP/PSV+Apn TCPL','CPAP/PSV+ApnPres','CPAP/PSV+ApnVol','MMV',
                            'MMV/AutoFlow','MMV/PSV','MMV/PSV/AutoFlow','P-CMV','PCV+',
                            'PCV+/PSV','PCV+Assist','PRES/AC','PRVC/AC','PRVC/SIMV','PSV/SBT',
                            'SIMV','SIMV/AutoFlow','SIMV/PRES','SIMV/PSV','SIMV/PSV/AutoFlow',
                            'SIMV/VOL','SYNCHRON MASTER','SYNCHRON SLAVE','VOL/AC'
                        )
                        OR vs.ventilator_mode_hamilton IN (
                            'APRV','APV (cmv)','Ambient','(S) CMV','P-CMV','SIMV',
                            'APV (simv)','P-SIMV','VS','ASV'
                        )
                        THEN 'InvasiveVent'
                    WHEN o2.o2_delivery_device_1 IN ('Bipap mask ', 'CPAP mask ')
                        OR vs.ventilator_mode_hamilton IN ('DuoPaP', 'NIV', 'NIV-ST')
                        THEN 'NonInvasiveVent'
                    WHEN o2.o2_delivery_device_1 IN ('High flow nasal cannula')
                        THEN 'HFNC'
                    WHEN o2.o2_delivery_device_1 IN (
                            'Non-rebreather','Face tent','Aerosol-cool','Venti mask ',
                            'Medium conc mask ','Ultrasonic neb','Vapomist','Oxymizer',
                            'High flow neb','Nasal cannula'
                        )
                        THEN 'SupplementalOxygen'
                    WHEN o2.o2_delivery_device_1 = 'None' THEN 'None'
                    ELSE NULL
                END AS ventilation_status
            FROM (
                SELECT stay_id, charttime FROM vent_setting_raw
                UNION
                SELECT stay_id, charttime FROM o2_device_pivot
            ) AS tm
            LEFT JOIN vent_setting_raw AS vs ON tm.stay_id = vs.stay_id AND tm.charttime = vs.charttime
            LEFT JOIN o2_device_pivot AS o2 ON tm.stay_id = o2.stay_id AND tm.charttime = o2.charttime
        ),
        vent_episodes_stg AS (
            SELECT
                stay_id, charttime, ventilation_status,
                LAG(charttime) OVER w AS charttime_lag,
                LEAD(charttime) OVER w AS charttime_lead,
                LAG(ventilation_status) OVER w AS ventilation_status_lag
            FROM vent_status_raw
            WHERE ventilation_status IS NOT NULL
            WINDOW w AS (PARTITION BY stay_id ORDER BY charttime NULLS FIRST)
        ),
        vent_episodes_flagged AS (
            SELECT
                *,
                CASE
                    WHEN ventilation_status_lag IS NULL THEN 1
                    WHEN DATEDIFF('hour', charttime_lag, charttime) >= 14 THEN 1
                    WHEN ventilation_status_lag <> ventilation_status THEN 1
                    ELSE 0
                END AS new_ventilation_event
            FROM vent_episodes_stg
        ),
        vent_episodes_seq AS (
            SELECT
                *,
                SUM(new_ventilation_event) OVER (
                    PARTITION BY stay_id ORDER BY charttime NULLS FIRST
                ) AS vent_seq
            FROM vent_episodes_flagged
        ),
        vent_episodes AS (
            SELECT
                stay_id,
                MIN(charttime) AS starttime,
                MAX(
                    CASE
                        WHEN charttime_lead IS NULL OR DATEDIFF('hour', charttime, charttime_lead) >= 14
                        THEN charttime ELSE charttime_lead
                    END
                ) AS endtime,
                MAX(ventilation_status) AS ventilation_status
            FROM vent_episodes_seq
            GROUP BY stay_id, vent_seq
            HAVING MIN(charttime) <> MAX(charttime)
        ),
        pf_ratio AS (
            SELECT
                s.stay_id,
                abg_art.charttime,
                abg_art.po2,
                COALESCE(fl.fio2, fc.fio2_chartevents) AS fio2_resolved,
                CASE
                    WHEN fl.fio2 IS NOT NULL THEN 100.0 * abg_art.po2 / fl.fio2
                    WHEN fc.fio2_chartevents IS NOT NULL THEN 100.0 * abg_art.po2 / fc.fio2_chartevents
                    ELSE NULL
                END AS pao2fio2ratio,
                CASE WHEN ve.stay_id IS NOT NULL THEN TRUE ELSE FALSE END AS on_vent
            FROM abg_art
            INNER JOIN stay_map_tbl AS s ON abg_art.hadm_id = s.hadm_id
            LEFT JOIN fio2_lab AS fl ON abg_art.specimen_id = fl.specimen_id
            LEFT JOIN LATERAL (
                SELECT fio2_ce.fio2_chartevents
                FROM fio2_ce
                WHERE fio2_ce.stay_id = s.stay_id
                    AND fio2_ce.charttime <= abg_art.charttime
                    AND fio2_ce.charttime > abg_art.charttime - INTERVAL '{FIO2_LOOKBACK_HOURS}' HOUR
                ORDER BY fio2_ce.charttime DESC
                LIMIT 1
            ) AS fc ON TRUE
            LEFT JOIN vent_episodes AS ve
                ON ve.stay_id = s.stay_id
                AND abg_art.charttime >= ve.starttime
                AND abg_art.charttime <= ve.endtime
                AND ve.ventilation_status = 'InvasiveVent'
        ),
        -- ================= COAGULATION / LIVER / RENAL-creatinine (unchanged, already verbatim) =================
        platelet AS (
            SELECT le.hadm_id, le.charttime, le.valuenum AS platelet
            FROM labevents AS le
            WHERE le.hadm_id IN ({hadm_ids_sql}) AND le.itemid = {ITEMID_PLATELET}
        ),
        bili AS (
            SELECT le.hadm_id, le.charttime, le.valuenum AS bilirubin
            FROM labevents AS le
            WHERE le.hadm_id IN ({hadm_ids_sql}) AND le.itemid = {ITEMID_BILIRUBIN_TOTAL}
        ),
        creat AS (
            SELECT le.hadm_id, le.charttime, le.valuenum AS creatinine
            FROM labevents AS le
            WHERE le.hadm_id IN ({hadm_ids_sql}) AND le.itemid = {ITEMID_CREATININE}
        ),
        -- ================= CARDIOVASCULAR: MAP + vasopressors (PB5, PB7) =================
        mbp AS (
            SELECT ce.stay_id, ce.charttime, ce.valuenum AS mbp
            FROM chartevents AS ce
            WHERE ce.stay_id IN ({stay_ids_sql})
                AND ce.itemid IN ({mbp_ids_sql})
                AND ce.valuenum > 0 AND ce.valuenum < 300
        ),
        vaso AS (
            SELECT
                ie.stay_id,
                CAST(ie.starttime AS TIMESTAMP) AS starttime,
                CAST(ie.endtime AS TIMESTAMP) AS endtime,
                MAX(
                    CASE WHEN ie.itemid = {ITEMID_NOREPINEPHRINE} THEN
                        -- PB5: norepinephrine-specific unit-bug correction,
                        -- verbatim from medication/norepinephrine.sql. The
                        -- other three pressors have NO such correction in
                        -- mimic-code (verified directly).
                        CASE
                            WHEN CAST(ie.rateuom AS VARCHAR) = 'mg/kg/min' AND CAST(ie.patientweight AS DOUBLE) = 1 THEN CAST(ie.rate AS DOUBLE)
                            WHEN CAST(ie.rateuom AS VARCHAR) = 'mg/kg/min' THEN CAST(ie.rate AS DOUBLE) * 1000.0
                            ELSE CAST(ie.rate AS DOUBLE)
                        END
                    END
                ) AS rate_norepi,
                MAX(CASE WHEN ie.itemid = {ITEMID_EPINEPHRINE} THEN CAST(ie.rate AS DOUBLE) END) AS rate_epi,
                MAX(CASE WHEN ie.itemid = {ITEMID_DOPAMINE} THEN CAST(ie.rate AS DOUBLE) END) AS rate_dopa,
                MAX(CASE WHEN ie.itemid = {ITEMID_DOBUTAMINE} THEN CAST(ie.rate AS DOUBLE) END) AS rate_dobu
            FROM inputevents AS ie
            WHERE ie.stay_id IN ({stay_ids_sql})
                AND ie.itemid IN ({ITEMID_NOREPINEPHRINE}, {ITEMID_EPINEPHRINE}, {ITEMID_DOPAMINE}, {ITEMID_DOBUTAMINE})
            GROUP BY ie.stay_id, ie.starttime, ie.endtime
        ),
        -- ================= CNS: GCS with carry-forward (PB3) =================
        gcs_raw AS (
            SELECT
                ce.stay_id, ce.charttime,
                MAX(CASE WHEN ce.itemid = {ITEMID_GCS_MOTOR} THEN ce.valuenum END) AS gcs_motor,
                MAX(CASE WHEN ce.itemid = {ITEMID_GCS_VERBAL} AND CAST(ce.value AS VARCHAR) = 'No Response-ETT' THEN 0
                         WHEN ce.itemid = {ITEMID_GCS_VERBAL} THEN ce.valuenum END) AS gcs_verbal,
                MAX(CASE WHEN ce.itemid = {ITEMID_GCS_EYES} THEN ce.valuenum END) AS gcs_eyes,
                ROW_NUMBER() OVER (PARTITION BY ce.stay_id ORDER BY ce.charttime ASC NULLS FIRST) AS rn
            FROM chartevents AS ce
            WHERE ce.stay_id IN ({stay_ids_sql})
                AND ce.itemid IN ({ITEMID_GCS_MOTOR}, {ITEMID_GCS_VERBAL}, {ITEMID_GCS_EYES})
            GROUP BY ce.stay_id, ce.charttime
        ),
        gcs_with_prev AS (
            -- PB3: the b2 self-join from gcs.sql -- previous row's components,
            -- only used if that previous row is within 6h
            SELECT
                b.stay_id, b.charttime, b.gcs_motor, b.gcs_verbal, b.gcs_eyes,
                b2.gcs_verbal AS gcs_verbal_prev,
                b2.gcs_motor AS gcs_motor_prev,
                b2.gcs_eyes AS gcs_eyes_prev
            FROM gcs_raw AS b
            LEFT JOIN gcs_raw AS b2
                ON b.stay_id = b2.stay_id
                AND b.rn = b2.rn + 1
                AND b2.charttime > b.charttime - INTERVAL '6' HOUR
        ),
        gcs AS (
            SELECT
                stay_id, charttime,
                CASE
                    WHEN gcs_verbal = 0 THEN 15
                    WHEN gcs_verbal IS NULL AND gcs_verbal_prev = 0 THEN 15
                    WHEN gcs_verbal_prev = 0
                        THEN COALESCE(gcs_motor, 6) + COALESCE(gcs_verbal, 5) + COALESCE(gcs_eyes, 4)
                    ELSE
                        COALESCE(gcs_motor, COALESCE(gcs_motor_prev, 6))
                        + COALESCE(gcs_verbal, COALESCE(gcs_verbal_prev, 5))
                        + COALESCE(gcs_eyes, COALESCE(gcs_eyes_prev, 4))
                END AS gcs_total
            FROM gcs_with_prev
        ),
        -- ================= RENAL: urine output, real 24h coverage-validity (PB4) =================
        urine_raw AS (
            SELECT
                oe.stay_id, oe.charttime,
                CASE WHEN oe.itemid = {ITEMID_URINE_OUTPUT_NEGATE} AND oe.value > 0
                     THEN -1 * oe.value ELSE oe.value END AS urineoutput
            FROM outputevents AS oe
            WHERE oe.stay_id IN ({stay_ids_sql}) AND oe.itemid IN ({urine_ids_sql})
        ),
        urine_tm AS (
            -- inter-measurement gap in minutes, mirrors urine_output_rate.sql's uo_tm CTE
            -- (anchored on ICU intime for the first measurement of a stay)
            SELECT
                u.stay_id, u.charttime, u.urineoutput,
                CASE
                    WHEN LAG(u.charttime) OVER w IS NULL
                        THEN DATEDIFF('minute', s.intime, u.charttime)
                    ELSE DATEDIFF('minute', LAG(u.charttime) OVER w, u.charttime)
                END AS tm_since_last_uo
            FROM urine_raw AS u
            INNER JOIN stay_map_tbl AS s ON u.stay_id = s.stay_id
            WINDOW w AS (PARTITION BY u.stay_id ORDER BY u.charttime NULLS FIRST)
        ),
        urine_24h AS (
            -- for each urine measurement, the trailing-24h volume AND the
            -- actual time coverage of that window (uo_tm_24hr) -- both are
            -- needed since sofa.sql only trusts the extrapolated volume when
            -- uo_tm_24hr >= 24h of real coverage
            SELECT
                io.stay_id, io.charttime,
                SUM(iosum.urineoutput) AS urineoutput_24hr,
                SUM(iosum.tm_since_last_uo) / 60.0 AS uo_tm_24hr
            FROM urine_tm AS io
            LEFT JOIN urine_tm AS iosum
                ON io.stay_id = iosum.stay_id
                AND io.charttime >= iosum.charttime
                AND io.charttime <= iosum.charttime + INTERVAL '23' HOUR
            GROUP BY io.stay_id, io.charttime
        ),
        -- ================= assemble per-hour worst component value =================
        hourly_components AS (
            SELECT
                g.stay_id, g.hr, g.hour_start, g.hour_end,
                (SELECT MIN(pf.pao2fio2ratio) FROM pf_ratio AS pf
                    WHERE pf.stay_id = g.stay_id AND pf.on_vent = FALSE
                        AND pf.charttime > g.hour_start AND pf.charttime <= g.hour_end) AS pf_novent,
                (SELECT MIN(pf.pao2fio2ratio) FROM pf_ratio AS pf
                    WHERE pf.stay_id = g.stay_id AND pf.on_vent = TRUE
                        AND pf.charttime > g.hour_start AND pf.charttime <= g.hour_end) AS pf_vent,
                (SELECT MIN(p.platelet) FROM platelet AS p
                    WHERE p.hadm_id = g.hadm_id
                        AND p.charttime > g.hour_start AND p.charttime <= g.hour_end) AS platelet_min,
                (SELECT MAX(b.bilirubin) FROM bili AS b
                    WHERE b.hadm_id = g.hadm_id
                        AND b.charttime > g.hour_start AND b.charttime <= g.hour_end) AS bilirubin_max,
                (SELECT MAX(c.creatinine) FROM creat AS c
                    WHERE c.hadm_id = g.hadm_id
                        AND c.charttime > g.hour_start AND c.charttime <= g.hour_end) AS creatinine_max,
                (SELECT MIN(m.mbp) FROM mbp AS m
                    WHERE m.stay_id = g.stay_id
                        AND m.charttime > g.hour_start AND m.charttime <= g.hour_end) AS mbp_min,
                (SELECT MAX(v.rate_norepi) FROM vaso AS v
                    WHERE v.stay_id = g.stay_id
                        AND v.endtime > g.hour_start AND v.starttime <= g.hour_end) AS rate_norepi_max,
                (SELECT MAX(v.rate_epi) FROM vaso AS v
                    WHERE v.stay_id = g.stay_id
                        AND v.endtime > g.hour_start AND v.starttime <= g.hour_end) AS rate_epi_max,
                (SELECT MAX(v.rate_dopa) FROM vaso AS v
                    WHERE v.stay_id = g.stay_id
                        AND v.endtime > g.hour_start AND v.starttime <= g.hour_end) AS rate_dopa_max,
                (SELECT MAX(v.rate_dobu) FROM vaso AS v
                    WHERE v.stay_id = g.stay_id
                        AND v.endtime > g.hour_start AND v.starttime <= g.hour_end) AS rate_dobu_max,
                (SELECT MIN(gc.gcs_total) FROM gcs AS gc
                    WHERE gc.stay_id = g.stay_id
                        AND gc.charttime > g.hour_start AND gc.charttime <= g.hour_end) AS gcs_min,
                (SELECT u.urineoutput_24hr FROM urine_24h AS u
                    WHERE u.stay_id = g.stay_id
                        AND u.uo_tm_24hr >= 24
                        AND u.charttime > g.hour_start AND u.charttime <= g.hour_end
                    ORDER BY u.charttime DESC LIMIT 1) AS urineoutput_24hr_valid,
                (SELECT u.uo_tm_24hr FROM urine_24h AS u
                    WHERE u.stay_id = g.stay_id
                        AND u.uo_tm_24hr >= 24
                        AND u.charttime > g.hour_start AND u.charttime <= g.hour_end
                    ORDER BY u.charttime DESC LIMIT 1) AS uo_tm_24hr_valid
            FROM hourly_grid AS g
        )
        SELECT * FROM hourly_components
    """
    df = con.execute(query).df()
    con.unregister("stay_map_tbl")
    return df


def _score_sofa_components(components: pd.DataFrame) -> pd.DataFrame:
    """
    Turn raw per-hour component values into 0-4 subscores using the exact
    thresholds from mimic-code's score/sofa.sql, then roll each subscore up
    over a trailing 24h window (worst value wins) and coalesce to 0 if never
    observed -- see ASSUMPTION A7.

    NOTE (vectorized after team review): this was originally six row-wise
    DataFrame.apply() calls, which is fine at --sample_size scale but would
    be slow across the full MIMIC-IV cohort (potentially tens of millions of
    hourly rows). Rewritten with np.select, which evaluates each threshold
    tier as a full-column boolean mask instead of iterating row by row.
    Thresholds and precedence order are unchanged -- np.select evaluates
    conditions in the order given and takes the first match, exactly
    mirroring the original if/elif cascade (and mimic-code's CASE WHEN
    ordering, which resolves the same way).
    """
    df = components.copy().sort_values(["stay_id", "hr"]).reset_index(drop=True)

    def _all_nan(*cols):
        return np.all([df[c].isna() for c in cols], axis=0)

    pf_vent, pf_novent = df["pf_vent"], df["pf_novent"]
    df["respiration"] = np.select(
        [
            pf_vent < 100,
            pf_vent < 200,
            pf_novent < 300,
            pf_vent < 300,
            pf_novent < 400,
            pf_vent < 400,
        ],
        [4, 3, 2, 2, 1, 1],
        default=0,
    ).astype(float)
    df.loc[_all_nan("pf_vent", "pf_novent"), "respiration"] = np.nan

    plt = df["platelet_min"]
    df["coagulation"] = np.select(
        [plt < 20, plt < 50, plt < 100, plt < 150], [4, 3, 2, 1], default=0
    ).astype(float)
    df.loc[plt.isna(), "coagulation"] = np.nan

    bili = df["bilirubin_max"]
    df["liver"] = np.select(
        [bili >= 12.0, bili >= 6.0, bili >= 2.0, bili >= 1.2], [4, 3, 2, 1], default=0
    ).astype(float)
    df.loc[bili.isna(), "liver"] = np.nan

    dopa, epi, norepi, dobu, mbp = (
        df["rate_dopa_max"], df["rate_epi_max"], df["rate_norepi_max"],
        df["rate_dobu_max"], df["mbp_min"],
    )
    df["cardiovascular"] = np.select(
        [
            dopa > 15,
            epi > 0.1,
            norepi > 0.1,
            dopa > 5,
            (epi > 0) & (epi <= 0.1),
            (norepi > 0) & (norepi <= 0.1),
            (dopa > 0) | (dobu > 0),
            mbp < 70,
        ],
        [4, 4, 4, 3, 3, 3, 2, 1],
        default=0,
    ).astype(float)
    df.loc[_all_nan("mbp_min", "rate_dopa_max", "rate_dobu_max", "rate_epi_max",
                     "rate_norepi_max"), "cardiovascular"] = np.nan

    gcs = df["gcs_min"]
    df["cns"] = np.select(
        [(gcs >= 13) & (gcs <= 14), (gcs >= 10) & (gcs <= 12),
         (gcs >= 6) & (gcs <= 9), gcs < 6],
        [1, 2, 3, 4],
        default=0,
    ).astype(float)
    df.loc[gcs.isna(), "cns"] = np.nan

    # PB4 (faithful port, replaces the earlier ASSUMPTION A11 hr>=18 proxy):
    # urine output is only trusted when uo_tm_24hr_valid (the actual time
    # covered by the trailing-24h window, from urine_output_rate.sql) is
    # itself >= 24h -- not a fixed hours-since-admission cutoff. The
    # extrapolated 24h volume matches sofa.sql's own
    # `uo.urineoutput_24hr / uo.uo_tm_24hr * 24` formula exactly.
    df["uo_24hr_extrapolated"] = np.where(
        df["uo_tm_24hr_valid"].notna() & (df["uo_tm_24hr_valid"] >= 24),
        df["urineoutput_24hr_valid"] / df["uo_tm_24hr_valid"] * 24,
        np.nan,
    )
    creat, uo = df["creatinine_max"], df["uo_24hr_extrapolated"]
    df["renal"] = np.select(
        [
            creat >= 5.0,
            uo < 200,
            (creat >= 3.5) & (creat < 5.0),
            uo < 500,
            (creat >= 2.0) & (creat < 3.5),
            (creat >= 1.2) & (creat < 2.0),
        ],
        [4, 4, 3, 3, 2, 1],
        default=0,
    ).astype(float)
    df.loc[_all_nan("creatinine_max", "uo_24hr_extrapolated"), "renal"] = np.nan

    component_cols = ["respiration", "coagulation", "liver", "cardiovascular", "cns", "renal"]

    # trailing 24h rolling MAX per component, per stay (ASSUMPTION A7),
    # coalesced to 0 if nothing observed in the window
    rolled = {}
    for col in component_cols:
        rolled[f"{col}_24h"] = (
            df.groupby("stay_id")[col]
            .rolling(window=SOFA_ROLLING_WINDOW_HOURS, min_periods=1)
            .max()
            .reset_index(level=0, drop=True)
        )
    rolled_df = pd.DataFrame(rolled, index=df.index).fillna(0)
    df = pd.concat([df, rolled_df], axis=1)
    df["sofa_total"] = rolled_df.sum(axis=1)

    return df[["stay_id", "hr", "hour_start", "hour_end", "sofa_total"] +
              [f"{c}_24h" for c in component_cols]]


def _detect_onset_per_stay(sofa_df: pd.DataFrame, suspicion_df: pd.DataFrame,
                            stay_map: pd.DataFrame) -> pd.DataFrame:
    """
    Onset = first hour where (sofa_total - baseline) >= SOFA_DELTA_THRESHOLD,
    restricted to the [-48h, +24h] window around that admission's suspicion-
    of-infection time (task spec / PROJECT_CONTEXT.md sec 5).

    baseline_sofa is the MEASURED SOFA value at hr=0 -- see ASSUMPTION A4.

    [Team decision, reverting the mimic-code-alignment change] This file
    previously used a fixed 0 baseline, verified to match mimic-code's
    generic sepsis3.sql (`sofa_score >= 2`, no baseline subtraction). That
    was a faithful port of a real reference, but mimic-code's sepsis3.sql is
    a retrospective cohort-labeling tool, not built for an early-prediction
    task. This project's own PROJECT_CONTEXT.md sec 5 names a DIFFERENT,
    more specific precedent for this exact -48h/+24h onset-window task:
    Moor et al. (2019)'s GP-TCN paper. "AI Gone Astray" (arXiv:2203.16452),
    which follows Moor et al.'s implementation directly, states explicitly:
    "we ... use the SOFA value in the first hour of a patient's ICU stay as
    the baseline score for later comparisons." A second, independent
    replication of the same task setup (arXiv:2511.08986) confirms the same
    convention. Team decision: revert to this measured hr=0 baseline for
    internal consistency with the project's own cited methodology, NOT
    mimic-code's generic convention. This is a deliberate choice between two
    legitimate, published conventions, not a bug fix -- see ASSUMPTION A4
    for the full citation trail.

    Because this reopens the early-onset concentration the fixed-0 baseline
    had reduced, it is now paired with the sepsis_within_4h_of_admission
    exclusion in assign_labels() -- exactly how the Moor et al. lineage
    itself operationalizes it (baseline at hr=0, PLUS a buffer exclusion on
    top; arXiv:2511.08986 uses a 6h buffer for the same reason).
    """
    results = []
    stay_to_hadm = stay_map.set_index("stay_id")["hadm_id"].to_dict()
    susp_by_hadm = suspicion_df.set_index("hadm_id")["suspicion_time"].to_dict() if not suspicion_df.empty else {}

    for stay_id, grp in sofa_df.groupby("stay_id"):
        hadm_id = stay_to_hadm.get(stay_id)
        susp_time = susp_by_hadm.get(hadm_id)
        grp = grp.sort_values("hr")

        # ASSUMPTION A4: measured hr=0 SOFA value, matching Moor et al. /
        # AI Gone Astray -- see the docstring above for the full citation
        # trail and why this reverts the earlier mimic-code-aligned fixed-0
        # baseline.
        baseline_row = grp[grp["hr"] == 0]
        baseline = float(baseline_row["sofa_total"].iloc[0]) if len(baseline_row) else 0.0

        onset_time, sofa_at_onset = None, None
        if susp_time is not None and pd.notna(susp_time):
            window_lo = susp_time - pd.Timedelta(hours=SOFA_WINDOW_BEFORE_HOURS)
            window_hi = susp_time + pd.Timedelta(hours=SOFA_WINDOW_AFTER_HOURS)
            in_window = grp[(grp["hour_end"] >= window_lo) & (grp["hour_end"] <= window_hi)]
            qualifying = in_window[in_window["sofa_total"] - baseline >= SOFA_DELTA_THRESHOLD]
            if len(qualifying):
                first_row = qualifying.sort_values("hour_end").iloc[0]
                onset_time = first_row["hour_end"]
                sofa_at_onset = float(first_row["sofa_total"])

        results.append({
            "stay_id": stay_id,
            "hadm_id": hadm_id,
            "suspicion_time": susp_time,
            "baseline_sofa": baseline,
            "sepsis_onset_time": onset_time,
            "sofa_at_onset": sofa_at_onset,
        })

    return pd.DataFrame(results)


def assign_labels(con: duckdb.DuckDBPyConnection, cohort: pd.DataFrame,
                   iv_antibiotics_only: bool = False,
                   blood_cultures_only: bool = False) -> pd.DataFrame:
    """
    Stage B: compute suspicion-of-infection + hourly SOFA trajectory for every
    structurally-eligible candidate in `cohort` (i.e. excluded_reason IS NULL
    coming out of build_cohort), determine sepsis onset, and produce the
    final locked schema:
        subject_id, hadm_id, sepsis_onset_time, sofa_at_onset, label,
        excluded_reason, split

    Also applies the "sepsis_within_4h_of_admission" exclusion here
    (excluded_reason populated, NOT labeled positive) since it requires the
    onset computation that only this stage can do -- see ASSUMPTION A3/A10.
    """
    eligible = cohort[cohort["excluded_reason"].isna()].copy()
    excluded_already = cohort[cohort["excluded_reason"].notna()].copy()

    if eligible.empty:
        final_excluded = excluded_already.copy()
        final_excluded["sepsis_onset_time"] = pd.NaT
        final_excluded["sofa_at_onset"] = np.nan
        final_excluded["label"] = 0
        return final_excluded[[
            "subject_id", "hadm_id", "sepsis_onset_time", "sofa_at_onset",
            "label", "excluded_reason", "split",
        ]]

    hadm_ids = eligible["hadm_id"].astype(int).tolist()
    stay_map = eligible[["stay_id", "hadm_id", "subject_id", "intime", "outtime"]].copy()

    print(f"[label_sepsis3]   [1/4] suspicion of infection for {len(hadm_ids)} "
          f"admissions...", file=sys.stderr)
    t0 = time.time()
    suspicion_df = _compute_suspicion_of_infection(
        con, hadm_ids, iv_only=iv_antibiotics_only, blood_only=blood_cultures_only
    )
    print(f"[label_sepsis3]   [1/4] done in {time.time() - t0:.0f}s "
          f"({len(suspicion_df)} admissions with a qualifying suspicion event)",
          file=sys.stderr)

    print(f"[label_sepsis3]   [2/4] hourly SOFA components for "
          f"{len(stay_map)} ICU stays -- this is usually the slow step on the "
          f"first run before the Parquet cache is warm...", file=sys.stderr)
    t0 = time.time()
    raw_components = _compute_hourly_sofa(con, stay_map)
    print(f"[label_sepsis3]   [2/4] done in {time.time() - t0:.0f}s "
          f"({len(raw_components)} stay-hours)", file=sys.stderr)

    print("[label_sepsis3]   [3/4] scoring SOFA components...", file=sys.stderr)
    t0 = time.time()
    sofa_df = _score_sofa_components(raw_components)
    print(f"[label_sepsis3]   [3/4] done in {time.time() - t0:.0f}s", file=sys.stderr)

    print("[label_sepsis3]   [4/4] detecting onset per stay...", file=sys.stderr)
    t0 = time.time()
    onset_df = _detect_onset_per_stay(sofa_df, suspicion_df, stay_map)
    print(f"[label_sepsis3]   [4/4] done in {time.time() - t0:.0f}s", file=sys.stderr)

    eligible = eligible.merge(
        onset_df[["stay_id", "sepsis_onset_time", "sofa_at_onset", "suspicion_time"]],
        on="stay_id", how="left",
    )

    # sepsis_within_4h_of_admission exclusion [renamed and re-derived per
    # team decision, replacing the earlier sepsis_at_admission check
    # entirely -- does NOT stack with any suspicion_time/hr0-SOFA condition
    # from before].
    #
    # With baseline_sofa reverted to the measured hr=0 value (ASSUMPTION
    # A4), a large share of onsets land very early in the stay -- this is
    # the SAME phenomenon the Moor et al. / AI Gone Astray lineage itself
    # hits, and the field's standard response is a buffer exclusion, not a
    # baseline change:
    #   - arXiv:2210.15056 (UnfoldML): excludes onset within the first 6h
    #     of ICU admission ("88.1% of sepsis onsets happened within the
    #     first 6 hours ... and are excluded from our study cohort")
    #   - arXiv:2511.08986 (replication of the AI Gone Astray / Moor et al.
    #     task setup): excludes "a sepsis onset before the 6-hour window"
    # Team decision: use a 4h buffer specifically (not 6h) for internal
    # consistency with this project's own LOCKED 4-hour rolling prediction
    # horizon (PROJECT_CONTEXT.md sec 5, matching SepsisCalc/SepsisLab) --
    # the 6h precedent above is documented here so a reviewer sees this was
    # a considered choice between two valid literature options, not an
    # oversight.
    #
    # Condition: exclude any stay where sepsis_onset_time <= intime + 4h.
    # This replaces the earlier suspicion_time/hr0-SOFA-based check
    # entirely -- that check existed only because the old fixed-0 baseline
    # made a delta-based test structurally unable to fire near admission;
    # now that baseline is a measured value again, sepsis_onset_time itself
    # is a well-defined, non-degenerate quantity near admission and can be
    # tested directly.
    within_4h_mask = (
        eligible["sepsis_onset_time"].notna()
        & (eligible["sepsis_onset_time"] <= eligible["intime"] + pd.Timedelta(hours=4))
    )
    eligible.loc[within_4h_mask, "excluded_reason"] = "sepsis_within_4h_of_admission"

    eligible["label"] = np.where(
        eligible["excluded_reason"].isna() & eligible["sepsis_onset_time"].notna(), 1, 0
    )
    # excluded rows and true negatives both carry label=0; excluded_reason
    # disambiguates "not part of the cohort" from "negative but included"
    eligible.loc[eligible["excluded_reason"].notna(), "sepsis_onset_time"] = pd.NaT
    eligible.loc[eligible["excluded_reason"].notna(), "sofa_at_onset"] = np.nan

    excluded_already["sepsis_onset_time"] = pd.NaT
    excluded_already["sofa_at_onset"] = np.nan
    excluded_already["label"] = 0

    final = pd.concat([eligible, excluded_already], ignore_index=True, sort=False)

    return final[[
        "subject_id", "hadm_id", "sepsis_onset_time", "sofa_at_onset",
        "label", "excluded_reason", "split",
    ]].reset_index(drop=True)


# ==============================================================================
# COHORT STATISTICS
# ==============================================================================

def compute_cohort_stats(final_df: pd.DataFrame, total_admissions_considered: int) -> dict:
    stats: dict = {}
    stats["total_admissions_considered"] = int(total_admissions_considered)

    exclusion_counts = final_df["excluded_reason"].value_counts(dropna=True)
    stats["exclusions"] = {
        str(reason): {
            "count": int(count),
            "pct_of_considered": round(100.0 * count / total_admissions_considered, 2)
            if total_admissions_considered else None,
        }
        for reason, count in exclusion_counts.items()
    }
    n_excluded_total = int(final_df["excluded_reason"].notna().sum())
    stats["total_excluded"] = n_excluded_total
    stats["pct_excluded_total"] = (
        round(100.0 * n_excluded_total / total_admissions_considered, 2)
        if total_admissions_considered else None
    )

    final_cohort = final_df[final_df["excluded_reason"].isna()].copy()
    stats["final_cohort_size"] = int(len(final_cohort))
    stats["sepsis_positive_rate_pct"] = (
        round(100.0 * final_cohort["label"].mean(), 2) if len(final_cohort) else None
    )

    # split breakdown + positive-rate consistency flag
    split_stats = {}
    pos_rates = []
    for split_name, grp in final_cohort.groupby("split"):
        pos_rate = round(100.0 * grp["label"].mean(), 2) if len(grp) else None
        split_stats[split_name] = {"n": int(len(grp)), "positive_rate_pct": pos_rate}
        if pos_rate is not None:
            pos_rates.append(pos_rate)
    stats["splits"] = split_stats
    if len(pos_rates) >= 2 and (max(pos_rates) - min(pos_rates)) > 5.0:
        stats["split_positive_rate_flag"] = (
            f"WARNING: split positive rates differ by "
            f"{round(max(pos_rates) - min(pos_rates), 2)} pts (>5pt threshold) -- "
            f"check split assignment / cohort composition."
        )

    n_unique_subjects = final_cohort["subject_id"].nunique()
    n_unique_hadm = final_cohort["hadm_id"].nunique()
    stats["n_unique_subject_ids"] = int(n_unique_subjects)
    stats["n_unique_hadm_ids"] = int(n_unique_hadm)
    if n_unique_subjects != n_unique_hadm:
        stats["subject_hadm_mismatch_flag"] = (
            f"WARNING: {n_unique_subjects} unique subject_ids vs "
            f"{n_unique_hadm} unique hadm_ids in final cohort -- expected "
            f"equal given the first-stay-only filter. Check for a bug in "
            f"the subject-level split or first-stay selection."
        )

    return stats


def _add_onset_timing_stats(stats: dict, cohort_with_intime: pd.DataFrame,
                             final_df: pd.DataFrame) -> dict:
    """Median/IQR of hours-from-admission-to-onset, positive patients only."""
    merged = final_df.merge(
        cohort_with_intime[["subject_id", "hadm_id", "intime"]],
        on=["subject_id", "hadm_id"], how="left",
    )
    pos = merged[(merged["label"] == 1) & merged["sepsis_onset_time"].notna()].copy()
    if len(pos):
        hours = (pos["sepsis_onset_time"] - pos["intime"]).dt.total_seconds() / 3600.0
        stats["hours_admission_to_onset_positive_patients"] = {
            "median": round(float(hours.median()), 2),
            "iqr_25": round(float(hours.quantile(0.25)), 2),
            "iqr_75": round(float(hours.quantile(0.75)), 2),
            "n": int(len(hours)),
        }

        # DIAGNOSTIC (added after review): a low median hours-to-onset could
        # be genuine (many ICU admissions ARE already deteriorating on
        # arrival) or could be an artifact of the trailing-24h rolling-max
        # SOFA window using min_periods=1 (ASSUMPTION A7) -- in the first few
        # hours that window is nearly empty, so a component that was simply
        # unmeasured at hr=0 (scored 0 via the missing->healthy convention)
        # and gets its first-ever measurement at hr=1 or hr=2 can look like a
        # 2+ point "increase" purely because data started arriving, not
        # because the patient got sicker. This does NOT distinguish the two
        # explanations -- it just quantifies how much of the onset
        # distribution is concentrated in that early, window-fragile period,
        # so the team can decide whether it needs a closer look before
        # trusting the full-cohort numbers.
        n_pos = len(hours)
        stats["early_onset_diagnostic"] = {
            "pct_onset_within_1h": round(100.0 * (hours <= 1).sum() / n_pos, 2),
            "pct_onset_within_2h": round(100.0 * (hours <= 2).sum() / n_pos, 2),
            "pct_onset_within_3h": round(100.0 * (hours <= 3).sum() / n_pos, 2),
            "note": (
                "High concentration here is consistent with either genuine "
                "early deterioration OR the min_periods=1 rolling-window "
                "artifact described in ASSUMPTION A7 -- this stat alone "
                "cannot tell them apart. Worth a manual chart-level spot "
                "check on a few hr<=2 onset cases before trusting the full "
                "cohort's positive rate."
            ),
        }
        if stats["early_onset_diagnostic"]["pct_onset_within_2h"] > 30.0:
            stats["early_onset_diagnostic"]["flag"] = (
                f"WARNING: {stats['early_onset_diagnostic']['pct_onset_within_2h']}% "
                f"of positive-label onsets occur within 2h of ICU admission -- "
                f"unusually concentrated. Recommend spot-checking a few of "
                f"these cases against raw chartevents/labevents before "
                f"trusting this cohort's positive rate."
            )
    else:
        stats["hours_admission_to_onset_positive_patients"] = None
        stats["early_onset_diagnostic"] = None
    return stats


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Build the locked Sepsis-3 cohort + labels from raw MIMIC-IV CSVs."
    )
    parser.add_argument("--mimic_hosp_dir", type=str, required=True,
                         help="Path to the MIMIC-IV 'hosp' directory (admissions.csv etc.)")
    parser.add_argument("--mimic_icu_dir", type=str, required=True,
                         help="Path to the MIMIC-IV 'icu' directory (icustays.csv etc.)")
    parser.add_argument("--out_dir", type=str, default="data/cohort",
                         help="Output directory for sepsis_labels.parquet and cohort_stats.json")
    parser.add_argument("--splits_path", type=str, default="data/splits/subject_splits.parquet",
                         help="Persisted subject-level split assignment (generated once, reused).")
    parser.add_argument("--sample_size", type=int, default=None,
                         help="If set, run on a random sample of this many first-ICU-stay "
                              "subjects instead of the full cohort (for quick local testing).")
    parser.add_argument("--cache_dir", type=str, default="data/cache",
                         help="Where to cache Parquet conversions of the large event tables "
                              "(chartevents/labevents/outputevents/inputevents). First run "
                              "pays a one-time conversion cost; every run after that (sample "
                              "or full) reads the small, columnar cache instead of re-scanning "
                              "the raw multi-GB CSVs from scratch. Pass --no_cache to disable "
                              "and always read raw CSV (slower, no disk cache written).")
    parser.add_argument("--no_cache", action="store_true",
                         help="Disable the Parquet cache and read raw CSV directly every time.")
    parser.add_argument("--min_valid_obs_hours", type=int, default=MIN_VALID_OBS_HOURS,
                         help="Minimum distinct hourly observation timepoints required to keep "
                              "an admission (see ASSUMPTION A8/A8.1: confirmed by direct "
                              "inspection of SepsisCalc's actual code that their own threshold "
                              "is 5 RAW hours, same granularity as this default -- no unit "
                              "conversion needed. Override if the team wants a different value "
                              "for other reasons.)")
    parser.add_argument("--iv_antibiotics_only", action="store_true",
                         help="Restrict antibiotics to IV routes only (see ASSUMPTION A2). "
                              "Default (unset) matches mimic-code: all non-topical routes.")
    parser.add_argument("--blood_cultures_only", action="store_true",
                         help="Restrict cultures to blood specimens only (see ASSUMPTION A1). "
                              "Default (unset) matches mimic-code/alistairewj: any specimen type.")
    args = parser.parse_args()

    hosp_dir = Path(args.mimic_hosp_dir)
    icu_dir = Path(args.mimic_icu_dir)
    out_dir = Path(args.out_dir)
    splits_path = Path(args.splits_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = None if args.no_cache else Path(args.cache_dir)
    print(f"[label_sepsis3] Connecting to MIMIC-IV CSVs via DuckDB "
          f"(hosp={hosp_dir}, icu={icu_dir}, "
          f"cache_dir={cache_dir if cache_dir else 'disabled'})...", file=sys.stderr)
    con = connect_duckdb(hosp_dir, icu_dir, args.sample_size, cache_dir=cache_dir)

    # BUG FIX (found after reviewing real-data output): this used to be a raw
    # `SELECT COUNT(*) FROM admissions`, i.e. the size of the WHOLE hospital
    # admissions table, completely independent of --sample_size. That made
    # every pct_of_considered / pct_excluded_total stat in cohort_stats.json
    # wrong -- e.g. a run on --sample_size 500 with 30 exclusions reported
    # "0.01% excluded" (30 / 546,028) instead of the correct 6.0% (30 / 500).
    # It would ALSO have been wrong on a full run: raw admission count
    # (multiple admissions per patient) is a different, larger denominator
    # than "candidates that actually entered our filtering pipeline" (one
    # first-ICU-stay row per subject, per Stage A). The correct denominator
    # for these stats is len(cohort) -- exactly the set of first-eligible-
    # stay candidate rows that excluded_reason actually gets assigned
    # against -- not a hospital-wide table count. The raw admissions.csv
    # count is still surfaced separately below for context.
    total_admissions_in_hosp_table = con.execute("SELECT COUNT(*) FROM admissions").fetchone()[0]

    print("[label_sepsis3] Stage A: build_cohort()...", file=sys.stderr)
    cohort = build_cohort(con, splits_path, sample_size=args.sample_size,
                           min_valid_obs_hours=args.min_valid_obs_hours)
    total_admissions_considered = len(cohort)
    print(f"[label_sepsis3]   -> {len(cohort)} first-eligible-stay candidates "
          f"({cohort['excluded_reason'].isna().sum()} pass structural filters)",
          file=sys.stderr)

    print("[label_sepsis3] Stage B: assign_labels()...", file=sys.stderr)
    final_df = assign_labels(
        con, cohort,
        iv_antibiotics_only=args.iv_antibiotics_only,
        blood_cultures_only=args.blood_cultures_only,
    )

    out_path = out_dir / "sepsis_labels.parquet"
    final_df.to_parquet(out_path, index=False)
    print(f"[label_sepsis3] Wrote {out_path} ({len(final_df)} rows)", file=sys.stderr)

    print("[label_sepsis3] Computing cohort statistics...", file=sys.stderr)
    stats = compute_cohort_stats(final_df, total_admissions_considered)
    stats["total_admissions_in_hosp_table"] = int(total_admissions_in_hosp_table)
    stats["note_sample_size"] = (
        f"Run with --sample_size {args.sample_size}: all counts/percentages above "
        f"are scoped to this sample, not the full hospital admissions table."
        if args.sample_size else
        "Full run (no --sample_size): total_admissions_considered is the number of "
        "first-eligible-stay candidates entering Stage A, not the raw admissions.csv "
        "row count (see total_admissions_in_hosp_table for that)."
    )
    stats = _add_onset_timing_stats(stats, cohort, final_df)

    stats_path = out_dir / "cohort_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, default=str)

    print(json.dumps(stats, indent=2, default=str))
    print(f"[label_sepsis3] Wrote {stats_path}", file=sys.stderr)

    if "split_positive_rate_flag" in stats:
        print(f"[label_sepsis3] {stats['split_positive_rate_flag']}", file=sys.stderr)
    if "subject_hadm_mismatch_flag" in stats:
        print(f"[label_sepsis3] {stats['subject_hadm_mismatch_flag']}", file=sys.stderr)


if __name__ == "__main__":
    main()