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

A5. Ventilation status (needed to choose the PaO2/FiO2 respiration
    thresholds) is normally derived in mimic-code from a large state machine
    over ventilator-setting and oxygen-delivery-device chartevents
    (treatment/ventilation.sql). That full state machine was not reproduced
    here to keep this file single-source and dependency-light. Instead,
    "on invasive ventilation at time T" is approximated as "a Ventilator Mode
    chartevent (itemid 223849) was recorded within +/-6h of T". This will
    under- or over-count ventilation status in edge cases (e.g. NIV-only
    patients, or vent charting gaps) relative to the full mimic-code logic.
    Flag for team review if respiration-component SOFA looks off.

A6. PaO2/FiO2 pairing: mimic-code's bg.sql has multi-step FiO2 imputation
    (blood-gas-panel FiO2 first, else nearest preceding chartevents FiO2
    value with additional bookkeeping). Here FiO2 is taken from the nearest
    chartevents FiO2 (itemid 223835) within a 4h lookback of the arterial
    blood gas PaO2; if none is found the ratio is left null (not defaulted to
    room air), consistent with mimic-code leaving it null rather than assuming
    21%.

A7. "Missing SOFA component -> assume healthy (0 points)" (task spec, item 2
    under Stage B) is operationalized using the SAME mechanism mimic-code
    itself uses to encode exactly this principle: each component is taken as
    the WORST (max severity) value in the trailing 24h window, and only
    coalesced to 0 if there is truly no value anywhere in that 24h window.
    This avoids a raw hour-to-hour flicker of the score that a literal
    "missing-this-instant -> 0" reading would otherwise produce, while
    satisfying the same non-dropping intent as the spec.

A8. [Corrected after team review] The task spec's ">= 5 valid hourly
    observation timepoints" was described as "adapted from SepsisCalc's own
    exclusion threshold." The original version of this file could not locate
    that constant and implemented "5" as a literal raw-hour count. Per direct
    review of SepsisCalc's actual code (`if len(file_data) < 5 or
    sorted(vector)[2] < 1`), their "5" is a row-count over their OWN
    3-HOUR-BINNED data, not raw hours -- i.e. their threshold corresponds to
    roughly 15 raw hours of coverage, not 5. Since this file deliberately
    uses raw unbinned timestamps (per the task's explicit instruction NOT to
    copy SepsisCalc's 3h binning step), "5" and "15" are not interchangeable
    here -- one is ~3x looser than SepsisCalc's actual precedent.
    MIN_VALID_OBS_HOURS is now exposed via --min_valid_obs_hours (default
    kept at 5, i.e. the task spec's literal number, NOT auto-changed to 15)
    so the team can make this call explicitly rather than have either
    reading silently baked in. Pass --min_valid_obs_hours 15 to match
    SepsisCalc's actual precedent instead of the task spec's literal wording.

A9. Sepsis is only detectable from data recorded during the ICU stay itself
    (chartevents/labevents/inputevents/outputevents are ICU-scoped tables).
    If suspicion of infection or the necessary SOFA lookback window falls
    mostly before ICU admission (e.g., antibiotics started in the ED), this
    pipeline cannot see that pre-ICU data and may under-detect onset in that
    window. Any onset time computed at or before ICU intime is excluded via
    excluded_reason="sepsis_at_admission" per the task spec, which is the
    correct behavior for this exclusion but does not fully solve the
    underlying visibility gap; flagging for team awareness.

A10. [Added after team review -- fixes a real bug] The "sepsis_at_admission"
    exclusion cannot be implemented as "sepsis_onset_time <= intime": onset
    time is always some hour_end from the hourly grid, and the earliest
    possible value is intime + 1h, so that comparison is structurally
    unsatisfiable and was silently dead code in the first version of this
    file (confirmed: cohort_stats.json showed zero admissions with this
    excluded_reason). It also cannot be fixed by testing the delta-from-
    baseline at hr=0, because baseline_sofa is DEFINED as the hr=0 SOFA value
    (ASSUMPTION A4), so that delta is identically 0 by construction. The
    corrected check instead tests two things directly: (1) suspicion_time
    <= intime (infection was already suspected at or before ICU arrival --
    this CAN be true, since antibiotics/cultures are looked up at the
    admission level, not stay-scoped, so e.g. ED-administered antibiotics
    count), AND (2) the ABSOLUTE hr=0 SOFA total is already >= 2, using the
    standard "premorbid SOFA assumed 0" convention (i.e., organ dysfunction
    was already substantial by the time ICU data starts). This is
    intentionally an absolute-level test, not a delta test, and is a
    genuinely separate judgment call from the delta-based onset detection
    used for the main label -- confirm this operationalization matches team
    intent before treating Milestone 0 as locked.

A11. Renal urine-output validity (`hr >= 18` in _score_sofa_components) is a
    simplification of mimic-code's actual 22-30h collection-window validity
    check (uo_tm_24hr). Listed here explicitly per team review feedback --
    this was previously only a code comment, not called out at module level.
================================================================================
"""



import argparse
import json
import sys
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
# chartevents
ITEMID_FIO2_CHARTEVENTS = 223835
ITEMID_MBP = (220052, 220181, 225312)
ITEMID_GCS_MOTOR = 223901
ITEMID_GCS_VERBAL = 223900
ITEMID_GCS_EYES = 220739
ITEMID_VENT_MODE = 223849  # ASSUMPTION A5: simplified ventilation-status proxy
# inputevents (vasopressor rate, uom already rate per mimic-code convention)
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

FIO2_LOOKBACK_HOURS = 4  # ASSUMPTION A6
VENT_FLAG_WINDOW_HOURS = 6  # ASSUMPTION A5
SOFA_ROLLING_WINDOW_HOURS = 24  # ASSUMPTION A7, matches mimic-code

DEFAULT_TRAIN_FRAC = 0.70
DEFAULT_VAL_FRAC = 0.15
# test frac is the remainder


# ==============================================================================
# DATA LOADING
# ==============================================================================

def connect_duckdb(hosp_dir: Path, icu_dir: Path, sample_size: Optional[int]) -> duckdb.DuckDBPyConnection:
    """
    Register the raw MIMIC-IV CSVs as DuckDB views so we can query them with
    SQL directly off disk (per user environment note: prefer duckdb over
    loading full tables into pandas memory).

    If sample_size is set, we still register full views for the small
    "dimension" tables (patients/admissions/icustays) but cap read of the huge
    event tables at read time via LIMIT in the queries that use them, applied
    AFTER filtering to the sampled subject_ids (see build_cohort/assign_labels)
    so that --sample_size gives a coherent, not-truncated-mid-table, subset.
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
        replace_clause = ""
        dt_cols = DATETIME_COLS.get(view_name, [])
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
    "sepsis_at_admission" exclusion (which needs suspicion-of-infection + SOFA
    computed first) is applied in assign_labels(), see ASSUMPTION A3 in the
    module docstring for why these two exclusion families are split this way.

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
    """).df()

    if sample_size is not None:
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
            -- one row per (stay_id, hr) from intime to outtime
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
        -- ---------------- respiration: PaO2/FiO2 ----------------
        abg AS (
            SELECT
                le.hadm_id, le.specimen_id, le.charttime,
                MAX(CASE WHEN le.itemid = {ITEMID_PO2} THEN le.valuenum END) AS po2,
                MAX(CASE WHEN le.itemid = {ITEMID_SPECIMEN_TYPE} THEN le.value END) AS specimen
            FROM labevents AS le
            WHERE le.hadm_id IN ({hadm_ids_sql})
                AND le.itemid IN ({ITEMID_PO2}, {ITEMID_SPECIMEN_TYPE})
            GROUP BY le.hadm_id, le.specimen_id, le.charttime
        ),
        abg_art AS (
            SELECT hadm_id, charttime, po2
            FROM abg
            WHERE specimen = 'ART.' AND po2 IS NOT NULL
        ),
        fio2_ce AS (
            SELECT stay_id, charttime, valuenum AS fio2
            FROM chartevents
            WHERE stay_id IN ({stay_ids_sql})
                AND itemid = {ITEMID_FIO2_CHARTEVENTS}
                AND valuenum > 0 AND valuenum <= 100
        ),
        vent_flag AS (
            SELECT DISTINCT stay_id, charttime AS vent_charttime
            FROM chartevents
            WHERE stay_id IN ({stay_ids_sql})
                AND itemid = {ITEMID_VENT_MODE}
        ),
        pf_ratio AS (
            SELECT
                s.stay_id, abg_art.charttime, abg_art.po2,
                f.fio2,
                CASE WHEN f.fio2 IS NOT NULL THEN 100.0 * abg_art.po2 / f.fio2 ELSE NULL END AS pao2fio2ratio,
                CASE WHEN v.stay_id IS NOT NULL THEN TRUE ELSE FALSE END AS on_vent
            FROM abg_art
            INNER JOIN stay_map_tbl AS s ON abg_art.hadm_id = s.hadm_id
            LEFT JOIN LATERAL (
                SELECT fio2_ce.fio2
                FROM fio2_ce
                WHERE fio2_ce.stay_id = s.stay_id
                    AND fio2_ce.charttime <= abg_art.charttime
                    AND fio2_ce.charttime > abg_art.charttime - INTERVAL '{FIO2_LOOKBACK_HOURS}' HOUR
                ORDER BY fio2_ce.charttime DESC
                LIMIT 1
            ) AS f ON TRUE
            LEFT JOIN LATERAL (
                SELECT vent_flag.stay_id
                FROM vent_flag
                WHERE vent_flag.stay_id = s.stay_id
                    AND ABS(DATEDIFF('minute', vent_flag.vent_charttime, abg_art.charttime)) <= {VENT_FLAG_WINDOW_HOURS * 60}
                LIMIT 1
            ) AS v ON TRUE
        ),
        -- ---------------- coagulation: platelets ----------------
        platelet AS (
            SELECT le.hadm_id, le.charttime, le.valuenum AS platelet
            FROM labevents AS le
            WHERE le.hadm_id IN ({hadm_ids_sql}) AND le.itemid = {ITEMID_PLATELET}
        ),
        -- ---------------- liver: bilirubin ----------------
        bili AS (
            SELECT le.hadm_id, le.charttime, le.valuenum AS bilirubin
            FROM labevents AS le
            WHERE le.hadm_id IN ({hadm_ids_sql}) AND le.itemid = {ITEMID_BILIRUBIN_TOTAL}
        ),
        -- ---------------- renal: creatinine ----------------
        creat AS (
            SELECT le.hadm_id, le.charttime, le.valuenum AS creatinine
            FROM labevents AS le
            WHERE le.hadm_id IN ({hadm_ids_sql}) AND le.itemid = {ITEMID_CREATININE}
        ),
        -- ---------------- cardiovascular: MAP ----------------
        mbp AS (
            SELECT ce.stay_id, ce.charttime, ce.valuenum AS mbp
            FROM chartevents AS ce
            WHERE ce.stay_id IN ({stay_ids_sql})
                AND ce.itemid IN ({mbp_ids_sql})
                AND ce.valuenum > 0 AND ce.valuenum < 300
        ),
        -- ---------------- cardiovascular: vasopressor rates ----------------
        vaso AS (
            SELECT
                ie.stay_id,
                CAST(ie.starttime AS TIMESTAMP) AS starttime,
                CAST(ie.endtime AS TIMESTAMP) AS endtime,
                MAX(CASE WHEN ie.itemid = {ITEMID_NOREPINEPHRINE} THEN ie.rate END) AS rate_norepi,
                MAX(CASE WHEN ie.itemid = {ITEMID_EPINEPHRINE} THEN ie.rate END) AS rate_epi,
                MAX(CASE WHEN ie.itemid = {ITEMID_DOPAMINE} THEN ie.rate END) AS rate_dopa,
                MAX(CASE WHEN ie.itemid = {ITEMID_DOBUTAMINE} THEN ie.rate END) AS rate_dobu
            FROM inputevents AS ie
            WHERE ie.stay_id IN ({stay_ids_sql})
                AND ie.itemid IN ({ITEMID_NOREPINEPHRINE}, {ITEMID_EPINEPHRINE}, {ITEMID_DOPAMINE}, {ITEMID_DOBUTAMINE})
            GROUP BY ie.stay_id, ie.starttime, ie.endtime
        ),
        -- ---------------- cns: GCS ----------------
        -- ASSUMPTION A12 (added after team review): the "gcs_verbal = 0 ->
        -- total GCS forced to 15" rule below was NOT flagged as an
        -- assumption in the original version of this file, even though it
        -- looks like a judgment call. Verified directly against mimic-code's
        -- live mimic-iv/concepts_duckdb/score/gcs.sql: that file contains
        -- the exact same rule for the current-row case (when a patient's
        -- verbal component is coded 'No Response-ETT' -- i.e. intubated and
        -- therefore unassessable -- GCS is forced to the best-case value of
        -- 15 rather than penalizing an unmeasurable component). This IS the
        -- standard MIMIC-concepts convention, not an invented simplification.
        -- What IS simplified relative to mimic-code: the reference also
        -- carries this forward from the previous row within a 6h window
        -- (so a later row with a still-missing verbal component, following a
        -- verbal=0 row, also gets forced to 15) and separately handles the
        -- case where only the current row's motor/eyes are missing but a
        -- prior row had verbal=0. That row-to-row carry-forward stitching is
        -- NOT reproduced here -- each row is scored independently other than
        -- the COALESCE defaults below. This could understate CNS-driven
        -- onset detection for intubated patients whose verbal component
        -- happens to not be re-charted every row. Confirm this omission is
        -- acceptable before treating Milestone 0 as locked.
        gcs_raw AS (
            SELECT
                ce.stay_id, ce.charttime,
                MAX(CASE WHEN ce.itemid = {ITEMID_GCS_MOTOR} THEN ce.valuenum END) AS gcs_motor,
                MAX(CASE WHEN ce.itemid = {ITEMID_GCS_VERBAL} AND CAST(ce.value AS VARCHAR) = 'No Response-ETT' THEN 0
                         WHEN ce.itemid = {ITEMID_GCS_VERBAL} THEN ce.valuenum END) AS gcs_verbal,
                MAX(CASE WHEN ce.itemid = {ITEMID_GCS_EYES} THEN ce.valuenum END) AS gcs_eyes
            FROM chartevents AS ce
            WHERE ce.stay_id IN ({stay_ids_sql})
                AND ce.itemid IN ({ITEMID_GCS_MOTOR}, {ITEMID_GCS_VERBAL}, {ITEMID_GCS_EYES})
            GROUP BY ce.stay_id, ce.charttime
        ),
        gcs AS (
            SELECT
                stay_id, charttime,
                CASE
                    WHEN gcs_verbal = 0 THEN 15
                    ELSE COALESCE(gcs_motor, 6) + COALESCE(gcs_verbal, 5) + COALESCE(gcs_eyes, 4)
                END AS gcs_total
            FROM gcs_raw
        ),
        -- ---------------- renal: urine output, 24h trailing sum ----------------
        urine_raw AS (
            SELECT
                oe.stay_id, oe.charttime,
                CASE WHEN oe.itemid = {ITEMID_URINE_OUTPUT_NEGATE} AND oe.value > 0
                     THEN -1 * oe.value ELSE oe.value END AS urineoutput
            FROM outputevents AS oe
            WHERE oe.stay_id IN ({stay_ids_sql}) AND oe.itemid IN ({urine_ids_sql})
        ),
        -- ---------------- per-hour worst component value ----------------
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
                (SELECT SUM(u.urineoutput) FROM urine_raw AS u
                    WHERE u.stay_id = g.stay_id
                        AND u.charttime > g.hour_end - INTERVAL '24' HOUR
                        AND u.charttime <= g.hour_end) AS uo_24hr
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

    # urine output only considered "valid" after >=18h of ICU stay (simplified
    # from mimic-code's 22-30h collection-window validity check -- see
    # ASSUMPTION A11)
    df["uo_24hr_valid"] = np.where(df["hr"] >= 18, df["uo_24hr"], np.nan)
    creat, uo = df["creatinine_max"], df["uo_24hr_valid"]
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
    df.loc[_all_nan("creatinine_max", "uo_24hr_valid"), "renal"] = np.nan

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
    For each stay: baseline SOFA = value at hr=0 (ASSUMPTION A4). Onset = first
    hour where (sofa_total - baseline) >= SOFA_DELTA_THRESHOLD, restricted to
    the [-48h, +24h] window around that admission's suspicion-of-infection
    time (task spec / PROJECT_CONTEXT.md sec 5).
    """
    results = []
    stay_to_hadm = stay_map.set_index("stay_id")["hadm_id"].to_dict()
    susp_by_hadm = suspicion_df.set_index("hadm_id")["suspicion_time"].to_dict() if not suspicion_df.empty else {}

    for stay_id, grp in sofa_df.groupby("stay_id"):
        hadm_id = stay_to_hadm.get(stay_id)
        susp_time = susp_by_hadm.get(hadm_id)
        grp = grp.sort_values("hr")

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

    Also applies the "sepsis_at_admission" exclusion here (excluded_reason
    populated, NOT labeled positive) since it requires the onset computation
    that only this stage can do -- see ASSUMPTION A3.
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

    suspicion_df = _compute_suspicion_of_infection(
        con, hadm_ids, iv_only=iv_antibiotics_only, blood_only=blood_cultures_only
    )
    raw_components = _compute_hourly_sofa(con, stay_map)
    sofa_df = _score_sofa_components(raw_components)
    onset_df = _detect_onset_per_stay(sofa_df, suspicion_df, stay_map)

    eligible = eligible.merge(
        onset_df[["stay_id", "sepsis_onset_time", "sofa_at_onset", "suspicion_time"]],
        on="stay_id", how="left",
    )

    # sepsis_at_admission exclusion: infection was already suspected AT OR
    # BEFORE ICU arrival (suspicion_time can predate intime -- e.g. antibiotics
    # started in the ED) AND organ dysfunction is already elevated at the very
    # first computed hour of the stay.
    #
    # NOTE (fixed after review): an earlier version of this check compared
    # sepsis_onset_time <= intime. That is structurally unsatisfiable --
    # onset_time is always some hour_end from the hourly grid, and the
    # earliest possible value is intime + 1h, so it can never be <= intime.
    # It also can't be fixed by comparing onset_time to intime at all, because
    # baseline_sofa is defined as the hr=0 SOFA value itself (ASSUMPTION A4),
    # which makes the delta at hr=0 identically 0 by construction -- a
    # delta-based test can never fire there either.
    #
    # Correct check: suspicion_time <= intime (infection suspected at/before
    # ICU arrival) AND the ABSOLUTE hr=0 SOFA total (not the delta-from-
    # baseline, which is 0 by definition at hr=0) already meets the
    # SOFA_DELTA_THRESHOLD, using the standard "premorbid SOFA assumed 0"
    # convention. This is a distinct check from the delta-based onset
    # detection used for the main label and is intentionally absolute rather
    # than relative -- see ASSUMPTION A10 in the module docstring.
    hr0_sofa = (
        sofa_df[sofa_df["hr"] == 0].set_index("stay_id")["sofa_total"]
        if not sofa_df.empty else pd.Series(dtype=float)
    )
    eligible["_hr0_sofa"] = eligible["stay_id"].map(hr0_sofa)
    already_septic_mask = (
        eligible["suspicion_time"].notna()
        & (eligible["suspicion_time"] <= eligible["intime"])
        & eligible["_hr0_sofa"].notna()
        & (eligible["_hr0_sofa"] >= SOFA_DELTA_THRESHOLD)
    )
    eligible.drop(columns=["_hr0_sofa"], inplace=True)
    eligible.loc[already_septic_mask, "excluded_reason"] = "sepsis_at_admission"

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
    else:
        stats["hours_admission_to_onset_positive_patients"] = None
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
    parser.add_argument("--min_valid_obs_hours", type=int, default=MIN_VALID_OBS_HOURS,
                         help="Minimum distinct hourly observation timepoints required to keep "
                              "an admission (see ASSUMPTION A8: task spec says 5, but "
                              "SepsisCalc's actual precedent -- once you account for their 3h "
                              "binning -- corresponds to ~15 raw hours. Pass 15 to match that "
                              "precedent instead of the task spec's literal number).")
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

    print(f"[label_sepsis3] Connecting to MIMIC-IV CSVs via DuckDB "
          f"(hosp={hosp_dir}, icu={icu_dir})...", file=sys.stderr)
    con = connect_duckdb(hosp_dir, icu_dir, args.sample_size)

    total_admissions_considered = con.execute("SELECT COUNT(*) FROM admissions").fetchone()[0]

    print("[label_sepsis3] Stage A: build_cohort()...", file=sys.stderr)
    cohort = build_cohort(con, splits_path, sample_size=args.sample_size,
                           min_valid_obs_hours=args.min_valid_obs_hours)
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
