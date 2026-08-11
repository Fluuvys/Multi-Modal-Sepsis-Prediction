"""
ehr_extraction.py

PURPOSE:
    Extract the irregular EHR time-series (vitals + labs) per admission from MIMIC-IV,
    preserving exact observation timestamps (do NOT bin/discretize here -- SDCA needs
    real elapsed-time values downstream, unlike MedFuse's hourly-binned approach).

REFERENCE: adapted from MedPatch's EHR extraction (17-variable standard set), see
    docs/gap_and_solution.md for why we're extending MedPatch's confidence mechanism
    conceptually rather than forking its code directly.

TODO:
    [ ] Pull the standard 17-variable set (5 categorical + 12 continuous, matching
        MedFuse/MedPatch convention for comparability)
    [ ] Keep raw timestamps, no fixed-interval resampling
    [ ] Output one time-series object per admission, aligned to sepsis_labels.csv

INPUT:  raw MIMIC-IV chartevents/labevents (see data/raw_links/)
OUTPUT: data/cohort/ehr_timeseries/ (one file per admission, or a single long-format
        file -- decide and document here once implemented)
"""
"""
preprocessing/ehr_extraction.py
================================================================================
PURPOSE:
    Extract the irregular EHR time-series (vitals + labs) per admission from MIMIC-IV,
    preserving exact observation timestamps. Per PROJECT_CONTEXT.md Rule #5, NO 
    model-specific reduction (e.g., hourly binning) happens here. SDCA needs real 
    elapsed-time values downstream.

TWO STAGES, KEPT AS SEPARATE FUNCTIONS INTERNALLY:
    Stage A -- extract_raw_events(): Connects to raw MIMIC-IV tables, extracts the 
               standard 17-variable set (5 categorical + 12 continuous), and applies 
               variable-specific cleaning/normalization.
    Stage B -- align_and_format(): Inner-joins against the Sepsis-3 cohort, derives 
               missing composite variables (e.g., GCS total), and computes the 
               critical 'hours_before_onset' column.

REFERENCE IMPLEMENTATIONS:
    - nyuad-cai/MedPatch (mimic4extract/ and ehr_utils/) for the standard 17-variable 
      set mapping and cleaning functions.

TODO checklist (Milestone 1 — EHR preprocessing pipeline):
    [x] Pull the standard 17-variable set matching MedFuse/MedPatch convention
    [x] Keep raw timestamps, no fixed-interval resampling
    [x] Output one long-format time-series object, aligned to sepsis_labels.parquet
    [x] Compute hours_before_onset for SDCA
    [x] Add per-variable coverage stats and a spot-check utility
    [ ] Team sign-off on the ASSUMPTION items below before this is truly "locked"
================================================================================
ASSUMPTIONS / DEVIATIONS FLAGGED FOR TEAM REVIEW — read before trusting output
================================================================================
A1. VARIABLE MAPPING PROVENANCE: The exact itemid->variable mapping and outlier 
    ranges from MedPatch (itemid_to_variable_map.csv, variable_ranges.csv) were 
    not fully provided. VARIABLE_ITEMID_MAP is a best-effort MetaVision mapping. 
    OUTLIER_RANGES is currently empty. The team must wire in the real ranges from 
    the CSV before final modeling.

A2. GCS TOTAL DERIVATION: MIMIC-IV MetaVision lacks a pre-computed "GCS total" 
    itemid. Derived in _derive_gcs_total(), a faithful port of mimic-code's 
    measurement/gcs.sql (matching the same logic already used in 
    label_sepsis3.py's SOFA CNS component, PB3). Handles two MIMIC-IV-specific 
    quirks: (1) intubated patients charted as verbal="No Response-ETT" get 
    verbal remapped to 0 and total forced to 15, rather than penalized as a 
    real low verbal score; (2) a missing component can carry forward from the 
    previous reading within 6h, componentwise, rather than requiring exact-
    timestamp completeness for all three. Verified against a synthetic 
    4-branch test case before merge.

A3. UNIT CONVERSIONS: Temperature (F to C), FiO2 (pct to fraction), Weight 
    (oz/lb to kg), Height (in to cm) apply heuristics from MedPatch. Blood 
    pressure parsing handles legacy "120/80" string artifacts. 'Capillary refill 
    rate' strings are strictly mapped to 0.0/1.0; unrecognized strings are dropped.

A4. OUT OF SCOPE (By Design): Discretizer / Normalizer (fixed-timestep binning, 
    z-score normalization) are intentionally excluded here per Context Rule #5. 
    This file only ever produces raw, unbinned, long-format observations. Model 
    adapters handle binning downstream.
================================================================================
python preprocessing/ehr_extraction.py \
  --data_root "/home/fluuvys-main/Research/Multi modal sepsis prediction/Data/mimic-iv-3.1" \
  --labels_path data/cohort/sepsis_labels.parquet \
  --output data/cohort/ehr_timeseries_sample.parquet \
  --sample_size 50


"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import duckdb

# ------------------------------------------------------------------------------
# 1. ITEMID MAP  (chartevents itemids unless noted "LABEVENTS")
#    Structure: variable_name -> list of (itemid, source_table, unit_note)
# ------------------------------------------------------------------------------

VARIABLE_ITEMID_MAP = {
    # ---- categorical (5) ----
    "Capillary refill rate":              [(223951, "chartevents", "text->binary, see CLEAN_FNS")],
    "Glascow coma scale eye opening":     [(220739, "chartevents", "1-4 scale")],
    "Glascow coma scale motor response":  [(223901, "chartevents", "1-6 scale")],
    "Glascow coma scale verbal response": [(223900, "chartevents", "1-5 scale")],
    # "Glascow coma scale total": NOT itemid-based for MIMIC-IV, derived in align_and_format()

    # ---- continuous (12) ----
    "Diastolic blood pressure": [
        (220051, "chartevents", "arterial line, mmHg"),
        (220180, "chartevents", "non-invasive cuff, mmHg"),
    ],
    "Fraction inspired oxygen": [
        (223835, "chartevents", "may be pct or fraction; see CLEAN_FNS"),
        (50816, "labevents", "drawn as part of blood gas panel; may be pct or fraction; see CLEAN_FNS"),
    ],
    "Glucose": [
        (220621, "chartevents", "serum glucose, mg/dL"),
        (225664, "chartevents", "fingerstick glucose, mg/dL"),
        (50931, "labevents", "serum glucose, mg/dL"),
    ],
    "Heart Rate": [(220045, "chartevents", "bpm")],
    "Height": [(226730, "chartevents", "cm")],
    "Mean blood pressure": [
        (220052, "chartevents", "arterial line, mmHg"),
        (220181, "chartevents", "non-invasive cuff, mmHg"),
        (225312, "chartevents", "manual mean BP, mmHg"),
    ],
    "Oxygen saturation": [(220277, "chartevents", "SpO2, pct")],
    "Respiratory rate": [
        (220210, "chartevents", "breaths/min"),
        (224690, "chartevents", "breaths/min, total"),
    ],
    "Systolic blood pressure": [
        (220050, "chartevents", "arterial line, mmHg"),
        (220179, "chartevents", "non-invasive cuff, mmHg"),
    ],
    "Temperature": [
        (223761, "chartevents", "Fahrenheit or Celsius depending on unit, see CLEAN_FNS"),
        (223762, "chartevents", "Fahrenheit or Celsius depending on unit, see CLEAN_FNS"),
    ],
    "Weight": [
        (224639, "chartevents", "daily weight, kg (unit-checked in CLEAN_FNS)"),
        (226512, "chartevents", "admission weight, kg (unit-checked in CLEAN_FNS)"),
    ],
    "pH": [
        (50820, "labevents", "pH, whole blood (VBG)"),
        (50831, "labevents", "pH, arterial (ABG)"),
        (223830, "chartevents", "pH, charted"),
    ],
}

CATEGORICAL_VARS = {
    "Capillary refill rate", "Glascow coma scale eye opening",
    "Glascow coma scale motor response", "Glascow coma scale verbal response",
    "Glascow coma scale total",
}
CONTINUOUS_VARS = {
    "Diastolic blood pressure", "Fraction inspired oxygen", "Glucose",
    "Heart Rate", "Height", "Mean blood pressure", "Oxygen saturation",
    "Respiratory rate", "Systolic blood pressure", "Temperature", "Weight", "pH",
}
ALL_17_VARIABLES = sorted(CATEGORICAL_VARS | CONTINUOUS_VARS)
assert len(ALL_17_VARIABLES) == 17, f"expected 17 variables, got {len(ALL_17_VARIABLES)}"

# Placeholder for resources/variable_ranges.csv (OUTLIER_LOW/HIGH, VALID_LOW/HIGH).
# Structure once populated: {variable_name: (outlier_low, valid_low, valid_high, outlier_high)}
# Values below outlier_low / above outlier_high -> dropped (NaN).
# Values between outlier and valid bounds -> clipped to the valid bound.
# Left empty until resources/variable_ranges.csv contents are available.
OUTLIER_RANGES = {}


def _clip_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Applies OUTLIER_RANGES clipping if populated; no-op while it's empty."""
    if not OUTLIER_RANGES:
        return df
    for var, (out_low, val_low, val_high, out_high) in OUTLIER_RANGES.items():
        idx = df["variable_name"] == var
        v = df.loc[idx, "value"]
        v = v.mask(v < out_low, np.nan)
        v = v.mask(v > out_high, np.nan)
        v = v.mask(v < val_low, val_low)
        v = v.mask(v > val_high, val_high)
        df.loc[idx, "value"] = v
    return df.dropna(subset=["value"])


# ------------------------------------------------------------------------------
# 2. STAGE A -- extract_raw_events
# ------------------------------------------------------------------------------

def extract_raw_events(data_root: Path, sample_hadm_ids=None) -> pd.DataFrame:
    """
    Reads icu/chartevents.csv and hosp/labevents.csv (+ icu/d_items.csv,
    hosp/d_labitems.csv for human-readable cross-checks only -- the actual
    itemid->variable_name mapping used for extraction is VARIABLE_ITEMID_MAP
    above, not these dictionary tables) and returns ALL matching observations
    in long format, with NO joins to the cohort and NO binning/resampling.

    Columns returned: hadm_id (int64), timestamp (datetime64), variable_name
    (str), value (float64).

    Parameters
    ----------
    data_root : Path
        Root of the local MIMIC-IV v3.1 export, expected to contain hosp/ and
        icu/ subfolders (e.g. .../Data/mimic-iv-3.1).
    sample_hadm_ids : Optional[list[int]]
        If given, restricts extraction to these hadm_ids (for --sample_size /
        spot-check use). Pushed down into the SQL WHERE clause so it also
        speeds up dev iteration on the full CSVs.
    """
    chartevents_path = data_root / "icu" / "chartevents.csv"
    labevents_path = data_root / "hosp" / "labevents.csv"
    for p in (chartevents_path, labevents_path):
        if not p.exists():
            raise FileNotFoundError(f"Expected MIMIC-IV file not found: {p}")

    con = duckdb.connect()

    chart_itemids = sorted({
        iid for mapping in VARIABLE_ITEMID_MAP.values()
        for (iid, table, _note) in mapping if table == "chartevents"
    })
    lab_itemids = sorted({
        iid for mapping in VARIABLE_ITEMID_MAP.values()
        for (iid, table, _note) in mapping if table == "labevents"
    })

    itemid_to_varname = {}
    for varname, mapping in VARIABLE_ITEMID_MAP.items():
        for (iid, _table, _note) in mapping:
            itemid_to_varname[iid] = varname  # 1:1 per itemid, safe

    sample_filter_chart = ""
    sample_filter_lab = ""
    if sample_hadm_ids:
        ids_sql = ",".join(str(int(i)) for i in sample_hadm_ids)
        sample_filter_chart = f"AND hadm_id IN ({ids_sql})"
        sample_filter_lab = f"AND hadm_id IN ({ids_sql})"

    chart_itemids_sql = ",".join(str(i) for i in chart_itemids)
    chart_query = f"""
        SELECT
            hadm_id,
            charttime AS timestamp,
            itemid,
            valuenum,
            value AS value_text,
            valueuom
        FROM read_csv_auto('{chartevents_path.as_posix()}', ignore_errors=true)
        WHERE hadm_id IS NOT NULL
          AND itemid IN ({chart_itemids_sql})
          {sample_filter_chart}
    """
    chart_df = con.execute(chart_query).fetchdf()

    lab_itemids_sql = ",".join(str(i) for i in lab_itemids)
    lab_query = f"""
        SELECT
            hadm_id,
            charttime AS timestamp,
            itemid,
            valuenum,
            value AS value_text,
            valueuom
        FROM read_csv_auto('{labevents_path.as_posix()}', ignore_errors=true)
        WHERE hadm_id IS NOT NULL
          AND itemid IN ({lab_itemids_sql})
          {sample_filter_lab}
    """
    lab_df = con.execute(lab_query).fetchdf()
    con.close()

    raw = pd.concat([chart_df, lab_df], ignore_index=True)
    raw["variable_name"] = raw["itemid"].map(itemid_to_varname)
    raw = raw.dropna(subset=["variable_name", "timestamp"])

    # Dedup: MIMIC-IV mirrors some lab results into chartevents at the exact
    # same timestamp with the exact same value (e.g. pH itemid 50820 in
    # labevents gets echoed as itemid 223830 in chartevents for bedside
    # display). Since VARIABLE_ITEMID_MAP maps multiple itemids to the same
    # variable_name, these survive as duplicate rows unless collapsed here.
    # This only removes EXACT (timestamp, value) matches within the same
    # variable -- two genuinely different readings at the same timestamp
    # (e.g. a serum glucose and a fingerstick glucose that happen to differ)
    # are left alone.
    before = len(raw)
    raw = raw.drop_duplicates(subset=["hadm_id", "timestamp", "variable_name", "valuenum"], keep="first")
    print(f"  deduped {before - len(raw):,} exact-match rows across chartevents/labevents mirrors")
    raw["hadm_id"] = raw["hadm_id"].astype("int64")
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])
    raw["valueuom"] = raw["valueuom"].fillna("")
    raw["value_text"] = raw["value_text"].astype(str)

    raw["value"] = _clean_events(raw)
    raw = raw.dropna(subset=["value"])

    out = raw[["hadm_id", "timestamp", "variable_name", "value"]].copy()
    out = _clip_outliers(out)
    return out.reset_index(drop=True)


# ------------------------------------------------------------------------------
# 2b. Per-variable cleaning, ported from mimic4extract/ehr_utils/preprocessing.py
#     (clean_crr, clean_sbp, clean_dbp, clean_fio2, clean_o2sat, clean_lab,
#     clean_temperature, clean_weight, clean_height) -- adapted to MIMIC-IV
#     column names (value/valuenum/valueuom instead of value/valuenum/valuenum-uom).
# ------------------------------------------------------------------------------

_NUMERIC_RE = re.compile(r"^(\d+(\.\d*)?|\.\d+)$")
_BP_PAIR_RE = re.compile(r"^(\d+)/(\d+)$")


def _clean_crr(df: pd.DataFrame) -> pd.Series:
    v = pd.Series(np.nan, index=df.index)
    text = df["value_text"].str.strip()
    v.loc[text.isin(["Normal <3 Seconds", "Brisk"])] = 0.0
    v.loc[text.isin(["Abnormal >3 Seconds", "Delayed"])] = 1.0
    return v


def _clean_bp_pair(df: pd.DataFrame, group: int) -> pd.Series:
    v = df["valuenum"].astype(float).copy()
    matches = df["value_text"].str.match(_BP_PAIR_RE)
    for idx in df.index[matches.fillna(False)]:
        m = _BP_PAIR_RE.match(df.loc[idx, "value_text"])
        v.loc[idx] = float(m.group(group))
    return v


def _clean_sbp(df: pd.DataFrame) -> pd.Series:
    return _clean_bp_pair(df, group=1)


def _clean_dbp(df: pd.DataFrame) -> pd.Series:
    return _clean_bp_pair(df, group=2)


def _clean_fio2(df: pd.DataFrame) -> pd.Series:
    v = df["valuenum"].astype(float).copy()
    is_torr = df["valueuom"].str.lower().str.contains("torr")
    idx = (~is_torr) & (v > 1.0)
    v.loc[idx] = v.loc[idx] / 100.0
    return v


def _clean_o2sat(df: pd.DataFrame) -> pd.Series:
    v = df["valuenum"].astype(float).copy()
    idx = v <= 1
    v.loc[idx] = v.loc[idx] * 100.0
    return v


def _clean_lab(df: pd.DataFrame) -> pd.Series:
    # Glucose / pH: reject non-numeric raw strings (e.g. "ERROR") rather than
    # coercing them; only cleanly-numeric raw text is kept.
    is_numeric = df["value_text"].str.match(_NUMERIC_RE)
    v = pd.Series(np.nan, index=df.index)
    v.loc[is_numeric.fillna(False)] = df.loc[is_numeric.fillna(False), "value_text"].astype(float)
    return v


def _clean_temperature(df: pd.DataFrame) -> pd.Series:
    v = df["valuenum"].astype(float).copy()
    is_f = df["valueuom"].str.lower().str.contains("f") | (v >= 79)
    v.loc[is_f] = (v.loc[is_f] - 32.0) * 5.0 / 9.0
    return v


def _clean_weight(df: pd.DataFrame) -> pd.Series:
    v = df["valuenum"].astype(float).copy()
    uom = df["valueuom"].str.lower()
    is_oz = uom.str.contains("oz")
    v.loc[is_oz] = v.loc[is_oz] / 16.0
    is_lb = is_oz | uom.str.contains("lb")
    v.loc[is_lb] = v.loc[is_lb] * 0.453592
    return v


def _clean_height(df: pd.DataFrame) -> pd.Series:
    v = df["valuenum"].astype(float).copy()
    is_in = df["valueuom"].str.lower().str.contains("in")
    v.loc[is_in] = np.round(v.loc[is_in] * 2.54)
    return v

def _clean_gcs_verbal(df: pd.DataFrame) -> pd.Series:
    """
    Maps MIMIC-IV's 'No Response-ETT' (intubated, verbal unassessable) to 0,
    matching the convention verified against mimic-code's gcs.sql and already
    used in label_sepsis3.py's SOFA CNS component (PB3). This 0 is a sentinel
    consumed by _derive_gcs_total() below, NOT a real "worst possible verbal
    score" to be used on its own.
    """
    v = df["valuenum"].astype(float).copy()
    is_ett = df["value_text"].str.strip() == "No Response-ETT"
    v.loc[is_ett] = 0.0
    return v


CLEAN_FNS = {
    "Capillary refill rate": _clean_crr,
    "Systolic blood pressure": _clean_sbp,
    "Diastolic blood pressure": _clean_dbp,
    "Fraction inspired oxygen": _clean_fio2,
    "Oxygen saturation": _clean_o2sat,
    "Glucose": _clean_lab,
    "pH": _clean_lab,
    "Temperature": _clean_temperature,
    "Weight": _clean_weight,
    "Height": _clean_height,
    "Glascow coma scale verbal response": _clean_gcs_verbal,
    # Not in CLEAN_FNS (used as valuenum directly, no special handling),
    # matching the reference: Heart Rate, Respiratory rate,
    # Mean blood pressure, and the 3 raw GCS sub-scores.
}


def _clean_events(raw: pd.DataFrame) -> pd.Series:
    value = raw["valuenum"].astype(float).copy()
    for var_name, fn in CLEAN_FNS.items():
        idx = raw["variable_name"] == var_name
        if idx.any():
            value.loc[idx] = fn(raw.loc[idx])
    return value

def _derive_gcs_total(events: pd.DataFrame) -> pd.DataFrame:
    """
    Faithful port of label_sepsis3.py's PB3 GCS-total logic (itself a port of
    mimic-code's measurement/gcs.sql), applied at the hadm_id grain instead
    of stay_id -- equivalent for this cohort, since build_cohort() already
    restricts to exactly one ICU stay per admission.

    Unlike the PREVIOUS version of this function (which required all three
    components at the EXACT same timestamp, dropping every partial-charting
    row), this ports mimic-code's carry-forward rule: a component missing at
    the current timestamp can use the PREVIOUS charted value of that same
    component, as long as that previous reading is within 6h. This changes
    both coverage (more rows now qualify) and correctness (intubated
    patients no longer get penalized for an unassessable verbal component).

    Branch logic (verbatim from gcs.sql / PB3), b = current row, b2 = the
    previous row (within 6h) for the same hadm_id:
      1. b.verbal == 0 (current row itself is "No Response-ETT",
         pre-mapped to 0 by _clean_gcs_verbal)               -> total = 15
      2. b.verbal IS NULL AND b2.verbal == 0                 -> total = 15
      3. b2.verbal == 0 (and b.verbal is real, non-null/non-zero)
         -> total = COALESCE(b.motor,6) + COALESCE(b.verbal,5) + COALESCE(b.eyes,4)
            (current row's own components only -- a fresh non-ETT verbal
            reading is trusted immediately, no b2 fallback)
      4. otherwise (normal carry-forward, componentwise)
         -> total = COALESCE(b.motor, COALESCE(b2.motor,6))
                   + COALESCE(b.verbal, COALESCE(b2.verbal,5))
                   + COALESCE(b.eyes, COALESCE(b2.eyes,4))
    """
    gcs_component_names = [
        "Glascow coma scale eye opening",
        "Glascow coma scale motor response",
        "Glascow coma scale verbal response",
    ]
    gcs_components = events[events["variable_name"].isin(gcs_component_names)]
    if gcs_components.empty:
        return pd.DataFrame(columns=["hadm_id", "timestamp", "variable_name", "value"])

    pivoted = gcs_components.pivot_table(
        index=["hadm_id", "timestamp"],
        columns="variable_name",
        values="value",
        aggfunc="first",
    ).reset_index()
    for c in gcs_component_names:
        if c not in pivoted.columns:
            pivoted[c] = np.nan
    pivoted = pivoted.rename(columns={
        "Glascow coma scale eye opening": "eyes",
        "Glascow coma scale motor response": "motor",
        "Glascow coma scale verbal response": "verbal",
    })
    pivoted = pivoted.sort_values(["hadm_id", "timestamp"]).reset_index(drop=True)

    grp = pivoted.groupby("hadm_id")
    pivoted["motor_prev"] = grp["motor"].shift(1)
    pivoted["verbal_prev"] = grp["verbal"].shift(1)
    pivoted["eyes_prev"] = grp["eyes"].shift(1)
    prev_timestamp = grp["timestamp"].shift(1)
    within_6h = (pivoted["timestamp"] - prev_timestamp) <= pd.Timedelta(hours=6)
    # matches gcs.sql's b2 JOIN condition exactly: outside 6h, treat as absent
    pivoted.loc[~within_6h.fillna(False), ["motor_prev", "verbal_prev", "eyes_prev"]] = np.nan

    b_verbal = pivoted["verbal"]
    b2_verbal = pivoted["verbal_prev"]

    branch1 = b_verbal == 0
    branch2 = b_verbal.isna() & (b2_verbal == 0)
    branch3 = (~branch1) & (~branch2) & (b2_verbal == 0)
    # branch4 = everything else (default)

    total_branch1_2 = pd.Series(15.0, index=pivoted.index)
    total_branch3 = (
        pivoted["motor"].fillna(6) + pivoted["verbal"].fillna(5) + pivoted["eyes"].fillna(4)
    )
    total_branch4 = (
        pivoted["motor"].fillna(pivoted["motor_prev"]).fillna(6)
        + pivoted["verbal"].fillna(pivoted["verbal_prev"]).fillna(5)
        + pivoted["eyes"].fillna(pivoted["eyes_prev"]).fillna(4)
    )

    gcs_total = pd.Series(np.nan, index=pivoted.index, dtype="float64")
    gcs_total = gcs_total.mask(branch1 | branch2, total_branch1_2)
    gcs_total = gcs_total.mask(branch3, total_branch3)
    gcs_total = gcs_total.mask(gcs_total.isna(), total_branch4)

    out = pivoted[["hadm_id", "timestamp"]].copy()
    out["variable_name"] = "Glascow coma scale total"
    out["value"] = gcs_total
    return out
# ------------------------------------------------------------------------------
# 3. STAGE B -- align_and_format
# ------------------------------------------------------------------------------

def align_and_format(raw_events: pd.DataFrame, cohort_labels: pd.DataFrame) -> pd.DataFrame:
    """
    Derives gcs_total, inner-joins raw_events against the Sepsis-3 cohort on
    hadm_id, and computes hours_before_onset = (sepsis_onset_time - timestamp)
    in hours. Returns the final schema:
        hadm_id (int), timestamp (datetime), variable_name (str),
        value (float), hours_before_onset (float)

    cohort_labels must contain columns: hadm_id, sepsis_onset_time.
    """
    required_cols = {"hadm_id", "sepsis_onset_time"}
    missing = required_cols - set(cohort_labels.columns)
    if missing:
        raise ValueError(f"cohort_labels missing required columns: {missing}")

    events = raw_events.copy()

    # --- derive "Glascow coma scale total" (see _derive_gcs_total for the
    #     full carry-forward/intubation-aware logic, ported from
    #     label_sepsis3.py's PB3 / mimic-code's gcs.sql) ---
    gcs_total_rows = _derive_gcs_total(events)
    if not gcs_total_rows.empty:
        events = pd.concat([events, gcs_total_rows], ignore_index=True)

    # --- inner join against cohort ---
    cohort = cohort_labels[["hadm_id", "sepsis_onset_time"]].copy()
    cohort["hadm_id"] = cohort["hadm_id"].astype("int64")
    cohort["sepsis_onset_time"] = pd.to_datetime(cohort["sepsis_onset_time"])

    merged = events.merge(cohort, on="hadm_id", how="inner")
    merged["hours_before_onset"] = (
        (merged["sepsis_onset_time"] - merged["timestamp"]).dt.total_seconds() / 3600.0
    )

    final = merged[["hadm_id", "timestamp", "variable_name", "value", "hours_before_onset"]].copy()
    final["hadm_id"] = final["hadm_id"].astype("int64")
    final["value"] = final["value"].astype("float64")
    final["hours_before_onset"] = final["hours_before_onset"].astype("float64")
    return final.reset_index(drop=True)


# ------------------------------------------------------------------------------
# 4. Spot-check utility
# ------------------------------------------------------------------------------

def spot_check(data_root: Path, final_df: pd.DataFrame, hadm_ids: list, n=3):
    """
    Prints raw chartevents/labevents rows for `hadm_ids` alongside the
    corresponding rows in final_df, so you can manually verify timestamps,
    values, and hours_before_onset against the source tables.
    """
    ids = hadm_ids[:n]
    print(f"\n{'='*80}\nSPOT-CHECK: hadm_ids = {ids}\n{'='*80}")
    for hid in ids:
        print(f"\n--- hadm_id {hid} : final_df rows (first 10) ---")
        subset = final_df[final_df["hadm_id"] == hid].sort_values("timestamp")
        print(subset.head(10).to_string(index=False))
        print(f"    -> {len(subset)} total rows for this hadm_id in final output")
    print(
        "\nManually cross-reference a few of the (timestamp, value) pairs above "
        "against `chartevents.csv` / `labevents.csv` filtered to this hadm_id "
        "and the relevant itemid from VARIABLE_ITEMID_MAP, and confirm "
        "hours_before_onset = (sepsis_onset_time - timestamp)/3600 by hand."
    )


# ------------------------------------------------------------------------------
# 5. Summary stats
# ------------------------------------------------------------------------------

def print_summary(final_df: pd.DataFrame, cohort_labels: pd.DataFrame):
    n_cohort = cohort_labels["hadm_id"].nunique()
    n_rows = len(final_df)
    n_hadm_final = final_df["hadm_id"].nunique()

    print(f"\n{'='*80}\nEXTRACTION SUMMARY\n{'='*80}")
    print(f"Total rows in ehr_timeseries.parquet : {n_rows:,}")
    print(f"Unique hadm_ids in final output        : {n_hadm_final:,}")
    print(f"Unique hadm_ids in sepsis_labels cohort : {n_cohort:,}")
    if n_hadm_final != n_cohort:
        print(
            "  WARNING: mismatch -- some cohort admissions have ZERO extracted "
            "observations across all 17 variables. Investigate before merging."
        )
    else:
        print("  OK: hadm_id count matches cohort exactly.")

    print(f"\nPer-variable coverage (% of cohort hadm_ids with >=1 observation):")
    for var in ALL_17_VARIABLES:
        have_var = final_df.loc[final_df["variable_name"] == var, "hadm_id"].nunique()
        pct = 100.0 * have_var / n_cohort if n_cohort else 0.0
        print(f"  {var:24s} {pct:6.2f}%  ({have_var}/{n_cohort})")


# ------------------------------------------------------------------------------
# 6. CLI
# ------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Extract MIMIC-IV EHR time-series for sepsis cohort.")
    parser.add_argument(
        "--data_root", type=str, required=True,
        help=r"Path to MIMIC-IV v3.1 root, e.g. D:\...\Data\mimic-iv-3.1 (must contain hosp/ and icu/)",
    )
    parser.add_argument(
        "--labels_path", type=str, default="data/cohort/sepsis_labels.parquet",
        help="Path to Milestone-0 sepsis_labels.parquet",
    )
    parser.add_argument(
        "--output", type=str, default="data/cohort/ehr_timeseries.parquet",
        help="Output path for the long-format EHR parquet",
    )
    parser.add_argument(
        "--sample_size", type=int, default=None,
        help="If set, restrict extraction to this many hadm_ids (from the cohort) for a fast dev run.",
    )
    parser.add_argument(
        "--spot_check_n", type=int, default=3,
        help="Number of hadm_ids to print for manual spot-checking.",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    labels_path = Path(args.labels_path)
    output_path = Path(args.output)

    if not labels_path.exists():
        print(f"ERROR: {labels_path} not found. Run Milestone 0 (label_sepsis3.py) first.", file=sys.stderr)
        sys.exit(1)

    cohort_labels = pd.read_parquet(labels_path)
    if "hadm_id" not in cohort_labels.columns or "sepsis_onset_time" not in cohort_labels.columns:
        print(
            "ERROR: sepsis_labels.parquet must contain 'hadm_id' and 'sepsis_onset_time' columns.",
            file=sys.stderr,
        )
        sys.exit(1)

    # BUG FIX: sample_hadm_ids used to be None whenever --sample_size wasn't
    # passed, which meant extract_raw_events() got no hadm_id filter at all
    # on a full run -- it scanned/extracted matching-itemid rows for the
    # ENTIRE hospital population, not just this cohort, and only discarded
    # the excess afterward in Stage B's join. Now it's ALWAYS derived from
    # cohort_labels: the full cohort by default, or a random subset of it
    # when --sample_size is set. sample_hadm_ids is never None again, so
    # extract_raw_events() always receives a real, correct filter.
    all_cohort_hadm_ids = cohort_labels["hadm_id"].drop_duplicates().tolist()
    if args.sample_size is not None:
        sample_hadm_ids = (
            cohort_labels["hadm_id"].drop_duplicates().sample(
                n=min(args.sample_size, cohort_labels["hadm_id"].nunique()),
                random_state=0,
            ).tolist()
        )
        print(f"--sample_size set: restricting to {len(sample_hadm_ids)} hadm_ids for this run.")
    else:
        sample_hadm_ids = all_cohort_hadm_ids
        print(f"No --sample_size: extracting for the full cohort ({len(sample_hadm_ids):,} hadm_ids).")

    print("Stage A: extracting raw events from chartevents/labevents ...")
    raw_events = extract_raw_events(data_root, sample_hadm_ids=sample_hadm_ids)
    print(f"  extracted {len(raw_events):,} raw observation rows.")

    print("Stage B: joining against cohort and computing hours_before_onset ...")
    cohort_for_join = cohort_labels[cohort_labels["hadm_id"].isin(sample_hadm_ids)]
    final_df = align_and_format(raw_events, cohort_for_join)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_parquet(output_path, index=False)
    print(f"Wrote {output_path} ({len(final_df):,} rows).")

    print_summary(final_df, cohort_for_join)

    hadm_ids_present = final_df["hadm_id"].drop_duplicates().tolist()
    spot_check(data_root, final_df, hadm_ids_present, n=args.spot_check_n)


if __name__ == "__main__":
    main()
