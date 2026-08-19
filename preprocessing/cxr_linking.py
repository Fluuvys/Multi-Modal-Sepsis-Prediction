from __future__ import annotations
"""
preprocessing/cxr_linking.py
================================================================================
PURPOSE:
    Join MIMIC-CXR studies to admissions in our locked cohort
    (data/cohort/sepsis_labels.parquet, Milestone 0 -- see
    preprocessing/label_sepsis3.py, NOT reimplemented here), keeping the FULL
    per-patient CXR sequence with real timestamps -- NOT just the most-recent
    one. This is the exact limitation MedPatch's own paper names as future
    work; we don't repeat it. Produces the single shared
    data/cohort/cxr_metadata.parquet that every baseline reproduction AND our
    own model consume -- see PROJECT_CONTEXT.md rule #5: no model-specific
    reduction (most-recent-CXR-only, etc.) happens in this file. That logic
    goes in each model's own adapter under models/baselines/ or models/ours/.
    This file's job stops at: real timestamps, a resized image on disk, and a
    precomputed hours_before_onset per study -- the minimum shared fact every
    downstream consumer needs and none of them should recompute differently.

RELATIONSHIP TO MILESTONE 0:
    Depends on data/cohort/sepsis_labels.parquet already existing (see
    label_sepsis3.py). We join against it to (a) restrict to admissions in
    the final cohort (excluded_reason IS NULL) and (b) pull sepsis_onset_time
    to compute hours_before_onset. We do NOT reimplement or second-guess any
    cohort/labeling logic here.

WHY THIS FILE ALSO NEEDS hosp/admissions.csv (a dependency notes_extraction.py
    did NOT have -- read this before running):
    sepsis_labels.parquet (per docs/data_schema.md) carries subject_id and
    hadm_id but NOT admittime/dischtime. MIMIC-CXR studies are keyed by
    subject_id only -- there is no hadm_id on a CXR study row. A patient can
    have multiple admissions in the cohort (or admissions outside the
    cohort entirely -- excluded stays, ED-only visits, prior hospitalizations
    the patient had before the cohort's index stay), so attributing a CXR to
    "the" hadm_id for that subject_id is only correct if we actually check
    which admission's time window the study's timestamp falls inside. That
    window (admittime, dischtime) lives in MIMIC-IV core's hosp/admissions
    table, not in anything Milestone 0 produced -- so this file additionally
    requires --mimic_iv_hosp_dir pointing at the folder containing
    admissions.csv[.gz]. See ASSUMPTION A1 for exactly how the window check
    works and A2 for what happens to CXRs that don't fall inside ANY cohort
    admission's window (unavoidably dropped, same spirit as notes_extraction
    .py's null-hadm_id radiology rows).

INPUT:
    MIMIC-CXR-JPG metadata (mimic-cxr-2.0.0-metadata.csv[.gz], see
        ASSUMPTION A3 for exactly which columns are used and how
        StudyDate/StudyTime are combined into a single timestamp)
        columns used: subject_id (int), study_id (int), dicom_id (str),
        StudyDate (int, YYYYMMDD), StudyTime (float, HHMMSS.ffffff, no
        guaranteed leading zeros)
    MIMIC-CXR-JPG image files (files/p<shard>/p<subject_id>/s<study_id>/
        <dicom_id>.jpg under --mimic_cxr_dir) -- read only to resize, per
        MedPatch's own resize.py convention (see ASSUMPTION A4). We do NOT
        read pixel data for any other purpose in this file.
    hosp/admissions.csv[.gz] (MIMIC-IV core, NOT part of Milestone 0's
        output) -- subject_id, hadm_id, admittime, dischtime. Used ONLY to
        resolve which cohort hadm_id a CXR belongs to (see above). We do NOT
        read anything else from hosp/ or icu/ -- all cohort membership and
        onset timing still comes through sepsis_labels.parquet only, per
        PROJECT_CONTEXT.md rule #2.

OUTPUT: data/cohort/cxr_metadata.parquet, exactly these 5 columns (locked,
    per the task's Definition of Done):
        hadm_id             int       join key
        study_id            string
        dicom_id             string
        timestamp           datetime  raw study time (StudyDate + StudyTime)
        image_path          string    path to the RESIZED jpg on disk, not
                                       the original -- see ASSUMPTION A4
        hours_before_onset  float     precomputed; NaN for admissions with
                                       no onset (label=0, included-negative)
    Also writes data/cohort/cxr_stats.json (counts, misattribution/leakage
    diagnostics) and data/cohort/cxr_spot_check.json (manual-verification
    aid, DoD requirement, specifically exercises the multi-admission
    cross-attribution case).

TODO checklist (Milestone 1 -- EHR/notes/CXR preprocessing pipelines):
    [x] Join CXR studies to admissions -- via real admission time-window
        overlap, not subject_id alone (ASSUMPTION A1)
    [x] Preserve full per-patient CXR sequence + timestamps (every dicom_id
        kept as its own row, nothing collapsed to most-recent)
    [x] Decide + document image preprocessing (ASSUMPTION A4 -- verified
        directly against MedPatch's own resize.py source, not guessed)
    [x] cxr_stats.json (required counts + misattribution diagnostics)
    [x] spot_check_cxr() (re-queries raw metadata directly, per DoD, and
        specifically targets a multi-admission subject)
    [ ] Team review of ASSUMPTIONS A1-A6 below, especially A1 (the admission-
        window join) and A4 (image resizing)
    [ ] PROJECT_CONTEXT.md sec 7 checklist item, same PR or same-day follow-up

================================================================================
ASSUMPTIONS / DEVIATIONS FLAGGED FOR TEAM REVIEW -- read before trusting output
================================================================================
Search "ASSUMPTION" in this file for the code sites. Summary:

A1. ADMISSION-WINDOW JOIN -- read this one first.
    A CXR study (subject_id, study_id, dicom_id, timestamp) is attributed to
    a cohort hadm_id IF AND ONLY IF: (a) that hadm_id belongs to the same
    subject_id, (b) that hadm_id is in the eligible cohort (excluded_reason
    IS NULL), AND (c) admittime <= timestamp <= dischtime for that specific
    hadm_id's admission window (from hosp/admissions.csv). This is a real
    interval join, not a subject_id-only join -- a patient with two
    admissions in the cohort gets their CXRs split correctly between the two
    stays based on which window the timestamp actually falls in. If (rare --
    e.g. same-day transfer/readmission) a timestamp falls inside more than
    one matching admission's window, we deterministically pick the admission
    with the EARLIEST admittime among the matches (ROW_NUMBER() ... ORDER BY
    admittime ASC) and log the count of ambiguous cases in cxr_stats.json
    ("n_cxr_matched_multiple_admissions") -- if that count is ever non-
    trivial on your real data, this tie-break policy needs team discussion,
    not silent trust.

A2. CXRs THAT MATCH ZERO ADMISSION WINDOWS are unavoidably dropped -- can't
    compute hours_before_onset or attribute to a cohort hadm_id without one.
    This includes: CXRs taken during a hospitalization that was EXCLUDED
    from the cohort (e.g. sepsis_within_4h_of_admission, los_under_12h),
    CXRs taken during a DIFFERENT hospitalization the same subject_id had
    that isn't in our cohort at all (prior/later unrelated admission), and
    ED-only or outpatient imaging with no inpatient admission window at all.
    Counted via compute_raw_table_counts() / n_cxr_unmatched_to_any_admission
    in cxr_stats.json, not silently discarded. This is the CXR-specific
    analogue of notes_extraction.py's ASSUMPTION A7 (null-hadm_id radiology
    rows).

A3. Timestamp construction: StudyDate (int, YYYYMMDD) and StudyTime (float,
    HHMMSS.ffffff -- MIMIC-CXR-JPG does NOT zero-pad this field, so 94032.5
    means 09:40:32.5, not 94:03:2.5) are combined via
    parse_study_datetime() below. I could not verify against your actual
    metadata CSV (no MIMIC-CXR access from where this was written) that
    every row has both fields non-null -- rows with a null StudyDate or
    StudyTime are dropped and counted separately in cxr_stats.json
    ("n_dropped_null_study_datetime") rather than silently coerced to NaT
    and passed downstream, since a NaT timestamp would fail every downstream
    admission-window and hours_before_onset computation anyway.

A4. IMAGE RESIZING -- verified directly against MedPatch's actual
    github.com/nyuad-cai/MedPatch/blob/main/resize.py source (not guessed,
    not reconstructed from the paper): they resize to a FIXED WIDTH of 512
    pixels, preserving aspect ratio (hsize = round(height * (512/width))),
    using PIL's Image.resize() with NO explicit resample filter passed (so
    whatever your installed Pillow version's default resample is -- this
    was NOT pinned in their script, flagged here rather than silently
    picking a filter they didn't specify). Their own output directory
    convention is a FLAT `resized/` folder keyed only by original filename
    (i.e. `resized/{dicom_id}.jpg`, no subject/study subfolders) -- we
    replicate that exactly (see IMAGE_RESIZE_BASEWIDTH,
    resize_and_cache_image() below) so image_path values in the shared
    output are directly comparable to what a MedPatch reproduction expects.
    Deviations from their script, both deliberate: (1) we drive resizing
    from the metadata CSV + admission join (so we only ever resize images
    that actually belong to our cohort, not every file under `files/` via
    glob like their script does -- resizing all ~377K MIMIC-CXR-JPG images
    when our cohort only touches a subset would be wasted compute), and
    (2) we use a thread pool sized by --resize_threads (default 10, matching
    their hardcoded value) rather than their hardcoded 10, so this is
    tunable per machine.
    TEAM ACTION: confirm your Pillow version's default resize() resample
    filter is acceptable, or pass --resize_resample_filter explicitly if a
    specific baseline reproduction needs bit-for-bit matching.

A5. Multiple dicom_id rows per study_id (different views -- PA/AP/lateral)
    are ALL kept as separate rows, per the "full sequence, not just most
    recent" requirement -- not deduplicated or reduced to one view per
    study here. Each dicom_id gets its own resized image and its own
    hours_before_onset (identical across dicom_ids in the same study, since
    they share a StudyDate/StudyTime).

A6. --sample_size samples eligible hadm_ids (not raw CXR rows),
    deterministically, ORDERED before sampling -- mirrors the same
    reproducibility fix documented in label_sepsis3.py's build_cohort() and
    notes_extraction.py's ASSUMPTION A9 (unordered results + .sample()
    silently sampling different rows across runs). Don't remove the
    .sort_values() below without re-reading those files' comments.
================================================================================
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import duckdb
import numpy as np
import pandas as pd
from PIL import Image

# ==============================================================================
# CONFIG / CONSTANTS
# ==============================================================================

# MedPatch resize.py convention -- see ASSUMPTION A4. Verified against live source.
IMAGE_RESIZE_BASEWIDTH = 512
DEFAULT_RESIZE_THREADS = 10

OUTPUT_COLUMNS = ["hadm_id", "study_id", "dicom_id", "timestamp", "image_path",
                  "hours_before_onset", "hours_since_admission"]

CXR_DATETIME_SOURCE_COLS = ["StudyDate", "StudyTime"]
ADMISSIONS_DATETIME_COLS = ["admittime", "dischtime"]


# ==============================================================================
# DATA LOADING (duckdb) -- kept simple (SELECT / JOIN / CAST / window
# function only) for the same reason as notes_extraction.py: it's the one
# part of this file that can't be unit-tested with plain pandas fixtures.
# Business logic (admission-window resolution, hours_before_onset, stats,
# spot-check comparison) lives in the pure-pandas section further down.
# ==============================================================================

def _ensure_parquet_cache(view_name: str, csv_path: Path, cache_dir: Path) -> Path:
    """One-time conversion of a raw CSV to Parquet, cached under cache_dir.
    Same convention as label_sepsis3.py / notes_extraction.py's helper of the
    same name (kept as a separate copy since this file must run standalone --
    if the team later adds a shared preprocessing/_io_utils.py, all three
    copies should move there instead of a fourth one being written)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = cache_dir / f"{view_name}.parquet"
    if parquet_path.exists() and parquet_path.stat().st_mtime >= csv_path.stat().st_mtime:
        return parquet_path

    print(f"[cxr_linking]   Building Parquet cache for {view_name} "
          f"(one-time cost, reused on every future run)...", file=sys.stderr)
    t0 = time.time()
    tmp_con = duckdb.connect(database=":memory:")
    tmp_con.execute("PRAGMA threads=4;")
    tmp_con.execute(f"""
        COPY (
            SELECT * FROM read_csv('{csv_path.as_posix()}',
                parallel=false, ignore_errors=false)
        ) TO '{parquet_path.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD);
    """)
    tmp_con.close()
    print(f"[cxr_linking]   -> {view_name}.parquet written in "
          f"{time.time() - t0:.0f}s", file=sys.stderr)
    return parquet_path


def _resolve_existing_path(path: Path) -> Path:
    """Falls back to a .gz sibling if the plain path doesn't exist -- same
    convention as notes_extraction.py's _register_note_view()."""
    if path.exists():
        return path
    gz_path = path.with_suffix(path.suffix + ".gz")
    if gz_path.exists():
        return gz_path
    raise FileNotFoundError(
        f"Expected file not found (checked both {path} and {gz_path})."
    )


def connect_duckdb(cxr_metadata_path: Path, admissions_path: Path,
                    cache_dir: Optional[Path] = None) -> duckdb.DuckDBPyConnection:
    """Registers `cxr_metadata` and `admissions` views. CXR metadata columns
    are cast explicitly so DuckDB's type sniffer doesn't silently fall back
    to VARCHAR on a sparse column; admissions' datetime columns likewise."""
    con = duckdb.connect(database=":memory:")
    con.execute("PRAGMA threads=4;")

    cxr_path = _resolve_existing_path(cxr_metadata_path)
    adm_path = _resolve_existing_path(admissions_path)

    if cache_dir is not None:
        cxr_parquet = _ensure_parquet_cache("cxr_metadata_raw", cxr_path, cache_dir)
        adm_parquet = _ensure_parquet_cache("admissions_raw", adm_path, cache_dir)
        con.execute(f"CREATE OR REPLACE VIEW cxr_metadata_raw AS "
                    f"SELECT * FROM read_parquet('{cxr_parquet.as_posix()}');")
        con.execute(f"CREATE OR REPLACE VIEW admissions_raw AS "
                    f"SELECT * FROM read_parquet('{adm_parquet.as_posix()}');")
    else:
        con.execute(f"CREATE OR REPLACE VIEW cxr_metadata_raw AS "
                    f"SELECT * FROM read_csv('{cxr_path.as_posix()}', "
                    f"parallel=false, ignore_errors=false);")
        con.execute(f"CREATE OR REPLACE VIEW admissions_raw AS "
                    f"SELECT * FROM read_csv('{adm_path.as_posix()}', "
                    f"parallel=false, ignore_errors=false);")

    casts = ", ".join(f"CAST({c} AS TIMESTAMP) AS {c}" for c in ADMISSIONS_DATETIME_COLS)
    con.execute(
        f"CREATE OR REPLACE VIEW admissions AS "
        f"SELECT * REPLACE ({casts}) FROM admissions_raw;"
    )
    con.execute("CREATE OR REPLACE VIEW cxr_metadata AS SELECT * FROM cxr_metadata_raw;")
    return con


def compute_raw_table_counts(con: duckdb.DuckDBPyConnection) -> dict:
    """Row counts for the FULL raw tables (independent of --sample_size and
    of cohort membership) -- denominators for reporting drop rates."""
    counts = {}
    for view in ("cxr_metadata", "admissions"):
        total = con.execute(f"SELECT COUNT(*) FROM {view}").fetchone()[0]
        counts[view] = {"total_rows": int(total)}
    null_study_dt = con.execute(
        "SELECT COUNT(*) FROM cxr_metadata WHERE StudyDate IS NULL OR StudyTime IS NULL"
    ).fetchone()[0]
    counts["cxr_metadata"]["null_study_date_or_time_rows"] = int(null_study_dt)
    return counts


def extract_cxr_with_admission_match(con: duckdb.DuckDBPyConnection,
                                      cohort_hadm_subject_df: pd.DataFrame) -> pd.DataFrame:
    """
    THE core admission-window join -- see ASSUMPTION A1. Registers the
    eligible cohort's (hadm_id, subject_id, admittime, dischtime) as a temp
    view, joins CXR rows to it on subject_id AND a real timestamp-within-
    window check (not subject_id alone), and deterministically picks the
    earliest-admittime match if more than one admission window contains a
    given study's timestamp.

    `cohort_hadm_subject_df` must already carry admittime/dischtime pulled
    from hosp/admissions.csv (see load_cohort_admission_windows()) -- this
    function does not touch sepsis_onset_time at all; that join happens
    later in compute_hours_before_onset(), same separation-of-concerns
    pattern as notes_extraction.py.
    """
    con.register("cohort_admissions", cohort_hadm_subject_df)
    query = """
        WITH parsed_cxr AS (
            SELECT
                subject_id, study_id, dicom_id, StudyDate, StudyTime,
                -- combine YYYYMMDD int + HHMMSS.ffffff float into TIMESTAMP.
                -- StudyTime is NOT zero-padded (e.g. 940.5 == 00:09:40.5),
                -- so we zero-pad the integer-seconds part to 6 digits before
                -- parsing -- see ASSUMPTION A3.
                CASE WHEN StudyDate IS NULL OR StudyTime IS NULL THEN NULL ELSE
                    strptime(
                        printf('%08d%06d', CAST(StudyDate AS BIGINT), CAST(FLOOR(StudyTime) AS BIGINT)),
                        '%Y%m%d%H%M%S'
                    ) + to_seconds(StudyTime - FLOOR(StudyTime))
                END AS timestamp
            FROM cxr_metadata
        ),
        matched AS (
            SELECT
                p.subject_id, p.study_id, p.dicom_id, p.timestamp,
                a.hadm_id, a.admittime, a.dischtime,
                ROW_NUMBER() OVER (
                    PARTITION BY p.study_id, p.dicom_id
                    ORDER BY a.admittime ASC
                ) AS rn,
                COUNT(*) OVER (PARTITION BY p.study_id, p.dicom_id) AS n_matches
            FROM parsed_cxr p
            INNER JOIN cohort_admissions a
                ON p.subject_id = a.subject_id
                AND p.timestamp >= a.admittime
                AND p.timestamp <= a.dischtime
            WHERE p.timestamp IS NOT NULL
        )
        SELECT subject_id, study_id, dicom_id, timestamp, hadm_id, n_matches
        FROM matched
        WHERE rn = 1
    """
    df = con.execute(query).df()
    con.unregister("cohort_admissions")
    return df


def _query_raw_cxr_for_subject(con: duckdb.DuckDBPyConnection, subject_id: int) -> pd.DataFrame:
    """Small, independent re-query of the raw CXR metadata for ONE subject --
    used only by spot_check_cxr(), deliberately NOT reusing
    extract_cxr_with_admission_match()'s machinery, so the comparison is
    meaningful. Parses timestamp with plain Python/pandas, not the SQL
    expression above, for a genuinely independent check."""
    raw = con.execute(
        f"SELECT subject_id, study_id, dicom_id, StudyDate, StudyTime "
        f"FROM cxr_metadata WHERE subject_id = {int(subject_id)}"
    ).df()
    raw["timestamp"] = parse_study_datetime(raw["StudyDate"], raw["StudyTime"])
    return raw


def _query_admission_windows_for_subject(con: duckdb.DuckDBPyConnection, subject_id: int) -> pd.DataFrame:
    return con.execute(
        f"SELECT hadm_id, subject_id, admittime, dischtime FROM admissions "
        f"WHERE subject_id = {int(subject_id)}"
    ).df()


# ==============================================================================
# PURE TRANSFORMATION LOGIC -- pandas only, no duckdb. Exercised directly by
# the test suite against hand-built fixtures.
# ==============================================================================

def load_sepsis_cohort(sepsis_labels_path: Path) -> pd.DataFrame:
    """Loads Milestone 0's output, restricted to the final cohort
    (excluded_reason IS NULL). Same logic as notes_extraction.py's
    load_sepsis_cohort() -- kept as its own copy per that file's own note
    about a future shared _io_utils.py."""
    if not sepsis_labels_path.exists():
        raise FileNotFoundError(
            f"{sepsis_labels_path} not found. This script depends on "
            f"Milestone 0 (preprocessing/label_sepsis3.py) -- run that "
            f"first; see PROJECT_CONTEXT.md sec 6 repo map."
        )
    df = pd.read_parquet(sepsis_labels_path)
    required_cols = {"subject_id", "hadm_id", "sepsis_onset_time", "excluded_reason", "icu_intime"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"{sepsis_labels_path} is missing expected column(s) {sorted(missing)} -- "
            f"if only icu_intime is missing, apply label_sepsis3.py's schema addition "
            f"(icu_intime/icu_los_hours/sepsis_onset_time_hours) first."
        )
    eligible = df[df["excluded_reason"].isna()].copy()
    dup = eligible["hadm_id"].duplicated()
    if dup.any():
        print(f"[cxr_linking] WARNING: {int(dup.sum())} duplicate hadm_id(s) in "
              f"the eligible cohort -- keeping first occurrence of each.", file=sys.stderr)
        eligible = eligible[~dup]
    eligible["hadm_id"] = eligible["hadm_id"].astype("int64")
    eligible["subject_id"] = eligible["subject_id"].astype("int64")
    return eligible[["subject_id", "hadm_id", "sepsis_onset_time", "icu_intime"]].reset_index(drop=True)


def load_cohort_admission_windows(con: duckdb.DuckDBPyConnection, cohort_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pulls admittime/dischtime from hosp/admissions.csv, restricted to the
    eligible cohort's hadm_ids (see module docstring for why this file needs
    admissions.csv at all -- sepsis_labels.parquet doesn't carry these
    times). One row per eligible hadm_id; raises if any eligible hadm_id
    isn't found in admissions.csv at all (would silently drop ALL of that
    admission's CXRs otherwise, which is a data problem worth surfacing
    loudly rather than a normal missingness case like ASSUMPTION A2).
    """
    con.register("cohort_hadm_ids", cohort_df[["hadm_id"]].drop_duplicates())
    windows = con.execute("""
        SELECT a.subject_id, a.hadm_id, a.admittime, a.dischtime
        FROM admissions AS a
        INNER JOIN cohort_hadm_ids AS c ON a.hadm_id = c.hadm_id
    """).df()
    con.unregister("cohort_hadm_ids")

    windows["hadm_id"] = windows["hadm_id"].astype("int64")
    windows["subject_id"] = windows["subject_id"].astype("int64")

    missing_hadm = set(cohort_df["hadm_id"]) - set(windows["hadm_id"])
    if missing_hadm:
        raise ValueError(
            f"{len(missing_hadm)} eligible hadm_id(s) from sepsis_labels.parquet "
            f"were not found in hosp/admissions.csv (e.g. {sorted(missing_hadm)[:5]}) -- "
            f"check --mimic_iv_hosp_dir points at the SAME MIMIC-IV version/cohort "
            f"the sepsis_labels.parquet was built from. Every CXR for these admissions "
            f"would otherwise be silently unattributable."
        )
    return windows


def parse_study_datetime(study_date: pd.Series, study_time: pd.Series) -> pd.Series:
    """
    Pure-pandas equivalent of the SQL timestamp expression in
    extract_cxr_with_admission_match() -- used only by the independent
    spot-check re-query (_query_raw_cxr_for_subject), so a bug in the SQL
    version wouldn't also be silently present in the check. See
    ASSUMPTION A3 for the zero-padding rationale.
    """
    date_str = study_date.astype("Int64").astype(str).str.zfill(8)
    whole_seconds = np.floor(study_time.astype(float))
    frac_seconds = study_time.astype(float) - whole_seconds
    time_str = whole_seconds.astype("Int64").astype(str).str.zfill(6)
    base = pd.to_datetime(date_str + time_str, format="%Y%m%d%H%M%S", errors="coerce")
    return base + pd.to_timedelta(frac_seconds, unit="s")


def compute_hours_before_onset(matched_df: pd.DataFrame, cohort_df: pd.DataFrame) -> pd.DataFrame:
    """
    INNER-joins matched CXR rows (already carrying the correctly-resolved
    hadm_id from extract_cxr_with_admission_match()) to
    sepsis_onset_time and computes hours_before_onset. Same sign convention
    as notes_extraction.py's ASSUMPTION A4: positive = recorded before
    onset; negative = at/after onset; NaN = no onset (label=0, included-
    negative admission).
    """
    merged = matched_df.merge(
        cohort_df[["hadm_id", "sepsis_onset_time", "icu_intime"]], on="hadm_id", how="inner",
    )
    merged["hours_before_onset"] = (
        merged["sepsis_onset_time"] - merged["timestamp"]
    ).dt.total_seconds() / 3600.0
    # SCHEMA ADDITION: rolling-task time axis, independent of sepsis_onset_time
    # (null for negatives) -- see experiments/dataset.py's SCHEMA GAP docstring.
    merged["hours_since_admission"] = (
        merged["timestamp"] - merged["icu_intime"]
    ).dt.total_seconds() / 3600.0
    return merged


def build_source_image_path(subject_id: int, study_id, dicom_id: str, files_root: Path) -> Path:
    """MIMIC-CXR-JPG's fixed directory convention:
    files/p<first 2 digits of subject_id>/p<subject_id>/s<study_id>/<dicom_id>.jpg"""
    subject_str = str(int(subject_id))
    study_str = str(study_id).lstrip("s")  # study_id column may or may not include the 's' prefix
    return files_root / f"p{subject_str[:2]}" / f"p{subject_str}" / f"s{study_str}" / f"{dicom_id}.jpg"


def resize_one_image(src_path: Path, dst_path: Path, basewidth: int,
                      resample_filter=None) -> Optional[str]:
    """Resize a single image to a fixed width, preserving aspect ratio --
    exact convention verified against MedPatch's resize.py (ASSUMPTION A4).
    Returns an error string on failure (missing file, corrupt image, etc.)
    or None on success. Skips (returns None immediately) if dst_path already
    exists -- same caching behavior as their script's `paths_done` check."""
    if dst_path.exists():
        return None
    if not src_path.exists():
        return f"source_not_found: {src_path}"
    try:
        img = Image.open(src_path)
        wpercent = basewidth / float(img.size[0])
        hsize = int(float(img.size[1]) * wpercent)
        img = img.resize((basewidth, hsize), resample=resample_filter) if resample_filter else img.resize((basewidth, hsize))
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(dst_path)
        return None
    except Exception as exc:  # corrupt/truncated image, permissions, etc.
        return f"{type(exc).__name__}: {exc}"


def resize_images_for_cohort(rows: pd.DataFrame, files_root: Path, resized_dir: Path,
                              basewidth: int, threads: int,
                              resample_filter=None) -> tuple[pd.Series, dict]:
    """
    Resizes every (subject_id, study_id, dicom_id) in `rows`, flat output
    directory keyed by dicom_id.jpg (MedPatch convention, ASSUMPTION A4).
    Returns (image_path Series aligned to rows.index, error_summary dict).
    """
    resized_dir.mkdir(parents=True, exist_ok=True)
    src_paths = [
        build_source_image_path(r.subject_id, r.study_id, r.dicom_id, files_root)
        for r in rows.itertuples(index=False)
    ]
    dst_paths = [resized_dir / f"{dicom_id}.jpg" for dicom_id in rows["dicom_id"]]

    errors: dict = {}

    def _task(i):
        err = resize_one_image(src_paths[i], dst_paths[i], basewidth, resample_filter)
        if err is not None:
            errors[i] = err

    print(f"[cxr_linking] Resizing {len(rows)} images (basewidth={basewidth}, "
          f"threads={threads}, output={resized_dir})...", file=sys.stderr)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=threads) as pool:
        list(pool.map(_task, range(len(rows))))
    print(f"[cxr_linking]   -> done in {time.time() - t0:.0f}s, "
          f"{len(errors)} error(s)", file=sys.stderr)

    image_path = pd.Series(
        [str(p) if i not in errors else None for i, p in enumerate(dst_paths)],
        index=rows.index,
    )
    error_summary = {
        "n_resize_errors": len(errors),
        "sample_errors": [errors[i] for i in list(errors.keys())[:10]],
    }
    return image_path, error_summary


def build_cxr_output(df: pd.DataFrame) -> pd.DataFrame:
    """Trims to the DoD-locked schema, exact column order + dtypes."""
    out = df[OUTPUT_COLUMNS].copy()
    out["hadm_id"] = out["hadm_id"].astype("int64")
    out["study_id"] = out["study_id"].astype(str)
    out["dicom_id"] = out["dicom_id"].astype(str)
    out["image_path"] = out["image_path"].astype(str)
    out["hours_before_onset"] = out["hours_before_onset"].astype(float)
    out["hours_since_admission"] = out["hours_since_admission"].astype(float)
    return out.reset_index(drop=True)

# ==============================================================================
# STATS
# ==============================================================================

def compute_cxr_stats(matched_df: pd.DataFrame, cxr_final_df: pd.DataFrame,
                       cohort_df: pd.DataFrame, raw_table_counts: dict,
                       resize_error_summary: Optional[dict]) -> dict:
    """Required DoD stats (CXR counts per admission, avg/admission, % zero-
    CXR admissions) plus admission-window matching diagnostics."""
    stats: dict = {}
    stats["raw_table_counts_full_source (not sample-limited)"] = raw_table_counts

    n_cohort_admissions = int(cohort_df["hadm_id"].nunique())
    stats["n_cohort_admissions_this_run"] = n_cohort_admissions
    stats["n_sepsis_positive_admissions_this_run"] = int(cohort_df["sepsis_onset_time"].notna().sum())

    stats["n_cxr_matched_to_some_admission"] = int(len(matched_df))
    stats["n_cxr_matched_multiple_admissions"] = int((matched_df["n_matches"] > 1).sum())

    stats["total_cxr"] = int(len(cxr_final_df))

    per_adm = cxr_final_df.groupby("hadm_id").size()
    all_cohort_hadm = set(cohort_df["hadm_id"])
    n_zero = len(all_cohort_hadm - set(per_adm.index))
    stats["avg_cxr_per_admission"] = (
        round(len(cxr_final_df) / n_cohort_admissions, 3) if n_cohort_admissions else None
    )
    stats["n_admissions_with_zero_cxr"] = int(n_zero)
    stats["pct_admissions_with_zero_cxr"] = (
        round(100.0 * n_zero / n_cohort_admissions, 2) if n_cohort_admissions else None
    )
    stats["cxr_per_admission_distribution"] = {
        "median": float(per_adm.median()) if len(per_adm) else 0.0,
        "p90": float(per_adm.quantile(0.9)) if len(per_adm) else 0.0,
        "max": int(per_adm.max()) if len(per_adm) else 0,
    }

    studies_per_admission = cxr_final_df.groupby("hadm_id")["study_id"].nunique()
    stats["avg_distinct_studies_per_admission"] = (
        round(float(studies_per_admission.mean()), 3) if len(studies_per_admission) else 0.0
    )
    stats["avg_dicoms_per_study"] = (
        round(len(cxr_final_df) / cxr_final_df["study_id"].nunique(), 3)
        if cxr_final_df["study_id"].nunique() else None
    )

    pos = cxr_final_df[cxr_final_df["hours_before_onset"].notna()]
    if len(pos):
        stats["hours_before_onset_on_positive_admissions"] = {
            "n": int(len(pos)),
            "median": round(float(pos["hours_before_onset"].median()), 2),
            "pct_at_or_after_onset": round(100.0 * (pos["hours_before_onset"] <= 0).mean(), 2),
        }

    if resize_error_summary is not None:
        stats["image_resize"] = resize_error_summary

    return stats


# ==============================================================================
# SPOT CHECK (DoD requirement: pick 2-3 admissions, confirm linked studies/
# timestamps match raw MIMIC-CXR metadata directly, confirm no cross-
# admission misattribution for a patient with multiple admissions)
# ==============================================================================

def pick_spot_check_admissions(cxr_final_df: pd.DataFrame, cohort_df: pd.DataFrame,
                                n: int = 3, seed: int = 20240115) -> list:
    """
    Deterministic selection biased to make the DoD's "confirm no cross-
    admission misattribution for a patient with multiple admissions" case
    satisfiable by construction: prefers a subject_id with >=2 admissions
    in the eligible cohort AND CXRs attributed to at least two of them.
    """
    hadm_to_subject = cohort_df.set_index("hadm_id")["subject_id"].to_dict()
    admissions_with_cxr = set(cxr_final_df["hadm_id"].unique())
    if not admissions_with_cxr:
        return []

    subject_to_hadms = {}
    for hadm_id, subj in hadm_to_subject.items():
        subject_to_hadms.setdefault(subj, set()).add(hadm_id)

    multi_admission_subjects = [
        subj for subj, hadms in subject_to_hadms.items()
        if len(hadms & admissions_with_cxr) >= 2
    ]

    rng = np.random.RandomState(seed)
    picked_hadm: list = []
    if multi_admission_subjects:
        chosen_subj = sorted(multi_admission_subjects)[rng.randint(len(multi_admission_subjects))]
        matched_hadms = sorted(subject_to_hadms[chosen_subj] & admissions_with_cxr)
        picked_hadm.extend(matched_hadms[:2])

    remaining_pool = sorted(admissions_with_cxr - set(picked_hadm))
    rng.shuffle(remaining_pool)
    for hadm in remaining_pool:
        if len(picked_hadm) >= n:
            break
        picked_hadm.append(hadm)
    return picked_hadm[:max(n, len(picked_hadm[:2]))]


def spot_check_cxr(con: duckdb.DuckDBPyConnection, cxr_final_df: pd.DataFrame,
                    cohort_df: pd.DataFrame, n: int = 3, seed: int = 20240115) -> list:
    picked = pick_spot_check_admissions(cxr_final_df, cohort_df, n=n, seed=seed)
    hadm_to_subject = cohort_df.set_index("hadm_id")["subject_id"].to_dict()
    onset_by_hadm = cohort_df.set_index("hadm_id")["sepsis_onset_time"].to_dict()

    results = []
    for hadm_id in picked:
        subject_id = hadm_to_subject[hadm_id]
        extracted_subset = cxr_final_df[cxr_final_df["hadm_id"] == hadm_id]

        raw_subject_cxr = _query_raw_cxr_for_subject(con, subject_id)
        raw_windows = _query_admission_windows_for_subject(con, subject_id)
        this_window = raw_windows[raw_windows["hadm_id"] == hadm_id]
        admittime = this_window["admittime"].iloc[0] if len(this_window) else None
        dischtime = this_window["dischtime"].iloc[0] if len(this_window) else None

        # independently recompute which raw CXR rows SHOULD fall in this
        # admission's window, using the pure-pandas parser (not the SQL one)
        if admittime is not None and dischtime is not None:
            expected = raw_subject_cxr[
                (raw_subject_cxr["timestamp"] >= admittime) & (raw_subject_cxr["timestamp"] <= dischtime)
            ]
        else:
            expected = raw_subject_cxr.iloc[0:0]

        expected_dicoms = set(expected["dicom_id"])
        extracted_dicoms = set(extracted_subset["dicom_id"])

        # cross-admission misattribution check: for a multi-admission
        # subject, confirm none of THIS admission's extracted CXRs actually
        # fall inside a DIFFERENT admission's window for the same subject.
        # Deliberately checked against ALL of that subject's real admission
        # windows (raw_windows), not just cohort ones -- a CXR landing inside
        # a non-cohort admission's window is just as real a misattribution
        # bug as landing inside another cohort admission's window would be.
        other_windows = raw_windows[raw_windows["hadm_id"] != hadm_id]
        n_other_cohort_admissions = int(
            ((cohort_df["subject_id"] == subject_id) & (cohort_df["hadm_id"] != hadm_id)).sum()
        )
        misattributed = []
        for _, row in extracted_subset.iterrows():
            for _, ow in other_windows.iterrows():
                if ow["admittime"] <= row["timestamp"] <= ow["dischtime"]:
                    misattributed.append({
                        "dicom_id": row["dicom_id"], "timestamp": str(row["timestamp"]),
                        "also_falls_in_hadm_id": int(ow["hadm_id"]),
                    })

        onset_time = onset_by_hadm.get(hadm_id)
        results.append({
            "hadm_id": int(hadm_id),
            "subject_id": int(subject_id),
            "n_other_cohort_admissions_for_this_subject": n_other_cohort_admissions,
            "n_other_raw_admissions_checked_for_misattribution": int(len(other_windows)),
            "admittime": str(admittime), "dischtime": str(dischtime),
            "sepsis_onset_time": None if onset_time is None or pd.isna(onset_time) else str(onset_time),
            "n_extracted": int(len(extracted_subset)),
            "n_expected_from_independent_reparse": int(len(expected)),
            "dicom_id_sets_match": expected_dicoms == extracted_dicoms,
            "missing_from_extracted": sorted(expected_dicoms - extracted_dicoms),
            "extra_in_extracted": sorted(extracted_dicoms - expected_dicoms),
            "cross_admission_misattributions_found": misattributed,
            "pass": bool(expected_dicoms == extracted_dicoms and not misattributed),
        })
    return results


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Link MIMIC-CXR studies to admissions for the locked sepsis "
                    "cohort (Milestone 1)."
    )
    parser.add_argument("--mimic_cxr_dir", type=str, required=True,
                         help="Path to the MIMIC-CXR-JPG version folder (containing "
                              "files/ and mimic-cxr-2.0.0-metadata.csv[.gz]).")
    parser.add_argument("--mimic_cxr_metadata_filename", type=str,
                         default="mimic-cxr-2.0.0-metadata.csv",
                         help="Filename (under --mimic_cxr_dir) of the metadata CSV.")
    parser.add_argument("--mimic_iv_hosp_dir", type=str, required=True,
                         help="Path to the MIMIC-IV core hosp/ folder containing "
                              "admissions.csv[.gz]. See module docstring for why this "
                              "file needs it (sepsis_labels.parquet has no admittime/"
                              "dischtime).")
    parser.add_argument("--sepsis_labels_path", type=str, default="data/cohort/sepsis_labels.parquet",
                         help="Milestone 0 output (label_sepsis3.py).")
    parser.add_argument("--out_dir", type=str, default="data/cohort",
                         help="Output directory for cxr_metadata.parquet, cxr_stats.json, "
                              "cxr_spot_check.json.")
    parser.add_argument("--resized_image_dir", type=str, default="data/cohort/cxr/resized",
                         help="Flat output directory for resized JPGs, keyed by "
                              "dicom_id.jpg -- matches MedPatch's resize.py convention "
                              "(ASSUMPTION A4).")
    parser.add_argument("--cache_dir", type=str, default="data/cache",
                         help="Parquet cache dir for the raw metadata/admissions CSVs -- "
                              "same convention as label_sepsis3.py / notes_extraction.py.")
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--sample_size", type=int, default=None,
                         help="If set, run on a random sample of this many eligible hadm_ids "
                              "instead of the full cohort. See ASSUMPTION A6.")
    parser.add_argument("--resize_basewidth", type=int, default=IMAGE_RESIZE_BASEWIDTH,
                         help="See ASSUMPTION A4.")
    parser.add_argument("--resize_threads", type=int, default=DEFAULT_RESIZE_THREADS)
    parser.add_argument("--skip_image_resize", action="store_true",
                         help="Dry-run mode: link metadata + compute hours_before_onset "
                              "WITHOUT resizing images (image_path will point at the "
                              "would-be resized path, which will NOT exist on disk). "
                              "Useful for quickly checking the admission-join logic on a "
                              "sample before paying the full resize cost. NOT valid for "
                              "the final full-cohort run -- the DoD requires an actual "
                              "resized image on disk at image_path.")
    parser.add_argument("--n_spot_check", type=int, default=3)
    parser.add_argument("--spot_check_seed", type=int, default=20240115)
    args = parser.parse_args()

    mimic_cxr_dir = Path(args.mimic_cxr_dir)
    cxr_metadata_path = mimic_cxr_dir / args.mimic_cxr_metadata_filename
    files_root = mimic_cxr_dir / "files"
    hosp_dir = Path(args.mimic_iv_hosp_dir)
    admissions_path = hosp_dir / "admissions.csv"
    sepsis_labels_path = Path(args.sepsis_labels_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    resized_dir = Path(args.resized_image_dir)
    cache_dir = None if args.no_cache else Path(args.cache_dir)

    print("[cxr_linking] Loading eligible cohort from sepsis_labels.parquet...", file=sys.stderr)
    cohort_df = load_sepsis_cohort(sepsis_labels_path)
    if args.sample_size is not None:
        cohort_df = cohort_df.sort_values("hadm_id").reset_index(drop=True)  # see ASSUMPTION A6
        cohort_df = cohort_df.sample(
            n=min(args.sample_size, len(cohort_df)), random_state=0
        ).reset_index(drop=True)
    print(f"[cxr_linking]   -> {len(cohort_df)} eligible admissions "
          f"({int(cohort_df['sepsis_onset_time'].notna().sum())} sepsis-positive)", file=sys.stderr)

    print(f"[cxr_linking] Connecting to MIMIC-CXR metadata + hosp/admissions via DuckDB...",
          file=sys.stderr)
    con = connect_duckdb(cxr_metadata_path, admissions_path, cache_dir=cache_dir)
    raw_table_counts = compute_raw_table_counts(con)

    print("[cxr_linking] Loading admission windows (admittime/dischtime) for the "
          "eligible cohort...", file=sys.stderr)
    admission_windows = load_cohort_admission_windows(con, cohort_df)

    print("[cxr_linking] [1/3] Joining CXR studies to admissions via timestamp-in-"
          "window match...", file=sys.stderr)
    t0 = time.time()
    matched_df = extract_cxr_with_admission_match(con, admission_windows)
    print(f"[cxr_linking] [1/3] done in {time.time() - t0:.0f}s "
          f"({len(matched_df)} CXR rows matched to a cohort admission)", file=sys.stderr)

    print("[cxr_linking] [2/3] Computing hours_before_onset...", file=sys.stderr)
    cxr_df = compute_hours_before_onset(matched_df, cohort_df)
    # NOTE: subject_id already flows through from extract_cxr_with_admission_match()
    # (it's part of the matched-row payload) -- do NOT re-merge it from cohort_df
    # here, that creates a duplicate subject_id_x/subject_id_y column collision.

    resize_error_summary = None
    if args.skip_image_resize:
        print("[cxr_linking] --skip_image_resize set: writing placeholder image_path "
              "values, NOT resizing. Do not use this output as a final deliverable.",
              file=sys.stderr)
        cxr_df["image_path"] = cxr_df["dicom_id"].apply(lambda d: str(resized_dir / f"{d}.jpg"))
    else:
        image_path, resize_error_summary = resize_images_for_cohort(
            cxr_df, files_root, resized_dir, args.resize_basewidth, args.resize_threads,
        )
        cxr_df["image_path"] = image_path
        n_failed = cxr_df["image_path"].isna().sum()
        if n_failed:
            print(f"[cxr_linking] WARNING: {n_failed} row(s) failed to resize -- "
                  f"dropping from output (see cxr_stats.json['image_resize']).",
                  file=sys.stderr)
            cxr_df = cxr_df[cxr_df["image_path"].notna()].copy()

    print("[cxr_linking] [3/3] Computing stats...", file=sys.stderr)
    output_df = build_cxr_output(cxr_df)
    stats = compute_cxr_stats(matched_df, output_df, cohort_df, raw_table_counts, resize_error_summary)

    out_path = out_dir / "cxr_metadata.parquet"
    output_df.to_parquet(out_path, index=False)
    print(f"[cxr_linking] Wrote {out_path} ({len(output_df)} rows)", file=sys.stderr)

    stats_path = out_dir / "cxr_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, default=str)
    print(json.dumps(stats, indent=2, default=str))
    print(f"[cxr_linking] Wrote {stats_path}", file=sys.stderr)

    print(f"[cxr_linking] Running spot-check on up to {args.n_spot_check} admission(s) "
          f"(biased toward a multi-admission subject if one exists)...", file=sys.stderr)
    spot_check_results = spot_check_cxr(
        con, output_df, cohort_df, n=args.n_spot_check, seed=args.spot_check_seed
    )
    spot_check_path = out_dir / "cxr_spot_check.json"
    with open(spot_check_path, "w") as f:
        json.dump(spot_check_results, f, indent=2, default=str)
    for r in spot_check_results:
        status = "PASS" if r["pass"] else "FAIL"
        print(f"[cxr_linking]   hadm_id={r['hadm_id']} (subject_id={r['subject_id']}, "
              f"{r['n_other_cohort_admissions_for_this_subject']} other cohort "
              f"admission(s) for this subject): {status} ({r['n_extracted']} CXRs)",
              file=sys.stderr)
    print(f"[cxr_linking] Wrote {spot_check_path}", file=sys.stderr)

    if any(not r["pass"] for r in spot_check_results):
        print("[cxr_linking] WARNING: at least one spot-check admission did NOT "
              "pass (either a mismatch vs. independent re-parse, or a cross-"
              "admission misattribution was found) -- see cxr_spot_check.json "
              "before trusting cxr_metadata.parquet.", file=sys.stderr)


if __name__ == "__main__":
    main()
    
    
    
# python preprocessing/cxr_linking.py \
#   --mimic_cxr_dir "/home/fluuvys-main/Research/Multi modal sepsis prediction/Data/cxr/files/mimic-cxr-jpg/2.0.0"\
#   --mimic_iv_hosp_dir "/home/fluuvys-main/Research/Multi modal sepsis prediction/Data/mimic-iv-3.1/hosp" \
#   --sepsis_labels_path "/home/fluuvys-main/Research/Multi modal sepsis prediction/Multi-Modal-Sepsis-Prediction/data/cohort/sepsis_labels.parquet" \
#   --out_dir data/cohort \
#   --sample_size 100 \
#   --skip_image_resize