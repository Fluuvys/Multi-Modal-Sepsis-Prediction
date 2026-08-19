from __future__ import annotations

"""
preprocessing/notes_extraction.py
================================================================================
PURPOSE:
    Extract radiology reports (RR) and discharge notes (DN) from raw
    MIMIC-IV-Note tables, restricted to the admissions in our locked cohort
    (data/cohort/sepsis_labels.parquet, Milestone 0 -- see
    preprocessing/label_sepsis3.py, NOT reimplemented here). Produces the
    single shared data/cohort/notes.parquet that every baseline reproduction
    AND our own model consume -- see PROJECT_CONTEXT.md rule #5: no
    model-specific reduction (most-recent-only, note-count caps, chunking
    actually applied, embedding, etc.) happens in this file. That logic goes
    in each model's own adapter under models/baselines/ or models/ours/.
    This file's job stops at: real timestamps, full text, and a precomputed
    hours_before_onset per note -- the minimum shared fact every downstream
    consumer needs and none of them should recompute differently.

RELATIONSHIP TO MILESTONE 0:
    Depends on data/cohort/sepsis_labels.parquet already existing (see
    label_sepsis3.py). We join against it ONLY to (a) restrict to admissions
    in the final cohort (excluded_reason IS NULL -- both sepsis-positive and
    included-negative admissions; excluded admissions get zero notes, by
    design) and (b) pull sepsis_onset_time to compute hours_before_onset.
    We do NOT reimplement or second-guess any cohort/labeling logic here.

INPUT (raw MIMIC-IV-Note, see docs/data_schema.md for the locked OUTPUT
schema; this section documents what we READ, per the task's requirement #4):
    discharge.csv  (table: discharge)
        columns used: note_id (str, PK), subject_id (int), hadm_id (int,
        always non-null in this table), note_type (str, expected 'DS'),
        note_seq (int), charttime (timestamp), storetime (timestamp),
        text (str, the raw discharge summary)
    radiology.csv  (table: radiology)
        same column set; note_type expected 'RR'; hadm_id CAN be null here
        (imaging not tied to an inpatient admission, e.g. ED-only or
        outpatient studies) -- those rows are unavoidably dropped, since
        hours_before_onset needs an hadm_id to join against. See
        ASSUMPTION A7.
    We do NOT read discharge_detail.csv / radiology_detail.csv (addendum
    authorship placeholders, CPT codes) -- out of scope for the locked
    output schema. See ASSUMPTION A8.
    We do NOT read anything from hosp/ or icu/ directly -- all cohort
    membership and onset timing comes through sepsis_labels.parquet only,
    per PROJECT_CONTEXT.md rule #2 (label_sepsis3.py is the single source
    of truth; don't re-derive any of its logic here).

OUTPUT: data/cohort/notes.parquet, exactly these 6 columns (locked, per the
    task's Definition of Done -- do not add columns here even for
    diagnostics; diagnostics go in notes_stats.json / notes_spot_check.json
    instead):
        hadm_id             int       join key
        note_id             string    MIMIC-IV-Note's own note_id, unique
                                       across both source tables already
                                       (format "{subject_id}-{DS|RR}-{seq}")
        note_type           string    "RR" or "DN" -- see ASSUMPTION A1
        timestamp           datetime  raw, real note timestamp
        raw_text            string    unchunked, unembedded raw text
        hours_before_onset  float     precomputed; NaN for admissions with
                                       no onset (label=0, included-negative)
    Also writes data/cohort/notes_stats.json (counts, leakage diagnostics)
    and data/cohort/notes_spot_check.json (manual-verification aid, DoD
    requirement).

TODO checklist (Milestone 1 -- EHR/notes/CXR preprocessing pipelines):
    [x] extract_raw_notes() / assign_note_type() / resolve_timestamp()
    [x] compute_hours_before_onset() joined against Milestone 0's cohort
    [x] chunk_note_text() utility (MedPatch 512-token convention, adapted --
        see ASSUMPTION A3/A5) -- NOT applied to the output; raw_text stays
        whole
    [x] notes_stats.json (required counts + leakage diagnostics)
    [x] spot_check_notes() (re-queries raw tables directly, per DoD)
    [ ] Team review of ASSUMPTIONS A1-A9 below, especially A3 (discharge-note
        leakage -- read this one first) and A2 (timestamp field choice)
    [ ] PROJECT_CONTEXT.md sec 7 checklist item, same PR or same-day follow-up

================================================================================
ASSUMPTIONS / DEVIATIONS FLAGGED FOR TEAM REVIEW -- read before trusting output
================================================================================
Search "ASSUMPTION" in this file for the code sites. Summary:

A1. Our OUTPUT note_type ('RR'/'DN') is derived from WHICH SOURCE TABLE a row
    came from (discharge.csv -> 'DN', radiology.csv -> 'RR'), not from that
    table's own note_type field. This is more robust than trusting the raw
    field verbatim: even if discharge.csv's note_type ever contains
    something other than the expected 'DS' (e.g. an addendum-specific
    code -- see A8), it's still a discharge note for our purposes. The raw
    note_type values actually observed per source table are logged to
    notes_stats.json ("raw_note_type_observed_by_source_table") so the team
    can confirm nothing unexpected is hiding in there. CONFIRMED USEFUL ON
    REAL DATA: radiology.csv contains a small number of raw note_type='AR'
    rows (addendum reports) alongside the expected 'RR' -- both correctly
    flow through to output note_type='RR' via this source-table mapping,
    exactly the scenario this design was meant to handle safely.

A2. [UPDATED -- see NOTE_TYPE_TIMESTAMP_FIELD below] Timestamp field choice
    is now PER-NOTE-TYPE, not a single global field, based on direct
    verification against the real data:
      - DN (discharge notes): charttime is EXACTLY midnight for 100% of
        rows in the full 331,793-row discharge table (confirmed via direct
        query) -- this is a fabricated date-only timestamp, not real
        time-of-day precision. storetime's midnight rate on the same table
        is 0.02% (68/331,793) -- genuinely precise. DN uses storetime.
      - RR (radiology reports): charttime's midnight rate is 0% and a
        direct --timestamp_field storetime comparison run showed no
        meaningful improvement (hours_before_onset distribution barely
        moved). RR stays on charttime.
    This matters because it changes what hours_before_onset actually means
    for DN notes -- previously (charttime-based) it was accurate only to
    within +/-24h given the fabricated midnight time; now (storetime-based)
    it reflects real time-of-day precision. Re-run and compare against any
    notes.parquet produced before this fix before trusting DN-based
    lead-time filtering downstream.
    compute_leakage_diagnostics() still checks charttime's midnight rate
    specifically (not whichever field is actually canonical) as an early-
    warning signal for this exact failure mode recurring on some future
    MIMIC-IV-Note release -- see the flag_DN_timestamp_precision note in
    that function for the current limitation of that check.

A3. DISCHARGE-NOTE LEAKAGE HANDLING -- read this one first.
    MedPatch's own paper (Al Jorf & Shamout, MLHC 2025 -- 
    github.com/nyuad-cai/MedPatch) does NOT do timestamp-based filtering of
    discharge notes. Their actual mechanism, confirmed by reading the paper
    directly, is coarser: they exclude discharge notes (DN) ENTIRELY from
    their in-hospital-mortality task (an early/during-stay outcome,
    evaluated on the first 48h of admission), explicitly because DN
    "contains information related to patient discharge, which would
    introduce information leakage" (their stated reasoning, paraphrased
    here). They only use DN for their OTHER task -- end-of-stay clinical-
    condition classification -- where the label itself describes the whole
    stay, so a note summarizing the whole stay isn't leakage for that task.
    Our sepsis-onset task is an early-prediction task in the same family as
    MedPatch's mortality task (a specific lead time before a specific
    event), not their end-of-stay classification task -- so by MedPatch's
    own precedent, DN is exactly the kind of input that needs leakage
    handling for OUR task too.
    But: PROJECT_CONTEXT.md rule #5 says model-specific reduction doesn't
    belong in this shared file, and the task spec explicitly requires
    keeping ALL notes (not dropping DN here). So this file ADAPTS, rather
    than forks, MedPatch's logic: instead of a one-time binary
    include/exclude decision made in this file, we (a) keep every DN with
    its real timestamp, (b) compute hours_before_onset so each model's
    adapter can reproduce a decision at the per-lead-time granularity our
    task actually needs (e.g. filter to hours_before_onset >= lead_time_h
    before an adapter feeds notes to its encoder), and (c) empirically
    validate, in compute_leakage_diagnostics() below, that DN timestamps
    actually behave the way that downstream filtering strategy assumes
    (i.e. cluster late / near end-of-stay), flagging anything that doesn't.
    TEAM DECISION NEEDED (per PROJECT_CONTEXT.md rule #2): whichever
    adapter reproduces MedPatch's own baseline should almost certainly
    exclude DN entirely for the shortest lead times (2h/4h/6h), mirroring
    their mortality-task choice, rather than relying solely on
    hours_before_onset filtering -- confirm this with the team before
    Milestone 2 (baseline reproductions).

A4. hours_before_onset sign convention: positive = note recorded before
    onset (the common case a lead-time filter cares about); negative =
    recorded at/after onset; NaN = admission has no onset at all
    (label=0, included-negative -- there's nothing to be "before", this is
    expected and correct, not a bug). The ">24h before onset is suspicious
    for a DN" threshold used in the leakage flag (DN_SUSPICIOUS_EARLY_HOURS)
    is a heuristic starting point, not a validated cutoff -- tune it once
    the team has looked at a real distribution.

A5. Chunking tokenizer default: DEFAULT_TOKENIZER_NAME points at a public
    BioBERT checkpoint on the HuggingFace Hub, since MedPatch's paper states
    they used BioBERT (Lee et al. 2020) for both RR and DN encoding, and
    your environment.yml earmarks `transformers` for "BioBERT / Clinical-
    Longformer". I do not have access to MedPatch's actual code (only their
    paper) and could not confirm the exact HF Hub checkpoint identifier
    they loaded -- verify/override via --tokenizer_name before relying on
    this for a MedPatch reproduction specifically. chunk_note_text() takes
    any HF-compatible tokenizer as a parameter (not hardcoded), so other
    baselines / our own model can pass in whatever tokenizer they actually
    use (e.g. Clinical-Longformer) -- hardcoding BioBERT's subword
    boundaries into the *shared* notes.parquet would violate
    PROJECT_CONTEXT.md rule #5 just as much as pre-embedding would, which
    is why raw_text is stored unchunked and this is offered only as an
    importable utility, not applied in this file's own main().

A6. Your environment lists MIMIC-IV core at v3.1, but a web search from
    where this was written found MIMIC-IV-Note still at v2.2 (tied to core
    MIMIC-IV v2.2's admission set) -- no newer note-module release was
    found. If that's still true when you run this, a small number of the
    newest hadm_ids in your v3.1-derived cohort may have genuinely zero
    notes for this reason alone, not a pipeline bug. Check
    physionet.org/content/mimic-iv-note/ for a newer release before
    treating "% admissions with zero notes" as purely a data-quality signal.

A7. Radiology rows with NULL hadm_id are unavoidably dropped (can't compute
    hours_before_onset or restrict-to-cohort without an hadm_id). Counted
    via compute_raw_table_counts() and reported in notes_stats.json
    ("raw_table_counts"), not silently discarded.

A8. discharge_detail.csv / radiology_detail.csv (addendum linkage, CPT
    codes) are intentionally NOT read -- out of scope for the locked 6-
    column output schema. If an admission has more than one discharge.csv
    row (e.g. an addendum, distinguished by note_seq), EACH is kept as its
    own note per the "keep ALL notes, not just the most recent" requirement
    -- they are not deduplicated or collapsed here. avg_DN_per_admission in
    notes_stats.json will reveal how often this happens on your real data.

A9. --sample_size samples eligible hadm_ids (not raw note rows),
    deterministically, ORDERED before sampling. This mirrors, on purpose,
    the exact reproducibility bug-fix documented in label_sepsis3.py's
    build_cohort() (unordered DuckDB/pandas results + .sample(random_state=0)
    silently sampling different rows across runs) -- see that file's
    comment for the full explanation. Don't remove the .sort_values() below
    without re-reading that.

A10. [CSV parsing -- fixed after direct verification, mirrors a real bug
    found in this exact way on label_sepsis3.py earlier in this project]
    Both discharge.csv and radiology.csv are read with `parallel=false,
    ignore_errors=false`, NOT DuckDB's default read_csv_auto(...,
    IGNORE_ERRORS=TRUE). DuckDB's parallel CSV reader can misjudge row
    boundaries on files with very large embedded-newline quoted fields
    (discharge summaries and radiology reports routinely have these), and
    IGNORE_ERRORS=TRUE then silently discards the vast majority of
    resulting "rows" rather than erroring. Confirmed directly: the default
    read_csv_auto parsed only 739/331,794 real discharge rows (0.22%) and
    16,608/2,321,355 real radiology rows (~0.7%) on this exact data;
    parallel=false recovered all 331,793 and 2,321,355 respectively (exact
    match to PhysioNet's published counts). ignore_errors=false is
    deliberate too -- if a genuinely malformed row exists, a loud failure
    is far better than another silent mass-drop like this one.

A11. [Fixed -- was a real, would-crash-on-first-run bug] resolve_timestamp()
    takes a per-note-type dict (NOTE_TYPE_TIMESTAMP_FIELD) now, not a
    single field string, per A2 above. main() previously still called it
    with args.timestamp_field (a plain string from argparse), which throws
    AttributeError('str' object has no attribute 'items') the moment Stage
    2 runs -- confirmed via direct reproduction before this fix, not just
    inferred from reading the diff. --timestamp_field is now a full
    OVERRIDE (forces BOTH note types onto the given field, ignoring the
    per-type default) rather than the sole field selector; leave it unset
    to use the verified per-type default from A2.
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

# Which MIMIC-IV-Note table maps to which OUTPUT note_type -- see ASSUMPTION A1.
SOURCE_TABLE_TO_NOTE_TYPE = {"discharge": "DN", "radiology": "RR"}

# datetime-like columns shared by both note tables (see connect_duckdb())
NOTE_DATETIME_COLS = ["charttime", "storetime"]

# Per-note-type canonical timestamp field -- see ASSUMPTION A2. Confirmed
# directly against the full discharge table (100% vs 0.02% midnight rate)
# and a real --timestamp_field storetime comparison run for RR.
NOTE_TYPE_TIMESTAMP_FIELD = {"DN": "storetime", "RR": "charttime"}

CHUNK_MAX_TOKENS = 512  # MedPatch convention, see chunk_note_text() / ASSUMPTION A3
DEFAULT_TOKENIZER_NAME = "dmis-lab/biobert-base-cased-v1.1"  # ASSUMPTION A5

DN_SUSPICIOUS_EARLY_HOURS = 24.0  # ASSUMPTION A4 -- heuristic, tune after real data review

OUTPUT_COLUMNS = ["hadm_id", "note_id", "note_type", "timestamp", "raw_text",
                  "hours_before_onset", "hours_since_admission"]


# ==============================================================================
# DATA LOADING (duckdb) -- everything below this point that touches `con` is
# intentionally kept simple (SELECT / JOIN / UNION ALL / CAST only, no window
# functions or LATERAL joins) precisely because it's the one part of this
# file that can't be unit-tested with plain pandas fixtures. Business logic
# (joins against cohort, hours_before_onset, stats, leakage checks, spot-
# check comparison) lives in the pure-pandas section further down instead,
# where it CAN be exercised directly against hand-built DataFrames.
# ==============================================================================

def _ensure_parquet_cache(view_name: str, csv_path: Path, cache_dir: Path) -> Path:
    """
    One-time conversion of a raw note CSV to Parquet, cached under cache_dir.
    Identical in spirit to label_sepsis3.py's helper of the same name (kept
    as a separate copy since this file must run standalone -- if the team
    later adds a shared preprocessing/_io_utils.py, both copies should move
    there instead of a third one being written). radiology.csv in particular
    is large (millions of free-text rows); repeated CSV re-parses across
    dev/test runs are exactly what this avoids.

    See ASSUMPTION A10 for why this uses parallel=false, ignore_errors=false
    instead of DuckDB's default read_csv_auto(..., IGNORE_ERRORS=TRUE) --
    that default silently parsed only ~0.2-0.7% of these specific files.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = cache_dir / f"{view_name}.parquet"
    if parquet_path.exists() and parquet_path.stat().st_mtime >= csv_path.stat().st_mtime:
        return parquet_path

    print(f"[notes_extraction]   Building Parquet cache for {view_name} "
          f"(one-time cost, reused on every future run)...", file=sys.stderr)
    t0 = time.time()
    tmp_con = duckdb.connect(database=":memory:")
    tmp_con.execute("PRAGMA threads=4;")
    tmp_con.execute(f"""
        COPY (
            SELECT * FROM read_csv('{csv_path.as_posix()}', parallel=false, ignore_errors=false)
        ) TO '{parquet_path.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD);
    """)
    tmp_con.close()
    print(f"[notes_extraction]   -> {view_name}.parquet written in "
          f"{time.time() - t0:.0f}s", file=sys.stderr)
    return parquet_path


def _register_note_view(con: duckdb.DuckDBPyConnection, view_name: str,
                         csv_path: Path, cache_dir: Optional[Path]) -> None:
    """Registers `view_name` (discharge.csv or radiology.csv) with charttime
    and storetime explicitly cast to TIMESTAMP -- avoids DuckDB's type
    sniffer silently falling back to VARCHAR on a sparse/all-null column.
    See ASSUMPTION A10 for the parallel=false, ignore_errors=false choice
    (applies to both the cached and --no_cache code paths below)."""
    if not csv_path.exists():
        gz_path = csv_path.with_suffix(csv_path.suffix + ".gz")
        if gz_path.exists():
            csv_path = gz_path
        else:
            raise FileNotFoundError(
                f"Expected MIMIC-IV-Note file not found (checked both "
                f"{csv_path} and {gz_path}). Pass the folder containing "
                f"discharge.csv and radiology.csv via --mimic_note_dir "
                f"(not shown in your directory listing, so there's no "
                f"default -- see the module docstring)."
            )

    casts = ", ".join(f"CAST({c} AS TIMESTAMP) AS {c}" for c in NOTE_DATETIME_COLS)

    if cache_dir is not None:
        parquet_path = _ensure_parquet_cache(view_name, csv_path, cache_dir)
        con.execute(
            f"CREATE OR REPLACE VIEW {view_name} AS "
            f"SELECT * REPLACE ({casts}) FROM read_parquet('{parquet_path.as_posix()}');"
        )
    else:
        con.execute(
            f"CREATE OR REPLACE VIEW {view_name} AS "
            f"SELECT * REPLACE ({casts}) FROM read_csv('{csv_path.as_posix()}', "
            f"parallel=false, ignore_errors=false);"
        )


def connect_duckdb(note_dir: Path, cache_dir: Optional[Path] = None) -> duckdb.DuckDBPyConnection:
    """Registers `discharge` and `radiology` views over the raw MIMIC-IV-Note
    CSVs. See module docstring for the exact columns each table provides."""
    con = duckdb.connect(database=":memory:")
    con.execute("PRAGMA threads=4;")
    _register_note_view(con, "discharge", note_dir / "discharge.csv", cache_dir)
    _register_note_view(con, "radiology", note_dir / "radiology.csv", cache_dir)
    return con


def compute_raw_table_counts(con: duckdb.DuckDBPyConnection) -> dict:
    """Row counts for the FULL raw tables (independent of --sample_size and
    of cohort membership) -- the denominators used to report how many rows
    were structurally undroppable-vs-dropped. See ASSUMPTION A7."""
    counts = {}
    for view in ("discharge", "radiology"):
        total = con.execute(f"SELECT COUNT(*) FROM {view}").fetchone()[0]
        null_hadm = con.execute(f"SELECT COUNT(*) FROM {view} WHERE hadm_id IS NULL").fetchone()[0]
        counts[view] = {"total_rows": int(total), "null_hadm_id_rows": int(null_hadm)}
    return counts


def extract_raw_notes(con: duckdb.DuckDBPyConnection, cohort_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pulls every discharge + radiology row whose hadm_id is in the eligible
    cohort. This is the ONLY place hadm_id-based filtering happens; a NULL
    hadm_id (radiology only, see ASSUMPTION A7) never matches the join and
    is dropped automatically by ordinary SQL NULL semantics -- no separate
    WHERE hadm_id IS NOT NULL needed. Everything downstream of this
    function's return value is a pure pandas transform (see next section).
    """
    con.register("cohort_hadm_ids", cohort_df[["hadm_id"]].drop_duplicates())
    query = """
        SELECT d.note_id, d.subject_id, d.hadm_id, d.note_type AS raw_note_type,
               d.note_seq, d.charttime, d.storetime, d.text AS raw_text,
               'discharge' AS source_table
        FROM discharge AS d
        INNER JOIN cohort_hadm_ids AS c ON d.hadm_id = c.hadm_id
        UNION ALL
        SELECT r.note_id, r.subject_id, r.hadm_id, r.note_type AS raw_note_type,
               r.note_seq, r.charttime, r.storetime, r.text AS raw_text,
               'radiology' AS source_table
        FROM radiology AS r
        INNER JOIN cohort_hadm_ids AS c ON r.hadm_id = c.hadm_id
    """
    df = con.execute(query).df()
    con.unregister("cohort_hadm_ids")
    return df


def _query_raw_notes_for_hadm(con: duckdb.DuckDBPyConnection, hadm_id: int) -> pd.DataFrame:
    """Small, independent re-query of the raw tables for ONE admission --
    used only by spot_check_notes(), deliberately NOT reusing any of
    extract_raw_notes()'s machinery, so the comparison is meaningful."""
    query = f"""
        SELECT note_id, hadm_id, note_type AS raw_note_type, charttime, storetime,
               text AS raw_text, 'discharge' AS source_table
        FROM discharge WHERE hadm_id = {int(hadm_id)}
        UNION ALL
        SELECT note_id, hadm_id, note_type AS raw_note_type, charttime, storetime,
               text AS raw_text, 'radiology' AS source_table
        FROM radiology WHERE hadm_id = {int(hadm_id)}
    """
    return con.execute(query).df()


# ==============================================================================
# PURE TRANSFORMATION LOGIC -- pandas only, no duckdb. Every function below
# takes and returns plain DataFrames/Series and is exercised directly by the
# test suite against hand-built fixtures.
# ==============================================================================

def load_sepsis_cohort(sepsis_labels_path: Path) -> pd.DataFrame:
    """
    Loads Milestone 0's output and restricts to the FINAL cohort
    (excluded_reason IS NULL) -- both sepsis-positive (label=1) and
    included-negative (label=0) admissions. Excluded admissions never reach
    the note-extraction join, so their notes are correctly absent from
    notes.parquet -- "restrict to included admissions only" per the task.
    """
    if not sepsis_labels_path.exists():
        raise FileNotFoundError(
            f"{sepsis_labels_path} not found. This script depends on "
            f"Milestone 0 (preprocessing/label_sepsis3.py) -- run that "
            f"first; see PROJECT_CONTEXT.md sec 6 repo map."
        )
    df = pd.read_parquet(sepsis_labels_path)
    required_cols = {"hadm_id", "sepsis_onset_time", "excluded_reason", "icu_intime"}
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
        print(f"[notes_extraction] WARNING: {int(dup.sum())} duplicate hadm_id(s) in "
              f"the eligible cohort -- sepsis_labels.parquet should have one row per "
              f"admission (subject-level first-stay-only, per label_sepsis3.py). "
              f"Keeping first occurrence of each.", file=sys.stderr)
        eligible = eligible[~dup]
    eligible["hadm_id"] = eligible["hadm_id"].astype("int64")
    return eligible[["hadm_id", "sepsis_onset_time", "icu_intime"]].reset_index(drop=True)


def assign_note_type(raw_df: pd.DataFrame) -> pd.Series:
    """See ASSUMPTION A1: output note_type comes from source_table, not the
    raw note_type field."""
    mapped = raw_df["source_table"].map(SOURCE_TABLE_TO_NOTE_TYPE)
    if mapped.isna().any():
        unknown = sorted(raw_df.loc[mapped.isna(), "source_table"].unique().tolist())
        raise ValueError(f"Unrecognized source_table value(s): {unknown} -- "
                          f"expected only {sorted(SOURCE_TABLE_TO_NOTE_TYPE)}.")
    return mapped


def resolve_timestamp(raw_df: pd.DataFrame, note_type_field: dict = NOTE_TYPE_TIMESTAMP_FIELD):
    """
    Returns (canonical_timestamp_series, fallback_stats_dict). Per-note-type
    field choice, not a single global field -- see ASSUMPTION A2. Confirmed
    directly against the full discharge table: charttime is exactly midnight
    for 100% of DN rows (a fabricated date-only timestamp) vs 0.02% for
    storetime, so DN uses storetime; RR's charttime looks fine as-is (0%
    midnight) and switching it showed no meaningful improvement, so RR stays
    on charttime.

    `raw_df` must already have a `source_table` column (assign_note_type()
    is NOT called here -- note type is derived from source_table directly,
    same mapping, so this can run either before or after assign_note_type()
    without needing raw_df["note_type"] to exist yet).

    `note_type_field` lets a caller override the per-type default (e.g.
    main()'s --timestamp_field forces BOTH types onto one field for
    debugging -- see ASSUMPTION A11).
    """
    resolved = pd.Series(pd.NaT, index=raw_df.index, dtype="datetime64[ns]")
    fallback_stats = {}
    for note_type, primary_field in note_type_field.items():
        mask = raw_df["source_table"].map(SOURCE_TABLE_TO_NOTE_TYPE) == note_type
        sub = raw_df.loc[mask]
        other_field = "storetime" if primary_field == "charttime" else "charttime"
        primary = sub[primary_field]
        fallback = sub[other_field]
        used_fallback = primary.isna() & fallback.notna()
        row_resolved = primary.where(~used_fallback, fallback)
        resolved.loc[mask] = row_resolved
        fallback_stats[note_type] = {
            "primary_field": primary_field,
            "n_used_fallback": int(used_fallback.sum()),
            "n_still_null_after_fallback": int(row_resolved.isna().sum()),
        }
    return resolved, fallback_stats


def compute_hours_before_onset(notes_df: pd.DataFrame, cohort_df: pd.DataFrame) -> pd.DataFrame:
    """
    INNER-joins notes to the eligible cohort's sepsis_onset_time (NaT for
    included-negative admissions -- see label_sepsis3.py ASSUMPTION A4/A10
    for how sepsis_onset_time itself is defined; not re-derived here) and
    computes hours_before_onset. See ASSUMPTION A4 for the sign convention.
    This is also, functionally, the "restrict to included admissions only"
    step for any note whose hadm_id wasn't already excluded upstream by
    extract_raw_notes()'s join -- with an inner join here too as a second,
    cheap safety net.
    """
    merged = notes_df.merge(
        cohort_df[["hadm_id", "sepsis_onset_time", "icu_intime"]], on="hadm_id", how="inner",
    )
    merged["hours_before_onset"] = (
        merged["sepsis_onset_time"] - merged["timestamp"]
    ).dt.total_seconds() / 3600.0
    merged["hours_since_admission"] = (
        merged["timestamp"] - merged["icu_intime"]
    ).dt.total_seconds() / 3600.0
    return merged


def build_notes_output(df: pd.DataFrame) -> pd.DataFrame:
    """Trims to the DoD-locked 6-column schema, exact column order + dtypes.
    Every other column computed along the way (source_table, raw_note_type,
    subject_id, note_seq, sepsis_onset_time, the non-canonical timestamp
    field) is diagnostic-only and intentionally NOT written to notes.parquet
    -- it still flows into notes_stats.json / notes_spot_check.json from the
    richer, pre-trim frame."""
    out = df[OUTPUT_COLUMNS].copy()
    out["hadm_id"] = out["hadm_id"].astype("int64")
    out["note_id"] = out["note_id"].astype(str)
    out["note_type"] = out["note_type"].astype(str)
    out["raw_text"] = out["raw_text"].astype(str)
    out["hours_before_onset"] = out["hours_before_onset"].astype(float)
    out["hours_since_admission"] = out["hours_since_admission"].astype(float)
    return out.reset_index(drop=True)


# ==============================================================================
# CHUNKING UTILITY -- MedPatch's convention, adapted (see ASSUMPTION A3/A5).
# Not called by this file's own main() by default (raw_text stays whole in
# the output); importable by each model's adapter, and optionally exercised
# here just for descriptive stats if --compute_chunking_stats is passed.
# ==============================================================================

def chunk_note_text(text, tokenizer=None, model_name: str = DEFAULT_TOKENIZER_NAME,
                     max_tokens: int = CHUNK_MAX_TOKENS) -> list:
    """
    MedPatch's chunking convention (Al Jorf & Shamout, MLHC 2025, sec 4.4,
    paraphrased): split a document into NON-OVERLAPPING chunks of
    `max_tokens` tokens; each chunk is later encoded independently and
    mean-pooled -- that encoding/pooling step happens in each model's own
    encoder, NOT here (PROJECT_CONTEXT.md rule #5). This function only does
    the split, and returns DECODED TEXT per chunk (not token ids), so any
    downstream tokenizer can re-encode it -- MedPatch used BioBERT for both
    RR and DN, but other baselines / our own model may use a different
    tokenizer (e.g. Clinical-Longformer per environment.yml), so baking
    BioBERT's specific subword boundaries into a *shared* utility would be
    exactly the kind of model-specific reduction this file's output is
    supposed to stay free of.

    `tokenizer`, if provided, must expose `.encode(text, add_special_tokens=False)
    -> list[int]` and `.decode(ids) -> str` (any HuggingFace tokenizer
    satisfies this). If omitted, a BioBERT tokenizer is loaded lazily via
    `transformers` (see ASSUMPTION A5) -- deliberately NOT imported at module
    level, so plain note extraction (this file's main path) never requires
    `transformers` or network access at all.
    """
    if text is None or (isinstance(text, float) and pd.isna(text)) or not str(text).strip():
        return []
    if tokenizer is None:
        from transformers import AutoTokenizer  # lazy import, see docstring above
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    token_ids = tokenizer.encode(str(text), add_special_tokens=False)
    if not token_ids:
        return []
    return [tokenizer.decode(token_ids[start:start + max_tokens])
            for start in range(0, len(token_ids), max_tokens)]


# ==============================================================================
# STATS / LEAKAGE DIAGNOSTICS
# ==============================================================================

def compute_leakage_diagnostics(notes_df: pd.DataFrame) -> dict:
    """
    Empirical checks that the METADATA (not this file's extraction logic)
    makes downstream temporal filtering (hours_before_onset >= lead_time_h,
    applied in each model's own adapter -- see ASSUMPTION A3) actually safe
    for discharge notes. This does not drop or gate anything itself.

    NOTE: pct_charttime_exactly_midnight always checks the raw `charttime`
    field specifically, regardless of which field is actually canonical for
    a given note type per NOTE_TYPE_TIMESTAMP_FIELD / ASSUMPTION A2. This is
    intentional -- it's an early-warning check for this exact class of
    problem recurring (e.g. on a future MIMIC-IV-Note release), not a
    real-time validity check of whatever `timestamp` ended up being used.
    Don't read "flag didn't fire" as "the canonical field is fine" -- check
    which field is canonical separately if that matters for your use case.
    """
    diag = {}
    for note_type in ("DN", "RR"):
        sub = notes_df[notes_df["note_type"] == note_type]
        entry = {"n_notes": int(len(sub))}
        if len(sub) and {"charttime", "storetime"}.issubset(sub.columns):
            gap_hours = (sub["storetime"] - sub["charttime"]).dt.total_seconds() / 3600.0
            entry["charttime_to_storetime_gap_hours"] = {
                "median": None if gap_hours.isna().all() else round(float(gap_hours.median()), 2),
                "pct_storetime_before_charttime": round(100.0 * (gap_hours < 0).mean(), 2),
            }
            midnight = (
                (sub["charttime"].dt.hour == 0)
                & (sub["charttime"].dt.minute == 0)
                & (sub["charttime"].dt.second == 0)
                & sub["charttime"].notna()
            )
            entry["pct_charttime_exactly_midnight"] = round(100.0 * midnight.mean(), 2)
        pos = sub[sub["hours_before_onset"].notna()]
        if len(pos):
            entry["hours_before_onset_on_positive_admissions"] = {
                "n": int(len(pos)),
                "median": round(float(pos["hours_before_onset"].median()), 2),
                "pct_at_or_after_onset": round(100.0 * (pos["hours_before_onset"] <= 0).mean(), 2),
                f"pct_more_than_{DN_SUSPICIOUS_EARLY_HOURS:.0f}h_before_onset":
                    round(100.0 * (pos["hours_before_onset"] > DN_SUSPICIOUS_EARLY_HOURS).mean(), 2),
            }
        diag[note_type] = entry

    for note_type, threshold in (("DN", 50.0),):
        sub = notes_df[notes_df["note_type"] == note_type]
        if len(sub):
            midnight_pct = diag[note_type].get("pct_charttime_exactly_midnight")
            if midnight_pct is not None and midnight_pct > threshold:
                diag[f"flag_{note_type}_timestamp_precision"] = (
                    f"WARNING: {midnight_pct}% of {note_type} raw charttimes are exactly "
                    f"midnight -- this looks like a date-only field with a fabricated "
                    f"00:00:00 time component, not real time-of-day precision (see "
                    f"ASSUMPTION A2). If NOTE_TYPE_TIMESTAMP_FIELD still has {note_type} "
                    f"on charttime, hours_before_onset for these notes carries up to "
                    f"~24h of hidden uncertainty even though the 'suspiciously early' "
                    f"check below didn't fire -- confirm which field is actually "
                    f"canonical for {note_type} before trusting hour-level filtering."
                )

    dn_pos = notes_df[(notes_df["note_type"] == "DN") & notes_df["hours_before_onset"].notna()]
    if len(dn_pos):
        pct_suspicious = 100.0 * (dn_pos["hours_before_onset"] > DN_SUSPICIOUS_EARLY_HOURS).mean()
        if pct_suspicious > 10.0:
            diag["flag"] = (
                f"WARNING: {round(pct_suspicious, 2)}% of discharge notes (DN) on sepsis-"
                f"positive admissions are timestamped more than {DN_SUSPICIOUS_EARLY_HOURS:.0f}h "
                f"before onset. Discharge summaries should generally cluster near end-of-stay "
                f"(well after onset, for admissions that survive to discharge); a high rate this "
                f"early suggests the canonical timestamp may not reliably reflect true note "
                f"availability for this cohort (ASSUMPTION A2). Manually check a few of these "
                f"specific admissions (notes_spot_check.json) before trusting downstream "
                f"hours_before_onset-based filtering for DN."
            )
    return diag


def compute_notes_stats(notes_df: pd.DataFrame, cohort_df: pd.DataFrame,
                         raw_table_counts: dict, timestamp_fallback_stats: dict,
                         timestamp_field_used, compute_exact_chunking_stats: bool = False,
                         tokenizer_name: str = DEFAULT_TOKENIZER_NAME,
                         chunk_max_tokens: int = CHUNK_MAX_TOKENS,
                         chunking_stats_sample_size: int = 2000,
                         rng_seed: int = 20240115) -> dict:
    """Required DoD stats (note counts per type, avg notes/admission, % zero-
    note admissions) plus leakage diagnostics and assumption-verification
    aids. `raw_table_counts` reflects the FULL raw tables regardless of
    --sample_size; everything else here reflects the (possibly sampled)
    cohort actually processed this run.

    `timestamp_field_used` is whatever main() actually used to resolve
    timestamps this run -- either the NOTE_TYPE_TIMESTAMP_FIELD dict (the
    normal, per-type-verified default) or a string describing a
    --timestamp_field override forcing both types onto one field. Recorded
    as-is (via json.dump's default=str) so notes_stats.json always shows
    exactly what was used, not just a nominal CLI value that might not
    match what actually happened -- see ASSUMPTION A11."""
    stats: dict = {}
    stats["timestamp_field_used"] = timestamp_field_used
    stats["timestamp_fallback"] = timestamp_fallback_stats
    stats["raw_table_counts_full_source (not sample-limited)"] = raw_table_counts

    n_cohort_admissions = int(cohort_df["hadm_id"].nunique())
    stats["n_cohort_admissions_this_run"] = n_cohort_admissions
    stats["n_sepsis_positive_admissions_this_run"] = int(cohort_df["sepsis_onset_time"].notna().sum())

    stats["raw_note_type_observed_by_source_table"] = {
        f"{src}:{raw_nt}": int(n)
        for (src, raw_nt), n in notes_df.groupby("source_table")["raw_note_type"].value_counts().items()
    }

    stats["total_notes"] = int(len(notes_df))
    stats["note_counts_per_type"] = {
        str(k): int(v) for k, v in notes_df["note_type"].value_counts().items()
    }

    notes_per_admission = notes_df.groupby("hadm_id").size()
    admissions_with_notes = set(notes_per_admission.index)
    all_cohort_hadm = set(cohort_df["hadm_id"])
    n_zero = len(all_cohort_hadm - admissions_with_notes)
    stats["avg_notes_per_admission"] = (
        round(len(notes_df) / n_cohort_admissions, 3) if n_cohort_admissions else None
    )
    stats["n_admissions_with_zero_notes"] = int(n_zero)
    stats["pct_admissions_with_zero_notes"] = (
        round(100.0 * n_zero / n_cohort_admissions, 2) if n_cohort_admissions else None
    )

    for nt in ("DN", "RR"):
        sub = notes_df[notes_df["note_type"] == nt]
        per_adm = sub.groupby("hadm_id").size()
        stats[f"pct_admissions_with_at_least_one_{nt}"] = (
            round(100.0 * len(per_adm) / n_cohort_admissions, 2) if n_cohort_admissions else None
        )
        stats[f"avg_{nt}_per_admission_among_those_with_at_least_one"] = (
            round(float(per_adm.mean()), 3) if len(per_adm) else 0.0
        )

    approx_words = notes_df["raw_text"].str.split().str.len()
    stats["approx_note_length_words (whitespace split, NOT a real tokenizer)"] = {
        "median": None if not len(notes_df) else float(approx_words.median()),
        "p90": None if not len(notes_df) else float(approx_words.quantile(0.9)),
        "pct_over_512_words": None if not len(notes_df) else round(100.0 * (approx_words > 512).mean(), 2),
    }

    if compute_exact_chunking_stats and len(notes_df):
        sample = notes_df.sample(
            n=min(chunking_stats_sample_size, len(notes_df)), random_state=rng_seed
        )
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            n_tokens = sample["raw_text"].apply(
                lambda t: len(tokenizer.encode(str(t), add_special_tokens=False))
            )
            stats["exact_chunking_stats"] = {
                "tokenizer_name": tokenizer_name,
                "sample_size": int(len(sample)),
                "median_tokens": float(n_tokens.median()),
                "pct_over_max_tokens": round(100.0 * (n_tokens > chunk_max_tokens).mean(), 2),
                "median_chunks_needed": float(np.ceil(n_tokens / chunk_max_tokens).median()),
            }
        except Exception as exc:  # network / model download / etc. -- don't fail the whole run over this
            stats["exact_chunking_stats"] = {"error": f"{type(exc).__name__}: {exc}"}

    stats["leakage_diagnostics"] = compute_leakage_diagnostics(notes_df)
    return stats


# ==============================================================================
# SPOT CHECK (DoD requirement: pick 2-3 admissions, confirm extracted notes/
# timestamps match the raw tables directly, confirm no post-prediction-
# window leakage for at least one example)
# ==============================================================================

def pick_spot_check_admissions(notes_df: pd.DataFrame, cohort_df: pd.DataFrame,
                                n: int = 3, seed: int = 20240115) -> list:
    """
    Deterministic selection biased to make the DoD's "confirm no post-
    prediction-window leakage for at least one example" satisfiable by
    construction: prefers one admission that is sepsis-positive AND has
    >=1 DN AND >=1 RR (so there's something concrete to check timing on),
    then fills the rest deterministically.
    """
    admissions_with_notes = notes_df["hadm_id"].unique()
    if len(admissions_with_notes) == 0:
        return []
    positive_hadm = set(cohort_df.loc[cohort_df["sepsis_onset_time"].notna(), "hadm_id"])
    has_dn = set(notes_df.loc[notes_df["note_type"] == "DN", "hadm_id"])
    has_rr = set(notes_df.loc[notes_df["note_type"] == "RR", "hadm_id"])
    ideal = sorted(positive_hadm & has_dn & has_rr & set(admissions_with_notes))

    rng = np.random.RandomState(seed)
    picked: list = []
    if ideal:
        picked.append(ideal[rng.randint(len(ideal))])
    remaining_pool = sorted(set(admissions_with_notes) - set(picked))
    rng.shuffle(remaining_pool)
    for hadm in remaining_pool:
        if len(picked) >= n:
            break
        picked.append(hadm)
    return picked[:n]


def _compare_notes_to_source(extracted_subset: pd.DataFrame, source_subset: pd.DataFrame) -> dict:
    """Pure comparison logic, unit-testable without duckdb: same set of
    note_ids, same raw_text, and `timestamp` matches ONE of the two raw
    fields (whichever was canonical -- see ASSUMPTION A2 / resolve_timestamp)."""
    result: dict = {"n_extracted": int(len(extracted_subset)), "n_source": int(len(source_subset))}
    extracted_ids = set(extracted_subset["note_id"])
    source_ids = set(source_subset["note_id"])
    result["note_id_sets_match"] = extracted_ids == source_ids
    result["missing_from_extracted"] = sorted(source_ids - extracted_ids)
    result["extra_in_extracted"] = sorted(extracted_ids - source_ids)

    mismatches = []
    merged = extracted_subset.merge(source_subset, on="note_id", suffixes=("_extracted", "_source"))
    for _, row in merged.iterrows():
        text_match = row["raw_text_extracted"] == row["raw_text_source"]
        ts_match = row["timestamp"] in (row["charttime_source"], row["storetime_source"])
        if not (text_match and ts_match):
            mismatches.append({
                "note_id": row["note_id"], "text_match": bool(text_match), "timestamp_match": bool(ts_match),
            })
    result["mismatches"] = mismatches
    result["pass"] = bool(result["note_id_sets_match"] and not mismatches)
    return result


def spot_check_notes(con: duckdb.DuckDBPyConnection, notes_df: pd.DataFrame,
                      cohort_df: pd.DataFrame, n: int = 3, seed: int = 20240115) -> list:
    picked = pick_spot_check_admissions(notes_df, cohort_df, n=n, seed=seed)
    onset_by_hadm = cohort_df.set_index("hadm_id")["sepsis_onset_time"].to_dict()
    results = []
    for hadm_id in picked:
        extracted_subset = notes_df[notes_df["hadm_id"] == hadm_id]
        source_subset = _query_raw_notes_for_hadm(con, hadm_id)
        comparison = _compare_notes_to_source(extracted_subset, source_subset)

        onset_time = onset_by_hadm.get(hadm_id)
        timeline = []
        for _, row in extracted_subset.sort_values("timestamp").iterrows():
            hbo = row["hours_before_onset"]
            timeline.append({
                "note_id": row["note_id"],
                "note_type": row["note_type"],
                "charttime": str(row.get("charttime")),
                "storetime": str(row.get("storetime")),
                "timestamp_used": str(row["timestamp"]),
                "hours_before_onset": None if pd.isna(hbo) else round(float(hbo), 2),
                "recorded_at_or_after_onset": None if pd.isna(hbo) else bool(hbo <= 0),
            })
        results.append({
            "hadm_id": int(hadm_id),
            "sepsis_onset_time": None if onset_time is None or pd.isna(onset_time) else str(onset_time),
            "comparison_vs_raw_tables": comparison,
            "note_timeline": timeline,
        })
    return results


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Extract MIMIC-IV-Note discharge summaries + radiology reports "
                    "for the locked sepsis cohort (Milestone 1)."
    )
    parser.add_argument("--mimic_note_dir", type=str, required=True,
                         help="Path to the MIMIC-IV-Note folder containing discharge.csv "
                              "and radiology.csv. Not shown in your directory listing, so "
                              "there's no default -- see module docstring.")
    parser.add_argument("--sepsis_labels_path", type=str, default="data/cohort/sepsis_labels.parquet",
                         help="Milestone 0 output (label_sepsis3.py).")
    parser.add_argument("--out_dir", type=str, default="data/cohort",
                         help="Output directory for notes.parquet, notes_stats.json, "
                              "notes_spot_check.json.")
    parser.add_argument("--cache_dir", type=str, default="data/cache",
                         help="Parquet cache dir for the raw note CSVs -- same cache_dir "
                              "convention as label_sepsis3.py, safe to share the same folder.")
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--sample_size", type=int, default=None,
                         help="If set, run on a random sample of this many eligible hadm_ids "
                              "instead of the full cohort (for quick local testing). See "
                              "ASSUMPTION A9.")
    parser.add_argument("--timestamp_field", choices=["charttime", "storetime"],
                         default=None,
                         help="OVERRIDE: force BOTH note types onto this single field, "
                              "ignoring the per-note-type default. Leave unset (default) "
                              "to use NOTE_TYPE_TIMESTAMP_FIELD (DN->storetime, "
                              "RR->charttime), which was verified directly against the "
                              "real data -- see ASSUMPTION A2. Mainly useful for "
                              "debugging or reverting to old single-field behavior.")
    parser.add_argument("--chunk_max_tokens", type=int, default=CHUNK_MAX_TOKENS)
    parser.add_argument("--tokenizer_name", type=str, default=DEFAULT_TOKENIZER_NAME,
                         help="See ASSUMPTION A5. Only used if --compute_chunking_stats is set "
                              "(chunk_note_text() itself is a utility for downstream adapters, "
                              "not called in this file's default path).")
    parser.add_argument("--compute_chunking_stats", action="store_true",
                         help="Optionally compute EXACT (tokenizer-based, not word-count-"
                              "approximate) long-document stats on a sample. Requires "
                              "transformers + downloading a tokenizer -- off by default so "
                              "plain extraction never needs network access.")
    parser.add_argument("--chunking_stats_sample_size", type=int, default=2000)
    parser.add_argument("--n_spot_check", type=int, default=3)
    parser.add_argument("--spot_check_seed", type=int, default=20240115)
    args = parser.parse_args()

    note_dir = Path(args.mimic_note_dir)
    sepsis_labels_path = Path(args.sepsis_labels_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = None if args.no_cache else Path(args.cache_dir)

    print("[notes_extraction] Loading eligible cohort from sepsis_labels.parquet...", file=sys.stderr)
    cohort_df = load_sepsis_cohort(sepsis_labels_path)
    if args.sample_size is not None:
        # ORDER BY before sampling -- see ASSUMPTION A9, mirrors
        # label_sepsis3.py's build_cohort() reproducibility fix exactly.
        cohort_df = cohort_df.sort_values("hadm_id").reset_index(drop=True)
        cohort_df = cohort_df.sample(
            n=min(args.sample_size, len(cohort_df)), random_state=0
        ).reset_index(drop=True)
    print(f"[notes_extraction]   -> {len(cohort_df)} eligible admissions "
          f"({int(cohort_df['sepsis_onset_time'].notna().sum())} sepsis-positive)", file=sys.stderr)

    print(f"[notes_extraction] Connecting to MIMIC-IV-Note CSVs via DuckDB "
          f"(note_dir={note_dir}, cache_dir={cache_dir if cache_dir else 'disabled'})...",
          file=sys.stderr)
    con = connect_duckdb(note_dir, cache_dir=cache_dir)
    raw_table_counts = compute_raw_table_counts(con)

    print("[notes_extraction] [1/3] Extracting discharge + radiology notes for cohort "
          "admissions...", file=sys.stderr)
    t0 = time.time()
    raw_df = extract_raw_notes(con, cohort_df)
    print(f"[notes_extraction] [1/3] done in {time.time() - t0:.0f}s "
          f"({len(raw_df)} raw note rows)", file=sys.stderr)

    raw_df["note_type"] = assign_note_type(raw_df)

    # BUG FIX (ASSUMPTION A11): resolve_timestamp() now takes a per-note-type
    # dict, not a single field string. --timestamp_field is a full override
    # (forces both types onto one field) rather than the sole selector.
    if args.timestamp_field is not None:
        note_type_field = {"DN": args.timestamp_field, "RR": args.timestamp_field}
        timestamp_field_used = f"OVERRIDE: both DN and RR forced to '{args.timestamp_field}'"
    else:
        note_type_field = NOTE_TYPE_TIMESTAMP_FIELD
        timestamp_field_used = dict(NOTE_TYPE_TIMESTAMP_FIELD)
    raw_df["timestamp"], timestamp_fallback_stats = resolve_timestamp(raw_df, note_type_field)

    print("[notes_extraction] [2/3] Joining against sepsis_onset_time, computing "
          "hours_before_onset...", file=sys.stderr)
    t0 = time.time()
    notes_df = compute_hours_before_onset(raw_df, cohort_df)
    print(f"[notes_extraction] [2/3] done in {time.time() - t0:.0f}s "
          f"({len(notes_df)} notes retained)", file=sys.stderr)

    print("[notes_extraction] [3/3] Computing stats + leakage diagnostics...", file=sys.stderr)
    stats = compute_notes_stats(
        notes_df, cohort_df, raw_table_counts, timestamp_fallback_stats, timestamp_field_used,
        compute_exact_chunking_stats=args.compute_chunking_stats,
        tokenizer_name=args.tokenizer_name, chunk_max_tokens=args.chunk_max_tokens,
        chunking_stats_sample_size=args.chunking_stats_sample_size,
        rng_seed=args.spot_check_seed,
    )
    stats["note_module_version_caveat"] = (
        "MIMIC-IV-Note had not (as of when this script was written) been re-released to "
        "match core MIMIC-IV v3.1 -- it remained at v2.2, tied to core MIMIC-IV v2.2's "
        "admission set. If your hosp/icu directories are v3.1, a small number of the "
        "newest hadm_ids in your cohort may genuinely have zero notes for this reason "
        "alone -- see ASSUMPTION A6. Check physionet.org/content/mimic-iv-note/ for a "
        "newer release before treating this run's zero-notes rate as purely a data-"
        "quality signal."
    )
    stats["note_sample_size"] = (
        f"Run with --sample_size {args.sample_size}: cohort-derived counts above are "
        f"scoped to this sample; raw_table_counts is not."
        if args.sample_size else
        "Full run (no --sample_size)."
    )

    output_df = build_notes_output(notes_df)
    out_path = out_dir / "notes.parquet"
    output_df.to_parquet(out_path, index=False)
    print(f"[notes_extraction] Wrote {out_path} ({len(output_df)} rows)", file=sys.stderr)

    stats_path = out_dir / "notes_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, default=str)
    print(json.dumps(stats, indent=2, default=str))
    print(f"[notes_extraction] Wrote {stats_path}", file=sys.stderr)
    for flag_key, flag_msg in stats.get("leakage_diagnostics", {}).items():
        if flag_key.startswith("flag"):
            print(f"[notes_extraction] {flag_msg}", file=sys.stderr)

    print(f"[notes_extraction] Running spot-check on {args.n_spot_check} admission(s)...",
          file=sys.stderr)
    spot_check_results = spot_check_notes(
        con, notes_df, cohort_df, n=args.n_spot_check, seed=args.spot_check_seed
    )
    spot_check_path = out_dir / "notes_spot_check.json"
    with open(spot_check_path, "w") as f:
        json.dump(spot_check_results, f, indent=2, default=str)
    for r in spot_check_results:
        status = "PASS" if r["comparison_vs_raw_tables"]["pass"] else "FAIL"
        print(f"[notes_extraction]   hadm_id={r['hadm_id']}: {status} "
              f"({r['comparison_vs_raw_tables']['n_extracted']} notes, "
              f"onset={r['sepsis_onset_time']})", file=sys.stderr)
    print(f"[notes_extraction] Wrote {spot_check_path}", file=sys.stderr)

    if any(r["comparison_vs_raw_tables"]["pass"] is False for r in spot_check_results):
        print("[notes_extraction] WARNING: at least one spot-check admission did NOT "
              "match the raw tables -- see notes_spot_check.json before trusting "
              "notes.parquet.", file=sys.stderr)


if __name__ == "__main__":
    main()