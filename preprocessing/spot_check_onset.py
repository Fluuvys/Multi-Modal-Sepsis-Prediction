"""
preprocessing/spot_check_onset.py
================================================================================
Manual spot-check helper for Milestone 0 sign-off (the last item in the
Definition of Done that stats alone can't substitute for).

Picks a handful of positive-label patients whose sepsis_onset_time falls in
a chosen hours-since-admission range (default 6-10h, matching the locked
run's IQR) and prints, for each, two things:

  1. The per-hour SOFA trajectory around onset -- BOTH the raw underlying
     values (platelet count, bilirubin, MBP, etc.) AND the derived 0-4
     component scores, for a few hours before onset through onset itself.
     This directly answers the question the diagnostic couldn't: does the
     SOFA jump reflect a genuinely NEW abnormal value, or did a component
     just get its first-ever measurement of the stay?

  2. The raw chartevents/labevents/outputevents/inputevents rows in a
     window around onset, so you can see the literal source data behind
     the trajectory in (1) -- not just our re-derived summary of it.

This is READ-ONLY. It does not change sepsis_labels.parquet, cohort_stats
.json, or any label. It imports label_sepsis3.py directly and calls its own
_compute_hourly_sofa / _score_sofa_components functions, so what you are
inspecting here is guaranteed to be exactly the computation that produced
the label -- not a reimplementation that could silently drift from it.

USAGE:
    python preprocessing/spot_check_onset.py \\
      --mimic_hosp_dir "/path/to/mimic-iv-3.1/hosp" \\
      --mimic_icu_dir "/path/to/mimic-iv-3.1/icu" \\
      --labels_path data/cohort/sepsis_labels.parquet \\
      --cache_dir data/cache \\
      --n 3 --hour_lo 6 --hour_hi 10

Re-uses the same Parquet cache label_sepsis3.py already built, so this runs
in seconds, not minutes -- it does not re-scan the raw MIMIC-IV CSVs.

Output: one text block per patient printed to stdout, and (optionally) an
equivalent .txt file per patient under --out_dir for pasting into a review
doc / PR description.
================================================================================
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import label_sepsis3 as m  # noqa: E402  (must come after sys.path insert)


# Human-readable labels for the itemids label_sepsis3.py actually uses, so
# the raw-row printout is legible without cross-referencing d_items/d_labitems.
ITEMID_LABELS = {
    m.ITEMID_PLATELET: "Platelet count (labs)",
    m.ITEMID_BILIRUBIN_TOTAL: "Total bilirubin (labs)",
    m.ITEMID_CREATININE: "Creatinine (labs)",
    m.ITEMID_PO2: "PaO2 (labs, blood gas)",
    m.ITEMID_SPECIMEN_TYPE: "Specimen type (labs, blood gas)",
    m.ITEMID_FIO2_LABEVENTS: "FiO2 (labs, blood gas panel)",
    m.ITEMID_FIO2_CHARTEVENTS: "FiO2 (chartevents)",
    m.ITEMID_VENT_MODE: "Ventilator mode (chartevents)",
    m.ITEMID_VENT_MODE_HAMILTON: "Ventilator mode - Hamilton (chartevents)",
    m.ITEMID_O2_DEVICE: "O2 delivery device (chartevents)",
    m.ITEMID_GCS_MOTOR: "GCS - Motor (chartevents)",
    m.ITEMID_GCS_VERBAL: "GCS - Verbal (chartevents)",
    m.ITEMID_GCS_EYES: "GCS - Eyes (chartevents)",
    m.ITEMID_NOREPINEPHRINE: "Norepinephrine rate (inputevents)",
    m.ITEMID_EPINEPHRINE: "Epinephrine rate (inputevents)",
    m.ITEMID_DOPAMINE: "Dopamine rate (inputevents)",
    m.ITEMID_DOBUTAMINE: "Dobutamine rate (inputevents)",
}
for _i in m.ITEMID_MBP:
    ITEMID_LABELS[_i] = "Mean arterial pressure (chartevents)"
for _i in m.ITEMID_URINE_OUTPUT:
    ITEMID_LABELS[_i] = "Urine output (outputevents)"


def pick_candidates(labels_path: Path, icu_dir: Path, n: int,
                     hour_lo: float, hour_hi: float, seed: int) -> pd.DataFrame:
    labels = pd.read_parquet(labels_path)
    icustays = pd.read_csv(
        icu_dir / "icustays.csv",
        usecols=["subject_id", "hadm_id", "stay_id", "intime", "outtime"],
    )
    icustays["intime"] = pd.to_datetime(icustays["intime"])
    icustays["outtime"] = pd.to_datetime(icustays["outtime"])

    pos = labels[labels["label"] == 1].merge(
        icustays, on=["subject_id", "hadm_id"], how="inner"
    )
    pos["hours_to_onset"] = (
        pd.to_datetime(pos["sepsis_onset_time"]) - pos["intime"]
    ).dt.total_seconds() / 3600.0

    candidates = pos[(pos["hours_to_onset"] >= hour_lo) & (pos["hours_to_onset"] <= hour_hi)]
    if candidates.empty:
        print(
            f"No positive patients with onset between {hour_lo}h and {hour_hi}h "
            f"found in {labels_path}. Try widening --hour_lo/--hour_hi.",
            file=sys.stderr,
        )
        sys.exit(1)

    return candidates.sample(n=min(n, len(candidates)), random_state=seed).reset_index(drop=True)


def print_trajectory(con, subject_id: int, hadm_id: int, stay_id: int,
                      intime: pd.Timestamp, outtime: pd.Timestamp,
                      onset_time: pd.Timestamp, hours_before: int, hours_after: int) -> None:
    stay_map = pd.DataFrame([{
        "stay_id": stay_id, "hadm_id": hadm_id, "subject_id": subject_id,
        "intime": intime, "outtime": outtime,
    }])
    raw = m._compute_hourly_sofa(con, stay_map)
    scored = m._score_sofa_components(raw)
    merged = raw.merge(
        scored[["stay_id", "hr", "sofa_total"] + [f"{c}_24h" for c in
            ["respiration", "coagulation", "liver", "cardiovascular", "cns", "renal"]]],
        on=["stay_id", "hr"],
    )

    onset_hr = merged.loc[
        (merged["hour_end"] - onset_time).abs().idxmin(), "hr"
    ]
    window = merged[
        (merged["hr"] >= onset_hr - hours_before) & (merged["hr"] <= onset_hr + hours_after)
    ].copy()
    window["is_onset_hour"] = window["hr"] == onset_hr

    display_cols = [
        "hr", "hour_end", "is_onset_hour", "sofa_total",
        "pf_novent", "pf_vent", "platelet_min", "bilirubin_max", "creatinine_max",
        "mbp_min", "rate_norepi_max", "rate_epi_max", "rate_dopa_max", "rate_dobu_max",
        "gcs_min", "urineoutput_24hr_valid", "uo_tm_24hr_valid",
        "respiration_24h", "coagulation_24h", "liver_24h", "cardiovascular_24h",
        "cns_24h", "renal_24h",
    ]
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print(window[display_cols].to_string(index=False))


def print_raw_events(con, hadm_id: int, stay_id: int,
                      window_start: pd.Timestamp, window_end: pd.Timestamp) -> None:
    itemids_sql = ",".join(str(i) for i in ITEMID_LABELS.keys())

    # NOTE: itemid is explicitly CAST to BIGINT in every branch below. This
    # is required for correctness, not just style -- DuckDB's UNION ALL
    # coerces a column to the common type across all branches, and if any
    # one source table's itemid gets type-sniffed as VARCHAR (e.g. an empty
    # or unusual CSV), the WHOLE unioned itemid column silently becomes
    # VARCHAR ("51265" instead of 51265), which then fails to match the
    # int-keyed ITEMID_LABELS lookup below with no error, just blank labels.
    rows = con.execute(f"""
        SELECT 'chartevents' AS source, charttime, CAST(itemid AS BIGINT) AS itemid,
               CAST(value AS VARCHAR) AS value
        FROM chartevents
        WHERE stay_id = {stay_id} AND itemid IN ({itemids_sql})
            AND charttime >= '{window_start}' AND charttime <= '{window_end}'
        UNION ALL
        SELECT 'labevents' AS source, charttime, CAST(itemid AS BIGINT) AS itemid,
               CAST(value AS VARCHAR) AS value
        FROM labevents
        WHERE hadm_id = {hadm_id} AND itemid IN ({itemids_sql})
            AND charttime >= '{window_start}' AND charttime <= '{window_end}'
        UNION ALL
        SELECT 'outputevents' AS source, charttime, CAST(itemid AS BIGINT) AS itemid,
               CAST(value AS VARCHAR) AS value
        FROM outputevents
        WHERE stay_id = {stay_id} AND itemid IN ({itemids_sql})
            AND charttime >= '{window_start}' AND charttime <= '{window_end}'
        UNION ALL
        SELECT 'inputevents' AS source, CAST(starttime AS TIMESTAMP) AS charttime,
               CAST(itemid AS BIGINT) AS itemid,
               CAST(rate AS VARCHAR) || ' ' || CAST(rateuom AS VARCHAR) AS value
        FROM inputevents
        WHERE stay_id = {stay_id} AND itemid IN ({itemids_sql})
            AND starttime >= '{window_start}' AND starttime <= '{window_end}'
        ORDER BY charttime
    """).df()

    if rows.empty:
        print("  (no raw events found in this window for the itemids we use)")
        return

    # second line of defense: force the dtype on the Python side too, in
    # case a future source table quirk reintroduces a type mismatch
    rows["itemid"] = rows["itemid"].astype("int64")
    rows["label"] = rows["itemid"].map(ITEMID_LABELS)
    pd.set_option("display.max_rows", 200)
    print(rows[["charttime", "source", "label", "itemid", "value"]].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description="Manual spot-check helper: print SOFA trajectory + raw events "
                     "around onset for a sample of positive-label patients."
    )
    parser.add_argument("--mimic_hosp_dir", type=str, required=True)
    parser.add_argument("--mimic_icu_dir", type=str, required=True)
    parser.add_argument("--labels_path", type=str, default="data/cohort/sepsis_labels.parquet")
    parser.add_argument("--cache_dir", type=str, default="data/cache",
                         help="Reuses the Parquet cache label_sepsis3.py already built.")
    parser.add_argument("--n", type=int, default=3, help="Number of patients to sample.")
    parser.add_argument("--hour_lo", type=float, default=6.0,
                         help="Lower bound (hours since admission) for onset time to sample from.")
    parser.add_argument("--hour_hi", type=float, default=10.0,
                         help="Upper bound (hours since admission) for onset time to sample from.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hours_before", type=int, default=3,
                         help="How many hours before onset to show in the trajectory table.")
    parser.add_argument("--hours_after", type=int, default=1,
                         help="How many hours after onset to show in the trajectory table.")
    parser.add_argument("--out_dir", type=str, default=None,
                         help="If set, also write one .txt report per patient here.")
    args = parser.parse_args()

    hosp_dir = Path(args.mimic_hosp_dir)
    icu_dir = Path(args.mimic_icu_dir)
    labels_path = Path(args.labels_path)
    cache_dir = Path(args.cache_dir)

    print(f"[spot_check] Selecting {args.n} positive patients with onset in "
          f"[{args.hour_lo}h, {args.hour_hi}h] from {labels_path}...", file=sys.stderr)
    candidates = pick_candidates(labels_path, icu_dir, args.n, args.hour_lo, args.hour_hi, args.seed)

    print(f"[spot_check] Connecting to MIMIC-IV via the existing Parquet cache "
          f"({cache_dir})...", file=sys.stderr)
    con = m.connect_duckdb(hosp_dir, icu_dir, sample_size=None, cache_dir=cache_dir)

    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    for _, row in candidates.iterrows():
        subject_id, hadm_id, stay_id = int(row.subject_id), int(row.hadm_id), int(row.stay_id)
        intime, outtime = row.intime, row.outtime
        onset_time = pd.to_datetime(row.sepsis_onset_time)
        hours_to_onset = row.hours_to_onset

        header = (
            f"\n{'='*100}\n"
            f"subject_id={subject_id}  hadm_id={hadm_id}  stay_id={stay_id}\n"
            f"intime={intime}  onset_time={onset_time}  "
            f"hours_to_onset={hours_to_onset:.2f}h  sofa_at_onset={row.sofa_at_onset}\n"
            f"{'='*100}\n"
            f"--- SOFA trajectory (hr = hours since admission; is_onset_hour marks the "
            f"detected onset hour) ---"
        )
        print(header)
        print_trajectory(con, subject_id, hadm_id, stay_id, intime, outtime,
                          onset_time, args.hours_before, args.hours_after)

        print(f"\n--- Raw events in [{args.hours_before}h before onset, "
              f"{args.hours_after}h after onset] ---")
        window_start = onset_time - pd.Timedelta(hours=args.hours_before)
        window_end = onset_time + pd.Timedelta(hours=args.hours_after)
        print_raw_events(con, hadm_id, stay_id, window_start, window_end)

        if out_dir:
            import io
            import contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                print(header)
                print_trajectory(con, subject_id, hadm_id, stay_id, intime, outtime,
                                  onset_time, args.hours_before, args.hours_after)
                print(f"\n--- Raw events in [{args.hours_before}h before onset, "
                      f"{args.hours_after}h after onset] ---")
                print_raw_events(con, hadm_id, stay_id, window_start, window_end)
            out_path = out_dir / f"spot_check_subject_{subject_id}.txt"
            out_path.write_text(buf.getvalue())
            print(f"[spot_check] Wrote {out_path}", file=sys.stderr)

    print(f"\n[spot_check] Done. Reviewed {len(candidates)} patients. "
          f"For each: check whether the SOFA jump at is_onset_hour reflects a "
          f"NEW abnormal raw value (visible in the raw-events table) versus a "
          f"component that simply had no earlier measurement to compare against.",
          file=sys.stderr)


if __name__ == "__main__":
    main()