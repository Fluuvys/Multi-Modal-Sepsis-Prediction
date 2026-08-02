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

# TODO: implementation goes here
