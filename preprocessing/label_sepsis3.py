"""
label_sepsis3.py

PURPOSE:
    Single source of truth for Sepsis-3 label construction on MIMIC-IV.
    Every downstream experiment depends on this file's output -- do NOT reimplement
    this logic anywhere else in the repo. If the definition needs to change, discuss
    with the team first (see PROJECT_CONTEXT.md rule #2).

DEFINITION (locked, see PROJECT_CONTEXT.md section 5):
    - Suspicion of infection: earlier timestamp of IV antibiotics + blood cultures,
      within 24h (antibiotics -> cultures) or 72h (cultures -> antibiotics).
    - Sepsis onset: suspicion of infection AND SOFA score increase >= 2 points,
      within a -48h/+24h window around suspicion.

TODO (assign as a GitHub issue, milestone 0):
    [ ] Implement suspicion-of-infection extraction from MIMIC-IV prescriptions +
        microbiologyevents tables
    [ ] Implement SOFA score computation over time from chartevents/labevents
    [ ] Combine into final sepsis onset timestamp per admission
    [ ] Output: one row per admission with columns
        [subject_id, hadm_id, sepsis_onset_time, sofa_at_onset, excluded_reason]
    [ ] Sanity check cohort size/positive rate against published Sepsis-3-on-MIMIC
        benchmarks before moving on -- do not proceed to modeling until this passes

INPUT:  raw MIMIC-IV tables (see data/raw_links/)
OUTPUT: data/cohort/sepsis_labels.csv
"""

# TODO: implementation goes here
