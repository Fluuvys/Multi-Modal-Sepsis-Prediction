"""
notes_extraction.py

PURPOSE:
    Extract radiology reports + discharge notes from MIMIC-IV-Note, keeping the FULL
    timestamped sequence per patient -- this is a deliberate difference from MedPatch,
    which uses only a capped/most-recent subset. We need the full sequence because
    SDCA's whole point is reasoning about note staleness over time.

TODO:
    [ ] Extract radiology reports with their real timestamps
    [ ] Extract discharge notes (careful: only usable for tasks/times where they don't
        leak future information relative to the prediction point -- discuss before use)
    [ ] Chunk long documents (>512 tokens) consistently with baseline reproductions
        for fair comparison
    [ ] Output one timestamped note sequence per admission

INPUT:  raw MIMIC-IV-Note tables
OUTPUT: data/cohort/notes/
"""

# TODO: implementation goes here
