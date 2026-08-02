"""
cxr_linking.py

PURPOSE:
    Join MIMIC-CXR studies to admissions by subject_id, keeping the FULL sequence of
    CXRs per patient with real timestamps -- NOT just the most-recent one (this is
    the exact limitation MedPatch names as their own future work; don't repeat it).

TODO:
    [ ] Join CXR studies to admissions
    [ ] Preserve full per-patient CXR sequence + timestamps
    [ ] Decide + document image preprocessing (resolution, normalization) matching
        baseline reproductions for fair comparison

INPUT:  raw MIMIC-CXR metadata + image files
OUTPUT: data/cohort/cxr/ (metadata) + image path references
"""

# TODO: implementation goes here
