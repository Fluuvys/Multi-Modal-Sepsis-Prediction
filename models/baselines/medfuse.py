"""
medfuse.py -- baseline reproduction of MedFuse (Hayat et al., 2022)

TS + CXR fusion via LSTM sequence fusion. 2 modalities only -- report this baseline's
results explicitly labeled as "TS+CXR (no notes)" in Table 2, don't present as a full
tri-modal comparison.

TODO:
    [ ] Reimplement (or adapt from public repo if available) LSTM-based sequence fusion
    [ ] Train/evaluate on our sepsis-onset labels + lead-time sweep
    [ ] Sanity check against original paper's reported ballpark on the ORIGINAL
        mortality task first, before trusting the sepsis-task numbers
"""

# TODO: implementation goes here
