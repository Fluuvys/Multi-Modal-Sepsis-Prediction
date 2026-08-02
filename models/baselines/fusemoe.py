"""
fusemoe.py -- baseline reproduction of FuseMoE (Han et al., 2024)

mTAND irregular-time encoding + sparse Mixture-of-Experts fusion (Laplace gating).
TS + notes + CXR (their repo also supports ECG -- we don't need it).

TODO:
    [ ] Pull mTAND implementation from FuseMoE's public repo
    [ ] Adapt fusion/MoE layer to our modality set and cohort
    [ ] Train/evaluate on our sepsis-onset labels + lead-time sweep
"""

# TODO: implementation goes here
