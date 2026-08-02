"""
sdca.py -- Staleness-Decay Cross-Attention (our primary contribution)

Multiplies a learned decay term into fusion/attention weights BEFORE the softmax,
computed from:
    - elapsed time since the observation's capture (Delta t)
    - current patient trajectory volatility (rolling stat from the TS stream)

Modality-agnostic by design -- must work identically for TS, notes, and CXR tokens.
See docs/gap_and_solution.md section "SDCA" for the full motivation.

TODO:
    [ ] Define the trajectory-volatility statistic precisely (e.g. rolling variance /
        rate-of-change over a fixed lookback window) -- document choice here once made
    [ ] Implement the decay function (start simple: learned exponential decay rate per
        modality, extend to volatility-conditioned form after backbone.py is validated)
    [ ] Support per-modality vs shared decay parameterization (needed for ablation)
    [ ] Support time-only vs volatility-aware decay (needed for ablation)
"""

# TODO: implementation goes here
