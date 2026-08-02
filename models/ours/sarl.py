"""
sarl.py -- Staleness-Aware Regularization Loss (our secondary contribution)

Auxiliary loss term added to the main task loss. Purpose: prevent SDCA's decay function
from collapsing to a degenerate solution under pure task-loss pressure -- e.g. ignoring
elapsed time entirely, or staying overconfident when only stale data is available.

TODO:
    [ ] Define the exact penalty (e.g. penalize low predictive entropy when all
        available modalities are above some staleness threshold) -- document the final
        formulation here once decided
    [ ] Implement as an addable term to the main task loss, with a tunable weight
    [ ] Needs an on/off ablation switch -- see docs/experiment_blueprint.md Table 3
"""

# TODO: implementation goes here
