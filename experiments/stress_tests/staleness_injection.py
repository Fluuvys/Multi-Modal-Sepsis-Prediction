"""
staleness_injection.py -- Table 4(b), the most direct test of our actual claim

Artificially ages the last CXR/note at inference to 6h/24h/48h/72h, reports AUROC
degradation curve. Don't skip this even if main results already look good -- see
docs/experiment_blueprint.md non-negotiables.

TODO:
    [ ] Implement timestamp manipulation at inference time
    [ ] Run across all models in Table 2
    [ ] Plot degradation curves (Figure 4) -- expect our model's slope to be flatter
"""

# TODO: implementation goes here
