"""
backbone.py -- OUR OWN fusion backbone (not forked from any baseline paper)

Standard multi-head cross-attention across the three modality token streams (TS, notes,
CXR), with a lightweight gating step and a shared decoder. This is deliberately a
"boring" architecture -- see docs/gap_and_solution.md for why all the novelty lives in
sdca.py and sarl.py, not here. This file should be the "backbone with SDCA/SARL removed"
row in the Table 3 ablation.

TODO:
    [ ] Implement standard multi-head cross-attention fusion over TS/notes/CXR tokens
    [ ] Implement shared decoder -> risk score per lead-time horizon
    [ ] Verify this alone (no SDCA/SARL) trains stably and produces sane AUROC/AUPRC
        before adding sdca.py / sarl.py on top
"""

# TODO: implementation goes here
