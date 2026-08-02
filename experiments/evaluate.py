"""
evaluate.py -- shared evaluation logic for every model/baseline

Computes AUROC, AUPRC (primary metric given class imbalance), and ECE/MCE for
calibration, with bootstrapped confidence intervals -- used identically across every
baseline and our own model so comparisons in Table 2 are apples-to-apples.

TODO:
    [ ] AUROC / AUPRC with bootstrapped CI (match methodology to MedPatch's approach
        for comparability)
    [ ] ECE / MCE / Brier score
    [ ] Output format consistent across all models -- one shared results schema
"""

# TODO: implementation goes here
