from __future__ import annotations
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
"""
evaluate.py -- shared evaluation logic for every model/baseline (PROJECT_CONTEXT.md
Section 7: "Metrics"). Every model must be scored through THIS file so Table 2 is
apples-to-apples -- don't let a baseline compute its own metrics.

Provides:
    compute_metrics(y_true, y_prob)          -- AUROC/AUPRC/ECE/Brier + bootstrap CIs
    lead_time_sweep(y_true, y_prob, hours)    -- stratify by TRUE distance-to-onset
                                                  (Table 2 / Figure 3, per Section 7:
                                                  "produced by stratifying test
                                                  predictions ... not by training
                                                  separate models")
    evaluate_both_protocols(...)              -- standard + strict pre-suspicion
                                                  (Section 4, required, not optional)
"""


from dataclasses import dataclass, asdict
from typing import Optional, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss


@dataclass
class MetricResult:
    auroc: float
    auroc_ci_lo: float
    auroc_ci_hi: float
    auprc: float
    auprc_ci_lo: float
    auprc_ci_hi: float
    ece: float
    brier: float
    n: int
    n_positive: int

    def to_dict(self):
        return asdict(self)


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Standard equal-width-bin ECE. MCE is the max per-bin gap instead of the
    weighted-mean gap -- swap np.average(...) for gaps.max() below if you need it too."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece, gaps, weights = 0.0, [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (y_prob > lo) & (y_prob <= hi) if lo > 0 else (y_prob >= lo) & (y_prob <= hi)
        if in_bin.sum() == 0:
            continue
        bin_acc = y_true[in_bin].mean()
        bin_conf = y_prob[in_bin].mean()
        gap = abs(bin_acc - bin_conf)
        weight = in_bin.sum() / len(y_prob)
        gaps.append(gap)
        weights.append(weight)
        ece += weight * gap
    return float(ece)


def _bootstrap_ci(y_true: np.ndarray, y_prob: np.ndarray, metric_fn, n_boot: int, rng: np.random.Generator,
                   alpha: float = 0.05):
    n = len(y_true)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_true[idx], y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue  # degenerate resample, skip (matters at small n / heavy class imbalance)
        stats.append(metric_fn(yt, yp))
    if not stats:
        return float("nan"), float("nan")
    lo = np.percentile(stats, 100 * alpha / 2)
    hi = np.percentile(stats, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


def compute_metrics(y_true: Sequence[float], y_prob: Sequence[float], n_boot: int = 1000,
                     seed: int = 1002) -> MetricResult:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    assert y_true.shape == y_prob.shape and y_true.ndim == 1

    rng = np.random.default_rng(seed)
    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    auroc_lo, auroc_hi = _bootstrap_ci(y_true, y_prob, roc_auc_score, n_boot, rng)
    auprc_lo, auprc_hi = _bootstrap_ci(y_true, y_prob, average_precision_score, n_boot, rng)

    return MetricResult(
        auroc=float(auroc), auroc_ci_lo=auroc_lo, auroc_ci_hi=auroc_hi,
        auprc=float(auprc), auprc_ci_lo=auprc_lo, auprc_ci_hi=auprc_hi,
        ece=expected_calibration_error(y_true, y_prob),
        brier=float(brier_score_loss(y_true, y_prob)),
        n=len(y_true), n_positive=int(y_true.sum()),
    )


def lead_time_sweep(y_true: Sequence[float], y_prob: Sequence[float],
                     hours_to_onset: Sequence[Optional[float]],
                     bins=(2.0, 4.0, 6.0, 12.0), n_boot: int = 1000, seed: int = 1002) -> dict:
    """Stratifies POSITIVE-timepoint predictions by true distance-to-onset, compares
    each against ALL negative-timepoint predictions (standard lead-time-sweep setup,
    matches SepsisCalc/SepsisLab framing referenced in PROJECT_CONTEXT.md Section 7).
    `hours_to_onset` is None for every negative-admission timepoint (see dataset.py) --
    those are always included as the negative class in each bin's comparison.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    hours = np.array([h if h is not None else np.inf for h in hours_to_onset], dtype=np.float64)

    is_neg = y_true == 0
    out = {}
    edges = [0.0] + list(bins)
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin_pos = (y_true == 1) & (hours > lo) & (hours <= hi)
        mask = in_bin_pos | is_neg
        if in_bin_pos.sum() < 2:
            out[f"{lo}-{hi}h"] = None
            continue
        out[f"{lo}-{hi}h"] = compute_metrics(y_true[mask], y_prob[mask], n_boot=n_boot, seed=seed).to_dict()
    return out


def evaluate_both_protocols(y_true, y_prob, used_only_pre_suspicion: Sequence[bool],
                             hours_to_onset: Optional[Sequence[Optional[float]]] = None,
                             n_boot: int = 1000, seed: int = 1002) -> dict:
    """PROJECT_CONTEXT.md Section 4 + rule #7: every reported result needs BOTH the
    standard protocol (all predictions) and the strict pre-suspicion protocol (only
    predictions built entirely from data available before suspicion_time). Never report
    only the standard number.

    NOTE: this function only *stratifies and scores* -- computing
    `used_only_pre_suspicion` per sample happens in dataset.py at data-loading time
    (using suspicion_time from label_sepsis3.py), not here.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    used_only_pre_suspicion = np.asarray(used_only_pre_suspicion, dtype=bool)

    result = {
        "standard": compute_metrics(y_true, y_prob, n_boot=n_boot, seed=seed).to_dict(),
    }
    if used_only_pre_suspicion.sum() < 2 or len(np.unique(y_true[used_only_pre_suspicion])) < 2:
        result["strict_pre_suspicion"] = None
        result["strict_pre_suspicion_note"] = (
            "too few eligible samples (need >=2 with both classes present) -- "
            "check suspicion_time coverage"
        )
    else:
        result["strict_pre_suspicion"] = compute_metrics(
            y_true[used_only_pre_suspicion], y_prob[used_only_pre_suspicion],
            n_boot=n_boot, seed=seed,
        ).to_dict()

    if hours_to_onset is not None:
        result["lead_time_sweep_standard"] = lead_time_sweep(
            y_true, y_prob, hours_to_onset, n_boot=n_boot, seed=seed
        )
    return result