"""Discrimination statistics.

Ref: Sec. III-H (AUROC, AUPRC, DeLong, 2,000 stratified bootstrap resamples),
Sec. IV-A (the reported AUROC gap and its DeLong p-value).

The AUROC is computed from average ranks rather than by sweeping thresholds, so
tied scores contribute half a concordant pair. That matters here because several
comparators binarise a continuous measurement and produce large tie blocks; the
ordinal-rank form would report a different number for the same predictions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pathanchor.metrics.intervals import (
    DEFAULT_LEVEL,
    DEFAULT_RESAMPLES,
    BootstrapResult,
    bootstrap_interval,
    normal_interval,
)
from pathanchor.utils.numerics import average_ranks
from pathanchor.utils.types import FloatArray, IntArray, Interval
from pathanchor.utils.validation import require_binary


@dataclass(frozen=True, slots=True)
class DeLongResult:
    """A paired DeLong contrast between two scoring rules."""

    auc_a: float
    auc_b: float
    difference: float
    standard_error: float
    z: float
    p_value: float
    interval: Interval

    def summary(self) -> str:
        return (
            f"delta={self.difference:+.4f} se={self.standard_error:.4f} "
            f"z={self.z:+.3f} p={self.p_value:.4f}"
        )


def auroc(labels: IntArray, scores: FloatArray) -> float:
    """Area under the ROC curve, computed from average ranks."""
    require_binary(labels, "auroc labels")
    if labels.shape[0] != scores.shape[0]:
        raise ValueError("auroc labels and scores disagree in length")
    positives = int(np.sum(labels == 1))
    negatives = int(labels.shape[0] - positives)
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = average_ranks(scores)
    rank_sum = float(np.sum(ranks[labels == 1]))
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def auprc(labels: IntArray, scores: FloatArray) -> float:
    """Average precision, the step-wise interpolated area under precision-recall."""
    require_binary(labels, "auprc labels")
    positives = int(np.sum(labels == 1))
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    ordered = labels[order]
    tp = np.cumsum(ordered)
    fp = np.cumsum(1 - ordered)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / positives
    increments = np.diff(np.concatenate([[0.0], recall]))
    return float(np.sum(precision * increments))


def sensitivity_specificity(
    labels: IntArray,
    scores: FloatArray,
    threshold: float,
) -> tuple[float, float]:
    predicted = scores >= threshold
    positives = labels == 1
    negatives = labels == 0
    sensitivity = float(np.mean(predicted[positives])) if positives.any() else float("nan")
    specificity = float(np.mean(~predicted[negatives])) if negatives.any() else float("nan")
    return (sensitivity, specificity)


def _midrank(values: FloatArray) -> FloatArray:
    return average_ranks(values)


def delong_components(
    labels: IntArray,
    score_matrix: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """AUCs and their covariance matrix by the fast DeLong construction."""
    require_binary(labels, "delong labels")
    if score_matrix.ndim == 1:
        score_matrix = score_matrix.reshape(-1, 1)
    if score_matrix.shape[0] != labels.shape[0]:
        raise ValueError("delong scores and labels disagree in length")
    positive = labels == 1
    negative = labels == 0
    n_pos = int(np.sum(positive))
    n_neg = int(np.sum(negative))
    if n_pos == 0 or n_neg == 0:
        raise ValueError("DeLong needs at least one positive and one negative")
    k = score_matrix.shape[1]
    aucs = np.empty(k, dtype=np.float64)
    v10 = np.zeros((n_pos, k), dtype=np.float64)
    v01 = np.zeros((n_neg, k), dtype=np.float64)
    for column in range(k):
        column_scores = score_matrix[:, column]
        total_rank = _midrank(column_scores)
        positive_rank = _midrank(column_scores[positive])
        negative_rank = _midrank(column_scores[negative])
        aucs[column] = (float(np.sum(total_rank[positive])) - n_pos * (n_pos + 1) / 2.0) / (
            n_pos * n_neg
        )
        v10[:, column] = (total_rank[positive] - positive_rank) / n_neg
        v01[:, column] = 1.0 - (total_rank[negative] - negative_rank) / n_pos
    s10 = np.cov(v10, rowvar=False, ddof=1) / n_pos
    s01 = np.cov(v01, rowvar=False, ddof=1) / n_neg
    covariance = np.atleast_2d(s10 + s01)
    return (aucs, covariance)


def delong_test(
    labels: IntArray,
    scores_a: FloatArray,
    scores_b: FloatArray,
    level: float = DEFAULT_LEVEL,
) -> DeLongResult:
    """Two-sided DeLong test of two correlated ROC curves."""
    from scipy import stats

    matrix = np.column_stack([scores_a, scores_b])
    aucs, covariance = delong_components(labels, matrix)
    variance = float(covariance[0, 0] + covariance[1, 1] - 2.0 * covariance[0, 1])
    variance = max(variance, 0.0)
    standard_error = float(np.sqrt(variance))
    difference = float(aucs[0] - aucs[1])
    if standard_error == 0.0:
        return DeLongResult(
            aucs[0], aucs[1], difference, 0.0, 0.0, 1.0, normal_interval(difference, 0.0, level)
        )
    z = difference / standard_error
    p_value = float(2.0 * (1.0 - stats.norm.cdf(abs(z))))
    return DeLongResult(
        auc_a=float(aucs[0]),
        auc_b=float(aucs[1]),
        difference=difference,
        standard_error=standard_error,
        z=float(z),
        p_value=p_value,
        interval=normal_interval(difference, standard_error, level),
    )


def auroc_interval(
    labels: IntArray,
    scores: FloatArray,
    resamples: int = DEFAULT_RESAMPLES,
    level: float = DEFAULT_LEVEL,
    seed: int = 20260101,
) -> BootstrapResult:
    def statistic(indices: IntArray) -> float:
        return auroc(labels[indices], scores[indices])

    return bootstrap_interval(labels, statistic, resamples=resamples, level=level, seed=seed)


def auprc_interval(
    labels: IntArray,
    scores: FloatArray,
    resamples: int = DEFAULT_RESAMPLES,
    level: float = DEFAULT_LEVEL,
    seed: int = 20260101,
) -> BootstrapResult:
    def statistic(indices: IntArray) -> float:
        return auprc(labels[indices], scores[indices])

    return bootstrap_interval(labels, statistic, resamples=resamples, level=level, seed=seed)


def delong_interval(
    labels: IntArray,
    scores: FloatArray,
    level: float = DEFAULT_LEVEL,
) -> Interval:
    aucs, covariance = delong_components(labels, scores)
    return normal_interval(float(aucs[0]), float(np.sqrt(max(covariance[0, 0], 0.0))), level)


def auroc_rank_identity(labels: IntArray, scores: FloatArray) -> float:
    """The U-statistic read of the same quantity, for cross-checking.

    Independent of the rank-sum algebra above: it counts concordant pairs
    directly, which is how the verification layer recomputes an AUROC without
    reusing this module.
    """
    positives = np.sort(scores[labels == 1])
    negatives = np.sort(scores[labels == 0])
    if positives.shape[0] == 0 or negatives.shape[0] == 0:
        return float("nan")
    greater = np.searchsorted(negatives, positives, side="left")
    equal = np.searchsorted(negatives, positives, side="right") - greater
    concordant = greater.sum() + 0.5 * equal.sum()
    return float(concordant / (positives.shape[0] * negatives.shape[0]))
