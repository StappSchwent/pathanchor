"""Calibration statistics.

Ref: Sec. III-H ("expected calibration error (ECE) ... using the 15 equal-width
bin method and Brier score"), Sec. IV-A (0.031 against 0.058).

The bin count is the manuscript's. Bins are equal-width on [0, 1] and an empty bin
contributes nothing, which matters because a well-calibrated model on a small
split leaves several bins unpopulated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pathanchor.metrics.intervals import (
    DEFAULT_LEVEL,
    DEFAULT_RESAMPLES,
    BootstrapResult,
    bootstrap_interval,
)
from pathanchor.utils.types import FloatArray, IntArray
from pathanchor.utils.validation import require_binary, require_probabilities

ECE_BINS: int = 15


@dataclass(frozen=True, slots=True)
class ReliabilityCurve:
    bin_lower: FloatArray
    bin_upper: FloatArray
    confidence: FloatArray
    accuracy: FloatArray
    count: IntArray

    def populated(self) -> int:
        return int(np.sum(self.count > 0))


@dataclass(frozen=True, slots=True)
class CalibrationSummary:
    brier: float
    ece: float
    bins: int
    populated_bins: int
    max_bin_gap: float


def brier_score(labels: IntArray, probabilities: FloatArray) -> float:
    require_binary(labels, "brier labels")
    require_probabilities(probabilities, "brier probabilities")
    if labels.shape[0] != probabilities.shape[0]:
        raise ValueError("brier labels and probabilities disagree in length")
    return float(np.mean((probabilities - labels.astype(np.float64)) ** 2))


def reliability_curve(
    labels: IntArray,
    probabilities: FloatArray,
    bins: int = ECE_BINS,
) -> ReliabilityCurve:
    require_binary(labels, "reliability labels")
    require_probabilities(probabilities, "reliability probabilities")
    edges = np.linspace(0.0, 1.0, bins + 1)
    slot = np.clip(np.searchsorted(edges, probabilities, side="right") - 1, 0, bins - 1)
    count = np.zeros(bins, dtype=np.int64)
    confidence = np.zeros(bins, dtype=np.float64)
    accuracy = np.zeros(bins, dtype=np.float64)
    for index in range(bins):
        selected = slot == index
        count[index] = int(np.sum(selected))
        if count[index] == 0:
            continue
        confidence[index] = float(np.mean(probabilities[selected]))
        accuracy[index] = float(np.mean(labels[selected]))
    return ReliabilityCurve(edges[:-1], edges[1:], confidence, accuracy, count)


def expected_calibration_error(
    labels: IntArray,
    probabilities: FloatArray,
    bins: int = ECE_BINS,
) -> float:
    curve = reliability_curve(labels, probabilities, bins)
    total = int(np.sum(curve.count))
    if total == 0:
        return 0.0
    weights = curve.count.astype(np.float64) / float(total)
    gaps = np.abs(curve.accuracy - curve.confidence)
    return float(np.sum(weights * gaps))


def max_bin_gap(labels: IntArray, probabilities: FloatArray, bins: int = ECE_BINS) -> float:
    curve = reliability_curve(labels, probabilities, bins)
    gaps = np.abs(curve.accuracy - curve.confidence)
    populated = curve.count > 0
    if not bool(np.any(populated)):
        return 0.0
    return float(np.max(gaps[populated]))


def calibration_summary(
    labels: IntArray,
    probabilities: FloatArray,
    bins: int = ECE_BINS,
) -> CalibrationSummary:
    curve = reliability_curve(labels, probabilities, bins)
    return CalibrationSummary(
        brier=brier_score(labels, probabilities),
        ece=expected_calibration_error(labels, probabilities, bins),
        bins=bins,
        populated_bins=curve.populated(),
        max_bin_gap=max_bin_gap(labels, probabilities, bins),
    )


def brier_interval(
    labels: IntArray,
    probabilities: FloatArray,
    resamples: int = DEFAULT_RESAMPLES,
    level: float = DEFAULT_LEVEL,
    seed: int = 20260101,
) -> BootstrapResult:
    def statistic(indices: IntArray) -> float:
        return brier_score(labels[indices], probabilities[indices])

    return bootstrap_interval(labels, statistic, resamples=resamples, level=level, seed=seed)


def ece_interval(
    labels: IntArray,
    probabilities: FloatArray,
    bins: int = ECE_BINS,
    resamples: int = DEFAULT_RESAMPLES,
    level: float = DEFAULT_LEVEL,
    seed: int = 20260101,
) -> BootstrapResult:
    def statistic(indices: IntArray) -> float:
        return expected_calibration_error(labels[indices], probabilities[indices], bins)

    return bootstrap_interval(labels, statistic, resamples=resamples, level=level, seed=seed)


def bin_membership(probabilities: FloatArray, bins: int = ECE_BINS) -> IntArray:
    """Bin index per probability, exposed so the assignment can be recomputed."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    membership: IntArray = np.clip(
        np.searchsorted(edges, probabilities, side="right") - 1, 0, bins - 1
    ).astype(np.int64)
    return membership


def hand_computed_ece(
    labels: IntArray,
    probabilities: FloatArray,
    bins: int = ECE_BINS,
) -> float:
    """The manuscript's definition written out longhand.

    The verification layer uses this form so the vectorised implementation above
    has a reference that shares no code with it.
    """
    membership = bin_membership(probabilities, bins)
    total = float(labels.shape[0])
    accumulated = 0.0
    for index in range(bins):
        selected = membership == index
        n_bin = float(np.sum(selected))
        if n_bin == 0.0:
            continue
        mean_probability = float(np.mean(probabilities[selected]))
        observed = float(np.mean(labels[selected]))
        accumulated += (n_bin / total) * abs(observed - mean_probability)
    return accumulated
