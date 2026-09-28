"""Clinical utility: net benefit and decision-curve analysis.

Ref: Sec. III-H ("net gain over critical probabilities ... 'treat-all' and
'treat-none' as the referential strategies, as well as ... a pre-set minimum
clinically relevant difference in the net effect = 0.02 at the value of 0.35"),
Sec. IV-C (0.182 for the framework against 0.104, 0.118, 0.131 and 0.000).

Net benefit is expressed per patient, so the identity
``net_benefit = prevalence - loss`` holds exactly and can be checked without this
module.
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
from pathanchor.utils.types import FloatArray, IntArray, Interval
from pathanchor.utils.validation import require_binary

MINIMUM_CLINICAL_DIFFERENCE: float = 0.02
OPERATING_THRESHOLD: float = 0.35


@dataclass(frozen=True, slots=True)
class NetBenefit:
    threshold: float
    value: float
    true_positives: int
    false_positives: int
    n: int

    def as_dict(self) -> dict[str, float]:
        return {
            "threshold": self.threshold,
            "net_benefit": self.value,
            "true_positives": float(self.true_positives),
            "false_positives": float(self.false_positives),
        }


@dataclass(frozen=True, slots=True)
class DecisionCurve:
    thresholds: FloatArray
    model: FloatArray
    treat_all: FloatArray
    treat_none: FloatArray
    prevalence: float

    def best_strategy(self, threshold: float) -> str:
        index = int(np.argmin(np.abs(self.thresholds - threshold)))
        candidates = {
            "model": float(self.model[index]),
            "treat_all": float(self.treat_all[index]),
            "treat_none": float(self.treat_none[index]),
        }
        return max(candidates, key=lambda name: (candidates[name], name))


def net_benefit(
    labels: IntArray,
    probabilities: FloatArray,
    threshold: float,
) -> float:
    require_binary(labels, "net benefit labels")
    if not 0.0 < threshold < 1.0:
        raise ValueError("the decision threshold must lie strictly inside (0, 1)")
    predicted = probabilities >= threshold
    n = float(labels.shape[0])
    true_positive = float(np.sum(predicted & (labels == 1)))
    false_positive = float(np.sum(predicted & (labels == 0)))
    odds = threshold / (1.0 - threshold)
    return true_positive / n - (false_positive / n) * odds


def net_benefit_detail(
    labels: IntArray,
    probabilities: FloatArray,
    threshold: float,
) -> NetBenefit:
    predicted = probabilities >= threshold
    odds = threshold / (1.0 - threshold)
    true_positive = int(np.sum(predicted & (labels == 1)))
    false_positive = int(np.sum(predicted & (labels == 0)))
    n = int(labels.shape[0])
    value = true_positive / n - (false_positive / n) * odds
    return NetBenefit(threshold, float(value), true_positive, false_positive, n)


def treat_all_net_benefit(prevalence: float, threshold: float) -> float:
    odds = threshold / (1.0 - threshold)
    return prevalence - (1.0 - prevalence) * odds


def treat_none_net_benefit() -> float:
    return 0.0


def standardised_net_benefit(value: float, prevalence: float) -> float:
    if prevalence <= 0.0:
        return 0.0
    return value / prevalence


def decision_curve(
    labels: IntArray,
    probabilities: FloatArray,
    thresholds: FloatArray | None = None,
) -> DecisionCurve:
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)
    prevalence = float(np.mean(labels))
    model = np.asarray(
        [net_benefit(labels, probabilities, float(item)) for item in thresholds], dtype=np.float64
    )
    all_treat = np.asarray(
        [treat_all_net_benefit(prevalence, float(item)) for item in thresholds], dtype=np.float64
    )
    none_treat = np.zeros_like(all_treat)
    return DecisionCurve(thresholds, model, all_treat, none_treat, prevalence)


def net_benefit_interval(
    labels: IntArray,
    probabilities: FloatArray,
    threshold: float = OPERATING_THRESHOLD,
    resamples: int = DEFAULT_RESAMPLES,
    level: float = DEFAULT_LEVEL,
    seed: int = 20260101,
) -> BootstrapResult:
    def statistic(indices: IntArray) -> float:
        return net_benefit(labels[indices], probabilities[indices], threshold)

    return bootstrap_interval(labels, statistic, resamples=resamples, level=level, seed=seed)


def net_benefit_difference_interval(
    labels: IntArray,
    probabilities_a: FloatArray,
    probabilities_b: FloatArray,
    threshold: float = OPERATING_THRESHOLD,
    resamples: int = DEFAULT_RESAMPLES,
    level: float = DEFAULT_LEVEL,
    seed: int = 20260101,
) -> BootstrapResult:
    rng = np.random.default_rng(seed)
    from pathanchor.metrics.intervals import stratified_indices

    def difference(indices: IntArray) -> float:
        return net_benefit(labels[indices], probabilities_a[indices], threshold) - net_benefit(
            labels[indices], probabilities_b[indices], threshold
        )

    observed = difference(np.arange(labels.shape[0], dtype=np.int64))
    draws = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        sample = stratified_indices(labels, rng)
        draws[index] = difference(sample)
    draws = draws[np.isfinite(draws)]
    from pathanchor.metrics.intervals import percentile_interval

    return BootstrapResult(
        value=float(observed),
        interval=percentile_interval(draws, level),
        resamples=resamples,
        standard_error=float(np.std(draws, ddof=1)) if draws.shape[0] > 1 else 0.0,
    )


def exceeds_minimum_difference(value: float, reference: float) -> bool:
    return (value - reference) > MINIMUM_CLINICAL_DIFFERENCE


def threshold_of_maximum_benefit(
    labels: IntArray,
    probabilities: FloatArray,
) -> float:
    curve = decision_curve(labels, probabilities)
    index = int(np.argmax(curve.model))
    return float(curve.thresholds[index])


def benefit_retention_range(
    labels: IntArray,
    probabilities: FloatArray,
    treat_none_crosses: float = 0.99,
) -> Interval:
    """The threshold span over which the model still beats treat-none."""
    curve = decision_curve(labels, probabilities, np.linspace(0.01, treat_none_crosses, 197))
    positive = curve.model > 0.0
    if not bool(np.any(positive)):
        return Interval(0.0, 0.0)
    low = float(curve.thresholds[np.argmax(positive)])
    high = float(curve.thresholds[len(positive) - 1 - int(np.argmax(positive[::-1]))])
    return Interval(low, high)
