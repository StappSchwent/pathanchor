"""Combined metric bundles and multiplicity control.

Ref: Sec. III-H (the reported quantity set), Sec. IV-D (six prespecified subgroup
families whose interaction tests form one multiplicity family).

The step-down adjustment is the Holm form. The correct implementation carries the
running *maximum* of the scaled p-values in sorted order and then clamps at one;
taking the running minimum instead would report every member of the family at the
family-wise Bonferroni level.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from pathanchor.metrics.calibration import (
    CalibrationSummary,
    brier_score,
    calibration_summary,
    expected_calibration_error,
)
from pathanchor.metrics.discrimination import auprc, auroc
from pathanchor.metrics.intervals import (
    DEFAULT_LEVEL,
    DEFAULT_RESAMPLES,
    BootstrapResult,
    bootstrap_interval,
    normal_interval,
)
from pathanchor.utils.types import FloatArray, IntArray, Interval


@dataclass(frozen=True, slots=True)
class MethodRow:
    """One row of the comparison table."""

    family: str
    method: str
    auroc: float
    auroc_interval: Interval
    auprc: float
    brier: float
    ece: float
    discordance: float
    framework: bool = False

    def as_dict(self) -> Mapping[str, object]:
        return {
            "family": self.family,
            "method": self.method,
            "auroc": self.auroc,
            "auroc_low": self.auroc_interval.low,
            "auroc_high": self.auroc_interval.high,
            "auprc": self.auprc,
            "brier": self.brier,
            "ece": self.ece,
            "discordance": self.discordance,
            "framework": self.framework,
        }


@dataclass(slots=True)
class MetricBundle:
    labels: IntArray
    scores: FloatArray
    cache: dict[str, object] = field(default_factory=dict)

    def discrimination(self) -> dict[str, float]:
        return {
            "auroc": auroc(self.labels, self.scores),
            "auprc": auprc(self.labels, self.scores),
        }

    def calibration(self) -> CalibrationSummary:
        return calibration_summary(self.labels, self.scores)

    def summary(self) -> dict[str, float]:
        discrimination = self.discrimination()
        calibration = self.calibration()
        return {
            "auroc": discrimination["auroc"],
            "auprc": discrimination["auprc"],
            "brier": calibration.brier,
            "ece": calibration.ece,
            "populated_bins": float(calibration.populated_bins),
        }


def discrimination_bundle(labels: IntArray, scores: FloatArray) -> dict[str, object]:
    trials = np.linspace(0.10, 0.90, 17)
    return {
        "auroc": auroc(labels, scores),
        "auprc": auprc(labels, scores),
        "brier": brier_score(labels, scores),
        "ece": expected_calibration_error(labels, scores),
        "thresholds": trials.tolist(),
    }


def bootstrap_pair(
    labels: IntArray,
    scores: FloatArray,
    resamples: int = DEFAULT_RESAMPLES,
    level: float = DEFAULT_LEVEL,
    seed: int = 20260101,
) -> tuple[BootstrapResult, BootstrapResult]:

    def auroc_statistic(indices: IntArray) -> float:
        return auroc(labels[indices], scores[indices])

    def auprc_statistic(indices: IntArray) -> float:
        return auprc(labels[indices], scores[indices])

    return (
        bootstrap_interval(labels, auroc_statistic, resamples=resamples, level=level, seed=seed),
        bootstrap_interval(
            labels, auprc_statistic, resamples=resamples, level=level, seed=seed + 1
        ),
    )


def holm_bonferroni(p_values: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values.

    Adjusted value at rank i is max over j <= i of (m - j + 1) * p_(j), clamped to
    one. The running maximum is what enforces monotonicity; without it the
    adjustment is not a step-down procedure.
    """
    m = len(p_values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda slot: (p_values[slot], slot))
    adjusted = [0.0] * m
    running = 0.0
    for rank, slot in enumerate(order):
        scaled = (m - rank) * float(p_values[slot])
        running = max(running, scaled)
        adjusted[slot] = min(running, 1.0)
    return adjusted


def benjamini_hochberg(p_values: Sequence[float]) -> list[float]:
    """Benjamini-Hochberg adjusted p-values, for comparison with the step-down."""
    m = len(p_values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda slot: (p_values[slot], slot))
    adjusted = [0.0] * m
    running = 1.0
    for rank in range(m - 1, -1, -1):
        slot = order[rank]
        scaled = float(p_values[slot]) * m / (rank + 1)
        running = min(running, scaled)
        adjusted[slot] = min(running, 1.0)
    return adjusted


def family_report(names: Sequence[str], p_values: Sequence[float]) -> list[dict[str, object]]:
    adjusted = holm_bonferroni(p_values)
    return [
        {
            "name": names[index],
            "p_value": float(p_values[index]),
            "holm_adjusted": float(adjusted[index]),
            "significant_at_005": bool(adjusted[index] < 0.05),
        }
        for index in range(len(names))
    ]


def interval_width_table(
    labels: IntArray,
    scores: FloatArray,
    comparisons: Mapping[str, FloatArray],
) -> list[dict[str, object]]:
    from pathanchor.metrics.discrimination import delong_test

    rows: list[dict[str, object]] = []
    for name, other in comparisons.items():
        result = delong_test(labels, scores, other)
        rows.append(
            {
                "against": name,
                "difference": result.difference,
                "low": result.interval.low,
                "high": result.interval.high,
                "p_value": result.p_value,
            }
        )
    return rows


def safe_interval(value: float, standard_error: float, level: float = DEFAULT_LEVEL) -> Interval:
    if not np.isfinite(standard_error):
        return Interval(value, value, level)
    return normal_interval(value, standard_error, level)
