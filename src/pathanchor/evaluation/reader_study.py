"""Three-arm reader study.

Ref: Sec. IV-G (24 clinicians, 120 vignettes, three conditions with a three-week
washout; treatment-selection accuracy 68.3% / 73.5% / 79.7%; over-reliance errors
14.2% and 6.7%; appropriate reliance 0.54 and 0.71; mean decision time 52 s and
74 s), Sec. IV-F (the fourteen-patient expert case review).

The module computes the reported contrasts from per-vignette records rather than
storing the published percentages, so a run with different records reports its own
numbers and the manuscript's stay in the claim map.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from pathanchor.metrics.intervals import (
    DEFAULT_LEVEL,
    DEFAULT_RESAMPLES,
    BootstrapResult,
    bootstrap_difference,
    bootstrap_interval,
)
from pathanchor.utils.types import BoolArray, FloatArray, IntArray


@dataclass(frozen=True, slots=True)
class VignetteOutcome:
    """One clinician's decision on one vignette under one condition."""

    clinician_id: str
    vignette_id: str
    condition: str
    correct: bool
    accepted_ai: bool
    ai_was_correct: bool
    seconds: float

    @property
    def over_reliance(self) -> bool:
        return self.accepted_ai and not self.ai_was_correct

    @property
    def appropriate_reliance(self) -> bool:
        return self.accepted_ai and self.ai_was_correct


CONDITIONS: tuple[str, ...] = ("unaided", "score_only_ai", "pathanchor")


def accuracy_by_condition(
    outcomes: Sequence[VignetteOutcome],
) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for item in outcomes:
        grouped.setdefault(item.condition, []).append(1.0 if item.correct else 0.0)
    return {
        condition: float(np.mean(values)) if values else float("nan")
        for condition, values in sorted(grouped.items())
    }


def over_reliance_rate(
    outcomes: Sequence[VignetteOutcome],
    condition: str,
) -> float:
    selected = [item for item in outcomes if item.condition == condition]
    if not selected:
        return float("nan")
    return float(np.mean([1.0 if item.over_reliance else 0.0 for item in selected]))


def appropriate_reliance_score(
    outcomes: Sequence[VignetteOutcome],
    condition: str,
) -> float:
    selected = [item for item in outcomes if item.condition == condition]
    if not selected:
        return float("nan")
    return float(np.mean([1.0 if item.appropriate_reliance else 0.0 for item in selected]))


def mean_decision_seconds(
    outcomes: Sequence[VignetteOutcome],
    condition: str,
) -> float:
    selected = [item.seconds for item in outcomes if item.condition == condition]
    if not selected:
        return float("nan")
    return float(np.mean(selected))


def decision_overhead(unaided: float, assisted: float) -> float:
    """Relative time overhead of the assisted condition."""
    if unaided <= 0.0:
        return float("nan")
    return (assisted - unaided) / unaided


def _paired_labels(outcomes: Sequence[VignetteOutcome]) -> IntArray:
    keys = sorted({(item.clinician_id, item.vignette_id) for item in outcomes})
    return np.arange(len(keys), dtype=np.int64)


def _paired_correct(
    outcomes: Sequence[VignetteOutcome],
    condition: str,
) -> tuple[list[tuple[str, str]], BoolArray]:
    keys = sorted({(item.clinician_id, item.vignette_id) for item in outcomes})
    index = {key: slot for slot, key in enumerate(keys)}
    values = np.zeros(len(keys), dtype=bool)
    seen = np.zeros(len(keys), dtype=bool)
    for item in outcomes:
        if item.condition != condition:
            continue
        slot = index[(item.clinician_id, item.vignette_id)]
        values[slot] = item.correct
        seen[slot] = True
    if not bool(np.all(seen)):
        missing = int(np.sum(~seen))
        raise ValueError(f"condition '{condition}' is missing {missing} clinician-vignette cells")
    return (keys, values)


@dataclass(frozen=True, slots=True)
class ReaderContrast:
    first: str
    second: str
    difference: float
    interval_low: float
    interval_high: float
    resamples: int

    def as_dict(self) -> Mapping[str, object]:
        return {
            "contrast": f"{self.first} vs {self.second}",
            "difference_points": self.difference * 100.0,
            "low_points": self.interval_low * 100.0,
            "high_points": self.interval_high * 100.0,
            "resamples": self.resamples,
        }


def paired_accuracy_contrast(
    outcomes: Sequence[VignetteOutcome],
    first: str,
    second: str,
    resamples: int = DEFAULT_RESAMPLES,
    level: float = DEFAULT_LEVEL,
    seed: int = 20260101,
) -> ReaderContrast:
    """Paired bootstrap interval for the accuracy difference of two conditions."""
    keys, first_correct = _paired_correct(outcomes, first)
    _, second_correct = _paired_correct(outcomes, second)
    labels = np.ones(len(keys), dtype=np.int64)

    def statistic_first(indices: IntArray) -> float:
        return float(np.mean(first_correct[indices]))

    def statistic_second(indices: IntArray) -> float:
        return float(np.mean(second_correct[indices]))

    result: BootstrapResult = bootstrap_difference(
        labels, statistic_first, statistic_second, resamples=resamples, level=level, seed=seed
    )
    return ReaderContrast(
        first=first,
        second=second,
        difference=result.value,
        interval_low=result.interval.low,
        interval_high=result.interval.high,
        resamples=resamples,
    )


def condition_summary(outcomes: Sequence[VignetteOutcome]) -> dict[str, Mapping[str, float]]:
    accuracy = accuracy_by_condition(outcomes)
    summary: dict[str, Mapping[str, float]] = {}
    for condition in sorted({item.condition for item in outcomes}):
        summary[condition] = {
            "accuracy": accuracy.get(condition, float("nan")),
            "over_reliance": over_reliance_rate(outcomes, condition),
            "appropriate_reliance": appropriate_reliance_score(outcomes, condition),
            "seconds": mean_decision_seconds(outcomes, condition),
        }
    return summary


def accuracy_interval(
    outcomes: Sequence[VignetteOutcome],
    condition: str,
    resamples: int = DEFAULT_RESAMPLES,
    level: float = DEFAULT_LEVEL,
    seed: int = 20260101,
) -> BootstrapResult:
    keys, correct = _paired_correct(outcomes, condition)
    labels = np.ones(len(keys), dtype=np.int64)

    def statistic(indices: IntArray) -> float:
        return float(np.mean(correct[indices]))

    return bootstrap_interval(labels, statistic, resamples=resamples, level=level, seed=seed)


@dataclass(frozen=True, slots=True)
class CaseAdjudication:
    """One expert's verdict on one of the reviewed test-split cases."""

    case_id: str
    defensible: bool
    path_confirms: bool
    note: str = ""


def case_review_summary(
    adjudications: Sequence[CaseAdjudication],
) -> dict[str, object]:
    if not adjudications:
        return {"cases": 0, "defensible": 0, "path_confirms": 0, "divergences": []}
    return {
        "cases": len(adjudications),
        "defensible": sum(1 for item in adjudications if item.defensible),
        "path_confirms": sum(1 for item in adjudications if item.path_confirms),
        "divergences": [
            {"case_id": item.case_id, "note": item.note}
            for item in adjudications
            if not (item.defensible and item.path_confirms)
        ],
    }


def enrichment_plan() -> Mapping[str, int]:
    """The structured sample of Sec. IV-F: fourteen cases with a stated mix."""
    return {
        "near the abstention boundary": 4,
        "BRAF-wild-type branch": 4,
        "beyond one prior line": 3,
        "without molecular profiling": 1,
    }


def time_overhead_summary(outcomes: Sequence[VignetteOutcome]) -> dict[str, float]:
    unaided = mean_decision_seconds(outcomes, "unaided")
    assisted = mean_decision_seconds(outcomes, "pathanchor")
    return {
        "unaided_seconds": unaided,
        "assisted_seconds": assisted,
        "relative_overhead": decision_overhead(unaided, assisted),
    }


def reliance_gap(
    outcomes: Sequence[VignetteOutcome],
    first: str,
    second: str,
) -> float:
    return over_reliance_rate(outcomes, first) - over_reliance_rate(outcomes, second)


def long_form_records(
    outcomes: Sequence[VignetteOutcome],
) -> FloatArray:
    """Stack the per-vignette outcomes into a numeric matrix for reporting."""
    return np.asarray(
        [
            [
                1.0 if item.correct else 0.0,
                1.0 if item.accepted_ai else 0.0,
                1.0 if item.ai_was_correct else 0.0,
                float(item.seconds),
            ]
            for item in outcomes
        ],
        dtype=np.float64,
    )
