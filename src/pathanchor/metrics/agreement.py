"""Agreement statistics for the path audit.

Ref: Sec. III-H ("Path-level auditability is evaluated by comparing the evidence
paths reported by the system with the independent decision of three clinicians ...
The level of agreement is indicated by Cohen's kappa"), Sec. IV-B (93.1% against
62.3% audit agreement).

Kappa is computed over the nominal categories the adjudicators actually used, so
a reader can see whether the agreement comes from a shared majority class or from
the full category set.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from pathanchor.metrics.intervals import (
    DEFAULT_LEVEL,
    DEFAULT_RESAMPLES,
    BootstrapResult,
    bootstrap_interval,
)
from pathanchor.utils.types import IntArray


@dataclass(frozen=True, slots=True)
class AgreementSummary:
    observed: float
    expected: float
    kappa: float
    n: int
    categories: int

    def as_dict(self) -> dict[str, float]:
        return {
            "observed_agreement": self.observed,
            "expected_agreement": self.expected,
            "kappa": self.kappa,
            "n": float(self.n),
            "categories": float(self.categories),
        }


def confusion_nominal(
    first: Sequence[str],
    second: Sequence[str],
    categories: Sequence[str],
) -> np.ndarray:
    index = {name: slot for slot, name in enumerate(categories)}
    matrix = np.zeros((len(categories), len(categories)), dtype=np.float64)
    for left, right in zip(first, second, strict=True):
        if left not in index or right not in index:
            continue
        matrix[index[left], index[right]] += 1.0
    return matrix


def cohen_kappa_from_matrix(matrix: np.ndarray) -> AgreementSummary:
    total = float(np.sum(matrix))
    if total == 0.0:
        return AgreementSummary(0.0, 0.0, 0.0, 0, int(matrix.shape[0]))
    observed = float(np.trace(matrix) / total)
    row = np.sum(matrix, axis=1) / total
    column = np.sum(matrix, axis=0) / total
    expected = float(np.sum(row * column))
    if abs(1.0 - expected) < 1e-12:
        kappa = 1.0 if observed >= 1.0 else 0.0
    else:
        kappa = (observed - expected) / (1.0 - expected)
    return AgreementSummary(
        observed=observed,
        expected=expected,
        kappa=float(kappa),
        n=int(total),
        categories=int(matrix.shape[0]),
    )


def cohen_kappa(
    first: Sequence[str],
    second: Sequence[str],
    categories: Sequence[str] | None = None,
) -> AgreementSummary:
    if len(first) != len(second):
        raise ValueError("the two rating sequences disagree in length")
    names = list(categories) if categories is not None else sorted(set(first) | set(second))
    return cohen_kappa_from_matrix(confusion_nominal(first, second, names))


def observed_agreement(first: Sequence[str], second: Sequence[str]) -> float:
    if not first:
        return 0.0
    hits = sum(1 for left, right in zip(first, second, strict=True) if left == right)
    return hits / len(first)


def chance_agreement(first: Sequence[str], second: Sequence[str]) -> float:
    names = sorted(set(first) | set(second))
    index = {name: slot for slot, name in enumerate(names)}
    left = np.zeros(len(names), dtype=np.float64)
    right = np.zeros(len(names), dtype=np.float64)
    for item in first:
        left[index[item]] += 1.0
    for item in second:
        right[index[item]] += 1.0
    if first:
        left /= float(len(first))
    if second:
        right /= float(len(second))
    return float(np.sum(left * right))


def kappa_from_definition(first: Sequence[str], second: Sequence[str]) -> float:
    """Cohen's kappa written directly from its definition.

    The verification layer calls this rather than the matrix form, so a defect in
    the contingency-table build cannot hide behind a matching implementation.
    """
    observed = observed_agreement(first, second)
    expected = chance_agreement(first, second)
    if abs(1.0 - expected) < 1e-12:
        return 1.0 if observed >= 1.0 else 0.0
    return (observed - expected) / (1.0 - expected)


def kappa_interval(
    first: Sequence[str],
    second: Sequence[str],
    categories: Sequence[str],
    resamples: int = DEFAULT_RESAMPLES,
    level: float = DEFAULT_LEVEL,
    seed: int = 20260101,
) -> BootstrapResult:
    labels = np.ones(len(first), dtype=np.int64)
    first_array = np.asarray(first)
    second_array = np.asarray(second)

    def statistic(indices: IntArray) -> float:
        return cohen_kappa(
            [str(item) for item in first_array[indices].tolist()],
            [str(item) for item in second_array[indices].tolist()],
            categories,
        ).kappa

    return bootstrap_interval(labels, statistic, resamples=resamples, level=level, seed=seed)


def category_share(ratings: Sequence[str]) -> Mapping[str, float]:
    if not ratings:
        return {}
    counts: dict[str, int] = {}
    for item in ratings:
        counts[item] = counts.get(item, 0) + 1
    return {name: count / len(ratings) for name, count in sorted(counts.items())}


def audit_replay_agreement(
    reported: Sequence[Sequence[str]],
    adjudicated: Sequence[Sequence[str]],
) -> AgreementSummary:
    """Agreement between the typed routes the system reported and those the
    adjudicators rebuilt from the graph.

    A returned route is the concatenation of its relation names, so two routes
    agree when every relation matches in order.
    """
    first = ["|".join(item) for item in reported]
    second = ["|".join(item) for item in adjudicated]
    return cohen_kappa(first, second)
