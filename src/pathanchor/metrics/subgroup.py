"""Prespecified subgroup analysis and the interaction test.

Ref: Sec. IV-D and Table III (subgroups by molecular class, treatment class, site,
region, sex and age band, with a reported interaction between BRAF status and
method at p = 0.008), Sec. III-H (the subgroup list is prespecified).

The interaction test compares the paired DeLong contrasts of two disjoint strata.
Because the strata share no patient, the variances add, and the contrast itself is
computed inside each stratum rather than on the pooled cohort -- the pooled
contrast is what the treatment-conditional finding says is misleading.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from pathanchor.metrics.discrimination import auroc, delong_components
from pathanchor.utils.types import BoolArray, FloatArray, IntArray

SUBGROUP_DIMENSIONS: tuple[str, ...] = (
    "Molecular",
    "Treatment class",
    "Site",
    "Region",
    "Sex",
    "Age band",
)


@dataclass(frozen=True, slots=True)
class StratumRow:
    dimension: str
    subgroup: str
    n: int
    framework_auroc: float
    baseline_auroc: float
    delta_percentage_points: float
    framework_only: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "dimension": self.dimension,
            "subgroup": self.subgroup,
            "n": self.n,
            "framework_auroc": self.framework_auroc,
            "baseline_auroc": self.baseline_auroc,
            "delta_pp": self.delta_percentage_points,
            "framework_only": self.framework_only,
        }


def stratum_indices(labels: Sequence[str]) -> dict[str, IntArray]:
    grouped: dict[str, list[int]] = {}
    for index, name in enumerate(labels):
        grouped.setdefault(str(name), []).append(index)
    return {name: np.asarray(items, dtype=np.int64) for name, items in sorted(grouped.items())}


def stratum_rows(
    labels: IntArray,
    framework_scores: FloatArray,
    baseline_scores: FloatArray,
    subgroup: Sequence[str],
    dimension: str,
) -> list[StratumRow]:
    rows: list[StratumRow] = []
    groups = stratum_indices(subgroup)
    for name, indices in groups.items():
        stratum_labels = labels[indices]
        if len(set(stratum_labels.tolist())) < 2:
            rows.append(
                StratumRow(
                    dimension,
                    name,
                    int(indices.shape[0]),
                    float("nan"),
                    float("nan"),
                    float("nan"),
                    framework_scores is baseline_scores,
                )
            )
            continue
        framework = auroc(stratum_labels, framework_scores[indices])
        baseline = auroc(stratum_labels, baseline_scores[indices])
        rows.append(
            StratumRow(
                dimension=dimension,
                subgroup=name,
                n=int(indices.shape[0]),
                framework_auroc=float(framework),
                baseline_auroc=float(baseline),
                delta_percentage_points=float((framework - baseline) * 100.0),
                framework_only=False,
            )
        )
    return rows


def paired_delta(
    labels: IntArray,
    framework_scores: FloatArray,
    baseline_scores: FloatArray,
) -> tuple[float, float]:
    """The framework-minus-baseline AUROC gap and its DeLong variance."""
    matrix = np.column_stack([framework_scores, baseline_scores])
    aucs, covariance = delong_components(labels, matrix)
    difference = float(aucs[0] - aucs[1])
    variance = float(covariance[0, 0] + covariance[1, 1] - 2.0 * covariance[0, 1])
    return (difference, max(variance, 0.0))


@dataclass(frozen=True, slots=True)
class InteractionResult:
    dimension: str
    first_stratum: str
    second_stratum: str
    delta_first: float
    delta_second: float
    difference: float
    standard_error: float
    z: float
    p_value: float
    significant_at_010: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "dimension": self.dimension,
            "first": self.first_stratum,
            "second": self.second_stratum,
            "delta_first": self.delta_first,
            "delta_second": self.delta_second,
            "difference": self.difference,
            "standard_error": self.standard_error,
            "z": self.z,
            "p_value": self.p_value,
            "significant_at_010": self.significant_at_010,
        }


def interaction_test(
    labels: IntArray,
    framework_scores: FloatArray,
    baseline_scores: FloatArray,
    first_mask: BoolArray,
    second_mask: BoolArray,
    dimension: str,
    first_name: str,
    second_name: str,
) -> InteractionResult:
    from scipy import stats

    if bool(np.any(first_mask & second_mask)):
        raise ValueError("the two interaction strata must be disjoint")
    delta_first, variance_first = paired_delta(
        labels[first_mask], framework_scores[first_mask], baseline_scores[first_mask]
    )
    delta_second, variance_second = paired_delta(
        labels[second_mask], framework_scores[second_mask], baseline_scores[second_mask]
    )
    variance = variance_first + variance_second
    standard_error = float(np.sqrt(variance))
    difference = float(delta_first - delta_second)
    if standard_error == 0.0:
        return InteractionResult(
            dimension,
            first_name,
            second_name,
            delta_first,
            delta_second,
            difference,
            0.0,
            0.0,
            1.0,
            False,
        )
    z = difference / standard_error
    p_value = float(2.0 * (1.0 - stats.norm.cdf(abs(z))))
    return InteractionResult(
        dimension=dimension,
        first_stratum=first_name,
        second_stratum=second_name,
        delta_first=float(delta_first),
        delta_second=float(delta_second),
        difference=difference,
        standard_error=standard_error,
        z=float(z),
        p_value=p_value,
        significant_at_010=p_value < 0.10,
    )


def subgroup_table(
    labels: IntArray,
    framework_scores: FloatArray,
    baseline_scores: FloatArray,
    dimensions: Mapping[str, Sequence[str]],
) -> list[StratumRow]:
    rows: list[StratumRow] = []
    for dimension, values in dimensions.items():
        rows.extend(stratum_rows(labels, framework_scores, baseline_scores, values, dimension))
    return rows


def largest_delta(rows: Sequence[StratumRow]) -> StratumRow:
    finite = [row for row in rows if np.isfinite(row.delta_percentage_points)]
    if not finite:
        raise ValueError("no stratum carries a finite difference")
    return max(finite, key=lambda row: row.delta_percentage_points)


def smallest_delta(rows: Sequence[StratumRow]) -> StratumRow:
    finite = [row for row in rows if np.isfinite(row.delta_percentage_points)]
    if not finite:
        raise ValueError("no stratum carries a finite difference")
    return min(finite, key=lambda row: row.delta_percentage_points)


def non_inferior_strata(rows: Sequence[StratumRow], margin_pp: float = 0.0) -> int:
    """Count strata where the framework is at or above the margin."""
    return sum(
        1
        for row in rows
        if np.isfinite(row.delta_percentage_points) and row.delta_percentage_points >= margin_pp
    )
