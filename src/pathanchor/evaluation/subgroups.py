"""Prespecified subgroup evaluation.

Ref: Table III and Sec. IV-D, with the subgroup list of Sec. III-H.

The table is built from the framework's own scores and the best comparator's
scores on the same records, so a stratum's difference is paired. The interaction
test joins the two molecular branches, which is the contrast the treatment-
conditional finding rests on.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from pathanchor.cohort.fixture import CohortFixture
from pathanchor.cohort.schema import PatientRecord, Split
from pathanchor.metrics.discrimination import auroc
from pathanchor.metrics.subgroup import (
    InteractionResult,
    StratumRow,
    interaction_test,
    stratum_rows,
)
from pathanchor.utils.types import FloatArray, IntArray


@dataclass(frozen=True, slots=True)
class SubgroupReport:
    rows: tuple[StratumRow, ...]
    interaction: InteractionResult | None
    strata_beaten: int
    strata_total: int

    def as_dicts(self) -> list[dict[str, object]]:
        return [row.as_dict() for row in self.rows]

    def summary(self) -> dict[str, object]:
        return {
            "rows": len(self.rows),
            "strata_beaten": self.strata_beaten,
            "strata_total": self.strata_total,
            "interaction_p": (
                self.interaction.p_value if self.interaction is not None else float("nan")
            ),
        }


def dimension_values(records: Sequence[PatientRecord], dimension: str) -> list[str]:
    mapping = {
        "Molecular": lambda record: record.molecular.value,
        "Treatment class": lambda record: record.treatment.value,
        "Site": lambda record: record.site.value,
        "Region": lambda record: record.region.value,
        "Sex": lambda record: record.sex.value,
        "Age band": lambda record: record.age_band.value,
    }
    if dimension not in mapping:
        raise ValueError(f"unknown subgroup dimension '{dimension}'")
    return [mapping[dimension](record) for record in records]


def subgroup_report(
    records: Sequence[PatientRecord],
    labels: IntArray,
    framework_scores: FloatArray,
    baseline_scores: FloatArray,
    dimensions: Sequence[str] = (
        "Molecular",
        "Treatment class",
        "Site",
        "Region",
        "Sex",
        "Age band",
    ),
) -> SubgroupReport:
    rows: list[StratumRow] = []
    for dimension in dimensions:
        rows.extend(
            stratum_rows(
                labels,
                framework_scores,
                baseline_scores,
                dimension_values(records, dimension),
                dimension,
            )
        )
    interaction = molecular_interaction(records, labels, framework_scores, baseline_scores)
    beaten = sum(
        1
        for row in rows
        if np.isfinite(row.delta_percentage_points) and row.delta_percentage_points >= 0.0
    )
    return SubgroupReport(
        rows=tuple(rows),
        interaction=interaction,
        strata_beaten=beaten,
        strata_total=sum(1 for row in rows if np.isfinite(row.delta_percentage_points)),
    )


def molecular_interaction(
    records: Sequence[PatientRecord],
    labels: IntArray,
    framework_scores: FloatArray,
    baseline_scores: FloatArray,
) -> InteractionResult | None:
    braf = np.asarray([record.is_braf_v600 for record in records], dtype=bool)
    wild = ~braf
    if not bool(np.any(braf)) or not bool(np.any(wild)):
        return None
    return interaction_test(
        labels,
        framework_scores,
        baseline_scores,
        braf,
        wild,
        dimension="Molecular",
        first_name="BRAF V600 mutant",
        second_name="BRAF wild-type",
    )


def split_rows(
    report: SubgroupReport,
    dimension: str,
) -> list[StratumRow]:
    return [row for row in report.rows if row.dimension == dimension]


def site_stability(report: SubgroupReport) -> dict[str, float]:
    rows = split_rows(report, "Site")
    values = [row.framework_auroc for row in rows if np.isfinite(row.framework_auroc)]
    if not values:
        return {"min": float("nan"), "max": float("nan"), "spread": float("nan")}
    return {
        "min": float(min(values)),
        "max": float(max(values)),
        "spread": float(max(values) - min(values)),
    }


def stratum_auroc(
    records: Sequence[PatientRecord],
    labels: IntArray,
    scores: FloatArray,
    dimension: str,
    subgroup: str,
) -> float:
    values = dimension_values(records, dimension)
    mask = np.asarray([value == subgroup for value in values], dtype=bool)
    if int(np.sum(mask)) < 2:
        return float("nan")
    return auroc(labels[mask], scores[mask])


def subgroup_counts(records: Sequence[PatientRecord], dimension: str) -> Mapping[str, int]:
    counts: dict[str, int] = {}
    for value in dimension_values(records, dimension):
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def evaluation_records(fixture: CohortFixture, split: Split) -> tuple[PatientRecord, ...]:
    return fixture.by_split(split)
