"""Guideline adjudicator agent.

Ref: Sec. III-D ("Guideline adjudicator (GA) maps the patient condition to the
guideline recommendations and provides C(p) ... It explains the precise statement
that supports each preferred action").

The adjudicator is the only agent that reads the guideline, and it reads it
before any score exists. The output therefore cannot depend on the estimator.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from pathanchor.agents.contracts import AdjudicationResult
from pathanchor.cohort.schema import PatientRecord
from pathanchor.guideline.feasibility import (
    FeasibilityLayer,
    FeasibilityResult,
    branch_disjointness,
    discordance_rate,
)
from pathanchor.guideline.statements import GuidelineTable, StatementRecord


@dataclass(slots=True)
class GuidelineAdjudicator:
    """Maps a patient state to the feasible treatment set."""

    layer: FeasibilityLayer

    @classmethod
    def from_table(cls, table: GuidelineTable) -> GuidelineAdjudicator:
        return cls(layer=FeasibilityLayer(table))

    def adjudicate(self, record: PatientRecord) -> AdjudicationResult:
        result = self.layer.resolve(record)
        return AdjudicationResult(patient_id=record.patient_id, feasibility=result)

    def adjudicate_many(self, records: Sequence[PatientRecord]) -> list[AdjudicationResult]:
        return [self.adjudicate(record) for record in records]

    def supporting_statements(self, option: str, result: FeasibilityResult) -> tuple[str, ...]:
        return result.statement_ids_for(option)


def feasibility_from_results(results: Sequence[AdjudicationResult]) -> list[FeasibilityResult]:
    return [item.feasibility for item in results]


def option_frequency(results: Sequence[FeasibilityResult]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        for option in result.options():
            counts[option] = counts.get(option, 0) + 1
    return dict(sorted(counts.items()))


def contraindication_frequency(results: Sequence[FeasibilityResult]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        for option in result.contraindicated:
            counts[option.value] = counts.get(option.value, 0) + 1
    return dict(sorted(counts.items()))


def audit_discordance(
    chosen: Sequence[str],
    results: Sequence[FeasibilityResult],
) -> dict[str, float]:
    """Guideline discordance, reported as a feasibility check rather than a
    finding: the projection makes the value zero by construction."""
    return {
        "discordance": discordance_rate(chosen, results),
        "n": float(len(results)),
        "feasible_options_per_patient": (
            sum(len(item.feasible) for item in results) / len(results) if results else 0.0
        ),
    }


def branch_report(
    records: Sequence[PatientRecord],
    results: Sequence[FeasibilityResult],
) -> Mapping[str, object]:
    return branch_disjointness(results, records)


def statement_table_summary(table: GuidelineTable) -> Mapping[str, object]:
    strengths: dict[str, int] = {}
    for record in table.records:
        strengths[record.strength.value] = strengths.get(record.strength.value, 0) + 1
    return {
        "guideline": table.guideline,
        "statements": len(table.records),
        "strengths": dict(sorted(strengths.items())),
        "actions": [item.value for item in table.actions()],
        "fingerprint": table.fingerprint()[:16],
    }


def predicates_for_option(records: Sequence[StatementRecord], option: str) -> tuple[str, ...]:
    return tuple(item.statement_id for item in records if item.action.value == option)
