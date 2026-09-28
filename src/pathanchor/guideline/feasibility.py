"""Construction of the guideline-feasible set C(p).

Ref: Sec. III-A (i) and Eq. (1); Sec. III-D (guideline adjudicator);
Sec. IV-B (removing the projection raises discordance from zero to 18.9%).

C(p) is derived from the guideline table alone: no term in it depends on the
patient's outcome or on any learned parameter, which is what makes guideline
discordance impossible by construction rather than discouraged by a penalty.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from pathanchor.cohort.schema import PatientRecord, TreatmentClass
from pathanchor.guideline.statements import (
    GuidelineTable,
    Predicate,
    StatementIndex,
    StatementRecord,
    Strength,
    prior_lines_token,
)

_STRENGTH_RANK: Mapping[Strength, int] = {
    Strength.PREFERRED: 0,
    Strength.RECOMMENDED: 1,
    Strength.OPTIONAL: 2,
    Strength.CONTRAINDICATED: 3,
}


@dataclass(frozen=True, slots=True)
class FeasibilityResult:
    patient_id: str
    guideline: str
    fingerprint: str
    feasible: tuple[TreatmentClass, ...]
    contraindicated: tuple[TreatmentClass, ...]
    supporting: Mapping[TreatmentClass, tuple[str, ...]]
    priority: tuple[TreatmentClass, ...]

    def options(self) -> tuple[str, ...]:
        return tuple(item.value for item in self.feasible)

    def is_feasible(self, option: str) -> bool:
        return option in set(self.options())

    def statement_ids_for(self, option: str) -> tuple[str, ...]:
        for candidate, identifiers in self.supporting.items():
            if candidate.value == option:
                return identifiers
        return ()

    def discordant(self, option: str) -> bool:
        return option not in set(self.options())


@dataclass(slots=True)
class FeasibilityLayer:
    """Applies the statement table to a patient state."""

    table: GuidelineTable
    index: StatementIndex = field(init=False)

    def __post_init__(self) -> None:
        self.index = StatementIndex.build(self.table)

    def _matching(self, record: PatientRecord) -> list[StatementRecord]:
        matched: list[StatementRecord] = []
        matched.extend(self.index.lookup(Predicate.MOLECULAR_CLASS, record.molecular.value))
        for statement in self.index.all_for(Predicate.PRIOR_LINES_AT_MOST):
            if int(prior_lines_token(record.prior_lines)) <= int(statement.value):
                matched.append(statement)
        for statement in self.index.all_for(Predicate.AGE_AT_LEAST):
            if record.age >= int(statement.value):
                matched.append(statement)
        for statement in self.index.all_for(Predicate.HISTOLOGY_AVAILABLE):
            if str(record.has_pathology).lower() == statement.value.lower():
                matched.append(statement)
        for statement in self.index.all_for(Predicate.IMAGING_AVAILABLE):
            if str(record.has_imaging).lower() == statement.value.lower():
                matched.append(statement)
        return matched

    def resolve(self, record: PatientRecord) -> FeasibilityResult:
        matched = self._matching(record)
        opened: dict[TreatmentClass, list[str]] = {}
        blocked: dict[TreatmentClass, list[str]] = {}
        best_rank: dict[TreatmentClass, int] = {}
        for statement in matched:
            if statement.strength is Strength.CONTRAINDICATED:
                blocked.setdefault(statement.action, []).append(statement.statement_id)
                continue
            opened.setdefault(statement.action, []).append(statement.statement_id)
            rank = _STRENGTH_RANK[statement.strength]
            best_rank[statement.action] = min(best_rank.get(statement.action, rank), rank)
        feasible = tuple(
            option for option in TreatmentClass if option in opened and option not in blocked
        )
        contraindicated = tuple(option for option in TreatmentClass if option in blocked)
        priority = tuple(
            sorted(feasible, key=lambda option: (best_rank.get(option, 9), option.value))
        )
        supporting: dict[TreatmentClass, tuple[str, ...]] = {
            option: tuple(opened[option]) for option in feasible
        }
        return FeasibilityResult(
            patient_id=record.patient_id,
            guideline=self.table.guideline,
            fingerprint=self.table.fingerprint(),
            feasible=feasible,
            contraindicated=contraindicated,
            supporting=supporting,
            priority=priority,
        )

    def resolve_many(self, records: Sequence[PatientRecord]) -> list[FeasibilityResult]:
        return [self.resolve(record) for record in records]


def default_feasibility_layer() -> FeasibilityLayer:
    from pathanchor.guideline.statements import default_table

    return FeasibilityLayer(default_table())


def feasible_counts(results: Sequence[FeasibilityResult]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        key = "|".join(result.options())
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def discordance_rate(
    chosen: Sequence[str],
    results: Sequence[FeasibilityResult],
) -> float:
    """Share of patients whose recommendation falls outside C(p)."""
    if len(chosen) != len(results):
        raise ValueError("chosen options and feasibility results disagree in length")
    if not results:
        return 0.0
    outside = sum(
        1 for option, result in zip(chosen, results, strict=True) if result.discordant(option)
    )
    return outside / len(results)


def branch_disjointness(
    results: Sequence[FeasibilityResult],
    records: Sequence[PatientRecord],
) -> dict[str, object]:
    """Check the manuscript's claim that the BRAF branches have disjoint
    feasible sets.

    The claim as printed in Sec. IV-D does not hold literally: both branches
    admit the immune-checkpoint classes. What holds is that the BRAF-directed
    class is admissible in exactly one branch, which is what this returns.
    """
    braf_options: set[str] = set()
    wild_options: set[str] = set()
    for record, result in zip(records, results, strict=True):
        if record.is_braf_v600:
            braf_options.update(result.options())
        else:
            wild_options.update(result.options())
    return {
        "braf_branch": sorted(braf_options),
        "wild_type_branch": sorted(wild_options),
        "shared": sorted(braf_options & wild_options),
        "literal_disjoint": not (braf_options & wild_options),
        "braf_directed_in_one_branch_only": TreatmentClass.BRAF_MEK.value in braf_options
        and TreatmentClass.BRAF_MEK.value not in wild_options,
    }
