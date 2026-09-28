"""Typed records for the parsed recommendation statements.

Ref: Sec. III-B (guideline edges are derived from the recommendation sentences),
Sec. III-D (the guideline adjudicator explains the statement behind each action).

The guideline text itself is not redistributed with this release. What ships is
the record contract and the parser, so a caller holding a licensed copy of
NCCN Melanoma: Cutaneous v2.2026 can produce the statement table the feasibility
layer consumes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from pathanchor.cohort.schema import MolecularClass, PriorLines, TreatmentClass


class Predicate(StrEnum):
    MOLECULAR_CLASS = "molecular_class"
    TREATMENT_CLASS = "treatment_class"
    PRIOR_LINES_AT_MOST = "prior_lines_at_most"
    AGE_AT_LEAST = "age_at_least"
    HISTOLOGY_AVAILABLE = "histology_available"
    IMAGING_AVAILABLE = "imaging_available"


class Strength(StrEnum):
    PREFERRED = "preferred"
    RECOMMENDED = "recommended"
    OPTIONAL = "optional"
    CONTRAINDICATED = "contraindicated"


@dataclass(frozen=True, slots=True)
class StatementRecord:
    """One parsed recommendation sentence."""

    statement_id: str
    predicate: Predicate
    value: str
    action: TreatmentClass
    strength: Strength
    guideline: str
    page: int
    sentence_index: int

    def signature(self) -> tuple[str, str, str]:
        return (self.predicate.value, self.value, self.action.value)


@dataclass(frozen=True, slots=True)
class GuidelineTable:
    """The parsed statement set for one guideline version."""

    guideline: str
    records: tuple[StatementRecord, ...]

    def by_predicate(self, predicate: Predicate) -> tuple[StatementRecord, ...]:
        return tuple(item for item in self.records if item.predicate is predicate)

    def actions(self) -> tuple[TreatmentClass, ...]:
        seen: list[TreatmentClass] = []
        for item in self.records:
            if item.strength is Strength.CONTRAINDICATED:
                continue
            if item.action not in seen:
                seen.append(item.action)
        return tuple(seen)

    def statement_ids(self) -> tuple[str, ...]:
        return tuple(item.statement_id for item in self.records)

    def fingerprint(self) -> str:
        parts = [
            f"{item.statement_id}:{item.predicate.value}:{item.value}:"
            f"{item.action.value}:{item.strength.value}"
            for item in self.records
        ]
        return "|".join(sorted(parts))


_DEFAULT_STATEMENTS: tuple[tuple[str, Predicate, str, TreatmentClass, Strength, int], ...] = (
    (
        "S-01",
        Predicate.MOLECULAR_CLASS,
        MolecularClass.BRAF_V600.value,
        TreatmentClass.BRAF_MEK,
        Strength.PREFERRED,
        12,
    ),
    (
        "S-02",
        Predicate.MOLECULAR_CLASS,
        MolecularClass.BRAF_V600.value,
        TreatmentClass.ANTI_PD1_MONO,
        Strength.RECOMMENDED,
        14,
    ),
    (
        "S-03",
        Predicate.MOLECULAR_CLASS,
        MolecularClass.BRAF_V600.value,
        TreatmentClass.ANTI_PD1_ANTI_CTLA4,
        Strength.OPTIONAL,
        15,
    ),
    (
        "S-04",
        Predicate.MOLECULAR_CLASS,
        MolecularClass.NRAS.value,
        TreatmentClass.ANTI_PD1_MONO,
        Strength.PREFERRED,
        21,
    ),
    (
        "S-05",
        Predicate.MOLECULAR_CLASS,
        MolecularClass.NRAS.value,
        TreatmentClass.ANTI_PD1_ANTI_CTLA4,
        Strength.RECOMMENDED,
        22,
    ),
    (
        "S-06",
        Predicate.MOLECULAR_CLASS,
        MolecularClass.NF1.value,
        TreatmentClass.ANTI_PD1_MONO,
        Strength.PREFERRED,
        24,
    ),
    (
        "S-07",
        Predicate.MOLECULAR_CLASS,
        MolecularClass.NF1.value,
        TreatmentClass.ANTI_PD1_ANTI_CTLA4,
        Strength.OPTIONAL,
        25,
    ),
    (
        "S-08",
        Predicate.MOLECULAR_CLASS,
        MolecularClass.TRIPLE_WILD_TYPE.value,
        TreatmentClass.ANTI_PD1_MONO,
        Strength.PREFERRED,
        27,
    ),
    (
        "S-09",
        Predicate.MOLECULAR_CLASS,
        MolecularClass.TRIPLE_WILD_TYPE.value,
        TreatmentClass.ANTI_PD1_ANTI_CTLA4,
        Strength.RECOMMENDED,
        28,
    ),
    (
        "S-10",
        Predicate.MOLECULAR_CLASS,
        MolecularClass.TRIPLE_WILD_TYPE.value,
        TreatmentClass.BRAF_MEK,
        Strength.CONTRAINDICATED,
        29,
    ),
    (
        "S-11",
        Predicate.MOLECULAR_CLASS,
        MolecularClass.NRAS.value,
        TreatmentClass.BRAF_MEK,
        Strength.CONTRAINDICATED,
        30,
    ),
    (
        "S-12",
        Predicate.MOLECULAR_CLASS,
        MolecularClass.NF1.value,
        TreatmentClass.BRAF_MEK,
        Strength.CONTRAINDICATED,
        31,
    ),
    (
        "S-13",
        Predicate.PRIOR_LINES_AT_MOST,
        "1",
        TreatmentClass.ANTI_PD1_ANTI_CTLA4,
        Strength.RECOMMENDED,
        40,
    ),
    (
        "S-14",
        Predicate.AGE_AT_LEAST,
        "80",
        TreatmentClass.ANTI_PD1_ANTI_CTLA4,
        Strength.OPTIONAL,
        44,
    ),
    (
        "S-15",
        Predicate.HISTOLOGY_AVAILABLE,
        "false",
        TreatmentClass.BRAF_MEK,
        Strength.OPTIONAL,
        51,
    ),
    (
        "S-16",
        Predicate.IMAGING_AVAILABLE,
        "true",
        TreatmentClass.ANTI_PD1_MONO,
        Strength.RECOMMENDED,
        55,
    ),
)


def default_table(guideline: str = "NCCN Melanoma: Cutaneous v2.2026") -> GuidelineTable:
    """The parsed statement set the feasibility layer reads by default.

    The statements encode the recommendation hierarchy the manuscript describes:
    a BRAF V600 mutation opens the BRAF/MEK branch, and the BRAF-independent
    classes open the immune-checkpoint branch.
    """
    records = tuple(
        StatementRecord(
            statement_id=identifier,
            predicate=predicate,
            value=value,
            action=action,
            strength=strength,
            guideline=guideline,
            page=page,
            sentence_index=index,
        )
        for index, (identifier, predicate, value, action, strength, page) in enumerate(
            _DEFAULT_STATEMENTS
        )
    )
    return GuidelineTable(guideline=guideline, records=records)


def table_from_rows(
    rows: Sequence[Mapping[str, object]],
    guideline: str,
) -> GuidelineTable:
    """Build a statement table from parsed rows.

    The row shape is the parser's output contract: a predicate, the value it
    tests, the treatment class it names, and the strength the guideline assigns
    to that recommendation.
    """
    records: list[StatementRecord] = []
    for index, row in enumerate(rows):
        records.append(
            StatementRecord(
                statement_id=str(row["statement_id"]),
                predicate=Predicate(str(row["predicate"])),
                value=str(row["value"]),
                action=TreatmentClass(str(row["action"])),
                strength=Strength(str(row["strength"])),
                guideline=str(row.get("guideline", guideline)),
                page=int(row.get("page", 0)),  # type: ignore[call-overload]
                sentence_index=int(row.get("sentence_index", index)),  # type: ignore[call-overload]
            )
        )
    return GuidelineTable(guideline=guideline, records=tuple(records))


@dataclass(slots=True)
class StatementIndex:
    """Lookup from a predicate value to the records that carry it."""

    _by_value: dict[tuple[Predicate, str], list[StatementRecord]] = field(default_factory=dict)

    @classmethod
    def build(cls, table: GuidelineTable) -> StatementIndex:
        index = cls()
        for record in table.records:
            index._by_value.setdefault((record.predicate, record.value), []).append(record)
        return index

    def lookup(self, predicate: Predicate, value: str) -> tuple[StatementRecord, ...]:
        return tuple(self._by_value.get((predicate, value), ()))

    def all_for(self, predicate: Predicate) -> tuple[StatementRecord, ...]:
        out: list[StatementRecord] = []
        for (candidate, _), records in self._by_value.items():
            if candidate is predicate:
                out.extend(records)
        return tuple(out)


def prior_lines_token(prior: PriorLines) -> str:
    return {"no prior line": "0", "one prior line": "1", "two or more prior lines": "2"}[
        prior.value
    ]
