"""Statement parsing and the feasible set of Eq. (1)."""

from __future__ import annotations

import pytest

from pathanchor.cohort.schema import MolecularClass, PriorLines, TreatmentClass
from pathanchor.guideline.feasibility import (
    branch_disjointness,
    default_feasibility_layer,
    discordance_rate,
    feasible_counts,
)
from pathanchor.guideline.statements import (
    Predicate,
    Strength,
    default_table,
    prior_lines_token,
    table_from_rows,
)


def test_default_table_parses() -> None:
    table = default_table()
    assert len(table.records) >= 16
    assert TreatmentClass.BRAF_MEK in table.actions()
    assert len(table.fingerprint()) > 0


def test_braf_directed_class_needs_the_mutation(records_factory) -> None:
    layer = default_feasibility_layer()
    braf = records_factory(MolecularClass.BRAF_V600)
    wild = records_factory(MolecularClass.NRAS)
    assert TreatmentClass.BRAF_MEK in layer.resolve(braf).feasible
    assert TreatmentClass.BRAF_MEK not in layer.resolve(wild).feasible
    assert TreatmentClass.BRAF_MEK in layer.resolve(wild).contraindicated


def test_discordance_is_zero_for_the_priority_option(records_factory) -> None:
    layer = default_feasibility_layer()
    results = [layer.resolve(records_factory(molecular)) for molecular in MolecularClass]
    chosen = [result.priority[0].value for result in results]
    assert discordance_rate(chosen, results) == 0.0


def test_discordance_counts_an_outside_option(records_factory) -> None:
    layer = default_feasibility_layer()
    results = [layer.resolve(records_factory(MolecularClass.NRAS))]
    assert discordance_rate([TreatmentClass.BRAF_MEK.value], results) == 1.0


def test_the_branches_share_the_checkpoint_classes(records_factory) -> None:
    layer = default_feasibility_layer()
    records = [records_factory(item) for item in MolecularClass]
    results = [layer.resolve(record) for record in records]
    report = branch_disjointness(results, records)
    assert not bool(report["literal_disjoint"])
    assert bool(report["braf_directed_in_one_branch_only"])


def test_prior_lines_token_mapping() -> None:
    assert prior_lines_token(PriorLines.NONE) == "0"
    assert prior_lines_token(PriorLines.TWO_OR_MORE) == "2"


def test_prior_line_statement_only_matches_inside_its_bound(records_factory) -> None:
    layer = default_feasibility_layer()
    early = records_factory(MolecularClass.TRIPLE_WILD_TYPE, prior_lines=PriorLines.NONE)
    advanced = records_factory(MolecularClass.TRIPLE_WILD_TYPE, prior_lines=PriorLines.TWO_OR_MORE)
    combo = TreatmentClass.ANTI_PD1_ANTI_CTLA4.value
    assert "S-13" in layer.resolve(early).statement_ids_for(combo)
    assert "S-13" not in layer.resolve(advanced).statement_ids_for(combo)
    assert combo in layer.resolve(advanced).feasible


def test_row_table_round_trip() -> None:
    rows = [
        {
            "statement_id": "X-1",
            "predicate": Predicate.MOLECULAR_CLASS.value,
            "value": MolecularClass.BRAF_V600.value,
            "action": TreatmentClass.BRAF_MEK.value,
            "strength": Strength.PREFERRED.value,
            "page": 3,
        }
    ]
    table = table_from_rows(rows, "caller supplied guideline label")
    assert len(table.records) == 1
    assert table.records[0].predicate is Predicate.MOLECULAR_CLASS


def test_feasible_counts_groups_configurations(records_factory) -> None:
    layer = default_feasibility_layer()
    results = [layer.resolve(records_factory(item)) for item in MolecularClass]
    counts = feasible_counts(results)
    assert sum(counts.values()) == len(results)


@pytest.mark.parametrize("molecular", list(MolecularClass))
def test_every_class_keeps_a_checkpoint_option(molecular, records_factory) -> None:
    layer = default_feasibility_layer()
    result = layer.resolve(records_factory(molecular))
    assert TreatmentClass.ANTI_PD1_MONO in result.feasible
