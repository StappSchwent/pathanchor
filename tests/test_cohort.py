"""Cohort fixture, marginal allocation and the partition contract."""

from __future__ import annotations

import numpy as np
import pytest

from pathanchor.cohort.fixture import CohortSpec, bayes_scores, build_cohort_fixture, scale_spec
from pathanchor.cohort.marginals import (
    MarginSpec,
    build_table,
    margin_error,
    margin_sums,
    split_proportionally,
    table_cells,
)
from pathanchor.cohort.schema import Split, region_of, subgroup_labels, validate_record
from pathanchor.cohort.splits import SplitFractions, partition_report, stratified_partition
from pathanchor.metrics.discrimination import auroc


def test_full_fixture_holds_every_reported_margin() -> None:
    fixture = build_cohort_fixture()
    assert fixture.size() == 4286
    assert fixture.margin_deviations == {
        "molecular_treatment": 0,
        "site": 0,
        "sex": 0,
        "age": 0,
    }
    assert fixture.molecular_counts() == {
        "BRAF V600 mutant": 1894,
        "NF1 altered": 531,
        "NRAS mutant": 926,
        "Triple wild-type": 935,
    }
    assert fixture.treatment_counts() == {
        "Anti-PD-1 + anti-CTLA-4": 1161,
        "Anti-PD-1 monotherapy": 2242,
        "BRAF/MEK inhibition": 883,
    }


def test_partition_sizes_and_positives() -> None:
    fixture = build_cohort_fixture()
    assert [fixture.size(split) for split in Split] == [2572, 858, 856]
    assert fixture.positive_counts == {
        "training": 1039,
        "calibration": 346,
        "testing": 346,
    }
    assert sum(fixture.positive_counts.values()) == 1731


def test_test_split_reproduces_the_subgroup_table() -> None:
    fixture = build_cohort_fixture()
    assert fixture.molecular_counts(Split.TEST) == {
        "BRAF V600 mutant": 378,
        "NF1 altered": 106,
        "NRAS mutant": 185,
        "Triple wild-type": 187,
    }
    assert fixture.treatment_counts(Split.TEST) == {
        "Anti-PD-1 + anti-CTLA-4": 232,
        "Anti-PD-1 monotherapy": 448,
        "BRAF/MEK inhibition": 176,
    }
    assert fixture.site_counts(Split.TEST) == {
        "Site A": 282,
        "Site B": 221,
        "Site C": 197,
        "Site D": 156,
    }
    assert fixture.region_counts(Split.TEST) == {"Region I": 503, "Region II": 353}


def test_record_contract_holds_for_every_record() -> None:
    fixture = build_cohort_fixture()
    for record in fixture.records:
        validate_record(record)
        labels = subgroup_labels(record)
        assert set(labels) == {
            "Molecular",
            "Treatment class",
            "Site",
            "Region",
            "Demographic / clinical",
        }
        assert record.region is region_of(record.site)


def test_bayes_scores_are_bounded_away_from_perfection() -> None:
    fixture = build_cohort_fixture()
    scores = bayes_scores(fixture, Split.TEST)
    labels = fixture.labels(Split.TEST)
    value = auroc(labels, scores)
    assert 0.55 < value < 0.95


def test_scaled_spec_still_holds_its_margins() -> None:
    scaled = scale_spec(CohortSpec(), 300, 60, 60, 121)
    fixture = build_cohort_fixture(scaled)
    assert fixture.size() == 300
    assert all(value == 0 for value in fixture.margin_deviations.values())
    assert sum(fixture.molecular_counts().values()) == 300


def test_allocation_is_exact_on_a_two_way_margin() -> None:
    shape = (3, 2)
    margins = [
        MarginSpec("rows", (0,), np.asarray([10, 20, 30], dtype=np.int64)),
        MarginSpec("columns", (1,), np.asarray([30, 30], dtype=np.int64)),
    ]
    table = build_table(shape, margins, 60)
    assert margin_error(table, margins) == {"rows": 0, "columns": 0}
    assert int(np.sum(table)) == 60


def test_allocation_rejects_an_impossible_total() -> None:
    margins = [MarginSpec("rows", (0,), np.asarray([1, 1], dtype=np.int64))]
    with pytest.raises(ValueError):
        build_table((2,), margins, 5)


def test_split_proportionally_preserves_cells() -> None:
    table = np.asarray([[10, 20], [30, 40]], dtype=np.int64)
    first, second = split_proportionally(table, 40)
    assert int(np.sum(first)) == 40
    np.testing.assert_array_equal(first + second, table)


def test_table_cells_expand_with_labels() -> None:
    table = np.asarray([[2, 0], [1, 3]], dtype=np.int64)
    cells = table_cells(table, [["a", "b"], ["x", "y"]])
    assert ("a", "x") in [names for names, _ in cells]
    assert sum(count for _, count in cells) == 6


def test_stratified_partition_keeps_sites_everywhere() -> None:
    fixture = build_cohort_fixture()
    partition = stratified_partition(fixture.records, SplitFractions(), seed=7)
    partition.check_disjoint()
    report = partition_report(partition)
    assert bool(report["patient_disjoint"])
    assert bool(report["sites_in_every_part"])


def test_margin_sums_reads_the_requested_axis() -> None:
    table = np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    np.testing.assert_array_equal(margin_sums(table, (0,)), np.asarray([6, 15]))
    np.testing.assert_array_equal(margin_sums(table, (1,)), np.asarray([5, 7, 9]))
