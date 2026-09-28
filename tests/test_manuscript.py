"""The manuscript's arithmetic against itself."""

from __future__ import annotations

from pathanchor.evaluation import manuscript


def test_every_family_is_non_empty() -> None:
    families = manuscript.all_families()
    assert set(families) == {
        "cohort",
        "table_three",
        "trust_axes",
        "panel",
        "pre_specified",
        "table_one",
        "abstract",
        "subgroup_text",
        "branch_disjointness",
    }
    for name, checks in families.items():
        assert checks, name


def test_cohort_arithmetic_closes() -> None:
    records = [check.record() for check in manuscript.cohort_arithmetic()]
    assert all(record["status"] == "PASS" for record in records)


def test_table_three_arithmetic_closes() -> None:
    records = [check.record() for check in manuscript.table_three_arithmetic()]
    assert all(record["status"] == "PASS" for record in records)


def test_panel_arithmetic_closes() -> None:
    records = [check.record() for check in manuscript.panel_arithmetic()]
    assert all(record["status"] == "PASS" for record in records)


def test_pre_specified_margin_is_unreachable_and_reported() -> None:
    checks = manuscript.pre_specified_margin()
    assert len(checks) == 1
    assert not checks[0].passes
    assert checks[0].computed < checks[0].reported


def test_subgroup_narrative_disagrees_with_the_table() -> None:
    checks = manuscript.subgroup_text_vs_table()
    assert all(not check.passes for check in checks)


def test_comparator_count_check_fails() -> None:
    checks = manuscript.table_one_claims()
    failing = [check.name for check in checks if not check.passes]
    assert "comparator_count" in failing


def test_rank_correlation_helpers() -> None:
    assert manuscript.spearman_rank_correlation([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 1.0
    assert manuscript.spearman_rank_correlation([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == -1.0


def test_flatten_and_inconsistency_grouping() -> None:
    records = manuscript.flatten(manuscript.all_families())
    families = manuscript.inconsistency_families(records)
    assert families
    assert all(isinstance(names, list) for names in families.values())
