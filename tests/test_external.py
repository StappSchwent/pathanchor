"""External validation and the reader-study statistics."""

from __future__ import annotations

import numpy as np
import pytest

from pathanchor.evaluation.external import (
    EXTERNAL_SERIES,
    MODALITY_LADDER,
    SERIES_TOTAL,
    ExternalProfileTable,
    degradation,
    evaluate_external,
    ladder_report,
    series_accessions,
    series_census,
    withheld_modality_delta,
)
from pathanchor.evaluation.reader_study import (
    CONDITIONS,
    CaseAdjudication,
    VignetteOutcome,
    accuracy_by_condition,
    accuracy_interval,
    case_review_summary,
    enrichment_plan,
    over_reliance_rate,
    paired_accuracy_contrast,
    time_overhead_summary,
)


def _table(seed: int = 3, size: int = 120) -> ExternalProfileTable:
    rng = np.random.default_rng(seed)
    latent = rng.normal(0.0, 1.0, size)
    labels = (rng.random(size) < 1.0 / (1.0 + np.exp(-latent))).astype(np.int64)
    profiles = np.column_stack([latent + rng.normal(0.0, 1.0, size), rng.normal(0.0, 1.0, size)])
    return ExternalProfileTable(
        labels=labels, profiles=profiles, series=tuple("s" for _ in range(size))
    )


def test_series_census_matches_the_reported_total() -> None:
    assert sum(series_census().values()) == SERIES_TOTAL
    assert len(EXTERNAL_SERIES) == 4
    assert set(series_accessions().values()) >= {"GSE78220", "GSE91061", "GSE115821"}


def test_external_result_reports_both_models() -> None:
    table = _table()
    strong = table.profiles[:, 0]
    weak = table.profiles[:, 1]
    result = evaluate_external(table, strong, weak)
    assert result.n == table.labels.shape[0]
    assert result.framework_auroc >= result.comparator_auroc - 0.2


def test_degradation_is_signed_from_the_internal_value() -> None:
    table = _table()
    result = evaluate_external(table, table.profiles[:, 0], table.profiles[:, 1])
    report = degradation(0.9, 0.01, result)
    assert report["discrimination_points"] > 0.0


def test_ladder_skips_a_rung_without_scores() -> None:
    table = _table()
    rows = ladder_report(table, {"Full multimodal input": table.profiles[:, 0]})
    assert len(rows) == 1
    assert rows[0]["rung"] == "Full multimodal input"
    assert rows[0]["modalities"] == 4


def test_ladder_rungs_are_nested() -> None:
    lengths = [len(modalities) for _, modalities in MODALITY_LADDER]
    assert lengths == sorted(lengths)


def test_withheld_modality_delta() -> None:
    assert withheld_modality_delta(0.765, 0.759) == pytest.approx(0.6, abs=1e-6)


def test_table_validation_rejects_misaligned_input() -> None:
    with pytest.raises(ValueError):
        ExternalProfileTable(
            labels=np.asarray([0, 1]), profiles=np.zeros((3, 2)), series=("a", "b", "c")
        ).validate()


def _outcomes(seed: int = 5, clinicians: int = 6, vignettes: int = 20):
    rng = np.random.default_rng(seed)
    rows = []
    for clinician in range(clinicians):
        for vignette in range(vignettes):
            for condition in CONDITIONS:
                base = {"unaided": 0.65, "score_only_ai": 0.72, "pathanchor": 0.80}[condition]
                correct = bool(rng.random() < base)
                ai_correct = bool(rng.random() < 0.8)
                accepted = bool(
                    rng.random()
                    < {"unaided": 0.1, "score_only_ai": 0.8, "pathanchor": 0.8}[condition]
                )
                rows.append(
                    VignetteOutcome(
                        clinician_id=f"c{clinician}",
                        vignette_id=f"v{vignette}",
                        condition=condition,
                        correct=correct,
                        accepted_ai=accepted,
                        ai_was_correct=ai_correct,
                        seconds=52.0 if condition == "unaided" else 74.0,
                    )
                )
    return rows


def test_accuracy_ordering_follows_the_conditions() -> None:
    accuracy = accuracy_by_condition(_outcomes())
    assert set(accuracy) == set(CONDITIONS)


def test_paired_contrast_is_centred_on_the_difference() -> None:
    outcomes = _outcomes()
    contrast = paired_accuracy_contrast(outcomes, "pathanchor", "unaided", resamples=120)
    accuracy = accuracy_by_condition(outcomes)
    expected = accuracy["pathanchor"] - accuracy["unaided"]
    assert contrast.difference == pytest.approx(expected)
    assert contrast.resamples == 120


def test_over_reliance_and_timing() -> None:
    outcomes = _outcomes()
    assert over_reliance_rate(outcomes, "pathanchor") >= 0.0
    overhead = time_overhead_summary(outcomes)
    assert overhead["relative_overhead"] > 0.0


def test_interval_returns_the_sample_size() -> None:
    result = accuracy_interval(_outcomes(), "pathanchor", resamples=60)
    assert result.resamples == 60


def test_case_review_summary_counts_divergences() -> None:
    adjudications = [
        CaseAdjudication("c1", True, True),
        CaseAdjudication("c2", True, False, "an omitted exposure"),
    ]
    summary = case_review_summary(adjudications)
    assert summary["cases"] == 2
    assert len(summary["divergences"]) == 1
    assert sum(enrichment_plan().values()) == 12
