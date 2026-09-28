"""Discrimination, calibration, utility and agreement statistics."""

from __future__ import annotations

import numpy as np
import pytest

from pathanchor.metrics.agreement import cohen_kappa, kappa_from_definition
from pathanchor.metrics.calibration import (
    ECE_BINS,
    bin_membership,
    brier_score,
    expected_calibration_error,
    hand_computed_ece,
    reliability_curve,
)
from pathanchor.metrics.clinical import (
    net_benefit,
    net_benefit_detail,
    treat_all_net_benefit,
)
from pathanchor.metrics.discrimination import (
    auprc,
    auroc,
    auroc_rank_identity,
    delong_test,
)
from pathanchor.metrics.summary import benjamini_hochberg, holm_bonferroni


def test_auroc_matches_the_pair_count() -> None:
    labels = np.asarray([0, 1, 0, 1, 1, 0, 0, 1, 1, 1], dtype=np.int64)
    scores = np.asarray([0.1, 0.9, 0.4, 0.35, 0.8, 0.2, 0.6, 0.7, 0.55, 0.95])
    hand = 0.0
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    for positive in positives.tolist():
        for negative in negatives.tolist():
            hand += 1.0 if positive > negative else 0.5 if positive == negative else 0.0
    hand /= positives.shape[0] * negatives.shape[0]
    assert auroc(labels, scores) == pytest.approx(hand)
    assert auroc_rank_identity(labels, scores) == pytest.approx(hand)


def test_auroc_of_all_ties_is_one_half() -> None:
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    assert auroc(labels, np.full(4, 0.3)) == pytest.approx(0.5)


def test_auprc_matches_longhand_average_precision() -> None:
    labels = np.asarray([1, 0, 1, 0, 1, 1, 0, 1, 0, 0], dtype=np.int64)
    scores = np.linspace(1.0, 0.1, 10)
    order = np.argsort(-scores)
    cumulative = np.cumsum(labels[order])
    precision_sum = sum(
        float(cumulative[position]) / float(position + 1)
        for position in range(labels.shape[0])
        if labels[order][position] == 1
    )
    assert auprc(labels, scores) == pytest.approx(precision_sum / int(np.sum(labels)))


def test_delong_reports_a_small_p_value_for_a_separated_pair() -> None:
    rng = np.random.default_rng(5)
    labels = (rng.random(300) < 0.4).astype(np.int64)
    strong = labels + rng.normal(0.0, 0.4, 300)
    weak = labels * 0.3 + rng.normal(0.0, 1.0, 300)
    result = delong_test(labels, strong, weak)
    assert result.difference > 0.0
    assert result.p_value < 0.01


def test_delong_of_a_model_against_itself_is_flat() -> None:
    rng = np.random.default_rng(6)
    labels = (rng.random(200) < 0.5).astype(np.int64)
    scores = rng.normal(0.0, 1.0, 200)
    result = delong_test(labels, scores, scores)
    assert result.difference == pytest.approx(0.0)
    assert result.p_value == pytest.approx(1.0)


def test_ece_vectorised_and_longhand_agree() -> None:
    rng = np.random.default_rng(7)
    probabilities = np.clip(rng.beta(2.0, 2.0, 500), 0.0, 1.0)
    labels = (rng.random(500) < probabilities).astype(np.int64)
    assert expected_calibration_error(labels, probabilities, ECE_BINS) == pytest.approx(
        hand_computed_ece(labels, probabilities, ECE_BINS), abs=1e-12
    )


def test_bin_membership_stays_inside_the_range() -> None:
    probabilities = np.asarray([0.0, 0.5, 1.0])
    membership = bin_membership(probabilities, ECE_BINS)
    assert int(membership.min()) == 0
    assert int(membership.max()) == ECE_BINS - 1


def test_reliability_curve_counts_sum_to_the_sample() -> None:
    rng = np.random.default_rng(8)
    probabilities = np.clip(rng.random(300), 0.0, 1.0)
    labels = (rng.random(300) < probabilities).astype(np.int64)
    curve = reliability_curve(labels, probabilities, ECE_BINS)
    assert int(np.sum(curve.count)) == 300
    assert curve.populated() > 0


def test_brier_is_the_mean_squared_error() -> None:
    labels = np.asarray([0, 1, 1], dtype=np.int64)
    probabilities = np.asarray([0.25, 0.75, 0.5])
    assert brier_score(labels, probabilities) == pytest.approx(
        float(np.mean((probabilities - labels) ** 2))
    )


def test_net_benefit_matches_the_longhand_form() -> None:
    rng = np.random.default_rng(9)
    probabilities = np.clip(rng.beta(2.0, 3.0, 400), 0.0, 1.0)
    labels = (rng.random(400) < probabilities).astype(np.int64)
    threshold = 0.35
    odds = threshold / (1.0 - threshold)
    predicted = probabilities >= threshold
    expected = (
        float(np.sum(predicted & (labels == 1))) / 400.0
        - float(np.sum(predicted & (labels == 0))) / 400.0 * odds
    )
    assert net_benefit(labels, probabilities, threshold) == pytest.approx(expected)
    detail = net_benefit_detail(labels, probabilities, threshold)
    assert detail.n == 400


def test_treat_all_reference_falls_with_the_threshold() -> None:
    assert treat_all_net_benefit(0.4, 0.2) > treat_all_net_benefit(0.4, 0.5)


def test_kappa_from_the_contingency_table_matches_the_definition() -> None:
    first = ["A", "B", "A", "C", "B", "A", "C", "C", "B", "A"]
    second = ["A", "B", "B", "C", "B", "A", "A", "C", "B", "B"]
    assert cohen_kappa(first, second, ["A", "B", "C"]).kappa == pytest.approx(
        kappa_from_definition(first, second)
    )


def test_kappa_of_a_perfect_agreement_is_one() -> None:
    ratings = ["A", "B", "C", "A", "B"]
    assert cohen_kappa(ratings, ratings, ["A", "B", "C"]).kappa == pytest.approx(1.0)


def test_holm_adjusted_values_match_the_step_down_definition() -> None:
    p_values = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216]
    computed = holm_bonferroni(p_values)
    m = len(p_values)
    order = sorted(range(m), key=lambda slot: p_values[slot])
    reference = [0.0] * m
    running = 0.0
    for rank, slot in enumerate(order):
        value = (m - rank) * p_values[slot]
        running = max(running, value)
        reference[slot] = min(running, 1.0)
    assert computed == pytest.approx(reference)
    assert benjamini_hochberg(p_values)[0] <= holm_bonferroni(p_values)[0] + 1e-12
