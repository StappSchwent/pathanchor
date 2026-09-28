"""Split conformal calibration, the monotone transform and abstention."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.isotonic import IsotonicRegression

from pathanchor.conformal.split import (
    SplitConformal,
    fit_isotonic,
    quantile_value,
)
from pathanchor.utils.numerics import quantile_index


def _split(seed: int, size: int = 400) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    latent = rng.normal(0.0, 1.0, size)
    labels = (rng.random(size) < 1.0 / (1.0 + np.exp(-latent))).astype(np.int64)
    scores = latent + rng.normal(0.0, 1.0, size)
    return (scores, labels)


def test_quantile_index_follows_the_definition() -> None:
    for n in (10, 51, 200, 858):
        for level in (0.90, 0.95):
            expected = min(max(int(np.ceil((n + 1) * level)), 1), n)
            assert quantile_index(n, level) == expected


def test_quantile_value_uses_the_index() -> None:
    values = np.asarray([0.9, 0.1, 0.4, 0.7, 0.2, 0.55, 0.35, 0.8, 0.65, 0.5])
    alpha = 0.10
    index = quantile_index(values.shape[0], 1.0 - alpha)
    assert quantile_value(values, alpha) == pytest.approx(float(np.sort(values)[index - 1]))


def test_isotonic_is_monotone_and_matches_a_second_solver() -> None:
    values, labels = _split(3, 300)
    order = np.argsort(values)
    fitted = fit_isotonic(values[order], labels[order].astype(np.float64))
    assert fitted.monotone_check()
    reference = IsotonicRegression(out_of_bounds="clip").fit(
        values[order], labels[order].astype(np.float64)
    )
    in_sample = float(
        np.max(np.abs(fitted(values[order]) - np.asarray(reference.predict(values[order]))))
    )
    assert in_sample < 0.05
    probe = np.linspace(float(values.min()), float(values.max()), 50)
    assert bool(np.all(np.diff(fitted(probe)) >= -1e-12))


def test_coverage_holds_at_the_nominal_level() -> None:
    train_scores, train_labels = _split(11, 600)
    calibration, calibration_labels = _split(12, 400)
    evaluation, evaluation_labels = _split(13, 400)
    layer = SplitConformal(alpha=0.10, tau_abs=0.35)
    layer.fit(train_scores, train_labels, calibration, calibration_labels)
    assert layer.coverage(evaluation, evaluation_labels) >= 0.85


def test_sets_are_ordered_and_clipped() -> None:
    scores, labels = _split(21, 200)
    layer = SplitConformal(alpha=0.10, tau_abs=0.35)
    layer.fit(scores, labels, scores, labels)
    low, high = layer.interval(scores)
    assert bool(np.all(low <= high))
    assert float(np.min(low)) >= 0.0
    assert float(np.max(high)) <= 1.0


def test_abstention_triggers_above_the_threshold() -> None:
    scores, labels = _split(31, 200)
    layer = SplitConformal(alpha=0.10, tau_abs=0.05)
    layer.fit(scores, labels, scores, labels)
    sets = layer.prediction_sets(scores)
    assert all(item.width > 0.0 for item in sets)
    assert layer.abstention_rate(scores) == 1.0


def test_wider_alpha_shrinks_the_set() -> None:
    scores, labels = _split(41, 300)
    narrow = SplitConformal(alpha=0.20, tau_abs=0.35).fit(scores, labels, scores, labels)
    wide = SplitConformal(alpha=0.05, tau_abs=0.35).fit(scores, labels, scores, labels)
    assert narrow.quantile <= wide.quantile


def test_unfitted_layer_refuses_to_score() -> None:
    layer = SplitConformal(alpha=0.10, tau_abs=0.35)
    with pytest.raises(RuntimeError):
        layer.interval(np.asarray([0.5]))
