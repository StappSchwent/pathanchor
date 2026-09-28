"""Numerical helpers shared across the estimator, the metrics and the checks.

Ref: Sec. III-E (monotone transform), Algorithm 1 line 6 (path marginalisation).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from pathanchor.utils.types import BoolArray, FloatArray, IntArray

_LOG_FLOAT_TINY = -1.0e300


def logsumexp(values: FloatArray, axis: int | None = None, keepdims: bool = False) -> FloatArray:
    """Stable log-sum-exp.

    The per-row maximum is kept as a dimension so a two-dimensional input yields a
    two-dimensional result rather than a squeezed tuple.
    """
    if axis is None:
        scalar_peak = float(np.max(values))
        scalar_total = scalar_peak + float(np.log(np.sum(np.exp(values - scalar_peak))))
        if keepdims:
            return np.asarray([[scalar_total]], dtype=np.float64)
        scalar: NDArray[np.float64] = np.asarray(scalar_total, dtype=np.float64).reshape(())
        return scalar
    row_peak: FloatArray = np.max(values, axis=axis, keepdims=True)
    finite_peak: FloatArray = np.where(np.isfinite(row_peak), row_peak, 0.0)
    row_total: FloatArray = finite_peak + np.log(
        np.sum(np.exp(values - finite_peak), axis=axis, keepdims=True)
    )
    if keepdims:
        return row_total
    reduced: FloatArray = np.squeeze(row_total, axis=axis)
    return reduced


def softmax(values: FloatArray, axis: int = -1) -> FloatArray:
    shifted = values - np.max(values, axis=axis, keepdims=True)
    weights: FloatArray = np.exp(shifted)
    normaliser: FloatArray = np.sum(weights, axis=axis, keepdims=True)
    result: FloatArray = weights / normaliser
    return result


def sigmoid(x: FloatArray) -> FloatArray:
    """Numerically stable logistic function evaluated elementwise."""
    out = np.empty_like(x)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def clip_probability(values: FloatArray, eps: float = 1e-9) -> FloatArray:
    return np.clip(values, eps, 1.0 - eps)


def average_ranks(values: FloatArray) -> FloatArray:
    """1-based ranks with tied values sharing their mean rank.

    This is the tie convention the U-statistic form of the AUROC assumes; using
    the ordinal rank instead silently changes the statistic whenever a score
    repeats, which happens for every binarised comparator.
    """
    order = np.argsort(values, kind="mergesort")
    ordered = values[order]
    ranks = np.empty(values.shape[0], dtype=np.float64)
    n = values.shape[0]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and ordered[j + 1] == ordered[i]:
            j += 1
        shared = 0.5 * (i + j) + 1.0
        ranks[order[i : j + 1]] = shared
        i = j + 1
    return ranks


def stable_softplus(x: FloatArray) -> FloatArray:
    return np.logaddexp(0.0, x)


def bernoulli_entropy(p: FloatArray) -> FloatArray:
    q = clip_probability(p)
    result: FloatArray = -(q * np.log(q) + (1.0 - q) * np.log(1.0 - q))
    return result


def standardise(
    values: FloatArray, mean: FloatArray, scale: FloatArray, eps: float = 1e-8
) -> FloatArray:
    result: FloatArray = (values - mean) / np.maximum(scale, eps)
    return result


def safe_divide(numerator: float, denominator: float, fallback: float = 0.0) -> float:
    if denominator == 0.0:
        return fallback
    return numerator / denominator


def quantile_index(n: int, level: float) -> int:
    """The 1-based index ceil((n + 1) * level), clamped into range.

    Used by the split-conformal layer of Eq. (2), where the calibration quantile
    is the ceil((n + 1)(1 - alpha))-th smallest nonconformity score.
    """
    raw = int(np.ceil((n + 1) * level))
    return int(np.clip(raw, 1, n))


def assert_finite(name: str, values: FloatArray) -> None:
    if not bool(np.all(np.isfinite(values))):
        raise FloatingPointError(f"{name} contains a non-finite entry")


def is_binary(labels: IntArray) -> bool:
    return bool(np.all((labels == 0) | (labels == 1)))


def one_hot(indices: IntArray, width: int) -> FloatArray:
    out = np.zeros((indices.shape[0], width), dtype=np.float64)
    out[np.arange(indices.shape[0]), indices] = 1.0
    return out


def binarise(values: FloatArray, threshold: float) -> BoolArray:
    result: BoolArray = values >= threshold
    return result


def concat_arrays(parts: list[FloatArray]) -> FloatArray:
    if not parts:
        return np.zeros(0, dtype=np.float64)
    result: FloatArray = np.concatenate(parts)
    return result


def outer_scale(vector: FloatArray, factor: float) -> NDArray[np.float64]:
    scaled: NDArray[np.float64] = vector * factor
    return scaled


def allocate_counts(total: int, weights: Sequence[float], floor: int = 1) -> list[int]:
    """Integer allocation by largest remainder, never smaller than ``floor``.

    The floor keeps every declared category present, which the resource censuses
    assert; without it a low-weighted category would silently vanish and the
    reported type count would be short.
    """
    if total < floor * len(weights):
        raise ValueError("total is too small to satisfy the per-category floor")
    normalised = np.asarray(weights, dtype=np.float64)
    if not bool(np.all(normalised > 0.0)):
        raise ValueError("all allocation weights must be positive")
    budget = total - floor * len(weights)
    shares = normalised / float(np.sum(normalised))
    raw = shares * budget
    base = np.floor(raw).astype(np.int64)
    remainder = int(budget - int(np.sum(base)))
    if remainder > 0:
        order = np.argsort(-(raw - base), kind="stable")
        base[order[:remainder]] += 1
    return [int(value) + floor for value in base.tolist()]
