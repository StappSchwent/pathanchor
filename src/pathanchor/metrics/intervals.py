"""Bootstrap intervals.

Ref: Sec. III-H ("expressed as % 95 confidence intervals based on the 2,000
stratified bootstrapped resamples").

Resampling is stratified on the label, so a bootstrap replicate keeps the
observed number of responders and the interval is not inflated by the class
ratio moving between replicates.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from pathanchor.utils.types import FloatArray, IntArray, Interval

DEFAULT_RESAMPLES: int = 2000
DEFAULT_LEVEL: float = 0.95


@dataclass(frozen=True, slots=True)
class BootstrapResult:
    value: float
    interval: Interval
    resamples: int
    standard_error: float


def stratified_indices(labels: IntArray, rng: np.random.Generator) -> IntArray:
    positives = np.nonzero(labels == 1)[0]
    negatives = np.nonzero(labels == 0)[0]
    parts: list[IntArray] = []
    if positives.shape[0] > 0:
        parts.append(rng.choice(positives, size=positives.shape[0], replace=True))
    if negatives.shape[0] > 0:
        parts.append(rng.choice(negatives, size=negatives.shape[0], replace=True))
    if not parts:
        return np.zeros(0, dtype=np.int64)
    return np.concatenate(parts).astype(np.int64)


def unstratified_indices(size: int, rng: np.random.Generator) -> IntArray:
    return rng.integers(0, size, size=size).astype(np.int64)


def percentile_interval(
    values: FloatArray,
    level: float = DEFAULT_LEVEL,
) -> Interval:
    if values.shape[0] == 0:
        return Interval(0.0, 0.0, level)
    tail = (1.0 - level) / 2.0
    low = float(np.quantile(values, tail))
    high = float(np.quantile(values, 1.0 - tail))
    return Interval(low, high, level)


def bootstrap_interval(
    labels: IntArray,
    statistic: Callable[[IntArray], float],
    resamples: int = DEFAULT_RESAMPLES,
    level: float = DEFAULT_LEVEL,
    seed: int = 20260101,
    stratified: bool = True,
) -> BootstrapResult:
    rng = np.random.default_rng(seed)
    observed = float(statistic(np.arange(labels.shape[0], dtype=np.int64)))
    draws = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        sample = (
            stratified_indices(labels, rng)
            if stratified
            else unstratified_indices(labels.shape[0], rng)
        )
        draws[index] = float(statistic(sample))
    finite = draws[np.isfinite(draws)]
    interval = percentile_interval(finite, level)
    error = float(np.std(finite, ddof=1)) if finite.shape[0] > 1 else 0.0
    return BootstrapResult(
        value=observed, interval=interval, resamples=resamples, standard_error=error
    )


def bootstrap_statistics(
    labels: IntArray,
    statistics: Sequence[Callable[[IntArray], float]],
    resamples: int = DEFAULT_RESAMPLES,
    level: float = DEFAULT_LEVEL,
    seed: int = 20260101,
) -> list[BootstrapResult]:
    rng_seed = seed
    results: list[BootstrapResult] = []
    for statistic in statistics:
        results.append(
            bootstrap_interval(labels, statistic, resamples=resamples, level=level, seed=rng_seed)
        )
        rng_seed += 1
    return results


def bootstrap_difference(
    labels: IntArray,
    statistic_a: Callable[[IntArray], float],
    statistic_b: Callable[[IntArray], float],
    resamples: int = DEFAULT_RESAMPLES,
    level: float = DEFAULT_LEVEL,
    seed: int = 20260101,
) -> BootstrapResult:
    """Paired bootstrap interval for the difference of two statistics.

    Both statistics are evaluated on the *same* replicate, which is what makes
    the interval paired and much tighter than differencing two independent ones.
    """
    rng = np.random.default_rng(seed)

    def difference(indices: IntArray) -> float:
        return float(statistic_a(indices) - statistic_b(indices))

    observed = difference(np.arange(labels.shape[0], dtype=np.int64))
    draws = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        sample = stratified_indices(labels, rng)
        draws[index] = difference(sample)
    finite = draws[np.isfinite(draws)]
    return BootstrapResult(
        value=observed,
        interval=percentile_interval(finite, level),
        resamples=resamples,
        standard_error=float(np.std(finite, ddof=1)) if finite.shape[0] > 1 else 0.0,
    )


def normal_interval(value: float, standard_error: float, level: float = DEFAULT_LEVEL) -> Interval:
    from scipy import stats

    z = float(stats.norm.ppf(1.0 - (1.0 - level) / 2.0))
    return Interval(value - z * standard_error, value + z * standard_error, level)
