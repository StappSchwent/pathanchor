"""Split conformal calibration and the abstention rule.

Ref: Sec. III-E and Eq. (2).

    V_i = |Y_i - q_hat(x_i)|,  q_hat a monotone transform of S_theta fitted on
    the training split
    Q(p) = [q_hat(x_p) - Q_{1-alpha}, q_hat(x_p) + Q_{1-alpha}] intersect [0, 1]

with Q_{1-alpha} the ceil((n+1)(1-alpha))-th smallest nonconformity score. The
layer changes how the score is read, not how it ranks, which is why it reports
no AUROC movement in Table II Panel A.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pathanchor.utils.numerics import quantile_index
from pathanchor.utils.types import FloatArray, IntArray
from pathanchor.utils.validation import require_binary, require_probabilities


@dataclass(frozen=True, slots=True)
class MonotoneTransform:
    """A non-decreasing step map, fitted by isotonic regression."""

    knots: FloatArray
    fitted: FloatArray

    def __call__(self, values: FloatArray) -> FloatArray:
        if self.knots.shape[0] == 0:
            return np.full(values.shape, float(np.mean(self.fitted)) if self.fitted.size else 0.5)
        slot = np.searchsorted(self.knots, values, side="right") - 1
        slot = np.clip(slot, 0, self.fitted.shape[0] - 1)
        mapped: FloatArray = self.fitted[slot]
        return np.clip(mapped, 0.0, 1.0)

    def monotone_check(self) -> bool:
        return bool(np.all(np.diff(self.fitted) >= -1e-12))


def fit_isotonic(values: FloatArray, targets: FloatArray) -> MonotoneTransform:
    """Pool-adjacent-violators isotonic regression.

    Written out rather than delegated so the fitted map can be compared against an
    independent solver, and so the monotonicity the conformal guarantee needs is
    a property of this code rather than of a library's default behaviour.
    """
    if values.shape[0] != targets.shape[0]:
        raise ValueError("isotonic fit needs equally long value and target arrays")
    if values.shape[0] == 0:
        return MonotoneTransform(np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64))
    order = np.argsort(values, kind="stable")
    xs = values[order]
    ys = targets[order]
    block_value: list[float] = []
    block_weight: list[float] = []
    block_end: list[int] = []
    index = 0
    n = xs.shape[0]
    while index < n:
        stop = index
        while stop + 1 < n and xs[stop + 1] == xs[index]:
            stop += 1
        weight = float(stop - index + 1)
        mean = float(np.mean(ys[index : stop + 1]))
        block_value.append(mean)
        block_weight.append(weight)
        block_end.append(stop)
        while len(block_value) >= 2 and block_value[-2] > block_value[-1]:
            value_right = block_value.pop()
            weight_right = block_weight.pop()
            end_right = block_end.pop()
            value_left = block_value.pop()
            weight_left = block_weight.pop()
            block_end.pop()
            merged = (value_left * weight_left + value_right * weight_right) / (
                weight_left + weight_right
            )
            block_value.append(merged)
            block_weight.append(weight_left + weight_right)
            block_end.append(end_right)
        index = stop + 1
    fitted_sorted = np.empty(n, dtype=np.float64)
    start = 0
    for value, end in zip(block_value, block_end, strict=True):
        fitted_sorted[start : end + 1] = value
        start = end + 1
    unique_x = np.unique(xs)
    first_positions = np.searchsorted(xs, unique_x, side="left")
    return MonotoneTransform(
        np.asarray(unique_x, dtype=np.float64),
        np.asarray(fitted_sorted[first_positions], dtype=np.float64),
    )


@dataclass(frozen=True, slots=True)
class PredictionSet:
    low: float
    high: float

    @property
    def width(self) -> float:
        return self.high - self.low

    def contains(self, value: float) -> bool:
        return self.low <= value <= self.high


@dataclass(slots=True)
class SplitConformal:
    """Split conformal over a monotone transform of the path-marginal score."""

    alpha: float = 0.10
    tau_abs: float = 0.35
    transform: MonotoneTransform | None = None
    quantile: float = 1.0
    n_calibration: int = 0
    nonconformity: FloatArray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))

    def fit(
        self,
        train_scores: FloatArray,
        train_labels: IntArray,
        calibration_scores: FloatArray,
        calibration_labels: IntArray,
    ) -> SplitConformal:
        require_binary(train_labels, "training labels")
        require_binary(calibration_labels, "calibration labels")
        if train_scores.shape[0] != train_labels.shape[0]:
            raise ValueError("training scores and labels disagree in length")
        if calibration_scores.shape[0] != calibration_labels.shape[0]:
            raise ValueError("calibration scores and labels disagree in length")
        self.transform = fit_isotonic(train_scores, train_labels.astype(np.float64))
        mapped = self.transform(calibration_scores)
        self.nonconformity = np.abs(calibration_labels.astype(np.float64) - mapped)
        self.n_calibration = int(self.nonconformity.shape[0])
        self.quantile = quantile_value(self.nonconformity, self.alpha)
        return self

    def apply_transform(self, scores: FloatArray) -> FloatArray:
        if self.transform is None:
            raise RuntimeError("the conformal layer has not been fitted")
        return self.transform(scores)

    def interval(self, scores: FloatArray) -> tuple[FloatArray, FloatArray]:
        if self.transform is None:
            raise RuntimeError("the conformal layer has not been fitted")
        centre = self.transform(scores)
        low = np.clip(centre - self.quantile, 0.0, 1.0)
        high = np.clip(centre + self.quantile, 0.0, 1.0)
        return (low, high)

    def prediction_sets(self, scores: FloatArray) -> list[PredictionSet]:
        low, high = self.interval(scores)
        return [
            PredictionSet(float(low[index]), float(high[index])) for index in range(low.shape[0])
        ]

    def abstains(self, prediction_sets: list[PredictionSet]) -> list[bool]:
        return [item.width > self.tau_abs for item in prediction_sets]

    def coverage(self, scores: FloatArray, labels: IntArray) -> float:
        require_binary(labels, "coverage labels")
        low, high = self.interval(scores)
        inside = (labels.astype(np.float64) >= low) & (labels.astype(np.float64) <= high)
        return float(np.mean(inside))

    def mean_width(self, scores: FloatArray) -> float:
        low, high = self.interval(scores)
        return float(np.mean(high - low))

    def abstention_rate(self, scores: FloatArray) -> float:
        sets = self.prediction_sets(scores)
        flags = self.abstains(sets)
        return float(np.mean(flags)) if flags else 0.0

    def nominal_level(self) -> float:
        return 1.0 - self.alpha


def quantile_value(values: FloatArray, alpha: float) -> float:
    """The ceil((n + 1)(1 - alpha))-th smallest value of ``values``.

    The finite-sample correction is the whole point of the split-conformal
    guarantee: dropping the +1 gives the empirical quantile, which under-covers at
    small n.
    """
    if values.shape[0] == 0:
        return float("inf")
    ordered = np.sort(values)
    index = quantile_index(int(ordered.shape[0]), 1.0 - alpha)
    return float(ordered[index - 1])


def empirical_coverage(intervals: list[PredictionSet], labels: IntArray) -> float:
    require_binary(labels, "coverage labels")
    if len(intervals) != labels.shape[0]:
        raise ValueError("intervals and labels disagree in length")
    inside = [
        1.0 if item.contains(float(label)) else 0.0
        for item, label in zip(intervals, labels.tolist(), strict=True)
    ]
    return float(np.mean(inside)) if inside else 0.0


def validate_conformal_inputs(scores: FloatArray, labels: IntArray, name: str) -> None:
    require_binary(labels, f"{name} labels")
    if scores.shape[0] != labels.shape[0]:
        raise ValueError(f"{name} scores and labels disagree in length")
    require_probabilities(np.clip(scores, 0.0, 1.0), f"{name} scores")
