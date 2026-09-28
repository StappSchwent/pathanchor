"""Integer contingency allocation that holds every reported marginal exactly.

Ref: Sec. III-F (cohort composition and the 60/20/20 partition), Table III
(prespecified subgroup counts on the test split).

The cohort description fixes several margins at once -- molecular class crossed
with treatment class, site, sex and age band -- and the test split fixes them
again on its own 856 records. A single margin can be met by rounding, but several
cannot: rounding a table to satisfy one margin moves another off target, and
where the margins meet the greedy repair stalls a unit short.

So the allocation is a small integer program. A dense table is first fitted by
iterative proportional fitting; the integer table is then the solution that
minimises the deviation from that fit subject to every margin holding exactly and
every cell staying within one unit of its fitted value. That last bound is what
makes it a controlled rounding rather than an arbitrary feasible point, and it is
what keeps the result stable when the margins change.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from pathanchor.utils.types import IntArray

_DENSITY_FLOOR = 1e-6


@dataclass(frozen=True, slots=True)
class MarginSpec:
    """One reported marginal, over a subset of the table's axes."""

    name: str
    axes: tuple[int, ...]
    target: IntArray

    def validate(self, shape: tuple[int, ...]) -> None:
        if len(self.target.shape) != len(self.axes):
            raise ValueError(f"margin '{self.name}' has a mismatched axis/target rank")
        for axis, width in zip(self.axes, self.target.shape, strict=True):
            if shape[axis] != width:
                raise ValueError(
                    f"margin '{self.name}' expects {width} levels on axis {axis}, "
                    f"the table has {shape[axis]}"
                )
        if int(np.min(self.target)) < 0:
            raise ValueError(f"margin '{self.name}' carries a negative target")


def _complement(ndim: int, axes: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(axis for axis in range(ndim) if axis not in axes)


def margin_sums(table: np.ndarray, axes: tuple[int, ...]) -> np.ndarray:
    totals: np.ndarray = table.sum(axis=_complement(table.ndim, axes))
    return totals


def _rescale(table: np.ndarray, axes: tuple[int, ...], factor: np.ndarray) -> np.ndarray:
    scaled: np.ndarray = table * np.expand_dims(factor, axis=_complement(table.ndim, axes))
    return scaled


def fit_table(
    shape: tuple[int, ...],
    margins: Sequence[MarginSpec],
    total: int,
    iterations: int = 500,
    tolerance: float = 1e-11,
    allowed: np.ndarray | None = None,
) -> np.ndarray:
    """Iterative proportional fitting over a table of the given shape."""
    for margin in margins:
        margin.validate(shape)
    if int(np.sum(margins[0].target)) != total:
        raise ValueError("the first margin does not sum to the requested total")
    table = np.ones(shape, dtype=np.float64) if allowed is None else allowed.astype(np.float64)
    for _ in range(iterations):
        largest = 0.0
        for margin in margins:
            current = margin_sums(table, margin.axes)
            factor = np.where(
                current > _DENSITY_FLOOR,
                margin.target.astype(np.float64) / np.maximum(current, _DENSITY_FLOOR),
                0.0,
            )
            table = _rescale(table, margin.axes, factor)
            if allowed is not None:
                table = np.where(allowed, table, 0.0)
            current = margin_sums(table, margin.axes)
            largest = max(largest, float(np.max(np.abs(current - margin.target))))
        if largest < tolerance:
            break
    return table * (total / float(np.sum(table)))


def margin_constraint_matrix(
    shape: tuple[int, ...],
    margins: Sequence[MarginSpec],
) -> np.ndarray:
    """Rows selecting the cells of one margin slice, for the integer program."""
    size = int(np.prod(shape))
    rows: list[np.ndarray] = []
    for margin in margins:
        flat_target = int(np.prod(margin.target.shape))
        slices = _slice_map(shape, margin).ravel()
        for cell in range(flat_target):
            row = np.zeros(size, dtype=np.float64)
            row[slices == cell] = 1.0
            rows.append(row)
    return np.vstack(rows) if rows else np.zeros((0, size), dtype=np.float64)


def _slice_map(shape: tuple[int, ...], margin: MarginSpec) -> np.ndarray:
    """Flat index of each table cell inside its margin slice."""
    widths = [int(value) for value in margin.target.shape]
    strides = [1] * len(widths)
    for position in range(len(widths) - 2, -1, -1):
        strides[position] = strides[position + 1] * widths[position + 1]
    index = np.zeros(shape, dtype=np.int64)
    for position, axis in enumerate(margin.axes):
        spread = [1] * len(shape)
        spread[axis] = shape[axis]
        coordinate = np.arange(shape[axis], dtype=np.int64).reshape(spread)
        index = index + coordinate * strides[position]
    return index


def controlled_round(
    fitted: np.ndarray,
    margins: Sequence[MarginSpec],
    total: int,
    allowed: np.ndarray | None = None,
    windows: Sequence[int] = (1, 2, 4, 8, 16),
) -> IntArray:
    """Round ``fitted`` to integers while holding every margin at its target.

    The objective is the summed absolute deviation from the fitted table; the
    deviation variables are auxiliary and continuous, while the cell counts are
    integral. Cells outside ``allowed`` are pinned to zero.

    The window bounds each cell to within a fixed distance of its fitted value, so
    the result stays a controlled rounding rather than an arbitrary feasible
    point. Tight windows are tried first and widened only when the margins are
    jointly infeasible inside them, which happens when two margins disagree about
    a cell by more than the window allows.
    """
    shape = fitted.shape
    size = int(np.prod(shape))
    flat = fitted.ravel()
    matrix = margin_constraint_matrix(shape, margins)
    target = np.concatenate([margin.target.ravel().astype(np.float64) for margin in margins])
    identity = np.eye(size, dtype=np.float64)
    deviation = LinearConstraint(
        np.vstack([np.hstack([identity, -identity]), np.hstack([-identity, -identity])]),
        np.concatenate([np.full(size, -np.inf), np.full(size, -np.inf)]),
        np.concatenate([flat, -flat]),
    )
    equality = LinearConstraint(
        np.hstack([matrix, np.zeros((matrix.shape[0], size), dtype=np.float64)]),
        target,
        target,
    )
    cost = np.concatenate([np.zeros(size, dtype=np.float64), np.ones(size, dtype=np.float64)])
    integrality = np.concatenate([np.ones(size, dtype=np.int8), np.zeros(size, dtype=np.int8)])
    last_message = "no window was attempted"
    for window in windows:
        lower = np.maximum(np.floor(flat) - window, 0.0)
        upper = np.minimum(np.ceil(flat) + window, float(total))
        if allowed is not None:
            mask = allowed.ravel()
            lower = np.where(mask, lower, 0.0)
            upper = np.where(mask, upper, 0.0)
        result = milp(
            c=cost,
            constraints=[equality, deviation],
            integrality=integrality,
            bounds=Bounds(
                np.concatenate([lower, np.zeros(size, dtype=np.float64)]),
                np.concatenate([upper, np.full(size, float(total), dtype=np.float64)]),
            ),
            options={"presolve": True, "disp": False},
        )
        if result.success and result.x is not None:
            solution: IntArray = np.rint(result.x[:size]).astype(np.int64).reshape(shape)
            return solution
        last_message = f"window {window}: {result.message}"
    raise RuntimeError(f"the marginal allocation did not solve ({last_message})")


def build_table(
    shape: tuple[int, ...],
    margins: Sequence[MarginSpec],
    total: int,
    iterations: int = 500,
    allowed: np.ndarray | None = None,
) -> IntArray:
    fitted = fit_table(shape, margins, total, iterations=iterations, allowed=allowed)
    table = controlled_round(fitted, margins, total, allowed=allowed)
    deviations = margin_error(table, margins)
    if any(value != 0 for value in deviations.values()):
        raise RuntimeError(f"the marginal allocation is not exact: {deviations}")
    return table


def margin_error(table: IntArray, margins: Sequence[MarginSpec]) -> dict[str, int]:
    """Largest absolute deviation per margin, for the allocation assertion."""
    return {
        margin.name: int(np.max(np.abs(margin_sums(table, margin.axes) - margin.target)))
        for margin in margins
    }


def split_proportionally(
    table: IntArray,
    first_total: int,
) -> tuple[IntArray, IntArray]:
    """Split a table into two integer tables of fixed sizes.

    Each cell is divided by largest remainder so both parts are integral and the
    pair sums back to the original cell, which keeps every margin of the whole
    intact in the union of the two parts.
    """
    if first_total < 0 or first_total > int(np.sum(table)):
        raise ValueError("the first part's size must lie inside the table total")
    flat = table.ravel().astype(np.float64)
    grand = float(np.sum(flat))
    exact = flat * (first_total / grand)
    first = np.floor(exact).astype(np.int64)
    shortfall = first_total - int(np.sum(first))
    if shortfall > 0:
        order = np.argsort(-(exact - first), kind="stable")
        first[order[:shortfall]] += 1
    second = table.ravel() - first
    if int(np.sum(first)) != first_total:
        raise RuntimeError("the first part does not hold the requested size")
    return (first.reshape(table.shape), second.reshape(table.shape))


def table_cells(
    table: IntArray,
    labels: Sequence[Sequence[str]],
) -> list[tuple[tuple[str, ...], int]]:
    """Expand a fitted table into (level names, count) pairs, dropping empties."""
    out: list[tuple[tuple[str, ...], int]] = []
    for index in np.ndindex(*table.shape):
        count = int(table[index])
        if count <= 0:
            continue
        out.append((tuple(labels[axis][index[axis]] for axis in range(table.ndim)), count))
    return out
