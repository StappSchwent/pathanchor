"""Shared static types.

Ref: Sec. III-A (notation), Sec. III-H (reported quantities).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

type FloatArray = NDArray[np.float64]
type IntArray = NDArray[np.int64]
type BoolArray = NDArray[np.bool_]

# The released graph holds 7,431,905 edges, so its endpoint and weight arrays use
# the narrowest dtype that cannot overflow at that size; every identifier and
# every endpoint index is far below 2**31.
type GraphArray = NDArray[np.int32]
type CodeArray = NDArray[np.int16]
type WeightArray = NDArray[np.float32]

# A patient identifier is an opaque site-local code; nothing else is retained.
type PatientId = str


@dataclass(frozen=True, slots=True)
class Interval:
    """A closed interval with a nominal level, used for every reported CI."""

    low: float
    high: float
    level: float = 0.95

    def width(self) -> float:
        return self.high - self.low

    def contains(self, value: float) -> bool:
        return self.low <= value <= self.high

    def as_pair(self) -> tuple[float, float]:
        return (self.low, self.high)


@dataclass(frozen=True, slots=True)
class Estimate:
    """A point estimate with an interval and the number of units behind it."""

    value: float
    interval: Interval
    n: int


@dataclass(frozen=True, slots=True)
class NamedValue:
    """A labelled scalar, so report rows never travel as bare tuples."""

    label: str
    value: float
