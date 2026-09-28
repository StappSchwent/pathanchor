"""Input validation at the boundaries: file reads, CLI arguments, splits.

Ref: Sec. III-A (record structure), Sec. III-F (cohort definition).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from pathanchor.utils.types import FloatArray, IntArray


class ValidationError(ValueError):
    """Raised when an input violates a documented precondition."""


def require_fields(record: Mapping[str, Any], fields: Iterable[str], context: str) -> None:
    missing = [name for name in fields if name not in record]
    if missing:
        joined = ", ".join(sorted(missing))
        raise ValidationError(f"{context}: missing field(s) {joined}")


def require_probabilities(values: FloatArray, name: str) -> None:
    if values.ndim != 1:
        raise ValidationError(f"{name} must be one-dimensional")
    if not bool(np.all(np.isfinite(values))):
        raise ValidationError(f"{name} contains a non-finite entry")
    if float(np.min(values)) < -1e-9 or float(np.max(values)) > 1.0 + 1e-9:
        raise ValidationError(f"{name} leaves the unit interval")


def require_binary(labels: IntArray, name: str) -> None:
    if labels.ndim != 1:
        raise ValidationError(f"{name} must be one-dimensional")
    if not bool(np.all((labels == 0) | (labels == 1))):
        raise ValidationError(f"{name} must hold only 0 and 1")


def require_aligned(*arrays: Sequence[object], names: tuple[str, ...]) -> None:
    lengths = {len(array) for array in arrays}
    if len(lengths) != 1:
        raise ValidationError(f"{names} disagree in length: {sorted(lengths)}")


def require_probability(value: float, name: str, *, closed: bool = True) -> float:
    if not np.isfinite(value):
        raise ValidationError(f"{name} must be finite")
    if closed:
        if value < 0.0 or value > 1.0:
            raise ValidationError(f"{name} must lie in [0, 1]")
    elif value <= 0.0 or value >= 1.0:
        raise ValidationError(f"{name} must lie strictly inside (0, 1)")
    return float(value)


def require_positive(value: float, name: str) -> float:
    if not np.isfinite(value) or value <= 0.0:
        raise ValidationError(f"{name} must be strictly positive")
    return float(value)


def require_non_negative(value: float, name: str) -> float:
    if not np.isfinite(value) or value < 0.0:
        raise ValidationError(f"{name} must be non-negative")
    return float(value)


def require_choice(value: str, allowed: Sequence[str], name: str) -> str:
    if value not in allowed:
        options = ", ".join(sorted(allowed))
        raise ValidationError(f"{name} must be one of {{{options}}}, got '{value}'")
    return value


def require_simplex(values: FloatArray, name: str, tolerance: float = 1e-6) -> None:
    if values.ndim != 1:
        raise ValidationError(f"{name} must be one-dimensional")
    if float(np.min(values)) < -tolerance:
        raise ValidationError(f"{name} has a negative entry")
    if abs(float(np.sum(values)) - 1.0) > tolerance:
        raise ValidationError(f"{name} does not sum to one")


def require_disjoint(groups: Mapping[str, Iterable[str]], context: str) -> None:
    seen: dict[str, str] = {}
    for group_name, members in groups.items():
        for member in members:
            if member in seen:
                raise ValidationError(
                    f"{context}: '{member}' appears in both '{seen[member]}' and '{group_name}'"
                )
            seen[member] = group_name
