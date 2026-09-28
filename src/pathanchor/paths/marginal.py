"""Path marginalisation over the admissible set of one treatment option.

Ref: Algorithm 1 line 6, S_theta(p, a) = log sum_{pi in P''(a)} exp s_theta(pi).
Ref: Table II Panel B, "mean-pool paths (no log-sum-exp)".

The marginal is a log-sum-exp, so a single strong path can carry an option while
the alternative aggregation averages it away with the rest of the set. The
alternative is implemented here rather than configured, because the comparison
between the two is one of the reported ablations.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from pathanchor.utils.types import FloatArray

if TYPE_CHECKING:
    from torch import Tensor

EMPTY_OPTION_SCORE: float = -60.0


def log_sum_exp_marginal(scores: Sequence[float]) -> float:
    """S_theta(p, a) for one option."""
    if len(scores) == 0:
        return EMPTY_OPTION_SCORE
    values = np.asarray(scores, dtype=np.float64)
    peak = float(np.max(values))
    return float(peak + np.log(float(np.sum(np.exp(values - peak)))))


def mean_pool_marginal(scores: Sequence[float]) -> float:
    if len(scores) == 0:
        return EMPTY_OPTION_SCORE
    return float(np.mean(np.asarray(scores, dtype=np.float64)))


def top_m_mean_marginal(scores: Sequence[float], m: int) -> float:
    if len(scores) == 0 or m <= 0:
        return EMPTY_OPTION_SCORE
    values = np.sort(np.asarray(scores, dtype=np.float64))[::-1]
    return float(np.mean(values[:m]))


def option_scores(
    scores_by_option: Mapping[str, Sequence[float]],
    aggregator: str = "logsumexp",
) -> dict[str, float]:
    """Aggregate per-path scores into one score per treatment option."""
    out: dict[str, float] = {}
    for option, values in scores_by_option.items():
        if aggregator == "logsumexp":
            out[option] = log_sum_exp_marginal(values)
        elif aggregator == "mean":
            out[option] = mean_pool_marginal(values)
        else:
            raise ValueError(f"unknown path aggregator '{aggregator}'")
    return out


def argmax_over_feasible(
    scores: Mapping[str, float],
    feasible: Sequence[str],
) -> tuple[str, float, bool]:
    """arg max over C(p) of Eq. (1).

    Returns the option, its score, and whether the feasible set was non-empty.
    An empty feasible set is a structural failure of the guideline parse, not a
    modelling outcome, so it is surfaced rather than defaulted away.
    """
    candidates = [option for option in feasible if option in scores]
    if not candidates:
        return ("", EMPTY_OPTION_SCORE, False)
    best = max(candidates, key=lambda option: (scores[option], option))
    return (best, float(scores[best]), True)


def unconstrained_argmax(scores: Mapping[str, float]) -> tuple[str, float]:
    """The same selection without the feasible set, for the Panel B comparison."""
    if not scores:
        return ("", EMPTY_OPTION_SCORE)
    best = max(scores, key=lambda option: (scores[option], option))
    return (best, float(scores[best]))


@dataclass(frozen=True, slots=True)
class ProjectionOutcome:
    option: str
    score: float
    feasible: bool
    restricted: bool

    def label(self) -> str:
        return self.option if self.feasible else "no feasible option"


def project(
    scores: Mapping[str, float],
    feasible: Sequence[str],
    enabled: bool = True,
) -> ProjectionOutcome:
    """Apply the guideline projection, or deliberately skip it.

    ``enabled=False`` reproduces the leave-one-out arm of Table II Panel B, where
    removing the projection lifts discrimination to 0.772 and raises guideline
    discordance from zero to 18.9%.
    """
    if not enabled:
        option, value = unconstrained_argmax(scores)
        return ProjectionOutcome(
            option=option, score=value, feasible=bool(option), restricted=False
        )
    option, value, ok = argmax_over_feasible(scores, feasible)
    return ProjectionOutcome(option=option, score=value, feasible=ok, restricted=True)


def marginal_gap(scores: Mapping[str, float], feasible: Sequence[str]) -> float:
    """Distance between the feasible choice and the unrestricted best."""
    if not scores:
        return 0.0
    _, restricted_value, ok = argmax_over_feasible(scores, feasible)
    if not ok:
        return 0.0
    _, free_value = unconstrained_argmax(scores)
    return float(free_value - restricted_value)


def as_array(scores: Mapping[str, float], order: Sequence[str]) -> FloatArray:
    return np.asarray(
        [float(scores.get(option, EMPTY_OPTION_SCORE)) for option in order], dtype=np.float64
    )


def segment_log_sum_exp(scores: Tensor, segments: Tensor, n_segments: int) -> Tensor:
    """Per-segment log-sum-exp over a flat path-score tensor.

    A patient's proposed paths for every feasible option travel as one flat batch
    with a segment index per path, so the marginalisation is one scatter-reduce
    rather than a Python loop over options and a tensor allocation per option.
    """
    import torch

    values: Tensor = torch.as_tensor(scores)
    index: Tensor = torch.as_tensor(segments, dtype=torch.long)
    if values.numel() == 0:
        return torch.full((n_segments,), EMPTY_OPTION_SCORE, dtype=torch.float32)
    totals = torch.full((n_segments,), EMPTY_OPTION_SCORE, dtype=values.dtype)
    for segment in range(n_segments):
        selected = values[index == segment]
        if selected.numel() == 0:
            continue
        totals[segment] = torch.logsumexp(selected, dim=0)
    return totals


def segment_mean(scores: Tensor, segments: Tensor, n_segments: int) -> Tensor:
    """The alternative aggregation of Table II Panel B, over the same segments."""
    import torch

    values: Tensor = torch.as_tensor(scores)
    index: Tensor = torch.as_tensor(segments, dtype=torch.long)
    totals = torch.full((n_segments,), EMPTY_OPTION_SCORE, dtype=values.dtype)
    for segment in range(n_segments):
        selected = values[index == segment]
        if selected.numel() == 0:
            continue
        totals[segment] = torch.mean(selected)
    return totals
