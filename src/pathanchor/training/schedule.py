"""Learning-rate schedules.

Ref: Sec. IV-G (the reported training budget), KICKOFF R3 (schedule defaults
follow the reported configuration).

Both schedules share one base class that owns the step counter; the base class
implements ``get_lr`` so the step and cosine branches are the only overrides and
neither can inherit an unimplemented method.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class ScheduleConfig:
    name: str = "cosine"
    warmup_steps: int = 100
    total_steps: int = 2000
    min_ratio: float = 0.05

    def validate(self) -> None:
        if self.total_steps <= 0:
            raise ValueError("total_steps must be positive")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if not 0.0 <= self.min_ratio <= 1.0:
            raise ValueError("min_ratio must lie in [0, 1]")


class WarmupSchedule(torch.optim.lr_scheduler.LRScheduler):
    """Base schedule: linear warmup followed by a shape owned by the subclass."""

    def __init__(
        self,
        optimiser: torch.optim.Optimizer,
        config: ScheduleConfig,
        last_epoch: int = -1,
    ) -> None:
        config.validate()
        self.config = config
        self.base_learning_rate = float(optimiser.param_groups[0]["lr"])
        super().__init__(optimiser, last_epoch=last_epoch)

    def _decay_ratio(self, step: int) -> float:
        raise NotImplementedError

    def get_lr(self) -> list[float]:  # type: ignore[override]
        step = max(self.last_epoch, 0)
        warmup = self.config.warmup_steps
        if warmup > 0 and step < warmup:
            ratio = float(step) / float(max(warmup, 1))
        else:
            ratio = self._decay_ratio(step - warmup)
        return [self.base_learning_rate * ratio for _ in self.optimizer.param_groups]


class CosineSchedule(WarmupSchedule):
    """Cosine decay to ``min_ratio`` of the base rate."""

    def _decay_ratio(self, step: int) -> float:
        span = max(self.config.total_steps - self.config.warmup_steps, 1)
        clipped = min(max(step, 0), span)
        cosine = 0.5 * (1.0 + math.cos(math.pi * clipped / span))
        return self.config.min_ratio + (1.0 - self.config.min_ratio) * cosine


class LinearSchedule(WarmupSchedule):
    """Linear decay to ``min_ratio`` of the base rate."""

    def _decay_ratio(self, step: int) -> float:
        span = max(self.config.total_steps - self.config.warmup_steps, 1)
        clipped = min(max(step, 0), span)
        fraction = float(clipped) / float(span)
        return 1.0 - (1.0 - self.config.min_ratio) * fraction


class ConstantSchedule(WarmupSchedule):
    """Warmup then a flat rate."""

    def _decay_ratio(self, step: int) -> float:
        del step
        return 1.0


def build_schedule(
    optimiser: torch.optim.Optimizer,
    config: ScheduleConfig,
) -> WarmupSchedule:
    if config.name == "cosine":
        return CosineSchedule(optimiser, config)
    if config.name == "linear":
        return LinearSchedule(optimiser, config)
    if config.name == "constant":
        return ConstantSchedule(optimiser, config)
    raise ValueError(f"unknown schedule '{config.name}'")


def schedule_trace(
    schedule: WarmupSchedule,
    steps: int,
) -> list[float]:
    """The rate at each step, for a report that does not have to re-derive it."""
    return [float(schedule.get_lr()[0]) for _ in range(steps)]
