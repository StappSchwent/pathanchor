"""Mixed precision and exponential moving averages.

Ref: KICKOFF R3 (precision is a configurable training parameter), Sec. IV-G (the
reported training cost).

Precision is opt-in rather than defaulted, because the manuscript reports no
precision and a silently-downcast run would not reproduce a full-precision one.
The moving average lives beside it because both are wrapper concerns the trainer
applies around the same forward pass.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class PrecisionConfig:
    enabled: bool = False
    dtype: str = "bf16"
    initial_scale: float = 1024.0

    def resolve(self) -> torch.dtype:
        if self.dtype in {"bf16", "bfloat16"}:
            return torch.bfloat16
        if self.dtype in {"fp16", "float16", "half"}:
            return torch.float16
        raise ValueError(f"unknown precision dtype '{self.dtype}'")


def autocast_context(
    config: PrecisionConfig,
    device: torch.device,
) -> AbstractContextManager[Any]:
    if not config.enabled:
        return nullcontext()
    if device.type != "cuda":
        # Autocast on a host device would change numerics without a speed signal,
        # so a CPU run stays in full precision and says so through the ledger.
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=config.resolve())


def make_grad_scaler(config: PrecisionConfig, device: torch.device) -> Any:
    """A scaler only where it is meaningful, so a host run is not silently scaled."""
    if not config.enabled or config.resolve() is not torch.float16 or device.type != "cuda":
        return None
    return torch.amp.GradScaler("cuda", init_scale=config.initial_scale)  # type: ignore[attr-defined]


class ExponentialMovingAverage:
    """A shadow copy of the parameters, updated after each optimiser step."""

    def __init__(self, module: nn.Module, decay: float = 0.999) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError("the decay must lie strictly inside (0, 1)")
        self.decay = float(decay)
        self.shadow: dict[str, torch.Tensor] = {
            name: parameter.detach().clone()
            for name, parameter in module.named_parameters()
            if parameter.requires_grad
        }
        self.updates = 0

    @torch.no_grad()
    def update(self, module: nn.Module) -> None:
        for name, parameter in module.named_parameters():
            if name not in self.shadow:
                continue
            self.shadow[name].mul_(self.decay).add_(parameter.detach(), alpha=1.0 - self.decay)
        self.updates += 1

    def copy_to(self, module: nn.Module) -> None:
        with torch.no_grad():
            for name, parameter in module.named_parameters():
                if name in self.shadow:
                    parameter.copy_(self.shadow[name])

    def state(self) -> Mapping[str, torch.Tensor]:
        return self.shadow


def half_precision_supported(device: torch.device) -> bool:
    return device.type == "cuda" and bool(torch.cuda.is_bf16_supported())


def iter_parameters_shadow(module: nn.Module) -> Iterator[tuple[str, torch.Tensor]]:
    for name, parameter in module.named_parameters():
        if parameter.requires_grad:
            yield (name, parameter.detach())
