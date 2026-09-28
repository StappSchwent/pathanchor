"""Optimiser construction and parameter groups.

Ref: Sec. IV-G (reporting the training budget), KICKOFF R3 (the paper's reported
optimiser settings are the defaults).

The manuscript reports a training budget in GPU-hours but no optimiser, learning
rate or weight-decay value, so the defaults below are declared engineering
defaults and appear in the deviation register. Parameter groups separate the
scoring head from the graph-side modules so they can be decayed differently
without changing either schedule.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class OptimiserConfig:
    name: str = "adamw"
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-2
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1.0e-8
    no_decay_on_bias: bool = True


DECAY_FREE_SUFFIXES: tuple[str, ...] = ("bias", "norm.weight", "norm.bias", "scale")


def decay_free(name: str) -> bool:
    return any(name.endswith(suffix) for suffix in DECAY_FREE_SUFFIXES)


def parameter_groups(
    module: nn.Module,
    config: OptimiserConfig,
) -> list[dict[str, object]]:
    """Split parameters into decayed and decay-free groups."""
    decayed: list[nn.Parameter] = []
    free: list[nn.Parameter] = []
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            continue
        if config.no_decay_on_bias and decay_free(name):
            free.append(parameter)
        else:
            decayed.append(parameter)
    groups: list[dict[str, object]] = []
    if decayed:
        groups.append({"params": decayed, "weight_decay": config.weight_decay})
    if free:
        groups.append({"params": free, "weight_decay": 0.0})
    if not groups:
        raise ValueError("the module has no trainable parameters")
    return groups


def build_optimiser(module: nn.Module, config: OptimiserConfig) -> torch.optim.Optimizer:
    groups = parameter_groups(module, config)
    if config.name == "adamw":
        return torch.optim.AdamW(
            groups,
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.epsilon,
        )
    if config.name == "adam":
        return torch.optim.Adam(
            groups,
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.epsilon,
        )
    if config.name == "sgd":
        return torch.optim.SGD(groups, lr=config.learning_rate, momentum=config.beta1)
    raise ValueError(f"unknown optimiser '{config.name}'")


def learning_rates(optimiser: torch.optim.Optimizer) -> dict[str, float]:
    return {
        f"group_{index}": float(group["lr"]) for index, group in enumerate(optimiser.param_groups)
    }


def gradient_norm(parameters: Iterable[nn.Parameter]) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        total += float(torch.sum(parameter.grad.detach() ** 2))
    return float(total**0.5)


def clip_gradients(
    parameters: Iterable[nn.Parameter],
    max_norm: float,
) -> float:
    """Clip in place and return the pre-clip norm."""
    collected: Sequence[nn.Parameter] = list(parameters)
    norm: float = float(torch.nn.utils.clip_grad_norm_(collected, max_norm).detach())
    return norm
