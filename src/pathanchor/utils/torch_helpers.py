"""Small torch helpers shared by the modules.

Ref: Sec. III-C (path coder), Sec. III-A (multimodal evidence record).
"""

from __future__ import annotations

from torch import Tensor, nn


def reset_linear(layer: nn.Linear) -> None:
    """Xavier-uniform initialisation for a linear layer, applied in place."""
    nn.init.xavier_uniform_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


def masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    """Mean over the rows a mask selects, safe when a row selects nothing."""
    weights = mask.unsqueeze(-1) if mask.dim() == values.dim() - 1 else mask
    numerator = (values * weights).sum(dim=1)
    denominator = weights.sum(dim=1).clamp(min=1.0)
    result: Tensor = numerator / denominator
    return result


def inverse_softplus(value: float) -> float:
    """The parameter whose softplus is ``value``, for a scale initialised to one."""
    import math

    return value + math.log(-math.expm1(-value))
