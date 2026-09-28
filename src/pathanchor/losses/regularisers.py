"""Regularisers on the retained path set.

Ref: Sec. III-C (a path is kept when its impact reaches the threshold set by
epsilon), Table II Panel B (the retained set is what the audit replays).

Two properties are worth paying for. The retained set should be small, because a
justification that cites every route explains nothing; and a path that sits just
below the retention threshold is a liability, because a small change to the
graph would flip it, so the margin term pushes near-threshold paths decisively to
one side.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class RetentionSparsityLoss(nn.Module):
    """Differentiable count of the paths that cross the retention threshold."""

    def __init__(self, epsilon: float, weight: float = 1.0, target: float = 8.0) -> None:
        super().__init__()
        self.epsilon = float(epsilon)
        self.weight = float(weight)
        self.target = float(target)

    def forward(self, scores: Tensor, mask: Tensor) -> Tensor:
        if scores.numel() == 0:
            return torch.zeros((), dtype=torch.float32, requires_grad=True)
        soft_retained = torch.sigmoid((scores.abs() - self.epsilon) * 20.0) * mask
        count = soft_retained.sum(dim=-1).mean()
        loss: Tensor = self.weight * torch.relu(count - self.target)
        return loss


class RetentionMarginLoss(nn.Module):
    """Hinge that pushes each path away from the retention boundary."""

    def __init__(self, epsilon: float, margin: float = 0.1, weight: float = 1.0) -> None:
        super().__init__()
        self.epsilon = float(epsilon)
        self.margin = float(margin)
        self.weight = float(weight)

    def forward(self, scores: Tensor, mask: Tensor) -> Tensor:
        if scores.numel() == 0:
            return torch.zeros((), dtype=torch.float32, requires_grad=True)
        slack = torch.relu(self.epsilon + self.margin - scores.abs()) * mask
        denominator = torch.clamp(mask.sum(), min=1.0)
        loss: Tensor = self.weight * slack.sum() / denominator
        return loss


class SignatureConsistencyLoss(nn.Module):
    """Penalty on paths whose relation sequence is not shared with any other
    patient in the batch.

    A route that only ever appears once cannot be cross-checked, so the term
    nudges the scorer toward routes the auditor can find again.
    """

    def __init__(self, weight: float = 0.1) -> None:
        super().__init__()
        self.weight = float(weight)

    def forward(self, signature_ids: Tensor, weights: Tensor) -> Tensor:
        if signature_ids.numel() == 0:
            return torch.zeros((), dtype=torch.float32, requires_grad=True)
        unique, inverse, counts = torch.unique(
            signature_ids, return_inverse=True, return_counts=True
        )
        del unique
        multiplicity = counts[inverse].to(weights.dtype)
        per_path = weights * torch.rsqrt(multiplicity)
        loss: Tensor = self.weight * (1.0 - torch.mean(per_path))
        return loss
