"""Response objective.

Ref: Sec. III-A (the target triple carries a response probability), Sec. III-E
(the calibrated value is derived from the path-marginal score).

The estimator predicts objective response under RECIST 1.1, so the supervised
term is a logistic loss on the path-marginal score. The manuscript does not state
a training objective for this head; the form below is a declared engineering
default and appears in the deviation register.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as functional


class ResponseLoss(nn.Module):
    """Binary logistic loss on a path-marginal score."""

    pos_weight: Tensor

    def __init__(self, weight: float = 1.0, positive_weight: float | None = None) -> None:
        super().__init__()
        self.weight = float(weight)
        self.register_buffer(
            "pos_weight",
            (
                torch.tensor([positive_weight], dtype=torch.float32)
                if positive_weight is not None
                else torch.tensor([1.0], dtype=torch.float32)
            ),
        )

    def forward(self, scores: Tensor, labels: Tensor) -> Tensor:
        targets = labels.to(scores.dtype)
        if self.pos_weight.item() != 1.0:
            raw = functional.binary_cross_entropy_with_logits(
                scores, targets, pos_weight=self.pos_weight.expand_as(scores), reduction="none"
            )
        else:
            raw = functional.binary_cross_entropy_with_logits(scores, targets, reduction="none")
        loss: Tensor = self.weight * torch.mean(raw)
        return loss


def balanced_positive_weight(labels: Tensor) -> float:
    """Ratio of negatives to positives, so a skewed fixture does not dominate."""
    positives = float(torch.sum(labels))
    negatives = float(labels.shape[0] - positives)
    if positives <= 0.0:
        return 1.0
    return max(1.0, negatives / positives)


def expected_response_probability(option_scores: Tensor) -> Tensor:
    """Soft-max weighted response probability over the feasible options.

    A hard arg max would stop the gradient at the selection step, so the
    selection is relaxed to a softmax over the scores of the admissible set.
    """
    weights = torch.softmax(option_scores, dim=-1)
    probabilities = torch.sigmoid(option_scores)
    pooled: Tensor = torch.sum(weights * probabilities, dim=-1)
    return pooled


def expected_response_loss(option_scores: Tensor, labels: Tensor) -> Tensor:
    probability = torch.clamp(expected_response_probability(option_scores), 1e-6, 1.0 - 1e-6)
    targets = labels.to(probability.dtype)
    terms = -(targets * torch.log(probability) + (1.0 - targets) * torch.log(1.0 - probability))
    loss: Tensor = torch.mean(terms)
    return loss
