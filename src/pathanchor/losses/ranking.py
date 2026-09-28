"""Ranking objectives over patients and over paths.

Ref: Sec. IV-A (the estimate is assessed by discrimination, AUROC), Sec. IV-D
(the treatment-conditional comparison `0.731 versus 0.765` is a ranking gap).

The manuscript reports discrimination as the primary utility axis and states that
hyperparameter tuning happens on the training split only; the pairwise form below
is a declared engineering default for that axis. The path-level term is a
listwise objective over one patient's proposed paths, which is what makes the
path scorer trainable before any marginalisation is applied.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class RankingConfig:
    patient_weight: float = 1.0
    path_weight: float = 0.5


class PairwiseRankingLoss(nn.Module):
    """RankNet over patient pairs whose labels differ."""

    def __init__(self, weight: float = 1.0) -> None:
        super().__init__()
        self.weight = float(weight)

    def forward(self, scores: Tensor, labels: Tensor) -> Tensor:
        difference = scores.unsqueeze(0) - scores.unsqueeze(1)
        targets = labels.unsqueeze(0) - labels.unsqueeze(1)
        positive = targets > 0
        if not bool(torch.any(positive)):
            return torch.zeros((), dtype=scores.dtype, requires_grad=scores.requires_grad)
        raw = torch.nn.functional.softplus(-difference[positive])
        loss: Tensor = self.weight * torch.mean(raw)
        return loss


class ListwisePathLoss(nn.Module):
    """Cross-entropy that promotes the paths with the highest edge-tier evidence.

    A path's target is its accumulated edge weight normalised inside the
    patient's proposal set, so the scorer learns an ordering that is defined by
    the graph rather than by the outcome.
    """

    def __init__(self, weight: float = 1.0) -> None:
        super().__init__()
        self.weight = float(weight)

    def forward(self, scores: Tensor, path_weights: Tensor, mask: Tensor) -> Tensor:
        """Cross-entropy between the score distribution and the evidence weights.

        A row whose mask selects nothing carries no path to rank, so it is excluded
        from the mean rather than contributing a zero row; feeding an all-masked row
        to a softmax would produce a division of zero by zero and poison every term
        downstream of it.
        """
        binary = mask > 0
        logits = scores.masked_fill(~binary, torch.finfo(scores.dtype).min)
        weights = torch.where(binary, path_weights, torch.zeros_like(path_weights))
        total = weights.sum(dim=-1, keepdim=True)
        populated = total.squeeze(-1) > 0
        targets = torch.where(
            populated.unsqueeze(-1),
            weights / total.clamp(min=1e-9),
            torch.zeros_like(weights),
        )
        log_probabilities = torch.log_softmax(logits, dim=-1)
        raw = -torch.sum(targets * log_probabilities, dim=-1)
        raw = torch.where(populated, raw, torch.zeros_like(raw))
        divisor = torch.clamp(populated.to(scores.dtype).sum(), min=1.0)
        loss: Tensor = self.weight * raw.sum() / divisor
        return loss


def contrastive_margin_loss(positive: Tensor, negative: Tensor, margin: float) -> Tensor:
    """Hinge on the score gap between a retained and a rejected path."""
    if positive.numel() == 0 or negative.numel() == 0:
        return torch.zeros((), dtype=torch.float32)
    gap = positive.mean() - negative.mean()
    return torch.relu(margin - gap)


def treatment_conditional_gap(
    scores_braf: Tensor,
    scores_wild: Tensor,
) -> Tensor:
    """Mean score gap between the two feasible branches.

    Sec. IV-D reports this gap as the quantitative form of the treatment-
    conditional finding (0.731 against 0.765 when the scorer is pooled).
    """
    if scores_braf.numel() == 0 or scores_wild.numel() == 0:
        return torch.zeros((), dtype=torch.float32)
    return scores_braf.mean() - scores_wild.mean()
