"""Composite objective.

Ref: Sec. IV-A (discrimination), Sec. III-C (retained path set),
Sec. IV-D (branch-conditional behaviour).

Each term reports its own value so a training log shows which objective moved,
which is what makes a single-batch overfit check meaningful rather than a check
that the total happened to drop.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

from pathanchor.losses.ranking import ListwisePathLoss, PairwiseRankingLoss
from pathanchor.losses.regularisers import RetentionMarginLoss, RetentionSparsityLoss
from pathanchor.losses.response import ResponseLoss, balanced_positive_weight


@dataclass(frozen=True, slots=True)
class LossWeights:
    response: float = 1.0
    patient_ranking: float = 1.0
    path_listwise: float = 0.5
    retention_sparsity: float = 0.05
    retention_margin: float = 0.05


@dataclass(slots=True)
class LossBreakdown:
    """Named term values for one batch."""

    terms: dict[str, float] = field(default_factory=dict)
    total: float = 0.0

    def as_dict(self) -> Mapping[str, float]:
        out = dict(sorted(self.terms.items()))
        out["total"] = self.total
        return out


class CompositeObjective(nn.Module):
    """The weighted sum of the response, ranking and retention terms."""

    def __init__(
        self,
        weights: LossWeights,
        epsilon: float,
        path_target: float = 8.0,
        ranking_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.weights = weights
        self.response = ResponseLoss(weight=weights.response)
        self.patient_ranking = PairwiseRankingLoss(weight=weights.patient_ranking)
        self.path_listwise = ListwisePathLoss(weight=weights.path_listwise)
        self.retention_sparsity = RetentionSparsityLoss(
            epsilon=epsilon, weight=weights.retention_sparsity, target=path_target
        )
        self.retention_margin = RetentionMarginLoss(
            epsilon=epsilon, weight=weights.retention_margin
        )
        self.ranking_scale = float(ranking_scale)

    def forward(
        self,
        patient_scores: Tensor,
        labels: Tensor,
        labels_braf: Tensor,
        labels_wild: Tensor,
        path_scores: Tensor,
        path_weights: Tensor,
        path_mask: Tensor,
    ) -> tuple[Tensor, LossBreakdown]:
        response = self.response(patient_scores, labels)
        ranking = self.patient_ranking(patient_scores, labels)
        ranking = ranking * self.ranking_scale
        listwise = self.path_listwise(path_scores, path_weights, path_mask)
        sparsity = self.retention_sparsity(path_scores, path_mask)
        margin = self.retention_margin(path_scores, path_mask)
        branch = _branch_term(patient_scores, labels_braf, labels_wild)
        total: Tensor = response + ranking + listwise + sparsity + margin + branch
        breakdown = LossBreakdown(
            terms={
                "response": float(response.detach()),
                "patient_ranking": float(ranking.detach()),
                "path_listwise": float(listwise.detach()),
                "retention_sparsity": float(sparsity.detach()),
                "retention_margin": float(margin.detach()),
                "branch_contrast": float(branch.detach()),
            },
            total=float(total.detach()),
        )
        return (total, breakdown)


def _branch_term(
    scores: Tensor, mask_braf: Tensor, mask_wild: Tensor, margin: float = 0.05
) -> Tensor:
    """Keep the two feasible branches from collapsing to the same score.

    Sec. IV-D reports that pooling the branches costs 3.4 points of AUROC; the
    term below is the training-side statement of the same property and is a
    declared engineering default, as no objective is reported in the manuscript.
    The gap is bounded above by the hinge, so the term cannot be reduced without
    limit by pushing the two branches apart.
    """
    braf = scores[mask_braf > 0]
    wild = scores[mask_wild > 0]
    if braf.numel() == 0 or wild.numel() == 0:
        return torch.zeros((), dtype=scores.dtype, requires_grad=scores.requires_grad)
    gap = torch.abs(braf.mean() - wild.mean())
    return torch.relu(margin - gap)


def positive_weight_for(labels: Tensor) -> float:
    return balanced_positive_weight(labels)
