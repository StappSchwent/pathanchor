"""Objective terms."""

from __future__ import annotations

import torch

from pathanchor.losses.composite import CompositeObjective, LossWeights, positive_weight_for
from pathanchor.losses.ranking import PairwiseRankingLoss, contrastive_margin_loss
from pathanchor.losses.regularisers import RetentionMarginLoss, RetentionSparsityLoss
from pathanchor.losses.response import (
    ResponseLoss,
    balanced_positive_weight,
    expected_response_probability,
)
from pathanchor.training.seed import set_seed


def test_response_loss_is_zero_for_perfect_logits() -> None:
    set_seed(20260101)
    loss = ResponseLoss()
    logits = torch.tensor([8.0, -8.0, 8.0])
    labels = torch.tensor([1.0, 0.0, 1.0])
    assert float(loss(logits, labels)) < 1e-3


def test_positive_weight_follows_the_class_ratio() -> None:
    labels = torch.tensor([1.0, 0.0, 0.0, 0.0])
    assert balanced_positive_weight(labels) == 3.0
    assert positive_weight_for(labels) == 3.0


def test_expected_probability_lies_between_the_option_probabilities() -> None:
    scores = torch.tensor([[0.5, 1.5, -0.5]])
    probability = expected_response_probability(scores)
    bounds = torch.sigmoid(scores)
    assert float(probability.min()) >= float(bounds.min()) - 1e-6
    assert float(probability.max()) <= float(bounds.max()) + 1e-6


def test_pairwise_ranking_loss_is_zero_for_a_perfect_order() -> None:
    set_seed(20260101)
    loss = PairwiseRankingLoss()
    scores = torch.tensor([40.0, 20.0, -40.0])
    labels = torch.tensor([1.0, 1.0, 0.0])
    assert float(loss(scores, labels)) < 1e-6


def test_pairwise_ranking_loss_is_flat_without_label_variation() -> None:
    loss = PairwiseRankingLoss()
    scores = torch.tensor([1.0, 2.0])
    labels = torch.tensor([1.0, 1.0])
    assert float(loss(scores, labels)) == 0.0


def test_retention_sparsity_is_zero_below_the_target() -> None:
    loss = RetentionSparsityLoss(epsilon=0.0, target=4.0)
    scores = torch.zeros((1, 4))
    mask = torch.ones((1, 4))
    assert float(loss(scores, mask)) == 0.0


def test_retention_margin_is_larger_near_the_threshold() -> None:
    loss = RetentionMarginLoss(epsilon=0.1, margin=0.05)
    mask = torch.ones((1, 2))
    near = loss(torch.tensor([[0.1, 0.1]]), mask)
    far = loss(torch.tensor([[1.0, 1.0]]), mask)
    assert float(near) > float(far)


def test_contrastive_margin_penalises_a_reversed_gap() -> None:
    positive = torch.tensor([0.0])
    negative = torch.tensor([1.0])
    assert float(contrastive_margin_loss(positive, negative, 0.5)) > 0.0


def test_composite_reports_every_term_and_backpropagates() -> None:
    set_seed(20260101)
    objective = CompositeObjective(LossWeights(), epsilon=0.0)
    patient_scores = torch.tensor([0.5, -0.3, 0.9, 0.2], requires_grad=True)
    labels = torch.tensor([1.0, 0.0, 1.0, 1.0])
    braf = torch.tensor([1.0, 1.0, 0.0, 0.0])
    wild = torch.tensor([0.0, 0.0, 1.0, 1.0])
    path_scores = torch.tensor(
        [[0.4, 0.2, -60.0], [0.1, -0.2, -60.0], [0.9, 0.8, 0.1], [0.3, 0.25, 0.2]]
    )
    path_weights = torch.tensor(
        [[1.0, 0.8, 0.0], [0.6, 0.5, 0.0], [1.0, 0.9, 0.7], [0.4, 0.3, 0.2]]
    )
    path_mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    total, breakdown = objective(
        patient_scores, labels, braf, wild, path_scores, path_weights, path_mask
    )
    assert set(breakdown.terms) == {
        "response",
        "patient_ranking",
        "path_listwise",
        "retention_sparsity",
        "retention_margin",
        "branch_contrast",
    }
    total.backward()
    assert patient_scores.grad is not None
    assert float(patient_scores.grad.abs().sum()) > 0.0
