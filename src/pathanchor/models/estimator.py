"""The estimator: multimodal features plus typed evidence paths.

Ref: Eq. (1) (the projected arg max), Algorithm 1 (the five-agent pipeline),
Table II Panel A (additive component contributions) and Panel B (leave-one-out).

The score of an option is the path marginal of Algorithm 1 line 6 added to a
feature-only response logit, so the feature arm and the graph arm are separate
summands and the component analysis can switch either off. Everything that
changes the *interpretation* of the score rather than the score itself --
conformal calibration, the guideline projection -- lives in the agents, not here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

from pathanchor.models.encoder import (
    EncoderConfig,
    ModalityBlock,
    MultimodalEncoder,
    ResponseHead,
)
from pathanchor.paths.rotation import RelationalRotationScorer, ScorerConfig
from pathanchor.paths.scorer import PathBatch
from pathanchor.utils.torch_helpers import inverse_softplus, reset_linear


@dataclass(frozen=True, slots=True)
class EstimatorConfig:
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    scorer: ScorerConfig = field(default_factory=ScorerConfig)
    typed_paths: bool = True
    use_graph_paths: bool = True
    use_base: bool = True
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-2


class UntypedPathReadout(nn.Module):
    """A bag-of-evidence readout that ignores relation identity and order.

    This is the "typed paths -> free-text rationale" arm of Table II Panel B: the
    graph is still read, but the output carries no relation typing, so an auditor
    cannot replay the route. It costs discrimination and, much more, agreement.
    """

    def __init__(self, tiers: int, hidden: int) -> None:
        super().__init__()
        self.tier_weight = nn.Parameter(torch.ones(tiers))
        self.projection = nn.Linear(1, hidden)
        self.output = nn.Linear(hidden, 1)
        reset_linear(self.projection)
        reset_linear(self.output)

    def forward(
        self,
        tier_ids: Tensor,
        edge_weights: Tensor,
        mask: Tensor,
        condition: Tensor | None = None,
    ) -> Tensor:
        scaled = self.tier_weight[tier_ids] * edge_weights
        numerator = torch.sum(scaled * mask, dim=1, keepdim=True)
        denominator = torch.clamp(mask.sum(dim=1, keepdim=True), min=1.0)
        pooled = numerator / denominator
        hidden = torch.tanh(self.projection(pooled))
        if condition is not None:
            width = min(hidden.shape[-1], condition.shape[-1])
            hidden = hidden.clone()
            hidden[:, :width] = hidden[:, :width] + condition[:, :width]
        score: Tensor = self.output(hidden).squeeze(-1)
        return score


class PathAnchorEstimator(nn.Module):
    """The trained part of the framework."""

    def __init__(self, config: EstimatorConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = MultimodalEncoder(config.encoder)
        self.response_head = ResponseHead(config.encoder.hidden)
        self.path_scorer = RelationalRotationScorer(config.scorer)
        self.untyped_readout = UntypedPathReadout(config.scorer.tiers, config.scorer.hidden)
        self.bridge = nn.Linear(config.encoder.hidden, config.scorer.hidden)
        self.base_scale = nn.Parameter(torch.tensor(inverse_softplus(1.0), dtype=torch.float32))
        reset_linear(self.bridge)

    def encode(self, features: Tensor, mask: Tensor) -> Tensor:
        encoded: Tensor = self.encoder(features, mask)
        return encoded

    def base_logit(self, encoded: Tensor) -> Tensor:
        logit: Tensor = self.response_head(encoded)
        return logit

    def conditions(self, encoded: Tensor) -> Tensor:
        bridged: Tensor = self.bridge(encoded)
        return bridged

    def path_scores(self, batch: PathBatch, encoded: Tensor) -> Tensor:
        condition = self.conditions(encoded)
        if self.config.typed_paths:
            typed: Tensor = self.path_scorer(
                batch.entity_indices,
                batch.entity_types,
                batch.relation_ids,
                batch.tier_ids,
                batch.edge_weights,
                batch.mask,
                condition,
            )
            return typed
        untyped: Tensor = self.untyped_readout(
            batch.tier_ids, batch.edge_weights, batch.mask, condition
        )
        return untyped

    def combine(self, base: Tensor, marginal: Tensor) -> Tensor:
        if not self.config.use_base:
            return marginal
        if not self.config.use_graph_paths:
            return base
        return torch.nn.functional.softplus(self.base_scale) * base + marginal

    def option_score(self, base: Tensor, marginal: Tensor) -> Tensor:
        return self.combine(base, marginal)


def marginalise_path_scores(scores: Tensor) -> Tensor:
    """log sum exp over one option's paths, with the empty set mapped to a floor."""
    if scores.numel() == 0:
        return torch.full((1,), -60.0, dtype=torch.float32)
    return torch.logsumexp(scores, dim=0, keepdim=True)


def mean_pool_scores(scores: Tensor) -> Tensor:
    if scores.numel() == 0:
        return torch.full((1,), -60.0, dtype=torch.float32)
    return torch.mean(scores, dim=0, keepdim=True)


def build_estimator(config: EstimatorConfig | None = None) -> PathAnchorEstimator:
    return PathAnchorEstimator(config if config is not None else EstimatorConfig())


def parameter_count(module: nn.Module) -> int:
    return sum(int(item.numel()) for item in module.parameters() if item.requires_grad)


def trainable_named_shapes(module: nn.Module) -> dict[str, tuple[int, ...]]:
    return {
        name: tuple(int(item) for item in parameter.shape)
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    }


def encoder_blocks(model: PathAnchorEstimator) -> int:
    return sum(1 for item in model.encoder.blocks if isinstance(item, ModalityBlock))
