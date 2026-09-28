"""Relational rotation in complex space and the attention readout.

Ref: Sec. III-C ("the path coder which has an attention readout driven by the
edge sequence"; the relational-path model of refs. [22], [23]).

A relation is a rotation in the complex plane, so composing the relations along a
path is a product of rotations and the path's endpoint prediction is the rotated
start compared with the observed end. The readout attends over the edge
representations rather than pooling them, which is what the leave-one-out
comparison against mean pooling isolates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn

from pathanchor.utils.torch_helpers import reset_linear
from pathanchor.utils.types import FloatArray


def entity_hash_features(indices: Tensor, dimension: int, seed: int = 20260101) -> Tensor:
    """A fixed, non-trainable encoding of entity identity.

    The released graph holds 386,204 entities; a trainable row per entity would
    dominate the parameter count for no reported benefit, so identity enters as a
    deterministic trigonometric projection of the entity index. The projection is
    fixed at construction, so it cannot absorb label information.
    """
    position = indices.to(torch.float64).unsqueeze(-1)
    frequency = torch.arange(1, dimension + 1, dtype=torch.float64) * math.pi
    phase = position * frequency / float(seed) * 1000.0
    stacked = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)
    features: Tensor = stacked.to(torch.float32)
    return features


def complex_multiply(left: Tensor, right: Tensor) -> Tensor:
    """Multiply two complex vectors held as interleaved real pairs.

    The last dimension is split into a real and an imaginary half; both inputs
    must share that layout.
    """
    half = left.shape[-1] // 2
    re_l, im_l = left[..., :half], left[..., half:]
    re_r, im_r = right[..., :half], right[..., half:]
    return torch.cat([re_l * re_r - im_l * im_r, re_l * im_r + im_l * re_r], dim=-1)


def rotation_from_phase(phase: Tensor) -> Tensor:
    """Turn an angle vector into the complex pair (cos, sin)."""
    return torch.cat([torch.cos(phase), torch.sin(phase)], dim=-1)


def composed_angle(phases: FloatArray) -> FloatArray:
    """Sum of rotation angles along a path.

    Used by the verification layer: rotating by r1 and then by r2 equals a single
    rotation by the sum of their angles, which is a closed form the module's
    composition can be checked against without calling the module.
    """
    total: FloatArray = np.sum(phases, axis=0)
    return total


def rotate_sequence(start: FloatArray, phases: FloatArray) -> FloatArray:
    """Rotate a complex start vector by each phase in turn.

    Independent of the torch module: the explicit complex product is written out
    so the check has something to disagree with.
    """
    current = np.array(start, dtype=np.float64, copy=True)
    half = current.shape[0] // 2
    for phase in phases:
        re_c = np.cos(phase)
        im_c = np.sin(phase)
        re_h, im_h = current[:half], current[half:]
        current = np.concatenate([re_h * re_c - im_h * im_c, re_h * im_c + im_h * re_c])
    rotated: FloatArray = current
    return rotated


@dataclass(frozen=True, slots=True)
class ScorerConfig:
    dimension: int = 32
    hidden: int = 64
    entity_types: int = 41
    relations: int = 42
    tiers: int = 4
    dropout: float = 0.0


class RelationalRotationScorer(nn.Module):
    """Scores a batch of typed paths.

    Inputs are index tensors, so the module never carries a per-entity parameter
    table and the same instance scores paths over any subgraph of the resource.
    """

    def __init__(self, config: ScorerConfig) -> None:
        super().__init__()
        self.config = config
        self.relation_phase = nn.Parameter(torch.zeros(config.relations, config.dimension))
        self.entity_offset = nn.Parameter(torch.zeros(config.entity_types, 2 * config.dimension))
        self.tier_scale = nn.Parameter(torch.ones(config.tiers))
        self.edge_projection = nn.Linear(2 * config.dimension + 2, config.hidden)
        self.attention = nn.Linear(config.hidden, 1)
        self.readout = nn.Linear(config.hidden, 1)
        self.dropout = nn.Dropout(config.dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.config.dimension)
        with torch.no_grad():
            self.relation_phase.uniform_(-math.pi, math.pi)
            self.entity_offset.uniform_(-bound, bound)
            self.tier_scale.fill_(1.0)
        reset_linear(self.edge_projection)
        reset_linear(self.attention)
        reset_linear(self.readout)

    def forward(
        self,
        entity_indices: Tensor,
        entity_types: Tensor,
        relation_ids: Tensor,
        tier_ids: Tensor,
        edge_weights: Tensor,
        mask: Tensor,
        condition: Tensor | None = None,
    ) -> Tensor:
        """Return one score per path.

        ``entity_indices`` has shape (B, L+1) and holds the walk, ``relation_ids``,
        ``tier_ids`` and ``edge_weights`` have shape (B, L), and ``mask`` marks the
        edges that exist in each path. ``condition`` is the patient embedding z_p
        that the readout is conditioned on.
        """
        walk = entity_hash_features(entity_indices, self.config.dimension)
        walk = walk + self.entity_offset[entity_types]
        phase = self.relation_phase[relation_ids]
        rotation = rotation_from_phase(phase)
        states = [walk[:, 0, :]]
        for step in range(relation_ids.shape[1]):
            rotated = complex_multiply(states[-1], rotation[:, step, :])
            states.append(rotated + self.entity_offset[entity_types[:, step + 1]])
        edge_states = torch.stack(states[1:], dim=1)
        scale = self.tier_scale[tier_ids].unsqueeze(-1)
        edge_inputs = torch.cat([edge_states, edge_weights.unsqueeze(-1), scale], dim=-1)
        projected = torch.tanh(self.edge_projection(edge_inputs))
        projected = self.dropout(projected)
        logits = self.attention(projected).squeeze(-1)
        negated = torch.finfo(logits.dtype).min * torch.ones_like(logits)
        logits = torch.where(mask > 0, logits, negated)
        weights = torch.softmax(logits, dim=-1)
        pooled = torch.sum(projected * weights.unsqueeze(-1), dim=1)
        if condition is not None:
            width = min(pooled.shape[-1], condition.shape[-1])
            pooled = pooled.clone()
            pooled[:, :width] = pooled[:, :width] + condition[:, :width]
        score: Tensor = self.readout(pooled).squeeze(-1)
        return score

    def mean_pool_forward(
        self,
        entity_indices: Tensor,
        entity_types: Tensor,
        relation_ids: Tensor,
        tier_ids: Tensor,
        edge_weights: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """The same encoder with the readout replaced by a masked mean.

        This is the "mean-pool paths (no log-sum-exp)" comparison of Table II
        Panel B: the readout changes, the rotation does not.
        """
        walk = entity_hash_features(entity_indices, self.config.dimension)
        walk = walk + self.entity_offset[entity_types]
        rotation = rotation_from_phase(self.relation_phase[relation_ids])
        states = [walk[:, 0, :]]
        for step in range(relation_ids.shape[1]):
            rotated = complex_multiply(states[-1], rotation[:, step, :])
            states.append(rotated + self.entity_offset[entity_types[:, step + 1]])
        edge_states = torch.stack(states[1:], dim=1)
        scale = self.tier_scale[tier_ids].unsqueeze(-1)
        edge_inputs = torch.cat([edge_states, edge_weights.unsqueeze(-1), scale], dim=-1)
        projected = torch.tanh(self.edge_projection(edge_inputs))
        numerator = torch.sum(projected * mask.unsqueeze(-1), dim=1)
        denominator = torch.clamp(mask.sum(dim=1, keepdim=True), min=1.0)
        pooled = numerator / denominator
        score: Tensor = self.readout(pooled).squeeze(-1)
        return score
