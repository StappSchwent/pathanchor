"""The path scorer: batching typed paths and applying the retention threshold.

Ref: Sec. III-C (path coder, epsilon threshold), Table II Panel B (top-m
retention, mean pooling).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from pathanchor.graph.index import RELATION_TYPE_IDS, TIER_IDS, CompactGraph
from pathanchor.paths.rotation import RelationalRotationScorer, ScorerConfig
from pathanchor.paths.signatures import EvidencePath


@dataclass(frozen=True, slots=True)
class PathBatch:
    entity_indices: Tensor
    entity_types: Tensor
    relation_ids: Tensor
    tier_ids: Tensor
    edge_weights: Tensor
    mask: Tensor
    lengths: Tensor

    @property
    def size(self) -> int:
        return int(self.entity_indices.shape[0])


def key_index(graph: CompactGraph) -> dict[str, int]:
    return {key: index for index, key in enumerate(graph.keys())}


def encode_paths(
    graph: CompactGraph,
    paths: Sequence[EvidencePath],
    max_hops: int,
    key_to_index: Mapping[str, int] | None = None,
) -> PathBatch:
    """Stack a list of paths into fixed-width index tensors.

    Shorter paths are padded with their own last edge and masked off, so the
    rotation loop can run over a rectangular batch without branching on length.
    """
    lookup = key_index(graph) if key_to_index is None else key_to_index
    n = len(paths)
    if n == 0:
        zeros = torch.zeros((0, max_hops + 1), dtype=torch.long)
        short = torch.zeros((0, max_hops), dtype=torch.long)
        return PathBatch(
            zeros,
            zeros,
            short,
            short,
            short.float(),
            short.float(),
            torch.zeros(0, dtype=torch.long),
        )
    entity_idx = torch.zeros((n, max_hops + 1), dtype=torch.long)
    entity_ty = torch.zeros((n, max_hops + 1), dtype=torch.long)
    relation = torch.zeros((n, max_hops), dtype=torch.long)
    tier = torch.zeros((n, max_hops), dtype=torch.long)
    weight = torch.zeros((n, max_hops), dtype=torch.float32)
    mask = torch.zeros((n, max_hops), dtype=torch.float32)
    lengths = torch.zeros(n, dtype=torch.long)
    types = graph.entity_type_array()
    for row, path in enumerate(paths):
        nodes = path.nodes()
        for column, key in enumerate(nodes):
            index = lookup[key]
            entity_idx[row, column] = index
            entity_ty[row, column] = int(types[index])
        for column, edge in enumerate(path.edges):
            relation[row, column] = RELATION_TYPE_IDS[edge.relation.value]
            tier[row, column] = TIER_IDS[edge.tier.value]
            weight[row, column] = float(edge.weight)
            mask[row, column] = 1.0
        lengths[row] = len(path)
        for column in range(len(path), max_hops):
            entity_idx[row, column + 1] = entity_idx[row, len(path)]
            entity_ty[row, column + 1] = entity_ty[row, len(path)]
    return PathBatch(entity_idx, entity_ty, relation, tier, weight, mask, lengths)


class PathScorer(nn.Module):
    """Scores an evidence path and decides whether it is retained."""

    def __init__(self, config: ScorerConfig, epsilon: float = 1e-3) -> None:
        super().__init__()
        self.config = config
        self.epsilon = float(epsilon)
        self.encoder = RelationalRotationScorer(config)

    def forward(self, batch: PathBatch) -> Tensor:
        scores: Tensor = self.encoder(
            batch.entity_indices,
            batch.entity_types,
            batch.relation_ids,
            batch.tier_ids,
            batch.edge_weights,
            batch.mask,
        )
        return scores

    def mean_pool(self, batch: PathBatch) -> Tensor:
        pooled: Tensor = self.encoder.mean_pool_forward(
            batch.entity_indices,
            batch.entity_types,
            batch.relation_ids,
            batch.tier_ids,
            batch.edge_weights,
            batch.mask,
        )
        return pooled

    def retain(self, scores: Tensor) -> Tensor:
        """Retention mask of Sec. III-C: keep a path whose impact reaches the
        threshold."""
        return (scores.abs() >= self.epsilon).to(torch.bool)

    def score_paths(
        self, graph: CompactGraph, paths: Sequence[EvidencePath], max_hops: int
    ) -> list[float]:
        batch = encode_paths(graph, paths, max_hops)
        if batch.size == 0:
            return []
        with torch.no_grad():
            values = self.forward(batch)
        return [float(item) for item in values.detach().cpu().tolist()]

    def retained_paths(
        self,
        graph: CompactGraph,
        paths: Sequence[EvidencePath],
        max_hops: int,
    ) -> list[EvidencePath]:
        batch = encode_paths(graph, paths, max_hops)
        if batch.size == 0:
            return []
        with torch.no_grad():
            values = self.forward(batch)
            keep = self.retain(values)
        return [path for path, flag in zip(paths, keep.tolist(), strict=True) if flag]


def top_m_paths(
    graph: CompactGraph,
    paths: Sequence[EvidencePath],
    scores: Sequence[float],
    m: int,
) -> list[EvidencePath]:
    """The m highest-scoring retained paths, which form E(p)."""
    if m <= 0 or not paths:
        return []
    order = sorted(range(len(paths)), key=lambda slot: (-scores[slot], paths[slot].canonical_key()))
    return [paths[slot] for slot in order[:m]]


def score_variance(scores: Sequence[float]) -> float:
    if not scores:
        return 0.0
    mean = sum(scores) / len(scores)
    return sum((value - mean) ** 2 for value in scores) / len(scores)
