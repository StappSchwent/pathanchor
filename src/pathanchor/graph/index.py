"""Compact array-backed graph: the representation the released resource uses.

Ref: Sec. III-B (Nodes, Edges), Sec. III-C (bounded expansion).

The full resource holds 386,204 entities and 7,431,905 edges. Object-per-edge
storage would cost gigabytes, so the shared graph lives in flat integer arrays
with a compressed-sparse-row adjacency on both directions, and materialises a
typed :class:`~pathanchor.graph.kg.KnowledgeGraph` only for a bounded
neighbourhood. Endpoints and weights use the narrowest dtypes that cannot
overflow at that size.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from pathanchor.graph.kg import Edge, Entity, GraphCensus, KnowledgeGraph
from pathanchor.graph.ontology import (
    NODE_TYPES,
    RELATION_TYPES,
    EvidenceTier,
    NodeType,
    RelationType,
    SourceOntology,
)
from pathanchor.utils.types import CodeArray, GraphArray, WeightArray

NODE_TYPE_IDS: Mapping[str, int] = {item.value: index for index, item in enumerate(NODE_TYPES)}
RELATION_TYPE_IDS: Mapping[str, int] = {
    item.value: index for index, item in enumerate(RELATION_TYPES)
}
TIER_IDS: Mapping[str, int] = {item.value: index for index, item in enumerate(EvidenceTier)}
NODE_TYPE_BY_ID: tuple[NodeType, ...] = NODE_TYPES
RELATION_BY_ID: tuple[RelationType, ...] = RELATION_TYPES
TIER_BY_ID: tuple[EvidenceTier, ...] = tuple(EvidenceTier)
SOURCE_BY_ID: tuple[SourceOntology, ...] = tuple(SourceOntology)
SOURCE_IDS: Mapping[str, int] = {item.value: index for index, item in enumerate(SourceOntology)}


@dataclass(frozen=True, slots=True)
class NeighbourBatch:
    """A CSR slice: the tail indices reachable from one node, with relation ids."""

    tails: GraphArray
    relations: CodeArray
    weights: WeightArray

    def __len__(self) -> int:
        return int(self.tails.shape[0])


class CompactGraph:
    """CSR adjacency over flat endpoint arrays."""

    def __init__(
        self,
        entity_keys: Sequence[str],
        entity_types: object,
        entity_sources: object,
        edge_head: object,
        edge_relation: object,
        edge_tail: object,
        edge_tier: object,
        edge_weight: object,
    ) -> None:
        types = np.asarray(entity_types, dtype=np.int16)
        sources = np.asarray(entity_sources, dtype=np.int16)
        head = np.asarray(edge_head, dtype=np.int32)
        relation = np.asarray(edge_relation, dtype=np.int16)
        tail = np.asarray(edge_tail, dtype=np.int32)
        tier = np.asarray(edge_tier, dtype=np.int16)
        weight = np.asarray(edge_weight, dtype=np.float32)
        if types.shape[0] != len(entity_keys):
            raise ValueError("entity key list and entity type array disagree in length")
        if head.shape[0] != relation.shape[0] or head.shape[0] != tail.shape[0]:
            raise ValueError("edge endpoint arrays disagree in length")
        if head.shape[0] > 0 and int(head.min()) < 0:
            raise ValueError("an edge endpoint is negative")
        self._keys: tuple[str, ...] = tuple(entity_keys)
        self._entity_types = types
        self._entity_sources = sources
        self._head = head
        self._relation = relation
        self._tail = tail
        self._tier = tier
        self._weight = weight
        self._out_ptr, self._out_tail, self._out_relation, self._out_weight, self._out_edge = _csr(
            head, tail, relation, weight, len(self._keys)
        )
        self._in_ptr, self._in_head, self._in_relation, _, self._in_edge = _csr(
            tail, head, relation, weight, len(self._keys)
        )

    @property
    def n_entities(self) -> int:
        return len(self._keys)

    @property
    def n_edges(self) -> int:
        return int(self._head.shape[0])

    def keys(self) -> tuple[str, ...]:
        return self._keys

    def key_of(self, index: int) -> str:
        return self._keys[index]

    def node_type_of(self, index: int) -> NodeType:
        return NODE_TYPE_BY_ID[int(self._entity_types[index])]

    def source_of(self, index: int) -> SourceOntology:
        return SOURCE_BY_ID[int(self._entity_sources[index])]

    def out_neighbours(self, index: int) -> NeighbourBatch:
        start = int(self._out_ptr[index])
        stop = int(self._out_ptr[index + 1])
        return NeighbourBatch(
            self._out_tail[start:stop],
            self._out_relation[start:stop],
            self._out_weight[start:stop],
        )

    def in_neighbours(self, index: int) -> NeighbourBatch:
        start = int(self._in_ptr[index])
        stop = int(self._in_ptr[index + 1])
        return NeighbourBatch(
            self._in_head[start:stop],
            self._in_relation[start:stop],
            np.ones(stop - start, dtype=np.float32),
        )

    def edge_arrays(self) -> tuple[GraphArray, CodeArray, GraphArray, CodeArray, WeightArray]:
        return (self._head, self._relation, self._tail, self._tier, self._weight)

    def out_edge_ids(self, index: int) -> GraphArray:
        start = int(self._out_ptr[index])
        stop = int(self._out_ptr[index + 1])
        return self._out_edge[start:stop]

    def edge_relation_of(self, edge_id: int) -> RelationType:
        return RELATION_BY_ID[int(self._relation[edge_id])]

    def edge_tier_of(self, edge_id: int) -> EvidenceTier:
        return TIER_BY_ID[int(self._tier[edge_id])]

    def edge_weight_of(self, edge_id: int) -> float:
        return float(self._weight[edge_id])

    def edge_tail_of(self, edge_id: int) -> int:
        return int(self._tail[edge_id])

    def entity_type_array(self) -> CodeArray:
        return self._entity_types

    def entity_source_array(self) -> CodeArray:
        return self._entity_sources

    def census(self) -> GraphCensus:
        per_type: dict[str, int] = defaultdict(int)
        for code in self._entity_types.tolist():
            per_type[NODE_TYPE_BY_ID[int(code)].value] += 1
        per_relation: dict[str, int] = defaultdict(int)
        for code in self._relation.tolist():
            per_relation[RELATION_BY_ID[int(code)].value] += 1
        per_tier: dict[str, int] = defaultdict(int)
        for code in self._tier.tolist():
            per_tier[TIER_BY_ID[int(code)].value] += 1
        return GraphCensus(
            entities=self.n_entities,
            edges=self.n_edges,
            node_types=len(per_type),
            relation_types=len(per_relation),
            entities_per_type=dict(per_type),
            edges_per_relation=dict(per_relation),
            edges_per_tier=dict(per_tier),
        )

    def breadth_first(
        self,
        anchors: Sequence[int],
        depth: int,
        branching: int,
        forward: bool = True,
        relation_filter: frozenset[int] | None = None,
    ) -> list[list[int]]:
        """Level-by-level expansion from a set of anchors.

        Each node contributes at most ``branching`` successors per level, chosen
        by descending edge weight with the node index breaking ties, so the
        expansion is deterministic without materialising the full frontier.
        """
        levels: list[list[int]] = [list(dict.fromkeys(int(item) for item in anchors))]
        seen: set[int] = set(levels[0])
        for _ in range(depth):
            frontier: list[int] = []
            for node in levels[-1]:
                batch = self.out_neighbours(node) if forward else self.in_neighbours(node)
                for candidate in _top_weighted(
                    batch.tails, batch.relations, batch.weights, branching, relation_filter
                ):
                    if candidate not in seen:
                        seen.add(candidate)
                        frontier.append(candidate)
            if not frontier:
                break
            levels.append(frontier)
        return levels

    def materialise(self, indices: Iterable[int], entities_only: bool = False) -> KnowledgeGraph:
        """Build a fully typed view over the induced subgraph of ``indices``."""
        chosen = sorted({int(index) for index in indices})
        if len(chosen) == 0:
            return KnowledgeGraph()
        selected = np.asarray(chosen, dtype=np.int32)
        position = {index: slot for slot, index in enumerate(chosen)}
        graph = KnowledgeGraph()
        for index in chosen:
            graph.add_entity(
                Entity(
                    key=self._keys[index],
                    node_type=NODE_TYPE_BY_ID[int(self._entity_types[index])],
                    label=self._keys[index],
                    source=SOURCE_BY_ID[int(self._entity_sources[index])],
                )
            )
        if entities_only:
            return graph
        member = np.zeros(self.n_entities, dtype=bool)
        member[selected] = True
        keep = member[self._head] & member[self._tail]
        heads = self._head[keep]
        relations = self._relation[keep]
        tails = self._tail[keep]
        tiers = self._tier[keep]
        for slot in range(int(keep.sum())):
            graph.add_edge(
                Edge(
                    head=self._keys[position[int(heads[slot])]],
                    relation=RELATION_BY_ID[int(relations[slot])],
                    tail=self._keys[position[int(tails[slot])]],
                    tier=TIER_BY_ID[int(tiers[slot])],
                )
            )
        return graph


def _csr(
    sources: GraphArray,
    targets: GraphArray,
    relations: CodeArray,
    weights: WeightArray,
    n_nodes: int,
) -> tuple[np.ndarray, GraphArray, CodeArray, WeightArray, GraphArray]:
    order = np.argsort(sources, kind="stable").astype(np.int32)
    sorted_sources = sources[order]
    counts = np.bincount(sorted_sources.astype(np.int64), minlength=n_nodes)
    pointers = np.zeros(n_nodes + 1, dtype=np.int64)
    np.cumsum(counts, out=pointers[1:])
    return (
        pointers.astype(np.int32),
        targets[order],
        relations[order],
        weights[order],
        order,
    )


def _top_weighted(
    tails: GraphArray,
    relations: CodeArray,
    weights: WeightArray,
    branching: int,
    relation_filter: frozenset[int] | None,
) -> list[int]:
    if tails.shape[0] == 0:
        return []
    if relation_filter is None:
        allowed = np.ones(tails.shape[0], dtype=bool)
    else:
        allowed = np.isin(
            relations.astype(np.int64), np.fromiter(sorted(relation_filter), dtype=np.int64)
        )
    indices = np.nonzero(allowed)[0]
    if indices.shape[0] == 0:
        return []
    if indices.shape[0] <= branching:
        ordered = indices
    else:
        keyed = sorted(
            (int(slot) for slot in indices.tolist()),
            key=lambda slot: (-float(weights[slot]), int(tails[slot])),
        )
        ordered = np.asarray(keyed[:branching], dtype=np.int64)
    return [int(tails[slot]) for slot in ordered]


def build_index(
    keys: Sequence[str],
    types: object,
    sources: object,
    head: object,
    relation: object,
    tail: object,
    tier: object,
    weight: object,
) -> CompactGraph:
    return CompactGraph(keys, types, sources, head, relation, tail, tier, weight)
