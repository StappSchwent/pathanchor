"""The typed knowledge graph and the patient subgraph.

Ref: Sec. III-B (Nodes, Edges, Patient subgraphs), Sec. III-C (typed paths).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass

from pathanchor.graph.ontology import (
    EvidenceTier,
    NodeType,
    RelationType,
    SourceOntology,
    is_admissible,
    tier_of,
)


class GraphError(ValueError):
    """Raised on a malformed entity, a duplicate key or a signature violation."""


@dataclass(frozen=True, slots=True)
class Entity:
    key: str
    node_type: NodeType
    label: str
    source: SourceOntology


@dataclass(frozen=True, slots=True)
class Edge:
    head: str
    relation: RelationType
    tail: str
    tier: EvidenceTier
    weight: float = 1.0

    def endpoints(self) -> tuple[str, str]:
        return (self.head, self.tail)


@dataclass(frozen=True, slots=True)
class GraphCensus:
    entities: int
    edges: int
    node_types: int
    relation_types: int
    entities_per_type: Mapping[str, int]
    edges_per_relation: Mapping[str, int]
    edges_per_tier: Mapping[str, int]

    def totals(self) -> tuple[int, int, int, int]:
        return (self.entities, self.edges, self.node_types, self.relation_types)


class KnowledgeGraph:
    """A typed, signature-checked directed multigraph.

    An edge whose head or tail type is not declared by the relation's signature
    is rejected at insertion, so no ill-typed edge can exist in the structure
    that the enumeration later traverses.
    """

    def __init__(self) -> None:
        self._entities: dict[str, Entity] = {}
        self._edges: list[Edge] = []
        self._edge_keys: set[tuple[str, RelationType, str]] = set()
        self._out: dict[str, list[Edge]] = defaultdict(list)
        self._in: dict[str, list[Edge]] = defaultdict(list)

    def __len__(self) -> int:
        return len(self._entities)

    @property
    def n_entities(self) -> int:
        return len(self._entities)

    @property
    def n_edges(self) -> int:
        return len(self._edges)

    def add_entity(self, entity: Entity) -> None:
        existing = self._entities.get(entity.key)
        if existing is not None:
            if existing.node_type is not entity.node_type:
                raise GraphError(
                    f"entity {entity.key} already declared as {existing.node_type.value}, "
                    f"cannot redeclare as {entity.node_type.value}"
                )
            return
        self._entities[entity.key] = entity

    def add_edge(self, edge: Edge) -> None:
        head = self._entities.get(edge.head)
        tail = self._entities.get(edge.tail)
        if head is None:
            raise GraphError(f"edge head {edge.head} is not in the graph")
        if tail is None:
            raise GraphError(f"edge tail {edge.tail} is not in the graph")
        if not is_admissible(edge.relation, head.node_type, tail.node_type):
            raise GraphError(
                f"ill-typed edge: {edge.relation.value} from {head.node_type.value} "
                f"to {tail.node_type.value} is outside the declared signature"
            )
        key = (edge.head, edge.relation, edge.tail)
        if key in self._edge_keys:
            return
        self._edge_keys.add(key)
        self._edges.append(edge)
        self._out[edge.head].append(edge)
        self._in[edge.tail].append(edge)

    def extend(self, entities: Iterable[Entity], edges: Iterable[Edge]) -> None:
        for entity in entities:
            self.add_entity(entity)
        for edge in edges:
            self.add_edge(edge)

    def entity(self, key: str) -> Entity:
        value = self._entities.get(key)
        if value is None:
            raise GraphError(f"entity {key} is not in the graph")
        return value

    def has_entity(self, key: str) -> bool:
        return key in self._entities

    def node_type_of(self, key: str) -> NodeType:
        return self.entity(key).node_type

    def entities(self) -> Iterator[Entity]:
        return iter(self._entities.values())

    def edges(self) -> Iterator[Edge]:
        return iter(self._edges)

    def out_edges(self, key: str) -> Sequence[Edge]:
        return self._out.get(key, ())

    def in_edges(self, key: str) -> Sequence[Edge]:
        return self._in.get(key, ())

    def degree(self, key: str) -> int:
        return len(self._out.get(key, ())) + len(self._in.get(key, ()))

    def entities_of_type(self, node_type: NodeType) -> list[Entity]:
        return [item for item in self._entities.values() if item.node_type is node_type]

    def census(self) -> GraphCensus:
        per_type: dict[str, int] = defaultdict(int)
        for entity in self._entities.values():
            per_type[entity.node_type.value] += 1
        per_relation: dict[str, int] = defaultdict(int)
        per_tier: dict[str, int] = defaultdict(int)
        for edge in self._edges:
            per_relation[edge.relation.value] += 1
            per_tier[edge.tier.value] += 1
        return GraphCensus(
            entities=len(self._entities),
            edges=len(self._edges),
            node_types=len(per_type),
            relation_types=len(per_relation),
            entities_per_type=dict(per_type),
            edges_per_relation=dict(per_relation),
            edges_per_tier=dict(per_tier),
        )


@dataclass(frozen=True, slots=True)
class PatientSubgraph:
    """The de-identified slice of the graph a patient's record grounds to.

    Each entry maps a multimodal field to the graph key it is anchored at, so a
    path proposal always starts from an entity the record actually measured
    (Sec. III-B, "Patient subgraphs").
    """

    patient_id: str
    anchors: Mapping[str, tuple[str, ...]]

    def keys(self) -> tuple[str, ...]:
        seen: list[str] = []
        for group in self.anchors.values():
            seen.extend(group)
        return tuple(dict.fromkeys(seen))

    def __iter__(self) -> Iterator[str]:
        """Iterate the anchored keys, so an auditor can walk the subgraph."""
        return iter(self.keys())

    def group(self, name: str) -> tuple[str, ...]:
        return tuple(self.anchors.get(name, ()))


def merge_subgraph(graph: KnowledgeGraph, subgraph: PatientSubgraph) -> KnowledgeGraph:
    """Project a patient subgraph into the shared graph.

    The shared graph is expected to already hold every referenced entity, so the
    projection is a pure read; the helper exists to make that assumption explicit
    at the call site rather than implicit in the enumeration.
    """
    missing = [key for key in subgraph if not graph.has_entity(key)]
    if missing:
        joined = ", ".join(sorted(missing)[:8])
        raise GraphError(f"patient subgraph references entities absent from the graph: {joined}")
    return graph


def tier_counts(edges: Iterable[Edge]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for edge in edges:
        counts[edge.tier.value] += 1
    return dict(counts)


def declared_tier(relation: RelationType) -> EvidenceTier:
    return tier_of(relation)
