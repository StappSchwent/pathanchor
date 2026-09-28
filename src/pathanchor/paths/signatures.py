"""Typed evidence paths and their validity notion.

Ref: Sec. III-C (relations declared on node types; a path is admissible only if
it is type-consistent).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from pathanchor.graph.ontology import EvidenceTier, NodeType, RelationType, is_admissible


@dataclass(frozen=True, slots=True)
class PathEdge:
    head: str
    relation: RelationType
    tail: str
    tier: EvidenceTier
    weight: float = 1.0


@dataclass(frozen=True, slots=True)
class EvidencePath:
    """An ordered walk in the graph.

    The path is the output artefact of the reasoning layer: it can be replayed
    against the graph by a third party, which is what makes the justification
    checkable without access to the model (Sec. III-A, evidentiary accountability).
    """

    edges: tuple[PathEdge, ...]

    def __post_init__(self) -> None:
        if len(self.edges) == 0:
            raise ValueError("an evidence path needs at least one edge")

    def __len__(self) -> int:
        return len(self.edges)

    @property
    def start(self) -> str:
        return self.edges[0].head

    @property
    def end(self) -> str:
        return self.edges[-1].tail

    def nodes(self) -> tuple[str, ...]:
        return (self.edges[0].head, *tuple(edge.tail for edge in self.edges))

    def relations(self) -> tuple[RelationType, ...]:
        return tuple(edge.relation for edge in self.edges)

    def weights(self) -> tuple[float, ...]:
        return tuple(edge.weight for edge in self.edges)

    def min_tier(self) -> EvidenceTier:
        order = [
            EvidenceTier.INFERRED,
            EvidenceTier.EXPERIMENTAL,
            EvidenceTier.CURATED,
            EvidenceTier.GUIDELINE,
        ]
        weakest = min(self.edges, key=lambda edge: order.index(edge.tier))
        return weakest.tier

    def canonical_key(self) -> str:
        """A content key that ignores endpoint identity.

        Two paths with the same relation sequence and the same tiers produce the
        same key, which is what lets the counterfactual critic talk about a route
        without naming a patient entity.
        """
        parts = [f"{edge.relation.value}:{edge.tier.value}" for edge in self.edges]
        return ">".join(parts)


def is_chain(path: EvidencePath) -> bool:
    return all(
        path.edges[index].tail == path.edges[index + 1].head for index in range(len(path.edges) - 1)
    )


def is_type_consistent(path: EvidencePath, type_of: Callable[[str], NodeType]) -> bool:
    for edge in path.edges:
        if not is_admissible(edge.relation, type_of(edge.head), type_of(edge.tail)):
            return False
    return True


def admissible(path: EvidencePath, type_of: Callable[[str], NodeType]) -> bool:
    """A path is admissible when it is a chain and every edge respects its
    relation signature. Both conditions are machine-checkable against the graph
    alone, so the audit does not require the scorer."""
    return is_chain(path) and is_type_consistent(path, type_of)


def admissible_signature_only(path: EvidencePath, type_of: Callable[[str], NodeType]) -> bool:
    """The ontology validator's own test, before chaining is considered."""
    return is_type_consistent(path, type_of)


def reject_reasons(path: EvidencePath, type_of: Callable[[str], NodeType]) -> tuple[str, ...]:
    reasons: list[str] = []
    if not is_chain(path):
        reasons.append("not_a_chain")
    for index, edge in enumerate(path.edges):
        if not is_admissible(edge.relation, type_of(edge.head), type_of(edge.tail)):
            reasons.append(f"ill_typed_edge@{index}:{edge.relation.value}")
    return tuple(reasons)


def path_lengths(paths: Sequence[EvidencePath]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for path in paths:
        counts[len(path)] = counts.get(len(path), 0) + 1
    return dict(sorted(counts.items()))


def dedupe_paths(paths: Sequence[EvidencePath]) -> list[EvidencePath]:
    seen: set[tuple[tuple[str, str, str], ...]] = set()
    out: list[EvidencePath] = []
    for path in paths:
        key = tuple((edge.head, edge.relation.value, edge.tail) for edge in path.edges)
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def describe(path: EvidencePath) -> tuple[tuple[str, str], ...]:
    return tuple((edge.relation.value, edge.tier.value) for edge in path.edges)
