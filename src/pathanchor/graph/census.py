"""Census reporting and the frozen graph snapshot.

Ref: Sec. III-B (386,204 entities in 41 node types; 7,431,905 relations in 42
relation types).
"""

from __future__ import annotations

import io
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pathanchor.graph.builder import CensusSpec
from pathanchor.graph.index import CompactGraph
from pathanchor.graph.kg import GraphCensus
from pathanchor.utils.serialization import write_bytes


def census_discrepancies(census: GraphCensus, spec: CensusSpec) -> list[str]:
    """Report every place the built graph misses its declared size."""
    problems: list[str] = []
    if census.entities != spec.entity_total:
        problems.append(f"entities {census.entities} != declared {spec.entity_total}")
    if census.edges != spec.edge_total:
        problems.append(f"edges {census.edges} != declared {spec.edge_total}")
    if census.node_types != 41:
        problems.append(f"node types {census.node_types} != declared 41")
    if census.relation_types != 42:
        problems.append(f"relation types {census.relation_types} != declared 42")
    return problems


def save_snapshot(graph: CompactGraph, path: Path) -> None:
    """Write the graph snapshot.

    The snapshot is written through the byte-level atomic writer because a
    compressed archive has no text form; the array order is fixed so two runs
    produce identical bytes.
    """
    head, relation, tail, tier, weight = graph.edge_arrays()
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        keys=np.asarray(graph.keys(), dtype=np.str_),
        entity_types=graph.entity_type_array(),
        entity_sources=graph.entity_source_array(),
        head=head,
        relation=relation,
        tail=tail,
        tier=tier,
        weight=weight,
    )
    write_bytes(path, buffer.getvalue())


def load_snapshot(path: Path) -> CompactGraph:
    """Read a snapshot. Keys are stored as a fixed-width unicode array, so no
    pickle is involved and a foreign archive cannot execute on load."""
    with np.load(path, allow_pickle=False) as payload:
        keys = [str(item) for item in payload["keys"].tolist()]
        return CompactGraph(
            keys,
            payload["entity_types"],
            payload["entity_sources"],
            payload["head"],
            payload["relation"],
            payload["tail"],
            payload["tier"],
            payload["weight"],
        )


def snapshot_digest(graph: CompactGraph) -> str:
    """Digest of the tensor payload, not of the container bytes.

    A compressed archive embeds no index, but a plain ``.npz`` write still lays
    out the members in call order; digesting the payload makes the value
    independent of the archive layer.
    """
    head, relation, tail, tier, weight = graph.edge_arrays()
    digest = np.zeros(0, dtype=np.float64)
    for part in (
        head.astype(np.float64),
        relation.astype(np.float64),
        tail.astype(np.float64),
        tier.astype(np.float64),
        weight,
    ):
        digest = np.concatenate([digest, np.asarray([float(part.sum()), float(part.shape[0])])])
    return "|".join(f"{value:.6f}" for value in digest.tolist())


@dataclass(frozen=True, slots=True)
class CensusReport:
    entities: int
    edges: int
    node_types: int
    relation_types: int
    entities_per_type: Mapping[str, int]
    edges_per_relation: Mapping[str, int]
    edges_per_tier: Mapping[str, int]
    discrepancies: tuple[str, ...]

    @property
    def exact(self) -> bool:
        return not self.discrepancies


def census_report(census: GraphCensus, spec: CensusSpec) -> CensusReport:
    return CensusReport(
        entities=census.entities,
        edges=census.edges,
        node_types=census.node_types,
        relation_types=census.relation_types,
        entities_per_type=census.entities_per_type,
        edges_per_relation=census.edges_per_relation,
        edges_per_tier=census.edges_per_tier,
        discrepancies=tuple(census_discrepancies(census, spec)),
    )


def top_relations(census: GraphCensus, limit: int = 8) -> Sequence[tuple[str, int]]:
    ordered = sorted(census.edges_per_relation.items(), key=lambda item: (-item[1], item[0]))
    return ordered[:limit]
