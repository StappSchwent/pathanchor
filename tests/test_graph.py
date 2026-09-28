"""Graph construction, census and snapshot round trip."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pathanchor.graph.builder import CensusSpec, build_declared_graph
from pathanchor.graph.census import (
    census_report,
    load_snapshot,
    save_snapshot,
    snapshot_digest,
)
from pathanchor.graph.index import CompactGraph
from pathanchor.graph.kg import Edge, Entity, GraphError, KnowledgeGraph
from pathanchor.graph.ontology import EvidenceTier, NodeType, RelationType, SourceOntology
from pathanchor.utils.numerics import allocate_counts


def _entity(key: str, node_type: NodeType) -> Entity:
    return Entity(key=key, node_type=node_type, label=key, source=SourceOntology.HGNC)


def test_census_matches_the_declaration() -> None:
    spec = CensusSpec(entity_total=3000, edge_total=12000, seed=3, core_size=32)
    graph = build_declared_graph(spec)
    census = graph.census()
    assert census.entities == spec.entity_total
    assert census.edges == spec.edge_total
    assert census.node_types == 41
    assert census.relation_types == 42
    assert census_report(census, spec).exact


def test_ill_typed_edge_is_rejected() -> None:
    graph = KnowledgeGraph()
    graph.add_entity(_entity("disease", NodeType.DISEASE))
    graph.add_entity(_entity("gene", NodeType.GENE))
    with pytest.raises(GraphError):
        graph.add_edge(Edge("disease", RelationType.TARGETS, "gene", EvidenceTier.CURATED))


def test_duplicate_edge_is_ignored() -> None:
    graph = KnowledgeGraph()
    graph.add_entity(_entity("drug", NodeType.DRUG))
    graph.add_entity(_entity("gene", NodeType.GENE))
    edge = Edge("drug", RelationType.TARGETS, "gene", EvidenceTier.CURATED)
    graph.add_edge(edge)
    graph.add_edge(edge)
    assert graph.n_edges == 1


def test_builder_is_deterministic() -> None:
    spec = CensusSpec(entity_total=3000, edge_total=12000, seed=11, core_size=32)
    first = build_declared_graph(spec)
    second = build_declared_graph(spec)
    assert first.keys() == second.keys()
    assert snapshot_digest(first) == snapshot_digest(second)


def test_snapshot_round_trip(tmp_path: Path, tiny_graph: CompactGraph) -> None:
    path = tmp_path / "snapshot.npz"
    save_snapshot(tiny_graph, path)
    restored = load_snapshot(path)
    assert restored.n_entities == tiny_graph.n_entities
    assert restored.n_edges == tiny_graph.n_edges
    assert restored.key_of(1) == tiny_graph.key_of(1)
    np.testing.assert_array_equal(restored.edge_arrays()[0], tiny_graph.edge_arrays()[0])


def test_materialise_returns_a_typed_subgraph(tiny_graph: CompactGraph) -> None:
    indices = list(range(40))
    subgraph = tiny_graph.materialise(indices)
    assert subgraph.n_entities <= len(indices)
    for entity in subgraph.entities():
        assert entity.node_type in set(NodeType)


def test_largest_remainder_preserves_the_total() -> None:
    allocated = allocate_counts(100, [1.0, 2.0, 3.0, 4.0], floor=0)
    assert sum(allocated) == 100
    assert allocated == sorted(allocated)
