"""Budgeted proposal against complete enumeration."""

from __future__ import annotations

from pathanchor.graph.builder import CensusSpec, build_declared_graph
from pathanchor.graph.grounding import EntityDirectory
from pathanchor.graph.index import CompactGraph
from pathanchor.graph.ontology import NodeType
from pathanchor.paths.enumerate import (
    PathEnumerator,
    ProposalBudget,
    exhaustive_paths,
    paths_by_length,
)


def _probe(tiny_graph: CompactGraph) -> tuple[list[int], int]:
    directory = EntityDirectory.from_graph(tiny_graph, core_size=64)
    anchors = [int(directory.pool(NodeType.CLINICAL_VARIABLE)[0])]
    target = int(directory.pool(NodeType.TREATMENT_CLASS)[0])
    return (anchors, target)


def test_budgeted_is_a_subset_of_complete(tiny_graph: CompactGraph) -> None:
    anchors, target = _probe(tiny_graph)
    enumerator = PathEnumerator(tiny_graph)
    budget = ProposalBudget(max_hops=3, max_per_pair=24, branching=6, min_hops=1, max_frontier=32)
    budgeted = enumerator.propose(anchors, (target,), budget)
    complete = exhaustive_paths(tiny_graph, anchors, target, budget.max_hops)
    assert {path.canonical_key() for path in budgeted} <= {
        path.canonical_key() for path in complete
    }


def test_hop_and_pair_bounds_hold(tiny_graph: CompactGraph) -> None:
    anchors, target = _probe(tiny_graph)
    enumerator = PathEnumerator(tiny_graph)
    budget = ProposalBudget(max_hops=2, max_per_pair=5, branching=4, min_hops=1, max_frontier=16)
    paths = enumerator.propose(anchors, (target,), budget)
    assert len(paths) <= budget.max_per_pair
    assert all(len(path) <= budget.max_hops for path in paths)


def test_every_path_chains(tiny_graph: CompactGraph) -> None:
    anchors, target = _probe(tiny_graph)
    enumerator = PathEnumerator(tiny_graph)
    budget = ProposalBudget(max_hops=3, max_per_pair=16, branching=6, min_hops=1, max_frontier=32)
    for path in enumerator.propose(anchors, (target,), budget):
        assert path.start == tiny_graph.key_of(anchors[0])
        assert path.end == tiny_graph.key_of(target)
        assert len(set(path.nodes())) == len(path.nodes())


def test_stats_report_the_bound(tiny_graph: CompactGraph) -> None:
    anchors, target = _probe(tiny_graph)
    enumerator = PathEnumerator(tiny_graph)
    budget = ProposalBudget(max_hops=3, max_per_pair=8, branching=4, min_hops=1, max_frontier=16)
    paths, stats = enumerator.propose_with_stats(anchors, (target,), budget)
    assert stats.proposals == len(paths)
    assert stats.expanded_edges >= len(paths)


def test_path_length_histogram(tiny_graph: CompactGraph) -> None:
    anchors, target = _probe(tiny_graph)
    enumerator = PathEnumerator(tiny_graph)
    paths = enumerator.propose(
        anchors,
        (target,),
        ProposalBudget(max_hops=3, max_per_pair=16, branching=6, min_hops=1, max_frontier=32),
    )
    histogram = paths_by_length(paths)
    assert sum(histogram.values()) == len(paths)


def test_unknown_endpoint_is_rejected() -> None:
    spec = CensusSpec(entity_total=3000, edge_total=12000, seed=5, core_size=64)
    graph = build_declared_graph(spec)
    enumerator = PathEnumerator(graph)
    paths = enumerator.propose(
        [0],
        (graph.n_entities + 5,),
        ProposalBudget(max_hops=2, branching=2, min_hops=1, max_frontier=8),
    )
    assert paths == []
