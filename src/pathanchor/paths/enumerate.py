"""Budgeted breadth-first proposal of evidence paths.

Ref: Sec. III-C ("breadth-first growth (k <= 4, at most 128 proposals per pair)").

The budget is what makes the proposal tractable on a 7.4-million-edge graph: a
node contributes at most ``branching`` successors per level, chosen by descending
evidence weight, and a pair stops once it holds ``max_per_pair`` proposals. The
exhaustive enumerator beside it ignores both bounds and exists to check the
budgeted one on a graph small enough to walk completely.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from pathanchor.graph.index import CompactGraph
from pathanchor.paths.signatures import EvidencePath, PathEdge, dedupe_paths


@dataclass(frozen=True, slots=True)
class ProposalBudget:
    max_hops: int = 4
    max_per_pair: int = 128
    branching: int = 48
    min_hops: int = 2
    max_frontier: int = 4096

    def validate(self) -> None:
        if self.max_hops < 1:
            raise ValueError("max_hops must be at least one")
        if self.max_per_pair < 1:
            raise ValueError("max_per_pair must be at least one")
        if self.branching < 1:
            raise ValueError("branching must be at least one")
        if self.min_hops < 1 or self.min_hops > self.max_hops:
            raise ValueError("min_hops must lie inside [1, max_hops]")
        if self.max_frontier < 1:
            raise ValueError("max_frontier must be at least one")


@dataclass(frozen=True, slots=True)
class ProposalStats:
    visited_nodes: int
    expanded_edges: int
    truncated: bool
    proposals: int


@dataclass(slots=True)
class _Partial:
    node: int
    nodes: tuple[int, ...]
    edge_ids: tuple[int, ...]
    weight: float


def materialise(graph: CompactGraph, partial: _Partial) -> EvidencePath:
    edges: list[PathEdge] = []
    for slot, edge_id in enumerate(partial.edge_ids):
        edges.append(
            PathEdge(
                head=graph.key_of(partial.nodes[slot]),
                relation=graph.edge_relation_of(edge_id),
                tail=graph.key_of(partial.nodes[slot + 1]),
                tier=graph.edge_tier_of(edge_id),
                weight=graph.edge_weight_of(edge_id),
            )
        )
    return EvidencePath(tuple(edges))


class PathEnumerator:
    """Proposes paths from a patient's anchors to a treatment entity.

    Expansion is level-synchronous and vectorised: one pass gathers every
    successor of the current beam, ranks them inside their source node, and keeps
    the best ``branching`` per node. The per-node Python loop then runs only over
    the surviving children rather than over the whole degree of the frontier,
    which is what keeps a four-hop proposal affordable on a graph with millions
    of edges.
    """

    def __init__(self, graph: CompactGraph) -> None:
        self._graph = graph
        self._tail = graph.edge_arrays()[2]
        self._weight = graph.edge_arrays()[4]

    def successors(self, node: int, branching: int) -> list[int]:
        edge_ids = self._graph.out_edge_ids(node)
        count = int(edge_ids.shape[0])
        if count == 0:
            return []
        if count <= branching:
            return [int(item) for item in edge_ids.tolist()]
        keyed = sorted(
            (int(item) for item in edge_ids.tolist()),
            key=lambda edge_id: (
                -self._graph.edge_weight_of(edge_id),
                self._graph.edge_tail_of(edge_id),
            ),
        )
        return keyed[:branching]

    def _expand(
        self,
        frontier: list[_Partial],
        branching: int,
    ) -> list[_Partial]:
        if not frontier:
            return []
        parts: list[np.ndarray] = []
        origins: list[np.ndarray] = []
        for position, item in enumerate(frontier):
            edge_ids = self._graph.out_edge_ids(item.node)
            if int(edge_ids.shape[0]) == 0:
                continue
            parts.append(edge_ids)
            origins.append(np.full(int(edge_ids.shape[0]), position, dtype=np.int64))
        if not parts:
            return []
        gathered = np.concatenate(parts)
        sources = np.concatenate(origins)
        weights = self._weight[gathered]
        tails = self._tail[gathered]
        order = np.lexsort((tails, -weights, sources))
        ordered_sources = sources[order]
        rank = np.arange(order.shape[0], dtype=np.int64)
        starts = np.searchsorted(ordered_sources, ordered_sources, side="left")
        selected = np.nonzero(rank - starts < branching)[0]
        children: list[_Partial] = []
        for position in [int(item) for item in selected.tolist()]:
            source = int(ordered_sources[position])
            edge_id = int(gathered[order[position]])
            parent = frontier[source]
            tail = int(self._tail[edge_id])
            if tail in parent.nodes:
                continue
            children.append(
                _Partial(
                    tail,
                    (*parent.nodes, tail),
                    (*parent.edge_ids, edge_id),
                    parent.weight + float(self._weight[edge_id]),
                )
            )
        return children

    def propose(
        self,
        anchors: Sequence[int],
        targets: Sequence[int],
        budget: ProposalBudget,
    ) -> list[EvidencePath]:
        paths, _ = self.propose_with_stats(anchors, targets, budget)
        return paths

    def propose_with_stats(
        self,
        anchors: Sequence[int],
        targets: Sequence[int],
        budget: ProposalBudget,
    ) -> tuple[list[EvidencePath], ProposalStats]:
        budget.validate()
        target_set = {int(index) for index in targets}
        starts = [int(index) for index in dict.fromkeys(anchors)]
        frontier: list[_Partial] = [_Partial(node, (node,), (), 0.0) for node in starts]
        found: list[_Partial] = []
        visited = set(starts)
        expanded = 0
        truncated = False
        for hop in range(1, budget.max_hops + 1):
            if not frontier:
                break
            following: list[_Partial] = []
            for child in self._expand(frontier, budget.branching):
                expanded += 1
                tail = child.node
                visited.add(tail)
                if tail in target_set and hop >= budget.min_hops:
                    found.append(child)
                elif hop < budget.max_hops:
                    following.append(child)
            if len(following) > budget.max_frontier:
                following.sort(key=lambda item: (-item.weight, item.nodes[-1]))
                following = following[: budget.max_frontier]
                truncated = True
            if len(found) >= budget.max_per_pair:
                truncated = True
                break
            frontier = following
        found.sort(key=lambda item: (-item.weight, len(item.nodes), item.nodes[-1]))
        if len(found) > budget.max_per_pair:
            truncated = True
        capped = found[: budget.max_per_pair]
        paths = dedupe_paths([materialise(self._graph, item) for item in capped])
        stats = ProposalStats(
            visited_nodes=len(visited),
            expanded_edges=expanded,
            truncated=truncated,
            proposals=len(paths),
        )
        return (paths, stats)


def exhaustive_paths(
    graph: CompactGraph,
    anchors: Sequence[int],
    target: int,
    max_hops: int,
) -> list[EvidencePath]:
    """Every simple path from an anchor to one target, with no branching bound.

    This is the reference the budgeted proposer is checked against; it is usable
    only on a graph small enough to walk completely.
    """
    results: list[_Partial] = []

    def walk(node: int, nodes: tuple[int, ...], edge_ids: tuple[int, ...], total: float) -> None:
        if len(edge_ids) >= max_hops:
            return
        for edge_id in [int(item) for item in graph.out_edge_ids(node).tolist()]:
            nxt = graph.edge_tail_of(edge_id)
            if nxt in nodes:
                continue
            extended_nodes = (*nodes, nxt)
            extended_edges = (*edge_ids, edge_id)
            extended_weight = total + graph.edge_weight_of(edge_id)
            if nxt == target:
                results.append(_Partial(nxt, extended_nodes, extended_edges, extended_weight))
            else:
                walk(nxt, extended_nodes, extended_edges, extended_weight)

    for anchor in dict.fromkeys(int(item) for item in anchors):
        walk(anchor, (anchor,), (), 0.0)
    results.sort(key=lambda item: (-item.weight, len(item.nodes), item.nodes[-1]))
    return dedupe_paths([materialise(graph, item) for item in results])


def paths_by_length(paths: Sequence[EvidencePath]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for path in paths:
        counts[len(path)] = counts.get(len(path), 0) + 1
    return dict(sorted(counts.items()))
