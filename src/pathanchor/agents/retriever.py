"""Retriever agent.

Ref: Sec. III-D ("Retriever (RE) ... collects information about the patient's
health, finds possible directions, and returns the proposed course").

The retriever anchors the record on the graph, expands within the proposal
budget towards each feasible option, and returns typed paths. It never returns a
sentence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from pathanchor.agents.contracts import RetrievalRequest, RetrievalResult
from pathanchor.graph.index import CompactGraph
from pathanchor.paths.enumerate import PathEnumerator, ProposalBudget, ProposalStats
from pathanchor.paths.signatures import EvidencePath


@dataclass(slots=True)
class Retriever:
    """Proposes candidate evidence paths for every option of a request."""

    graph: CompactGraph
    targets: Mapping[str, int]
    budget: ProposalBudget
    _enumerator: PathEnumerator = field(init=False)

    def __post_init__(self) -> None:
        self._enumerator = PathEnumerator(self.graph)

    def run(self, request: RetrievalRequest) -> RetrievalResult:
        proposals: dict[str, tuple[EvidencePath, ...]] = {}
        visited = 0
        expanded = 0
        truncated = False
        total = 0
        for option in request.options:
            target = self.targets.get(option)
            if target is None:
                proposals[option] = ()
                continue
            paths, stats = self._enumerator.propose_with_stats(
                request.anchors, (target,), self.budget
            )
            proposals[option] = tuple(paths)
            visited += stats.visited_nodes
            expanded += stats.expanded_edges
            truncated = truncated or stats.truncated
            total += stats.proposals
        combined = ProposalStats(
            visited_nodes=visited,
            expanded_edges=expanded,
            truncated=truncated,
            proposals=total,
        )
        return RetrievalResult(
            patient_id=request.patient_id,
            proposals=proposals,
            anchor_count=len(request.anchors),
            stats=combined,
        )

    def options(self) -> tuple[str, ...]:
        return tuple(self.targets.keys())


def collect_proposals(
    results: Sequence[RetrievalResult],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        for option, paths in result.proposals.items():
            counts[option] = counts.get(option, 0) + len(paths)
    return dict(sorted(counts.items()))
