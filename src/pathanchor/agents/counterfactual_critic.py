"""Counterfactual critic agent.

Ref: Sec. III-D ("Counterfactual critic (CC) forms erroneous questions -- routes
that should be present if its suggested advice was inaccurate -- and reduces the
influence of routes where the relevant neighbourhood does not pass the question").

The falsifying query for a route is its relation sequence: if the same sequence
also reaches a competing treatment option from the same patient, the route does
not separate the options and its influence is divided by the number of options it
reaches. That turns self-critique from a prompting strategy into a countable
operation.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from pathanchor.agents.contracts import CritiqueResult
from pathanchor.paths.signatures import EvidencePath


@dataclass(frozen=True, slots=True)
class FalsifyingQuery:
    """The route that would have to exist for a recommendation to be wrong."""

    signature: str
    reaches: tuple[str, ...]

    def falsifies(self, option: str) -> bool:
        return option in self.reaches and len(self.reaches) > 1


def build_queries(
    validated: Mapping[str, Sequence[EvidencePath]],
) -> dict[str, FalsifyingQuery]:
    reached: dict[str, set[str]] = defaultdict(set)
    for option, paths in validated.items():
        for path in paths:
            reached[path.canonical_key()].add(option)
    return {
        signature: FalsifyingQuery(signature=signature, reaches=tuple(sorted(options)))
        for signature, options in reached.items()
    }


def influence_factors(
    validated: Mapping[str, Sequence[EvidencePath]],
    queries: Mapping[str, FalsifyingQuery],
) -> dict[str, tuple[tuple[EvidencePath, float], ...]]:
    weighted: dict[str, tuple[tuple[EvidencePath, float], ...]] = {}
    for option, paths in validated.items():
        items: list[tuple[EvidencePath, float]] = []
        for path in paths:
            query = queries.get(path.canonical_key())
            if query is None:
                items.append((path, 1.0))
                continue
            if not query.falsifies(option):
                items.append((path, 1.0))
                continue
            penalty = 1.0 / float(len(query.reaches))
            items.append((path, penalty))
        weighted[option] = tuple(items)
    return weighted


@dataclass(slots=True)
class CounterfactualCritic:
    """Reweights validated paths by whether they separate the options."""

    def run(
        self,
        patient_id: str,
        anchors: Sequence[int],
        options: Sequence[str],
        validated: Mapping[str, tuple[EvidencePath, ...]],
    ) -> CritiqueResult:
        del anchors
        subset = {option: validated.get(option, ()) for option in options}
        queries = build_queries(subset)
        weighted = influence_factors(subset, queries)
        downweighted = sum(1 for items in weighted.values() for _, factor in items if factor < 1.0)
        non_discriminative = tuple(
            sorted(signature for signature, query in queries.items() if len(query.reaches) > 1)
        )
        return CritiqueResult(
            patient_id=patient_id,
            weighted=weighted,
            downweighted=downweighted,
            non_discriminative_signatures=non_discriminative,
        )


def weighted_option_paths(
    critique: CritiqueResult,
    option: str,
    threshold: float = 1.0,
) -> tuple[EvidencePath, ...]:
    """Paths the critic leaves at full influence for one option."""
    items = critique.weighted.get(option, ())
    return tuple(path for path, factor in items if factor >= threshold)


def suppress_factor_distribution(critiques: Sequence[CritiqueResult]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for critique in critiques:
        for items in critique.weighted.values():
            for _, factor in items:
                counts[f"{factor:.4f}"] += 1
    return dict(sorted(counts.items()))
