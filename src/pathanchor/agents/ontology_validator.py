"""Ontology validator agent.

Ref: Sec. III-D ("Ontology validator (OV) ... applies type signatures, gets rid
of ill-typed paths and unifies entity identity across vocabularies. It produces
the validated set of paths along with a coverage statistic").

The validator never scores a path; it only decides whether the path is a
well-formed object. That separation is what lets an auditor replay the same
filter against the graph and reach the same verdict.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field

from pathanchor.agents.contracts import RetrievalResult, ValidationResult
from pathanchor.graph.index import CompactGraph
from pathanchor.graph.ontology import NodeType
from pathanchor.paths.signatures import (
    EvidencePath,
    PathEdge,
    is_chain,
    reject_reasons,
)

CANONICAL_VOCABULARY: frozenset[NodeType] = frozenset(
    {
        NodeType.DISEASE,
        NodeType.MORPHOLOGY,
        NodeType.ANATOMIC_SITE,
        NodeType.GENE,
        NodeType.PROTEIN,
        NodeType.PROTEIN_CODING_LOCUS,
        NodeType.COMPOUND,
        NodeType.DRUG,
        NodeType.PATHWAY,
        NodeType.INTERACTION_EVIDENCE,
        NodeType.VARIANT,
        NodeType.SOMATIC_MUTATION,
        NodeType.COPY_NUMBER_ALTERATION,
        NodeType.PHENOTYPE,
        NodeType.CLINICAL_VARIABLE,
        NodeType.STAGE,
        NodeType.TREATMENT,
        NodeType.TREATMENT_CLASS,
        NodeType.REGIMEN,
        NodeType.GUIDELINE_STATEMENT,
        NodeType.GUIDELINE_RECOMMENDATION,
    }
)


@dataclass(slots=True)
class OntologyValidator:
    """Signature filter plus cross-vocabulary identity unification."""

    graph: CompactGraph
    aliases: Mapping[str, str] = field(default_factory=dict)
    enforce_signatures: bool = True
    _lookup: dict[str, int] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self._lookup = {key: index for index, key in enumerate(self.graph.keys())}

    def type_of(self, key: str) -> NodeType:
        index = self._lookup.get(key)
        if index is None:
            raise KeyError(f"entity {key} is not in the graph")
        return self.graph.node_type_of(index)

    def canonicalise(self, path: EvidencePath) -> EvidencePath:
        if not self.aliases:
            return path
        edges: list[PathEdge] = []
        for edge in path.edges:
            head = self.aliases.get(edge.head, edge.head)
            tail = self.aliases.get(edge.tail, edge.tail)
            if head == edge.head and tail == edge.tail:
                edges.append(edge)
                continue
            edges.append(
                PathEdge(
                    head=head, relation=edge.relation, tail=tail, tier=edge.tier, weight=edge.weight
                )
            )
        return EvidencePath(tuple(edges))

    def coverage_of(self, paths: list[EvidencePath]) -> float:
        """Share of path endpoints grounded in a vocabulary that fixes identity."""
        total = 0
        canonical = 0
        for path in paths:
            for key in path.nodes():
                total += 1
                if self.type_of(key) in CANONICAL_VOCABULARY:
                    canonical += 1
        if total == 0:
            return 0.0
        return canonical / total

    def run(self, result: RetrievalResult) -> ValidationResult:
        validated: dict[str, tuple[EvidencePath, ...]] = {}
        reasons: dict[str, int] = defaultdict(int)
        rejected = 0
        accepted_paths: list[EvidencePath] = []
        for option, paths in result.proposals.items():
            kept: list[EvidencePath] = []
            for path in paths:
                unified = self.canonicalise(path)
                if not is_chain(unified):
                    reasons["not_a_chain"] += 1
                    rejected += 1
                    continue
                if not self.enforce_signatures:
                    kept.append(unified)
                    continue
                failures = reject_reasons(unified, self.type_of)
                if failures:
                    for failure in failures:
                        reasons[failure.split(":")[0]] += 1
                    rejected += 1
                    continue
                kept.append(unified)
            validated[option] = tuple(kept)
            accepted_paths.extend(kept)
        return ValidationResult(
            patient_id=result.patient_id,
            validated=validated,
            rejected=rejected,
            rejection_reasons=dict(sorted(reasons.items())),
            coverage=self.coverage_of(accepted_paths),
        )


def rejection_profile(results: list[ValidationResult]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for result in results:
        for reason, value in result.rejection_reasons.items():
            counts[reason] += value
    return dict(sorted(counts.items()))


def mean_coverage(results: list[ValidationResult]) -> float:
    if not results:
        return 0.0
    return sum(result.coverage for result in results) / len(results)
