"""Node types, relation types and their signatures."""

from __future__ import annotations

import pytest

from pathanchor.graph.ontology import (
    NODE_TYPE_SOURCE,
    NODE_TYPES,
    RELATION_TYPES,
    SIGNATURES,
    EvidenceTier,
    NodeType,
    RelationType,
    is_admissible,
    tier_of,
    validate_ontology,
)


def test_declared_counts() -> None:
    assert len(NODE_TYPES) == 41
    assert len(RELATION_TYPES) == 42


def test_every_relation_has_a_signature() -> None:
    assert set(SIGNATURES) == set(RELATION_TYPES)


def test_every_node_type_declares_a_source() -> None:
    assert set(NODE_TYPES) == set(NODE_TYPE_SOURCE)


def test_validation_is_idempotent() -> None:
    validate_ontology()


def test_targets_signature_rejects_an_ontology_class_head() -> None:
    assert is_admissible(RelationType.TARGETS, NodeType.DRUG, NodeType.GENE)
    assert not is_admissible(RelationType.TARGETS, NodeType.ONTOLOGY_CLASS, NodeType.GENE)


def test_subsumed_by_is_reflexive_over_ontology_classes() -> None:
    assert is_admissible(RelationType.SUBSUMED_BY, NodeType.ONTOLOGY_CLASS, NodeType.ONTOLOGY_CLASS)


def test_tiers_are_declared_per_relation() -> None:
    tiers = {tier_of(relation) for relation in RELATION_TYPES}
    assert EvidenceTier.GUIDELINE in tiers
    assert EvidenceTier.INFERRED not in tiers


@pytest.mark.parametrize("relation", list(RelationType))
def test_signature_sides_are_non_empty(relation: RelationType) -> None:
    signature = SIGNATURES[relation]
    assert signature.head and signature.tail
