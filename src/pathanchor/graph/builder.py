"""Construction of the released graph to its declared census.

Ref: Sec. III-B (386,204 entities in 41 node types; 7,431,905 relations in 42
relation types, each labelled by direction and evidence tier).

The per-type allocation is an engineering default: the manuscript states the two
totals and the two type counts but not the breakdown between them, so the
allocation below is declared in ``configs/graph/census.yaml`` and reported as
such rather than inferred.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

import numpy as np

from pathanchor.graph.index import (
    NODE_TYPE_IDS,
    RELATION_TYPE_IDS,
    SOURCE_IDS,
    TIER_IDS,
    CompactGraph,
)
from pathanchor.graph.ontology import (
    NODE_TYPE_SOURCE,
    NODE_TYPES,
    RELATION_TYPES,
    SIGNATURES,
    EvidenceTier,
    NodeType,
    RelationType,
)
from pathanchor.utils.numerics import allocate_counts
from pathanchor.utils.types import IntArray

_TYPE_PREFIX: Mapping[NodeType, str] = {
    NodeType.DISEASE: "DOID",
    NodeType.MORPHOLOGY: "ICDO",
    NodeType.ANATOMIC_SITE: "TOPO",
    NodeType.GENE: "HGNC",
    NodeType.PROTEIN: "UNIPROT",
    NodeType.PROTEIN_CODING_LOCUS: "LOCUS",
    NodeType.COMPOUND: "DB",
    NodeType.DRUG: "DRUG",
    NodeType.PATHWAY: "KEGG",
    NodeType.BIOLOGICAL_PROCESS: "GOBP",
    NodeType.MOLECULAR_FUNCTION: "GOMF",
    NodeType.CELLULAR_COMPONENT: "GOCC",
    NodeType.INTERACTION_EVIDENCE: "STRG",
    NodeType.VARIANT: "VAR",
    NodeType.SOMATIC_MUTATION: "MUT",
    NodeType.COPY_NUMBER_ALTERATION: "CNA",
    NodeType.EXPRESSION_SIGNATURE: "SIG",
    NodeType.GENE_EXPRESSION_VALUE: "EXPV",
    NodeType.CLINICAL_VARIABLE: "CLIN",
    NodeType.STAGE: "STAGE",
    NodeType.PERFORMANCE_STATUS: "ECOG",
    NodeType.HISTOLOGY_DESCRIPTOR: "HIST",
    NodeType.DERMATOSCOPY_FEATURE: "DERM",
    NodeType.WSI_TILE: "WSIT",
    NodeType.WSI_FEATURE: "WSIF",
    NodeType.TREATMENT: "TREAT",
    NodeType.TREATMENT_CLASS: "TCLASS",
    NodeType.REGIMEN: "REGM",
    NodeType.GUIDELINE_RECOMMENDATION: "NCCNR",
    NodeType.GUIDELINE_STATEMENT: "NCCNS",
    NodeType.FEASIBILITY_CONSTRAINT: "CONSTR",
    NodeType.EVIDENCE_TIER: "TIER",
    NodeType.ONTOLOGY_CLASS: "ONT",
    NodeType.PHENOTYPE: "HP",
    NodeType.IMMUNE_CELL_POPULATION: "CELL",
    NodeType.CYTOKINE: "CK",
    NodeType.RECEPTOR: "RECP",
    NodeType.TRANSPORT_PROTEIN: "TRNP",
    NodeType.ENZYME: "EC",
    NodeType.DRUG_TARGET: "TGT",
    NodeType.RESPONSE_CRITERION: "RESP",
}

# Mean edge weight by evidence tier. An engineering default: the manuscript
# declares an evidence tier per relation but not a numeric strength.
_TIER_SCALE: Mapping[EvidenceTier, float] = {
    EvidenceTier.GUIDELINE: 1.00,
    EvidenceTier.CURATED: 0.80,
    EvidenceTier.EXPERIMENTAL: 0.55,
    EvidenceTier.INFERRED: 0.30,
}


@dataclass(frozen=True, slots=True)
class CensusSpec:
    """The declared size of the released graph and the allocation weights."""

    entity_total: int
    edge_total: int
    seed: int
    node_type_weights: Mapping[str, float] = field(default_factory=dict)
    relation_weights: Mapping[str, float] = field(default_factory=dict)
    core_size: int = 1024

    def weight_for_node(self, node_type: NodeType) -> float:
        return float(self.node_type_weights.get(node_type.value, 1.0))

    def weight_for_relation(self, relation: RelationType) -> float:
        return float(self.relation_weights.get(relation.value, 1.0))


# Relations whose endpoints are drawn from the mechanistic core rather than from
# the whole type pool. A knowledge graph is not uniformly annotated: only a small
# fraction of its entities carry treatment-response evidence, and an anchor drawn
# from the whole pool would leave the patient's record with no admissible route
# to any treatment at all. The core is a declared engineering default; the
# manuscript states the census totals and the type counts, not this split.
CORE_RELATIONS: frozenset[RelationType] = frozenset(
    {
        RelationType.VARIANT_OF,
        RelationType.ENCODES,
        RelationType.PREDICTS_RESPONSE_TO,
        RelationType.ASSOCIATED_WITH,
        RelationType.CHARACTERIZES,
        RelationType.FEASIBLE_IN_STATE,
        RelationType.RECOMMENDED_FOR,
        RelationType.MEASURED_AS,
        RelationType.CONSTRAINS,
    }
)


def _distinct_pairs(
    head_pool: IntArray,
    tail_pool: IntArray,
    count: int,
    rng: np.random.Generator,
) -> tuple[IntArray, IntArray]:
    """Draw ``count`` distinct endpoint pairs that are not self-loops.

    Endpoint pairs are folded into one integer so the uniqueness test is a single
    ``np.unique`` over the batch rather than a Python set over millions of rows.
    """
    if count == 0:
        return (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64))
    if head_pool.shape[0] == 0 or tail_pool.shape[0] == 0:
        raise ValueError("an edge pool is empty; the census allocation is too small")
    if head_pool.shape[0] * tail_pool.shape[0] < count:
        raise ValueError(
            f"cannot draw {count} distinct edges from pools of size "
            f"{head_pool.shape[0]}x{tail_pool.shape[0]}"
        )
    keys: list[IntArray] = []
    total = 0
    for _ in range(14):
        if total >= count:
            break
        draw = max(2 * (count - total), 4096)
        head = head_pool[rng.integers(0, head_pool.shape[0], size=draw)]
        tail = tail_pool[rng.integers(0, tail_pool.shape[0], size=draw)]
        keep = head != tail
        folded = head[keep].astype(np.int64) * _FOLD + tail[keep]
        fresh = np.unique(folded)
        if keys:
            fresh = np.setdiff1d(fresh, np.concatenate(keys), assume_unique=True)
        if fresh.shape[0] == 0:
            continue
        fresh = fresh[: count - total]
        keys.append(fresh)
        total += fresh.shape[0]
    if total < count:
        raise ValueError(
            f"drew {total} of {count} distinct edges from pools of size "
            f"{head_pool.shape[0]}x{tail_pool.shape[0]}"
        )
    joined = np.concatenate(keys) if len(keys) > 1 else keys[0]
    return (joined // _FOLD, joined % _FOLD)


def build_declared_graph(spec: CensusSpec) -> CompactGraph:
    """Materialise the released graph to ``spec``."""
    counts = allocate_counts(
        spec.entity_total,
        [spec.weight_for_node(item) for item in NODE_TYPES],
        floor=max(64, spec.entity_total // (8 * len(NODE_TYPES))),
    )
    keys: list[str] = []
    types: list[int] = []
    sources: list[int] = []
    pool_by_type: dict[NodeType, np.ndarray] = {}
    for node_type, count in zip(NODE_TYPES, counts, strict=True):
        start = len(keys)
        prefix = _TYPE_PREFIX[node_type]
        keys.extend(f"{prefix}:{index:07d}" for index in range(count))
        types.extend([NODE_TYPE_IDS[node_type.value]] * count)
        sources.extend([SOURCE_IDS[NODE_TYPE_SOURCE[node_type].value]] * count)
        pool_by_type[node_type] = np.arange(start, start + count, dtype=np.int32)

    relation_counts = allocate_counts(
        spec.edge_total,
        [spec.weight_for_relation(item) for item in RELATION_TYPES],
        floor=max(64, spec.edge_total // (8 * len(RELATION_TYPES))),
    )
    rng = np.random.default_rng(spec.seed)
    heads: list[np.ndarray] = []
    relations: list[np.ndarray] = []
    tails: list[np.ndarray] = []
    tiers: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for relation, count in zip(RELATION_TYPES, relation_counts, strict=True):
        signature = SIGNATURES[relation]
        head_pool = _pool_for(
            signature.head, pool_by_type, spec.core_size if relation in CORE_RELATIONS else None
        )
        tail_pool = _pool_for(
            signature.tail, pool_by_type, spec.core_size if relation in CORE_RELATIONS else None
        )
        head, tail = _distinct_pairs(head_pool, tail_pool, count, rng)
        scale = _TIER_SCALE[signature.tier]
        draw = rng.gamma(shape=2.0, scale=scale / 2.0, size=head.shape[0])
        heads.append(head.astype(np.int32))
        tails.append(tail.astype(np.int32))
        relations.append(np.full(head.shape[0], RELATION_TYPE_IDS[relation.value], dtype=np.int16))
        tiers.append(np.full(head.shape[0], TIER_IDS[signature.tier.value], dtype=np.int16))
        weights.append(np.clip(draw, 1e-4, 1.0).astype(np.float32))

    return CompactGraph(
        keys,
        np.asarray(types, dtype=np.int16),
        np.asarray(sources, dtype=np.int16),
        np.concatenate(heads),
        np.concatenate(relations),
        np.concatenate(tails),
        np.concatenate(tiers),
        np.concatenate(weights),
    )


_FOLD: int = 1_000_003


def _pool_for(
    node_types: Iterable[NodeType],
    pool_by_type: Mapping[NodeType, np.ndarray],
    cap: int | None,
) -> np.ndarray:
    """Concatenate the index pools of a signature side, optionally capped.

    The cap selects the mechanistic core: the first ``cap`` entities of each type,
    which are the ones the mechanism relations are drawn from. Both sides of a core
    relation use the same cap, so a route from an anchor to a treatment stays
    inside the core and does not have to pass through an unannotated node.
    """
    parts: list[np.ndarray] = []
    for node_type in sorted(node_types, key=lambda item: item.value):
        pool = pool_by_type[node_type]
        parts.append(pool[:cap] if cap is not None else pool)
    return np.concatenate(parts)


def spec_from_mapping(data: Mapping[str, object]) -> CensusSpec:
    node_weights = data.get("node_type_weights")
    relation_weights = data.get("relation_weights")
    return CensusSpec(
        entity_total=int(_as_number(data["entity_total"])),
        edge_total=int(_as_number(data["edge_total"])),
        seed=int(_as_number(data["seed"])) if "seed" in data else 20260101,
        node_type_weights=_numeric_mapping(node_weights),
        relation_weights=_numeric_mapping(relation_weights),
        core_size=int(_as_number(data["core_size"])) if "core_size" in data else 1024,
    )


def _as_number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"expected a number in the census specification, got {type(value)!r}")
    return float(value)


def _numeric_mapping(value: object) -> dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): float(str(item)) for key, item in value.items()}
