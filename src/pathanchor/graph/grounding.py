"""Grounding of a patient record onto typed graph entities.

Ref: Sec. III-B (Patient subgraphs), Sec. III-D (the retriever's anchor set).

A record grounds to entities of the type each field belongs to, chosen by a
stable digest of the field label rather than by a random draw, so the same
record always anchors at the same entities and a rerun is byte-identical.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from hashlib import sha256

import numpy as np

from pathanchor.cohort.schema import PatientRecord
from pathanchor.graph.index import NODE_TYPE_IDS, CompactGraph
from pathanchor.graph.kg import GraphError, PatientSubgraph
from pathanchor.graph.ontology import NodeType


@dataclass(slots=True)
class EntityDirectory:
    """Indices of the entities of each type, so grounding is a lookup.

    ``core_size`` restricts the lookup to the first entities of each type, which
    are the ones the mechanism relations are drawn from. A record whose anchor sat
    outside the core would have no admissible route to any treatment, so the
    anchoring and the construction have to agree on the same core.
    """

    _pools: dict[NodeType, np.ndarray] = field(default_factory=dict)
    core_size: int = 1024

    @classmethod
    def from_graph(cls, graph: CompactGraph, core_size: int = 1024) -> EntityDirectory:
        pools: dict[NodeType, np.ndarray] = {}
        codes = graph.entity_type_array()
        for node_type in NodeType:
            code = NODE_TYPE_IDS[node_type.value]
            pools[node_type] = np.nonzero(codes == code)[0].astype(np.int32)
        return cls(_pools=pools, core_size=core_size)

    def pool(self, node_type: NodeType) -> np.ndarray:
        values = self._pools.get(node_type)
        if values is None or values.shape[0] == 0:
            raise GraphError(f"graph holds no entity of type {node_type.value}")
        return values[: self.core_size] if self.core_size > 0 else values

    def pick(self, node_type: NodeType, token: str) -> int:
        values = self.pool(node_type)
        digest = sha256(f"{node_type.value}|{token}".encode()).digest()
        slot = int.from_bytes(digest[:8], "big") % int(values.shape[0])
        return int(values[slot])

    def keys(
        self, node_type: NodeType, tokens: Sequence[str], graph: CompactGraph
    ) -> tuple[str, ...]:
        return tuple(graph.key_of(self.pick(node_type, token)) for token in tokens)


_MOLECULAR_FIELD_TYPE: Mapping[str, NodeType] = {
    "braf_v600": NodeType.SOMATIC_MUTATION,
    "nras": NodeType.SOMATIC_MUTATION,
    "nf1": NodeType.COPY_NUMBER_ALTERATION,
    "tmb": NodeType.CLINICAL_VARIABLE,
    "pdl1": NodeType.CLINICAL_VARIABLE,
}


def ground_record(
    graph: CompactGraph,
    directory: EntityDirectory,
    record: PatientRecord,
) -> PatientSubgraph:
    """Project one record onto the shared graph.

    The four groups correspond to the four modalities of the evidence record
    (x_clin, x_mol, x_hist, x_img); a modality the record does not carry
    contributes no anchor rather than a stand-in for one.
    """
    anchors: dict[str, tuple[str, ...]] = {}

    molecular: list[str] = []
    for name, present in record.molecular_flags().items():
        if not present:
            continue
        node_type = _MOLECULAR_FIELD_TYPE.get(name, NodeType.VARIANT)
        molecular.append(graph.key_of(directory.pick(node_type, f"{record.patient_id}|{name}")))
    if molecular:
        anchors["molecular"] = tuple(molecular)

    clinical_tokens = [
        f"{record.patient_id}|stage",
        f"{record.patient_id}|age",
        f"{record.patient_id}|sex",
        f"{record.patient_id}|ecog",
    ]
    anchors["clinical"] = directory.keys(NodeType.CLINICAL_VARIABLE, clinical_tokens, graph)

    if record.has_pathology:
        anchors["histology"] = directory.keys(
            NodeType.HISTOLOGY_DESCRIPTOR,
            [f"{record.patient_id}|histology", f"{record.patient_id}|morphology"],
            graph,
        )
    if record.has_imaging:
        anchors["image"] = directory.keys(NodeType.WSI_FEATURE, [f"{record.patient_id}|wsi"], graph)
    return PatientSubgraph(patient_id=record.patient_id, anchors=anchors)


def anchor_indices(graph: CompactGraph, subgraph: PatientSubgraph) -> list[int]:
    """Resolve a subgraph's anchors back to indices, for the compact traversal."""
    lookup = {key: index for index, key in enumerate(graph.keys())}
    missing = [key for key in subgraph if key not in lookup]
    if missing:
        joined = ", ".join(sorted(missing)[:6])
        raise GraphError(f"patient subgraph anchors absent from the compact graph: {joined}")
    return [lookup[key] for key in subgraph]


def ground_treatments(
    graph: CompactGraph,
    directory: EntityDirectory,
    options: Sequence[str],
) -> dict[str, int]:
    """Anchor each treatment option at one treatment-class entity.

    The anchor is shared across patients, so a path's target is a property of the
    option rather than of the record and the same walk can be replayed by an
    auditor. One entity per option, not one per (option, patient): a treatment
    class is a single concept in the graph.
    """
    pool = directory.pool(NodeType.TREATMENT_CLASS)
    ordered = sorted(options)
    if len(ordered) > int(pool.shape[0]):
        raise GraphError("the graph holds fewer treatment classes than there are options")
    return {option: int(pool[index]) for index, option in enumerate(ordered)}


def ground_regimens(
    graph: CompactGraph,
    directory: EntityDirectory,
    options: Sequence[str],
) -> dict[str, int]:
    pool = directory.pool(NodeType.REGIMEN)
    ordered = sorted(options)
    if len(ordered) > int(pool.shape[0]):
        raise GraphError("the graph holds fewer regimens than there are options")
    return {option: int(pool[index]) for index, option in enumerate(ordered)}
