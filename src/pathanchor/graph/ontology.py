"""Node types, relation types and their signatures.

Ref: Sec. III-B (Nodes, Edges), Sec. III-C (relation signatures).

The graph carries 41 node types and 42 relation types. Each relation declares
the head and tail types it may connect; that declaration is what makes a path
machine-checkable without re-running the model (Sec. III-C).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum


class NodeType(StrEnum):
    """Entity types. ICD-O-3 and SNOMED CT fix the disease side, HGNC the gene
    side, DrugBank the compound side, KEGG the pathway side, STRING the
    interaction side."""

    DISEASE = "DISEASE"
    MORPHOLOGY = "MORPHOLOGY"
    ANATOMIC_SITE = "ANATOMIC_SITE"
    GENE = "GENE"
    PROTEIN = "PROTEIN"
    PROTEIN_CODING_LOCUS = "PROTEIN_CODING_LOCUS"
    COMPOUND = "COMPOUND"
    DRUG = "DRUG"
    PATHWAY = "PATHWAY"
    BIOLOGICAL_PROCESS = "BIOLOGICAL_PROCESS"
    MOLECULAR_FUNCTION = "MOLECULAR_FUNCTION"
    CELLULAR_COMPONENT = "CELLULAR_COMPONENT"
    INTERACTION_EVIDENCE = "INTERACTION_EVIDENCE"
    VARIANT = "VARIANT"
    SOMATIC_MUTATION = "SOMATIC_MUTATION"
    COPY_NUMBER_ALTERATION = "COPY_NUMBER_ALTERATION"
    EXPRESSION_SIGNATURE = "EXPRESSION_SIGNATURE"
    GENE_EXPRESSION_VALUE = "GENE_EXPRESSION_VALUE"
    CLINICAL_VARIABLE = "CLINICAL_VARIABLE"
    STAGE = "STAGE"
    PERFORMANCE_STATUS = "PERFORMANCE_STATUS"
    HISTOLOGY_DESCRIPTOR = "HISTOLOGY_DESCRIPTOR"
    DERMATOSCOPY_FEATURE = "DERMATOSCOPY_FEATURE"
    WSI_TILE = "WSI_TILE"
    WSI_FEATURE = "WSI_FEATURE"
    TREATMENT = "TREATMENT"
    TREATMENT_CLASS = "TREATMENT_CLASS"
    REGIMEN = "REGIMEN"
    GUIDELINE_RECOMMENDATION = "GUIDELINE_RECOMMENDATION"
    GUIDELINE_STATEMENT = "GUIDELINE_STATEMENT"
    FEASIBILITY_CONSTRAINT = "FEASIBILITY_CONSTRAINT"
    EVIDENCE_TIER = "EVIDENCE_TIER"
    ONTOLOGY_CLASS = "ONTOLOGY_CLASS"
    PHENOTYPE = "PHENOTYPE"
    IMMUNE_CELL_POPULATION = "IMMUNE_CELL_POPULATION"
    CYTOKINE = "CYTOKINE"
    RECEPTOR = "RECEPTOR"
    TRANSPORT_PROTEIN = "TRANSPORT_PROTEIN"
    ENZYME = "ENZYME"
    DRUG_TARGET = "DRUG_TARGET"
    RESPONSE_CRITERION = "RESPONSE_CRITERION"


class RelationType(StrEnum):
    """Edge types. Direction and evidence tier are part of the declaration."""

    SUBCLASS_OF = "SUBCLASS_OF"
    SUBSUMED_BY = "SUBSUMED_BY"
    PART_OF = "PART_OF"
    HAS_PART = "HAS_PART"
    TARGETS = "TARGETS"
    TARGETED_BY = "TARGETED_BY"
    ENCODES = "ENCODES"
    ENCODED_BY = "ENCODED_BY"
    HAS_VARIANT = "HAS_VARIANT"
    VARIANT_OF = "VARIANT_OF"
    PARTICIPATES_IN = "PARTICIPATES_IN"
    HAS_PARTICIPANT = "HAS_PARTICIPANT"
    INTERACTS_WITH = "INTERACTS_WITH"
    INTERACTION_EVIDENCE_FOR = "INTERACTION_EVIDENCE_FOR"
    EXPRESSED_IN = "EXPRESSED_IN"
    EXPRESSES = "EXPRESSES"
    ASSOCIATED_WITH = "ASSOCIATED_WITH"
    RISK_FACTOR_FOR = "RISK_FACTOR_FOR"
    PREDICTS_RESPONSE_TO = "PREDICTS_RESPONSE_TO"
    CONTRAINDICATED_WITH = "CONTRAINDICATED_WITH"
    RECOMMENDED_FOR = "RECOMMENDED_FOR"
    PREFERRED_OVER = "PREFERRED_OVER"
    FEASIBLE_IN_STATE = "FEASIBLE_IN_STATE"
    CONSTRAINS = "CONSTRAINS"
    DERIVED_FROM_STATEMENT = "DERIVED_FROM_STATEMENT"
    SUPPORTS = "SUPPORTS"
    REFUTES = "REFUTES"
    MEASURED_AS = "MEASURED_AS"
    OBSERVED_VALUE = "OBSERVED_VALUE"
    HAS_MAGNITUDE = "HAS_MAGNITUDE"
    HAS_DIRECTION = "HAS_DIRECTION"
    UPREGULATES = "UPREGULATES"
    DOWNREGULATES = "DOWNREGULATES"
    ACTIVATES = "ACTIVATES"
    INHIBITS = "INHIBITS"
    MEDIATES = "MEDIATES"
    MEDIATED_BY = "MEDIATED_BY"
    CO_OCCURS_WITH = "CO_OCCURS_WITH"
    MEMBER_OF = "MEMBER_OF"
    HAS_MEMBER = "HAS_MEMBER"
    CHARACTERIZES = "CHARACTERIZES"
    OBSERVED_IN = "OBSERVED_IN"


class EvidenceTier(StrEnum):
    """Declared strength of the evidence an edge carries."""

    GUIDELINE = "guideline"
    CURATED = "curated"
    EXPERIMENTAL = "experimental"
    INFERRED = "inferred"


class SourceOntology(StrEnum):
    ICD_O_3 = "ICD-O-3"
    SNOMED_CT = "SNOMED CT"
    HGNC = "HGNC"
    DRUG_BANK = "DrugBank"
    KEGG = "KEGG"
    STRING = "STRING"
    HETIONET = "Hetionet"
    GUIDELINE = "NCCN Melanoma: Cutaneous v2.2026"
    COHORT = "cohort record"


NODE_TYPES: tuple[NodeType, ...] = tuple(NodeType)
RELATION_TYPES: tuple[RelationType, ...] = tuple(RelationType)

NODE_TYPE_COUNT: int = len(NODE_TYPES)
RELATION_TYPE_COUNT: int = len(RELATION_TYPES)

_G = NodeType.GENE
_PL = NodeType.PROTEIN_CODING_LOCUS
_V = NodeType.VARIANT
_OC = NodeType.ONTOLOGY_CLASS


@dataclass(frozen=True, slots=True)
class RelationSignature:
    """The head and tail types a relation may legally connect."""

    relation: RelationType
    head: frozenset[NodeType]
    tail: frozenset[NodeType]
    tier: EvidenceTier


def _sig(
    relation: RelationType,
    head: tuple[NodeType, ...],
    tail: tuple[NodeType, ...],
    tier: EvidenceTier,
) -> RelationSignature:
    return RelationSignature(relation, frozenset(head), frozenset(tail), tier)


SIGNATURES: Mapping[RelationType, RelationSignature] = {
    RelationType.SUBCLASS_OF: _sig(
        RelationType.SUBCLASS_OF,
        (_OC, NodeType.DISEASE, NodeType.DRUG, NodeType.PATHWAY),
        (_OC, NodeType.DISEASE, NodeType.DRUG, NodeType.PATHWAY),
        EvidenceTier.CURATED,
    ),
    RelationType.SUBSUMED_BY: _sig(
        RelationType.SUBSUMED_BY,
        (_OC, NodeType.DISEASE, NodeType.MORPHOLOGY),
        (_OC, NodeType.DISEASE, NodeType.MORPHOLOGY),
        EvidenceTier.CURATED,
    ),
    RelationType.PART_OF: _sig(
        RelationType.PART_OF,
        (NodeType.PATHWAY, NodeType.BIOLOGICAL_PROCESS, NodeType.PROTEIN, NodeType.DISEASE),
        (NodeType.PATHWAY, NodeType.BIOLOGICAL_PROCESS, NodeType.PROTEIN, NodeType.DISEASE),
        EvidenceTier.CURATED,
    ),
    RelationType.HAS_PART: _sig(
        RelationType.HAS_PART,
        (NodeType.PATHWAY, NodeType.BIOLOGICAL_PROCESS, NodeType.COMPOUND),
        (NodeType.PROTEIN, NodeType.GENE, NodeType.COMPOUND, NodeType.PATHWAY),
        EvidenceTier.CURATED,
    ),
    RelationType.TARGETS: _sig(
        RelationType.TARGETS,
        (NodeType.DRUG, NodeType.COMPOUND),
        (_G, NodeType.PROTEIN, NodeType.DRUG_TARGET, NodeType.RECEPTOR, NodeType.ENZYME),
        EvidenceTier.CURATED,
    ),
    RelationType.TARGETED_BY: _sig(
        RelationType.TARGETED_BY,
        (_G, NodeType.PROTEIN, NodeType.DRUG_TARGET, NodeType.RECEPTOR, NodeType.ENZYME),
        (NodeType.DRUG, NodeType.COMPOUND),
        EvidenceTier.CURATED,
    ),
    RelationType.ENCODES: _sig(
        RelationType.ENCODES,
        (_G, _PL),
        (
            NodeType.PROTEIN,
            NodeType.ENZYME,
            NodeType.RECEPTOR,
            NodeType.TRANSPORT_PROTEIN,
            NodeType.CYTOKINE,
        ),
        EvidenceTier.CURATED,
    ),
    RelationType.ENCODED_BY: _sig(
        RelationType.ENCODED_BY,
        (
            NodeType.PROTEIN,
            NodeType.ENZYME,
            NodeType.RECEPTOR,
            NodeType.TRANSPORT_PROTEIN,
            NodeType.CYTOKINE,
        ),
        (_G, _PL),
        EvidenceTier.CURATED,
    ),
    RelationType.HAS_VARIANT: _sig(
        RelationType.HAS_VARIANT,
        (_G, _PL, NodeType.DISEASE),
        (_V, NodeType.SOMATIC_MUTATION, NodeType.COPY_NUMBER_ALTERATION),
        EvidenceTier.EXPERIMENTAL,
    ),
    RelationType.VARIANT_OF: _sig(
        RelationType.VARIANT_OF,
        (_V, NodeType.SOMATIC_MUTATION, NodeType.COPY_NUMBER_ALTERATION),
        (_G, _PL),
        EvidenceTier.EXPERIMENTAL,
    ),
    RelationType.PARTICIPATES_IN: _sig(
        RelationType.PARTICIPATES_IN,
        (_G, NodeType.PROTEIN, NodeType.COMPOUND, NodeType.DRUG),
        (NodeType.PATHWAY, NodeType.BIOLOGICAL_PROCESS),
        EvidenceTier.CURATED,
    ),
    RelationType.HAS_PARTICIPANT: _sig(
        RelationType.HAS_PARTICIPANT,
        (NodeType.PATHWAY, NodeType.BIOLOGICAL_PROCESS),
        (_G, NodeType.PROTEIN, NodeType.COMPOUND, NodeType.DRUG),
        EvidenceTier.CURATED,
    ),
    RelationType.INTERACTS_WITH: _sig(
        RelationType.INTERACTS_WITH,
        (NodeType.PROTEIN, _G, NodeType.COMPOUND),
        (NodeType.PROTEIN, _G, NodeType.COMPOUND),
        EvidenceTier.EXPERIMENTAL,
    ),
    RelationType.INTERACTION_EVIDENCE_FOR: _sig(
        RelationType.INTERACTION_EVIDENCE_FOR,
        (NodeType.INTERACTION_EVIDENCE,),
        (NodeType.PROTEIN, _G),
        EvidenceTier.EXPERIMENTAL,
    ),
    RelationType.EXPRESSED_IN: _sig(
        RelationType.EXPRESSED_IN,
        (_G, NodeType.PROTEIN, NodeType.EXPRESSION_SIGNATURE),
        (
            NodeType.DISEASE,
            NodeType.ANATOMIC_SITE,
            NodeType.IMMUNE_CELL_POPULATION,
            NodeType.WSI_TILE,
        ),
        EvidenceTier.EXPERIMENTAL,
    ),
    RelationType.EXPRESSES: _sig(
        RelationType.EXPRESSES,
        (
            NodeType.DISEASE,
            NodeType.ANATOMIC_SITE,
            NodeType.IMMUNE_CELL_POPULATION,
            NodeType.WSI_TILE,
        ),
        (_G, NodeType.PROTEIN, NodeType.EXPRESSION_SIGNATURE),
        EvidenceTier.EXPERIMENTAL,
    ),
    RelationType.ASSOCIATED_WITH: _sig(
        RelationType.ASSOCIATED_WITH,
        (
            NodeType.DISEASE,
            NodeType.PHENOTYPE,
            NodeType.CLINICAL_VARIABLE,
            NodeType.STAGE,
            NodeType.CYTOKINE,
            NodeType.PATHWAY,
        ),
        (
            NodeType.DISEASE,
            NodeType.PHENOTYPE,
            NodeType.CLINICAL_VARIABLE,
            NodeType.RESPONSE_CRITERION,
            NodeType.TREATMENT_CLASS,
        ),
        EvidenceTier.CURATED,
    ),
    RelationType.RISK_FACTOR_FOR: _sig(
        RelationType.RISK_FACTOR_FOR,
        (NodeType.CLINICAL_VARIABLE, NodeType.VARIANT, NodeType.PHENOTYPE),
        (NodeType.DISEASE, NodeType.PHENOTYPE),
        EvidenceTier.CURATED,
    ),
    RelationType.PREDICTS_RESPONSE_TO: _sig(
        RelationType.PREDICTS_RESPONSE_TO,
        (
            _G,
            NodeType.PROTEIN,
            NodeType.EXPRESSION_SIGNATURE,
            NodeType.PATHWAY,
            NodeType.CLINICAL_VARIABLE,
            NodeType.VARIANT,
        ),
        (
            NodeType.TREATMENT,
            NodeType.TREATMENT_CLASS,
            NodeType.REGIMEN,
            NodeType.RESPONSE_CRITERION,
        ),
        EvidenceTier.EXPERIMENTAL,
    ),
    RelationType.CONTRAINDICATED_WITH: _sig(
        RelationType.CONTRAINDICATED_WITH,
        (NodeType.CLINICAL_VARIABLE, NodeType.PHENOTYPE, NodeType.TREATMENT),
        (NodeType.TREATMENT, NodeType.TREATMENT_CLASS, NodeType.CLINICAL_VARIABLE),
        EvidenceTier.GUIDELINE,
    ),
    RelationType.RECOMMENDED_FOR: _sig(
        RelationType.RECOMMENDED_FOR,
        (NodeType.TREATMENT, NodeType.TREATMENT_CLASS, NodeType.REGIMEN),
        (NodeType.STAGE, NodeType.PHENOTYPE, NodeType.GUIDELINE_RECOMMENDATION),
        EvidenceTier.GUIDELINE,
    ),
    RelationType.PREFERRED_OVER: _sig(
        RelationType.PREFERRED_OVER,
        (NodeType.TREATMENT, NodeType.TREATMENT_CLASS, NodeType.REGIMEN),
        (NodeType.TREATMENT, NodeType.TREATMENT_CLASS, NodeType.REGIMEN),
        EvidenceTier.GUIDELINE,
    ),
    RelationType.FEASIBLE_IN_STATE: _sig(
        RelationType.FEASIBLE_IN_STATE,
        (NodeType.TREATMENT, NodeType.TREATMENT_CLASS, NodeType.REGIMEN),
        (
            NodeType.STAGE,
            NodeType.PHENOTYPE,
            NodeType.CLINICAL_VARIABLE,
            NodeType.FEASIBILITY_CONSTRAINT,
        ),
        EvidenceTier.GUIDELINE,
    ),
    RelationType.CONSTRAINS: _sig(
        RelationType.CONSTRAINS,
        (NodeType.GUIDELINE_STATEMENT, NodeType.GUIDELINE_RECOMMENDATION),
        (NodeType.FEASIBILITY_CONSTRAINT, NodeType.TREATMENT_CLASS),
        EvidenceTier.GUIDELINE,
    ),
    RelationType.DERIVED_FROM_STATEMENT: _sig(
        RelationType.DERIVED_FROM_STATEMENT,
        (NodeType.FEASIBILITY_CONSTRAINT, NodeType.GUIDELINE_RECOMMENDATION),
        (NodeType.GUIDELINE_STATEMENT,),
        EvidenceTier.GUIDELINE,
    ),
    RelationType.SUPPORTS: _sig(
        RelationType.SUPPORTS,
        (
            NodeType.INTERACTION_EVIDENCE,
            NodeType.PATHWAY,
            NodeType.EXPRESSION_SIGNATURE,
            NodeType.GENE_EXPRESSION_VALUE,
        ),
        (
            NodeType.RESPONSE_CRITERION,
            NodeType.GUIDELINE_RECOMMENDATION,
            NodeType.PHENOTYPE,
            NodeType.EXPRESSION_SIGNATURE,
            NodeType.PATHWAY,
        ),
        EvidenceTier.EXPERIMENTAL,
    ),
    RelationType.REFUTES: _sig(
        RelationType.REFUTES,
        (
            NodeType.INTERACTION_EVIDENCE,
            NodeType.EXPRESSION_SIGNATURE,
            NodeType.GENE_EXPRESSION_VALUE,
        ),
        (NodeType.PHENOTYPE, NodeType.RESPONSE_CRITERION, NodeType.EXPRESSION_SIGNATURE),
        EvidenceTier.EXPERIMENTAL,
    ),
    RelationType.MEASURED_AS: _sig(
        RelationType.MEASURED_AS,
        (NodeType.CLINICAL_VARIABLE, NodeType.PHENOTYPE, NodeType.GENE),
        (
            NodeType.GENE_EXPRESSION_VALUE,
            NodeType.HISTOLOGY_DESCRIPTOR,
            NodeType.CLINICAL_VARIABLE,
            NodeType.WSI_FEATURE,
        ),
        EvidenceTier.CURATED,
    ),
    RelationType.OBSERVED_VALUE: _sig(
        RelationType.OBSERVED_VALUE,
        (NodeType.DISEASE, NodeType.PHENOTYPE, NodeType.IMMUNE_CELL_POPULATION),
        (
            NodeType.CLINICAL_VARIABLE,
            NodeType.GENE_EXPRESSION_VALUE,
            NodeType.HISTOLOGY_DESCRIPTOR,
            NodeType.DERMATOSCOPY_FEATURE,
            NodeType.WSI_FEATURE,
        ),
        EvidenceTier.EXPERIMENTAL,
    ),
    RelationType.HAS_MAGNITUDE: _sig(
        RelationType.HAS_MAGNITUDE,
        (
            NodeType.GENE_EXPRESSION_VALUE,
            NodeType.CLINICAL_VARIABLE,
            NodeType.WSI_FEATURE,
            NodeType.HISTOLOGY_DESCRIPTOR,
            NodeType.DERMATOSCOPY_FEATURE,
        ),
        (NodeType.ONTOLOGY_CLASS,),
        EvidenceTier.EXPERIMENTAL,
    ),
    RelationType.HAS_DIRECTION: _sig(
        RelationType.HAS_DIRECTION,
        (NodeType.GENE_EXPRESSION_VALUE, NodeType.CLINICAL_VARIABLE),
        (NodeType.ONTOLOGY_CLASS,),
        EvidenceTier.EXPERIMENTAL,
    ),
    RelationType.UPREGULATES: _sig(
        RelationType.UPREGULATES,
        (_G, NodeType.PROTEIN, NodeType.EXPRESSION_SIGNATURE),
        (_G, NodeType.PROTEIN, NodeType.PATHWAY, NodeType.CYTOKINE),
        EvidenceTier.EXPERIMENTAL,
    ),
    RelationType.DOWNREGULATES: _sig(
        RelationType.DOWNREGULATES,
        (_G, NodeType.PROTEIN, NodeType.EXPRESSION_SIGNATURE),
        (_G, NodeType.PROTEIN, NodeType.PATHWAY, NodeType.CYTOKINE),
        EvidenceTier.EXPERIMENTAL,
    ),
    RelationType.ACTIVATES: _sig(
        RelationType.ACTIVATES,
        (NodeType.PROTEIN, _G, NodeType.CYTOKINE, NodeType.DRUG, NodeType.PATHWAY),
        (NodeType.PROTEIN, NodeType.PATHWAY, NodeType.RECEPTOR, NodeType.BIOLOGICAL_PROCESS),
        EvidenceTier.EXPERIMENTAL,
    ),
    RelationType.INHIBITS: _sig(
        RelationType.INHIBITS,
        (NodeType.DRUG, NodeType.COMPOUND, NodeType.PROTEIN, NodeType.CYTOKINE),
        (
            NodeType.PROTEIN,
            NodeType.PATHWAY,
            NodeType.ENZYME,
            NodeType.RECEPTOR,
            NodeType.BIOLOGICAL_PROCESS,
        ),
        EvidenceTier.CURATED,
    ),
    RelationType.MEDIATES: _sig(
        RelationType.MEDIATES,
        (NodeType.PATHWAY, NodeType.CYTOKINE, NodeType.PROTEIN),
        (
            NodeType.PHENOTYPE,
            NodeType.BIOLOGICAL_PROCESS,
            NodeType.DISEASE,
            NodeType.RESPONSE_CRITERION,
        ),
        EvidenceTier.EXPERIMENTAL,
    ),
    RelationType.MEDIATED_BY: _sig(
        RelationType.MEDIATED_BY,
        (
            NodeType.PHENOTYPE,
            NodeType.BIOLOGICAL_PROCESS,
            NodeType.DISEASE,
            NodeType.RESPONSE_CRITERION,
        ),
        (NodeType.PATHWAY, NodeType.CYTOKINE, NodeType.PROTEIN),
        EvidenceTier.EXPERIMENTAL,
    ),
    RelationType.CO_OCCURS_WITH: _sig(
        RelationType.CO_OCCURS_WITH,
        (NodeType.VARIANT, _G, NodeType.PHENOTYPE, NodeType.CLINICAL_VARIABLE),
        (NodeType.VARIANT, _G, NodeType.PHENOTYPE, NodeType.CLINICAL_VARIABLE),
        EvidenceTier.EXPERIMENTAL,
    ),
    RelationType.MEMBER_OF: _sig(
        RelationType.MEMBER_OF,
        (NodeType.PROTEIN, NodeType.IMMUNE_CELL_POPULATION, NodeType.PATHWAY),
        (NodeType.PATHWAY, NodeType.IMMUNE_CELL_POPULATION, NodeType.ONTOLOGY_CLASS),
        EvidenceTier.CURATED,
    ),
    RelationType.HAS_MEMBER: _sig(
        RelationType.HAS_MEMBER,
        (NodeType.PATHWAY, NodeType.IMMUNE_CELL_POPULATION, NodeType.ONTOLOGY_CLASS),
        (NodeType.PROTEIN, NodeType.IMMUNE_CELL_POPULATION, NodeType.PATHWAY),
        EvidenceTier.CURATED,
    ),
    RelationType.CHARACTERIZES: _sig(
        RelationType.CHARACTERIZES,
        (
            NodeType.HISTOLOGY_DESCRIPTOR,
            NodeType.DERMATOSCOPY_FEATURE,
            NodeType.WSI_FEATURE,
            NodeType.MORPHOLOGY,
        ),
        (NodeType.DISEASE, NodeType.PHENOTYPE, NodeType.MORPHOLOGY),
        EvidenceTier.CURATED,
    ),
    RelationType.OBSERVED_IN: _sig(
        RelationType.OBSERVED_IN,
        (
            NodeType.VARIANT,
            NodeType.SOMATIC_MUTATION,
            NodeType.EXPRESSION_SIGNATURE,
            NodeType.HISTOLOGY_DESCRIPTOR,
        ),
        (NodeType.DISEASE, NodeType.ANATOMIC_SITE, NodeType.PHENOTYPE),
        EvidenceTier.EXPERIMENTAL,
    ),
}

NODE_TYPE_SOURCE: Mapping[NodeType, SourceOntology] = {
    NodeType.DISEASE: SourceOntology.ICD_O_3,
    NodeType.MORPHOLOGY: SourceOntology.ICD_O_3,
    NodeType.ANATOMIC_SITE: SourceOntology.ICD_O_3,
    NodeType.GENE: SourceOntology.HGNC,
    NodeType.PROTEIN: SourceOntology.HGNC,
    NodeType.PROTEIN_CODING_LOCUS: SourceOntology.HGNC,
    NodeType.COMPOUND: SourceOntology.DRUG_BANK,
    NodeType.DRUG: SourceOntology.DRUG_BANK,
    NodeType.PATHWAY: SourceOntology.KEGG,
    NodeType.BIOLOGICAL_PROCESS: SourceOntology.KEGG,
    NodeType.MOLECULAR_FUNCTION: SourceOntology.KEGG,
    NodeType.CELLULAR_COMPONENT: SourceOntology.KEGG,
    NodeType.INTERACTION_EVIDENCE: SourceOntology.STRING,
    NodeType.VARIANT: SourceOntology.SNOMED_CT,
    NodeType.SOMATIC_MUTATION: SourceOntology.SNOMED_CT,
    NodeType.COPY_NUMBER_ALTERATION: SourceOntology.SNOMED_CT,
    NodeType.EXPRESSION_SIGNATURE: SourceOntology.HETIONET,
    NodeType.GENE_EXPRESSION_VALUE: SourceOntology.COHORT,
    NodeType.CLINICAL_VARIABLE: SourceOntology.SNOMED_CT,
    NodeType.STAGE: SourceOntology.SNOMED_CT,
    NodeType.PERFORMANCE_STATUS: SourceOntology.SNOMED_CT,
    NodeType.HISTOLOGY_DESCRIPTOR: SourceOntology.SNOMED_CT,
    NodeType.DERMATOSCOPY_FEATURE: SourceOntology.SNOMED_CT,
    NodeType.WSI_TILE: SourceOntology.COHORT,
    NodeType.WSI_FEATURE: SourceOntology.COHORT,
    NodeType.TREATMENT: SourceOntology.DRUG_BANK,
    NodeType.TREATMENT_CLASS: SourceOntology.DRUG_BANK,
    NodeType.REGIMEN: SourceOntology.DRUG_BANK,
    NodeType.GUIDELINE_RECOMMENDATION: SourceOntology.GUIDELINE,
    NodeType.GUIDELINE_STATEMENT: SourceOntology.GUIDELINE,
    NodeType.FEASIBILITY_CONSTRAINT: SourceOntology.GUIDELINE,
    NodeType.EVIDENCE_TIER: SourceOntology.HETIONET,
    NodeType.ONTOLOGY_CLASS: SourceOntology.SNOMED_CT,
    NodeType.PHENOTYPE: SourceOntology.SNOMED_CT,
    NodeType.IMMUNE_CELL_POPULATION: SourceOntology.HETIONET,
    NodeType.CYTOKINE: SourceOntology.HGNC,
    NodeType.RECEPTOR: SourceOntology.HGNC,
    NodeType.TRANSPORT_PROTEIN: SourceOntology.HGNC,
    NodeType.ENZYME: SourceOntology.HGNC,
    NodeType.DRUG_TARGET: SourceOntology.DRUG_BANK,
    NodeType.RESPONSE_CRITERION: SourceOntology.SNOMED_CT,
}


def is_admissible(relation: RelationType, head: NodeType, tail: NodeType) -> bool:
    """True when the relation may legally connect a head of one type to a tail
    of the other. This is the type-consistency test of Sec. III-C."""
    signature = SIGNATURES.get(relation)
    if signature is None:
        return False
    return head in signature.head and tail in signature.tail


def tier_of(relation: RelationType) -> EvidenceTier:
    return SIGNATURES[relation].tier


def validate_ontology() -> None:
    """Fail loudly if the declared type counts or signature coverage drift."""
    if NODE_TYPE_COUNT != 41:
        raise ValueError(f"declared node type count is {NODE_TYPE_COUNT}, expected 41")
    if RELATION_TYPE_COUNT != 42:
        raise ValueError(f"declared relation type count is {RELATION_TYPE_COUNT}, expected 42")
    missing = [item.relation.value for item in SIGNATURES.values()]
    absent = [relation.value for relation in RELATION_TYPES if relation.value not in missing]
    if absent:
        raise ValueError(f"relation types without a signature: {absent}")
    for signature in SIGNATURES.values():
        if signature.head and signature.tail:
            continue
        raise ValueError(f"relation {signature.relation.value} has an empty signature side")
    for node_type in NODE_TYPES:
        if node_type not in NODE_TYPE_SOURCE:
            raise ValueError(f"node type {node_type.value} has no declared source ontology")


validate_ontology()
