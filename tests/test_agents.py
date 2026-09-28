"""The five agents and their typed interfaces."""

from __future__ import annotations

import numpy as np
import pytest

from pathanchor.agents.conformal_calibrator import ConformalCalibrator, coverage_report
from pathanchor.agents.contracts import (
    AdjudicatorProtocol,
    CalibratorProtocol,
    CriticProtocol,
    RetrievalRequest,
    RetrieverProtocol,
    ValidatorProtocol,
)
from pathanchor.agents.counterfactual_critic import CounterfactualCritic, build_queries
from pathanchor.agents.guideline_adjudicator import GuidelineAdjudicator
from pathanchor.agents.ontology_validator import OntologyValidator
from pathanchor.agents.retriever import Retriever
from pathanchor.conformal.split import SplitConformal
from pathanchor.graph.grounding import EntityDirectory, ground_record
from pathanchor.graph.index import CompactGraph
from pathanchor.graph.ontology import NodeType
from pathanchor.guideline.statements import default_table
from pathanchor.paths.enumerate import ProposalBudget


def _targets(directory: EntityDirectory) -> dict[str, int]:
    pool = directory.pool(NodeType.TREATMENT_CLASS)
    return {f"option-{index}": int(pool[index]) for index in range(3)}


def test_protocols_are_satisfied(
    smoke_graph: CompactGraph, smoke_directory: EntityDirectory, smoke_runtime
) -> None:
    retriever = Retriever(
        graph=smoke_graph, targets=_targets(smoke_directory), budget=smoke_runtime.budget
    )
    validator = OntologyValidator(graph=smoke_graph)
    adjudicator = GuidelineAdjudicator.from_table(default_table())
    critic = CounterfactualCritic()
    scores = np.linspace(0.0, 1.0, 40)
    labels = (scores > 0.5).astype(np.int64)
    layer = SplitConformal(alpha=0.10, tau_abs=0.35).fit(scores, labels, scores, labels)
    calibrator = ConformalCalibrator(layer=layer)
    assert isinstance(retriever, RetrieverProtocol)
    assert isinstance(validator, ValidatorProtocol)
    assert isinstance(adjudicator, AdjudicatorProtocol)
    assert isinstance(critic, CriticProtocol)
    assert isinstance(calibrator, CalibratorProtocol)


def test_retriever_proposes_for_every_option(
    smoke_graph: CompactGraph, smoke_directory: EntityDirectory, records_factory
) -> None:
    budget = ProposalBudget(max_hops=3, max_per_pair=16, branching=8, min_hops=1, max_frontier=32)
    retriever = Retriever(graph=smoke_graph, targets=_targets(smoke_directory), budget=budget)
    record = records_factory()
    subgraph = ground_record(smoke_graph, smoke_directory, record)
    lookup = {key: index for index, key in enumerate(smoke_graph.keys())}
    anchors = tuple(lookup[key] for key in subgraph)
    result = retriever.run(
        RetrievalRequest(
            patient_id=record.patient_id,
            anchors=anchors,
            options=tuple(_targets(smoke_directory).keys()),
        )
    )
    assert set(result.proposals) == set(_targets(smoke_directory))
    assert result.anchor_count == len(anchors)


def test_validator_rejects_an_ill_typed_route(
    smoke_graph: CompactGraph, smoke_directory: EntityDirectory, records_factory
) -> None:
    from pathanchor.agents.contracts import RetrievalResult
    from pathanchor.graph.ontology import EvidenceTier, RelationType
    from pathanchor.paths.enumerate import ProposalStats
    from pathanchor.paths.signatures import EvidencePath, PathEdge

    ill_typed = EvidencePath(
        (PathEdge("HGNC:0000000", RelationType.TARGETS, "TREAT:0000000", EvidenceTier.CURATED),)
    )
    result = RetrievalResult(
        patient_id="probe",
        proposals={"option-0": (ill_typed,)},
        anchor_count=1,
        stats=ProposalStats(0, 0, False, 1),
    )
    validator = OntologyValidator(graph=smoke_graph)
    outcome = validator.run(result)
    assert outcome.rejected == 1
    assert outcome.validated["option-0"] == ()
    del records_factory, smoke_directory


def test_validator_keeps_a_typed_route(
    smoke_graph: CompactGraph, smoke_directory: EntityDirectory
) -> None:
    from pathanchor.agents.contracts import RetrievalResult
    from pathanchor.agents.retriever import Retriever as _Retriever
    from pathanchor.paths.enumerate import PathEnumerator, ProposalStats

    budget = ProposalBudget(max_hops=3, max_per_pair=16, branching=8, min_hops=1, max_frontier=32)
    targets = _targets(smoke_directory)
    enumerator = PathEnumerator(smoke_graph)
    anchor = int(smoke_directory.pool(NodeType.CLINICAL_VARIABLE)[0])
    paths = enumerator.propose([anchor], (targets["option-0"],), budget)
    if not paths:
        pytest.skip("no route proposed for the probe pair")
    result = RetrievalResult(
        patient_id="probe",
        proposals={"option-0": tuple(paths)},
        anchor_count=1,
        stats=ProposalStats(0, 0, False, len(paths)),
    )
    outcome = OntologyValidator(graph=smoke_graph).run(result)
    assert len(outcome.validated["option-0"]) == len(paths)
    assert outcome.coverage > 0.0
    del _Retriever


def test_critic_reweights_a_shared_signature(
    smoke_graph: CompactGraph, smoke_directory: EntityDirectory
) -> None:
    budget = ProposalBudget(max_hops=2, max_per_pair=16, branching=8, min_hops=1, max_frontier=32)
    from pathanchor.paths.enumerate import PathEnumerator

    enumerator = PathEnumerator(smoke_graph)
    anchor = int(smoke_directory.pool(NodeType.CLINICAL_VARIABLE)[0])
    targets = _targets(smoke_directory)
    proposals = {
        name: tuple(enumerator.propose([anchor], (target,), budget))
        for name, target in targets.items()
    }
    queries = build_queries(proposals)
    shared = [key for key, query in queries.items() if len(query.reaches) > 1]
    critic = CounterfactualCritic()
    outcome = critic.run("probe", (anchor,), tuple(proposals), proposals)
    if not shared:
        pytest.skip("no signature reaches two options for the probe pair")
    assert outcome.downweighted > 0
    assert all(factor <= 1.0 for items in outcome.weighted.values() for _, factor in items)


def test_calibrator_defers_on_a_wide_set(smoke_runtime) -> None:
    rng = np.random.default_rng(4)
    labels = (rng.random(80) < 0.5).astype(np.int64)
    scores = labels + rng.normal(0.0, 2.0, 80)
    layer = SplitConformal(alpha=0.20, tau_abs=0.01).fit(scores, labels, scores, labels)
    calibrator = ConformalCalibrator(layer=layer)
    result = calibrator.run("p1", "option-0", 0.5)
    assert result.abstains
    report = coverage_report(layer, scores, labels)
    assert set(report) == {"coverage", "nominal", "mean_width", "quantile", "n_calibration"}
    del smoke_runtime
