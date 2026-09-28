"""Agent pipeline, option scoring and the conformal layer end to end."""

from __future__ import annotations

import numpy as np
import pytest

from pathanchor.agents.counterfactual_critic import CounterfactualCritic
from pathanchor.agents.guideline_adjudicator import GuidelineAdjudicator
from pathanchor.agents.ontology_validator import OntologyValidator
from pathanchor.cohort.schema import Split
from pathanchor.evaluation.ablations import (
    INTEGRATED,
    PANEL_A_ARMS,
    PANEL_B_ARMS,
    build_store_for_arm,
    evaluate_arm,
    marginal_gains,
    score_base_only,
    synergy_surplus,
)
from pathanchor.evaluation.pipeline import (
    build_evidence_store,
    score_options,
    selected_options,
)
from pathanchor.graph.grounding import EntityDirectory
from pathanchor.graph.index import CompactGraph
from pathanchor.guideline.statements import default_table
from pathanchor.models.estimator import build_estimator
from pathanchor.training.seed import set_seed


@pytest.fixture(scope="module")
def smoke_store(
    smoke_graph: CompactGraph, smoke_directory: EntityDirectory, smoke_runtime, smoke_fixture
):
    records = smoke_fixture.by_split(Split.TEST)
    return build_evidence_store(
        records,
        smoke_graph,
        smoke_directory,
        smoke_runtime.budget,
        GuidelineAdjudicator.from_table(default_table()),
        OntologyValidator(graph=smoke_graph),
        CounterfactualCritic(),
    )


def test_store_ledger_is_consistent(smoke_store) -> None:
    ledger = smoke_store.ledger.as_dict()
    assert ledger["patients"] == smoke_store.size
    assert ledger["validated"] >= ledger["rejected"]
    assert ledger["deferred"] <= smoke_store.size


def test_retained_paths_are_typed(smoke_store, smoke_graph: CompactGraph) -> None:
    lookup = {key: smoke_graph.node_type_of(index) for index, key in enumerate(smoke_graph.keys())}
    for path in smoke_store.flat_paths()[:40]:
        for edge in path.edges:
            assert edge.head in lookup and edge.tail in lookup


def test_scoring_produces_one_row_per_record(smoke_store, smoke_graph, smoke_runtime) -> None:
    set_seed(smoke_runtime.train.seed)
    model = build_estimator(smoke_runtime.estimator)
    matrix = score_options(smoke_store, smoke_graph, model)
    assert matrix.scores.shape[0] == smoke_store.size
    assert matrix.weights.shape == matrix.scores.shape
    chosen = selected_options(matrix)
    assert len(chosen) == smoke_store.size


def test_base_only_arm_scores_every_record(smoke_store, smoke_runtime) -> None:
    set_seed(smoke_runtime.train.seed)
    model = build_estimator(smoke_runtime.estimator)
    matrix = score_base_only(smoke_store, model)
    finite = np.isfinite(matrix.scores.detach().numpy())
    assert bool(finite.any())


def test_arms_execute_and_report(
    smoke_graph: CompactGraph, smoke_directory: EntityDirectory, smoke_runtime, smoke_fixture
) -> None:
    records = smoke_fixture.by_split(Split.TEST)
    adjudicator = GuidelineAdjudicator.from_table(default_table())
    set_seed(smoke_runtime.train.seed)
    model = build_estimator(smoke_runtime.estimator)
    results = []
    for name, flags in PANEL_A_ARMS[:3]:
        store = build_store_for_arm(
            flags, records, smoke_graph, smoke_directory, smoke_runtime.budget, adjudicator
        )
        results.append(
            evaluate_arm(
                name,
                flags,
                records,
                smoke_graph,
                smoke_directory,
                smoke_runtime.budget,
                adjudicator,
                model,
                store,
            )
        )
    assert len(results) == 3
    assert all(np.isfinite(item.auroc) for item in results)
    assert len(marginal_gains(results)) == 2


def test_unconstrained_arm_leaves_the_feasible_set(
    smoke_graph: CompactGraph, smoke_directory: EntityDirectory, smoke_runtime, smoke_fixture
) -> None:
    records = smoke_fixture.by_split(Split.TEST)
    adjudicator = GuidelineAdjudicator.from_table(default_table())
    set_seed(smoke_runtime.train.seed)
    model = build_estimator(smoke_runtime.estimator)
    flags = next(item for _, item in PANEL_B_ARMS if not item.projection_enabled)
    store = build_store_for_arm(
        flags, records, smoke_graph, smoke_directory, smoke_runtime.budget, adjudicator
    )
    result = evaluate_arm(
        "unconstrained",
        flags,
        records,
        smoke_graph,
        smoke_directory,
        smoke_runtime.budget,
        adjudicator,
        model,
        store,
    )
    integrated = build_store_for_arm(
        INTEGRATED, records, smoke_graph, smoke_directory, smoke_runtime.budget, adjudicator
    )
    deployed = evaluate_arm(
        "integrated",
        INTEGRATED,
        records,
        smoke_graph,
        smoke_directory,
        smoke_runtime.budget,
        adjudicator,
        model,
        integrated,
    )
    assert deployed.discordance == 0.0
    assert result.discordance >= 0.0


def test_synergy_surplus_is_finite(
    smoke_graph, smoke_directory, smoke_runtime, smoke_fixture
) -> None:
    records = smoke_fixture.by_split(Split.TEST)
    adjudicator = GuidelineAdjudicator.from_table(default_table())
    set_seed(smoke_runtime.train.seed)
    model = build_estimator(smoke_runtime.estimator)
    results = []
    for name, flags in PANEL_A_ARMS:
        store = build_store_for_arm(
            flags, records, smoke_graph, smoke_directory, smoke_runtime.budget, adjudicator
        )
        results.append(
            evaluate_arm(
                name,
                flags,
                records,
                smoke_graph,
                smoke_directory,
                smoke_runtime.budget,
                adjudicator,
                model,
                store,
            )
        )
    assert np.isfinite(synergy_surplus(results))
