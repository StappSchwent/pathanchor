"""Minimal training loop on the smoke configuration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from pathanchor.agents.counterfactual_critic import CounterfactualCritic
from pathanchor.agents.guideline_adjudicator import GuidelineAdjudicator
from pathanchor.agents.ontology_validator import OntologyValidator
from pathanchor.cohort.schema import Split
from pathanchor.evaluation.pipeline import build_evidence_store
from pathanchor.graph.grounding import EntityDirectory
from pathanchor.graph.index import CompactGraph
from pathanchor.guideline.statements import default_table
from pathanchor.losses.composite import LossWeights
from pathanchor.models.estimator import build_estimator, parameter_count
from pathanchor.training.checkpoint import checkpoint_round_trip
from pathanchor.training.engine import TrainConfig, TrainingEngine, fold_rows
from pathanchor.training.seed import set_seed


@pytest.fixture
def smoke_engine(
    smoke_graph: CompactGraph, smoke_directory: EntityDirectory, smoke_runtime, smoke_fixture
):
    records = smoke_fixture.by_split(Split.TEST)
    store = build_evidence_store(
        records,
        smoke_graph,
        smoke_directory,
        smoke_runtime.budget,
        GuidelineAdjudicator.from_table(default_table()),
        OntologyValidator(graph=smoke_graph),
        CounterfactualCritic(),
    )
    set_seed(smoke_runtime.train.seed)
    model = build_estimator(smoke_runtime.estimator)
    config = TrainConfig(
        epochs=6,
        batch_size=8,
        max_steps=30,
        learning_rate=1.0e-2,
        warmup_steps=2,
        seed=smoke_runtime.train.seed,
        path_cap=4,
        epsilon=smoke_runtime.path_epsilon,
    )
    return TrainingEngine(model, store, smoke_graph, config, LossWeights())


def test_store_is_populated(smoke_engine) -> None:
    assert smoke_engine.store.size > 0
    assert smoke_engine.store.ledger.patients == smoke_engine.store.size


def test_training_runs_and_reports_terms(smoke_engine) -> None:
    history = smoke_engine.fit()
    assert history.steps > 0
    summary = history.summary()
    assert summary["steps"] == history.steps
    assert "response" in summary["terms"]


def test_loss_moves_over_the_run(smoke_engine) -> None:
    history = smoke_engine.fit()
    assert len(history.losses) >= 20
    head = float(np.mean(history.losses[:2]))
    tail = float(np.mean(history.losses[-2:]))
    assert tail < head


def test_parameters_move_after_a_step(smoke_engine) -> None:
    engine = smoke_engine
    before = [parameter.detach().clone() for parameter in engine.model.parameters()]
    engine.fit()
    moved = max(
        float((parameter.detach() - original).abs().max())
        for parameter, original in zip(engine.model.parameters(), before, strict=True)
    )
    assert moved > 0.0


def test_checkpoint_round_trip(tmp_path: Path, smoke_engine) -> None:
    report = checkpoint_round_trip(smoke_engine.model, tmp_path, smoke_engine.config.seed)
    assert bool(report["identical"])
    assert bool(report["seed_restored"])


def test_fold_rows_matches_hand_built_rows() -> None:
    flat = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    index = torch.tensor([0, 0, 1, 2, 2])
    rows = fold_rows(flat, index, 3, 3)
    expected = torch.tensor([[1.0, 2.0, -60.0], [3.0, -60.0, -60.0], [4.0, 5.0, -60.0]])
    assert torch.equal(rows, expected)


def test_parameter_count_is_reported(smoke_engine) -> None:
    assert parameter_count(smoke_engine.model) > 0
