"""Encoder, path scorer and the estimator's combination."""

from __future__ import annotations

import torch

from pathanchor.models.encoder import (
    FEATURE_DIM,
    MODALITY_ORDER,
    EncoderConfig,
    MultimodalEncoder,
    modality_mask,
)
from pathanchor.models.estimator import (
    EstimatorConfig,
    PathAnchorEstimator,
    build_estimator,
    marginalise_path_scores,
    mean_pool_scores,
    parameter_count,
)
from pathanchor.paths.scorer import PathBatch
from pathanchor.training.seed import set_seed


def _empty_batch(width: int, size: int = 1) -> PathBatch:
    long_zeros = torch.zeros((size, width + 1), dtype=torch.long)
    short_zeros = torch.zeros((size, width), dtype=torch.long)
    return PathBatch(
        long_zeros,
        long_zeros,
        short_zeros,
        short_zeros,
        short_zeros.float(),
        short_zeros.float(),
        torch.zeros(size, dtype=torch.long),
    )


def test_feature_width_matches_the_modality_slices() -> None:
    assert FEATURE_DIM == 42
    assert len(MODALITY_ORDER) == 4


def test_modality_mask_expands_to_the_feature_width() -> None:
    present = torch.tensor([[1.0, 1.0, 0.0, 1.0]])
    expanded = modality_mask(present)
    assert tuple(expanded.shape) == (1, FEATURE_DIM)
    assert float(expanded[0, -1]) == 1.0


def test_encoder_shapes() -> None:
    set_seed(20260101)
    encoder = MultimodalEncoder(EncoderConfig(hidden=16, blocks=1))
    features = torch.randn((5, FEATURE_DIM))
    modality = torch.ones((5, 4))
    encoded = encoder(features, modality)
    assert tuple(encoded.shape) == (5, 16)


def test_withheld_modality_changes_the_embedding() -> None:
    set_seed(20260101)
    encoder = MultimodalEncoder(EncoderConfig(hidden=16, blocks=1))
    features = torch.ones((1, FEATURE_DIM))
    full = encoder(features, torch.ones((1, 4)))
    partial = encoder(features, torch.tensor([[1.0, 1.0, 1.0, 0.0]]))
    assert not torch.allclose(full, partial)


def test_estimator_combination_switches() -> None:
    config = EstimatorConfig()
    set_seed(20260101)
    model = PathAnchorEstimator(config)
    base = torch.tensor([1.0])
    marginal = torch.tensor([0.5])
    combined = model.combine(base, marginal)
    assert float(combined.detach()) > float(marginal) - 1e-6
    without_graph = PathAnchorEstimator(EstimatorConfig(use_graph_paths=False, use_base=True))
    assert torch.allclose(without_graph.combine(base, marginal), base)
    without_base = PathAnchorEstimator(EstimatorConfig(use_base=False))
    assert torch.allclose(without_base.combine(base, marginal), marginal)


def test_estimator_scores_an_empty_path_batch() -> None:
    set_seed(20260101)
    model = build_estimator(EstimatorConfig())
    encoded = torch.zeros((1, model.config.encoder.hidden))
    values = model.path_scores(_empty_batch(4), encoded)
    assert tuple(values.shape) == (1,)


def test_marginalisation_floors_the_empty_set() -> None:
    empty = torch.zeros(0)
    assert float(marginalise_path_scores(empty)[0]) == -60.0
    assert float(mean_pool_scores(empty)[0]) == -60.0
    values = torch.tensor([0.5, 0.25])
    assert float(marginalise_path_scores(values)[0]) > float(mean_pool_scores(values)[0])


def test_parameter_count_is_positive_and_small() -> None:
    model = build_estimator(EstimatorConfig())
    count = parameter_count(model)
    assert 0 < count < 500_000


def test_typed_and_untyped_readouts_differ() -> None:
    set_seed(20260101)
    typed = build_estimator(EstimatorConfig(typed_paths=True))
    untyped = build_estimator(EstimatorConfig(typed_paths=False))
    batch = _empty_batch(4)
    encoded = torch.zeros((1, typed.config.encoder.hidden))
    assert not torch.allclose(
        typed.path_scores(batch, encoded), untyped.path_scores(batch, encoded)
    )
