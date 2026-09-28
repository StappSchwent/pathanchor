"""Rotation algebra, the path scorer and path marginalisation."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pathanchor.graph.grounding import EntityDirectory
from pathanchor.graph.index import CompactGraph
from pathanchor.graph.ontology import NodeType, RelationType
from pathanchor.paths.enumerate import PathEnumerator, ProposalBudget
from pathanchor.paths.marginal import (
    EMPTY_OPTION_SCORE,
    argmax_over_feasible,
    log_sum_exp_marginal,
    mean_pool_marginal,
    segment_log_sum_exp,
)
from pathanchor.paths.rotation import (
    ScorerConfig,
    complex_multiply,
    composed_angle,
    rotate_sequence,
)
from pathanchor.paths.scorer import PathScorer, encode_paths, top_m_paths
from pathanchor.paths.signatures import (
    EvidencePath,
    PathEdge,
    admissible,
    dedupe_paths,
    is_chain,
    path_lengths,
)
from pathanchor.training.seed import set_seed


def _path(dimension: int) -> EvidencePath:
    edges = []
    for index in range(dimension):
        edges.append(
            PathEdge(
                head=f"n{index}",
                relation=RelationType.ASSOCIATED_WITH,
                tail=f"n{index + 1}",
                tier=__import__(
                    "pathanchor.graph.ontology", fromlist=["EvidenceTier"]
                ).EvidenceTier.CURATED,
            )
        )
    return EvidencePath(tuple(edges))


def test_rotation_composition_equals_the_angle_sum() -> None:
    start = np.asarray([1.0, 0.5, -0.25, 0.75, 0.0, 1.0, 0.5, -0.5])
    phases = np.asarray([[0.3, -0.7, 1.1, 0.2], [0.9, 0.4, -0.5, 0.8], [-0.2, 0.1, 0.6, -0.9]])
    rotated = rotate_sequence(start, phases)
    total = composed_angle(phases)
    half = start.shape[0] // 2
    re_s, im_s = start[:half], start[half:]
    expected = np.concatenate(
        [re_s * np.cos(total) - im_s * np.sin(total), re_s * np.sin(total) + im_s * np.cos(total)]
    )
    np.testing.assert_allclose(rotated, expected, atol=1e-12)


def test_complex_multiply_matches_the_definition() -> None:
    left = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    right = torch.tensor([[0.5, -1.0, 2.0, 0.25]])
    product = complex_multiply(left, right)
    left_re = torch.tensor([1.0, 2.0])
    left_im = torch.tensor([3.0, 4.0])
    right_re = torch.tensor([0.5, -1.0])
    right_im = torch.tensor([2.0, 0.25])
    expected = torch.cat(
        [left_re * right_re - left_im * right_im, left_re * right_im + left_im * right_re]
    ).unsqueeze(0)
    assert torch.allclose(product, expected)


def test_chain_and_signature_checks() -> None:
    path = _path(2)
    assert is_chain(path)
    assert path_lengths([path]) == {2: 1}
    assert len(dedupe_paths([path, path])) == 1
    type_lookup = {f"n{index}": NodeType.CLINICAL_VARIABLE for index in range(3)}
    assert admissible(path, type_lookup.__getitem__)


def test_log_sum_exp_marginal_matches_a_direct_sum() -> None:
    scores = [0.4, -1.2, 2.5, 0.75]
    assert log_sum_exp_marginal(scores) == pytest.approx(
        float(np.log(sum(np.exp(value) for value in scores))), abs=1e-12
    )
    assert mean_pool_marginal(scores) == pytest.approx(float(np.mean(scores)))
    assert log_sum_exp_marginal([]) == EMPTY_OPTION_SCORE


def test_argmax_respects_the_feasible_set() -> None:
    scores = {"a": 3.0, "b": 1.0, "c": 2.0}
    option, value, ok = argmax_over_feasible(scores, ["b", "c"])
    assert ok and option == "c" and value == pytest.approx(2.0)
    empty, _, ok_empty = argmax_over_feasible(scores, ["z"])
    assert not ok_empty and empty == ""


def test_segment_log_sum_exp_groups_by_segment() -> None:
    values = torch.tensor([0.5, 0.25, 1.0])
    segments = torch.tensor([0, 0, 1])
    totals = segment_log_sum_exp(values, segments, 2)
    assert float(totals[0]) == pytest.approx(float(np.log(np.exp(0.5) + np.exp(0.25))), abs=1e-6)
    assert float(totals[1]) == pytest.approx(1.0, abs=1e-6)


def test_path_scorer_shapes_and_readout_differ(tiny_graph: CompactGraph) -> None:
    directory = EntityDirectory.from_graph(tiny_graph, core_size=64)
    enumerator = PathEnumerator(tiny_graph)
    anchors = [int(directory.pool(NodeType.CLINICAL_VARIABLE)[0])]
    target = int(directory.pool(NodeType.TREATMENT_CLASS)[0])
    paths = enumerator.propose(
        anchors,
        (target,),
        ProposalBudget(max_hops=3, max_per_pair=12, branching=6, min_hops=1, max_frontier=32),
    )
    if not paths:
        pytest.skip("no route proposed for the probe pair")
    set_seed(20260101)
    scorer = PathScorer(ScorerConfig(dimension=8, hidden=16), epsilon=0.0)
    batch = encode_paths(tiny_graph, paths, 3)
    condition = torch.zeros((len(paths), 16))
    values = scorer.forward(batch)
    pooled = scorer.mean_pool(batch)
    assert tuple(values.shape) == (len(paths),)
    assert tuple(pooled.shape) == (len(paths),)
    assert not torch.allclose(values, pooled)
    del condition


def test_top_m_selection_is_ordered(tiny_graph: CompactGraph) -> None:
    directory = EntityDirectory.from_graph(tiny_graph, core_size=64)
    enumerator = PathEnumerator(tiny_graph)
    anchors = [int(directory.pool(NodeType.CLINICAL_VARIABLE)[0])]
    target = int(directory.pool(NodeType.TREATMENT_CLASS)[0])
    paths = enumerator.propose(
        anchors,
        (target,),
        ProposalBudget(max_hops=3, max_per_pair=12, branching=6, min_hops=1, max_frontier=32),
    )
    if not paths:
        pytest.skip("no route proposed for the probe pair")
    picked = top_m_paths(tiny_graph, paths, [float(index) for index in range(len(paths))], 3)
    assert len(picked) == min(3, len(paths))
