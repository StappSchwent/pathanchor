"""Conformal calibration entry point.

Ref: Sec. III-E and Eq. (2); Table II Panel A (the layer moves no AUROC).
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from pathanchor.agents.counterfactual_critic import CounterfactualCritic
from pathanchor.agents.guideline_adjudicator import GuidelineAdjudicator
from pathanchor.agents.ontology_validator import OntologyValidator
from pathanchor.cohort.fixture import build_cohort_fixture
from pathanchor.cohort.schema import PatientRecord, Split
from pathanchor.conformal.split import SplitConformal
from pathanchor.evaluation.ablations import row_scores
from pathanchor.evaluation.pipeline import build_evidence_store, score_options
from pathanchor.evaluation.registry import RuntimeConfig, resolve
from pathanchor.evaluation.reporting import ReportSection, render_mapping, render_sections
from pathanchor.graph.builder import build_declared_graph
from pathanchor.graph.grounding import EntityDirectory
from pathanchor.graph.index import CompactGraph
from pathanchor.guideline.statements import default_table
from pathanchor.metrics.discrimination import auroc
from pathanchor.models.estimator import PathAnchorEstimator, build_estimator
from pathanchor.training.seed import set_seed
from pathanchor.utils.logging import configure_logging, get_logger
from pathanchor.utils.serialization import write_text
from pathanchor.utils.types import FloatArray, IntArray

LOGGER = get_logger("cli.calibrate")


def parse(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="pathanchor-calibrate",
        description="Fit the split-conformal layer and report its coverage and width.",
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("override", nargs="*", default=[])
    return parser.parse_args(argv)


def scores_for(
    records: Sequence[PatientRecord],
    graph: CompactGraph,
    directory: EntityDirectory,
    runtime: RuntimeConfig,
    model: PathAnchorEstimator,
) -> FloatArray:
    store = build_evidence_store(
        records,
        graph,
        directory,
        runtime.budget,
        GuidelineAdjudicator.from_table(default_table()),
        OntologyValidator(graph=graph),
        CounterfactualCritic(),
    )
    return row_scores(score_options(store, graph, model))


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse(argv)
    configure_logging()
    runtime = resolve(arguments.config, tuple(arguments.override))
    graph = build_declared_graph(runtime.census)
    directory = EntityDirectory.from_graph(graph, core_size=runtime.census.core_size)
    fixture = build_cohort_fixture(runtime.cohort)
    set_seed(runtime.train.seed)
    model = build_estimator(runtime.estimator)
    train = fixture.by_split(Split.TRAIN)[: arguments.limit]
    calibration = fixture.by_split(Split.CALIBRATION)[: arguments.limit]
    test = fixture.by_split(Split.TEST)[: arguments.limit]
    train_scores = scores_for(train, graph, directory, runtime, model)
    calibration_scores = scores_for(calibration, graph, directory, runtime, model)
    test_scores = scores_for(test, graph, directory, runtime, model)
    layer = SplitConformal(alpha=runtime.conformal_alpha, tau_abs=runtime.conformal_tau_abs)
    layer.fit(train_scores, labels_of(train), calibration_scores, labels_of(calibration))
    coverage = layer.coverage(test_scores, labels_of(test))
    width = layer.mean_width(test_scores)
    deferred = layer.abstention_rate(test_scores)
    body = render_sections(
        [
            ReportSection(
                "Layer",
                render_mapping(
                    {
                        "alpha": layer.alpha,
                        "tau_abs": layer.tau_abs,
                        "quantile": layer.quantile,
                        "calibration records": float(layer.n_calibration),
                        "nominal level": layer.nominal_level(),
                    }
                ),
            ),
            ReportSection(
                "Held-out behaviour",
                render_mapping(
                    {
                        "coverage": coverage,
                        "mean width": width,
                        "abstention rate": deferred,
                        "auroc unchanged": auroc(labels_of(test), test_scores),
                    }
                ),
            ),
        ]
    )
    if arguments.report is not None:
        write_text(arguments.report, body)
    LOGGER.info("coverage %.4f at nominal %.2f", coverage, layer.nominal_level())
    print(body)
    return 0


def labels_of(records: Sequence[PatientRecord]) -> IntArray:
    return np.asarray([record.response for record in records], dtype=np.int64)


if __name__ == "__main__":
    raise SystemExit(main())
