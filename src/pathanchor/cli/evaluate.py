"""Evaluation entry point.

Ref: Sec. IV-A to IV-E (the reported tables), Sec. III-H (the metric set).
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from pathanchor.agents.conformal_calibrator import ConformalCalibrator, coverage_report
from pathanchor.agents.counterfactual_critic import CounterfactualCritic
from pathanchor.agents.guideline_adjudicator import GuidelineAdjudicator
from pathanchor.agents.ontology_validator import OntologyValidator
from pathanchor.cohort.fixture import build_cohort_fixture
from pathanchor.cohort.schema import PatientRecord, Split
from pathanchor.conformal.split import SplitConformal
from pathanchor.evaluation.ablations import row_scores, score_base_only
from pathanchor.evaluation.pipeline import build_evidence_store, score_options, selected_options
from pathanchor.evaluation.registry import RuntimeConfig, resolve
from pathanchor.evaluation.reporting import (
    ReportSection,
    render_counts,
    render_mapping,
    render_sections,
)
from pathanchor.evaluation.subgroups import subgroup_report
from pathanchor.graph.builder import build_declared_graph
from pathanchor.graph.grounding import EntityDirectory
from pathanchor.graph.index import CompactGraph
from pathanchor.guideline.statements import default_table
from pathanchor.metrics.calibration import calibration_summary
from pathanchor.metrics.clinical import decision_curve, net_benefit
from pathanchor.metrics.discrimination import auprc, auroc
from pathanchor.models.estimator import PathAnchorEstimator, build_estimator
from pathanchor.training.seed import set_seed
from pathanchor.utils.logging import configure_logging, get_logger
from pathanchor.utils.serialization import write_text
from pathanchor.utils.types import FloatArray, IntArray

LOGGER = get_logger("cli.evaluate")


def parse(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="pathanchor-evaluate",
        description="Score a split, report the trust axes and the subgroup table.",
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--split", default=Split.TEST.value)
    parser.add_argument("override", nargs="*", default=[])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse(argv)
    configure_logging()
    runtime = resolve(arguments.config, tuple(arguments.override))
    split = Split(arguments.split)
    graph = build_declared_graph(runtime.census)
    directory = EntityDirectory.from_graph(graph, core_size=runtime.census.core_size)
    fixture = build_cohort_fixture(runtime.cohort)
    records = fixture.by_split(split)
    adjudicator = GuidelineAdjudicator.from_table(default_table())
    store = build_evidence_store(
        records,
        graph,
        directory,
        runtime.budget,
        adjudicator,
        OntologyValidator(graph=graph),
        CounterfactualCritic(),
    )
    set_seed(runtime.train.seed)
    model = build_estimator(runtime.estimator)
    matrix = score_options(store, graph, model)
    baseline = score_base_only(store, model)
    labels = store.labels()
    framework = row_scores(matrix)
    comparator = row_scores(baseline)
    probabilities = 1.0 / (1.0 + np.exp(-framework))
    layer = SplitConformal(alpha=runtime.conformal_alpha, tau_abs=runtime.conformal_tau_abs)
    training_records = fixture.by_split(Split.TRAIN)
    training_store = build_evidence_store(
        training_records,
        graph,
        directory,
        runtime.budget,
        adjudicator,
        OntologyValidator(graph=graph),
        CounterfactualCritic(),
    )
    training_scores = row_scores(score_options(training_store, graph, model))
    calibration_records = fixture.by_split(Split.CALIBRATION)
    calibration_scores = _split_scores(
        calibration_records, graph, directory, runtime, adjudicator, model
    )
    layer.fit(
        training_scores,
        training_store.labels(),
        calibration_scores,
        _labels_of(calibration_records),
    )
    coverage = coverage_report(layer, framework, labels)
    calibrator = ConformalCalibrator(layer=layer)
    decisions = [
        calibrator.decide(
            item.patient_id, option or "-", float(framework[row]), item.paths, bool(option)
        )
        for row, (item, option) in enumerate(
            zip(store.items, selected_options(matrix), strict=True)
        )
    ]
    deferred = sum(1 for item in decisions if item.deferred)
    calibration = calibration_summary(labels, probabilities)
    curve = decision_curve(labels, probabilities)
    subgroup = subgroup_report(records, labels, framework, comparator, runtime.subgroup_dimensions)
    body = render_sections(
        [
            ReportSection(
                "Record set",
                render_mapping(
                    {
                        "split": split.value,
                        "records": store.size,
                        "retained routes": len(store.flat_paths()),
                    }
                ),
            ),
            ReportSection("Agent ledger", render_mapping(dict(store.ledger.as_dict()))),
            ReportSection(
                "Discrimination",
                render_mapping(
                    {
                        "auroc": auroc(labels, framework),
                        "auprc": auprc(labels, framework),
                        "baseline auroc": auroc(labels, comparator),
                    }
                ),
            ),
            ReportSection(
                "Calibration and utility",
                render_mapping(
                    {
                        "brier": calibration.brier,
                        "ece": calibration.ece,
                        "net benefit at 0.35": net_benefit(labels, probabilities, 0.35),
                        "best strategy at 0.35": curve.best_strategy(0.35),
                        "coverage": coverage["coverage"],
                        "mean width": coverage["mean_width"],
                        "deferred": float(deferred),
                    }
                ),
            ),
            ReportSection(
                "Subgroup summary",
                render_counts(
                    {
                        str(key): int(value) if isinstance(value, bool) else value
                        for key, value in subgroup.summary().items()
                        if isinstance(value, (int, bool))
                    }
                ),
            ),
            ReportSection(
                "Subgroup rows",
                render_mapping(
                    {row.subgroup: row.delta_percentage_points for row in subgroup.rows}
                ),
            ),
        ]
    )
    if arguments.report is not None:
        write_text(arguments.report, body)
    LOGGER.info("evaluated %d records", store.size)
    print(body)
    return 0


def _split_scores(
    records: Sequence[PatientRecord],
    graph: CompactGraph,
    directory: EntityDirectory,
    runtime: RuntimeConfig,
    adjudicator: GuidelineAdjudicator,
    model: PathAnchorEstimator,
) -> FloatArray:
    store = build_evidence_store(
        records,
        graph,
        directory,
        runtime.budget,
        adjudicator,
        OntologyValidator(graph=graph),
        CounterfactualCritic(),
    )
    return row_scores(score_options(store, graph, model))


def _labels_of(records: Sequence[PatientRecord]) -> IntArray:
    return np.asarray([record.response for record in records], dtype=np.int64)


if __name__ == "__main__":
    raise SystemExit(main())
