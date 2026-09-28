"""Training entry point.

Ref: Sec. IV-G (the reported training budget), Algorithm 1 (the optimised score).
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from pathanchor.agents.counterfactual_critic import CounterfactualCritic
from pathanchor.agents.guideline_adjudicator import GuidelineAdjudicator
from pathanchor.agents.ontology_validator import OntologyValidator
from pathanchor.cohort.fixture import build_cohort_fixture
from pathanchor.cohort.schema import Split
from pathanchor.evaluation.pipeline import build_evidence_store
from pathanchor.evaluation.registry import repo_root, resolve
from pathanchor.evaluation.reporting import ReportSection, render_mapping, render_sections
from pathanchor.graph.builder import build_declared_graph
from pathanchor.graph.grounding import EntityDirectory
from pathanchor.guideline.statements import default_table
from pathanchor.models.estimator import build_estimator
from pathanchor.training.checkpoint import save_checkpoint
from pathanchor.training.engine import TrainingEngine
from pathanchor.utils.logging import configure_logging, get_logger
from pathanchor.utils.serialization import write_text

LOGGER = get_logger("cli.train")


def parse(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="pathanchor-train",
        description="Fit the path scorer on the training split of the cohort fixture.",
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("override", nargs="*", default=[])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse(argv)
    configure_logging()
    runtime = resolve(arguments.config, tuple(arguments.override))
    root = repo_root(arguments.config)
    graph = build_declared_graph(runtime.census)
    directory = EntityDirectory.from_graph(graph, core_size=runtime.census.core_size)
    fixture = build_cohort_fixture(runtime.cohort)
    records = fixture.by_split(Split.TRAIN)
    store = build_evidence_store(
        records,
        graph,
        directory,
        runtime.budget,
        GuidelineAdjudicator.from_table(default_table()),
        OntologyValidator(graph=graph),
        CounterfactualCritic(),
    )
    model = build_estimator(runtime.estimator)
    engine = TrainingEngine(model, store, graph, runtime.train)
    history = engine.fit()
    summary = history.summary()
    body = render_sections(
        [
            ReportSection("Configuration", render_mapping(runtime.summary())),
            ReportSection("Training", render_mapping(summary)),
            ReportSection(
                "Agent ledger",
                render_mapping(dict(store.ledger.as_dict())),
            ),
        ]
    )
    if arguments.report is not None:
        write_text(arguments.report, body)
    if arguments.checkpoint is not None:
        save_checkpoint(
            arguments.checkpoint,
            model,
            engine.optimiser,
            engine.schedule,
            history.epochs,
            history.steps,
            runtime.train.seed,
            history.losses,
            runtime.train.seed,
        )
    LOGGER.info("training finished after %d steps", history.steps)
    print(body)
    del root
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
