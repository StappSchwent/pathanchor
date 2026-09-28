"""Ablation entry point.

Ref: Table II Panel A and Panel B, Sec. IV-B.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from pathanchor.agents.guideline_adjudicator import GuidelineAdjudicator
from pathanchor.cohort.fixture import build_cohort_fixture
from pathanchor.cohort.schema import Split
from pathanchor.evaluation.ablations import (
    PANEL_A_ARMS,
    PANEL_B_ARMS,
    AblationFlags,
    ArmResult,
    build_store_for_arm,
    evaluate_arm,
    leave_one_out_deltas,
    marginal_gains,
    panel_rows,
    synergy_surplus,
)
from pathanchor.evaluation.registry import RuntimeConfig, resolve
from pathanchor.evaluation.reporting import ReportSection, render_sections, render_table
from pathanchor.graph.builder import build_declared_graph
from pathanchor.graph.grounding import EntityDirectory
from pathanchor.graph.index import CompactGraph
from pathanchor.guideline.statements import default_table
from pathanchor.models.estimator import build_estimator
from pathanchor.training.seed import set_seed
from pathanchor.utils.logging import configure_logging, get_logger
from pathanchor.utils.serialization import write_text

LOGGER = get_logger("cli.ablate")

PANELS: dict[str, tuple[tuple[str, AblationFlags], ...]] = {
    "panel_a": PANEL_A_ARMS,
    "panel_b": PANEL_B_ARMS,
}


def parse(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="pathanchor-ablate",
        description="Run a component panel or a leave-one-out panel.",
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--panel", choices=sorted(PANELS), default="panel_b")
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("override", nargs="*", default=[])
    return parser.parse_args(argv)


def run_panel(
    runtime: RuntimeConfig,
    graph: CompactGraph,
    directory: EntityDirectory,
    panel: tuple[tuple[str, AblationFlags], ...],
    limit: int,
) -> list[ArmResult]:
    fixture = build_cohort_fixture(runtime.cohort)
    records = fixture.by_split(Split.TEST)[:limit]
    adjudicator = GuidelineAdjudicator.from_table(default_table())
    set_seed(runtime.train.seed)
    model = build_estimator(runtime.estimator)
    results: list[ArmResult] = []
    for name, flags in panel:
        store = build_store_for_arm(flags, records, graph, directory, runtime.budget, adjudicator)
        results.append(
            evaluate_arm(
                name, flags, records, graph, directory, runtime.budget, adjudicator, model, store
            )
        )
    return results


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse(argv)
    configure_logging()
    runtime = resolve(arguments.config, tuple(arguments.override))
    graph = build_declared_graph(runtime.census)
    directory = EntityDirectory.from_graph(graph, core_size=runtime.census.core_size)
    results = run_panel(runtime, graph, directory, PANELS[arguments.panel], arguments.limit)
    rows = panel_rows(results)
    table = render_table(
        ["arm", "disabled", "auroc", "ece", "discordance", "routes"],
        [
            [
                row["arm"],
                row["disabled"],
                row["auroc"],
                row["ece"],
                row["discordance"],
                row["retained_paths"],
            ]
            for row in rows
        ],
        title=f"{arguments.panel} on {len(results)} arms",
    )
    gains = marginal_gains(results)
    deltas = leave_one_out_deltas(results)
    body = render_sections(
        [
            ReportSection("Arms", table),
            ReportSection(
                "Marginal gains (percentage points)",
                "\n".join(f"{name}: {value:+.2f}" for name, value in gains.items()) or "none",
            ),
            ReportSection(
                "Leave-one-out deltas (percentage points)",
                "\n".join(f"{name}: {value:+.2f}" for name, value in deltas.items()) or "none",
            ),
            ReportSection("Synergy surplus", f"{synergy_surplus(results):+.2f} pp"),
        ]
    )
    if arguments.report is not None:
        write_text(arguments.report, body)
    LOGGER.info("ran %d arms", len(results))
    print(body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
