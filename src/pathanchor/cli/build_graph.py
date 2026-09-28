"""Graph construction entry point.

Ref: Sec. III-B (the released graph and its census), DATA AND CODE AVAILABILITY
(the derived knowledge-graph release).
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from pathanchor.evaluation.registry import repo_root, resolve
from pathanchor.evaluation.reporting import ReportSection, render_sections
from pathanchor.graph.builder import build_declared_graph
from pathanchor.graph.census import census_report, save_snapshot, snapshot_digest
from pathanchor.utils.logging import configure_logging, get_logger
from pathanchor.utils.serialization import write_text

LOGGER = get_logger("cli.build_graph")


def parse(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="pathanchor-build-graph",
        description="Materialise the released graph and write its snapshot and census.",
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse(argv)
    configure_logging()
    config_path = arguments.config
    if config_path is None:
        config_path = Path(__file__).resolve().parents[3] / "configs" / "experiment" / "main.yaml"
    root = repo_root(config_path)
    runtime = resolve(config_path)
    graph = build_declared_graph(runtime.census)
    census = graph.census()
    report = census_report(census, runtime.census)
    snapshot = (
        arguments.out if arguments.out is not None else root / "data" / "graph" / "snapshot.npz"
    )
    save_snapshot(graph, snapshot)
    body = render_sections(
        [
            ReportSection(
                "Graph census",
                "\n".join(
                    [
                        f"entities        {census.entities}",
                        f"edges           {census.edges}",
                        f"node types      {census.node_types}",
                        f"relation types  {census.relation_types}",
                        f"discrepancies   {list(report.discrepancies)}",
                    ]
                ),
            ),
            ReportSection(
                "Snapshot", f"written to {snapshot.name}\npayload digest {snapshot_digest(graph)}"
            ),
        ]
    )
    if arguments.report is not None:
        write_text(arguments.report, body)
    LOGGER.info("graph census: %d entities, %d edges", census.entities, census.edges)
    print(body)
    return 0 if report.exact else 1


if __name__ == "__main__":
    raise SystemExit(main())
