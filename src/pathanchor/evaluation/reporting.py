"""Report rendering.

Ref: Sec. IV-A to IV-G (the reported tables and the numbers each one carries).

Reports are plain text, never markdown, because this release ships exactly one
Markdown file. Column widths are derived from the data rather than fixed, so a
long value widens its column instead of being silently truncated.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np


def render_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[object]],
    title: str = "",
    note: str = "",
) -> str:
    cells = [[_cell(value) for value in row] for row in rows]
    all_rows = [list(headers), *cells]
    widths = [max(len(row[column]) for row in all_rows) for column in range(len(headers))]
    lines: list[str] = []
    if title:
        lines.append(title)
        lines.append("-" * max(len(title), sum(widths) + 2 * len(widths)))
    header = "  ".join(headers[column].ljust(widths[column]) for column in range(len(headers)))
    lines.append(header)
    lines.append("  ".join("-" * width for width in widths))
    for row in cells:
        lines.append("  ".join(row[column].ljust(widths[column]) for column in range(len(headers))))
    if note:
        lines.append("")
        lines.append(note)
    return "\n".join(lines)


def _cell(value: object) -> str:
    if isinstance(value, float):
        if not np.isfinite(value):
            return "n/a"
        return f"{value:.4f}"
    return str(value)


@dataclass(frozen=True, slots=True)
class ReportSection:
    title: str
    body: str

    def render(self) -> str:
        return f"{self.title}\n{'=' * len(self.title)}\n{self.body}\n"


def render_sections(sections: Iterable[ReportSection]) -> str:
    return "\n".join(section.render() for section in sections)


def render_mapping(entries: Mapping[str, object], title: str = "") -> str:
    rows = [[key, value] for key, value in entries.items()]
    return render_table(["key", "value"], rows, title=title)


def render_counts(entries: Mapping[str, int], title: str = "") -> str:
    rows = [[key, value] for key, value in entries.items()]
    return render_table(["item", "count"], rows, title=title)


def render_estimates(
    entries: Mapping[str, tuple[float, float, float]],
    title: str = "",
) -> str:
    rows = [[key, value[0], value[1], value[2]] for key, value in entries.items()]
    return render_table(["quantity", "value", "low", "high"], rows, title=title)


def render_status(checks: Sequence[Mapping[str, object]], title: str = "") -> str:
    rows = [
        [check.get("name", ""), check.get("status", ""), check.get("evidence", "")]
        for check in checks
    ]
    return render_table(["check", "status", "evidence"], rows, title=title)


def status_tally(checks: Sequence[Mapping[str, object]]) -> dict[str, int]:
    tally: dict[str, int] = {}
    for check in checks:
        status = str(check.get("status", "UNKNOWN"))
        tally[status] = tally.get(status, 0) + 1
    return dict(sorted(tally.items()))


def overall_status(tally: Mapping[str, int], manuscript_failures: int = 0) -> str:
    """The verdict for the release as a whole.

    A manuscript-arithmetic failure does not make the code wrong, so it is
    reported beside the code verdict rather than folded into it.
    """
    if tally.get("FAIL", 0) > 0:
        return "FAIL"
    blocked = tally.get("BLOCKED", 0) + tally.get("NOT_RUN", 0)
    if blocked == 0 and manuscript_failures == 0:
        return "PASS"
    return "PARTIALLY_VERIFIED"
