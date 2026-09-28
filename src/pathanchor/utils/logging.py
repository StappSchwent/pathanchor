"""Logging configuration.

Ref: Sec. IV (one plain-text report per reported quantity).
"""

from __future__ import annotations

import logging
import sys

_FORMAT = "%(asctime)s %(levelname)-8s %(name)s %(message)s"
_DATEFMT = "%Y-%m-%dT%H:%M:%S"
_CONFIGURED = False


def configure_logging(level: str = "INFO", stream: object | None = None) -> None:
    global _CONFIGURED
    handler = logging.StreamHandler(stream if stream is not None else sys.stderr)  # type: ignore[arg-type]
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
    root = logging.getLogger("pathanchor")
    root.handlers = [handler]
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    if not _CONFIGURED:
        configure_logging()
    if name.startswith("pathanchor"):
        return logging.getLogger(name)
    return logging.getLogger(f"pathanchor.{name}")


def strip_timestamps(text: str) -> str:
    """Drop the leading timestamp of each line.

    Captured log text that reaches a digested artefact must not carry the clock,
    so anything recorded from a log stream goes through here first.
    """
    out: list[str] = []
    for line in text.splitlines():
        parts = line.split(" ", 2)
        if len(parts) == 3 and len(parts[0]) == 19 and parts[0][10] == "T":
            out.append(f"{parts[1]} {parts[2]}")
        else:
            out.append(line)
    return "\n".join(out)
