"""Verification entry point.

Ref: DATA AND CODE AVAILABILITY (the frozen configuration files and seeds),
Sec. IV (the reported tables).
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

from pathanchor.evaluation.verification import main as verify_main

if __name__ == "__main__":
    raise SystemExit(verify_main(sys.argv[1:]))


def main(argv: Sequence[str] | None = None) -> int:
    return verify_main(argv)
