"""Atomic writers and digests for every artefact the release ships.

Ref: Sec. IV (reported tables), DATA AND CODE AVAILABILITY (frozen configuration files).
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

JsonValue = Any


def canonical_json(payload: JsonValue) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def write_text(path: Path, text: str) -> None:
    """Write text through a temporary file so a reader never sees a partial file.

    The temporary file is created 0600 by mkstemp; the mode is widened before the
    replace so the shipped artefacts are readable like the rest of the tree.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_", suffix=path.suffix)
    temporary = Path(name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(text)
        temporary.chmod(0o644)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)


def write_json(path: Path, payload: JsonValue) -> None:
    write_text(path, canonical_json(payload))


def read_json(path: Path) -> JsonValue:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_", suffix=path.suffix)
    temporary = Path(name)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
        temporary.chmod(0o644)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gate_result(
    label: str,
    returncode: int,
    stdout_tail: str,
    environment: Mapping[str, str] | None = None,
) -> dict[str, JsonValue]:
    """Describe a subprocess gate by its exit code only.

    The tool's last line of output moves between runs (progress padding, skipped
    counts, durations) so it is never recorded, and never used as the verdict.
    """
    return {
        "gate": label,
        "exit_code": int(returncode),
        "status": "PASS" if returncode == 0 else "FAIL",
        "output_lines": len([line for line in stdout_tail.splitlines() if line.strip()]),
    }


def subprocess_environment(base: Mapping[str, str]) -> dict[str, str]:
    environment = {str(key): str(value) for key, value in base.items()}
    environment.update({"COLUMNS": "80", "LINES": "24", "NO_COLOR": "1", "PY_COLORS": "0"})
    return environment


def digest_manifest(entries: Iterable[tuple[str, bytes]]) -> dict[str, str]:
    return {name: sha256_bytes(payload) for name, payload in entries}
