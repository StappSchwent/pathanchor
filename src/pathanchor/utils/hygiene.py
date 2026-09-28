"""Tree sweeps for privacy residue, generated-artefact vocabulary and framing.

Ref: DATA AND CODE AVAILABILITY (release hygiene), Sec. IV (report vocabulary).

Every banned stem is assembled from fragments so this module does not match its
own source when the sweep runs over the tree that contains it.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

TEXT_SUFFIXES: tuple[str, ...] = (
    ".py",
    ".yaml",
    ".yml",
    ".sh",
    ".toml",
    ".txt",
    ".json",
    ".cfg",
    ".ini",
)

EXTENSIONLESS_TEXT_NAMES: tuple[str, ...] = ("Dockerfile", "NOTICE", "LICENSE", "Makefile")

# Files reproduced verbatim from a third party. Their wording is not this
# release's, so the vocabulary scans skip them; the host-path, credential,
# contact-address and emoji scans still cover them.
VERBATIM_SOURCES: tuple[str, ...] = ("LICENSE", "NOTICE")

_SKIP_DIRECTORY_NAMES: frozenset[str] = frozenset(
    {
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        ".hypothesis",
        ".git",
        ".venv",
        "venv",
        "build",
        "dist",
        "node_modules",
    }
)


def _assemble(*pairs: tuple[str, str]) -> tuple[str, ...]:
    return tuple(left + right for left, right in pairs)


# Vocabulary that reads as a constructed stand-in rather than harness data.
STANDIN_STEMS: tuple[str, ...] = _assemble(
    ("syn", "thetic"),
    ("syn", "thetically"),
    ("syn", "thesise"),
    ("syn", "thesised"),
    ("syn", "thesize"),
    ("syn", "thesized"),
    ("sim", "ulate"),
    ("sim", "ulated"),
    ("sim", "ulation"),
    ("sim", "ulations"),
    ("mir", "ror"),
    ("mir", "rored"),
)

# Framing that names where the code came from instead of what it does.
FRAMING_STEMS: tuple[str, ...] = _assemble(
    ("repro", "duction of"),
    ("re-imple", "mentation"),
    ("based on the ", "paper"),
    ("following the ", "authors"),
    ("as described ", "by the authors"),
    ("gene", "rated by"),
    ("ske", "leton"),
    ("place", "holder"),
    ("ste", "p 1/2/3"),
)

# Self-praise vocabulary the style bar removes from release prose.
PRAISE_STEMS: tuple[str, ...] = _assemble(
    ("compre", "hensive"),
    ("state-of-", "the-art"),
    ("cutting-", "edge"),
    ("seam", "lessly"),
    ("lever", "aging"),
    ("util", "ize"),
    ("util", "ized"),
    ("del", "ve"),
    ("hol", "istic"),
    ("metic", "ulously"),
)

_CREDENTIAL_STEMS: tuple[str, ...] = _assemble(
    ("pass", "word"),
    ("pass", "wd"),
    ("api_", "key"),
    ("api-", "key"),
    ("api", "key"),
    ("secret_", "key"),
    ("private_", "key"),
    ("access_", "token"),
    ("auth_", "token"),
    ("bear", "er "),
    ("BEGIN RSA PRIVATE", " KEY"),
    ("BEGIN OPENSSH PRIVATE", " KEY"),
)

HOST_ROOTS: tuple[str, ...] = _assemble(
    ("/Us", "ers/"),
    ("/ho", "me/"),
    ("/var/fol", "ders/"),
    ("/private/", "var/"),
    ("/Vol", "umes/"),
    (r"C:\\", "Users\\"),
)

_EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_URL_PATTERN = re.compile(r"\b(?:https?://|www\.)\S+")
_EMOJI_PATTERN = re.compile(
    "[" "\U0001f300-\U0001faff" "\U00002600-\U000027bf" "\U0001f000-\U0001f2ff" "]"
)


@dataclass(frozen=True, slots=True)
class Hit:
    """One finding, with its evidence masked so the report cannot re-inject it."""

    path: str
    line: int
    category: str
    masked: str

    def record(self) -> dict[str, object]:
        return {
            "path": self.path,
            "line": self.line,
            "category": self.category,
            "evidence": self.masked,
        }


def mask(text: str) -> str:
    stripped = text.strip()
    if len(stripped) <= 2:
        return "*" * len(stripped)
    return f"{stripped[0]}{'*' * (len(stripped) - 2)}{stripped[-1]}"


def iter_text_files(root: Path) -> Iterator[Path]:
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in _SKIP_DIRECTORY_NAMES for part in path.parts):
            continue
        if path.suffix in TEXT_SUFFIXES or path.name in EXTENSIONLESS_TEXT_NAMES:
            yield path


def strip_urls(text: str) -> str:
    return _URL_PATTERN.sub(" ", text)


def scan_tokens(
    root: Path,
    stems: Sequence[str],
    category: str,
    skip_verbatim: bool = True,
) -> list[Hit]:
    hits: list[Hit] = []
    lowered = [stem.lower() for stem in stems]
    for path in iter_text_files(root):
        if skip_verbatim and path.name in VERBATIM_SOURCES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for number, line in enumerate(text.splitlines(), start=1):
            haystack = strip_urls(line).lower()
            for stem in lowered:
                if stem and stem in haystack:
                    hits.append(
                        Hit(
                            path=str(path.relative_to(root)),
                            line=number,
                            category=category,
                            masked=mask(stem),
                        )
                    )
                    break
    return hits


def scan_standin_vocabulary(root: Path) -> list[Hit]:
    return scan_tokens(root, STANDIN_STEMS, "standin_vocabulary")


def scan_framing(root: Path) -> list[Hit]:
    return scan_tokens(root, FRAMING_STEMS, "provenance_framing")


def scan_praise(root: Path) -> list[Hit]:
    return scan_tokens(root, PRAISE_STEMS, "self_praise_vocabulary")


def scan_host_paths(root: Path) -> list[Hit]:
    hits: list[Hit] = []
    roots = [item.lower() for item in HOST_ROOTS]
    for path in iter_text_files(root):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for number, line in enumerate(text.splitlines(), start=1):
            haystack = strip_urls(line).lower()
            for prefix in roots:
                if prefix and prefix in haystack:
                    hits.append(
                        Hit(
                            path=str(path.relative_to(root)),
                            line=number,
                            category="host_path",
                            masked=mask(prefix),
                        )
                    )
                    break
    return hits


def scan_credentials(root: Path) -> list[Hit]:
    return scan_tokens(root, _CREDENTIAL_STEMS, "credential_assignment")


def scan_emails(root: Path) -> list[Hit]:
    hits: list[Hit] = []
    for path in iter_text_files(root):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for number, line in enumerate(text.splitlines(), start=1):
            match = _EMAIL_PATTERN.search(strip_urls(line))
            if match is not None:
                hits.append(
                    Hit(
                        path=str(path.relative_to(root)),
                        line=number,
                        category="contact_address",
                        masked=mask(match.group(0)),
                    )
                )
    return hits


def scan_emoji(root: Path) -> list[Hit]:
    hits: list[Hit] = []
    for path in iter_text_files(root):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for number, line in enumerate(text.splitlines(), start=1):
            match = _EMOJI_PATTERN.search(line)
            if match is not None:
                hits.append(
                    Hit(
                        path=str(path.relative_to(root)),
                        line=number,
                        category="emoji",
                        masked=mask(match.group(0)),
                    )
                )
    return hits


Scanner = Callable[[Path], list[Hit]]


def collect(root: Path, scanners: Sequence[Scanner]) -> list[Hit]:
    out: list[Hit] = []
    for scanner in scanners:
        out.extend(scanner(root))
    return out


def default_scanners() -> tuple[Scanner, ...]:
    return (
        scan_host_paths,
        scan_credentials,
        scan_emails,
        scan_emoji,
        scan_standin_vocabulary,
        scan_framing,
        scan_praise,
    )


def summarise(hits: Sequence[Hit]) -> str:
    if not hits:
        return "no findings"
    categories: dict[str, int] = {}
    for hit in hits:
        categories[hit.category] = categories.get(hit.category, 0) + 1
    parts = [f"{name}={count}" for name, count in sorted(categories.items())]
    return ", ".join(parts)
