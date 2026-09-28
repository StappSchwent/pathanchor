"""YAML configuration loading with dotted command-line overrides.

Ref: Sec. III (configuration surface), Sec. IV (one config per reported experiment).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml

_CONFIG_ROOT = ("configs",)


class ConfigError(ValueError):
    """Raised when a configuration key is missing or has the wrong shape."""


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"configuration file not found: {path.name}")
    with path.open("r", encoding="utf-8") as handle:
        raw: Any = yaml.safe_load(handle)
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigError(f"configuration root of {path.name} must be a mapping")
    return {str(key): value for key, value in raw.items()}


def load_layered(path: Path, seen: frozenset[Path] = frozenset()) -> dict[str, Any]:
    """Load a configuration and every file it names under ``base``.

    A layer may be given as one path or a list of paths, each resolved relative to
    the file that names it. Later layers override earlier ones, and the file's own
    keys override all of them, so an experiment file states only what it changes.
    """
    resolved = path.resolve()
    if resolved in seen:
        raise ConfigError(f"configuration cycle through {path.name}")
    document = load_yaml(path)
    declared = document.pop("base", None)
    if declared is None:
        return document
    names = [declared] if isinstance(declared, str) else list(declared)
    merged: dict[str, Any] = {}
    for name in names:
        child = (path.parent / str(name)).resolve()
        merged = deep_merge(merged, load_layered(child, seen | {resolved}))
    return deep_merge(merged, document)


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {str(k): v for k, v in base.items()}
    for key, value in override.items():
        name = str(key)
        current = merged.get(name)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[name] = deep_merge(current, value)
        else:
            merged[name] = value
    return merged


def parse_override(text: str) -> tuple[list[str], Any]:
    if "=" not in text:
        raise ConfigError(f"override '{text}' must have the form key.path=value")
    key, _, value = text.partition("=")
    path = [part for part in key.strip().split(".") if part]
    if not path:
        raise ConfigError(f"override '{text}' has an empty key path")
    parsed: Any = yaml.safe_load(value)
    return path, parsed


def apply_overrides(data: Mapping[str, Any], overrides: Iterable[str]) -> dict[str, Any]:
    result = {str(k): v for k, v in data.items()}
    for text in overrides:
        path, value = parse_override(text)
        cursor: dict[str, Any] = result
        for part in path[:-1]:
            child = cursor.get(part)
            if not isinstance(child, dict):
                child = {}
                cursor[part] = child
            cursor = child
        cursor[path[-1]] = value
    return result


def load_config(path: Path, overrides: Iterable[str] = ()) -> dict[str, Any]:
    return apply_overrides(load_layered(path), overrides)


def section(data: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ConfigError(f"section '{key}' must be a mapping")
    return {str(k): v for k, v in value.items()}


def require(data: Mapping[str, Any], key: str) -> Any:
    if key not in data:
        raise ConfigError(f"required key '{key}' is absent")
    return data[key]


def get_float(data: Mapping[str, Any], key: str, default: float | None = None) -> float:
    value = data.get(key, default)
    if value is None:
        raise ConfigError(f"key '{key}' has no value and no default")
    if isinstance(value, bool):
        raise ConfigError(f"key '{key}' must be a number, got a boolean")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"key '{key}' must be a number") from exc


def get_int(data: Mapping[str, Any], key: str, default: int | None = None) -> int:
    value = data.get(key, default)
    if value is None:
        raise ConfigError(f"key '{key}' has no value and no default")
    if isinstance(value, bool):
        raise ConfigError(f"key '{key}' must be an integer, got a boolean")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"key '{key}' must be an integer") from exc


def get_str(data: Mapping[str, Any], key: str, default: str | None = None) -> str:
    value = data.get(key, default)
    if value is None:
        raise ConfigError(f"key '{key}' has no value and no default")
    return str(value)


def get_bool(data: Mapping[str, Any], key: str, default: bool | None = None) -> bool:
    value = data.get(key, default)
    if value is None:
        raise ConfigError(f"key '{key}' has no value and no default")
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "yes", "1", "on"}:
        return True
    if text in {"false", "no", "0", "off"}:
        return False
    raise ConfigError(f"key '{key}' must be a boolean")


def get_float_list(data: Mapping[str, Any], key: str) -> list[float]:
    value = data.get(key)
    if value is None:
        raise ConfigError(f"key '{key}' has no value")
    if not isinstance(value, (list, tuple)):
        raise ConfigError(f"key '{key}' must be a sequence of numbers")
    return [float(item) for item in value]


def get_str_list(data: Mapping[str, Any], key: str) -> list[str]:
    value = data.get(key)
    if value is None:
        raise ConfigError(f"key '{key}' has no value")
    if not isinstance(value, (list, tuple)):
        raise ConfigError(f"key '{key}' must be a sequence of strings")
    return [str(item) for item in value]


def repo_root(config_path: Path) -> Path:
    """Resolve the repository root from a file under configs/.

    A path of the shape <root>/configs/<group>/<name>.yaml resolves to <root>,
    which is one level above the obvious parent-of-parent.
    """
    resolved = config_path.resolve()
    parts = resolved.parts
    for index in range(len(parts) - 1, -1, -1):
        if parts[index] == _CONFIG_ROOT[0]:
            return Path(*parts[:index])
    return resolved.parent


def relative_to_root(path: Path, root: Path) -> str:
    """Render a path for any written artefact.

    Anything that reaches disk is stored relative to the repository root so a
    report never carries a host path.
    """
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return path.name
