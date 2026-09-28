"""Checkpointing.

Ref: DATA AND CODE AVAILABILITY (the frozen configuration files and random seeds
needed to reproduce every table).

Writes go through a temporary file and ``os.replace`` so a reader never sees a
half-written checkpoint, and the restored run restores the data-order seed as well
as the model weights -- otherwise a resumed run reshuffles differently and the
reported numbers stop matching.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class CheckpointContents:
    epoch: int
    step: int
    seed: int
    model_state: Mapping[str, Any]
    optimiser_state: Mapping[str, Any] | None
    scheduler_state: Mapping[str, Any] | None
    history: tuple[float, ...]
    data_order_seed: int

    def summary(self) -> dict[str, object]:
        return {
            "epoch": self.epoch,
            "step": self.step,
            "seed": self.seed,
            "tensors": len(self.model_state),
            "epochs_recorded": len(self.history),
        }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimiser: torch.optim.Optimizer | None,
    scheduler: object | None,
    epoch: int,
    step: int,
    seed: int,
    history: list[float],
    data_order_seed: int,
) -> Path:
    payload = {
        "epoch": epoch,
        "step": step,
        "seed": seed,
        "model_state": model.state_dict(),
        "optimiser_state": optimiser.state_dict() if optimiser is not None else None,
        "scheduler_state": getattr(scheduler, "state_dict", lambda: None)(),
        "history": list(history),
        "data_order_seed": data_order_seed,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_ckpt_", suffix=".pt")
    os.close(handle)
    temporary = Path(name)
    temporary.chmod(0o644)
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)
    return path


def load_checkpoint(path: Path) -> CheckpointContents:
    payload: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
    return CheckpointContents(
        epoch=int(payload["epoch"]),
        step=int(payload["step"]),
        seed=int(payload["seed"]),
        model_state=payload["model_state"],
        optimiser_state=payload.get("optimiser_state"),
        scheduler_state=payload.get("scheduler_state"),
        history=tuple(float(value) for value in payload.get("history", [])),
        data_order_seed=int(payload.get("data_order_seed", payload["seed"])),
    )


def restore(
    path: Path,
    model: nn.Module,
    optimiser: torch.optim.Optimizer | None = None,
    scheduler: object | None = None,
) -> CheckpointContents:
    contents = load_checkpoint(path)
    model.load_state_dict(contents.model_state)
    if optimiser is not None and contents.optimiser_state is not None:
        optimiser.load_state_dict(dict(contents.optimiser_state))
    if (
        scheduler is not None
        and contents.scheduler_state is not None
        and hasattr(scheduler, "load_state_dict")
    ):
        scheduler.load_state_dict(contents.scheduler_state)
    return contents


def state_digest(state: Mapping[str, Any]) -> str:
    """A digest of the tensor payloads, not of the container bytes.

    ``torch.save`` embeds container metadata, so a file hash of a checkpoint moves
    on every write while the parameters are identical; summing the payload makes
    the value a function of the model rather than of the serialiser.
    """
    parts: list[str] = []
    for name in sorted(state):
        value = state[name]
        if isinstance(value, torch.Tensor):
            flat = value.detach().to(torch.float64).cpu()
            parts.append(f"{name}:{flat.shape}:{float(flat.sum()):.6f}")
    return "|".join(parts)


def checkpoint_round_trip(
    model: nn.Module,
    directory: Path,
    seed: int,
) -> dict[str, object]:
    """Save and reload a model, reporting whether the payload survived.

    Writes into a caller-supplied scratch directory so a verification run cannot
    leave a checkpoint in the release tree.
    """
    before = state_digest(model.state_dict())
    path = save_checkpoint(
        directory / "round_trip.pt",
        model,
        None,
        None,
        0,
        0,
        seed,
        [],
        seed,
    )
    clone = type(model)(model.config)
    contents = restore(path, clone)
    after = state_digest(clone.state_dict())
    return {
        "identical": before == after,
        "tensors": len(contents.model_state),
        "seed_restored": contents.seed == seed,
        "path_name": path.name,
    }


def history_moves(history: list[float], window: int = 5) -> bool:
    """True when the trailing loss is below the leading loss over ``window`` steps."""
    if len(history) < window * 2:
        return False
    head = float(np.mean(history[:window]))
    tail = float(np.mean(history[-window:]))
    return tail < head
