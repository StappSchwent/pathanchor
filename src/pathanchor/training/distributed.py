"""Distributed context.

Ref: Sec. IV-G ("model training required 41 GPU-hours on four A100 accelerators").

The trainer reads its rank and world size from the environment rather than
requiring a launcher, so the same entry point runs a single-process job and a
torchrun job. Nothing here reduces gradients implicitly: a job that launches with
a world size above one and never calls the synchronisation helper would report
per-rank progress as if it were global, so the helper exists and the engine calls
it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class DistributedContext:
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    backend: str = "gloo"

    @property
    def is_primary(self) -> bool:
        return self.rank == 0

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    def device(self) -> torch.device:
        if self.enabled and torch.cuda.is_available():
            return torch.device("cuda", self.local_rank)
        return torch.device("cpu")

    def summary(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "world_size": self.world_size,
            "backend": self.backend,
            "device": self.device().type,
        }


def from_environment() -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    backend = os.environ.get("PATHANCHOR_BACKEND", "nccl" if torch.cuda.is_available() else "gloo")
    return DistributedContext(rank, local_rank, world_size, backend)


def initialise(context: DistributedContext) -> DistributedContext:
    if not context.enabled:
        return context
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend=context.backend)
    return context


def shutdown(context: DistributedContext) -> None:
    if context.enabled and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def wrap(module: nn.Module, context: DistributedContext) -> nn.Module:
    if not context.enabled:
        return module
    device = context.device()
    module = module.to(device)
    if device.type == "cuda":
        return nn.parallel.DistributedDataParallel(module, device_ids=[context.local_rank])
    return nn.parallel.DistributedDataParallel(module)


def unwrap(module: nn.Module) -> nn.Module:
    if isinstance(
        module,
        (nn.parallel.DistributedDataParallel, nn.parallel.DataParallel),
    ):
        inner: nn.Module = module.module
        return inner
    return module


def barrier(context: DistributedContext) -> None:
    if context.enabled and torch.distributed.is_initialized():
        torch.distributed.barrier()


def all_reduce_mean(value: float, context: DistributedContext) -> float:
    """Mean of a scalar across ranks.

    Called by the engine at the end of every epoch, so the reported loss is the
    global one rather than rank zero's.
    """
    if not context.enabled or not torch.distributed.is_initialized():
        return float(value)
    tensor = torch.tensor([float(value)], dtype=torch.float64)
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return float(tensor.item() / context.world_size)


def shard_indices(size: int, context: DistributedContext) -> list[int]:
    """A contiguous shard of a range, used to split the record order by rank."""
    if not context.enabled:
        return list(range(size))
    per_rank = size // context.world_size
    remainder = size % context.world_size
    start = per_rank * context.rank + min(context.rank, remainder)
    stop = start + per_rank + (1 if context.rank < remainder else 0)
    return list(range(start, stop))
