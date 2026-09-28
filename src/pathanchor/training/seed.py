"""Seed control.

Ref: DATA AND CODE AVAILABILITY ("the frozen configuration files and random seeds
needed to reproduce every table").

Every entry point that allocates a tensor calls this before the first
allocation, not merely before fitting, or the initialisation is drawn from the
process default and the run is not reproducible.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> int:
    """Seed python, NumPy and torch, and return the seed that was applied."""
    if seed < 0:
        raise ValueError("the seed must be non-negative")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed


def numpy_generator(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def derived_seed(base: int, salt: str) -> int:
    """A stable child seed, so independent stages do not share a stream."""
    from hashlib import sha256

    digest = sha256(f"{base}|{salt}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def torch_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
