"""Reproducible G3.1 initialization and crash-safe checkpoint writes."""
from __future__ import annotations

import os
from pathlib import Path
import tempfile

import torch


def seed_training_rng(seed: int) -> None:
    """Seed CPU and available CUDA generators before model initialization."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint_atomic(payload: object, destination: str | Path) -> None:
    """Serialize beside the destination, fsync, then atomically replace it.

    A serialization or promotion failure leaves an existing checkpoint untouched.
    The temporary file is always on the destination filesystem so os.replace
    does not cross devices.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
