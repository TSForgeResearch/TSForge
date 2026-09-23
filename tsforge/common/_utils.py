import logging
import os
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Union
import math
import numpy as np
import yaml
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def mix_seed(*parts: int) -> int:
    """
    Combine integers into a well-scattered 63-bit seed (splitmix64 mixing).

    Keying a generator on consecutive integers (step, step+1, ...) can leave
    neighbouring streams correlated. Mixing removes that, so per-step seeds
    behave like independent draws rather than adjacent ones.
    """
    M = (1 << 64) - 1
    h = 0
    for p in parts:
        h = (h + (int(p) & M) + 0x9E3779B97F4A7C15) & M
        h ^= h >> 30
        h = (h * 0xBF58476D1CE4E5B9) & M
        h ^= h >> 27
        h = (h * 0x94D049BB133111EB) & M
        h ^= h >> 31
    return h & ((1 << 63) - 1)


def set_determinism(seed: int, strict: bool = True) -> None:
    """
    Pin the RNG streams before the model is constructed.

    theta_0 is drawn from the global torch stream, and torch's default seed is
    randomised per process, so seeding inside fit() happens after init and is
    too late to make initial weights reproducible.

    Parameters
    ----------
    seed    Base seed for python / numpy / torch (CPU and all CUDA devices).
    strict  Also pin kernel selection: deterministic cuDNN algorithms, and
            raise on any op with no deterministic CUDA implementation rather
            than let it vary silently. Turn off only for profiling.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)           # also seeds all CUDA devices
    torch.cuda.manual_seed_all(seed)  # explicit; redundant with the above

    if strict:
        # Required before the first cuBLAS GEMM, or use_deterministic_algorithms
        # raises when one runs.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)


def build_optimizer(params, mcfg) -> torch.optim.Optimizer:
    """
    Single source of truth for optimiser hyperparameters.

    These were previously hardcoded in three call sites (train(),
    _distributed_worker() and BaseModel.setup_training()), which is how three
    copies of the same literals drift apart. Defaults here are identical to
    those literals, so this changes no behaviour.
    """
    name = getattr(mcfg, "optimizer_name", "AdamW")
    if name != "AdamW":
        raise ValueError(
            f"optimizer_name={name!r} is not supported; only 'AdamW' is wired "
            f"up. This key used to be read by nothing, so setting it had no "
            f"effect — it now fails loudly instead."
        )
    return torch.optim.AdamW(
        params,
        lr           = mcfg.learning_rate,
        betas        = tuple(getattr(mcfg, "betas", (0.9, 0.999))),
        eps          = getattr(mcfg, "eps", 1e-8),
        weight_decay = getattr(mcfg, "weight_decay", 1e-2),
    )


class CheckpointManager:
    def __init__(self, checkpoint_dir: str, checkpoint_step: int = 1000):
        self.checkpoint_dir  = Path(checkpoint_dir)
        self.checkpoint_step = checkpoint_step
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def step(self, step: int, model: nn.Module, **extra) -> None:
        if step % self.checkpoint_step != 0:
            return
        path = self.checkpoint_dir / f"ckpt_step={step:07d}.pt"
        torch.save({"step": step, "model_state_dict": model.state_dict(), **extra}, path)
        logger.info("Checkpoint saved → %s", path)

    def load(self, path: str, model: nn.Module, map_location: str = "cpu") -> dict:
        payload = torch.load(path, map_location=map_location)
        model.load_state_dict(payload["model_state_dict"])
        logger.info("Checkpoint loaded ← %s", path)
        return payload


class EarlyStopper:
    """
    Counts validation checks without improvement.
    Patience is in units of *checks*, not steps.

    Parameters
    ----------
    patience : int   — number of checks allowed without improvement
    mode     : str   — "min" (lower is better) or "max"
    min_delta: float — minimum change to count as improvement
    """

    def __init__(self, patience: int, mode: str = "min", min_delta: float = 0.0):
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got '{mode}'.")
        self.patience   = patience
        self.mode       = mode
        self.min_delta  = min_delta
        self._best      = math.inf if mode == "min" else -math.inf
        self._counter   = 0

    @property
    def best(self) -> float:
        return self._best

    def step(self, metric: float) -> bool:
        """
        Call after each validation check.
        Returns True if training should stop.
        """
        improved = (
            metric < self._best - self.min_delta
            if self.mode == "min"
            else metric > self._best + self.min_delta
        )
        if improved:
            self._best    = metric
            self._counter = 0
        else:
            self._counter += 1

        return self._counter >= self.patience

    def state_dict(self) -> dict:
        return {"best": self._best, "counter": self._counter}

    def load_state_dict(self, d: dict):
        self._best    = d["best"]
        self._counter = d["counter"]
