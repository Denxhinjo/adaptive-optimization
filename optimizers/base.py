"""Abstract base class shared by every optimizer in this library.

Design notes
------------
Every optimizer implements a single method, :meth:`Optimizer.step`, that
maps ``(params, grad) -> new_params``. All per-coordinate/scalar state
(moment estimates, accumulators, iteration counters, ...) is owned by the
optimizer instance, not by the caller. This mirrors the interface used by
PyTorch/JAX optimizers while remaining a ~20-line contract implemented in
pure NumPy.

``params`` may be an array of any shape (a vector for logistic regression,
a ``(d, K)`` matrix for softmax regression, or a 2-D point for benchmark
functions) -- every update rule here is defined element-wise, so shape is
irrelevant to the math.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


class Optimizer(ABC):
    """Base class for all first-order optimizers.

    Parameters
    ----------
    lr : float
        Base learning rate (step size). Must be strictly positive.
    """

    #: Human-readable name used in plots, tables and the Streamlit UI.
    name: str = "Optimizer"

    def __init__(self, lr: float = 1e-3) -> None:
        if lr <= 0:
            raise ValueError(f"learning rate must be positive, got {lr}")
        self.lr = float(lr)
        self.t: int = 0

    @abstractmethod
    def step(self, params: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """Apply one update step and return the new parameters.

        Implementations must increment ``self.t`` and are free to lazily
        initialize any internal state on the first call using
        ``np.zeros_like(params)``.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Clear all internal state (moment estimates, counters, ...).

        Called before every fresh training run so the same optimizer
        instance can be reused across experiments without leaking state.
        """
        self.t = 0
        for key, value in list(self.__dict__.items()):
            if key in ("lr", "t"):
                continue
            if isinstance(value, np.ndarray) or value is None:
                self.__dict__[key] = None

    def get_config(self) -> dict[str, Any]:
        """Return hyperparameters (excluding runtime state) for logging."""
        return {
            k: v
            for k, v in self.__dict__.items()
            if not isinstance(v, np.ndarray) and k != "t"
        }

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        cfg = ", ".join(f"{k}={v}" for k, v in self.get_config().items())
        return f"{self.name}({cfg})"


@dataclass
class StepResult:
    """Lightweight record of a single optimization step, used by trainers
    and the benchmark-function animators to reconstruct trajectories."""

    iteration: int
    params: np.ndarray
    loss: float
    grad_norm: float
    extra: dict[str, Any] = field(default_factory=dict)
