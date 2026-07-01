"""Metrics used to evaluate and compare optimizers: loss/accuracy curves,
gradient norms, wall-clock runtime, peak memory usage, convergence speed
and stability.
"""
from __future__ import annotations

import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np


def gradient_norm(grad: np.ndarray) -> float:
    """L2 norm of the (flattened) gradient."""
    return float(np.linalg.norm(np.asarray(grad).ravel()))


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


@contextmanager
def track_peak_memory_mb():
    """Context manager yielding a callable that returns peak memory (MB)
    allocated by Python objects during the ``with`` block, measured via
    :mod:`tracemalloc`. Lightweight, cross-platform, no extra dependency.
    """
    was_tracing = tracemalloc.is_tracing()
    if not was_tracing:
        tracemalloc.start()
    tracemalloc.reset_peak()

    result = {"peak_mb": 0.0}
    try:
        yield result
    finally:
        _, peak = tracemalloc.get_traced_memory()
        result["peak_mb"] = peak / (1024 ** 2)
        if not was_tracing:
            tracemalloc.stop()


@dataclass
class RunMetrics:
    """Full metric history + summary statistics for a single
    (optimizer, model, dataset) training run, as produced by
    :class:`experiments.trainer.Trainer`.
    """

    optimizer_name: str
    dataset_name: str = ""

    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    train_acc: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)
    grad_norm: list[float] = field(default_factory=list)
    epoch_times: list[float] = field(default_factory=list)

    total_runtime_sec: float = 0.0
    peak_memory_mb: float = 0.0
    n_iterations: int = 0

    def record_epoch(
        self,
        train_loss: float,
        grad_norm_value: float,
        epoch_time: float,
        val_loss: float | None = None,
        train_accuracy: float | None = None,
        val_accuracy: float | None = None,
    ) -> None:
        self.train_loss.append(train_loss)
        self.grad_norm.append(grad_norm_value)
        self.epoch_times.append(epoch_time)
        if val_loss is not None:
            self.val_loss.append(val_loss)
        if train_accuracy is not None:
            self.train_acc.append(train_accuracy)
        if val_accuracy is not None:
            self.val_acc.append(val_accuracy)
        self.n_iterations = len(self.train_loss)

    # ------------------------------------------------------------------ #
    # Derived statistics
    # ------------------------------------------------------------------ #
    def convergence_epoch(self, tol: float = 1e-3) -> int:
        """First epoch (1-indexed) at which the training loss gets within
        ``tol`` (relative) of the best loss observed across the whole run.
        Used as a simple, comparable "convergence speed" statistic.
        """
        if not self.train_loss:
            return -1
        best = min(self.train_loss)
        threshold = best + tol * abs(best) + tol
        for i, loss in enumerate(self.train_loss, start=1):
            if loss <= threshold:
                return i
        return len(self.train_loss)

    def stability(self, tail_fraction: float = 0.2) -> float:
        """Coefficient of variation of the loss over the final
        ``tail_fraction`` of training -- a proxy for how much the
        optimizer oscillates once it has (roughly) converged. Lower is
        more stable.
        """
        if len(self.train_loss) < 5:
            return 0.0
        n_tail = max(2, int(len(self.train_loss) * tail_fraction))
        tail = np.asarray(self.train_loss[-n_tail:])
        mean = np.mean(tail)
        if mean == 0:
            return float(np.std(tail))
        return float(np.std(tail) / abs(mean))

    def final(self, series: list[float]) -> float | None:
        return series[-1] if series else None

    def summary(self) -> dict:
        """One row of a cross-optimizer comparison table."""
        return {
            "optimizer": self.optimizer_name,
            "dataset": self.dataset_name,
            "final_train_loss": self.final(self.train_loss),
            "final_val_loss": self.final(self.val_loss),
            "final_train_acc": self.final(self.train_acc),
            "test_accuracy": self.final(self.val_acc),
            "final_grad_norm": self.final(self.grad_norm),
            "convergence_epoch": self.convergence_epoch(),
            "stability": self.stability(),
            "n_iterations": self.n_iterations,
            "runtime_sec": round(self.total_runtime_sec, 4),
            "peak_memory_mb": round(self.peak_memory_mb, 3),
        }


class Timer:
    """Minimal wall-clock stopwatch context manager."""

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        self.elapsed = 0.0
        return self

    def __exit__(self, *exc) -> None:
        self.elapsed = time.perf_counter() - self._start
