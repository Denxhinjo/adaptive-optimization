"""Run an optimizer over a 2-D :class:`~benchmarks.functions.BenchmarkFunction`
and record its trajectory, loss curve and gradient-norm curve, for use by
:mod:`visualization.trajectory` (static contour + path plots and animated
GIFs) and the Streamlit benchmark playground.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from benchmarks.functions import BenchmarkFunction
from optimizers.base import Optimizer
from optimizers.lbfgs import LBFGS
from utils.metrics import gradient_norm


@dataclass
class BenchmarkRun:
    optimizer_name: str
    function_name: str
    trajectory: np.ndarray  # (T+1, 2)
    losses: list[float] = field(default_factory=list)
    grad_norms: list[float] = field(default_factory=list)

    @property
    def final_point(self) -> np.ndarray:
        return self.trajectory[-1]

    @property
    def distance_to_nearest_minimum(self) -> float:
        return float("nan")  # filled in by caller, who knows `minima`


def run_optimizer_on_benchmark(
    fn: BenchmarkFunction,
    optimizer: Optimizer,
    start: tuple[float, float] | None = None,
    n_iters: int = 200,
    tol: float = 1e-10,
) -> BenchmarkRun:
    """Run ``optimizer`` on ``fn`` starting from ``start`` (defaults to
    ``fn.default_start``) for up to ``n_iters`` steps, stopping early if the
    gradient norm drops below ``tol``.
    """
    optimizer.reset()
    point = np.array(start if start is not None else fn.default_start, dtype=float)

    trajectory = [point.copy()]
    losses = [fn.value_at(point)]
    grad_norms = []

    is_lbfgs = isinstance(optimizer, LBFGS)

    def closure(p: np.ndarray) -> tuple[float, np.ndarray]:
        return fn.value_at(p), fn.grad(p)

    for _ in range(n_iters):
        grad = fn.grad(point)
        g_norm = gradient_norm(grad)
        grad_norms.append(g_norm)
        if g_norm < tol or not np.isfinite(g_norm):
            break

        if is_lbfgs:
            point = optimizer.step(point, grad, closure=closure)
        else:
            point = optimizer.step(point, grad)

        if not np.all(np.isfinite(point)):
            break

        trajectory.append(point.copy())
        losses.append(fn.value_at(point))

    grad_norms.append(gradient_norm(fn.grad(point)))

    return BenchmarkRun(
        optimizer_name=getattr(optimizer, "name", type(optimizer).__name__),
        function_name=fn.name,
        trajectory=np.array(trajectory),
        losses=losses,
        grad_norms=grad_norms,
    )


def run_all_optimizers_on_benchmark(
    fn: BenchmarkFunction,
    optimizers: dict[str, Optimizer],
    start: tuple[float, float] | None = None,
    n_iters: int = 200,
) -> dict[str, BenchmarkRun]:
    return {
        name: run_optimizer_on_benchmark(fn, opt, start=start, n_iters=n_iters)
        for name, opt in optimizers.items()
    }
