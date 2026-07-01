r"""Classic 2-D non-convex benchmark functions used to visualize and
stress-test optimizer trajectories.

Each function is wrapped in a :class:`BenchmarkFunction` exposing:

* ``f(x, y)``     -- value, vectorized over NumPy arrays (for contour plots)
* ``grad(point)`` -- analytic gradient at a single ``(2,)`` point (for
  optimizer trajectories)
* ``domain``      -- a sensible ``(xlim, ylim)`` viewing window
* ``minima``      -- known global minimizer(s)
* ``default_start``  -- a challenging starting point (near a saddle/ridge)

All formulas and analytic gradients below are hand-derived; see
``tests/test_benchmarks.py`` for gradient-checks against central finite
differences.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class BenchmarkFunction:
    name: str
    f: Callable[[np.ndarray, np.ndarray], np.ndarray]
    grad: Callable[[np.ndarray], np.ndarray]
    domain: tuple[tuple[float, float], tuple[float, float]]
    minima: list[tuple[float, float]]
    default_start: tuple[float, float]
    description: str = ""

    def value_at(self, point: np.ndarray) -> float:
        x, y = point
        return float(self.f(np.asarray(x), np.asarray(y)))


# --------------------------------------------------------------------- #
# Rosenbrock: f(x, y) = (a - x)^2 + b (y - x^2)^2
# --------------------------------------------------------------------- #
def _rosenbrock_f(x, y, a=1.0, b=100.0):
    return (a - x) ** 2 + b * (y - x ** 2) ** 2


def _rosenbrock_grad(point, a=1.0, b=100.0):
    x, y = point
    dx = -2 * (a - x) - 4 * b * x * (y - x ** 2)
    dy = 2 * b * (y - x ** 2)
    return np.array([dx, dy])


ROSENBROCK = BenchmarkFunction(
    name="Rosenbrock",
    f=_rosenbrock_f,
    grad=_rosenbrock_grad,
    domain=((-2.0, 2.0), (-1.0, 3.0)),
    minima=[(1.0, 1.0)],
    default_start=(-1.5, 2.5),
    description="Narrow curved valley ('banana function'); a classic "
    "ill-conditioning stress test for fixed-step first-order methods.",
)


# --------------------------------------------------------------------- #
# Himmelblau: f(x, y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2
# --------------------------------------------------------------------- #
def _himmelblau_f(x, y):
    return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2


def _himmelblau_grad(point):
    x, y = point
    t1 = x ** 2 + y - 11
    t2 = x + y ** 2 - 7
    dx = 4 * x * t1 + 2 * t2
    dy = 2 * t1 + 4 * y * t2
    return np.array([dx, dy])


HIMMELBLAU = BenchmarkFunction(
    name="Himmelblau",
    f=_himmelblau_f,
    grad=_himmelblau_grad,
    domain=((-5.0, 5.0), (-5.0, 5.0)),
    minima=[
        (3.0, 2.0),
        (-2.805118, 3.131312),
        (-3.779310, -3.283186),
        (3.584428, -1.848126),
    ],
    default_start=(-4.0, 4.0),
    description="Four symmetric global minima; illustrates how different "
    "optimizers/starting points converge to different basins.",
)


# --------------------------------------------------------------------- #
# Beale: f(x, y) = (1.5 - x + xy)^2 + (2.25 - x + xy^2)^2 + (2.625 - x + xy^3)^2
# --------------------------------------------------------------------- #
def _beale_f(x, y):
    t1 = 1.5 - x + x * y
    t2 = 2.25 - x + x * y ** 2
    t3 = 2.625 - x + x * y ** 3
    return t1 ** 2 + t2 ** 2 + t3 ** 2


def _beale_grad(point):
    x, y = point
    t1 = 1.5 - x + x * y
    t2 = 2.25 - x + x * y ** 2
    t3 = 2.625 - x + x * y ** 3
    dx = 2 * t1 * (y - 1) + 2 * t2 * (y ** 2 - 1) + 2 * t3 * (y ** 3 - 1)
    dy = 2 * t1 * x + 2 * t2 * (2 * x * y) + 2 * t3 * (3 * x * y ** 2)
    return np.array([dx, dy])


BEALE = BenchmarkFunction(
    name="Beale",
    f=_beale_f,
    grad=_beale_grad,
    domain=((-4.5, 4.5), (-4.5, 4.5)),
    minima=[(3.0, 0.5)],
    default_start=(-3.5, -4.0),
    description="Sharp, steep-walled corners around a flat plateau; "
    "punishes overly aggressive fixed step sizes.",
)


# --------------------------------------------------------------------- #
# Rastrigin: f(x) = A*n + sum(x_i^2 - A*cos(2*pi*x_i)), A = 10, n = 2
# --------------------------------------------------------------------- #
def _rastrigin_f(x, y, A=10.0):
    return (
        2 * A
        + (x ** 2 - A * np.cos(2 * np.pi * x))
        + (y ** 2 - A * np.cos(2 * np.pi * y))
    )


def _rastrigin_grad(point, A=10.0):
    x, y = point
    dx = 2 * x + 2 * np.pi * A * np.sin(2 * np.pi * x)
    dy = 2 * y + 2 * np.pi * A * np.sin(2 * np.pi * y)
    return np.array([dx, dy])


RASTRIGIN = BenchmarkFunction(
    name="Rastrigin",
    f=_rastrigin_f,
    grad=_rastrigin_grad,
    domain=((-5.12, 5.12), (-5.12, 5.12)),
    minima=[(0.0, 0.0)],
    default_start=(-3.2, 4.0),
    description="Highly multi-modal with many regularly-spaced local "
    "minima; probes an optimizer's tendency to get trapped locally.",
)


BENCHMARK_REGISTRY: dict[str, BenchmarkFunction] = {
    fn.name: fn for fn in (ROSENBROCK, HIMMELBLAU, BEALE, RASTRIGIN)
}


def get_benchmark(name: str) -> BenchmarkFunction:
    try:
        return BENCHMARK_REGISTRY[name]
    except KeyError as exc:
        valid = ", ".join(BENCHMARK_REGISTRY)
        raise ValueError(f"Unknown benchmark '{name}'. Valid options: {valid}") from exc
