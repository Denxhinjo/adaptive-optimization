"""Unit tests for the 2-D benchmark functions: analytic gradients are
checked against central finite differences, and every known global
minimum is verified to have (near) zero value and gradient."""
from __future__ import annotations

import numpy as np
import pytest

from benchmarks import BEALE, HIMMELBLAU, RASTRIGIN, ROSENBROCK, BENCHMARK_REGISTRY


def finite_diff_grad(fn, point: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    grad = np.zeros(2)
    for i in range(2):
        p_plus, p_minus = point.copy(), point.copy()
        p_plus[i] += eps
        p_minus[i] -= eps
        grad[i] = (fn.value_at(p_plus) - fn.value_at(p_minus)) / (2 * eps)
    return grad


@pytest.mark.parametrize("fn", list(BENCHMARK_REGISTRY.values()), ids=list(BENCHMARK_REGISTRY))
def test_analytic_gradient_matches_finite_differences(fn):
    rng = np.random.default_rng(42)
    (x0, x1), (y0, y1) = fn.domain
    for _ in range(10):
        point = np.array([rng.uniform(x0, x1), rng.uniform(y0, y1)])
        analytic = fn.grad(point)
        numeric = finite_diff_grad(fn, point)
        np.testing.assert_allclose(analytic, numeric, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("fn", list(BENCHMARK_REGISTRY.values()), ids=list(BENCHMARK_REGISTRY))
def test_global_minima_have_zero_value_and_gradient(fn):
    for mx, my in fn.minima:
        point = np.array([mx, my])
        assert fn.value_at(point) == pytest.approx(0.0, abs=1e-6)
        grad = fn.grad(point)
        assert np.linalg.norm(grad) == pytest.approx(0.0, abs=1e-4)


def test_rosenbrock_known_values():
    assert ROSENBROCK.value_at(np.array([1.0, 1.0])) == pytest.approx(0.0)
    assert ROSENBROCK.value_at(np.array([0.0, 0.0])) == pytest.approx(1.0)


def test_get_benchmark_lookup():
    from benchmarks import get_benchmark

    assert get_benchmark("Rosenbrock") is ROSENBROCK
    with pytest.raises(ValueError):
        get_benchmark("NotAFunction")


def test_himmelblau_has_four_minima():
    assert len(HIMMELBLAU.minima) == 4


def test_beale_and_rastrigin_registered():
    assert BEALE.name in BENCHMARK_REGISTRY
    assert RASTRIGIN.name in BENCHMARK_REGISTRY
