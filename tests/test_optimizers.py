"""Unit tests for every optimizer: convex convergence, monotonic-ish
decrease behaviour, and state-reset correctness."""
from __future__ import annotations

import numpy as np
import pytest

from optimizers import (
    LBFGS,
    Adagrad,
    AdagradNorm,
    Adam,
    GradientDescent,
    RMSProp,
    SGD,
    build_optimizer,
)
from optimizers.base import Optimizer


def quadratic_loss_grad(w: np.ndarray):
    """f(w) = 0.5 * ||w - target||^2, a simple strongly-convex bowl."""
    target = np.array([3.0, -2.0, 1.5])
    return 0.5 * float(np.sum((w - target) ** 2)), (w - target)


TARGET = np.array([3.0, -2.0, 1.5])

OPTIMIZER_FACTORIES = {
    "GradientDescent": lambda: GradientDescent(lr=0.5),
    "SGD": lambda: SGD(lr=0.5),
    "SGD-momentum": lambda: SGD(lr=0.1, momentum=0.9),
    "Adagrad": lambda: Adagrad(lr=1.0),
    "AdagradNorm": lambda: AdagradNorm(lr=2.0),
    # RMSProp's EMA-based accumulator (unlike Adagrad's monotonic sum) keeps a
    # nonzero floor step size near the optimum, so a small lr is needed for
    # tight convergence within a fixed iteration budget -- see rmsprop.py docstring.
    "RMSProp": lambda: RMSProp(lr=0.01),
    "Adam": lambda: Adam(lr=0.5),
    "LBFGS": lambda: LBFGS(lr=1.0),
}


@pytest.mark.parametrize("name", OPTIMIZER_FACTORIES)
def test_converges_on_convex_quadratic(name):
    optimizer: Optimizer = OPTIMIZER_FACTORIES[name]()
    w = np.zeros(3)
    is_lbfgs = isinstance(optimizer, LBFGS)

    for _ in range(500):
        loss, grad = quadratic_loss_grad(w)
        if is_lbfgs:
            w = optimizer.step(w, grad, closure=quadratic_loss_grad)
        else:
            w = optimizer.step(w, grad)

    assert np.linalg.norm(w - TARGET) < 1e-2, f"{name} failed to converge: w={w}"


@pytest.mark.parametrize("name", OPTIMIZER_FACTORIES)
def test_reset_clears_state(name):
    optimizer = OPTIMIZER_FACTORIES[name]()
    w = np.zeros(3)
    for _ in range(10):
        _, grad = quadratic_loss_grad(w)
        w = optimizer.step(w, grad) if not isinstance(optimizer, LBFGS) else optimizer.step(
            w, grad, closure=quadratic_loss_grad
        )
    assert optimizer.t == 10

    optimizer.reset()
    assert optimizer.t == 0
    # every ndarray-valued state should be cleared back to None
    for key, value in optimizer.__dict__.items():
        if key in ("lr", "t"):
            continue
        assert not isinstance(value, np.ndarray) or value is None

    if isinstance(optimizer, AdagradNorm):
        assert optimizer.b2 == pytest.approx(optimizer.b0 ** 2)
    if isinstance(optimizer, LBFGS):
        assert len(optimizer._s) == 0
        assert len(optimizer._y) == 0
        assert optimizer._prev_params is None


def test_lbfgs_converges_faster_than_gd_on_ill_conditioned_quadratic():
    """Sanity check that L-BFGS's curvature information helps on a
    poorly-conditioned quadratic where plain GD struggles."""
    A = np.diag([1.0, 100.0])
    target = np.array([1.0, 1.0])

    def loss_grad(w):
        diff = w - target
        return float(0.5 * diff @ A @ diff), A @ diff

    def run(optimizer, is_lbfgs, n_iters):
        w = np.zeros(2)
        for _ in range(n_iters):
            loss, grad = loss_grad(w)
            w = optimizer.step(w, grad, closure=loss_grad) if is_lbfgs else optimizer.step(w, grad)
        return np.linalg.norm(w - target)

    gd_error = run(GradientDescent(lr=0.0198), False, 50)
    lbfgs_error = run(LBFGS(lr=1.0), True, 50)

    assert lbfgs_error < gd_error


@pytest.mark.parametrize("bad_lr", [0.0, -1.0])
def test_negative_or_zero_lr_rejected(bad_lr):
    with pytest.raises(ValueError):
        SGD(lr=bad_lr)


def test_build_optimizer_registry():
    opt = build_optimizer("Adam", lr=0.01)
    assert isinstance(opt, Adam)
    assert opt.lr == 0.01

    with pytest.raises(ValueError):
        build_optimizer("NotAnOptimizer")


def test_optimizers_work_on_matrix_shaped_params():
    """Softmax regression uses (d, K) matrices -- every optimizer must
    handle arbitrary-shape params via element-wise NumPy ops."""
    target = np.array([[1.0, -1.0], [0.5, 2.0]])

    def loss_grad(W):
        return float(0.5 * np.sum((W - target) ** 2)), (W - target)

    for name, factory in OPTIMIZER_FACTORIES.items():
        optimizer = factory()
        W = np.zeros((2, 2))
        is_lbfgs = isinstance(optimizer, LBFGS)
        for _ in range(300):
            _, grad = loss_grad(W)
            W = optimizer.step(W, grad, closure=loss_grad) if is_lbfgs else optimizer.step(W, grad)
        assert np.linalg.norm(W - target) < 1e-1, f"{name} failed on matrix params"
