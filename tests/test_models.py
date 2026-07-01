"""Unit tests for LogisticRegression and SoftmaxRegression: gradient
correctness (via finite differences), training convergence, and the
label-convention handling ({0,1} vs {-1,+1})."""
from __future__ import annotations

import numpy as np
import pytest

from datasets.loaders import make_synthetic_binary, make_synthetic_multiclass
from models import LogisticRegression, SoftmaxRegression
from optimizers import Adam


def numerical_grad(f, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    grad = np.zeros_like(x, dtype=float)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        orig = x[idx]
        x[idx] = orig + eps
        f_plus = f(x)
        x[idx] = orig - eps
        f_minus = f(x)
        x[idx] = orig
        grad[idx] = (f_plus - f_minus) / (2 * eps)
    return grad


def test_logistic_regression_gradient_matches_finite_differences():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 5))
    y = rng.integers(0, 2, size=30)
    model = LogisticRegression(l2=0.1)
    w = rng.normal(size=5) * 0.1

    analytic = model.grad(w, X, y)
    numeric = numerical_grad(lambda w_: model.loss(w_, X, y), w.copy())

    np.testing.assert_allclose(analytic, numeric, atol=1e-4)


def test_softmax_regression_gradient_matches_finite_differences():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(25, 4))
    y = rng.integers(0, 3, size=25)
    model = SoftmaxRegression(n_classes=3, l2=0.05)
    W = rng.normal(size=(4, 3)) * 0.1

    analytic = model.grad(W, X, y)
    numeric = numerical_grad(lambda W_: model.loss(W_, X, y), W.copy())

    np.testing.assert_allclose(analytic, numeric, atol=1e-4)


@pytest.mark.parametrize("labels", [np.array([0, 1, 0, 1]), np.array([-1, 1, -1, 1])])
def test_logistic_regression_accepts_both_label_conventions(labels):
    model = LogisticRegression()
    normalized = model._normalize_labels(labels)
    assert set(normalized.tolist()) == {0, 1}
    np.testing.assert_array_equal(normalized, np.array([0, 1, 0, 1]))


def test_logistic_regression_fits_synthetic_data():
    ds = make_synthetic_binary(n_samples=600, n_features=10, seed=0)
    model = LogisticRegression(l2=1e-3)
    model.fit(ds.X_train, ds.y_train, Adam(lr=0.1), epochs=60, batch_size=64, seed=0)
    acc = model.score(ds.X_test, ds.y_test)
    assert acc > 0.75


def test_softmax_regression_fits_synthetic_data():
    ds = make_synthetic_multiclass(n_samples=900, n_features=10, n_classes=3, seed=0)
    model = SoftmaxRegression(n_classes=3, l2=1e-3)
    model.fit(ds.X_train, ds.y_train, Adam(lr=0.1), epochs=60, batch_size=64, seed=0)
    acc = model.score(ds.X_test, ds.y_test)
    assert acc > 0.6


def test_score_raises_if_not_fitted():
    model = LogisticRegression()
    with pytest.raises(RuntimeError):
        model.score(np.zeros((2, 2)), np.array([0, 1]))
