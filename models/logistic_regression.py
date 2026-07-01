r"""Binary logistic regression with L2 regularization.

Model
-----
.. math::
    p(y=1 \mid x; w) = \sigma(w^\top x) = \frac{1}{1 + e^{-w^\top x}}

Loss (binary cross-entropy, averaged over the batch, plus L2 penalty)
-----------------------------------------------------------------------
.. math::
    \mathcal L(w) = -\frac{1}{n} \sum_{i=1}^n \Big[
        y_i \log \sigma(w^\top x_i) + (1-y_i) \log\big(1 - \sigma(w^\top x_i)\big)
    \Big] + \frac{\lambda}{2} \lVert w \rVert_2^2

Gradient
--------
.. math::
    \nabla \mathcal L(w) = \frac{1}{n} X^\top (\sigma(Xw) - y) + \lambda w

Labels are accepted as either ``{0, 1}`` or ``{-1, +1}`` and are
internally normalized to ``{0, 1}`` (the convention used throughout this
module); :meth:`predict` returns labels in the same ``{0, 1}`` space.
"""
from __future__ import annotations

import numpy as np

from .base import BaseModel


def _sigmoid(z: np.ndarray) -> np.ndarray:
    # Numerically stable logistic sigmoid.
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[~pos])
    out[~pos] = exp_z / (1.0 + exp_z)
    return out


def _to_zero_one(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    if set(np.unique(y).tolist()) <= {-1, 1}:
        return (y > 0).astype(float)
    return y.astype(float)


class LogisticRegression(BaseModel):
    """Binary logistic regression trained by an arbitrary first-order
    :class:`~optimizers.base.Optimizer` from this package."""

    name = "Logistic Regression"

    def _init_weights(self, n_features: int) -> np.ndarray:
        return np.zeros(n_features)

    def _normalize_labels(self, y: np.ndarray) -> np.ndarray:
        return _to_zero_one(y).astype(int)

    def loss(self, weights: np.ndarray, X, y: np.ndarray) -> float:
        y = _to_zero_one(y)
        z = X @ weights
        # log(1+exp(-|z|)) form of the stable log-sum-exp for BCE-with-logits
        bce = np.mean(np.logaddexp(0.0, -z) + (1.0 - y) * z)
        return float(bce) + self._l2_penalty(weights)

    def grad(self, weights: np.ndarray, X, y: np.ndarray) -> np.ndarray:
        y = _to_zero_one(y)
        n = X.shape[0]
        p = _sigmoid(np.asarray(X @ weights).ravel())
        residual = p - y
        g = (X.T @ residual) / n
        g = np.asarray(g).ravel()
        return g + self._l2_grad(weights)

    def predict_proba(self, weights: np.ndarray, X) -> np.ndarray:
        z = np.asarray(X @ weights).ravel()
        return _sigmoid(z)

    def predict(self, weights: np.ndarray, X, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(weights, X) >= threshold).astype(int)
