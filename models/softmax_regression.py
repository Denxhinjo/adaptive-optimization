r"""Multi-class softmax (multinomial logistic) regression with L2
regularization.

Model
-----
For :math:`K` classes and weight matrix :math:`W \in \mathbb{R}^{d \times K}`:

.. math::
    p(y=k \mid x; W) = \frac{e^{(Wx)_k}}{\sum_{j=1}^K e^{(Wx)_j}}
    \quad \text{(softmax)}

Loss (categorical cross-entropy, averaged over the batch, plus L2 penalty)
----------------------------------------------------------------------------
.. math::
    \mathcal L(W) = -\frac{1}{n} \sum_{i=1}^n \log p(y_i \mid x_i; W)
    + \frac{\lambda}{2} \lVert W \rVert_F^2

Computed in a numerically stable way via the log-sum-exp trick:
:math:`\log \sum_j e^{z_j} = m + \log \sum_j e^{z_j - m}` with
:math:`m = \max_j z_j`.

Gradient
--------
.. math::
    \nabla_W \mathcal L = \frac{1}{n} X^\top (P - Y_{\text{onehot}}) + \lambda W

where :math:`P \in \mathbb{R}^{n \times K}` stacks the softmax
probabilities for every sample.

Parameters here are a **matrix** ``(n_features, n_classes)`` rather than a
vector; every optimizer in :mod:`optimizers` operates element-wise so this
requires no special-casing anywhere else in the library.
"""
from __future__ import annotations

import numpy as np

from .base import BaseModel


class SoftmaxRegression(BaseModel):
    """Multinomial (softmax) logistic regression for multi-class
    classification, trained by an arbitrary first-order
    :class:`~optimizers.base.Optimizer` from this package."""

    name = "Softmax Regression"

    def __init__(self, n_classes: int, l2: float = 0.0, fit_intercept: bool = True) -> None:
        super().__init__(l2=l2, fit_intercept=fit_intercept)
        if n_classes < 2:
            raise ValueError("n_classes must be >= 2")
        self.n_classes = n_classes

    def _init_weights(self, n_features: int) -> np.ndarray:
        return np.zeros((n_features, self.n_classes))

    @staticmethod
    def _softmax(scores: np.ndarray) -> np.ndarray:
        scores = scores - scores.max(axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        return exp_scores / exp_scores.sum(axis=1, keepdims=True)

    def loss(self, weights: np.ndarray, X, y: np.ndarray) -> float:
        y = np.asarray(y).astype(int)
        scores = np.asarray(X @ weights)
        shifted = scores - scores.max(axis=1, keepdims=True)
        log_z = np.log(np.exp(shifted).sum(axis=1))
        correct = shifted[np.arange(len(y)), y]
        ce = float(np.mean(log_z - correct))
        return ce + self._l2_penalty(weights)

    def grad(self, weights: np.ndarray, X, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y).astype(int)
        n = X.shape[0]
        scores = np.asarray(X @ weights)
        P = self._softmax(scores)
        P[np.arange(n), y] -= 1.0
        g = np.asarray(X.T @ P) / n
        return g + self._l2_grad(weights)

    def predict_proba(self, weights: np.ndarray, X) -> np.ndarray:
        scores = np.asarray(X @ weights)
        return self._softmax(scores)

    def predict(self, weights: np.ndarray, X) -> np.ndarray:
        return np.argmax(self.predict_proba(weights, X), axis=1)

    def _l2_penalty(self, weights: np.ndarray) -> float:
        if self.l2 == 0.0:
            return 0.0
        w = weights[1:] if self.fit_intercept else weights
        return 0.5 * self.l2 * float(np.sum(w * w))

    def _l2_grad(self, weights: np.ndarray) -> np.ndarray:
        if self.l2 == 0.0:
            return np.zeros_like(weights)
        g = self.l2 * weights
        if self.fit_intercept:
            g = g.copy()
            g[0, :] = 0.0
        return g
