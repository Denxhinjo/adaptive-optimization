"""Abstract base class for the linear models in this package.

Both :class:`~models.logistic_regression.LogisticRegression` and
:class:`~models.softmax_regression.SoftmaxRegression` are *stateless with
respect to the optimization loop*: ``loss(weights, X, y)`` and
``grad(weights, X, y)`` are pure functions of an explicit parameter array,
which is exactly what every optimizer in :mod:`optimizers` expects. This
also lets the exact same loss/grad pair be reused unmodified for animated
benchmark-style visualizations and for :mod:`experiments.trainer`.

``fit()`` is a thin sklearn-flavoured convenience wrapper around
:class:`experiments.trainer.Trainer` for users who just want
``model.fit(X, y).predict(X_test)`` without touching the lower-level API.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import scipy.sparse as sp

from optimizers.base import Optimizer

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Shared functionality for linear classification models.

    Parameters
    ----------
    l2 : float, default 0.0
        L2 (ridge) regularization strength. The bias/intercept term is
        excluded from the penalty, following standard practice.
    fit_intercept : bool, default True
        Whether to prepend a constant-1 column to the design matrix.
    """

    name: str = "BaseModel"

    def __init__(self, l2: float = 0.0, fit_intercept: bool = True) -> None:
        if l2 < 0:
            raise ValueError("l2 must be non-negative")
        self.l2 = l2
        self.fit_intercept = fit_intercept
        self.weights: np.ndarray | None = None
        self.history: dict[str, Any] | None = None

    # ------------------------------------------------------------------ #
    # Methods every subclass must implement
    # ------------------------------------------------------------------ #
    @abstractmethod
    def _init_weights(self, n_features: int) -> np.ndarray: ...

    @abstractmethod
    def loss(self, weights: np.ndarray, X, y: np.ndarray) -> float: ...

    @abstractmethod
    def grad(self, weights: np.ndarray, X, y: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def predict_proba(self, weights: np.ndarray, X) -> np.ndarray: ...

    @abstractmethod
    def predict(self, weights: np.ndarray, X) -> np.ndarray: ...

    # ------------------------------------------------------------------ #
    # Shared utilities
    # ------------------------------------------------------------------ #
    def _add_intercept(self, X):
        if not self.fit_intercept:
            return X
        n = X.shape[0]
        if sp.issparse(X):
            ones = sp.csr_matrix(np.ones((n, 1)))
            return sp.hstack([ones, X], format="csr")
        return np.hstack([np.ones((n, 1)), X])

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
            g[0] = 0.0
        return g

    def loss_and_grad(self, weights: np.ndarray, X, y: np.ndarray) -> tuple[float, np.ndarray]:
        return self.loss(weights, X, y), self.grad(weights, X, y)

    def _normalize_labels(self, y: np.ndarray) -> np.ndarray:
        """Map labels into the convention used internally by this model.
        Overridden by :class:`~models.logistic_regression.LogisticRegression`
        to accept both ``{0, 1}`` and ``{-1, +1}`` labels."""
        return y

    def score(self, X, y: np.ndarray) -> float:
        """Mean accuracy on ``(X, y)``."""
        if self.weights is None:
            raise RuntimeError("model is not fitted yet")
        X = self._add_intercept(X)
        preds = self.predict(self.weights, X)
        return float(np.mean(preds == self._normalize_labels(y)))

    def fit(
        self,
        X,
        y: np.ndarray,
        optimizer: Optimizer,
        epochs: int = 100,
        batch_size: int | None = None,
        X_val=None,
        y_val: np.ndarray | None = None,
        shuffle: bool = True,
        seed: int = 0,
        verbose: bool = False,
        track_memory: bool = False,
    ) -> "BaseModel":
        """Fit the model to ``(X, y)`` using ``optimizer``.

        A thin convenience wrapper around :class:`experiments.trainer.Trainer`
        that mutates ``self.weights``/``self.history`` in place and returns
        ``self`` so calls can be chained, e.g. ``model.fit(...).predict(...)``.
        """
        from experiments.trainer import Trainer  # local import breaks a cycle

        Xb = self._add_intercept(X)
        Xb_val = self._add_intercept(X_val) if X_val is not None else None

        trainer = Trainer(
            model=self,
            optimizer=optimizer,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            verbose=verbose,
            track_memory=track_memory,
        )
        result = trainer.fit(Xb, y, X_val=Xb_val, y_val=y_val)
        self.weights = result.final_params
        self.history = result.history
        return self

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.name}(l2={self.l2}, fit_intercept={self.fit_intercept})"
