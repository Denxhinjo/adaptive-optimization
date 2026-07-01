r"""Vanilla (full-batch) Gradient Descent.

Update rule
-----------
.. math::
    w_{t+1} = w_t - \eta \, \nabla f(w_t)

where :math:`\eta` is a fixed (or externally scheduled) learning rate and
:math:`\nabla f(w_t)` is the *full-batch* gradient, i.e. computed over the
entire dataset (or, for benchmark functions, the exact analytic gradient).

Optionally supports classical (heavy-ball) momentum:

.. math::
    v_{t+1} &= \mu v_t + \nabla f(w_t) \\
    w_{t+1} &= w_t - \eta v_{t+1}

Advantages
    * Simple, deterministic, easy to reason about and to prove convergence for.
    * Exact descent direction -> monotone loss decrease for small enough
      :math:`\eta` on convex, L-smooth objectives.
Disadvantages
    * Requires a full pass over the data per step -> expensive for large
      datasets.
    * A single global learning rate struggles with ill-conditioned
      (elongated) loss landscapes; convergence is only linear.
"""
from __future__ import annotations

import numpy as np

from .base import Optimizer


class GradientDescent(Optimizer):
    """Full-batch gradient descent, optionally with heavy-ball momentum.

    Parameters
    ----------
    lr : float
        Learning rate :math:`\\eta`.
    momentum : float, default 0.0
        Momentum coefficient :math:`\\mu \\in [0, 1)`. ``0.0`` recovers
        vanilla gradient descent.
    """

    name = "Gradient Descent"

    def __init__(self, lr: float = 0.1, momentum: float = 0.0) -> None:
        super().__init__(lr)
        if not (0.0 <= momentum < 1.0):
            raise ValueError("momentum must be in [0, 1)")
        self.momentum = momentum
        self.velocity: np.ndarray | None = None

    def step(self, params: np.ndarray, grad: np.ndarray) -> np.ndarray:
        self.t += 1
        if self.momentum == 0.0:
            return params - self.lr * grad

        if self.velocity is None:
            self.velocity = np.zeros_like(params)
        self.velocity = self.momentum * self.velocity + grad
        return params - self.lr * self.velocity
