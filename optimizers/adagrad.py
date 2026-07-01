r"""Adagrad (Duchi, Hazan & Singer, 2011).

Update rule
-----------
Adagrad maintains a **per-coordinate** running sum of squared gradients and
uses it to give infrequently-updated coordinates larger effective step
sizes:

.. math::
    G_t &= G_{t-1} + g_t \odot g_t \\
    w_{t+1} &= w_t - \eta \, \frac{g_t}{\sqrt{G_t} + \varepsilon}

where :math:`\odot` is element-wise multiplication and the square
root/division are also element-wise.

Advantages
    * No manual per-parameter tuning: sparse/rare features automatically
      get larger updates, dense/frequent ones get smaller updates.
    * Strong theoretical guarantees for convex, sparse-gradient problems
      (e.g. NLP bag-of-words models).
Disadvantages
    * :math:`G_t` is monotonically non-decreasing, so the effective
      learning rate decays to zero and training can stall prematurely on
      non-convex/long-horizon problems.
"""
from __future__ import annotations

import numpy as np

from .base import Optimizer


class Adagrad(Optimizer):
    """Per-coordinate adaptive gradient descent.

    Parameters
    ----------
    lr : float
        Base learning rate :math:`\\eta`.
    eps : float
        Numerical-stability constant :math:`\\varepsilon`.
    """

    name = "Adagrad"

    def __init__(self, lr: float = 0.01, eps: float = 1e-8) -> None:
        super().__init__(lr)
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.eps = eps
        self.G: np.ndarray | None = None

    def step(self, params: np.ndarray, grad: np.ndarray) -> np.ndarray:
        self.t += 1
        if self.G is None:
            self.G = np.zeros_like(params)
        self.G += grad * grad
        return params - self.lr * grad / (np.sqrt(self.G) + self.eps)
