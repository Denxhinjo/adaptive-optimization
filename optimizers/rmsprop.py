r"""RMSProp (Tieleman & Hinton, 2012, unpublished -- Coursera lecture 6e).

Update rule
-----------
Replaces Adagrad's ever-growing sum of squared gradients with an
**exponential moving average**, so the effective step size no longer decays
to zero:

.. math::
    v_t &= \rho \, v_{t-1} + (1 - \rho) \, g_t \odot g_t \\
    w_{t+1} &= w_t - \eta \, \frac{g_t}{\sqrt{v_t} + \varepsilon}

:math:`\rho` (typically 0.9) controls the effective averaging window
(:math:`\approx 1/(1-\rho)` recent steps).

Advantages
    * Fixes Adagrad's vanishing-step-size problem -> works well on
      non-stationary / non-convex objectives (deep nets, RNNs).
    * Still fully per-coordinate adaptive.
Disadvantages
    * Introduces an extra hyperparameter (:math:`\rho`) to tune.
    * No bias correction for the EMA, so early steps (small :math:`t`)
      slightly under-estimate the true second moment.
"""
from __future__ import annotations

import numpy as np

from .base import Optimizer


class RMSProp(Optimizer):
    """Root Mean Square Propagation.

    Parameters
    ----------
    lr : float
        Base learning rate :math:`\\eta`.
    rho : float
        EMA decay rate :math:`\\rho \\in (0, 1)` for the squared-gradient
        accumulator.
    eps : float
        Numerical-stability constant :math:`\\varepsilon`.
    """

    name = "RMSProp"

    def __init__(self, lr: float = 1e-3, rho: float = 0.9, eps: float = 1e-8) -> None:
        super().__init__(lr)
        if not (0.0 < rho < 1.0):
            raise ValueError("rho must be in (0, 1)")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.rho = rho
        self.eps = eps
        self.v: np.ndarray | None = None

    def step(self, params: np.ndarray, grad: np.ndarray) -> np.ndarray:
        self.t += 1
        if self.v is None:
            self.v = np.zeros_like(params)
        self.v = self.rho * self.v + (1 - self.rho) * grad * grad
        return params - self.lr * grad / (np.sqrt(self.v) + self.eps)
