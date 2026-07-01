r"""AdaGrad-Norm (Ward, Wu & Bottou, 2020 -- "AdaGrad stepsizes: sharp
convergence over nonconvex landscapes").

Update rule
-----------
A **scalar** (rather than per-coordinate) variant of Adagrad: a single
accumulator tracks the running sum of squared *gradient norms*, and every
coordinate shares the same adaptive step size:

.. math::
    b_t^2 &= b_{t-1}^2 + \lVert g_t \rVert_2^2 \\
    w_{t+1} &= w_t - \eta \, \frac{g_t}{b_t + \varepsilon}

with :math:`b_0` a small positive initial value.

Advantages
    * Matches Adagrad's parameter-free convergence guarantees while using
      O(1) extra memory instead of O(d) -- attractive for very
      high-dimensional models.
    * Ward et al. prove it converges to a stationary point at rate
      :math:`O(\log T / \sqrt{T})` for smooth non-convex objectives,
      *without knowing the smoothness constant in advance*.
Disadvantages
    * Coordinates are coupled through one global scale, so it cannot adapt
      as finely to anisotropic curvature as coordinate-wise Adagrad/Adam.
"""
from __future__ import annotations

import numpy as np

from .base import Optimizer


class AdagradNorm(Optimizer):
    """Scalar adaptive step size driven by the global gradient norm.

    Parameters
    ----------
    lr : float
        Base learning rate :math:`\\eta`.
    eps : float
        Numerical-stability constant :math:`\\varepsilon`.
    b0 : float
        Initial value of the scalar accumulator :math:`b_0` (must be > 0
        to avoid an infinite first step).
    """

    name = "AdaGrad-Norm"

    def __init__(self, lr: float = 1.0, eps: float = 1e-8, b0: float = 1e-3) -> None:
        super().__init__(lr)
        if eps <= 0:
            raise ValueError("eps must be positive")
        if b0 <= 0:
            raise ValueError("b0 must be positive")
        self.eps = eps
        self.b0 = b0
        self.b2: float = b0 * b0

    def reset(self) -> None:
        super().reset()
        self.b2 = self.b0 * self.b0

    def step(self, params: np.ndarray, grad: np.ndarray) -> np.ndarray:
        self.t += 1
        self.b2 += float(np.sum(grad * grad))
        return params - self.lr * grad / (np.sqrt(self.b2) + self.eps)
