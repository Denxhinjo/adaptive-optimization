r"""Adam -- Adaptive Moment Estimation (Kingma & Ba, 2014).

Update rule
-----------
Combines momentum (a first-moment EMA of the gradient) with RMSProp-style
per-coordinate scaling (a second-moment EMA of the squared gradient), plus
bias correction for both moments:

.. math::
    m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
    v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t \odot g_t \\
    \hat m_t &= \frac{m_t}{1 - \beta_1^t}, \qquad
    \hat v_t = \frac{v_t}{1 - \beta_2^t} \\
    w_{t+1} &= w_t - \eta \, \frac{\hat m_t}{\sqrt{\hat v_t} + \varepsilon}

Bias correction matters most in early iterations: since :math:`m_0 = v_0 =
0`, the raw EMAs are biased toward zero, and dividing by :math:`1 -
\beta_i^t` compensates for that.

Advantages
    * Combines the benefits of momentum (smoothing, escaping shallow local
      structure) and adaptive per-coordinate scaling.
    * Robust default choice across a very wide range of ML problems with
      little tuning; the de-facto standard for deep learning.
Disadvantages
    * Can converge to a sharper / worse-generalizing minimum than
      well-tuned SGD in some settings.
    * :math:`v_t` (an EMA, not a running max) can occasionally shrink,
      leading to problematic increases in the effective step size --
      motivated the AMSGrad fix (not implemented here).
"""
from __future__ import annotations

import numpy as np

from .base import Optimizer


class Adam(Optimizer):
    """Adaptive Moment Estimation.

    Parameters
    ----------
    lr : float
        Base learning rate :math:`\\eta`.
    beta1 : float
        Decay rate for the first-moment (mean) estimate.
    beta2 : float
        Decay rate for the second-moment (uncentered variance) estimate.
    eps : float
        Numerical-stability constant :math:`\\varepsilon`.
    """

    name = "Adam"

    def __init__(
        self,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        super().__init__(lr)
        if not (0.0 <= beta1 < 1.0) or not (0.0 <= beta2 < 1.0):
            raise ValueError("beta1 and beta2 must be in [0, 1)")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m: np.ndarray | None = None
        self.v: np.ndarray | None = None

    def step(self, params: np.ndarray, grad: np.ndarray) -> np.ndarray:
        self.t += 1
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)

        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad * grad

        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        return params - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
