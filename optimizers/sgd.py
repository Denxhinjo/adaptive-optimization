r"""Stochastic Gradient Descent (with optional momentum).

Update rule
-----------
At each step a mini-batch :math:`B_t` of size :math:`b` is sampled uniformly
from the dataset and the update uses the mini-batch gradient
:math:`g_t = \nabla f_{B_t}(w_t)` as an unbiased estimator of the true
gradient:

.. math::
    w_{t+1} = w_t - \eta \, g_t

With momentum:

.. math::
    v_{t+1} &= \mu v_t + g_t \\
    w_{t+1} &= w_t - \eta v_{t+1}

The mini-batch sampling itself lives in :mod:`experiments.trainer` (it is a
property of the training loop, not the optimizer); this class implements
only the parameter update given whatever stochastic gradient it is handed.

Advantages
    * O(batch size) cost per step -> scales to large datasets.
    * Injected gradient noise can help escape shallow local minima / saddle
      points.
Disadvantages
    * Noisy updates -> oscillation near the optimum; typically needs a
      decaying learning rate to converge.
    * Still uses one global step size, so it inherits GD's sensitivity to
      ill-conditioning.
"""
from __future__ import annotations

import numpy as np

from .base import Optimizer


class SGD(Optimizer):
    """Stochastic Gradient Descent with optional momentum and LR decay.

    Parameters
    ----------
    lr : float
        Base learning rate :math:`\\eta`.
    momentum : float, default 0.0
        Momentum coefficient :math:`\\mu \\in [0, 1)`.
    lr_decay : float, default 0.0
        If > 0, the effective learning rate at step ``t`` is
        ``lr / (1 + lr_decay * t)`` (a standard 1/t schedule).
    """

    name = "SGD"

    def __init__(self, lr: float = 0.01, momentum: float = 0.0, lr_decay: float = 0.0) -> None:
        super().__init__(lr)
        if not (0.0 <= momentum < 1.0):
            raise ValueError("momentum must be in [0, 1)")
        if lr_decay < 0.0:
            raise ValueError("lr_decay must be non-negative")
        self.momentum = momentum
        self.lr_decay = lr_decay
        self.velocity: np.ndarray | None = None

    def step(self, params: np.ndarray, grad: np.ndarray) -> np.ndarray:
        self.t += 1
        eff_lr = self.lr / (1.0 + self.lr_decay * self.t) if self.lr_decay else self.lr

        if self.momentum == 0.0:
            return params - eff_lr * grad

        if self.velocity is None:
            self.velocity = np.zeros_like(params)
        self.velocity = self.momentum * self.velocity + grad
        return params - eff_lr * self.velocity
