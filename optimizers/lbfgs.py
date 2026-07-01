r"""Limited-memory BFGS (L-BFGS), implemented from scratch with the
two-loop recursion (Nocedal, 1980) and an Armijo backtracking line search.

Unlike the first-order methods in this package, L-BFGS is a **quasi-Newton**
method: it builds a low-rank approximation of the inverse Hessian from the
last ``m`` gradient/parameter differences and uses it to precondition the
gradient before taking a step.

Update rule
-----------
Given the last :math:`m` curvature pairs :math:`(s_i, y_i)` with
:math:`s_i = w_{i+1} - w_i` and :math:`y_i = g_{i+1} - g_i`, the two-loop
recursion computes the search direction :math:`d_t = -H_t g_t` (an implicit
application of the inverse-Hessian approximation :math:`H_t`) without ever
forming the :math:`d \times d` matrix explicitly:

.. math::
    q &\leftarrow g_t \\
    \text{for } i = t-1, \dots, t-m:\quad
        \alpha_i &= \rho_i \, s_i^\top q, \quad q \leftarrow q - \alpha_i y_i,
        \quad \rho_i = 1 / (y_i^\top s_i) \\
    r &\leftarrow \left(\frac{s_{t-1}^\top y_{t-1}}{y_{t-1}^\top y_{t-1}}\right) q \\
    \text{for } i = t-m, \dots, t-1:\quad
        \beta &= \rho_i \, y_i^\top r, \quad r \leftarrow r + s_i(\alpha_i - \beta) \\
    d_t &= -r

A step size :math:`\eta_t` is then chosen by Armijo backtracking line search
along :math:`d_t` and :math:`w_{t+1} = w_t + \eta_t d_t`.

Advantages
    * Superlinear local convergence -- typically an order of magnitude
      fewer iterations than first-order methods on smooth, well-behaved
      (e.g. strongly convex) objectives.
    * Only O(m*d) memory, versus O(d^2) for full BFGS.
Disadvantages
    * Requires the *full* (batch) gradient and a line search that
      re-evaluates the objective -- expensive per iteration and not
      naturally stochastic/mini-batch friendly.
    * Line search plus curvature-pair bookkeeping makes it a poor fit for
      noisy objectives (mini-batch SGD-style training).

This module also exposes :class:`LBFGSScipyReference`, a thin wrapper
around ``scipy.optimize.minimize(method="L-BFGS-B")`` used purely as a
correctness baseline for the from-scratch implementation above.
"""
from __future__ import annotations

from collections import deque
from typing import Callable

import numpy as np

from .base import Optimizer

LossGradFn = Callable[[np.ndarray], tuple[float, np.ndarray]]


class LBFGS(Optimizer):
    """From-scratch limited-memory BFGS with two-loop recursion.

    Parameters
    ----------
    lr : float
        Initial step size handed to the line search (only used for the
        very first iteration and as an upper bound thereafter).
    memory : int
        Number of curvature pairs :math:`(s_i, y_i)` to retain, ``m``.
    c1 : float
        Armijo sufficient-decrease constant.
    max_ls_iters : int
        Maximum backtracking line-search steps per outer iteration.
    """

    name = "L-BFGS"

    def __init__(
        self,
        lr: float = 1.0,
        memory: int = 10,
        c1: float = 1e-4,
        max_ls_iters: int = 20,
    ) -> None:
        super().__init__(lr)
        if memory < 1:
            raise ValueError("memory must be >= 1")
        self.memory = memory
        self.c1 = c1
        self.max_ls_iters = max_ls_iters
        self._s: deque[np.ndarray] = deque(maxlen=memory)
        self._y: deque[np.ndarray] = deque(maxlen=memory)
        self._prev_params: np.ndarray | None = None
        self._prev_grad: np.ndarray | None = None

    def reset(self) -> None:
        super().reset()
        self._s.clear()
        self._y.clear()
        self._prev_params = None
        self._prev_grad = None

    def _two_loop_direction(self, grad: np.ndarray) -> np.ndarray:
        q = grad.copy()
        alphas = []
        rhos = []
        for s, y in zip(reversed(self._s), reversed(self._y), strict=True):
            rho = 1.0 / (y.ravel() @ s.ravel() + 1e-12)
            alpha = rho * (s.ravel() @ q.ravel())
            q = q - alpha * y
            alphas.append(alpha)
            rhos.append(rho)

        if self._s:
            s_last, y_last = self._s[-1], self._y[-1]
            gamma = (s_last.ravel() @ y_last.ravel()) / (
                y_last.ravel() @ y_last.ravel() + 1e-12
            )
        else:
            gamma = 1.0
        r = gamma * q

        for (s, y), alpha, rho in zip(
            zip(self._s, self._y, strict=True), reversed(alphas), reversed(rhos), strict=True
        ):
            beta = rho * (y.ravel() @ r.ravel())
            r = r + s * (alpha - beta)

        return -r

    def step(
        self,
        params: np.ndarray,
        grad: np.ndarray,
        closure: LossGradFn | None = None,
    ) -> np.ndarray:
        """Take one L-BFGS step.

        Parameters
        ----------
        params, grad : current point and gradient at ``params``.
        closure : callable ``params -> (loss, grad)`` used for the Armijo
            line search. If omitted, falls back to a fixed step of size
            ``lr`` along the quasi-Newton direction (no guarantee of
            sufficient decrease -- only recommended for benchmark
            functions where the caller supplies a closure).
        """
        self.t += 1

        if self._prev_params is not None:
            s = params - self._prev_params
            y = grad - self._prev_grad
            if y.ravel() @ s.ravel() > 1e-10:  # curvature condition
                self._s.append(s)
                self._y.append(y)

        direction = self._two_loop_direction(grad)

        step_size = self.lr
        if closure is not None:
            loss0, _ = closure(params)
            directional_deriv = grad.ravel() @ direction.ravel()
            for _ in range(self.max_ls_iters):
                candidate = params + step_size * direction
                loss_new, _ = closure(candidate)
                if loss_new <= loss0 + self.c1 * step_size * directional_deriv:
                    break
                step_size *= 0.5

        new_params = params + step_size * direction

        self._prev_params = params.copy()
        self._prev_grad = grad.copy()
        return new_params


class LBFGSScipyReference:
    """Baseline wrapper around ``scipy.optimize.minimize(method="L-BFGS-B")``.

    Not part of the :class:`~optimizers.base.Optimizer` hierarchy (SciPy
    owns its own iteration loop internally) -- used only to sanity-check
    that :class:`LBFGS` above converges to the same optimum.
    """

    name = "L-BFGS (SciPy reference)"

    def __init__(self, max_iter: int = 200, tol: float = 1e-8) -> None:
        self.max_iter = max_iter
        self.tol = tol

    def minimize(
        self,
        loss_and_grad: LossGradFn,
        x0: np.ndarray,
    ):
        from scipy.optimize import minimize

        shape = x0.shape

        def fun(flat_x):
            loss, grad = loss_and_grad(flat_x.reshape(shape))
            return float(loss), np.asarray(grad).ravel()

        result = minimize(
            fun,
            x0.ravel(),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": self.max_iter, "ftol": self.tol},
        )
        result.x = result.x.reshape(shape)
        return result
