"""Generic training loop shared by every model/optimizer/dataset
combination in this project.

:class:`Trainer` is intentionally decoupled from any specific model: it
only requires an object exposing ``_init_weights``, ``loss``, ``grad``,
``predict`` and ``_normalize_labels`` (i.e. the
:class:`~models.base.BaseModel` interface) and an
:class:`~optimizers.base.Optimizer`. This is what lets
:mod:`experiments.compare_optimizers` run *every* optimizer over *every*
model/dataset combination with one code path, and is also what
:meth:`models.base.BaseModel.fit` delegates to internally.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import numpy as np

from optimizers.base import Optimizer
from optimizers.lbfgs import LBFGS
from utils.logging_config import get_logger
from utils.metrics import RunMetrics, Timer, accuracy, gradient_norm, track_peak_memory_mb

logger = get_logger("trainer")


@dataclass
class TrainResult:
    final_params: np.ndarray
    history: RunMetrics


class Trainer:
    """Runs a mini-batch (or full-batch) training loop for one
    ``(model, optimizer)`` pair and records :class:`~utils.metrics.RunMetrics`.

    Parameters
    ----------
    model : object implementing the :class:`~models.base.BaseModel` interface.
    optimizer : Optimizer
        Any optimizer from :mod:`optimizers`. If an :class:`~optimizers.lbfgs.LBFGS`
        instance is passed, ``batch_size`` is ignored and full-batch updates
        with an Armijo line search are used (L-BFGS is a deterministic
        quasi-Newton method and is not designed for stochastic gradients).
    epochs : int
        Number of passes over the training data.
    batch_size : int or None
        Mini-batch size. ``None`` means full-batch (one "epoch" = one step).
    """

    def __init__(
        self,
        model: Any,
        optimizer: Optimizer,
        epochs: int = 100,
        batch_size: int | None = None,
        shuffle: bool = True,
        seed: int = 0,
        verbose: bool = False,
        track_memory: bool = False,
        log_every: int = 10,
        on_epoch_end=None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.verbose = verbose
        self.track_memory = track_memory
        self.log_every = log_every
        self.on_epoch_end = on_epoch_end

    def fit(self, X, y: np.ndarray, X_val=None, y_val: np.ndarray | None = None) -> TrainResult:
        model = self.model
        optimizer = self.optimizer
        optimizer.reset()

        rng = np.random.default_rng(self.seed)
        n_samples = X.shape[0]
        params = model._init_weights(X.shape[1])

        is_lbfgs = isinstance(optimizer, LBFGS)
        batch_size = n_samples if (self.batch_size is None or is_lbfgs) else min(self.batch_size, n_samples)

        def closure(p: np.ndarray) -> tuple[float, np.ndarray]:
            return model.loss_and_grad(p, X, y)

        history = RunMetrics(optimizer_name=getattr(optimizer, "name", type(optimizer).__name__))

        mem = {"peak_mb": 0.0}
        mem_ctx = track_peak_memory_mb() if self.track_memory else nullcontext(mem)
        with mem_ctx as mem, Timer() as total_timer:
            for epoch in range(1, self.epochs + 1):
                with Timer() as epoch_timer:
                    order = rng.permutation(n_samples) if self.shuffle else np.arange(n_samples)

                    for start in range(0, n_samples, batch_size):
                        idx = order[start : start + batch_size]
                        Xb, yb = X[idx], y[idx]

                        grad = model.grad(params, Xb, yb)
                        if is_lbfgs:
                            params = optimizer.step(params, grad, closure=closure)
                        else:
                            params = optimizer.step(params, grad)

                train_loss = model.loss(params, X, y)
                full_grad = model.grad(params, X, y)
                grad_norm_value = gradient_norm(full_grad)
                train_preds = model.predict(params, X)
                train_accuracy = accuracy(model._normalize_labels(y), train_preds)

                val_loss = val_accuracy = None
                if X_val is not None and y_val is not None:
                    val_loss = model.loss(params, X_val, y_val)
                    val_preds = model.predict(params, X_val)
                    val_accuracy = accuracy(model._normalize_labels(y_val), val_preds)

                history.record_epoch(
                    train_loss=train_loss,
                    grad_norm_value=grad_norm_value,
                    epoch_time=epoch_timer.elapsed,
                    val_loss=val_loss,
                    train_accuracy=train_accuracy,
                    val_accuracy=val_accuracy,
                )

                if self.on_epoch_end is not None:
                    self.on_epoch_end(epoch, params, history)

                if self.verbose and (epoch % self.log_every == 0 or epoch == self.epochs):
                    msg = f"[{history.optimizer_name}] epoch {epoch}/{self.epochs} loss={train_loss:.5f}"
                    if val_loss is not None:
                        msg += f" val_loss={val_loss:.5f} val_acc={val_accuracy:.4f}"
                    logger.info(msg)

        history.total_runtime_sec = total_timer.elapsed
        history.peak_memory_mb = mem["peak_mb"]

        return TrainResult(final_params=params, history=history)
