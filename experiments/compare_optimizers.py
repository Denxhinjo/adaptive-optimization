"""Run every optimizer over a fixed (model, dataset) pair and produce a
single comparison table + per-optimizer training histories.

This is the code path behind both the comparison notebook and the
Streamlit dashboard's "Compare optimizers" tab.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from models.base import BaseModel
from optimizers.base import Optimizer
from utils.logging_config import get_logger
from utils.metrics import RunMetrics

from .trainer import Trainer, TrainResult

logger = get_logger("compare_optimizers")


@dataclass
class ComparisonResult:
    table: pd.DataFrame
    histories: dict[str, RunMetrics] = field(default_factory=dict)
    results: dict[str, TrainResult] = field(default_factory=dict)


def compare_optimizers(
    model_factory,
    optimizers: dict[str, Optimizer],
    X_train,
    y_train,
    X_test=None,
    y_test=None,
    epochs: int = 100,
    batch_size: int | None = None,
    seed: int = 0,
    verbose: bool = False,
    track_memory: bool = True,
) -> ComparisonResult:
    """Train a fresh model instance with every optimizer in ``optimizers``
    on the same data/hyperparameters and collect a comparison table.

    Parameters
    ----------
    model_factory : callable
        Zero-arg callable returning a *fresh* :class:`~models.base.BaseModel`
        instance (e.g. ``lambda: LogisticRegression(l2=1e-4)``). A fresh
        model is built per-optimizer to avoid state leaking between runs.
    optimizers : dict[str, Optimizer]
        Mapping of display name -> optimizer instance.
    """
    rows = []
    histories: dict[str, RunMetrics] = {}
    results: dict[str, TrainResult] = {}

    for opt_name, optimizer in optimizers.items():
        logger.info("Training with optimizer: %s", opt_name)
        model: BaseModel = model_factory()
        Xb = model._add_intercept(X_train)
        Xb_test = model._add_intercept(X_test) if X_test is not None else None

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            epochs=epochs,
            batch_size=batch_size,
            seed=seed,
            verbose=verbose,
            track_memory=track_memory,
        )
        result = trainer.fit(Xb, y_train, X_val=Xb_test, y_val=y_test)
        result.history.optimizer_name = opt_name

        histories[opt_name] = result.history
        results[opt_name] = result
        rows.append(result.history.summary())

    table = pd.DataFrame(rows).set_index("optimizer")
    return ComparisonResult(table=table, histories=histories, results=results)
