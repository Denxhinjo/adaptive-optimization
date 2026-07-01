"""Integration tests for the Trainer, compare_optimizers, and
benchmark_runner orchestration layers."""
from __future__ import annotations

import numpy as np

from benchmarks import ROSENBROCK
from datasets.loaders import make_synthetic_binary
from experiments.benchmark_runner import run_all_optimizers_on_benchmark, run_optimizer_on_benchmark
from experiments.compare_optimizers import compare_optimizers
from experiments.trainer import Trainer
from models import LogisticRegression
from optimizers import Adam, SGD, LBFGS


def test_trainer_reduces_loss_over_epochs():
    ds = make_synthetic_binary(n_samples=400, n_features=8, seed=0)
    model = LogisticRegression(l2=1e-4)
    Xb = model._add_intercept(ds.X_train)
    trainer = Trainer(model=model, optimizer=Adam(lr=0.1), epochs=20, batch_size=32, seed=0)
    result = trainer.fit(Xb, ds.y_train)

    assert result.history.train_loss[0] > result.history.train_loss[-1]
    assert result.history.n_iterations == 20
    assert len(result.history.grad_norm) == 20


def test_trainer_records_positive_runtime():
    """Regression test: total_runtime_sec must be read after the timing
    `with` block exits, not before (Timer.elapsed is only set on __exit__)."""
    ds = make_synthetic_binary(n_samples=200, n_features=5, seed=5)
    model = LogisticRegression()
    Xb = model._add_intercept(ds.X_train)
    trainer = Trainer(model=model, optimizer=SGD(lr=0.1), epochs=5, batch_size=32)
    result = trainer.fit(Xb, ds.y_train)
    assert result.history.total_runtime_sec > 0.0


def test_trainer_records_validation_metrics_when_provided():
    ds = make_synthetic_binary(n_samples=300, n_features=6, seed=1)
    model = LogisticRegression()
    Xb = model._add_intercept(ds.X_train)
    Xb_val = model._add_intercept(ds.X_test)
    trainer = Trainer(model=model, optimizer=SGD(lr=0.5), epochs=10, batch_size=None)
    result = trainer.fit(Xb, ds.y_train, X_val=Xb_val, y_val=ds.y_test)

    assert len(result.history.val_loss) == 10
    assert len(result.history.val_acc) == 10
    assert all(0.0 <= acc <= 1.0 for acc in result.history.val_acc)


def test_trainer_supports_lbfgs_full_batch():
    ds = make_synthetic_binary(n_samples=200, n_features=5, seed=2)
    model = LogisticRegression(l2=1e-3)
    Xb = model._add_intercept(ds.X_train)
    trainer = Trainer(model=model, optimizer=LBFGS(lr=1.0), epochs=15, batch_size=32)  # batch_size ignored
    result = trainer.fit(Xb, ds.y_train)
    assert result.history.train_loss[-1] < result.history.train_loss[0]


def test_compare_optimizers_produces_one_row_per_optimizer():
    ds = make_synthetic_binary(n_samples=300, n_features=6, seed=3)
    optimizers = {"SGD": SGD(lr=0.3), "Adam": Adam(lr=0.1)}
    result = compare_optimizers(
        lambda: LogisticRegression(l2=1e-4),
        optimizers,
        ds.X_train,
        ds.y_train,
        ds.X_test,
        ds.y_test,
        epochs=10,
        batch_size=32,
    )
    assert set(result.table.index) == {"SGD", "Adam"}
    assert "test_accuracy" in result.table.columns
    assert "runtime_sec" in result.table.columns


def test_run_optimizer_on_benchmark_tracks_trajectory():
    run = run_optimizer_on_benchmark(ROSENBROCK, Adam(lr=0.05), n_iters=50)
    assert run.trajectory.shape[1] == 2
    assert len(run.losses) == run.trajectory.shape[0]
    # loss should generally decrease from the starting point
    assert run.losses[-1] < run.losses[0]


def test_run_all_optimizers_on_benchmark_returns_all_keys():
    optimizers = {"SGD": SGD(lr=0.001), "Adam": Adam(lr=0.05)}
    runs = run_all_optimizers_on_benchmark(ROSENBROCK, optimizers, n_iters=30)
    assert set(runs.keys()) == {"SGD", "Adam"}
    for run in runs.values():
        assert np.all(np.isfinite(run.trajectory))
