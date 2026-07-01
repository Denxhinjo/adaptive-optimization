"""Unit tests for utils.metrics: RunMetrics summary statistics and the
memory/timing helpers."""
from __future__ import annotations

import time

import numpy as np

from utils.metrics import RunMetrics, Timer, accuracy, gradient_norm, track_peak_memory_mb


def test_gradient_norm():
    assert gradient_norm(np.array([3.0, 4.0])) == 5.0


def test_accuracy():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    assert accuracy(y_true, y_pred) == 0.75


def test_run_metrics_convergence_epoch():
    hist = RunMetrics(optimizer_name="test")
    for loss in [10.0, 5.0, 1.0, 1.0001, 1.00005, 1.00001]:
        hist.record_epoch(train_loss=loss, grad_norm_value=0.1, epoch_time=0.01)
    epoch = hist.convergence_epoch(tol=1e-3)
    assert epoch <= 6
    assert epoch >= 3


def test_run_metrics_stability_lower_for_flatter_tail():
    stable = RunMetrics(optimizer_name="stable")
    noisy = RunMetrics(optimizer_name="noisy")
    for i in range(20):
        stable.record_epoch(train_loss=1.0 + 1e-6 * i, grad_norm_value=0.1, epoch_time=0.01)
        noisy.record_epoch(train_loss=1.0 + 0.5 * ((-1) ** i), grad_norm_value=0.1, epoch_time=0.01)
    assert stable.stability() < noisy.stability()


def test_run_metrics_summary_keys():
    hist = RunMetrics(optimizer_name="opt", dataset_name="ds")
    hist.record_epoch(
        train_loss=1.0, grad_norm_value=0.5, epoch_time=0.01,
        val_loss=1.1, train_accuracy=0.8, val_accuracy=0.75,
    )
    summary = hist.summary()
    for key in (
        "optimizer", "dataset", "final_train_loss", "final_val_loss",
        "final_train_acc", "test_accuracy", "final_grad_norm",
        "convergence_epoch", "stability", "n_iterations", "runtime_sec", "peak_memory_mb",
    ):
        assert key in summary


def test_timer_measures_elapsed_time():
    with Timer() as t:
        time.sleep(0.05)
    assert t.elapsed >= 0.04


def test_track_peak_memory_mb_reports_nonnegative():
    with track_peak_memory_mb() as mem:
        _ = [np.zeros(1000) for _ in range(100)]
    assert mem["peak_mb"] >= 0.0
