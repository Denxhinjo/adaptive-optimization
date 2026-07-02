"""Smoke tests for the visualization module: every plotting function
should run headlessly and return a matplotlib Figure without raising."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from benchmarks import ROSENBROCK
from experiments.benchmark_runner import run_all_optimizers_on_benchmark
from optimizers import SGD, Adam
from utils.metrics import RunMetrics
from visualization.plots import (
    plot_accuracy_curves,
    plot_comparison_bars,
    plot_confusion_matrix,
    plot_gradient_norms,
    plot_loss_curves,
    plot_runtime_vs_loss,
)
from visualization.trajectory import plot_contour, plot_contour_with_trajectories, plot_convergence_curves


def _dummy_history(name: str) -> RunMetrics:
    hist = RunMetrics(optimizer_name=name)
    for i in range(10):
        hist.record_epoch(
            train_loss=1.0 / (i + 1),
            grad_norm_value=0.5 / (i + 1),
            epoch_time=0.01,
            val_loss=1.1 / (i + 1),
            train_accuracy=0.5 + i * 0.02,
            val_accuracy=0.5 + i * 0.02,
        )
    return hist


def test_plot_loss_and_accuracy_curves():
    histories = {"SGD": _dummy_history("SGD"), "Adam": _dummy_history("Adam")}
    for fig in (
        plot_loss_curves(histories),
        plot_accuracy_curves(histories),
        plot_gradient_norms(histories),
        plot_runtime_vs_loss(histories),
    ):
        assert fig is not None
        plt.close(fig)


def test_plot_comparison_bars():
    import pandas as pd

    table = pd.DataFrame(
        {"runtime_sec": [0.1, 0.2], "test_accuracy": [0.9, 0.95]}, index=["SGD", "Adam"]
    )
    table.index.name = "optimizer"
    fig = plot_comparison_bars(table, "runtime_sec")
    assert fig is not None
    plt.close(fig)


def test_plot_confusion_matrix():
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    fig = plot_confusion_matrix(y_true, y_pred, class_names=["neg", "pos"])
    assert fig is not None
    plt.close(fig)


def test_plot_contour_and_trajectories():
    fig1 = plot_contour(ROSENBROCK)
    plt.close(fig1)

    runs = run_all_optimizers_on_benchmark(
        ROSENBROCK, {"SGD": SGD(lr=0.001), "Adam": Adam(lr=0.05)}, n_iters=30
    )
    fig2 = plot_contour_with_trajectories(ROSENBROCK, runs)
    fig3 = plot_convergence_curves(runs)
    plt.close(fig2)
    plt.close(fig3)


def test_overlapping_trajectories_stay_visually_distinguishable():
    """Regression test: two optimizers that take nearly the same route
    (e.g. a well-tuned Adam vs. L-BFGS on Rosenbrock) must not be plotted
    identically -- same color *and* linestyle would let one fully hide the
    other. Every trajectory line must have a distinct (color, linestyle) pair.
    """
    from optimizers import LBFGS

    runs = run_all_optimizers_on_benchmark(
        ROSENBROCK, {"Adam": Adam(lr=0.05), "L-BFGS": LBFGS(lr=1.0)}, n_iters=50
    )
    fig = plot_contour_with_trajectories(ROSENBROCK, runs)
    ax = fig.axes[0]

    # First two Line2D artists on the axes are the trajectory lines (one per
    # optimizer, added before the start/end markers).
    trajectory_lines = [ln for ln in ax.get_lines() if ln.get_label() in runs][:2]
    styles = {(ln.get_color(), ln.get_linestyle()) for ln in trajectory_lines}
    assert len(styles) == len(trajectory_lines), "overlapping trajectories must differ in color or linestyle"
    plt.close(fig)
