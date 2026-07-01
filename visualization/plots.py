"""Static comparison plots: loss/accuracy/gradient-norm curves, comparison
bar charts, and confusion matrices. Every function returns a
``matplotlib.figure.Figure`` so callers can either ``fig.savefig(...)`` or
hand it to ``st.pyplot(fig)`` in the Streamlit dashboard.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.metrics import RunMetrics

from .style import apply_style, color_for

apply_style()


def _plot_series(
    histories: dict[str, RunMetrics],
    attr: str,
    ylabel: str,
    title: str,
    log_scale: bool = True,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, (name, hist) in enumerate(histories.items()):
        series = getattr(hist, attr)
        if not series:
            continue
        ax.plot(
            range(1, len(series) + 1),
            series,
            label=name,
            color=color_for(name, i),
            linewidth=2,
        )
    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    return fig


def plot_loss_curves(
    histories: dict[str, RunMetrics], which: str = "train_loss", log_scale: bool = True
) -> plt.Figure:
    title = "Training Loss" if which == "train_loss" else "Validation Loss"
    return _plot_series(histories, which, "Loss", title, log_scale=log_scale)


def plot_accuracy_curves(histories: dict[str, RunMetrics], which: str = "val_acc") -> plt.Figure:
    title = "Validation Accuracy" if which == "val_acc" else "Training Accuracy"
    return _plot_series(histories, which, "Accuracy", title, log_scale=False)


def plot_gradient_norms(histories: dict[str, RunMetrics]) -> plt.Figure:
    return _plot_series(histories, "grad_norm", "||grad||_2", "Gradient Norm", log_scale=True)


def plot_comparison_bars(table: pd.DataFrame, metric: str, ylabel: str | None = None) -> plt.Figure:
    """Bar chart comparing a single summary metric (e.g. ``runtime_sec``,
    ``test_accuracy``, ``convergence_epoch``) across optimizers."""
    fig, ax = plt.subplots(figsize=(7, 4))
    names = table.index.tolist()
    values = table[metric].values
    colors = [color_for(n, i) for i, n in enumerate(names)]
    bars = ax.bar(names, values, color=colors)
    ax.set_ylabel(ylabel or metric)
    ax.set_title(metric.replace("_", " ").title())
    ax.bar_label(bars, fmt="%.3g", padding=3, fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str] | None = None, title: str = "Confusion Matrix"
) -> plt.Figure:
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    labels = class_names or [str(i) for i in range(cm.shape[0])]

    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=8,
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def plot_runtime_vs_loss(histories: dict[str, RunMetrics]) -> plt.Figure:
    """Loss vs. cumulative wall-clock time -- a fairer comparison than
    loss-vs-epoch when optimizers have very different per-step cost
    (e.g. L-BFGS's line search vs. plain SGD)."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, (name, hist) in enumerate(histories.items()):
        if not hist.train_loss:
            continue
        cum_time = np.cumsum(hist.epoch_times)
        ax.plot(cum_time, hist.train_loss, label=name, color=color_for(name, i), linewidth=2)
    ax.set_yscale("log")
    ax.set_xlabel("Wall-clock time (s)")
    ax.set_ylabel("Training loss")
    ax.set_title("Loss vs. Wall-Clock Time")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    return fig
