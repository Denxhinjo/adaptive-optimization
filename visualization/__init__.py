"""Matplotlib visualizations: training curves, comparison charts, and
animated benchmark-function trajectories."""
from .plots import (
    plot_accuracy_curves,
    plot_comparison_bars,
    plot_confusion_matrix,
    plot_gradient_norms,
    plot_loss_curves,
    plot_runtime_vs_loss,
)
from .trajectory import (
    animate_trajectories,
    plot_contour,
    plot_contour_with_trajectories,
    plot_convergence_curves,
)

__all__ = [
    "plot_loss_curves",
    "plot_accuracy_curves",
    "plot_gradient_norms",
    "plot_comparison_bars",
    "plot_confusion_matrix",
    "plot_runtime_vs_loss",
    "plot_contour",
    "plot_contour_with_trajectories",
    "plot_convergence_curves",
    "animate_trajectories",
]
