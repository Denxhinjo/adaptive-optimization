"""Contour plots and animated GIFs of optimizer trajectories on the 2-D
benchmark functions in :mod:`benchmarks`.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

from benchmarks.functions import BenchmarkFunction
from experiments.benchmark_runner import BenchmarkRun

from .style import apply_style, color_for, linestyle_for

apply_style()


def _meshgrid(fn: BenchmarkFunction, n: int = 300):
    (x0, x1), (y0, y1) = fn.domain
    X, Y = np.meshgrid(np.linspace(x0, x1, n), np.linspace(y0, y1, n))
    Z = fn.f(X, Y)
    return X, Y, Z


def plot_contour(fn: BenchmarkFunction, ax: plt.Axes | None = None, levels: int = 40) -> plt.Figure:
    """Static contour/heatmap of ``fn`` with its global minima marked."""
    X, Y, Z = _meshgrid(fn)
    fig = ax.figure if ax is not None else None
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))

    z_plot = np.log1p(Z - Z.min())  # log-scale contours: huge dynamic range near minima
    ax.contourf(X, Y, z_plot, levels=levels, cmap="viridis", alpha=0.9)
    ax.contour(X, Y, z_plot, levels=levels, colors="black", linewidths=0.25, alpha=0.4)

    for mx, my in fn.minima:
        ax.plot(mx, my, marker="*", markersize=16, color="white", markeredgecolor="black", zorder=5)

    ax.set_xlim(*fn.domain[0])
    ax.set_ylim(*fn.domain[1])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"{fn.name} function")
    return fig


def plot_contour_with_trajectories(
    fn: BenchmarkFunction, runs: dict[str, BenchmarkRun], levels: int = 40
) -> plt.Figure:
    """Overlay every optimizer's trajectory on a shared contour plot."""
    fig, ax = plt.subplots(figsize=(8, 7))
    plot_contour(fn, ax=ax, levels=levels)

    # Different optimizers can take nearly the same route (e.g. a well-tuned
    # Adam vs. L-BFGS on Rosenbrock) -- solid-on-solid would let the
    # later-drawn line fully hide the earlier one. Vary linestyle + alpha so
    # every trajectory stays visible even when paths coincide.
    for i, (name, run) in enumerate(runs.items()):
        traj = run.trajectory
        c = color_for(name, i)
        ls = linestyle_for(i)
        ax.plot(
            traj[:, 0], traj[:, 1], color=c, linewidth=2.2, linestyle=ls, alpha=0.85,
            marker="o", markersize=2.5, label=name,
        )
        ax.plot(traj[0, 0], traj[0, 1], marker="s", color=c, markersize=8, markeredgecolor="black")
        ax.plot(traj[-1, 0], traj[-1, 1], marker="X", color=c, markersize=10, markeredgecolor="black")

    # Overlaying trajectories triggers matplotlib autoscale, which can shrink
    # the view to the trajectory's extent -- pin it back to the full domain.
    ax.set_xlim(*fn.domain[0])
    ax.set_ylim(*fn.domain[1])

    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"Optimizer trajectories on {fn.name}")
    fig.tight_layout()
    return fig


def plot_convergence_curves(runs: dict[str, BenchmarkRun]) -> plt.Figure:
    """Loss vs. iteration for each optimizer on a benchmark function."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, (name, run) in enumerate(runs.items()):
        losses = np.asarray(run.losses)
        ax.plot(
            losses - min(losses.min(), 0) + 1e-12, label=name,
            color=color_for(name, i), linestyle=linestyle_for(i), linewidth=2, alpha=0.85,
        )
    ax.set_yscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("f(x, y) (shifted, log scale)")
    ax.set_title("Convergence Curves")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    return fig


def animate_trajectories(
    fn: BenchmarkFunction,
    runs: dict[str, BenchmarkRun],
    levels: int = 40,
    interval_ms: int = 60,
    save_path: str | None = None,
) -> animation.FuncAnimation:
    """Animate every optimizer's trajectory converging on ``fn`` in
    lock-step. Returns the :class:`~matplotlib.animation.FuncAnimation`;
    if ``save_path`` is given (``.gif`` or ``.mp4``) the animation is also
    written to disk.
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    plot_contour(fn, ax=ax, levels=levels)
    ax.set_title(f"Optimizer trajectories on {fn.name}")
    ax.set_xlim(*fn.domain[0])
    ax.set_ylim(*fn.domain[1])

    max_len = max(len(run.trajectory) for run in runs.values())

    lines, points = {}, {}
    for i, name in enumerate(runs):
        c = color_for(name, i)
        (line,) = ax.plot([], [], color=c, linewidth=2.2, linestyle=linestyle_for(i), alpha=0.85, label=name)
        (point,) = ax.plot([], [], marker="o", color=c, markersize=7, markeredgecolor="black")
        lines[name] = line
        points[name] = point
    ax.legend(loc="upper right", fontsize=8)

    def update(frame):
        artists = []
        for name, run in runs.items():
            traj = run.trajectory
            idx = min(frame, len(traj) - 1)
            lines[name].set_data(traj[: idx + 1, 0], traj[: idx + 1, 1])
            points[name].set_data([traj[idx, 0]], [traj[idx, 1]])
            artists += [lines[name], points[name]]
        return artists

    ani = animation.FuncAnimation(fig, update, frames=max_len, interval=interval_ms, blit=True)

    if save_path is not None:
        writer = animation.PillowWriter(fps=max(1, 1000 // interval_ms))
        ani.save(save_path, writer=writer)

    return ani
