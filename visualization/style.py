"""Shared color palette so every plot (matplotlib, Streamlit, notebooks)
uses the same color for a given optimizer."""
from __future__ import annotations

OPTIMIZER_COLORS: dict[str, str] = {
    "Gradient Descent": "#8E8E93",
    "SGD": "#FF9F0A",
    "Adagrad": "#34C759",
    "AdaGrad-Norm": "#30B0C7",
    "RMSProp": "#5E5CE6",
    "Adam": "#FF375F",
    "L-BFGS": "#0A84FF",
    "L-BFGS (SciPy reference)": "#000000",
}

DEFAULT_CYCLE = list(OPTIMIZER_COLORS.values())


def color_for(name: str, fallback_index: int = 0) -> str:
    if name in OPTIMIZER_COLORS:
        return OPTIMIZER_COLORS[name]
    return DEFAULT_CYCLE[fallback_index % len(DEFAULT_CYCLE)]


# On problems like Rosenbrock, a well-tuned adaptive method's path can nearly
# coincide with L-BFGS's -- same color trajectories drawn solid-on-solid can
# fully occlude one another. Distinct dash patterns keep every trajectory
# visible even when two optimizers take almost the same route.
LINESTYLE_CYCLE = ["-", "--", "-.", (0, (1, 1)), (0, (3, 1, 1, 1))]


def linestyle_for(fallback_index: int) -> str | tuple:
    return LINESTYLE_CYCLE[fallback_index % len(LINESTYLE_CYCLE)]


PLOT_STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.grid": True,
    "grid.color": "#E5E5EA",
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "legend.frameon": False,
}


def apply_style() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(PLOT_STYLE)
