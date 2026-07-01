"""Generate every figure/table used in the README: benchmark-function
trajectories, and optimizer comparisons on a9a (logistic regression) and
MNIST (softmax regression).

Run once from the repo root:
    python -m experiments.generate_results

Writes tracked, README-ready assets to ``assets/`` and raw CSV dumps to
``results/`` (gitignored).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import BENCHMARK_REGISTRY  # noqa: E402
from datasets.loaders import load_a9a, load_mnist  # noqa: E402
from experiments.benchmark_runner import run_all_optimizers_on_benchmark  # noqa: E402
from experiments.compare_optimizers import ComparisonResult, compare_optimizers  # noqa: E402
from models import LogisticRegression, SoftmaxRegression  # noqa: E402
from optimizers import build_optimizer  # noqa: E402
from utils.logging_config import get_logger  # noqa: E402
from visualization.plots import (  # noqa: E402
    plot_accuracy_curves,
    plot_comparison_bars,
    plot_confusion_matrix,
    plot_gradient_norms,
    plot_loss_curves,
    plot_runtime_vs_loss,
)
from visualization.trajectory import (  # noqa: E402
    animate_trajectories,
    plot_contour_with_trajectories,
    plot_convergence_curves,
)

logger = get_logger("generate_results")

ASSETS = ROOT / "assets"
RESULTS = ROOT / "results"
ASSETS.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

BENCHMARK_LR = {
    "Gradient Descent": {"Rosenbrock": 0.0009, "Himmelblau": 0.001, "Beale": 0.0008, "Rastrigin": 0.01},
    "SGD": {"Rosenbrock": 0.0009, "Himmelblau": 0.001, "Beale": 0.0008, "Rastrigin": 0.01},
    "Adagrad": {"Rosenbrock": 0.5, "Himmelblau": 0.5, "Beale": 0.3, "Rastrigin": 0.5},
    "AdaGrad-Norm": {"Rosenbrock": 1.0, "Himmelblau": 1.0, "Beale": 0.5, "Rastrigin": 1.0},
    "RMSProp": {"Rosenbrock": 0.01, "Himmelblau": 0.05, "Beale": 0.01, "Rastrigin": 0.05},
    "Adam": {"Rosenbrock": 0.05, "Himmelblau": 0.2, "Beale": 0.02, "Rastrigin": 0.1},
    "L-BFGS": {"Rosenbrock": 1.0, "Himmelblau": 1.0, "Beale": 1.0, "Rastrigin": 1.0},
}
BENCHMARK_OPTIMIZERS = ["SGD", "Adagrad", "RMSProp", "Adam", "L-BFGS"]


def savefig(fig, name: str, dpi: int = 130) -> None:
    path = ASSETS / name
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", path)


def compare_with_full_batch_gd(model_factory, optimizer_lrs, X_train, y_train, X_test, y_test, epochs, batch_size):
    """Like ``compare_optimizers``, but runs "Gradient Descent" as true
    full-batch (batch_size=None) while every other optimizer uses
    ``batch_size`` -- otherwise "Gradient Descent" would just be SGD without
    momentum on the same mini-batches, which defeats the point of showing
    full-batch vs. mini-batch behavior side by side.
    """
    gd_lr = optimizer_lrs["Gradient Descent"]
    mini_batch_lrs = {k: v for k, v in optimizer_lrs.items() if k != "Gradient Descent"}

    gd_result = compare_optimizers(
        model_factory, {"Gradient Descent": build_optimizer("Gradient Descent", lr=gd_lr)},
        X_train, y_train, X_test, y_test, epochs=epochs, batch_size=None, track_memory=True,
    )
    rest_result = compare_optimizers(
        model_factory, {name: build_optimizer(name, lr=lr) for name, lr in mini_batch_lrs.items()},
        X_train, y_train, X_test, y_test, epochs=epochs, batch_size=batch_size, track_memory=True,
    )

    table = pd.concat([gd_result.table, rest_result.table]).loc[list(optimizer_lrs)]
    histories = {**gd_result.histories, **rest_result.histories}
    results = {**gd_result.results, **rest_result.results}
    return ComparisonResult(table=table, histories=histories, results=results)


def generate_benchmark_figures() -> None:
    for fn_name, fn in BENCHMARK_REGISTRY.items():
        optimizers = {
            name: build_optimizer(name, lr=BENCHMARK_LR[name][fn_name]) for name in BENCHMARK_OPTIMIZERS
        }
        runs = run_all_optimizers_on_benchmark(fn, optimizers, n_iters=400)

        fig1 = plot_contour_with_trajectories(fn, runs)
        savefig(fig1, f"benchmark_{fn_name.lower()}_trajectories.png")

        fig2 = plot_convergence_curves(runs)
        savefig(fig2, f"benchmark_{fn_name.lower()}_convergence.png")

        gif_path = ASSETS / f"benchmark_{fn_name.lower()}_animation.gif"
        animate_trajectories(fn, runs, save_path=str(gif_path))
        logger.info("wrote %s", gif_path)


def generate_a9a_results() -> None:
    ds = load_a9a()
    optimizer_lrs = {
        "Gradient Descent": 0.5,
        "SGD": 0.05,
        "Adagrad": 0.5,
        "AdaGrad-Norm": 1.0,
        "RMSProp": 0.01,
        "Adam": 0.01,
        "L-BFGS": 1.0,
    }
    t0 = time.time()
    result = compare_with_full_batch_gd(
        lambda: LogisticRegression(l2=1e-4),
        optimizer_lrs,
        ds.X_train, ds.y_train, ds.X_test, ds.y_test,
        epochs=60, batch_size=256,
    )
    logger.info("a9a comparison finished in %.1fs", time.time() - t0)

    result.table.to_csv(RESULTS / "a9a_comparison.csv")
    with open(ASSETS / "a9a_comparison_table.md", "w") as f:
        f.write(result.table.round(5).to_markdown())

    savefig(plot_loss_curves(result.histories), "a9a_loss_curves.png")
    savefig(plot_accuracy_curves(result.histories), "a9a_accuracy_curves.png")
    savefig(plot_gradient_norms(result.histories), "a9a_gradient_norms.png")
    savefig(plot_runtime_vs_loss(result.histories), "a9a_runtime_vs_loss.png")
    savefig(plot_comparison_bars(result.table, "runtime_sec", "seconds"), "a9a_runtime_bars.png")
    savefig(plot_comparison_bars(result.table, "test_accuracy", "accuracy"), "a9a_accuracy_bars.png")

    best_name = result.table["test_accuracy"].astype(float).idxmax()
    best_result = result.results[best_name]
    tmp_model = LogisticRegression(l2=1e-4)
    Xb_test = tmp_model._add_intercept(ds.X_test)
    preds = tmp_model.predict(best_result.final_params, Xb_test)
    y_true = tmp_model._normalize_labels(ds.y_test)
    savefig(
        plot_confusion_matrix(y_true, preds, class_names=["<=50K", ">50K"], title=f"a9a -- {best_name}"),
        "a9a_confusion_matrix.png",
    )


def generate_mnist_results() -> None:
    ds = load_mnist()
    optimizer_lrs = {
        "Gradient Descent": 0.5,
        "SGD": 0.5,
        "Adagrad": 0.5,
        "AdaGrad-Norm": 1.0,
        "RMSProp": 0.01,
        "Adam": 0.01,
        "L-BFGS": 1.0,
    }

    t0 = time.time()
    result = compare_with_full_batch_gd(
        lambda: SoftmaxRegression(n_classes=ds.n_classes, l2=1e-4),
        optimizer_lrs,
        ds.X_train, ds.y_train, ds.X_test, ds.y_test,
        epochs=25, batch_size=256,
    )
    logger.info("mnist comparison finished in %.1fs", time.time() - t0)

    result.table.to_csv(RESULTS / "mnist_comparison.csv")
    with open(ASSETS / "mnist_comparison_table.md", "w") as f:
        f.write(result.table.round(5).to_markdown())

    savefig(plot_loss_curves(result.histories), "mnist_loss_curves.png")
    savefig(plot_accuracy_curves(result.histories), "mnist_accuracy_curves.png")
    savefig(plot_gradient_norms(result.histories), "mnist_gradient_norms.png")
    savefig(plot_runtime_vs_loss(result.histories), "mnist_runtime_vs_loss.png")
    savefig(plot_comparison_bars(result.table, "runtime_sec", "seconds"), "mnist_runtime_bars.png")
    savefig(plot_comparison_bars(result.table, "test_accuracy", "accuracy"), "mnist_accuracy_bars.png")

    best_name = result.table["test_accuracy"].astype(float).idxmax()
    best_result = result.results[best_name]

    tmp_model = SoftmaxRegression(n_classes=ds.n_classes, l2=1e-4)
    Xb_test = tmp_model._add_intercept(ds.X_test)
    preds = tmp_model.predict(best_result.final_params, Xb_test)
    savefig(
        plot_confusion_matrix(
            ds.y_test, preds, class_names=[str(i) for i in range(10)], title=f"MNIST -- {best_name}"
        ),
        "mnist_confusion_matrix.png",
    )


if __name__ == "__main__":
    logger.info("Generating benchmark-function figures...")
    generate_benchmark_figures()

    logger.info("Generating a9a (logistic regression) results...")
    generate_a9a_results()

    logger.info("Generating MNIST (softmax regression) results...")
    generate_mnist_results()

    logger.info("Done. See %s and %s", ASSETS, RESULTS)
