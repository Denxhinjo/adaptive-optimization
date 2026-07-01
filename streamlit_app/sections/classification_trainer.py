from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from experiments.trainer import Trainer
from models import LogisticRegression, SoftmaxRegression
from optimizers import OPTIMIZER_REGISTRY, build_optimizer
from utils.metrics import RunMetrics
from visualization.plots import (
    plot_accuracy_curves,
    plot_confusion_matrix,
    plot_gradient_norms,
    plot_loss_curves,
)

from dashboard_utils import DATASET_LABELS, format_seconds, get_dataset

DEFAULT_LR = {
    "Gradient Descent": 0.5,
    "SGD": 0.1,
    "Adagrad": 0.5,
    "AdaGrad-Norm": 1.0,
    "RMSProp": 0.01,
    "Adam": 0.01,
    "L-BFGS": 1.0,
}


def render() -> None:
    st.title("Train a Classifier")
    st.caption("Logistic regression (binary) or softmax regression (multi-class), your choice of optimizer.")

    with st.sidebar:
        st.header("Training configuration")
        dataset_label = st.selectbox("Dataset", list(DATASET_LABELS))
        optimizer_name = st.selectbox("Optimizer", list(OPTIMIZER_REGISTRY), index=5)
        lr = st.number_input(
            "Learning rate", min_value=1e-6, max_value=10.0,
            value=DEFAULT_LR.get(optimizer_name, 0.01), format="%.5f",
        )
        epochs = st.slider("Epochs", 1, 200, 30)
        batch_size_option = st.selectbox("Batch size", ["Full batch", "32", "64", "128", "256"], index=2)
        batch_size = None if batch_size_option == "Full batch" else int(batch_size_option)
        l2 = st.select_slider("L2 regularization", options=[0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1], value=1e-4)
        refresh_every = st.slider("Live-chart refresh (epochs)", 1, 20, 5)
        track_memory = st.checkbox("Track peak memory usage", value=True)
        start = st.button("Start training", type="primary", use_container_width=True)

    if not start:
        st.info("Configure training in the sidebar and click **Start training**.", icon="⚙️")
        return

    dataset = get_dataset(dataset_label)
    st.write(
        f"**{dataset.name}** -- task: `{dataset.task}`, "
        f"train: {dataset.X_train.shape}, test: {dataset.X_test.shape}, classes: {dataset.n_classes}"
    )

    if dataset.task == "binary":
        model = LogisticRegression(l2=l2)
    else:
        model = SoftmaxRegression(n_classes=dataset.n_classes, l2=l2)

    optimizer = build_optimizer(optimizer_name, lr=lr)

    progress = st.progress(0.0, text="Training...")
    chart_cols = st.columns(3)
    loss_slot = chart_cols[0].empty()
    acc_slot = chart_cols[1].empty()
    grad_slot = chart_cols[2].empty()

    def on_epoch_end(epoch: int, params: np.ndarray, history: RunMetrics) -> None:
        progress.progress(epoch / epochs, text=f"Training... epoch {epoch}/{epochs}")
        if epoch % refresh_every != 0 and epoch != epochs:
            return
        histories = {optimizer_name: history}
        f1 = plot_loss_curves(histories)
        loss_slot.pyplot(f1)
        plt.close(f1)
        f2 = plot_accuracy_curves(histories, which="val_acc" if history.val_acc else "train_acc")
        acc_slot.pyplot(f2)
        plt.close(f2)
        f3 = plot_gradient_norms(histories)
        grad_slot.pyplot(f3)
        plt.close(f3)

    Xb = model._add_intercept(dataset.X_train)
    Xb_test = model._add_intercept(dataset.X_test)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        epochs=epochs,
        batch_size=batch_size,
        track_memory=track_memory,
        on_epoch_end=on_epoch_end,
    )
    result = trainer.fit(Xb, dataset.y_train, X_val=Xb_test, y_val=dataset.y_test)
    model.weights = result.final_params
    progress.progress(1.0, text="Done.")

    summary = result.history.summary()
    st.subheader("Training statistics")
    stat_cols = st.columns(5)
    stat_cols[0].metric("Final train loss", f"{summary['final_train_loss']:.5f}")
    stat_cols[1].metric("Test accuracy", f"{summary['test_accuracy']:.2%}" if summary["test_accuracy"] else "N/A")
    stat_cols[2].metric("Runtime", format_seconds(summary["runtime_sec"]))
    stat_cols[3].metric("Peak memory", f"{summary['peak_memory_mb']:.2f} MB")
    stat_cols[4].metric("Converged at epoch", summary["convergence_epoch"])

    st.subheader("Confusion matrix (test set)")
    preds = model.predict(model.weights, Xb_test)
    y_true = model._normalize_labels(dataset.y_test)
    fig_cm = plot_confusion_matrix(y_true, preds, title=f"{model.name} on {dataset.name}")
    st.pyplot(fig_cm)
    plt.close(fig_cm)

    with st.expander("Raw per-epoch history"):
        st.dataframe(
            {
                "epoch": list(range(1, len(result.history.train_loss) + 1)),
                "train_loss": result.history.train_loss,
                "val_loss": result.history.val_loss or [None] * len(result.history.train_loss),
                "val_acc": result.history.val_acc or [None] * len(result.history.train_loss),
                "grad_norm": result.history.grad_norm,
            },
            use_container_width=True,
        )
