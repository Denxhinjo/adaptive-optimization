from __future__ import annotations

import matplotlib.pyplot as plt
import streamlit as st

from experiments.compare_optimizers import compare_optimizers as run_comparison
from models import LogisticRegression, SoftmaxRegression
from optimizers import OPTIMIZER_REGISTRY, build_optimizer
from visualization.plots import (
    plot_accuracy_curves,
    plot_comparison_bars,
    plot_gradient_norms,
    plot_loss_curves,
    plot_runtime_vs_loss,
)

from dashboard_utils import DATASET_LABELS, get_dataset

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
    st.title("Compare Optimizers Side-by-Side")
    st.caption("Train every selected optimizer on the same model/dataset/hyperparameters and compare.")

    with st.sidebar:
        st.header("Comparison configuration")
        dataset_label = st.selectbox("Dataset", list(DATASET_LABELS), key="cmp_dataset")
        selected = st.multiselect(
            "Optimizers", list(OPTIMIZER_REGISTRY),
            default=["SGD", "Adagrad", "RMSProp", "Adam", "L-BFGS"],
        )
        epochs = st.slider("Epochs", 1, 200, 25, key="cmp_epochs")
        batch_size_option = st.selectbox("Batch size", ["Full batch", "32", "64", "128"], index=2, key="cmp_bs")
        batch_size = None if batch_size_option == "Full batch" else int(batch_size_option)
        l2 = st.select_slider("L2 regularization", options=[0.0, 1e-5, 1e-4, 1e-3, 1e-2], value=1e-4, key="cmp_l2")

        with st.expander("Per-optimizer learning rates"):
            lrs = {
                name: st.number_input(
                    f"{name} lr", min_value=1e-6, max_value=10.0,
                    value=DEFAULT_LR.get(name, 0.01), format="%.5f", key=f"cmp_lr_{name}",
                )
                for name in selected
            }

        run = st.button("Run comparison", type="primary", use_container_width=True)

    if not run:
        st.info("Configure the comparison in the sidebar and click **Run comparison**.", icon="⚙️")
        return
    if not selected:
        st.warning("Select at least one optimizer.")
        return

    dataset = get_dataset(dataset_label)

    def model_factory():
        if dataset.task == "binary":
            return LogisticRegression(l2=l2)
        return SoftmaxRegression(n_classes=dataset.n_classes, l2=l2)

    optimizers = {name: build_optimizer(name, lr=lrs[name]) for name in selected}

    with st.spinner(f"Training {len(optimizers)} optimizers..."):
        result = run_comparison(
            model_factory,
            optimizers,
            dataset.X_train,
            dataset.y_train,
            dataset.X_test,
            dataset.y_test,
            epochs=epochs,
            batch_size=batch_size,
        )

    st.subheader("Comparison table")
    st.dataframe(result.table, use_container_width=True)
    st.download_button(
        "Download table as CSV",
        result.table.to_csv().encode("utf-8"),
        file_name=f"comparison_{dataset.name}.csv",
        mime="text/csv",
    )

    st.subheader("Curves")
    c1, c2 = st.columns(2)
    with c1:
        fig = plot_loss_curves(result.histories)
        st.pyplot(fig)
        plt.close(fig)
    with c2:
        which = "val_acc" if dataset.X_test is not None else "train_acc"
        fig = plot_accuracy_curves(result.histories, which=which)
        st.pyplot(fig)
        plt.close(fig)

    c3, c4 = st.columns(2)
    with c3:
        fig = plot_gradient_norms(result.histories)
        st.pyplot(fig)
        plt.close(fig)
    with c4:
        fig = plot_runtime_vs_loss(result.histories)
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Summary bar charts")
    b1, b2, b3 = st.columns(3)
    with b1:
        fig = plot_comparison_bars(result.table, "runtime_sec", "seconds")
        st.pyplot(fig)
        plt.close(fig)
    with b2:
        fig = plot_comparison_bars(result.table, "convergence_epoch", "epoch")
        st.pyplot(fig)
        plt.close(fig)
    with b3:
        fig = plot_comparison_bars(result.table, "peak_memory_mb", "MB")
        st.pyplot(fig)
        plt.close(fig)
