from __future__ import annotations

import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

from benchmarks import BENCHMARK_REGISTRY
from experiments.benchmark_runner import run_all_optimizers_on_benchmark
from optimizers import OPTIMIZER_REGISTRY, build_optimizer
from visualization.trajectory import animate_trajectories, plot_contour_with_trajectories, plot_convergence_curves

DEFAULT_LR = {
    "Gradient Descent": 0.001,
    "SGD": 0.001,
    "Adagrad": 0.5,
    "AdaGrad-Norm": 1.0,
    "RMSProp": 0.01,
    "Adam": 0.05,
    "L-BFGS": 1.0,
}


def render() -> None:
    st.title("Benchmark Function Playground")
    st.caption("Watch optimizers race across classic non-convex 2-D landscapes.")

    left, right = st.columns([1, 2])

    with left:
        fn_name = st.selectbox("Benchmark function", list(BENCHMARK_REGISTRY), index=0)
        fn = BENCHMARK_REGISTRY[fn_name]
        st.caption(fn.description)

        selected = st.multiselect(
            "Optimizers to race",
            list(OPTIMIZER_REGISTRY),
            default=["SGD", "Adagrad", "RMSProp", "Adam"],
        )
        n_iters = st.slider("Iterations", 10, 1000, 300, step=10)

        use_default_start = st.checkbox("Use default (challenging) start point", value=True)
        if use_default_start:
            start = fn.default_start
        else:
            (x0, x1), (y0, y1) = fn.domain
            sx = st.slider("start x", float(x0), float(x1), float(fn.default_start[0]))
            sy = st.slider("start y", float(y0), float(y1), float(fn.default_start[1]))
            start = (sx, sy)

        st.markdown("**Learning rates**")
        lrs = {}
        for name in selected:
            lrs[name] = st.number_input(
                f"{name} lr", min_value=1e-6, max_value=10.0, value=DEFAULT_LR.get(name, 0.01),
                format="%.5f", key=f"bm_lr_{name}",
            )

        run = st.button("Run race", type="primary", use_container_width=True)
        make_gif = st.checkbox("Also render animated GIF (slower)", value=False)

    if not run:
        with right:
            fig = plot_contour_with_trajectories(fn, {})
            st.pyplot(fig)
            plt.close(fig)
        return

    if not selected:
        st.warning("Select at least one optimizer.")
        return

    optimizers = {name: build_optimizer(name, lr=lrs[name]) for name in selected}
    runs = run_all_optimizers_on_benchmark(fn, optimizers, start=start, n_iters=n_iters)

    with right:
        fig1 = plot_contour_with_trajectories(fn, runs)
        st.pyplot(fig1)
        plt.close(fig1)

        fig2 = plot_convergence_curves(runs)
        st.pyplot(fig2)
        plt.close(fig2)

    st.subheader("Final results")
    rows = []
    for name, r in runs.items():
        best_dist = min(
            ((r.final_point[0] - mx) ** 2 + (r.final_point[1] - my) ** 2) ** 0.5
            for mx, my in fn.minima
        )
        rows.append(
            {
                "optimizer": name,
                "final_x": round(float(r.final_point[0]), 4),
                "final_y": round(float(r.final_point[1]), 4),
                "final_value": round(float(r.losses[-1]), 6),
                "iterations_run": len(r.trajectory) - 1,
                "distance_to_nearest_minimum": round(best_dist, 4),
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)

    if make_gif:
        with st.spinner("Rendering animation..."):
            tmp_path = Path(tempfile.gettempdir()) / "adaptive_opt_benchmark_race.gif"
            animate_trajectories(fn, runs, save_path=str(tmp_path))
            st.image(str(tmp_path), caption=f"Optimizer trajectories on {fn.name}")
