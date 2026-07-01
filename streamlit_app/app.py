"""Adaptive Optimization Algorithms from Scratch -- Streamlit dashboard.

Run locally with:
    streamlit run streamlit_app/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import streamlit as st  # noqa: E402 -- sys.path must be patched above before this import

from sections import about, benchmark_playground, classification_trainer, compare_optimizers  # noqa: E402

st.set_page_config(
    page_title="Adaptive Optimization Algorithms from Scratch",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGES = {
    "Overview": about,
    "Benchmark Playground": benchmark_playground,
    "Train a Classifier": classification_trainer,
    "Compare Optimizers": compare_optimizers,
}

st.sidebar.title("Adaptive Optimization")
st.sidebar.caption("First-order optimizers implemented from scratch in NumPy.")
page_name = st.sidebar.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")

st.sidebar.divider()
st.sidebar.markdown(
    "**Implemented optimizers**\n"
    "- Gradient Descent\n"
    "- SGD (+ momentum)\n"
    "- Adagrad\n"
    "- AdaGrad-Norm\n"
    "- RMSProp\n"
    "- Adam\n"
    "- L-BFGS\n"
)
PAGES[page_name].render()
