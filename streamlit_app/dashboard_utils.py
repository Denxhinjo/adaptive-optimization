from __future__ import annotations

import streamlit as st

from datasets.loaders import Dataset, load_dataset

DATASET_LABELS = {
    "Synthetic (binary, instant)": ("synthetic-binary", {}),
    "Synthetic (multi-class, instant)": ("synthetic-multiclass", {}),
    "a9a (binary, ~32K rows)": ("a9a", {}),
    "MNIST (multi-class, 60K rows)": ("mnist", {}),
}


@st.cache_data(show_spinner="Loading dataset...")
def get_dataset(label: str) -> Dataset:
    name, kwargs = DATASET_LABELS[label]
    return load_dataset(name, **kwargs)


def format_seconds(s: float) -> str:
    if s < 1:
        return f"{s * 1000:.0f} ms"
    return f"{s:.2f} s"
