"""Dataset loaders for a9a, MNIST, and fast synthetic classification data."""
from .loaders import (
    DATA_DIR,
    DATASET_REGISTRY,
    Dataset,
    load_a9a,
    load_dataset,
    load_mnist,
    make_synthetic_binary,
    make_synthetic_multiclass,
)

__all__ = [
    "Dataset",
    "DATA_DIR",
    "DATASET_REGISTRY",
    "load_a9a",
    "load_mnist",
    "make_synthetic_binary",
    "make_synthetic_multiclass",
    "load_dataset",
]
