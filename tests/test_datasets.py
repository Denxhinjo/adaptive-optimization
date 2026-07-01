"""Unit tests for dataset loaders. The a9a/MNIST tests are skipped
automatically if the raw files haven't been downloaded/cached yet, so the
full suite still runs quickly and offline via the synthetic generators."""
from __future__ import annotations

import pytest
import scipy.sparse as sp

from datasets.loaders import (
    DATA_DIR,
    load_a9a,
    load_dataset,
    load_mnist,
    make_synthetic_binary,
    make_synthetic_multiclass,
)

A9A_CACHED = (DATA_DIR / "a9a").exists() and (DATA_DIR / "a9a.t").exists()
MNIST_CACHED = (DATA_DIR / "mnist").exists() and (DATA_DIR / "mnist.t").exists()


def test_make_synthetic_binary_shapes():
    ds = make_synthetic_binary(n_samples=200, n_features=10, seed=0)
    assert ds.task == "binary"
    assert ds.n_classes == 2
    assert sp.issparse(ds.X_train)
    assert ds.X_train.shape[1] == 10
    assert ds.X_train.shape[0] + ds.X_test.shape[0] == 200


def test_make_synthetic_multiclass_shapes():
    ds = make_synthetic_multiclass(n_samples=300, n_features=8, n_classes=4, seed=0)
    assert ds.task == "multiclass"
    assert ds.n_classes == 4
    assert set(ds.y_train.tolist()) <= {0, 1, 2, 3}


def test_load_dataset_registry():
    ds = load_dataset("synthetic-binary", n_samples=100, n_features=5)
    assert ds.name == "synthetic-binary"


def test_load_dataset_unknown_name_raises():
    with pytest.raises(ValueError):
        load_dataset("not-a-real-dataset")


@pytest.mark.skipif(not A9A_CACHED, reason="a9a not downloaded/cached locally")
def test_load_a9a():
    ds = load_a9a()
    assert ds.n_features == 123
    assert ds.n_classes == 2
    assert set(ds.y_train.tolist()) <= {-1.0, 1.0}


@pytest.mark.skipif(not MNIST_CACHED, reason="MNIST not downloaded/cached locally")
def test_load_mnist():
    ds = load_mnist()
    assert ds.n_features == 784
    assert ds.n_classes == 10
