"""Dataset loading utilities.

Downloads LIBSVM-format datasets (a9a, MNIST) on first use and caches them
under :data:`DATA_DIR` (``datasets/raw`` by default, overridable via the
``ADAPTIVE_OPT_DATA_DIR`` environment variable). Also provides a synthetic
data generator for fast unit tests / dashboard demos that don't need the
full datasets.

Both real datasets are served in sparse LIBSVM format via
``sklearn.datasets.load_svmlight_file`` -- every optimizer/model in this
library works directly on ``scipy.sparse`` matrices without densifying.
"""
from __future__ import annotations

import bz2
import logging
import os
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

DATA_DIR = Path(os.environ.get("ADAPTIVE_OPT_DATA_DIR", Path(__file__).resolve().parent / "raw"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

_URLS = {
    "a9a": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a9a",
    "a9a.t": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a9a.t",
    "mnist": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.bz2",
    "mnist.t": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.t.bz2",
}


@dataclass
class Dataset:
    """Container bundling a train/test split with metadata for the UI."""

    name: str
    X_train: sp.csr_matrix
    y_train: np.ndarray
    X_test: sp.csr_matrix
    y_test: np.ndarray
    n_features: int
    n_classes: int
    task: str  # "binary" | "multiclass"

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"Dataset(name={self.name!r}, task={self.task}, "
            f"train={self.X_train.shape}, test={self.X_test.shape}, "
            f"n_classes={self.n_classes})"
        )


def _fetch(key: str) -> Path:
    """Download (if needed) and decompress (if needed) a cached raw file."""
    url = _URLS[key]
    local = DATA_DIR / url.rsplit("/", 1)[1]
    already_decompressed = local.suffix == ".bz2" and local.with_suffix("").exists()
    if not local.exists() and not already_decompressed:
        logger.info("Downloading %s -> %s", url, local)
        urllib.request.urlretrieve(url, local)  # noqa: S310 - trusted, fixed URL list
    if local.suffix == ".bz2":
        out = local.with_suffix("")
        if not out.exists():
            logger.info("Decompressing %s -> %s", local, out)
            with bz2.open(local, "rb") as fi, open(out, "wb") as fo:
                shutil.copyfileobj(fi, fo)
        return out
    return local


def load_a9a() -> Dataset:
    """UCI Adult ('a9a') census-income binary classification task.

    123 sparse binary features, labels in ``{-1, +1}``, ~32.5K train /
    16.3K test examples. Source: LIBSVM datasets (Chang & Lin).
    """
    Xtr, ytr = load_svmlight_file(str(_fetch("a9a")), n_features=123)
    Xte, yte = load_svmlight_file(str(_fetch("a9a.t")), n_features=123)
    return Dataset(
        name="a9a",
        X_train=Xtr.tocsr(),
        y_train=ytr,
        X_test=Xte.tocsr(),
        y_test=yte,
        n_features=123,
        n_classes=2,
        task="binary",
    )


def load_mnist() -> Dataset:
    """MNIST handwritten digits, LIBSVM multiclass encoding.

    784 pixel-intensity features (rescaled to [0, 1]), 10 classes,
    60K train / 10K test examples.
    """
    Xtr, ytr = load_svmlight_file(str(_fetch("mnist")), n_features=784)
    Xte, yte = load_svmlight_file(str(_fetch("mnist.t")), n_features=784)
    # The LIBSVM encoding stores raw pixel intensities in [0, 255], not [0, 1] --
    # rescale so gradient magnitudes are well-conditioned for a linear model.
    Xtr = Xtr.tocsr() / 255.0
    Xte = Xte.tocsr() / 255.0
    return Dataset(
        name="mnist",
        X_train=Xtr,
        y_train=ytr.astype(int),
        X_test=Xte,
        y_test=yte.astype(int),
        n_features=784,
        n_classes=10,
        task="multiclass",
    )


def make_synthetic_binary(
    n_samples: int = 2000,
    n_features: int = 20,
    seed: int = 0,
    test_size: float = 0.2,
) -> Dataset:
    """Fast, dependency-free synthetic binary classification dataset --
    used by unit tests and the Streamlit dashboard's "quick demo" mode so
    users don't have to wait on a real download."""
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=max(2, n_features // 2),
        random_state=seed,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    return Dataset(
        name="synthetic-binary",
        X_train=sp.csr_matrix(X_train),
        y_train=y_train,
        X_test=sp.csr_matrix(X_test),
        y_test=y_test,
        n_features=n_features,
        n_classes=2,
        task="binary",
    )


def make_synthetic_multiclass(
    n_samples: int = 3000,
    n_features: int = 20,
    n_classes: int = 4,
    seed: int = 0,
    test_size: float = 0.2,
) -> Dataset:
    """Fast synthetic multi-class dataset, analogous to
    :func:`make_synthetic_binary`."""
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=max(2, n_features // 2),
        n_classes=n_classes,
        n_clusters_per_class=1,
        random_state=seed,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    return Dataset(
        name="synthetic-multiclass",
        X_train=sp.csr_matrix(X_train),
        y_train=y_train,
        X_test=sp.csr_matrix(X_test),
        y_test=y_test,
        n_features=n_features,
        n_classes=n_classes,
        task="multiclass",
    )


DATASET_REGISTRY = {
    "a9a": load_a9a,
    "mnist": load_mnist,
    "synthetic-binary": make_synthetic_binary,
    "synthetic-multiclass": make_synthetic_multiclass,
}


def load_dataset(name: str, **kwargs) -> Dataset:
    try:
        loader = DATASET_REGISTRY[name]
    except KeyError as exc:
        valid = ", ".join(DATASET_REGISTRY)
        raise ValueError(f"Unknown dataset '{name}'. Valid options: {valid}") from exc
    return loader(**kwargs)
