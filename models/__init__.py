"""Linear classification models trained with the optimizers in
:mod:`optimizers`. See each module's docstring for loss/gradient formulas.
"""
from .base import BaseModel
from .logistic_regression import LogisticRegression
from .softmax_regression import SoftmaxRegression

__all__ = ["BaseModel", "LogisticRegression", "SoftmaxRegression"]
