"""From-scratch first-order and quasi-Newton optimizers.

All optimizers share the :class:`~optimizers.base.Optimizer` interface:
``step(params, grad) -> new_params``, with internal state (moments,
accumulators, ...) owned by the instance. See each module's docstring for
the exact update equations, and the project README for a side-by-side
comparison table.
"""
from .base import Optimizer, StepResult
from .gradient_descent import GradientDescent
from .sgd import SGD
from .adagrad import Adagrad
from .adagrad_norm import AdagradNorm
from .rmsprop import RMSProp
from .adam import Adam
from .lbfgs import LBFGS, LBFGSScipyReference

__all__ = [
    "Optimizer",
    "StepResult",
    "GradientDescent",
    "SGD",
    "Adagrad",
    "AdagradNorm",
    "RMSProp",
    "Adam",
    "LBFGS",
    "LBFGSScipyReference",
    "OPTIMIZER_REGISTRY",
    "build_optimizer",
]

#: Name -> class registry, used by the Streamlit dashboard and experiment
#: scripts to construct optimizers from user-selected strings.
OPTIMIZER_REGISTRY: dict[str, type] = {
    "Gradient Descent": GradientDescent,
    "SGD": SGD,
    "Adagrad": Adagrad,
    "AdaGrad-Norm": AdagradNorm,
    "RMSProp": RMSProp,
    "Adam": Adam,
    "L-BFGS": LBFGS,
}


def build_optimizer(name: str, **kwargs) -> Optimizer:
    """Construct an optimizer instance by name using :data:`OPTIMIZER_REGISTRY`."""
    try:
        cls = OPTIMIZER_REGISTRY[name]
    except KeyError as exc:
        valid = ", ".join(OPTIMIZER_REGISTRY)
        raise ValueError(f"Unknown optimizer '{name}'. Valid options: {valid}") from exc
    return cls(**kwargs)
