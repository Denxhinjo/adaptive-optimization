"""Training loop, cross-optimizer comparison, and benchmark-function runner."""
from .benchmark_runner import BenchmarkRun, run_all_optimizers_on_benchmark, run_optimizer_on_benchmark
from .compare_optimizers import ComparisonResult, compare_optimizers
from .trainer import Trainer, TrainResult

__all__ = [
    "Trainer",
    "TrainResult",
    "compare_optimizers",
    "ComparisonResult",
    "BenchmarkRun",
    "run_optimizer_on_benchmark",
    "run_all_optimizers_on_benchmark",
]
