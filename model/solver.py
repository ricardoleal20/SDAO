"""
Solver interface for testing and benchmarking models.

In here, we'll select:
    - The model to test
    - The objective function
    - The optimization algorithm
    - The number of experiments
"""
from typing import Callable, TypedDict, Sequence
import numpy as np
# Import TQDM for the progress bar
from tqdm import tqdm


class ExperimentFunction(TypedDict):
    """Set the experiment function, along with their domain."""
    name: str
    call: Callable[[np.ndarray], float | int]
    domain: tuple[float, float]


class BenchmarkResult(TypedDict):
    """Set the benchmark result dictionary"""
    iteration: int
    best_value: float
    best_position: np.ndarray
    time: float
    memory: float


class Solver:
    """Solver interface for testing and benchmarking models."""

    _n_of_exp: int
    _functions: Sequence[ExperimentFunction]
    # Slots
    __slots__ = ["_n_of_exp", "_functions"]

    def __init__(
        self,
        num_experiments: int = 10,
        *,
        functions: Sequence[ExperimentFunction]
    ) -> None:
        self._n_of_exp = num_experiments
        self._functions = functions

    def benchmark(
        self,
        dimension: int,
        model: Callable[[
            # The objective function
            Callable[[np.ndarray], float | int],
            # The domain
            Sequence[tuple[float, float]] | tuple[float, float],
            # The dimension
            int
        ], tuple[float, np.ndarray]],
    ) -> list[BenchmarkResult]:
        """Benchmark the model using the given functions."""
        results: list[BenchmarkResult] = []
        for func in self._functions:
            print(f"Running benchmark for {func['name']}...")
            for i in tqdm(range(self._n_of_exp)):
                best_value, best_position = model(
                    func["call"], func["domain"], dimension)
                # With this, append the results
                results.append({
                    "iteration": i,
                    "best_value": best_value,
                    "best_position": best_position,
                    "time": 0.0,
                    "memory": 0.0
                })
        # Return the results
        return results
