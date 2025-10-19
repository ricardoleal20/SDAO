"""
Solver interface for testing and benchmarking models.

In here, we'll select:
    - The model to test
    - The objective function
    - The optimization algorithm
    - The number of experiments
"""

import time
import tracemalloc
from typing import Callable, Optional, Sequence, TypedDict

import numpy as np

# Import TQDM for the progress bar
from tqdm import tqdm
from typing_extensions import NotRequired

# Local imports
from model.soa.template import StepSolution


class ExperimentFunction(TypedDict):
    """Set the experiment function, along with their domain."""

    name: str
    call: Callable[[np.ndarray], float | int | np.signedinteger]
    domain: tuple[float, float]
    dimension: NotRequired[int]
    optimal_value: NotRequired[float]
    optimal_x_value: NotRequired[np.ndarray | list[float]]


class BenchmarkResult(TypedDict):
    """Set the benchmark result dictionary"""

    iteration: int
    best_value: float
    best_position: np.ndarray
    error: Optional[float]  # Expected error...
    error_x: Optional[float]  # Expected error in x array
    function: str
    time: float
    memory: float
    trajectory: list[StepSolution]


class Solver:
    """Solver interface for testing and benchmarking models."""

    _n_of_exp: int
    _functions: Sequence[ExperimentFunction]
    # Slots
    __slots__ = ["_n_of_exp", "_functions"]

    def __init__(
        self, num_experiments: int = 10, *, functions: Sequence[ExperimentFunction]
    ) -> None:
        self._n_of_exp = num_experiments
        self._functions = functions

    def benchmark(
        self,
        dimension: int,
        model: Callable[
            [
                # The objective function
                Callable[[np.ndarray], float | int | np.signedinteger],
                # The domain
                Sequence[tuple[float, float]] | tuple[float, float],
                # The dimension
                int,
            ],
            tuple[float, np.ndarray],
        ],
        trajectory: Callable[[], list[StepSolution]],
        *,
        store_trajectory: bool = False,
        profile_memory: bool = False,
        show_progress: bool = False,
    ) -> list[BenchmarkResult]:
        """Benchmark the model using the given functions."""
        results: list[BenchmarkResult] = []
        total_functions = len(self._functions)
        for i, func in enumerate(self._functions, start=1):
            # Get the dimension of the function, just in case that
            # we do not have it on the Experimental Function...
            print(f"Running benchmark for {func['name']}... {i / total_functions:.2%}")
            iter_obj = (
                tqdm(range(self._n_of_exp)) if show_progress else range(self._n_of_exp)
            )
            for i in iter_obj:
                # Calculate the time taken in this iteration
                start_time = time.time()
                # Also, calculate the memory usage
                if profile_memory:
                    tracemalloc.start()
                # Run the model
                best_value, best_position = model(
                    func["call"], func["domain"], func.get("dimension", dimension)
                )
                if profile_memory:
                    _, peak = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                else:
                    peak = 0
                # With this, append the results
                results.append(
                    {
                        "iteration": i,
                        "best_value": best_value,
                        "best_position": best_position,
                        "function": func["name"],
                        "trajectory": trajectory() if store_trajectory else [],
                        "time": (time.time() - start_time)
                        * 1000,  # to convert it to microseconds
                        # Memory usage is initially in kilobytes, converting it to megabytes
                        "memory": peak / 1024.0,
                        "error": np.abs(
                            best_value - func.get("optimal_value", float("inf"))
                        )
                        if func.get("optimal_value") is not None
                        else None,
                        "error_x": _get_error_x(best_position, func),
                    }
                )
        # Return the results
        return results


def _get_error_x(
    best_position: np.ndarray, func: ExperimentFunction
) -> Optional[float]:
    if func.get("optimal_x_value") is None:
        # This is a placeholder for the error_x calculation when optimal_x_value is None
        return None
    # Then, we should check if we have a predetermined optimal_x_value with the same length
    # as the dimension
    expected_x_array = func.get("optimal_x_value", [])
    if len(expected_x_array) != len(best_position):
        # Multiply it by the given dimension...
        expected_x_array = expected_x_array * len(best_position)
    # Return the absolute error...
    return sum(
        np.abs(best_position[i] - expected_x_array[i])
        for i in range(len(best_position))
    )
