"""
Define the performance metrics for the scenario demos.

This metrics are going to be used to evaluate the performance of all
the algorithms in front of each set of functions that had been implemented.
"""

from collections import defaultdict
import numpy as np
import pydash as _py

# Local imports
from model.solver import BenchmarkResult
from model.__main__ import functions_due_to_scenario

BENCH_DATA = dict[str, list[BenchmarkResult]]


def __create_best_solution_per_alg_dict() -> dict[str, dict[str, list[float]]]:
    """Create the dictionary that will store the best solutions for each algorithm and function."""
    return defaultdict(lambda: defaultdict(list))


def best_solution(data: BENCH_DATA, *_args, **_kwargs) -> dict[str, dict[str, float]]:
    """Based on the benchmark results, return the best solution found."""
    best_solution_per_alg = __create_best_solution_per_alg_dict()
    # Iterate over the algorithms to find their experiments
    for algorithm, experiments in data.items():
        # Iterate over the experiments
        for experiment in experiments:
            # Then, based on the experiments, find the best solution for each metric
            best_solution_per_alg[algorithm][experiment["function"]].append(
                abs(
                    # Get the difference between the best solution and the real value
                    experiment["error"] if experiment["error"] else float("inf")
                )
            )
    # Then, by the end, get the average of the best solutions for each metric
    best_solution = {
        algorithm: {function: np.mean(values) for function, values in metrics.items()}
        for algorithm, metrics in best_solution_per_alg.items()
    }
    return best_solution  # type: ignore


def convergence_rate(data: BENCH_DATA, scenario: int) -> dict[str, dict[str, float]]:
    """Based on the benchmark results, return the convergence rate of the solutions found."""
    convergence_rate_per_alg = __create_best_solution_per_alg_dict()
    # Get the functions for this scenario
    functions = functions_due_to_scenario(scenario)
    # Then, iterate over the algorithms to find in which point do we get a
    # solution that is (solution - optimal < 1e-3)
    for algorithm, experiments in data.items():
        # Iterate over the experiments
        for experiment in experiments:
            # Get the optimal solution from this method
            function = _py.find(
                functions, lambda x: x["name"] == experiment["function"]
            )
            if not function or function.get("optimal_value", None) is None:
                continue
            optimal_value = function.get("optimal_value", int(1e6))
            # Iterate over the trajectory
            find_value: bool = False
            for trajectory in experiment["trajectory"]:
                # In the trajectory, we found (Iteration, Best Solution)
                (iteration, best_solution) = trajectory
                # Check if the error is less than 1e-4
                if abs(best_solution - optimal_value) < 1e-4:
                    convergence_rate_per_alg[algorithm][experiment["function"]].append(
                        iteration
                    )
                    find_value = True
                    break
            # If we didn't find the value, append the last iteration
            if not find_value:
                convergence_rate_per_alg[algorithm][experiment["function"]].append(
                    iteration
                )
    # Then, by the end, get the average of the convergence rates for each metric
    convergence_rate = {
        algorithm: {
            function: round(np.mean(values), 0) for function, values in metrics.items()
        }
        for algorithm, metrics in convergence_rate_per_alg.items()
    }
    return convergence_rate  # type: ignore


def stability(data: BENCH_DATA, *_args, **_kwargs) -> dict[str, dict[str, float]]:
    """Based on the benchmark results, return the stability of the solutions found."""
    stability_per_alg = __create_best_solution_per_alg_dict()
    # Iterate over the algorithms to find their experiments
    for algorithm, experiments in data.items():
        # Iterate over the experiments
        for experiment in experiments:
            # Then, based on the experiments, find the stability of the solutions
            stability_per_alg[algorithm][experiment["function"]].append(
                abs(
                    # Get the difference between the best solution and the real value
                    experiment["error"] if experiment["error"] else float("inf")
                )
            )
    # Then, by the end, get the average of the stability of the solutions for each metric
    stability = {
        algorithm: {function: np.std(values) for function, values in metrics.items()}
        for algorithm, metrics in stability_per_alg.items()
    }
    return stability  # type: ignore


def time_complexity(data: BENCH_DATA, *_args, **_kwargs) -> dict[str, dict[str, float]]:
    """Based on the benchmark results, return the time complexity of the algorithms."""
    time_complexity_per_alg = __create_best_solution_per_alg_dict()
    # Iterate over the algorithms to find their experiments
    for algorithm, experiments in data.items():
        # Iterate over the experiments
        for experiment in experiments:
            # Then, based on the experiments, find the time complexity of the algorithms
            time_complexity_per_alg[algorithm][experiment["function"]].append(
                experiment["time"]
            )
    # Then, by the end, get the average of the time complexity of the algorithms for each metric
    time_complexity = {
        algorithm: {function: np.mean(values) for function, values in metrics.items()}
        for algorithm, metrics in time_complexity_per_alg.items()
    }
    return time_complexity  # type: ignore


# DEFINE THE METRICS AS A TUPLE
METRICS = (
    best_solution,
    convergence_rate,
    stability,
)
