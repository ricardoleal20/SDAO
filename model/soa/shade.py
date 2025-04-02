"""
Implementation of the SHADE algorithm with
Iterative Local Search for Large-Scale Global Optimization.

Link: https://ieeexplore.ieee.org/document/8477755
"""

from typing import Callable, Sequence
import numpy as np

# Local imports
from model.soa.template import Algorithm, StepSolution


class SHADEwithILS(Algorithm):  # pylint: disable=R0903
    """Implementation of SHADE with Iterative Local Search."""

    _n_iterations: int
    _n_population: int
    _memory_size: int
    _memory_cr: np.ndarray
    _memory_f: np.ndarray
    _iterations: list[StepSolution]
    _verbose: bool

    __slots__ = [
        "_n_population",
        "_n_iterations",
        "_memory_size",
        "_memory_cr",
        "_memory_f",
        "_iterations",
        "_verbose",
    ]

    def __init__(
        self,
        n_population: int = 50,
        n_iterations: int = 1000,
        memory_size: int = 10,
        *,
        verbose: bool = False,
    ):
        """
        Initialize the optimizer.

        Parameters:
        - population_size: int, the size of the population.
        - max_iterations: int, the maximum number of iterations.
        - H: int, the size of the memory for adaptation.
        """
        self._n_population = n_population
        self._n_iterations = n_iterations
        self._memory_size = memory_size
        self._memory_cr = np.full(memory_size, 0.5)
        self._memory_f = np.full(memory_size, 0.5)
        # Extra params
        self._verbose = verbose
        self._iterations = []

    def optimize(  # pylint: disable=R0914
        self,
        objective_fn: Callable[[np.ndarray], float | int],
        bounds: Sequence[tuple[float, float]] | tuple[float, float],
        dimension: int,
    ) -> tuple[float, np.ndarray]:
        """
        Perform optimization using SHADE with Iterative Local Search.

        Parameters:
        - objective_fn: Callable, the objective function to minimize.
        - bounds: Sequence of tuples indicating the bounds for each dimension.
        - dimension: int, the number of dimensions.

        Returns:
        - tuple of best fitness and best solution found.
        """
        self._iterations = []
        fn_bounds = [bounds] if isinstance(bounds, tuple) else bounds
        # Initialize population
        population = np.array(
            [
                np.random.uniform(low, high, size=dimension)
                for low, high in fn_bounds
                for _ in range(self._n_population)
            ]
        )

        fitness = np.array([objective_fn(ind) for ind in population])

        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        archive = []

        for iteration in range(self._n_iterations):
            new_population = population.copy()
            cr_pop = np.zeros(self._n_population)
            f_pop = np.zeros(self._n_population)

            # Initialize idx
            idx = 0
            for i in range(self._n_population):
                # Adaptation of CR and F
                idx = np.random.randint(self._memory_size)
                cr_pop[i] = np.clip(np.random.normal(self._memory_cr[idx], 0.1), 0, 1)
                f_pop[i] = np.clip(np.random.normal(self._memory_f[idx], 0.1), 0, 1)

                # Mutation and Crossover
                r1, r2, r3 = np.random.choice(
                    [j for j in range(self._n_population) if j != i], 3, replace=False
                )
                mutant = population[r1] + f_pop[i] * (population[r2] - population[r3])
                mutant = np.clip(
                    mutant, [b[0] for b in fn_bounds], [b[1] for b in fn_bounds]
                )

                trial = np.where(
                    np.random.rand(len(population[i])) < cr_pop[i],
                    mutant,
                    population[i],
                )
                trial_fitness = objective_fn(trial)

                # Selection
                if trial_fitness < fitness[i]:
                    new_population[i] = trial
                    fitness[i] = trial_fitness
                    archive.append(population[i])

            # Update memory
            if len(archive) > 0:
                archive = archive[-self._memory_size :]
                successful_cr = cr_pop[: len(archive)]
                successful_f = f_pop[: len(archive)]
                delta_fitness = np.abs(fitness[: len(archive)] - fitness[best_idx])
                # Get the sum of the delta fitness and implement it on the memory
                sum_delta_fitness = np.sum(delta_fitness)
                if sum_delta_fitness != 0:
                    self._memory_cr[idx] = (
                        np.sum(delta_fitness * successful_cr) / sum_delta_fitness
                    )
                    self._memory_f[idx] = (
                        np.sum(delta_fitness * successful_f) / sum_delta_fitness
                    )
                else:
                    # Fill with random values
                    self._memory_cr[idx] = np.random.uniform(0, 1)
                    self._memory_f[idx] = np.random.uniform(0, 1)

            population = new_population

            # Local Search
            for i in range(self._n_population):
                local_candidate = population[i] + np.random.uniform(
                    -0.1, 0.1, len(population[i])
                )
                local_candidate = np.clip(
                    local_candidate,
                    [b[0] for b in fn_bounds],
                    [b[1] for b in fn_bounds],
                )
                local_fitness = objective_fn(local_candidate)
                if local_fitness < fitness[i]:
                    population[i] = local_candidate
                    fitness[i] = local_fitness

            # Update best solution
            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < best_fitness:
                best_solution = population[current_best_idx]
                best_fitness = fitness[current_best_idx]

            if self._verbose:
                print(f"Iteration {iteration + 1}, Best Fitness: {best_fitness}")
            self._iterations.append((iteration, best_fitness))

        return best_fitness, best_solution
