"""
Include the Stochastic Fractal Search algorithm as one of the
State of Art (SoA) algorithms to compare with the
SDAO algorithm.

Link: https://www.sciencedirect.com/science/article/abs/pii/S0950705114002822?via%3Dihub
"""

from typing import Sequence, Callable
import numpy as np

# Local imports
from model.soa.template import Algorithm


class StochasticFractalSearch(Algorithm):
    """Stochastic Fractal Search algorithm."""

    __slots__ = [
        "_n_population",
        "_n_iterations",
        "_fractal_factor",
        "_dim",
        "_verbose",
    ]

    def __init__(
        self,
        n_population: int = 50,
        n_iterations: int = 100,
        fractal_factor: float = 0.9,
        *,
        verbose: bool = False,
    ):
        """
        Initialize the Stochastic Fractal Search algorithm.

        Parameters:
        - population_size: int, number of particles in the population.
        - max_iterations: int, maximum number of iterations.
        - fractal_factor: float, factor controlling the fractal dimension (0 < fractal_factor <= 1).
        """
        self._n_population = n_population
        self._n_iterations = n_iterations
        self._fractal_factor = fractal_factor
        self._verbose = verbose

    def __init_population(
        self,
        bounds: Sequence[tuple[float, float]] | tuple[float, float],
        dimension: int,
    ) -> np.ndarray:
        """Initialize the population randomly within the bounds."""
        if isinstance(bounds, tuple):
            return np.array(
                [
                    np.random.uniform(bounds[0], bounds[1], size=dimension)
                    for _ in range(self._n_population)
                ]
            )
        # If not...
        return np.array(
            [
                [np.random.uniform(low, high, size=dimension) for low, high in bounds]
                for _ in range(self._n_population)
            ]
        )

    def __evaluate_population(
        self, population: np.ndarray, obj_func: Callable[[np.ndarray], float | int]
    ) -> np.ndarray:
        """Evaluate the objective function for each individual in the population."""
        return np.array([obj_func(ind) for ind in population])

    def perturb(
        self,
        population: np.ndarray,
        fitness,
        bounds: Sequence[tuple[float, float]] | tuple[float, float],
        dimension: int,
    ) -> np.ndarray:
        """Apply stochastic perturbation to the population based on their fitness values."""

        # Get the b_limits
        b_limits = [bounds] if isinstance(bounds, tuple) else bounds

        new_population = population.copy()
        for i in range(self._n_population):
            perturbation = self._fractal_factor * np.random.normal(size=dimension)
            new_population[i] += perturbation * (1.0 / (1.0 + fitness[i]))
            new_population[i] = np.clip(
                new_population[i], [b[0] for b in b_limits], [b[1] for b in b_limits]
            )
        return new_population

    def optimize(
        self,
        objective_fn: Callable[[np.ndarray], float | int],
        bounds: Sequence[tuple[float, float]] | tuple[float, float],
        dimension: int,
    ) -> tuple[float, np.ndarray]:
        """Perform the optimization using Stochastic Fractal Search."""
        # Initialize population
        population = self.__init_population(bounds, dimension)
        fitness = self.__evaluate_population(population, objective_fn)

        best_solution = population[np.argmin(fitness)]
        best_fitness = np.min(fitness)

        for iteration in range(self._n_iterations):
            # Perturb population
            population = self.perturb(population, fitness, bounds, dimension)
            fitness = self.__evaluate_population(population, objective_fn)

            # Update best solution
            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < best_fitness:
                best_solution = population[current_best_idx]
                best_fitness = fitness[current_best_idx]

            if self._verbose:
                print(
                    f"Iteration {iteration + 1}/{self._n_iterations}: Best Fitness = {best_fitness}"
                )

        return best_fitness, best_solution
