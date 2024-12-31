"""

"""
from typing import Callable, Sequence
import numpy as np
# Local imports
from model.soa.template import Algorithm


class PathRelinking(Algorithm):
    """Implementation of Path Relinking for large-scale global optimization."""
    _n_iterations: int
    _n_population: int
    _elite_ratio: float
    _verbose: bool

    __slots__ = [
        "_n_population", "_n_iterations",
        "_elite_ratio", "_verbose"
    ]

    def __init__(
        self,
        n_population: int = 50,
        n_iterations: int = 1000,
        elite_ratio: float = 0.2,
        verbose: bool = False
    ):
        """
        Initialize the optimizer.

        Parameters:
        - population_size: int, the size of the population.
        - max_iterations: int, the maximum number of iterations.
        - elite_ratio: float, the proportion of elite solutions to use for path relinking.
        """
        self._n_population = n_population
        self._n_iterations = n_iterations
        self._elite_ratio = elite_ratio
        self._verbose = verbose

    def optimize(  # pylint: disable=R0914
        self,
        objective_fn: Callable[[np.ndarray], float | int],
        bounds: Sequence[tuple[float, float]] | tuple[float, float],
        dimension: int
    ) -> tuple[float, np.ndarray]:
        """
        Perform optimization using Path Relinking.

        Parameters:
        - objective_fn: Callable, the objective function to minimize.
        - bounds: Sequence of tuples indicating the bounds for each dimension.
        - dimension: int, the number of dimensions.

        Returns:
        - tuple of best fitness and best solution found.
        """
        fn_bounds = [bounds] if isinstance(bounds, tuple) else bounds
        # Initialize population
        population = np.array([
            [np.random.uniform(low, high) for low, high in fn_bounds]
            for _ in range(self._n_population)
        ])
        fitness = np.array([objective_fn(ind) for ind in population])

        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        for iteration in range(self._n_iterations):
            # Select elite solutions
            elite_count = max(1, int(self._elite_ratio * self._n_population))
            elite_indices = np.argsort(fitness)[:elite_count]
            elites = population[elite_indices]

            # Path relinking between elites and other solutions
            new_population = population.copy()
            for i in range(self._n_population):
                if i not in elite_indices:
                    elite_partner = elites[np.random.randint(elite_count)]
                    new_solution = self.path_relinking(
                        population[i], elite_partner, fn_bounds)  # type: ignore
                    new_population[i] = new_solution

            # Evaluate new population
            fitness = np.array([objective_fn(ind) for ind in new_population])

            # Update best solution
            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < best_fitness:
                best_solution = new_population[current_best_idx]
                best_fitness = fitness[current_best_idx]

            population = new_population

            if self._verbose:
                print(
                    f"Iteration {iteration + 1}, Best Fitness: {best_fitness}")

        return best_fitness, best_solution

    def path_relinking(
        self,
        start: np.ndarray,
        target: np.ndarray,
        bounds: Sequence[tuple[float, float]]
    ) -> np.ndarray:
        """
        Perform path relinking between two solutions.

        Parameters:
        - start: ndarray, the starting solution.
        - target: ndarray, the target solution.
        - bounds: Sequence of tuples indicating the bounds for each dimension.

        Returns:
        - ndarray, the new solution generated.
        """
        step = np.random.uniform(0, 1)
        new_solution = start + step * (target - start)
        return np.clip(new_solution, [b[0] for b in bounds], [b[1] for b in bounds])
