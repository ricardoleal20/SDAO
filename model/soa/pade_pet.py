"""
Dimension-improvements-based adaptation of control parameters in Differential Evolution.

Paper: Dimension improvements based adaptation of control parameters in Differential Evolution:
        A fitness-value-independent approach.
Link: https://www.sciencedirect.com/science/article/abs/pii/S0957417423003494
"""

from typing import Callable, Sequence, cast
import numpy as np

# Local imports
from model.soa.template import Algorithm, StepSolution


class PaDE_PET(Algorithm):  # pylint: disable=R0903
    """DE with dimension-wise, fitness-independent adaptation of F and CR.

    - Adapts per-dimension CR and F based on whether changes in each dimension
      participated in accepted trials (dimension improvements), avoiding dependence
      on the magnitude of fitness values.
    - Uses component-wise F (mutation factor) and CR (crossover probability).
    """

    _n_population: int
    _n_iterations: int
    _iterations: list[StepSolution]
    _verbose: bool

    # Per-dimension parameters
    _cr_dim: np.ndarray
    _f_dim: np.ndarray

    __slots__ = [
        "_n_population",
        "_n_iterations",
        "_iterations",
        "_verbose",
        "_cr_dim",
        "_f_dim",
    ]

    def __init__(
        self,
        n_population: int = 50,
        n_iterations: int = 500,
        *,
        verbose: bool = False,
    ) -> None:
        self._n_population = int(n_population)
        self._n_iterations = int(n_iterations)
        self._verbose = verbose
        self._iterations = []
        # Defer _cr_dim/_f_dim initialization until optimize() when dimension is known
        self._cr_dim = np.empty(0, dtype=float)
        self._f_dim = np.empty(0, dtype=float)

    # ====================================== #
    #              Public methods            #
    # ====================================== #
    def optimize(  # pylint: disable=R0914
        self,
        objective_fn: Callable[[np.ndarray], float | int],
        bounds: Sequence[tuple[float, float]] | tuple[float, float],
        dimension: int,
    ) -> tuple[float, np.ndarray]:
        """Optimize the objective function using DE with per-dimension adaptation."""
        self._iterations = []

        # Normalize per-dimension bounds
        fn_bounds: list[tuple[float, float]]
        if isinstance(bounds, tuple):
            # Disambiguate if it's a single (low, high) or a tuple of per-dimension bounds
            if (
                len(bounds) == 2
                and isinstance(bounds[0], (int, float))
                and isinstance(bounds[1], (int, float))
            ):
                low_f = float(bounds[0])
                high_f = float(bounds[1])
                fn_bounds = [(low_f, high_f) for _ in range(dimension)]
            else:
                seq_bounds = cast(list[tuple[float, float]], list(bounds))
                if len(seq_bounds) != dimension:
                    raise ValueError("Length of bounds must match dimension")
                fn_bounds = [(float(lb), float(ub)) for (lb, ub) in seq_bounds]
        else:
            if len(bounds) != dimension:
                raise ValueError("Length of bounds must match dimension")
            seq2 = cast(Sequence[tuple[float, float]], bounds)
            fn_bounds = [(float(lb), float(ub)) for (lb, ub) in seq2]

        low_vec = np.array([b[0] for b in fn_bounds], dtype=float)
        high_vec = np.array([b[1] for b in fn_bounds], dtype=float)

        # Initialize per-dimension CR and F
        # Start neutral, then adapt by dimension improvements
        self._cr_dim = np.full(dimension, 0.5, dtype=float)
        # Slightly higher F initially to encourage exploration in early stages
        self._f_dim = np.full(dimension, 0.7, dtype=float)

        # Initialize population uniformly
        rand_mat = np.random.rand(self._n_population, dimension)
        population = low_vec + rand_mat * (high_vec - low_vec)
        fitness = np.asarray([objective_fn(ind) for ind in population], dtype=float)

        best_idx = int(np.argmin(fitness))
        best_solution = population[best_idx].copy()
        best_fitness = float(fitness[best_idx])

        # Counters for dimension improvements per iteration
        attempt_counts = np.zeros(dimension, dtype=int)
        success_counts = np.zeros(dimension, dtype=int)

        for iteration in range(self._n_iterations):
            new_population = population.copy()
            new_fitness = fitness.copy()

            for i in range(self._n_population):
                # Select 3 distinct indices different from i
                idxs = list(range(self._n_population))
                idxs.remove(i)
                a_idx, b_idx, c_idx = np.random.choice(idxs, size=3, replace=False)
                a, b, c = population[a_idx], population[b_idx], population[c_idx]

                # Mutation: component-wise F
                mutant = a + self._f_dim * (b - c)
                mutant = np.clip(mutant, low_vec, high_vec)

                # Binomial crossover: component-wise CR
                cross_rand = np.random.rand(dimension)
                cross_mask = cross_rand < self._cr_dim
                # Ensure at least one dimension crosses
                if not np.any(cross_mask):
                    cross_mask[np.random.randint(0, dimension)] = True
                trial = np.where(cross_mask, mutant, population[i])

                # Clip trial to bounds
                trial = np.clip(trial, low_vec, high_vec)

                # Selection
                trial_fit = float(objective_fn(trial))
                # Update attempt counters for crossed dimensions
                attempt_counts += cross_mask.astype(int)
                if trial_fit <= fitness[i]:
                    new_population[i] = trial
                    new_fitness[i] = trial_fit
                    # Dimension improvements: any crossed dimension that is part of an accepted trial
                    success_counts += cross_mask.astype(int)

            # Commit new population
            population = new_population
            fitness = new_fitness

            # Update best
            current_best_idx = int(np.argmin(fitness))
            if fitness[current_best_idx] < best_fitness:
                best_fitness = float(fitness[current_best_idx])
                best_solution = population[current_best_idx].copy()

            # Adapt per-dimension parameters based on observed success ratios
            with np.errstate(divide="ignore", invalid="ignore"):
                success_ratio = np.divide(
                    success_counts.astype(float),
                    np.maximum(1, attempt_counts).astype(float),
                )

            # Move CR towards the empirical success ratio
            self._cr_dim = 0.7 * self._cr_dim + 0.3 * success_ratio
            self._cr_dim = np.clip(self._cr_dim, 0.05, 0.95)

            # Adjust F inversely to success (more success -> smaller F for exploitation)
            adjust = 1.0 + 0.6 * (0.5 - success_ratio)
            self._f_dim = np.clip(self._f_dim * adjust, 0.1, 1.0)

            # Reset counters for next iteration
            attempt_counts.fill(0)
            success_counts.fill(0)

            # Bookkeeping
            if self._verbose and (iteration % 10 == 0 or iteration == self._n_iterations - 1):
                print(f"Iteration {iteration} | Best fitness: {best_fitness}")
            self._iterations.append((iteration, best_fitness))

        return best_fitness, best_solution


