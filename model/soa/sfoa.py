"""
Implementation of the Starfish Optimization Algorithm (SFOA).

Link: https://link.springer.com/article/10.1007/s00521-024-10694-1
"""

from typing import Callable, Sequence, cast
import numpy as np

# Local imports
from model.soa.template import Algorithm, StepSolution


class SFOA(Algorithm):  # pylint: disable=R0903
    """Starfish Optimization Algorithm (SFOA)."""

    _n_population: int
    _n_iterations: int
    _exploration_rate: float
    _spiral_rate: float
    _regeneration_prob: float
    _iterations: list[StepSolution]
    _verbose: bool

    __slots__ = [
        "_n_population",
        "_n_iterations",
        "_exploration_rate",
        "_spiral_rate",
        "_regeneration_prob",
        "_iterations",
        "_verbose",
    ]

    def __init__(
        self,
        n_population: int = 50,
        n_iterations: int = 500,
        exploration_rate: float = 0.6,
        spiral_rate: float = 0.3,
        regeneration_prob: float = 0.05,
        *,
        verbose: bool = False,
    ) -> None:
        """Initialize the SFOA optimizer.

        Parameters:
        - n_population: Number of starfish (agents) in the population
        - n_iterations: Maximum number of iterations
        - exploration_rate: Probability of applying exploration/foraging moves
        - spiral_rate: Probability of applying spiral (attacking) moves
        - regeneration_prob: Probability of regenerating a stagnated starfish
        - verbose: Enable per-iteration progress logging
        """
        self._n_population = int(n_population)
        self._n_iterations = int(n_iterations)
        self._exploration_rate = float(exploration_rate)
        self._spiral_rate = float(spiral_rate)
        self._regeneration_prob = float(regeneration_prob)
        self._verbose = verbose
        self._iterations = []

    # ====================================== #
    #              Public methods            #
    # ====================================== #
    def optimize(  # pylint: disable=R0914
        self,
        objective_fn: Callable[[np.ndarray], float | int],
        bounds: Sequence[tuple[float, float]] | tuple[float, float],
        dimension: int,
    ) -> tuple[float, np.ndarray]:
        """Optimize the objective function using SFOA.

        Returns the best fitness and best solution found.
        """
        self._iterations = []

        # Normalize bounds into a per-dimension list for uniform handling
        fn_bounds: list[tuple[float, float]]
        if isinstance(bounds, tuple):
            # Disambiguate: could be (low, high) or tuple of per-dimension bounds
            if (
                len(bounds) == 2
                and isinstance(bounds[0], (int, float))
                and isinstance(bounds[1], (int, float))
            ):
                low_f = float(bounds[0])
                high_f = float(bounds[1])
                fn_bounds = [(low_f, high_f) for _ in range(dimension)]
            else:
                # Treat as sequence of per-dimension bounds
                seq_bounds = cast(list[tuple[float, float]], list(bounds))
                if len(seq_bounds) != dimension:
                    raise ValueError("Length of bounds must match dimension")
                fn_bounds = [(float(low), float(high)) for (low, high) in seq_bounds]
        else:
            if len(bounds) != dimension:
                raise ValueError("Length of bounds must match dimension")
            fn_bounds = [(float(low), float(high)) for (low, high) in bounds]

        # Initialize population uniformly within bounds
        low_vec = np.array([b[0] for b in fn_bounds], dtype=float)
        high_vec = np.array([b[1] for b in fn_bounds], dtype=float)
        rand_mat = np.random.rand(self._n_population, dimension)
        population = low_vec + rand_mat * (high_vec - low_vec)  # shape: (n_population, dim)

        fitness = np.asarray([objective_fn(ind) for ind in population], dtype=float)
        best_idx = int(np.argmin(fitness))
        best_solution = population[best_idx].copy()
        best_fitness = float(fitness[best_idx])

        # Track stagnation to enable regeneration
        stagnation_counters = np.zeros(self._n_population, dtype=int)
        last_improvement_fitness = fitness.copy()

        for iteration in range(self._n_iterations):
            new_population = population.copy()
            new_fitness = fitness.copy()

            # Adaptive step scales
            exploration_scale = 0.6 * (1.0 - iteration / max(1, self._n_iterations - 1))
            exploitation_scale = 0.2 + 0.4 * (iteration / max(1, self._n_iterations - 1))

            for i in range(self._n_population):
                current = population[i]

                # Movement selection
                r_move = np.random.rand()
                candidate = current.copy()

                if r_move < self._exploration_rate:
                    # Foraging/exploration: roam towards random peer and random direction
                    peer_idx = self.__rand_index_excluding(i)
                    peer = population[peer_idx]
                    random_direction = np.random.uniform(-1.0, 1.0, size=dimension)
                    candidate = self.__forage_move(
                        current,
                        peer,
                        best_solution,
                        random_direction,
                        exploration_scale,
                    )
                elif r_move < self._exploration_rate + self._spiral_rate:
                    # Spiral/attacking: orbit around best and shrink radius over time
                    candidate = self.__spiral_move(
                        current,
                        best_solution,
                        exploitation_scale,
                    )
                else:
                    # Aggregation/exploitation: move towards best with local randomization
                    candidate = self.__aggregate_move(
                        current,
                        best_solution,
                        exploitation_scale,
                    )

                # Regeneration for stagnated or by probability
                if (
                    np.random.rand() < self._regeneration_prob
                    or stagnation_counters[i] > max(10, self._n_iterations // 20)
                ):
                    candidate = self.__regenerate_within_bounds(fn_bounds, dimension)

                # Enforce bounds
                candidate = self.__clip_to_bounds(candidate, fn_bounds)

                # Greedy selection
                cand_fit = float(objective_fn(candidate))
                if cand_fit <= fitness[i]:
                    new_population[i] = candidate
                    new_fitness[i] = cand_fit
                    if cand_fit + 1e-12 < last_improvement_fitness[i]:
                        stagnation_counters[i] = 0
                        last_improvement_fitness[i] = cand_fit
                    else:
                        stagnation_counters[i] += 1
                else:
                    stagnation_counters[i] += 1

            # Update global best
            population = new_population
            fitness = new_fitness
            current_best_idx = int(np.argmin(fitness))
            if fitness[current_best_idx] < best_fitness:
                best_fitness = float(fitness[current_best_idx])
                best_solution = population[current_best_idx].copy()

            # Opposition-based attempt on worst individual to encourage exploration
            worst_idx = int(np.argmax(fitness))
            opposite = self.__opposite_of(population[worst_idx], fn_bounds)
            opposite = self.__clip_to_bounds(opposite, fn_bounds)
            opp_fit = float(objective_fn(opposite))
            if opp_fit < fitness[worst_idx]:
                population[worst_idx] = opposite
                fitness[worst_idx] = opp_fit
                if opp_fit < best_fitness:
                    best_fitness = opp_fit
                    best_solution = opposite.copy()

            # Iteration bookkeeping
            if self._verbose and (iteration % 10 == 0 or iteration == self._n_iterations - 1):
                print(f"Iteration {iteration} | Best fitness: {best_fitness}")
            self._iterations.append((iteration, best_fitness))

        return best_fitness, best_solution

    def trajectory(self) -> list[StepSolution]:
        return self._iterations

    # ====================================== #
    #             Private methods            #
    # ====================================== #
    def __forage_move(
        self,
        current: np.ndarray,
        peer: np.ndarray,
        best: np.ndarray,
        random_direction: np.ndarray,
        scale: float,
    ) -> np.ndarray:
        """Foraging/exploration move.

        Combines a move towards a random peer, a random roaming component, and a mild pull to best.
        """
        w_peer = np.random.uniform(0.5, 1.0)
        w_roam = np.random.uniform(0.5, 1.0)
        w_best = np.random.uniform(0.1, 0.4)
        step = (
            w_peer * (peer - current)
            + w_roam * random_direction
            + w_best * (best - current)
        )
        return current + scale * step

    def __spiral_move(
        self,
        current: np.ndarray,
        center: np.ndarray,
        scale: float,
    ) -> np.ndarray:
        """Spiral/attacking move around the best solution.

        Implements a 2D-equivalent spiral projected independently per dimension using random angles.
        """
        direction = current - center
        radius = np.linalg.norm(direction) + 1e-12
        unit = direction / radius
        theta = np.random.uniform(-np.pi, np.pi, size=current.shape)
        spiral_component = unit * np.cos(theta) + unit * np.sin(theta)
        shrink = np.random.uniform(0.6, 0.95)
        return center + shrink * direction + scale * spiral_component

    def __aggregate_move(
        self,
        current: np.ndarray,
        best: np.ndarray,
        scale: float,
    ) -> np.ndarray:
        """Aggregation/exploitation move towards the best with local randomization."""
        local_random = np.random.normal(0.0, 1.0, size=current.shape)
        return current + scale * (best - current) + 0.05 * scale * local_random

    def __regenerate_within_bounds(
        self, bounds: Sequence[tuple[float, float]], dimension: int
    ) -> np.ndarray:
        """Generate a random position within the given bounds."""
        return np.array([np.random.uniform(low, high) for (low, high) in bounds])

    def __clip_to_bounds(
        self, position: np.ndarray, bounds: Sequence[tuple[float, float]]
    ) -> np.ndarray:
        clipped = position.copy()
        for j, (low, high) in enumerate(bounds):
            clipped[j] = float(np.clip(clipped[j], low, high))
        return clipped

    def __opposite_of(
        self, position: np.ndarray, bounds: Sequence[tuple[float, float]]
    ) -> np.ndarray:
        opposite = np.empty_like(position)
        for j, (low, high) in enumerate(bounds):
            opposite[j] = low + high - position[j]
        return opposite

    def __rand_index_excluding(self, excluded: int) -> int:
        idx = excluded
        while idx == excluded:
            idx = int(np.random.randint(0, self._n_population))
        return idx
