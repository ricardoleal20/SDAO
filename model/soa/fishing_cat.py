"""
Fishing Cat Optimizer (FCO).

Paper: Fishing Cat Optimizer – a novel metaheuristic.
Link: https://www.emerald.com/ec/article-abstract/42/2/780/1244283/Fishing-cat-optimizer-a-novel-metaheuristic
"""

from typing import Callable, Sequence, cast
import numpy as np

# Local imports
from model.soa.template import Algorithm, StepSolution


class FCO(Algorithm):  # pylint: disable=R0903
    """Fishing Cat Optimizer (FCO).

    Implements four characteristic phases inspired by fishing cat hunting behavior:
    - Ambush (cautious local observation)
    - Detection (exploratory scan)
    - Dive (fast attacking exploitation)
    - Trapping (local consolidation around best)
    """

    _n_population: int
    _n_iterations: int
    _p_ambush: float
    _p_dive: float
    _p_trap: float
    _iterations: list[StepSolution]
    _verbose: bool

    __slots__ = [
        "_n_population",
        "_n_iterations",
        "_p_ambush",
        "_p_dive",
        "_p_trap",
        "_iterations",
        "_verbose",
    ]

    def __init__(
        self,
        n_population: int = 50,
        n_iterations: int = 500,
        ambush_prob: float = 0.25,
        dive_prob: float = 0.35,
        trap_prob: float = 0.20,
        *,
        verbose: bool = False,
    ) -> None:
        self._n_population = int(n_population)
        self._n_iterations = int(n_iterations)
        self._p_ambush = float(ambush_prob)
        self._p_dive = float(dive_prob)
        self._p_trap = float(trap_prob)
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
        self._iterations = []

        # Normalize per-dimension bounds
        fn_bounds: list[tuple[float, float]]
        if isinstance(bounds, tuple):
            # Disambiguate between global (low, high) and per-dimension tuple of tuples
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
            seq2 = cast(Sequence[tuple[float, float]] , bounds)
            fn_bounds = [(float(lb), float(ub)) for (lb, ub) in seq2]

        low_vec = np.array([b[0] for b in fn_bounds], dtype=float)
        high_vec = np.array([b[1] for b in fn_bounds], dtype=float)

        # Initialize population uniformly
        rand_mat = np.random.rand(self._n_population, dimension)
        population = low_vec + rand_mat * (high_vec - low_vec)
        fitness = np.asarray([objective_fn(ind) for ind in population], dtype=float)

        best_idx = int(np.argmin(fitness))
        best_solution = population[best_idx].copy()
        best_fitness = float(fitness[best_idx])

        # Phase scheduling: decrease exploration and increase exploitation over time
        for iteration in range(self._n_iterations):
            new_population = population.copy()
            new_fitness = fitness.copy()

            frac = iteration / max(1, self._n_iterations - 1)
            exploration_scale = 0.6 * (1.0 - frac)
            exploitation_scale = 0.2 + 0.6 * frac

            for i in range(self._n_population):
                current = population[i]
                r = np.random.rand()
                candidate = current.copy()

                if r < self._p_ambush:
                    # Ambush: cautious local observation around best and current
                    candidate = self.__ambush_move(
                        current, best_solution, exploitation_scale
                    )
                elif r < self._p_ambush + self._p_dive:
                    # Dive: fast approach towards best (exploit)
                    candidate = self.__dive_move(
                        current, best_solution, exploitation_scale
                    )
                elif r < self._p_ambush + self._p_dive + self._p_trap:
                    # Trapping: local consolidation near best within a shrinking radius
                    candidate = self.__trap_move(
                        current, best_solution, fn_bounds, exploitation_scale
                    )
                else:
                    # Detection: exploratory scanning using peer guidance and random roaming
                    peer_idx = self.__rand_index_excluding(i)
                    peer = population[peer_idx]
                    random_dir = np.random.uniform(-1.0, 1.0, size=dimension)
                    candidate = self.__detect_move(
                        current, peer, random_dir, exploration_scale
                    )

                # Clip to bounds
                candidate = np.clip(candidate, low_vec, high_vec)

                # Greedy selection
                cand_fit = float(objective_fn(candidate))
                if cand_fit <= fitness[i]:
                    new_population[i] = candidate
                    new_fitness[i] = cand_fit

            # Update population
            population = new_population
            fitness = new_fitness

            # Update best
            current_best_idx = int(np.argmin(fitness))
            if fitness[current_best_idx] < best_fitness:
                best_fitness = float(fitness[current_best_idx])
                best_solution = population[current_best_idx].copy()

            # Optional: mild opposition on worst to encourage exploration
            worst_idx = int(np.argmax(fitness))
            opposite = self.__opposite_of(population[worst_idx], fn_bounds)
            opposite = np.clip(opposite, low_vec, high_vec)
            opp_fit = float(objective_fn(opposite))
            if opp_fit < fitness[worst_idx]:
                population[worst_idx] = opposite
                fitness[worst_idx] = opp_fit
                if opp_fit < best_fitness:
                    best_fitness = opp_fit
                    best_solution = opposite.copy()

            if self._verbose and (iteration % 10 == 0 or iteration == self._n_iterations - 1):
                print(f"Iteration {iteration} | Best fitness: {best_fitness}")
            self._iterations.append((iteration, best_fitness))

        return best_fitness, best_solution

    # ====================================== #
    #             Private methods            #
    # ====================================== #
    def __ambush_move(
        self, current: np.ndarray, best: np.ndarray, scale: float
    ) -> np.ndarray:
        # Small local move biased to best
        jitter = np.random.normal(0.0, 0.1, size=current.shape)
        return current + 0.3 * scale * (best - current) + 0.05 * jitter

    def __detect_move(
        self,
        current: np.ndarray,
        peer: np.ndarray,
        random_dir: np.ndarray,
        scale: float,
    ) -> np.ndarray:
        # Exploratory scan using peer guidance and roaming
        w_peer = np.random.uniform(0.5, 1.0)
        w_roam = np.random.uniform(0.5, 1.0)
        step = w_peer * (peer - current) + w_roam * random_dir
        return current + scale * step

    def __dive_move(
        self, current: np.ndarray, best: np.ndarray, scale: float
    ) -> np.ndarray:
        # Fast attacking move towards best with possible overshoot
        direction = best - current
        gain = np.random.uniform(1.0, 2.0)
        noise = np.random.normal(0.0, 0.05, size=current.shape)
        return current + gain * scale * direction + noise

    def __trap_move(
        self,
        current: np.ndarray,
        best: np.ndarray,
        bounds: Sequence[tuple[float, float]],
        scale: float,
    ) -> np.ndarray:
        # Sample within a shrinking hyper-rectangle around best
        radius = 0.2 * (1.0 - 0.5 * scale)
        sampled = np.empty_like(current)
        for j, (lb, ub) in enumerate(bounds):
            span = (ub - lb) * radius
            lo = max(lb, best[j] - span)
            hi = min(ub, best[j] + span)
            sampled[j] = np.random.uniform(lo, hi)
        # Blend with current to avoid abrupt jumps
        return 0.5 * sampled + 0.5 * (current + scale * (best - current))

    def __opposite_of(
        self, position: np.ndarray, bounds: Sequence[tuple[float, float]]
    ) -> np.ndarray:
        opposite = np.empty_like(position)
        for j, (lb, ub) in enumerate(bounds):
            opposite[j] = lb + ub - position[j]
        return opposite

    def __rand_index_excluding(self, excluded: int) -> int:
        idx = excluded
        while idx == excluded:
            idx = int(np.random.randint(0, self._n_population))
        return idx


