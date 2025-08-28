"""
Implementation of the Gyro Fireworks Algorithm (GFA).

Paper: https://pubs.aip.org/aip/adv/article/14/8/085210/3306702/Gyro-fireworks-algorithm-A-new-metaheuristic#
"""

from typing import Callable, Sequence, cast
import numpy as np

# Local imports
from model.soa.template import Algorithm, StepSolution


class GFA(Algorithm):  # pylint: disable=R0902, R0903
    """Gyro Fireworks Algorithm (GFA).

    High-level mechanics (concise):
    - Maintain a set of fireworks (agents) and generate explosion sparks around them
    - Better fireworks generate more sparks with smaller amplitude; worse fireworks
      generate fewer sparks with larger amplitude
    - Apply a gyro-rotation operator that rotates spark vectors around the global best
      using a decaying angle to progressively bias exploitation
    - Optionally apply Gaussian mutation to some sparks to increase diversity
    - Select the best fireworks for the next iteration from the union of parents+sparks
    """

    _n_fireworks: int
    _n_iterations: int
    _min_sparks: int
    _max_sparks: int
    _base_amplitude: float
    _gyro_decay: float
    _gaussian_mutation_prob: float
    _iterations: list[StepSolution]
    _verbose: bool

    __slots__ = [
        "_n_fireworks",
        "_n_iterations",
        "_min_sparks",
        "_max_sparks",
        "_base_amplitude",
        "_gyro_decay",
        "_gaussian_mutation_prob",
        "_iterations",
        "_verbose",
    ]

    def __init__(
        self,
        n_fireworks: int = 25,
        n_iterations: int = 500,
        *,
        min_sparks: int = 5,
        max_sparks: int = 40,
        base_amplitude: float = 0.2,
        gyro_decay: float = 0.985,
        gaussian_mutation_prob: float = 0.1,
        verbose: bool = False,
    ) -> None:
        self._n_fireworks = int(n_fireworks)
        self._n_iterations = int(n_iterations)
        self._min_sparks = int(min_sparks)
        self._max_sparks = int(max_sparks)
        self._base_amplitude = float(base_amplitude)
        self._gyro_decay = float(gyro_decay)
        self._gaussian_mutation_prob = float(gaussian_mutation_prob)
        self._verbose = verbose
        self._iterations = []

    # ====================================== #
    #              Public methods            #
    # ====================================== #
    def optimize(  # pylint: disable=R0914
        self,
        objective_fn: Callable[[np.ndarray], float | int | np.signedinteger],
        bounds: Sequence[tuple[float, float]] | tuple[float, float],
        dimension: int,
    ) -> tuple[float, np.ndarray]:
        """Optimize the objective function using GFA.

        Returns the best fitness and best solution found.
        """
        self._iterations = []

        # Normalize bounds into per-dimension list
        fn_bounds: list[tuple[float, float]]
        if isinstance(bounds, tuple):
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
            fn_bounds = [(float(lb), float(ub)) for (lb, ub) in bounds]

        low_vec = np.array([b[0] for b in fn_bounds], dtype=float)
        high_vec = np.array([b[1] for b in fn_bounds], dtype=float)
        span_vec = high_vec - low_vec

        # Initialize fireworks uniformly within bounds
        rand_mat = np.random.rand(self._n_fireworks, dimension)
        fireworks = low_vec + rand_mat * span_vec
        fitness = np.asarray([objective_fn(ind) for ind in fireworks], dtype=float)

        best_idx = int(np.argmin(fitness))
        best_solution = fireworks[best_idx].copy()
        best_fitness = float(fitness[best_idx])

        # Initial gyro angle (radians); decays over iterations
        gyro_angle = 1.0  # ~57 degrees initial rotation

        for iteration in range(self._n_iterations):
            # Compute dynamic sparks and amplitudes per firework
            f_min = float(np.min(fitness))
            f_max = float(np.max(fitness))
            denom = max(1e-12, f_max - f_min)

            # More sparks for better fireworks; amplitude smaller for better
            ranks = (f_max - fitness) / denom  # in [0, 1]
            sparks_counts = np.round(
                self._min_sparks + ranks * (self._max_sparks - self._min_sparks)
            ).astype(int)
            amplitudes = self._base_amplitude * (fitness - f_min) / denom
            amplitudes = np.clip(amplitudes, 0.05 * self._base_amplitude, self._base_amplitude)

            # Generate sparks
            all_candidates = [*fireworks]
            all_fitness = [*fitness]

            for i, center in enumerate(fireworks):
                num_sparks = max(self._min_sparks, int(sparks_counts[i]))
                amp = float(amplitudes[i])
                amp_vec = amp * span_vec

                # Vectorized spark generation per firework
                noise = np.random.uniform(-1.0, 1.0, size=(num_sparks, dimension))
                sparks = center + noise * amp_vec

                # Gyro rotation around global best
                if best_solution is not None:  # mypy appeasement
                    sparks = self.__gyro_rotate_points(sparks, best_solution, gyro_angle)

                # Gaussian mutation (with probability per spark)
                if self._gaussian_mutation_prob > 0.0:
                    mask = np.random.rand(num_sparks) < self._gaussian_mutation_prob
                    if np.any(mask):
                        sigma = 0.05 * span_vec
                        gauss = np.random.normal(0.0, 1.0, size=(mask.sum(), dimension))
                        sparks[mask] = sparks[mask] + gauss * sigma

                # Clip sparks to bounds
                sparks = np.clip(sparks, low_vec, high_vec)

                # Evaluate and collect
                sparks_fit = [float(objective_fn(s)) for s in sparks]
                all_candidates.extend(list(sparks))
                all_fitness.extend(sparks_fit)

            # Selection: keep the best distinct fireworks for next iteration
            all_candidates_arr = np.asarray(all_candidates, dtype=float)
            all_fitness_arr = np.asarray(all_fitness, dtype=float)
            idx_sorted = np.argsort(all_fitness_arr)
            selected = all_candidates_arr[idx_sorted][: self._n_fireworks]
            selected_fit = all_fitness_arr[idx_sorted][: self._n_fireworks]

            fireworks = selected
            fitness = selected_fit

            # Update best
            current_best_idx = int(np.argmin(fitness))
            if fitness[current_best_idx] < best_fitness:
                best_fitness = float(fitness[current_best_idx])
                best_solution = fireworks[current_best_idx].copy()

            # Mild opposition on the current worst to re-diversify
            worst_idx = int(np.argmax(fitness))
            opposite = self.__opposite_of(fireworks[worst_idx], fn_bounds)
            opposite = np.clip(opposite, low_vec, high_vec)
            opp_fit = float(objective_fn(opposite))
            if opp_fit < fitness[worst_idx]:
                fireworks[worst_idx] = opposite
                fitness[worst_idx] = opp_fit
                if opp_fit < best_fitness:
                    best_fitness = opp_fit
                    best_solution = opposite.copy()

            # Bookkeeping and schedule
            if self._verbose and (iteration % 10 == 0 or iteration == self._n_iterations - 1):
                print(f"Iteration {iteration} | Best fitness: {best_fitness}")
            self._iterations.append((iteration, best_fitness))

            # Decay the gyro angle to increase exploitation over time
            gyro_angle *= self._gyro_decay

        return best_fitness, best_solution

    def trajectory(self) -> list[StepSolution]:
        return self._iterations

    # ====================================== #
    #             Private methods            #
    # ====================================== #
    def __gyro_rotate_points(
        self, points: np.ndarray, center: np.ndarray, angle: float
    ) -> np.ndarray:
        """Rotate each point around `center` by `angle` in a high-dimensional sense.

        The rotation is performed by mixing the radial vector with a random perpendicular
        vector. For each point p: v = p - center, u = v/||v||. Sample random r and build a
        perpendicular direction w = r - (r·u)u. Then: v' = cos(a)*v + sin(a)*||v||*w/||w||.
        """
        rotated = points.copy()
        if angle == 0.0:
            return rotated

        diffs = rotated - center  # shape: (n, d)
        norms = np.linalg.norm(diffs, axis=1)
        # Avoid division by zero by leaving points with ~zero radius unchanged
        nonzero_mask = norms > 1e-12
        if not np.any(nonzero_mask):
            return rotated

        idxs = np.where(nonzero_mask)[0]
        u = diffs[idxs] / norms[idxs][:, None]
        r = np.random.normal(0.0, 1.0, size=u.shape)
        # Remove projection of r onto u to build a perpendicular component
        proj = (np.sum(r * u, axis=1)[:, None]) * u
        w = r - proj
        w_norms = np.linalg.norm(w, axis=1)

        # For rare degenerate cases, resample to avoid zero vector
        deg_mask = w_norms <= 1e-12
        if np.any(deg_mask):
            w[deg_mask] = np.random.normal(0.0, 1.0, size=(int(np.sum(deg_mask)), u.shape[1]))
            proj2 = (np.sum(w[deg_mask] * u[deg_mask], axis=1)[:, None]) * u[deg_mask]
            w[deg_mask] = w[deg_mask] - proj2
            w_norms[deg_mask] = np.linalg.norm(w[deg_mask], axis=1)

        w_unit = w / w_norms[:, None]

        ca = np.cos(angle)
        sa = np.sin(angle)
        rotated[idxs] = (
            center
            + ca * diffs[idxs]
            + sa * (norms[idxs][:, None] * w_unit)
        )
        return rotated

    def __opposite_of(
        self, position: np.ndarray, bounds: Sequence[tuple[float, float]]
    ) -> np.ndarray:
        opposite = np.empty_like(position)
        for j, (lb, ub) in enumerate(bounds):
            opposite[j] = lb + ub - position[j]
        return opposite
