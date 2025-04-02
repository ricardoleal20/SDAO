"""
Two Level PSO algorithm implementation.

Link: https://link.springer.com/article/10.1007/s11721-019-00167-w
"""

from typing import Callable, Sequence
import numpy as np

# Local imports
from model.soa.template import Algorithm, StepSolution


class TLPSO(Algorithm):  # pylint: disable=R0903
    """Implementation of Two-Level Particle Swarm Optimization (TLPSO)."""

    _verbose: bool
    _global_swarm_size: int
    _local_swarm_size: int
    _n_iterations: int
    _iterations: list[StepSolution]

    __slots__ = [
        "_global_swarm_size",
        "_local_swarm_size",
        "_n_iterations",
        "_verbose",
        "_iterations",
    ]

    def __init__(
        self,
        global_swarm_size: int = 10,
        local_swarm_size: int = 5,
        max_iterations: int = 1000,
        *,
        verbose: bool = False,
    ):
        """Initialize the optimizer.

        Parameters:
        - global_swarm_size: int, number of particles in the global swarm.
        - local_swarm_size: int, number of particles in each local swarm.
        - max_iterations: int, maximum number of iterations.
        """
        self._global_swarm_size = global_swarm_size
        self._local_swarm_size = local_swarm_size
        self._n_iterations = max_iterations
        self._verbose = verbose
        self._iterations = []

    def optimize(
        self,
        objective_fn: Callable[[np.ndarray], float | int],
        bounds: Sequence[tuple[float, float]] | tuple[float, float],
        dimension: int,
    ) -> tuple[float, np.ndarray]:
        """
        Perform optimization using TLPSO.

        Parameters:
        - objective_fn: Callable, the objective function to minimize.
        - bounds: Sequence of tuples indicating the bounds for each dimension.
        - dimension: int, the number of dimensions.

        Returns:
        - tuple of best fitness and best solution found.
        """
        if not isinstance(bounds, tuple):
            raise NotImplementedError(
                "TLPSO only supports a single range for all dimensions."
            )
        self._iterations = []
        # Initialize global swarm
        global_positions = np.array(
            [
                np.random.uniform(bounds[0], bounds[1], size=dimension)
                for _ in range(self._global_swarm_size)
            ]
        )
        global_velocities = np.zeros((self._global_swarm_size, dimension))
        global_best_position = np.zeros(dimension)
        global_best_fitness = np.inf

        # Initialize local swarms
        local_swarms = [
            {
                "positions": np.array(
                    [
                        np.random.uniform(bounds[0], bounds[1], size=dimension)
                        for _ in range(self._local_swarm_size)
                    ]
                ),
                "velocities": np.zeros((self._local_swarm_size, dimension)),
                "best_positions": None,
                "best_fitness": np.inf,
            }
            for _ in range(self._global_swarm_size)
        ]

        for iteration in range(self._n_iterations):
            # Update local swarms
            for idx, local_swarm in enumerate(local_swarms):
                positions = local_swarm["positions"]
                velocities = local_swarm["velocities"]
                fitness = np.array([objective_fn(pos) for pos in positions])

                # Update local bests
                for i in range(self._local_swarm_size):
                    if fitness[i] < local_swarm["best_fitness"]:
                        local_swarm["best_positions"] = positions[i]
                        local_swarm["best_fitness"] = fitness[i]

                # Update velocities and positions
                for i in range(self._local_swarm_size):
                    inertia = 0.5 * velocities[i]
                    cognitive = (
                        2.0
                        * np.random.rand(dimension)
                        * (local_swarm["best_positions"] - positions[i])
                    )
                    social = (
                        2.0
                        * np.random.rand(dimension)
                        * (global_positions[idx] - positions[i])
                    )
                    velocities[i] = inertia + cognitive + social
                    positions[i] += velocities[i]
                    positions[i] = np.clip(positions[i], bounds[0], bounds[1])

                local_swarm["positions"] = positions

                # Update global best for this swarm
                if local_swarm["best_fitness"] < global_best_fitness:
                    global_best_fitness = local_swarm["best_fitness"]
                    global_best_position = local_swarm["best_positions"]

            # Update global swarm
            for i in range(self._global_swarm_size):
                # global_fitness = objective_fn(global_positions[i])

                # Update velocities and positions
                inertia = 0.5 * global_velocities[i]
                cognitive = (
                    2.0
                    * np.random.rand(dimension)
                    * (global_best_position - global_positions[i])
                )
                global_velocities[i] = inertia + cognitive
                global_positions[i] += global_velocities[i]
                global_positions[i] = np.clip(global_positions[i], bounds[0], bounds[1])

            if self._verbose:
                print(
                    f"Iteration {iteration + 1}, Global Best Fitness: {global_best_fitness}"
                )
            self._iterations.append((iteration, global_best_fitness))

        return global_best_fitness, global_best_position
