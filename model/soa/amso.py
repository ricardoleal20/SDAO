"""
AMSO: Adaptive Multi-Source Optimization

Link: https://dl.acm.org/doi/10.1145/3321707.3321713
"""

from typing import Callable, Sequence
import numpy as np

# Local imports
from model.soa.template import Algorithm, StepSolution


class AMSO(Algorithm):  # pylint: disable=R0903
    """Implementation of Adaptive Multi-Swarm Optimization (AMSO)."""

    _n_iterations: int
    _num_swarms: int
    _swarm_size: int
    _verbose: bool
    _iterations: list[StepSolution]

    __slots__ = [
        "_num_swarms",
        "_swarm_size",
        "_n_iterations",
        "_verbose",
        "_iterations",
    ]

    def __init__(
        self,
        num_swarms: int = 5,
        swarm_size: int = 20,
        n_iterations: int = 1000,
        *,
        verbose: bool = False,
    ):
        """
        Initialize the optimizer.

        Parameters:
        - num_swarms: int, the number of swarms.
        - swarm_size: int, the size of each swarm.
        - n_iterations: int, the maximum number of iterations.
        """
        self._num_swarms = num_swarms
        self._swarm_size = swarm_size
        self._n_iterations = n_iterations
        self._verbose = verbose
        self._iterations = []

    def optimize(  # pylint: disable=R0914
        self,
        objective_fn: Callable[[np.ndarray], float | int],
        bounds: Sequence[tuple[float, float]] | tuple[float, float],
        dimension: int,
    ) -> tuple[float, np.ndarray]:
        """
        Perform optimization using AMSO.

        Parameters:
        - objective_fn: Callable, the objective function to minimize.
        - bounds: Sequence of tuples indicating the bounds for each dimension.
        - dimension: int, the number of dimensions.

        Returns:
        - tuple of best fitness and best solution found.
        """
        if not isinstance(bounds, tuple):
            raise NotImplementedError(
                "The AMSO algorithm only supports uniform bounds, not dinamic per dimension."
            )
        # Initialize swarms
        swarms = [
            {
                "positions": np.array(
                    [
                        np.random.uniform(bounds[0], bounds[1], size=dimension)
                        for _ in range(self._swarm_size)
                    ]
                ),
                "velocities": np.zeros((self._swarm_size, dimension)),
                "best_positions": None,
                "best_fitness": np.inf,
            }
            for _ in range(self._num_swarms)
        ]

        global_best_position = np.zeros(dimension)
        global_best_fitness = np.inf

        for iteration in range(self._n_iterations):
            for swarm in swarms:
                positions = swarm["positions"]
                velocities = swarm["velocities"]
                fitness = np.array([objective_fn(pos) for pos in positions])

                # Update swarm bests
                for i in range(self._swarm_size):
                    if fitness[i] < swarm["best_fitness"]:
                        swarm["best_positions"] = positions[i]
                        swarm["best_fitness"] = fitness[i]

                # Update global best
                swarm_best_fitness = swarm["best_fitness"]
                if swarm_best_fitness < global_best_fitness:
                    global_best_position = swarm["best_positions"]
                    global_best_fitness = swarm_best_fitness

                # Update velocities and positions
                for i in range(self._swarm_size):
                    inertia = 0.5 * velocities[i]
                    cognitive = (
                        2.0
                        * np.random.rand(dimension)
                        * (swarm["best_positions"] - positions[i])
                    )
                    social = (
                        2.0
                        * np.random.rand(dimension)
                        * (global_best_position - positions[i])
                    )
                    velocities[i] = inertia + cognitive + social
                    positions[i] += velocities[i]
                    positions[i] = np.clip(positions[i], bounds[0], bounds[1])

                swarm["positions"] = positions
            if self._verbose:
                print(
                    f"Iteration {iteration + 1}, Global Best Fitness: {global_best_fitness}"
                )
            # Append the iteration data
            self._iterations.append((iteration, global_best_fitness))

        return global_best_fitness, global_best_position
