"""
Include the Algebraic Stochastic Gradient Descent (ASGD) optimizer.
The ASGD optimizer is a variant of the SGD optimizer that uses an algebraic
approach to optimize the learning rate.

Link: https://arxiv.org/abs/2204.05923
"""

from typing import Callable, Sequence
import numpy as np

# Local imports
from model.soa.template import Algorithm


class AlgebraicSGD(Algorithm):  # pylint: disable=R0903
    """Implementation of Algebraically Converging Stochastic Gradient Descent."""

    learning_rate: float
    n_iterations: int
    _verbose: bool

    __slots__ = ["learning_rate", "n_iterations", "_verbose"]

    def __init__(
        self,
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
        *,
        verbose: bool = False,
    ):
        """
        Initialize the optimizer.

        Parameters:
        - learning_rate: float, initial learning rate.
        - max_iterations: int, maximum number of iterations.
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self._verbose = verbose

    def optimize(
        self,
        objective_fn: Callable[[np.ndarray], float | int],
        bounds: Sequence[tuple[float, float]] | tuple[float, float],
        dimension: int,
    ) -> tuple[float, np.ndarray]:
        """
        Perform optimization using the Algebraically Converging SGD method.

        Parameters:
        - objective_fn: Callable, the objective function to minimize.
        - bounds: Sequence of tuples indicating the bounds for each dimension.
        - dimension: int, the number of dimensions.

        Returns:
        - tuple of best fitness and best solution found.
        """
        fn_bounds = [bounds] if isinstance(bounds, tuple) else bounds

        # Initialize position randomly within bounds
        position = np.array(
            [np.random.uniform(low, high, dimension) for low, high in fn_bounds]
        )[0]  # Get the first object... to get the only "particle" moving here

        best_position = position.copy()
        best_fitness = objective_fn(position)

        for iteration in range(1, self.n_iterations + 1):
            # Calculate gradient using finite difference
            gradient = self._compute_gradient(objective_fn, position)

            # Update position using the algebraically decaying learning rate
            position -= (self.learning_rate / iteration) * gradient

            # Ensure the position stays within bounds
            position = np.clip(
                position, [b[0] for b in fn_bounds], [b[1] for b in fn_bounds]
            )

            # Evaluate the objective function
            fitness = objective_fn(position)

            # Update best position and fitness if improvement found
            if fitness < best_fitness:
                best_fitness = fitness
                best_position = position.copy()

            if self._verbose:
                print(f"Iteration {iteration}, Best Fitness: {best_fitness}")

        return best_fitness, best_position

    @staticmethod
    def _compute_gradient(
        objective_fn: Callable[[np.ndarray], float | int], position: np.ndarray
    ) -> np.ndarray:
        """
        Compute the gradient of the objective function using finite differences.

        Parameters:
        - objective_fn: Callable, the objective function to minimize.
        - position: ndarray, current position.
        - bounds: Sequence of tuples indicating the bounds for each dimension.

        Returns:
        - ndarray of gradients for each dimension.
        """
        epsilon = 1e-8
        gradient = np.zeros_like(position)

        for i in range(len(position)):
            perturbed_position = position.copy()

            perturbed_position[i] += epsilon
            f_plus = objective_fn(perturbed_position)

            perturbed_position[i] -= 2 * epsilon
            f_minus = objective_fn(perturbed_position)

            gradient[i] = (f_plus - f_minus) / (2 * epsilon)

        return gradient
