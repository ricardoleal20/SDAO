"""
Include the template for all the algorithms to develop...
"""

from abc import ABC, abstractmethod
from typing import Callable, Sequence
import numpy as np

StepSolution = tuple[int, float]


class Algorithm(ABC):  # pylint: disable=R0903
    """Create the Algorithm interface"""

    _iterations: list[StepSolution]

    @abstractmethod
    def optimize(
        self,
        objective_fn: Callable[[np.ndarray], float | int | np.signedinteger],
        bounds: Sequence[tuple[float, float]] | tuple[float, float],
        dimension: int,
    ) -> tuple[float, np.ndarray]:
        """..."""

    def get_iterations_solutions(self) -> list[StepSolution]:
        """Return the list of iterations"""
        return self._iterations
