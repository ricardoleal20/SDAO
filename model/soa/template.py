"""
Include the template for all the algorithms to develop...
"""

from abc import ABC, abstractmethod
from typing import Callable, Sequence
import numpy as np


class Algorithm(ABC):  # pylint: disable=R0903
    """Create the Algorithm interface"""

    @abstractmethod
    def optimize(
        self,
        objective_fn: Callable[[np.ndarray], float | int],
        bounds: Sequence[tuple[float, float]] | tuple[float, float],
        dimension: int,
    ) -> tuple[float, np.ndarray]:
        """..."""
