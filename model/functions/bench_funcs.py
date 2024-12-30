"""
Run the normal benchmark test functions!

Define the normal benchmark functions to test the algorithms, including
their name and possible domain.
"""
from typing import TYPE_CHECKING
import numpy as np
if TYPE_CHECKING:
    from model.solver import ExperimentFunction


def rastrigin(x: np.ndarray) -> float | int:
    """Rastrigin function.
    
    f(x^d) = A * d + sum_{i=1}^{d} [x_i^2 - A*cos(2*pi*x_i)]

    where $d$ is the dimension of the input vector $x$.
    """
    dim = len(x)
    param = 10
    return param * dim + float(np.sum(x**2 - param * np.cos(2 * np.pi * x)))


# ========================================================= #
# DEFINE ALL THE STOCH FUNCTIONS WITH THEIR NAME AND DOMAIN #
stoch_funcs: list["ExperimentFunction"] = [
    {"name": "Rastrigin", "call": rastrigin, "domain": (-5.12, 5.12)}
]
