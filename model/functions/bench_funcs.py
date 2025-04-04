"""
Run the normal benchmark test functions!

Define the normal benchmark functions to test the algorithms, including
their name and possible domain.
"""

from typing import TYPE_CHECKING
import math
import numpy as np

if TYPE_CHECKING:
    from model.solver import ExperimentFunction


def rastrigin_function(x: np.ndarray) -> float | int:
    """Rastrigin function.

    f(x^d) = A * d + sum_{i=1}^{d} [x_i^2 - A*cos(2*pi*x_i)]

    where $d$ is the dimension of the input vector $x$.
    """
    dim = len(x)
    param = 10
    return param * dim + float(np.sum(x**2 - param * np.cos(2 * np.pi * x)))


def sphere_function(x: np.ndarray) -> float | np.signedinteger:
    """
    Sphere Function: f(x) = sum(x_i^2)

    **Arguments**:
        - x: np.ndarray - Position vector.

    **Returns**:
        - float: The value of the sphere function at the given position.
    """
    return np.sum(x**2)


def rosenbrock_function(x: np.ndarray) -> float:
    """
    Rosenbrock Function, using a=1 and b=100:
        f(x) = sum_{i=1}^{d-1} [100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2]

    **Arguments**:
        - x: np.ndarray - Position vector.

    **Returns**:
        - float: The value of the Rosenbrock function at the given position.
    """
    return sum((1 - x[:-1]) ** 2 + 100 * (x[1:] - x[:-1] ** 2) ** 2)


def ackley_function(x: np.ndarray) -> float:
    """
    Ackley Function:
    f(x) = -20*exp(-0.2*sqrt(1/d * sum(x_i^2))) - exp(1/d * sum(cos(2*pi*x_i))) + 20 + e

    **Arguments**:
        - x: np.ndarray - Position vector.

    **Returns**:
        - float: The value of the Ackley function at the given position.
    """
    d = len(x)
    sum_sq = np.sum(x**2)
    sum_cos = np.sum(np.cos(2 * math.pi * x))
    return (
        -20 * math.exp(-0.2 * math.sqrt(sum_sq / d))
        - math.exp(sum_cos / d)
        + 20
        + math.e
    )


def schwefel_function(x: np.ndarray) -> float:
    """
    Schwefel Function: f(x) = 418.9829*d - sum_{i=1}^{d} [x_i * sin(sqrt(|x_i|))]

    **Arguments**:
        - x: np.ndarray - Position vector.

    **Returns**:
        - float: The value of the Schwefel function at the given position.
    """
    d = len(x)
    return 418.9829 * d - np.sum(x * np.sin(np.sqrt(np.abs(x))))


def drop_wave_function(x: np.ndarray) -> float:
    """
    Drop-Wave Function: f(x) = - (1 + cos(12*sqrt(x^2 + y^2))) / (0.5*(x^2 + y^2) + 2)
    where x and y are the two dimensions of the position vector.

    **Arguments**:
        - x: np.ndarray - Position vector.

    **Returns**:
        - float: The value of the Drop-Wave function at the given position.
    """
    quadratic_sum = np.sum([y**2 for y in x])
    numerator = 1 + math.cos(12 * math.sqrt(quadratic_sum))
    denominator = 0.5 * (quadratic_sum) + 2
    return 1 - (numerator / denominator)


def booth_function(x: np.ndarray) -> float:
    """
    Booth Function: f(x, y) = (x + 2y - 7)^2 + (2x + y - 5)^2
    where x and y are the two dimensions of the position vector.

    **Arguments**:
        - x: np.ndarray - Position vector.

    **Returns**:
        - float: The value of the Booth function at the given position.
    """
    x0, y = x[0], x[1] if len(x) > 1 else 0
    return (x0 + 2 * y - 7) ** 2 + (2 * x0 + y - 5) ** 2


def beale_function(x: np.ndarray) -> float:
    """
    Beale Function: f(x, y) = (1.5 - x + xy)^2 + (2.25 - x + xy^2)^2 + (2.625 - x + xy^3)^2
    where x and y are the two dimensions of the position vector.

    **Arguments**:
        - x: np.ndarray - Position vector.

    **Returns**:
        - float: The value of the Beale function at the given position.
    """
    x0, y = x[0], x[1] if len(x) > 1 else 0
    return (
        (1.5 - x0 + x0 * y) ** 2
        + (2.25 - x0 + x0 * y**2) ** 2
        + (2.625 - x0 + x0 * y**3) ** 2
    )


def weierstrass_function(x: np.ndarray) -> float:
    """
    Weierstrass Function:
        f(x) = sum_{i=1}^{d} sum_{k=0}^{20} [a^k * cos(2*pi*b^k*(x_i + 0.5))]
                - d*sum_{k=0}^{20} a^k * cos(pi*b^k)
    where a = 0.5 and b = 3.

    **Arguments**:
        - x: np.ndarray - Position vector.

    **Returns**:
        - float: The value of the Weierstrass function at the given position.
    """
    a = 0.5
    b = 3.0
    d = len(x)
    sum1 = sum(
        a**k * np.cos(2 * math.pi * b**k * (xi + 0.5)) for xi in x for k in range(21)
    )
    sum2 = sum(a**k * np.cos(math.pi * b**k) for k in range(21))
    return sum1 - d * sum2


def griewank_function(x: np.ndarray) -> float:
    """
    Griewank's Function: f(x) = sum_{i=1}^{d} [x_i^2 / 4000] - prod_{i=1}^{d} cos(x_i / sqrt(i)) + 1

    **Arguments**:
        - x: np.ndarray - Position vector.

    **Returns**:
        - float: The value of the Griewank's function at the given position.
    """
    sum_sq = np.sum(x**2) / 4000
    prod_cos = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
    return sum_sq - prod_cos + 1  # type: ignore


def happy_cat_function(x: np.ndarray) -> float:
    """
    HappyCat Function:
        f(x) = (|x^2 - 4|^0.25 + 0.5*(x^2 - 4) + 0.5)
                + sum_{i=1}^{d} [1/(8*i) * (x_i^2 - 1)^2]

    **Arguments**:
        - x: np.ndarray - Position vector.

    **Returns**:
        - float: The value of the HappyCat function at the given position.
    """
    sum_sq = np.sum(x**2)
    additional_sum = sum(
        (1 / (8 * (i + 1))) * (xi**2 - 1) ** 2 for i, xi in enumerate(x)
    )
    return (
        np.abs(sum_sq - 4) ** 0.25
        + 0.5 * (sum_sq - 4)
        + 0.5
        + additional_sum
        # We add this to ensure the function HappyCat get's a minimum value of 0'
        - 5.065084580073288
    )


def schaffer_f7_function(x: np.ndarray) -> float:
    """
    Schaffer's F7 Function:
        f(x, y) = 0.5 + (sin(sqrt(x^2 + y^2))^2 - 0.5) / (1 + 0.001*(x^2 + y^2))^2
    where x and y are the two dimensions of the position vector.

    **Arguments**:
        - x: np.ndarray - Position vector.

    **Returns**:
        - float: The value of the Schaffer's F7 function at the given position.
    """
    sq_sum = np.sum([y**2 for y in x])
    return 0.5 + (math.sin(math.sqrt(sq_sum)) ** 2 - 0.5) / (1 + 0.001 * sq_sum) ** 2


def expanded_schaffer_f6_function(x: np.ndarray) -> float:
    """
    Expanded Schaffer's F6 Function:

        f(x, y) = 0.5 + (sin(sqrt(x^2 + y^2))^2 - 0.5)
                        / (1 + 0.001*(x^2 + y^2))^2
    where x and y are the two dimensions of the position vector.

    **Arguments**:
        - x: np.ndarray - Position vector.

    **Returns**:
        - float: The value of the Expanded Schaffer's F6 function at the given position.
    """
    sq_sum = np.sum([y**2 for y in x])
    return 0.5 + (math.sin(math.sqrt(sq_sum)) ** 2 - 0.5) / (1 + 0.001 * sq_sum) ** 2


def xin_she_yang_1_function(x: np.ndarray) -> float | np.signedinteger:
    """
    Xin-She Yang's 1 Function: f(x) = sum_{i=1}^{d} |x_i|^i

    **Arguments**:
        - x: np.ndarray - Position vector.

    **Returns**:
        - float: The value of the Xin-She Yang's 1 function at the given position.
    """
    return np.sum(np.abs(x) ** np.arange(1, len(x) + 1))


def salomon_function(x: np.ndarray) -> float:
    """
    Salomon Function: f(x) = 1 - cos(2*pi*sqrt(sum_{i=1}^{d} x_i^2)) + 0.1*sqrt(sum_{i=1}^{d} x_i^2)

    **Arguments**:
        - x: np.ndarray - Position vector.

    **Returns**:
        - float: The value of the Salomon function at the given position.
    """
    sum_sq = np.sum(x**2)
    return 1 - math.cos(2 * math.pi * math.sqrt(sum_sq)) + 0.1 * math.sqrt(sum_sq)


# ========================================================= #
# DEFINE ALL THE STOCH FUNCTIONS WITH THEIR NAME AND DOMAIN #
bench_funcs: list["ExperimentFunction"] = [
    {  # Sphere: optimum f(x)=0 at x = (0,0,...,0)
        "name": "Sphere",
        "call": sphere_function,
        "domain": (-5.12, 5.12),
        "optimal_value": 0,
        "optimal_x_value": [0],
    },
    {  # Rosenbrock: optimum f(x)=0 at x = (1,1,...,1)
        "name": "Rosenbrock",
        "call": rosenbrock_function,
        "domain": (-5.0, 10.0),
        "optimal_value": 0,
        "optimal_x_value": [1],
    },
    {  # Rastrigin: optimum f(x)=0 at x = (0,0,...,0)
        "name": "Rastrigin",
        "call": rastrigin_function,
        "domain": (-5.12, 5.12),
        "optimal_value": 0,
        "optimal_x_value": [0],
    },
    {  # Ackley: optimum f(x)=0 at x = (0,0,...,0)
        "name": "Ackley",
        "call": ackley_function,
        "domain": (-32.768, 32.768),
        "optimal_value": 0,
        "optimal_x_value": [0],
    },
    {  # Schwefel: optimum f(x)=0 at x_i = 420.9687 for every i
        "name": "Schwefel",
        "call": schwefel_function,
        "domain": (-500, 500),
        "optimal_value": 0,
        "optimal_x_value": [420.9687],
    },
    {  # Drop-Wave: optimum f(x)=0 at (0,0)
        "name": "Drop-Wave",
        "call": drop_wave_function,
        "domain": (-5.12, 5.12),
        "optimal_value": 0,
        "optimal_x_value": [0, 0],
    },
    {  # Booth: optimum f(x)=0 at (1,3)
        "name": "Booth",
        "call": booth_function,
        "domain": (-10, 10),
        "optimal_value": 0,
        "optimal_x_value": [1, 3],
    },
    {  # Beale: optimum f(x)=0 at (3,0.5)
        "name": "Beale",
        "call": beale_function,
        "domain": (-4.5, 4.5),
        "optimal_value": 0,
        "optimal_x_value": [3, 0.5],
    },
    # The Weierstrass function is commented out in the code.
    {  # Griewank: optimum f(x)=0 at x = (0, 0, ... ,0)
        "name": "Griewank",
        "call": griewank_function,
        "domain": (-600, 600),
        "optimal_value": 0,
        "optimal_x_value": [0],
    },
    {  # Happy Cat: optimum f(x)=0 by design (minimizer not trivial)
        "name": "Happy Cat",
        "call": happy_cat_function,
        "domain": (-2.0, 2.0),
        "optimal_value": 0,
    },
    {  # Schaffer F7: optimum f(x)=0 at (0,0)
        "name": "Schaffer F7",
        "call": schaffer_f7_function,
        "domain": (-100, 100),
        "optimal_value": 0,
        "optimal_x_value": [0, 0],
    },
    {  # Expanded Schaffer F6: optimum f(x)=0 at (0,0)
        "name": "Expanded Schaffer F6",
        "call": expanded_schaffer_f6_function,
        "domain": (-100, 100),
        "optimal_value": 0,
        "optimal_x_value": [0, 0],
    },
    {  # Xin-She Yang 1: optimum f(x)=0 at x = (0,0,...,0)
        "name": "Xin-She Yang 1",
        "call": xin_she_yang_1_function,
        "domain": (-10, 10),
        "optimal_value": 0,
        "optimal_x_value": [0],
    },
    {  # Salomon: optimum f(x)=0 at x = (0,0,...,0)
        "name": "Salomon",
        "call": salomon_function,
        "domain": (-100, 100),
        "optimal_value": 0,
        "optimal_x_value": [0],
    },
]
