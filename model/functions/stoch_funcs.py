"""
Run the Stochastic benchmark test functions!

Define the stochastic benchmark functions to test the algorithms, including
their name and possible domain.
"""

from typing import TYPE_CHECKING
import math
import random
import numpy as np

if TYPE_CHECKING:
    from model.solver import ExperimentFunction


def stochastic_rastrigin_function(x: np.ndarray) -> float | int:
    """Rastrigin function.

    f(x^d) = A * d + sum_{i=1}^{d} [x_i^2 - A*cos(2*pi*x_i)]

    where $d$ is the dimension of the input vector $x$.
    """
    dim = len(x)
    param = 10
    # Gaussian noise with mean 0 and std deviation 0.1
    noise = random.gauss(0, 1)
    return param * dim + float(np.sum(x**2 - param * np.cos(2 * np.pi * x))) + noise


def stochastic_sphere_function(x: np.ndarray) -> float:
    """
    Sphere Function: f(x) = sum(x_i^2)

    **Arguments**:
        - x: np.ndarray - Position vector.

    **Returns**:
        - float: The value of the sphere function at the given position.
    """
    # Gaussian noise with mean 0 and std deviation 0.1
    noise = random.gauss(0, 0.1)
    return float(np.sum(x**2)) + noise


def stochastic_rosenbrock_function(x: np.ndarray) -> float:
    """
    Rosenbrock Function, using a=1 and b=100:
        f(x) = sum_{i=1}^{d-1} [100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2]

    **Arguments**:
        - x: np.ndarray - Position vector.

    **Returns**:
        - float: The value of the Rosenbrock function at the given position.
    """
    # Gaussian noise with mean 0 and std deviation 0.1
    noise = random.gauss(0, 0.1)
    return sum((1 - x[:-1]) ** 2 + 100 * (x[1:] - x[:-1] ** 2) ** 2) + noise


def stochastic_ackley_function(x: np.ndarray) -> float:
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

    # Gaussian noise with mean 0 and std deviation 0.1
    noise = random.gauss(0, 0.1)

    return (
        -20 * math.exp(-0.2 * math.sqrt(sum_sq / d))
        - math.exp(sum_cos / d)
        + 20
        + math.e
        + noise
    )


def stochastic_schwefel_function(x: np.ndarray) -> float:
    """
    Schwefel Function: f(x) = 418.9829*d - sum_{i=1}^{d} [x_i * sin(sqrt(|x_i|))]

    **Arguments**:
        - x: np.ndarray - Position vector.

    **Returns**:
        - float: The value of the Schwefel function at the given position.
    """
    # Gaussian noise with mean 0 and std deviation 0.1
    noise = random.uniform(-1, 1)

    d = len(x)
    return 418.9829 * d - np.sum(x * np.sin(np.sqrt(np.abs(x)))) + noise


def stochastic_drop_wave_function(x: np.ndarray) -> float:
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

    # Gaussian noise with mean 0 and std deviation 0.1
    noise = random.gauss(0, 0.05)

    return 1 - (numerator / denominator) + noise


def stochastic_booth_function(x: np.ndarray) -> float:
    """
    Booth Function: f(x, y) = (x + 2y - 7)^2 + (2x + y - 5)^2
    where x and y are the two dimensions of the position vector.

    **Arguments**:
        - x: np.ndarray - Position vector.

    **Returns**:
        - float: The value of the Booth function at the given position.
    """
    # Gaussian noise with mean 0 and std deviation 0.1
    noise = random.gauss(0, 0.1)

    x0, y = x[0], x[1] if len(x) > 1 else 0
    return (x0 + 2 * y - 7) ** 2 + (2 * x0 + y - 5) ** 2 + noise


def stochastic_beale_function(x: np.ndarray) -> float:
    """
    Beale Function: f(x, y) = (1.5 - x + xy)^2 + (2.25 - x + xy^2)^2 + (2.625 - x + xy^3)^2
    where x and y are the two dimensions of the position vector.

    **Arguments**:
        - x: np.ndarray - Position vector.

    **Returns**:
        - float: The value of the Beale function at the given position.
    """
    # Gaussian noise with mean 0 and std deviation 0.1
    noise = random.gauss(0, 0.1)

    x0, y = x[0], x[1] if len(x) > 1 else 0
    return (
        (1.5 - x0 + x0 * y) ** 2
        + (2.25 - x0 + x0 * y**2) ** 2
        + (2.625 - x0 + x0 * y**3) ** 2
    ) + noise


def stochastic_weierstrass_function(x: np.ndarray) -> float:
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

    # Gaussian noise with mean 0 and std deviation 0.1
    noise = random.gauss(0, 0.1)
    return sum1 - d * sum2 + noise


def stochastic_griewank_function(x: np.ndarray) -> float:
    """
    Griewank's Function: f(x) = sum_{i=1}^{d} [x_i^2 / 4000] - prod_{i=1}^{d} cos(x_i / sqrt(i)) + 1

    **Arguments**:
        - x: np.ndarray - Position vector.

    **Returns**:
        - float: The value of the Griewank's function at the given position.
    """
    sum_sq = np.sum(x**2) / 4000
    prod_cos = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))

    # Gaussian noise with mean 0 and std deviation 0.1
    noise = random.gauss(0, 0.1)

    return sum_sq - prod_cos + 1 + noise  # type: ignore


def stochastic_happy_cat_function(x: np.ndarray) -> float:
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

    # Gaussian noise with mean 0 and std deviation 0.1
    noise = random.gauss(0, 0.1)
    return (
        np.abs(sum_sq - 4) ** 0.25 + 0.5 * (sum_sq - 4) + 0.5 + additional_sum + noise
    )


def stochastic_salomon_function(x: np.ndarray) -> float:
    """
    Salomon Function: f(x) = 1 - cos(2*pi*sqrt(sum_{i=1}^{d} x_i^2)) + 0.1*sqrt(sum_{i=1}^{d} x_i^2)

    **Arguments**:
        - x: np.ndarray - Position vector.

    **Returns**:
        - float: The value of the Salomon function at the given position.
    """
    # Gaussian noise with mean 0 and std deviation 0.1
    noise = random.gauss(0, 0.1)

    sum_sq = np.sum(x**2)
    return (
        1 - math.cos(2 * math.pi * math.sqrt(sum_sq)) + 0.1 * math.sqrt(sum_sq) + noise
    )


# ========================================================= #
# DEFINE ALL THE STOCH FUNCTIONS WITH THEIR NAME AND DOMAIN #
stoch_funcs: list["ExperimentFunction"] = [
    {"name": "Sphere", "call": stochastic_sphere_function, "domain": (-5.12, 5.12)},
    {
        "name": "Rosenbrock",
        "call": stochastic_rosenbrock_function,
        "domain": (-5.0, 10.0),
    },
    {
        "name": "Rastrigin",
        "call": stochastic_rastrigin_function,
        "domain": (-5.12, 5.12),
    },
    {"name": "Ackley", "call": stochastic_ackley_function, "domain": (-32.768, 32.768)},
    {"name": "Schwefel", "call": stochastic_schwefel_function, "domain": (-500, 500)},
    {
        "name": "Drop-Wave",
        "call": stochastic_drop_wave_function,
        "domain": (-5.12, 5.12),
    },
    {"name": "Booth", "call": stochastic_booth_function, "domain": (-10, 10)},
    {"name": "Beale", "call": stochastic_beale_function, "domain": (-4.5, 4.5)},
    # {"name": "Weierstrass", "call": weierstrass_function,
    #    "domain": (-0.5, 0.5)},
    {"name": "Griewank", "call": stochastic_griewank_function, "domain": (-600, 600)},
    {"name": "Happy Cat", "call": stochastic_happy_cat_function, "domain": (-2.0, 2.0)},
    {"name": "Salomon", "call": stochastic_salomon_function, "domain": (-100, 100)},
]  # type: ignore
