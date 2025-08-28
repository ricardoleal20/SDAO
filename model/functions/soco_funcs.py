"""
SOCO11 deterministic benchmark test functions.

Define the SOCO11 benchmark functions to test the algorithms, including
their name and possible domain.
"""

from typing import TYPE_CHECKING
import math
import numpy as np

if TYPE_CHECKING:
    from model.solver import ExperimentFunction


def soco_sphere_function(x: np.ndarray) -> float:
    """
    Sphere Function: f(x) = sum(x_i^2)

    Arguments:
        - x: np.ndarray - Position vector.

    Returns:
        - float: The value of the sphere function at the given position.
    """
    return float(np.sum(x**2))


def soco_rosenbrock_function(x: np.ndarray) -> float:
    """
    Rosenbrock Function (a=1, b=100):
        f(x) = sum_{i=1}^{d-1} [100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2]

    Arguments:
        - x: np.ndarray - Position vector.

    Returns:
        - float: The value of the Rosenbrock function.
    """
    if x.shape[0] < 2:
        return 0.0
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))  # type: ignore


def soco_rastrigin_function(x: np.ndarray) -> float:
    """
    Rastrigin Function:
        f(x) = 10*d + sum_{i=1}^{d} [x_i^2 - 10*cos(2*pi*x_i)]

    Arguments:
        - x: np.ndarray - Position vector.

    Returns:
        - float: The value of the Rastrigin function.
    """
    d = x.shape[0]
    a = 10.0
    return float(a * d + np.sum(x**2 - a * np.cos(2.0 * np.pi * x)))


def soco_ackley_function(x: np.ndarray) -> float:
    """
    Ackley Function:
        f(x) = -20*exp(-0.2*sqrt(1/d * sum(x_i^2)))
               - exp(1/d * sum(cos(2*pi*x_i))) + 20 + e

    Arguments:
        - x: np.ndarray - Position vector.

    Returns:
        - float: The value of the Ackley function.
    """
    d = x.shape[0]
    sum_sq = float(np.sum(x**2))
    sum_cos = float(np.sum(np.cos(2.0 * math.pi * x)))
    term1 = -20.0 * math.exp(-0.2 * math.sqrt(sum_sq / max(d, 1)))
    term2 = -math.exp(sum_cos / max(d, 1))
    return float(term1 + term2 + 20.0 + math.e)


def soco_griewank_function(x: np.ndarray) -> float:
    """
    Griewank Function:
        f(x) = sum_{i=1}^{d} [x_i^2 / 4000] - prod_{i=1}^{d} cos(x_i / sqrt(i)) + 1

    Arguments:
        - x: np.ndarray - Position vector.

    Returns:
        - float: The value of the Griewank function.
    """
    d = x.shape[0]
    sum_sq = float(np.sum(x**2)) / 4000.0
    if d == 0:
        prod_cos = 1.0
    else:
        prod_cos = float(np.prod(np.cos(x / np.sqrt(np.arange(1, d + 1, dtype=float)))))
    return float(sum_sq - prod_cos + 1.0)


def soco_schwefel_function(x: np.ndarray) -> float:
    """
    Schwefel Function (2.26):
        f(x) = 418.9829*d - sum_{i=1}^{d} [x_i * sin(sqrt(|x_i|))]

    Arguments:
        - x: np.ndarray - Position vector.

    Returns:
        - float: The value of the Schwefel function.
    """
    d = x.shape[0]
    return float(418.9829 * d - np.sum(x * np.sin(np.sqrt(np.abs(x)))))


def soco_weierstrass_function(x: np.ndarray) -> float:
    """
    Weierstrass Function:
        f(x) = sum_{i=1}^{d} sum_{k=0}^{20} [a^k * cos(2*pi*b^k*(x_i + 0.5))]
               - d*sum_{k=0}^{20} a^k * cos(2*pi*b^k*0.5)
    where a = 0.5 and b = 3.

    Arguments:
        - x: np.ndarray - Position vector.

    Returns:
        - float: The value of the Weierstrass function.
    """
    a = 0.5
    b = 3.0
    d = x.shape[0]
    k_values = np.arange(21)
    a_powers = a**k_values
    b_powers = b**k_values
    x_expanded = x[:, np.newaxis] + 0.5  # (d, 1)
    cos_args = 2.0 * math.pi * b_powers * x_expanded  # (d, 21)
    cos_values = np.cos(cos_args)
    sum1 = float(np.sum(a_powers * cos_values))
    sum2 = float(np.sum(a_powers * np.cos(2.0 * math.pi * b_powers * 0.5)))
    return float(sum1 - d * sum2)


def soco_bent_cigar_function(x: np.ndarray) -> float:
    """
    Bent Cigar Function:
        f(x) = x_1^2 + 1e6 * sum_{i=2}^{d} x_i^2
    """
    if x.shape[0] == 0:
        return 0.0
    return float(x[0] ** 2 + 1e6 * np.sum(x[1:] ** 2))


def soco_discus_function(x: np.ndarray) -> float:
    """
    Discus Function:
        f(x) = 1e6 * x_1^2 + sum_{i=2}^{d} x_i^2
    """
    if x.shape[0] == 0:
        return 0.0
    return float(1e6 * x[0] ** 2 + np.sum(x[1:] ** 2))


def soco_elliptic_function(x: np.ndarray) -> float:
    """
    Elliptic Function:
        f(x) = sum_{i=1}^{d} [10^(6 * ((i-1)/(d-1))) * x_i^2]
    """
    d = x.shape[0]
    if d == 0:
        return 0.0
    # Use linspace to avoid division by zero when d == 1 (exponents -> [0.0])
    exponents = np.linspace(0.0, 1.0, d)
    return float(np.sum((10.0 ** (6.0 * exponents)) * (x**2)))  # type: ignore


def soco_zakharov_function(x: np.ndarray) -> float:
    """
    Zakharov Function:
        f(x) = sum(x_i^2) + (0.5 * sum(i*x_i))^2 + (0.5 * sum(i*x_i))^4
    """
    d = x.shape[0]
    indices = np.arange(1, d + 1)
    s1 = float(np.sum(x**2))
    s2 = float(0.5 * np.sum(indices * x))
    return float(s1 + s2**2 + s2**4)


def soco_levy_function(x: np.ndarray) -> float:
    """
    Levy Function:
        Standard Levy with w_i = 1 + (x_i - 1)/4.
        f(x) = sin^2(pi*w_1)
               + sum_{i=1}^{d-1} (w_i - 1)^2 * [1 + 10*sin^2(pi*w_i + 1)]
               + (w_d - 1)^2 * [1 + sin^2(2*pi*w_d)].
    """
    if x.shape[0] == 0:
        return 0.0
    w = 1.0 + (x - 1.0) / 4.0
    term1 = float(np.sin(np.pi * w[0]) ** 2)
    if x.shape[0] > 1:
        term2 = float(
            np.sum((w[:-1] - 1.0) ** 2 * (1.0 + 10.0 * (np.sin(np.pi * w[:-1] + 1.0) ** 2)))
        )
    else:
        term2 = 0.0
    term3 = float((w[-1] - 1.0) ** 2 * (1.0 + (np.sin(2.0 * np.pi * w[-1]) ** 2)))
    return float(term1 + term2 + term3)


# ========================================================= #
# DEFINE ALL THE SOCO11 FUNCTIONS WITH THEIR NAME AND DOMAIN #
soco_funcs: list["ExperimentFunction"] = [
    {
        "name": "SOCO11 Sphere",
        "call": soco_sphere_function,
        "domain": (-100.0, 100.0),
        "optimal_x_value": [0.0],
        "optimal_value": 0.0,
        "dimension": 500,
    },
    {
        "name": "SOCO11 Rosenbrock",
        "call": soco_rosenbrock_function,
        "domain": (-30.0, 30.0),
        "optimal_x_value": [1.0],
        "optimal_value": 0.0,
        "dimension": 500,
    },
    {
        "name": "SOCO11 Rastrigin",
        "call": soco_rastrigin_function,
        "domain": (-5.12, 5.12),
        "optimal_x_value": [0.0],
        "optimal_value": 0.0,
        "dimension": 500,
    },
    {
        "name": "SOCO11 Ackley",
        "call": soco_ackley_function,
        "domain": (-32.768, 32.768),
        "optimal_x_value": [0.0],
        "optimal_value": 0.0,
        "dimension": 500,
    },
    {
        "name": "SOCO11 Griewank",
        "call": soco_griewank_function,
        "domain": (-600.0, 600.0),
        "optimal_x_value": [0.0],
        "optimal_value": 0.0,
        "dimension": 500,
    },
    {
        "name": "SOCO11 Schwefel",
        "call": soco_schwefel_function,
        "domain": (-500.0, 500.0),
        "optimal_x_value": [420.9687],
        "optimal_value": 0.0,
        "dimension": 500,
    },
    {
        "name": "SOCO11 Weierstrass",
        "call": soco_weierstrass_function,
        "domain": (-0.5, 0.5),
        "optimal_x_value": [0.0],
        "optimal_value": 0.0,
        "dimension": 500,
    },
    {
        "name": "SOCO11 Bent Cigar",
        "call": soco_bent_cigar_function,
        "domain": (-100.0, 100.0),
        "optimal_x_value": [0.0],
        "optimal_value": 0.0,
        "dimension": 500,
    },
    {
        "name": "SOCO11 Discus",
        "call": soco_discus_function,
        "domain": (-100.0, 100.0),
        "optimal_x_value": [0.0],
        "optimal_value": 0.0,
        "dimension": 500,
    },
    {
        "name": "SOCO11 Elliptic",
        "call": soco_elliptic_function,
        "domain": (-100.0, 100.0),
        "optimal_x_value": [0.0],
        "optimal_value": 0.0,
        "dimension": 500,
    },
    {
        "name": "SOCO11 Zakharov",
        "call": soco_zakharov_function,
        "domain": (-100.0, 100.0),
        "optimal_x_value": [0.0],
        "optimal_value": 0.0,
        "dimension": 500,
    },
    {
        "name": "SOCO11 Levy",
        "call": soco_levy_function,
        "domain": (-10.0, 10.0),
        "optimal_x_value": [1.0],
        "optimal_value": 0.0,
        "dimension": 500,
    },
]


