"""
Include the CEC functions
"""

from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from model.solver import ExperimentFunction


def cec_shifted_sphere_function(x: np.ndarray) -> float:
    """Shifted Sphere Function.

    f(x) = sum((x - shift)^2)

    Args:
        x (np.ndarray): Input vector.

    Returns:
        float: Function value.
    """
    shift = np.full_like(x, 0.5)  # constant shift vector
    z = x - shift
    return np.sum(z**2)


def cec_shifted_rosenbrock_function(x: np.ndarray) -> float:
    """Shifted Rosenbrock Function.

    f(x) = sum_{i=1}^{n-1} [100 * (z_{i+1} - z_i^2)^2 + (z_i - 1)^2],
    where z = x - shift + 1.

    Args:
        x (np.ndarray): Input vector.

    Returns:
        float: Function value.
    """
    shift = np.full_like(x, 0.5)
    z = x - shift + 1.0
    return np.sum(100.0 * (z[1:] - z[:-1] ** 2) ** 2 + (z[:-1] - 1) ** 2)  # type: ignore


def cec_shifted_rastrigin_function(x: np.ndarray) -> float:
    """Shifted Rastrigin Function.

    f(x) = 10*n + sum((x - shift)^2 - 10*cos(2*pi*(x - shift)))

    Args:
        x (np.ndarray): Input vector.

    Returns:
        float: Function value.
    """
    shift = np.full_like(x, 0.5)
    z = x - shift
    n = x.shape[0]
    return 10 * n + np.sum(z**2 - 10 * np.cos(2 * np.pi * z))


def cec_shifted_schwefel_function(x: np.ndarray) -> float:
    """Shifted Schwefel Function.

    f(x) = 418.9829 * n - sum((x - shift) * sin(sqrt(|x - shift|)))

    Args:
        x (np.ndarray): Input vector.

    Returns:
        float: Function value.
    """
    shift = np.full_like(x, 420.968746)
    z = x - shift
    n = x.shape[0]
    return 418.9829 * n - np.sum(z * np.sin(np.sqrt(np.abs(z))))


def cec_shifted_griewank_function(x: np.ndarray) -> float:
    """Shifted Griewank Function.

    f(x) = sum(z^2)/4000 - prod(cos(z_i/sqrt(i))) + 1,
    where z = x - shift.

    Args:
        x (np.ndarray): Input vector.

    Returns:
        float: Function value.
    """
    shift = np.full_like(x, 0.5)
    z = x - shift
    sum_term = np.sum(z**2) / 4000.0
    prod_term = np.prod(np.cos(z / np.sqrt(np.arange(1, x.shape[0] + 1))))
    return sum_term - prod_term + 1


def cec_shifted_ackley_function(x: np.ndarray) -> float:
    """Shifted Ackley Function.

    f(x) = -20 * exp(-0.2 * sqrt(sum(z^2)/n)) - exp(sum(cos(2*pi*z))/n) + 20 + e,
    where z = x - shift.

    Args:
        x (np.ndarray): Input vector.

    Returns:
        float: Function value.
    """
    shift = np.full_like(x, 0.5)
    z = x - shift
    n = x.shape[0]
    term1 = -20 * np.exp(-0.2 * np.sqrt(np.sum(z**2) / n))
    term2 = -np.exp(np.sum(np.cos(2 * np.pi * z)) / n)
    return term1 + term2 + 20 + np.e


def cec_shifted_weierstrass_function(x: np.ndarray) -> float:
    """Shifted Weierstrass Function.

    f(x) = sum_{i=1}^{n} sum_{k=0}^{k_max} [a^k * cos(2*pi*b^k*(z_i + 0.5))]
           - n * sum_{k=0}^{k_max} [a^k * cos(2*pi*b^k*0.5)],
    where z = x - shift.

    Args:
        x (np.ndarray): Input vector.

    Returns:
        float: Function value.
    """
    shift = np.full_like(x, 0.5)
    z = x - shift
    a = 0.5
    b = 3
    k_max = 20
    n = x.shape[0]
    sum1 = 0.0
    for i in range(n):
        for k in range(k_max + 1):
            sum1 += a**k * np.cos(2 * np.pi * b**k * (z[i] + 0.5))
    sum2 = 0.0
    for k in range(k_max + 1):
        sum2 += a**k * np.cos(2 * np.pi * b**k * 0.5)
    return sum1 - n * sum2


def cec_shifted_bent_cigar_function(x: np.ndarray) -> float:
    """Shifted Bent Cigar Function.

    f(x) = z_1^2 + 1e6 * sum_{i=2}^{n} z_i^2,
    where z = x - shift.

    Args:
        x (np.ndarray): Input vector.

    Returns:
        float: Function value.
    """
    shift = np.full_like(x, 0.5)
    z = x - shift
    return z[0] ** 2 + 1e6 * np.sum(z[1:] ** 2)


def cec_shifted_discus_function(x: np.ndarray) -> float:
    """Shifted Discus Function.

    f(x) = 1e6 * z_1^2 + sum_{i=2}^{n} z_i^2,
    where z = x - shift.

    Args:
        x (np.ndarray): Input vector.

    Returns:
        float: Function value.
    """
    shift = np.full_like(x, 0.5)
    z = x - shift
    return 1e6 * z[0] ** 2 + np.sum(z[1:] ** 2)


def cec_shifted_elliptic_function(x: np.ndarray) -> float:
    """Shifted Elliptic Function.

    f(x) = sum_{i=1}^{n} [10^(6 * ((i-1)/(n-1))) * z_i^2],
    where z = x - shift.

    Args:
        x (np.ndarray): Input vector.

    Returns:
        float: Function value.
    """
    shift = np.full_like(x, 0.5)
    z = x - shift
    n = x.shape[0]
    exponents = np.linspace(0, 1, n)
    return np.sum(10 ** (6 * exponents) * (z**2))  # type: ignore


# 11. Expanded Scaffer F6 Function (applied to consecutive pairs)
def cec_expanded_scaffer_f6(x: np.ndarray) -> float:
    """Expanded Scaffer F6 Function.

    f(x) = (1/(n-1)) * sum_{i=1}^{n-1} ScafferF6(z_i, z_{i+1}),
    where z = x - shift and ScafferF6 is defined as:
      ScafferF6(a, b) = 0.5 + (sin^2(sqrt(a^2 + b^2)) - 0.5) / (1 + 0.001*(a^2+b^2))^2.

    Args:
        x (np.ndarray): Input vector.

    Returns:
        float: Function value.
    """

    def scaffer_f6(a: float, b: float) -> float:
        r = np.sqrt(a**2 + b**2)
        return 0.5 + (np.sin(r) ** 2 - 0.5) / ((1 + 0.001 * (a**2 + b**2)) ** 2)

    z = x.copy() - 0.5  # shifting
    f = 0.0
    n = x.shape[0]
    for i in range(n - 1):
        f += scaffer_f6(z[i], z[i + 1])
    return f / (n - 1)


def cec_shifted_happy_cat_function(x: np.ndarray) -> float:
    """Shifted Happy Cat Function.

    f(x) = |sum(z^2) - n|^(0.25) + (0.5 * sum(z^2) + sum(z))/n + 0.5,
    where z = x - shift.

    Args:
        x (np.ndarray): Input vector.

    Returns:
        float: Function value.
    """
    shift = np.full_like(x, 0.5)
    z = x - shift
    n = x.shape[0]
    sum_z2 = np.sum(z**2)
    term1 = np.abs(sum_z2 - n) ** 0.25
    term2 = (0.5 * sum_z2 + np.sum(z)) / n + 0.5
    return term1 + term2


def cec_shifted_hgbat_function(x: np.ndarray) -> float:
    """Shifted HGBat Function.

    A variant of the Happy Cat function with a different exponent.
    f(x) = |sum(z^2) - n|^(0.5) + (0.5 * sum(z^2) + sum(z))/n + 0.5,
    where z = x - shift.

    Args:
        x (np.ndarray): Input vector.

    Returns:
        float: Function value.
    """
    shift = np.full_like(x, 0.5)
    z = x - shift
    n = x.shape[0]
    sum_z2 = np.sum(z**2)
    return np.abs(sum_z2 - n) ** 0.5 + (0.5 * sum_z2 + np.sum(z)) / n + 0.5


def cec_shifted_non_continuous_rastrigin_function(x: np.ndarray) -> float:
    """Shifted Non-Continuous Rastrigin Function.

    This version rounds z values if |z_i| > 0.5.
    f(x) = 10*n + sum(z_nc^2 - 10*cos(2*pi*z_nc)),
    where z = x - shift and z_nc is the non-continuous modification of z.

    Args:
        x (np.ndarray): Input vector.

    Returns:
        float: Function value.
    """
    shift = np.full_like(x, 0.5)
    z = x - shift
    z_nc = np.where(np.abs(z) > 0.5, np.round(z * 2) / 2.0, z)
    n = x.shape[0]
    return 10 * n + np.sum(z_nc**2 - 10 * np.cos(2 * np.pi * z_nc))


def cec_hybrid_composition_function(x: np.ndarray) -> float:
    """Hybrid Composition Function.

    A simple combination of the Sphere, Rastrigin, and Ackley functions.
    f(x) = w1 * f1(x) + w2 * f2(x) + w3 * f3(x),
    where f1, f2, f3 are the shifted Sphere, Rastrigin, and Ackley functions respectively.

    Args:
        x (np.ndarray): Input vector.

    Returns:
        float: Function value.
    """
    f1 = cec_shifted_sphere_function(x)
    f2 = cec_shifted_rastrigin_function(x)
    f3 = cec_shifted_ackley_function(x)
    w1, w2, w3 = 0.3, 0.3, 0.4  # predefined weights
    return w1 * f1 + w2 * f2 + w3 * f3


# List of CEC benchmark functions
cec_funcs: list["ExperimentFunction"] = [
    {
        "name": "CEC Shifted Sphere",
        "call": cec_shifted_sphere_function,
        "domain": (-100, 100),
    },
    {
        "name": "CEC Shifted Rosenbrock",
        "call": cec_shifted_rosenbrock_function,
        "domain": (-30, 30),
    },
    {
        "name": "CEC Shifted Rastrigin",
        "call": cec_shifted_rastrigin_function,
        "domain": (-5.12, 5.12),
    },
    {
        "name": "CEC Shifted Schwefel",
        "call": cec_shifted_schwefel_function,
        "domain": (-500, 500),
    },
    {
        "name": "CEC Shifted Griewank",
        "call": cec_shifted_griewank_function,
        "domain": (-600, 600),
    },
    {
        "name": "CEC Shifted Ackley",
        "call": cec_shifted_ackley_function,
        "domain": (-32, 32),
    },
    # {
    #     "name": "CEC Shifted Weierstrass",
    #     "call": cec_shifted_weierstrass_function,
    #     "domain": (-0.5, 0.5),
    # },
    {
        "name": "CEC Shifted Bent Cigar",
        "call": cec_shifted_bent_cigar_function,
        "domain": (-100, 100),
    },
    {
        "name": "CEC Shifted Discus",
        "call": cec_shifted_discus_function,
        "domain": (-100, 100),
    },
    {
        "name": "CEC Shifted Elliptic",
        "call": cec_shifted_elliptic_function,
        "domain": (-100, 100),
    },
    {
        "name": "CEC Expanded Scaffer F6",
        "call": cec_expanded_scaffer_f6,
        "domain": (-100, 100),
    },
    {
        "name": "CEC Shifted Happy Cat",
        "call": cec_shifted_happy_cat_function,
        "domain": (-2, 2),
    },
    {
        "name": "CEC Shifted HGBat",
        "call": cec_shifted_hgbat_function,
        "domain": (-100, 100),
    },
    {
        "name": "CEC Shifted Non-Continuous Rastrigin",
        "call": cec_shifted_non_continuous_rastrigin_function,
        "domain": (-5.12, 5.12),
    },
    {
        "name": "CEC Hybrid Composition",
        "call": cec_hybrid_composition_function,
        "domain": (-5, 5),
    },
]
