"""
Include the CEC functions
"""

from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from model.solver import ExperimentFunction


# Constants and cached helpers to avoid repeated allocations
PI2: float = 2.0 * np.pi


@lru_cache(maxsize=None)
def _inv_sqrt_indices(n: int) -> np.ndarray:
    """Return 1/sqrt(1..n) as a cached vector.

    The returned array must be treated as read-only by callers.
    """
    return 1.0 / np.sqrt(np.arange(1, n + 1, dtype=float))


@lru_cache(maxsize=None)
def _elliptic_weights(n: int) -> np.ndarray:
    """Weights for the elliptic function: 10 ** (6 * ((i-1)/(n-1)))."""
    if n == 1:
        # When n == 1, original implementation behaved like exponent = 0 -> weight = 1.0
        return np.array([1.0], dtype=float)
    exponents = np.linspace(0.0, 1.0, n, dtype=float)
    return np.power(10.0, 6.0 * exponents, dtype=float)


@lru_cache(maxsize=None)
def _zakharov_coeffs(n: int) -> np.ndarray:
    """Return 0.5 * (1..n) as float for Zakharov's weighted sum."""
    return 0.5 * np.arange(1, n + 1, dtype=float)


@lru_cache(maxsize=None)
def _arange_1_to_n(n: int) -> np.ndarray:
    """Return integer array [1, 2, ..., n] (cached)."""
    return np.arange(1, n + 1, dtype=int)


@lru_cache(maxsize=None)
def _weierstrass_params(k_max: int = 20) -> tuple[np.ndarray, np.ndarray, float]:
    """Cached powers and constant term for Weierstrass.

    Returns (a_pow, b_pow, sum2_const) where:
    - a_pow[k] = a**k, a = 0.5
    - b_pow[k] = b**k, b = 3.0
    - sum2_const = sum_{k=0}^{k_max} a^k * cos(2*pi*b^k*0.5)
    """
    a: float = 0.5
    b: float = 3.0
    k = np.arange(0, k_max + 1, dtype=int)
    a_pow = np.power(a, k, dtype=float)
    b_pow = np.power(b, k, dtype=float)
    sum2_const = float(np.sum(a_pow * np.cos(PI2 * b_pow * 0.5)))
    return a_pow, b_pow, sum2_const


def cec_shifted_sphere_function(x: np.ndarray) -> float:
    """Shifted Sphere Function.

    f(x) = sum((x - shift)^2)

    Args:
        x (np.ndarray): Input vector.

    Returns:
        float: Function value.
    """
    z = x - 0.5
    return float(np.dot(z, z))


def cec_shifted_rosenbrock_function(x: np.ndarray) -> float:
    """Shifted Rosenbrock Function.

    f(x) = sum_{i=1}^{n-1} [100 * (z_{i+1} - z_i^2)^2 + (z_i - 1)^2],
    where z = x - shift + 1.

    Args:
        x (np.ndarray): Input vector.

    Returns:
        float: Function value.
    """
    z = x - 0.5 + 1.0
    zi = z[:-1]
    zip1 = z[1:]
    return float(np.sum(100.0 * (zip1 - zi * zi) ** 2 + (zi - 1.0) ** 2))


def cec_shifted_rastrigin_function(x: np.ndarray) -> float:
    """Shifted Rastrigin Function.

    f(x) = 10*n + sum((x - shift)^2 - 10*cos(2*pi*(x - shift)))

    Args:
        x (np.ndarray): Input vector.

    Returns:
        float: Function value.
    """
    z = x - 0.5
    n = x.shape[0]
    return float(10 * n + (np.dot(z, z) - np.sum(10.0 * np.cos(PI2 * z))))


def cec_shifted_schwefel_function(x: np.ndarray) -> float:
    """Shifted Schwefel Function.

    f(x) = 418.9829 * n - sum((x - shift) * sin(sqrt(|x - shift|)))

    Args:
        x (np.ndarray): Input vector.

    Returns:
        float: Function value.
    """
    z = x - 420.968746
    n = x.shape[0]
    return float(418.9829 * n - np.sum(z * np.sin(np.sqrt(np.abs(z)))))


def cec_shifted_griewank_function(x: np.ndarray) -> float:
    """Shifted Griewank Function.

    f(x) = sum(z^2)/4000 - prod(cos(z_i/sqrt(i))) + 1,
    where z = x - shift.

    Args:
        x (np.ndarray): Input vector.

    Returns:
        float: Function value.
    """
    n = x.shape[0]
    z = x - 0.5
    sum_term = float(np.dot(z, z)) / 4000.0
    prod_term = np.prod(np.cos(z * _inv_sqrt_indices(n)))
    return float(sum_term - prod_term + 1)


def cec_shifted_ackley_function(x: np.ndarray) -> float:
    """Shifted Ackley Function.

    f(x) = -20 * exp(-0.2 * sqrt(sum(z^2)/n)) - exp(sum(cos(2*pi*z))/n) + 20 + e,
    where z = x - shift.

    Args:
        x (np.ndarray): Input vector.

    Returns:
        float: Function value.
    """
    z = x - 0.5
    n = x.shape[0]
    sumsq = float(np.dot(z, z))
    term1 = -20.0 * float(np.exp(-0.2 * np.sqrt(sumsq / n)))
    mean_cos = float(np.mean(np.cos(PI2 * z)))
    term2 = -float(np.exp(mean_cos))
    return float(term1 + term2 + 20.0 + np.e)


def cec_shifted_sum_of_different_powers_function(x: np.ndarray) -> float:
    """Shifted Sum of Different Powers Function (CEC F2 variant without rotation).

    f(x) = sum_{i=1}^{n} |z_i|^{i+1}, where z = x - shift.

    Args:
        x (np.ndarray): Input vector.

    Returns:
        float: Function value.
    """
    z = x - 0.5
    indices = _arange_1_to_n(x.shape[0])
    return float(np.sum(np.power(np.abs(z), indices + 1)))


def cec_shifted_zakharov_function(x: np.ndarray) -> float:
    """Shifted Zakharov Function (CEC F3 variant without rotation).

    f(x) = sum(z_i^2) + (0.5 * sum(i * z_i))^2 + (0.5 * sum(i * z_i))^4,
    where z = x - shift.

    Args:
        x (np.ndarray): Input vector.

    Returns:
        float: Function value.
    """
    z = x - 0.5
    n = x.shape[0]
    s1 = float(np.dot(z, z))
    weighted_sum = float(np.sum(_zakharov_coeffs(n) * z))
    return s1 + weighted_sum**2 + weighted_sum**4


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
    z = x - 0.5
    n = x.shape[0]
    k_max = 20
    a_pow, b_pow, sum2_const = _weierstrass_params(k_max)
    # angles shape: (n, k_max+1)
    angles = PI2 * (z[:, None] + 0.5) * b_pow[None, :]
    cos_terms = np.cos(angles)
    # sum over k for each i, then sum over i
    sum1 = float(np.sum(cos_terms @ a_pow))
    return float(sum1 - n * sum2_const)


def cec_shifted_bent_cigar_function(x: np.ndarray) -> float:
    """Shifted Bent Cigar Function.

    f(x) = z_1^2 + 1e6 * sum_{i=2}^{n} z_i^2,
    where z = x - shift.

    Args:
        x (np.ndarray): Input vector.

    Returns:
        float: Function value.
    """
    z = x - 0.5
    return float(z[0] ** 2 + 1e6 * np.dot(z[1:], z[1:]))


def cec_shifted_discus_function(x: np.ndarray) -> float:
    """Shifted Discus Function.

    f(x) = 1e6 * z_1^2 + sum_{i=2}^{n} z_i^2,
    where z = x - shift.

    Args:
        x (np.ndarray): Input vector.

    Returns:
        float: Function value.
    """
    z = x - 0.5
    return float(1e6 * z[0] ** 2 + np.dot(z[1:], z[1:]))


def cec_shifted_elliptic_function(x: np.ndarray) -> float:
    """Shifted Elliptic Function.

    f(x) = sum_{i=1}^{n} [10^(6 * ((i-1)/(n-1))) * z_i^2],
    where z = x - shift.

    Args:
        x (np.ndarray): Input vector.

    Returns:
        float: Function value.
    """
    z = x - 0.5
    n = x.shape[0]
    weights = _elliptic_weights(n)
    return float(np.dot(weights, np.square(z)))


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

    z = x - 0.5
    n = x.shape[0]
    if n <= 1:
        return 0.0
    a = z[:-1]
    b = z[1:]
    s = a * a + b * b
    r = np.sqrt(s)
    denom = (1.0 + 0.001 * s) ** 2
    vals = 0.5 + (np.sin(r) ** 2 - 0.5) / denom
    return float(np.mean(vals))


def cec_shifted_lunacek_bi_rastrigin_function(x: np.ndarray) -> float:
    """Shifted Lunacek bi-Rastrigin Function (CEC F7 variant without rotation).

    Using common parameters: mu1=2.5, s = 1 - 1/(2*sqrt(D+20) - 8.2), mu2 = -sqrt((mu1^2-1)/s).

    f(x) = min( sum((z - mu1)^2), D + s * sum((z - mu2)^2) )
           + 10 * sum(1 - cos(2*pi*(z - mu1)))

    where z = x - shift.
    """
    z = x - 0.5
    d = x.shape[0]
    mu1 = 2.5
    s = 1.0 - 1.0 / (2.0 * np.sqrt(d + 20.0) - 8.2)
    mu2 = -np.sqrt((mu1**2 - 1.0) / s)
    zm1 = z - mu1
    zm2 = z - mu2
    term_quadratic_1 = float(np.dot(zm1, zm1))
    term_quadratic_2 = float(d + s * np.dot(zm2, zm2))
    rastrigin_term = float(10.0 * np.sum(1.0 - np.cos(PI2 * zm1)))
    return float(min(term_quadratic_1, term_quadratic_2) + rastrigin_term)


def cec_shifted_levy_function(x: np.ndarray) -> float:
    """Shifted Levy Function (CEC F9 variant without rotation).

    Standard Levy with shift: w_i = 1 + (z_i - 1)/4, z = x - shift.

    f(x) = sin^2(pi*w_1)
           + sum_{i=1}^{n-1} (w_i - 1)^2 * [1 + 10*sin^2(pi*w_i + 1)]
           + (w_n - 1)^2 * [1 + sin^2(2*pi*w_n)].

    Args:
        x (np.ndarray): Input vector.

    Returns:
        float: Function value.
    """
    z = x - 0.5
    w = 1.0 + (z - 1.0) / 4.0
    term1 = np.sin(np.pi * w[0]) ** 2
    if x.shape[0] > 1:
        w_head = w[:-1]
        t = w_head - 1.0
        term2 = np.sum(t * t * (1.0 + 10.0 * (np.sin(np.pi * w_head + 1.0) ** 2)))
    else:
        term2 = 0.0
    t_last = w[-1] - 1.0
    term3 = t_last * t_last * (1.0 + (np.sin(2.0 * np.pi * w[-1]) ** 2))
    return float(term1 + term2 + term3)


def cec_shifted_happy_cat_function(x: np.ndarray) -> float:
    """Shifted Happy Cat Function.

    f(x) = |sum(z^2) - n|^(0.25) + (0.5 * sum(z^2) + sum(z))/n + 0.5,
    where z = x - shift.

    Args:
        x (np.ndarray): Input vector.

    Returns:
        float: Function value.
    """
    z = x - 0.5
    n = x.shape[0]
    sum_z2 = float(np.dot(z, z))
    term1 = np.abs(sum_z2 - n) ** 0.25
    term2 = (0.5 * sum_z2 + np.sum(z)) / n + 0.5
    return float(term1 + term2)


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
    z = x - 0.5
    n = x.shape[0]
    sum_z2 = float(np.dot(z, z))
    return float(np.abs(sum_z2 - n) ** 0.5 + (0.5 * sum_z2 + np.sum(z)) / n + 0.5)


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
    z = x - 0.5
    z_nc = np.where(np.abs(z) > 0.5, np.round(z * 2) / 2.0, z)
    n = x.shape[0]
    return float(10 * n + np.sum(z_nc**2 - 10 * np.cos(PI2 * z_nc)))


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
    return float(w1 * f1 + w2 * f2 + w3 * f3)


# ----------------------------
# Hybrid helper and functions
# ----------------------------
def _hybrid_weighted_sum(x: np.ndarray, funcs: list, weights: list[float]) -> float:
    values = [f(x) for f in funcs]
    total = float(np.sum(np.array(values) * np.array(weights)))
    return total


def cec_hybrid_1(x: np.ndarray) -> float:
    """Hybrid Function 1 (N=3): Sphere, Rastrigin, Ackley."""
    funcs = [
        cec_shifted_sphere_function,
        cec_shifted_rastrigin_function,
        cec_shifted_ackley_function,
    ]
    weights = [0.3, 0.3, 0.4]
    return _hybrid_weighted_sum(x, funcs, weights)


def cec_hybrid_2(x: np.ndarray) -> float:
    """Hybrid Function 2 (N=3): Rosenbrock, Griewank, Schwefel."""
    funcs = [
        cec_shifted_rosenbrock_function,
        cec_shifted_griewank_function,
        cec_shifted_schwefel_function,
    ]
    weights = [0.3, 0.3, 0.4]
    return _hybrid_weighted_sum(x, funcs, weights)


def cec_hybrid_3(x: np.ndarray) -> float:
    """Hybrid Function 3 (N=3): Bent Cigar, Discus, Elliptic."""
    funcs = [
        cec_shifted_bent_cigar_function,
        cec_shifted_discus_function,
        cec_shifted_elliptic_function,
    ]
    weights = [0.34, 0.33, 0.33]
    return _hybrid_weighted_sum(x, funcs, weights)


def cec_hybrid_4(x: np.ndarray) -> float:
    """Hybrid Function 4 (N=4): Sphere, Rosenbrock, Rastrigin, Ackley."""
    funcs = [
        cec_shifted_sphere_function,
        cec_shifted_rosenbrock_function,
        cec_shifted_rastrigin_function,
        cec_shifted_ackley_function,
    ]
    weights = [0.25, 0.25, 0.25, 0.25]
    return _hybrid_weighted_sum(x, funcs, weights)


def cec_hybrid_5(x: np.ndarray) -> float:
    """Hybrid Function 5 (N=4): Griewank, Schwefel, Happy Cat, HGBat."""
    funcs = [
        cec_shifted_griewank_function,
        cec_shifted_schwefel_function,
        cec_shifted_happy_cat_function,
        cec_shifted_hgbat_function,
    ]
    weights = [0.25, 0.25, 0.25, 0.25]
    return _hybrid_weighted_sum(x, funcs, weights)


def cec_hybrid_6(x: np.ndarray) -> float:
    """Hybrid Function 6 (N=4): Weierstrass, Non-Continuous Rastrigin, Expanded Scaffer F6, Levy."""
    funcs = [
        cec_shifted_weierstrass_function,
        cec_shifted_non_continuous_rastrigin_function,
        cec_expanded_scaffer_f6,
        cec_shifted_levy_function,
    ]
    weights = [0.25, 0.25, 0.25, 0.25]
    return _hybrid_weighted_sum(x, funcs, weights)


def cec_hybrid_7(x: np.ndarray) -> float:
    """Hybrid Function 7 (N=5): Sphere, Zakharov, Rosenbrock, Rastrigin, Ackley."""
    funcs = [
        cec_shifted_sphere_function,
        cec_shifted_zakharov_function,
        cec_shifted_rosenbrock_function,
        cec_shifted_rastrigin_function,
        cec_shifted_ackley_function,
    ]
    weights = [0.2, 0.2, 0.2, 0.2, 0.2]
    return _hybrid_weighted_sum(x, funcs, weights)


def cec_hybrid_8(x: np.ndarray) -> float:
    """Hybrid Function 8 (N=5): Griewank, Schwefel, Levy, Weierstrass, Expanded Scaffer F6."""
    funcs = [
        cec_shifted_griewank_function,
        cec_shifted_schwefel_function,
        cec_shifted_levy_function,
        cec_shifted_weierstrass_function,
        cec_expanded_scaffer_f6,
    ]
    weights = [0.2, 0.2, 0.2, 0.2, 0.2]
    return _hybrid_weighted_sum(x, funcs, weights)


def cec_hybrid_9(x: np.ndarray) -> float:
    """Hybrid Function 9 (N=5): Bent Cigar, Discus, Elliptic, Happy Cat, HGBat."""
    funcs = [
        cec_shifted_bent_cigar_function,
        cec_shifted_discus_function,
        cec_shifted_elliptic_function,
        cec_shifted_happy_cat_function,
        cec_shifted_hgbat_function,
    ]
    weights = [0.2, 0.2, 0.2, 0.2, 0.2]
    return _hybrid_weighted_sum(x, funcs, weights)


def cec_hybrid_10(x: np.ndarray) -> float:
    """Hybrid Function 10 (N=6): Sphere, Zakharov, Rosenbrock, Rastrigin, Schwefel, Ackley."""
    funcs = [
        cec_shifted_sphere_function,
        cec_shifted_zakharov_function,
        cec_shifted_rosenbrock_function,
        cec_shifted_rastrigin_function,
        cec_shifted_schwefel_function,
        cec_shifted_ackley_function,
    ]
    weights = [1 / 6.0] * 6
    return _hybrid_weighted_sum(x, funcs, weights)


# ---------------------------------
# Composition helper and functions
# ---------------------------------
def _composition_weighted_sum(
    x: np.ndarray,
    funcs: list,
    biases: list[float],
    sigmas: list[float],
) -> float:
    z = x - 0.5
    d2 = float(np.sum(z**2))
    ws = np.array([np.exp(-d2 / (2.0 * (s**2))) for s in sigmas])
    if float(np.sum(ws)) == 0.0:
        ws = np.ones_like(ws)
    ws = ws / float(np.sum(ws))
    vals = np.array([f(x) for f in funcs], dtype=float)
    total = float(np.sum(ws * (vals + np.array(biases, dtype=float))))
    return total


def cec_composition_1(x: np.ndarray) -> float:
    """Composition Function 1 (N=3)."""
    funcs = [
        cec_shifted_sphere_function,
        cec_shifted_rastrigin_function,
        cec_shifted_ackley_function,
    ]
    biases = [0.0, 100.0, 200.0]
    sigmas = [10.0, 20.0, 30.0]
    return _composition_weighted_sum(x, funcs, biases, sigmas)


def cec_composition_2(x: np.ndarray) -> float:
    """Composition Function 2 (N=3)."""
    funcs = [
        cec_shifted_rosenbrock_function,
        cec_shifted_griewank_function,
        cec_shifted_schwefel_function,
    ]
    biases = [0.0, 100.0, 200.0]
    sigmas = [10.0, 15.0, 20.0]
    return _composition_weighted_sum(x, funcs, biases, sigmas)


def cec_composition_3(x: np.ndarray) -> float:
    """Composition Function 3 (N=4)."""
    funcs = [
        cec_shifted_bent_cigar_function,
        cec_shifted_discus_function,
        cec_shifted_elliptic_function,
        cec_shifted_weierstrass_function,
    ]
    biases = [0.0, 100.0, 200.0, 300.0]
    sigmas = [10.0, 15.0, 20.0, 25.0]
    return _composition_weighted_sum(x, funcs, biases, sigmas)


def cec_composition_4(x: np.ndarray) -> float:
    """Composition Function 4 (N=4)."""
    funcs = [
        cec_shifted_sphere_function,
        cec_shifted_rosenbrock_function,
        cec_shifted_rastrigin_function,
        cec_shifted_ackley_function,
    ]
    biases = [0.0, 100.0, 200.0, 300.0]
    sigmas = [5.0, 10.0, 15.0, 20.0]
    return _composition_weighted_sum(x, funcs, biases, sigmas)


def cec_composition_5(x: np.ndarray) -> float:
    """Composition Function 5 (N=5)."""
    funcs = [
        cec_shifted_griewank_function,
        cec_shifted_schwefel_function,
        cec_shifted_levy_function,
        cec_shifted_weierstrass_function,
        cec_expanded_scaffer_f6,
    ]
    biases = [0.0, 100.0, 200.0, 300.0, 400.0]
    sigmas = [10.0, 10.0, 15.0, 20.0, 25.0]
    return _composition_weighted_sum(x, funcs, biases, sigmas)


def cec_composition_6(x: np.ndarray) -> float:
    """Composition Function 6 (N=5)."""
    funcs = [
        cec_shifted_bent_cigar_function,
        cec_shifted_discus_function,
        cec_shifted_elliptic_function,
        cec_shifted_happy_cat_function,
        cec_shifted_hgbat_function,
    ]
    biases = [0.0, 100.0, 200.0, 300.0, 400.0]
    sigmas = [5.0, 10.0, 15.0, 10.0, 10.0]
    return _composition_weighted_sum(x, funcs, biases, sigmas)


def cec_composition_7(x: np.ndarray) -> float:
    """Composition Function 7 (N=6)."""
    funcs = [
        cec_shifted_sphere_function,
        cec_shifted_zakharov_function,
        cec_shifted_rosenbrock_function,
        cec_shifted_rastrigin_function,
        cec_shifted_schwefel_function,
        cec_shifted_ackley_function,
    ]
    biases = [0.0, 100.0, 200.0, 300.0, 400.0, 500.0]
    sigmas = [5.0, 7.5, 10.0, 12.5, 15.0, 17.5]
    return _composition_weighted_sum(x, funcs, biases, sigmas)


def cec_composition_8(x: np.ndarray) -> float:
    """Composition Function 8 (N=6)."""
    funcs = [
        cec_shifted_griewank_function,
        cec_shifted_schwefel_function,
        cec_shifted_levy_function,
        cec_shifted_weierstrass_function,
        cec_expanded_scaffer_f6,
        cec_shifted_non_continuous_rastrigin_function,
    ]
    biases = [0.0, 100.0, 200.0, 300.0, 400.0, 500.0]
    sigmas = [7.0, 9.0, 11.0, 13.0, 15.0, 17.0]
    return _composition_weighted_sum(x, funcs, biases, sigmas)


def cec_composition_9(x: np.ndarray) -> float:
    """Composition Function 9 (N=3)."""
    funcs = [
        cec_shifted_bent_cigar_function,
        cec_shifted_discus_function,
        cec_shifted_elliptic_function,
    ]
    biases = [0.0, 100.0, 200.0]
    sigmas = [5.0, 10.0, 15.0]
    return _composition_weighted_sum(x, funcs, biases, sigmas)


def cec_composition_10(x: np.ndarray) -> float:
    """Composition Function 10 (N=3)."""
    funcs = [
        cec_shifted_happy_cat_function,
        cec_shifted_hgbat_function,
        cec_shifted_rastrigin_function,
    ]
    biases = [0.0, 100.0, 200.0]
    sigmas = [5.0, 10.0, 15.0]
    return _composition_weighted_sum(x, funcs, biases, sigmas)


# List of CEC benchmark functions
cec_funcs: list["ExperimentFunction"] = [
    {
        "name": "CEC Shifted Sphere",
        "call": cec_shifted_sphere_function,
        "domain": (-100, 100),
        "optimal_value": 0.0,
        # The function is f(x) = Σ (xᵢ – 0.5)².
        # It is minimized (with value 0) when xᵢ = 0.5 for every i
        "optimal_x_value": [0.5],
    },
    {
        "name": "CEC Shifted Sum of Different Powers",
        "call": cec_shifted_sum_of_different_powers_function,
        "domain": (-100, 100),
        "optimal_value": 0.0,
        "optimal_x_value": [0.5],
    },
    {
        "name": "CEC Shifted Zakharov",
        "call": cec_shifted_zakharov_function,
        "domain": (-100, 100),
        "optimal_value": 0.0,
        "optimal_x_value": [0.5],
    },
    {
        "name": "CEC Shifted Rosenbrock",
        "call": cec_shifted_rosenbrock_function,
        "domain": (-30, 30),
        "optimal_value": 0.0,
        # he shift the function uses z = x – 0.5 + 1 so that the
        # usual Rosenbrock formulation becomes
        # Σ[100*(zᵢ₊₁ – zᵢ²)² + (zᵢ – 1)²]
        "optimal_x_value": [0.5],
    },
    {
        "name": "CEC Shifted Levy",
        "call": cec_shifted_levy_function,
        "domain": (-100, 100),
        "optimal_value": 0.0,
        "optimal_x_value": [1.5],
    },
    {
        "name": "CEC Shifted Rastrigin",
        "call": cec_shifted_rastrigin_function,
        "domain": (-5.12, 5.12),
        "optimal_value": 0.0,
        # With f(x) = 10·d + Σ[(xᵢ – 0.5)² – 10 cos(2π(xᵢ – 0.5))]
        # the best value is reached when (xᵢ – 0.5) = 0
        # (so that cos(0)=1), yielding 10·d – 10·d = 0
        "optimal_x_value": [0.5],
    },
    {
        "name": "CEC Hybrid Function 1 (N=3)",
        "call": cec_hybrid_1,
        "domain": (-100, 100),
        "optimal_value": 0.0,
        "optimal_x_value": [0.5],
    },
    {
        "name": "CEC Hybrid Function 2 (N=3)",
        "call": cec_hybrid_2,
        "domain": (-100, 100),
        "optimal_value": 0.0,
        "optimal_x_value": [0.5],
    },
    {
        "name": "CEC Hybrid Function 3 (N=3)",
        "call": cec_hybrid_3,
        "domain": (-100, 100),
        "optimal_value": 0.0,
        "optimal_x_value": [0.5],
    },
    {
        "name": "CEC Hybrid Function 4 (N=4)",
        "call": cec_hybrid_4,
        "domain": (-100, 100),
        "optimal_value": 0.0,
        "optimal_x_value": [0.5],
    },
    {
        "name": "CEC Hybrid Function 5 (N=4)",
        "call": cec_hybrid_5,
        "domain": (-100, 100),
        "optimal_value": 0.0,
        "optimal_x_value": [0.5],
    },
    {
        "name": "CEC Hybrid Function 6 (N=4)",
        "call": cec_hybrid_6,
        "domain": (-100, 100),
        "optimal_value": 0.0,
        "optimal_x_value": [0.5],
    },
    {
        "name": "CEC Hybrid Function 7 (N=5)",
        "call": cec_hybrid_7,
        "domain": (-100, 100),
        "optimal_value": 0.0,
        "optimal_x_value": [0.5],
    },
    {
        "name": "CEC Hybrid Function 8 (N=5)",
        "call": cec_hybrid_8,
        "domain": (-100, 100),
        "optimal_value": 0.0,
        "optimal_x_value": [0.5],
    },
    {
        "name": "CEC Hybrid Function 9 (N=5)",
        "call": cec_hybrid_9,
        "domain": (-100, 100),
        "optimal_value": 0.0,
        "optimal_x_value": [0.5],
    },
    {
        "name": "CEC Hybrid Function 10 (N=6)",
        "call": cec_hybrid_10,
        "domain": (-100, 100),
        "optimal_value": 0.0,
        "optimal_x_value": [0.5],
    },
    {
        "name": "CEC Composition Function 1 (N=3)",
        "call": cec_composition_1,
        "domain": (-100, 100),
        "optimal_value": 0.0,
        "optimal_x_value": [0.5],
    },
    {
        "name": "CEC Composition Function 2 (N=3)",
        "call": cec_composition_2,
        "domain": (-100, 100),
        "optimal_value": 0.0,
        "optimal_x_value": [0.5],
    },
    {
        "name": "CEC Composition Function 3 (N=4)",
        "call": cec_composition_3,
        "domain": (-100, 100),
        "optimal_value": 0.0,
        "optimal_x_value": [0.5],
    },
    {
        "name": "CEC Composition Function 4 (N=4)",
        "call": cec_composition_4,
        "domain": (-100, 100),
        "optimal_value": 0.0,
        "optimal_x_value": [0.5],
    },
    {
        "name": "CEC Composition Function 5 (N=5)",
        "call": cec_composition_5,
        "domain": (-100, 100),
        "optimal_value": 0.0,
        "optimal_x_value": [0.5],
    },
    {
        "name": "CEC Composition Function 6 (N=5)",
        "call": cec_composition_6,
        "domain": (-100, 100),
        "optimal_value": 0.0,
        "optimal_x_value": [0.5],
    },
    {
        "name": "CEC Composition Function 7 (N=6)",
        "call": cec_composition_7,
        "domain": (-100, 100),
        "optimal_value": 0.0,
        "optimal_x_value": [0.5],
    },
    {
        "name": "CEC Composition Function 8 (N=6)",
        "call": cec_composition_8,
        "domain": (-100, 100),
        "optimal_value": 0.0,
        "optimal_x_value": [0.5],
    },
    {
        "name": "CEC Composition Function 9 (N=3)",
        "call": cec_composition_9,
        "domain": (-100, 100),
        "optimal_value": 0.0,
        "optimal_x_value": [0.5],
    },
    {
        "name": "CEC Composition Function 10 (N=3)",
        "call": cec_composition_10,
        "domain": (-100, 100),
        "optimal_value": 0.0,
        "optimal_x_value": [0.5],
    },
    {
        "name": "CEC Shifted Schwefel",
        "call": cec_shifted_schwefel_function,
        "domain": (-500, 500),
        "optimal_value": 0.0,
        # The Schwefel function is written as f(x) = 418.9829·d –
        # Σ[(xᵢ – 420.968746)·sin(√|xᵢ – 420.968746|)]
        # Here the “shift” is defined as 420.968746; therefore the
        # function’s z value is x – 420.968746 and we need
        # z = 420.968746. In other words, optimal x is
        # 420.968746 + 420.968746 = 841.937492 in every coordinate
        "optimal_x_value": [841.937492],
    },
    {
        "name": "CEC Shifted Griewank",
        "call": cec_shifted_griewank_function,
        "domain": (-600, 600),
        "optimal_value": 0.0,
        # Since f(x) = (Σ (xᵢ – 0.5)²)/4000 – ∏ cos((xᵢ – 0.5)/√(i)) + 1,
        # setting x = 0.5 makes the squared term zero and each cosine
        # becomes cos(0)=1. Hence f = 0 – 1 + 1 = 0
        "optimal_x_value": [0.5],
    },
    {
        "name": "CEC Shifted Lunacek bi-Rastrigin",
        "call": cec_shifted_lunacek_bi_rastrigin_function,
        "domain": (-100, 100),
        "optimal_value": 0.0,
        "optimal_x_value": [0.5],
    },
    {
        "name": "CEC Shifted Ackley",
        "call": cec_shifted_ackley_function,
        "domain": (-32, 32),
        "optimal_value": 0.0,
        # With z = x – 0.5 the Ackley terms are optimized by
        # having z = 0 (so that the exponential terms take the
        # “best‐case” values) yielding f = 0.
        "optimal_x_value": [0.5],
    },
    # {
    #     "name": "CEC Shifted Weierstrass",
    #     "call": cec_shifted_weierstrass_function,
    #     "domain": (-0.5, 0.5),
    #     "optimal_value": 0.0,
    #     "optimal_x_value": [0.5],
    # },
    {
        "name": "CEC Shifted Bent Cigar",
        "call": cec_shifted_bent_cigar_function,
        "domain": (-100, 100),
        "optimal_value": 0.0,
        # f(x) = (x₁ – 0.5)² + 1e6 · Σ₍ᵢ₌₂₎ (xᵢ – 0.5)²
        # is minimized by forcing x – 0.5 = 0
        "optimal_x_value": [0.5],
    },
    {
        "name": "CEC Shifted Discus",
        "call": cec_shifted_discus_function,
        "domain": (-100, 100),
        "optimal_value": 0.0,
        # f(x) = 1e6 · (x₁ – 0.5)² + Σ₍ᵢ₌₂₎ (xᵢ – 0.5)²
        # is minimized when x = 0.5 in every coordinate
        "optimal_x_value": [0.5],
    },
    {
        "name": "CEC Shifted Elliptic",
        "call": cec_shifted_elliptic_function,
        "domain": (-100, 100),
        "optimal_value": 0.0,
        # f(x) = Σ₍ᵢ₌₁₎ (10^6)^(i/(D-1)) · (xᵢ – 0.5)²
        # is minimized when x = 0.5 in every coordinate
        "optimal_x_value": [0.5],
    },
    {
        "name": "CEC Expanded Scaffer F6",
        "call": cec_expanded_scaffer_f6,
        "domain": (-100, 100),
        "optimal_value": 0.0,
        # The function is defined (using z = x – 0.5) so that each
        # ScafferF6 term has its minimum (0) when its two arguments
        # are 0; thus x = 0.5 is optimal
        "optimal_x_value": [0.5],
    },
    {
        "name": "CEC Shifted Happy Cat",
        "call": cec_shifted_happy_cat_function,
        "domain": (-2, 2),
        "optimal_value": 0.0,
        "optimal_x_value": [-0.5],
    },
    {
        "name": "CEC Shifted HGBat",
        "call": cec_shifted_hgbat_function,
        "domain": (-100, 100),
        "optimal_value": 0.0,
        # Its definition is similar to the Happy Cat function
        # except that the exponent on the first term is 0.5 rather than 0.25.
        # if z = –1 then
        #   |Σ(z²) – d|^(0.5) = |d – d|^(0.5) = 0
        #   and (0.5·d – d)/d + 0.5 = 0, so the minimum is 0 when x = –0.5.
        "optimal_x_value": [-0.5],
    },
    {
        "name": "CEC Shifted Non-Continuous Rastrigin",
        "call": cec_shifted_non_continuous_rastrigin_function,
        "domain": (-5.12, 5.12),
        "optimal_value": 0.0,
        "optimal_x_value": [0.5],
    },
    {
        "name": "CEC Hybrid Composition",
        "call": cec_hybrid_composition_function,
        "domain": (-5, 5),
        "optimal_value": 0.0,
        "optimal_x_value": [0.5],
    },
]
