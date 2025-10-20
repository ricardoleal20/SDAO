"""
SOCO11 deterministic benchmark test functions (optimized/vectorized).

This module provides efficient NumPy implementations for the common SOCO
benchmarks and exposes them in a list of ExperimentFunction descriptors.

Note: This file currently includes 12 base functions (as before) with
optimized math. Integration of the remaining SOCO11 functions (for a total
of 19) will use the official shift/rotation data downloaded into
`third_party/soco11/` in a follow-up step.
"""

import math
import os
import re
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from model.solver import ExperimentFunction


def soco_sphere_function(x: np.ndarray) -> float:
    """Sphere: f(x) = sum(x_i^2)."""
    return float(np.dot(x, x))


def soco_rosenbrock_function(x: np.ndarray) -> float:
    """Rosenbrock (a=1, b=100): sum[100(x_{i+1} - x_i^2)^2 + (1 - x_i)^2]."""
    if x.shape[0] < 2:
        return 0.0
    xi = x[:-1]
    xnext = x[1:]
    return float(np.sum(100.0 * (xnext - xi * xi) ** 2 + (1.0 - xi) ** 2))


def soco_rastrigin_function(x: np.ndarray) -> float:
    """Rastrigin: 10*d + sum[x_i^2 - 10 cos(2π x_i)]."""
    d = x.shape[0]
    a = 10.0
    return float(a * d + (np.dot(x, x) - np.sum(a * np.cos(2.0 * math.pi * x))))


def soco_ackley_function(x: np.ndarray) -> float:
    """Ackley standard form."""
    d = max(1, x.shape[0])
    sum_sq = float(np.dot(x, x))
    term1 = -20.0 * math.exp(-0.2 * math.sqrt(sum_sq / d))
    mean_cos = float(np.cos(2.0 * math.pi * x).mean())
    term2 = -math.exp(mean_cos)
    return float(term1 + term2 + 20.0 + math.e)


def soco_griewank_function(x: np.ndarray) -> float:
    """Griewank: sum(x_i^2)/4000 - prod(cos(x_i/sqrt(i))) + 1."""
    d = x.shape[0]
    sum_sq = float(np.dot(x, x)) / 4000.0
    if d == 0:
        prod_cos = 1.0
    else:
        idx = np.sqrt(np.arange(1, d + 1, dtype=float))
        prod_cos = float(np.prod(np.cos(x / idx)))
    return float(sum_sq - prod_cos + 1.0)


def soco_schwefel_function(x: np.ndarray) -> float:
    """Schwefel 2.26: 418.9829*d - sum[x_i sin(sqrt(|x_i|))]."""
    d = x.shape[0]
    return float(418.9829 * d - np.sum(x * np.sin(np.sqrt(np.abs(x)))))


def soco_weierstrass_function(x: np.ndarray) -> float:
    """Weierstrass: sum_i sum_{k=0}^{20} a^k cos(2π b^k(x_i+0.5)) - d*sum_{k} a^k cos(2π b^k*0.5)."""
    a = 0.5
    b = 3.0
    d = x.shape[0]
    k_values = np.arange(21, dtype=float)
    a_powers = a**k_values
    b_powers = b**k_values
    x_expanded = x[:, np.newaxis] + 0.5  # (d, 1)
    cos_args = 2.0 * math.pi * b_powers * x_expanded  # (d, 21)
    cos_values = np.cos(cos_args)
    sum1 = float(np.sum(a_powers * cos_values))
    sum2 = float(np.sum(a_powers * np.cos(2.0 * math.pi * b_powers * 0.5)))
    return float(sum1 - d * sum2)


def soco_bent_cigar_function(x: np.ndarray) -> float:
    """Bent Cigar: x_1^2 + 1e6 * sum_{i=2..d} x_i^2."""
    if x.shape[0] == 0:
        return 0.0
    return float(x[0] ** 2 + 1e6 * np.dot(x[1:], x[1:]))


def soco_discus_function(x: np.ndarray) -> float:
    """Discus: 1e6 * x_1^2 + sum_{i=2..d} x_i^2."""
    if x.shape[0] == 0:
        return 0.0
    return float(1e6 * x[0] ** 2 + np.dot(x[1:], x[1:]))


def soco_elliptic_function(x: np.ndarray) -> float:
    """Elliptic: sum[10^(6 * ((i-1)/(d-1))) * x_i^2]."""
    d = x.shape[0]
    if d == 0:
        return 0.0
    exponents = np.linspace(0.0, 1.0, d)
    weights = 10.0 ** (6.0 * exponents)
    return float(np.dot(weights, np.square(x)))


def soco_zakharov_function(x: np.ndarray) -> float:
    """Zakharov: sum(x_i^2) + (0.5 * sum(i*x_i))^2 + (0.5 * sum(i*x_i))^4."""
    d = x.shape[0]
    indices = np.arange(1, d + 1, dtype=float)
    s1 = float(np.dot(x, x))
    s2 = float(0.5 * np.dot(indices, x))
    return float(s1 + s2**2 + s2**4)


def soco_levy_function(x: np.ndarray) -> float:
    """Levy (standard): uses w_i = 1 + (x_i - 1)/4."""
    if x.shape[0] == 0:
        return 0.0
    w = 1.0 + (x - 1.0) / 4.0
    term1 = float(np.sin(np.pi * w[0]) ** 2)
    if x.shape[0] > 1:
        t = w[:-1] - 1.0
        term2 = float(
            np.sum(t * t * (1.0 + 10.0 * (np.sin(np.pi * w[:-1] + 1.0) ** 2)))
        )
    else:
        term2 = 0.0
    t_last = w[-1] - 1.0
    term3 = float(t_last * t_last * (1.0 + (np.sin(2.0 * np.pi * w[-1]) ** 2)))
    return float(term1 + term2 + term3)


# ============================= #
# Official SOCO11 shifted set   #
# ============================= #

_BASE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "third_party", "soco11"
)
_CEC08_DATA = os.path.join(_BASE_DIR, "cec08", "data.h")
_F7F11_DATA = os.path.join(_BASE_DIR, "updated-F7-F11", "f7f11data.h")
_F12_19_DATA = os.path.join(_BASE_DIR, "functions12-19", "cec08data.h")


@lru_cache(maxsize=None)
def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _extract_array(path: str, var_name: str) -> np.ndarray:
    text = _read_text(path)
    m = re.search(
        rf"double\s+{re.escape(var_name)}\s*\[[^\]]*\]\s*=\s*\{{([^\}}]*)\}};",
        text,
        flags=re.DOTALL,
    )
    if not m:
        raise RuntimeError(f"Array {var_name} not found in {path}")
    body = m.group(1)
    nums = [
        float(tok)
        for tok in re.findall(
            r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+(?:[eE][-+]?\d+)?", body
        )
    ]
    return np.array(nums, dtype=float)


@lru_cache(maxsize=None)
def _bias_vector() -> np.ndarray:
    return _extract_array(_CEC08_DATA, "f_bias")


@lru_cache(maxsize=None)
def _shift_vec(name: str) -> np.ndarray:
    if name in ("sphere", "schwefel", "rosenbrock", "rastrigin", "griewank", "ackley"):
        return _extract_array(_CEC08_DATA, name)
    if name in ("f7", "f8", "f9", "f10", "f11", "f15", "f19") and os.path.exists(
        _F7F11_DATA
    ):
        return _extract_array(_F7F11_DATA, name)
    return _extract_array(_F12_19_DATA, name)


def _slice_shift(vec: np.ndarray, d: int) -> np.ndarray:
    if vec.shape[0] < d:
        raise ValueError("Shift vector too short for requested dimension")
    return vec[:d]


def so11_shifted_sphere(x: np.ndarray) -> float:
    z = x - _slice_shift(_shift_vec("sphere"), x.shape[0])
    return float(np.dot(z, z) + _bias_vector()[0])


def so11_schwefel_problem(x: np.ndarray) -> float:
    if x.shape[0] == 0:
        return float(_bias_vector()[1])
    F = abs(float(x[0]))
    sv = _slice_shift(_shift_vec("schwefel"), x.shape[0])
    z_rest = np.abs(x[1:] - sv[1:])
    if z_rest.size > 0:
        F = max(F, float(np.max(z_rest)))
    return float(F + _bias_vector()[1])


def so11_shifted_rosenbrock(x: np.ndarray) -> float:
    sv = _slice_shift(_shift_vec("rosenbrock"), x.shape[0])
    z = x - sv + 1.0
    if z.shape[0] < 2:
        return float(_bias_vector()[2])
    zi = z[:-1]
    zip1 = z[1:]
    F = float(np.sum(100.0 * (zi * zi - zip1) ** 2 + (zi - 1.0) ** 2))
    return float(F + _bias_vector()[2])


def so11_shifted_rastrigin(x: np.ndarray) -> float:
    sv = _slice_shift(_shift_vec("rastrigin"), x.shape[0])
    z = x - sv
    F = float(np.sum(z * z - 10.0 * np.cos(2.0 * math.pi * z) + 10.0))
    return float(F + _bias_vector()[3])


def so11_shifted_griewank(x: np.ndarray) -> float:
    sv = _slice_shift(_shift_vec("griewank"), x.shape[0])
    z = x - sv
    d = max(1, x.shape[0])
    F1 = float(np.dot(z, z)) / 4000.0
    idx = np.sqrt(np.arange(1, d + 1, dtype=float))
    F2 = float(np.prod(np.cos(z / idx))) if d > 0 else 1.0
    return float(F1 - F2 + 1.0 + _bias_vector()[4])


def so11_shifted_ackley(x: np.ndarray) -> float:
    sv = _slice_shift(_shift_vec("ackley"), x.shape[0])
    z = x - sv
    d = max(1, x.shape[0])
    Sum1 = float(np.dot(z, z))
    Sum2 = float(np.sum(np.cos(2.0 * math.pi * z)))
    F = (
        -20.0 * math.exp(-0.2 * math.sqrt(Sum1 / d))
        - math.exp(Sum2 / d)
        + 20.0
        + math.e
        + float(_bias_vector()[5])
    )
    return float(F)


def _f10_base(x: float, y: float) -> float:
    p = x * x + y * y
    z = p**0.25
    t = math.sin(50.0 * (p**0.1))
    return z * (t * t + 1.0)


def so11_schwefel2_22(x: np.ndarray) -> float:
    sv = _slice_shift(_shift_vec("f7"), x.shape[0])
    v = np.abs(x - sv)
    return float(np.sum(v) + np.prod(v))


def so11_schwefel1_2(x: np.ndarray) -> float:
    sv = _slice_shift(_shift_vec("f8"), x.shape[0])
    diff = x - sv
    csum = np.cumsum(diff)
    return float(np.sum(csum * csum))


def so11_extended_f10(x: np.ndarray) -> float:
    sv = _slice_shift(_shift_vec("f9"), x.shape[0])
    z = x - sv
    if z.shape[0] == 0:
        return 0.0
    total = 0.0
    for i in range(z.shape[0] - 1):
        total += _f10_base(float(z[i]), float(z[i + 1]))
    total += _f10_base(float(z[-1]), float(z[0]))
    return float(total)


def so11_bohachevsky(x: np.ndarray) -> float:
    sv = _slice_shift(_shift_vec("f10"), x.shape[0])
    z = x - sv
    if z.shape[0] < 2:
        return 0.0
    current = float(z[0])
    s = 0.0
    for i in range(1, z.shape[0]):
        nxt = float(z[i])
        s += current * current + 2.0 * nxt * nxt
        s += 0.7 - (
            0.3 * math.cos(3.0 * math.pi * current)
            + 0.4 * math.cos(4.0 * math.pi * nxt)
        )
        current = nxt
    return float(s)


def so11_schaffer(x: np.ndarray) -> float:
    sv = _slice_shift(_shift_vec("f11"), x.shape[0])
    z = x - sv
    if z.shape[0] < 2:
        return 0.0
    s = 0.0
    cur = float(z[0])
    cur2 = cur * cur
    for i in range(1, z.shape[0]):
        nxt = float(z[i])
        nxt2 = nxt * nxt
        aux = cur2 + nxt2
        cur2 = nxt2
        s += (aux**0.25) * (math.sin(50.0 * (aux**0.1)) ** 2 + 1.0)
    return float(s)


def _divide_functions(s: np.ndarray, m: float) -> tuple[np.ndarray, np.ndarray]:
    dim = s.shape[0]
    if m <= 0.5:
        shared = int(math.floor(dim * m))
        partrest_is_part2 = True
    else:
        shared = int(math.floor(dim * (1.0 - m)))
        partrest_is_part2 = False
    part1 = np.empty(dim, dtype=float)
    part2 = np.empty(dim, dtype=float)
    for i in range(shared):
        part1[i] = s[2 * i]
        part2[i] = s[2 * i + 1]
    total = dim - shared
    rest = 2 * shared
    if partrest_is_part2:
        part2[shared : shared + (total - shared)] = s[rest : rest + (total - shared)]
        size1 = shared
        size2 = dim - shared
        return part1[:size1].copy(), part2[:size2].copy()
    else:
        part1[shared : shared + (total - shared)] = s[rest : rest + (total - shared)]
        size1 = dim - shared
        size2 = shared
        return part1[:size1].copy(), part2[:size2].copy()


def so11_hybrid_12(x: np.ndarray) -> float:
    part1, part2 = _divide_functions(x, 0.25)
    f1 = so11_extended_f10(part1)
    f2 = so11_shifted_sphere(part2) - float(_bias_vector()[0])
    return float(f1 + f2)


def so11_hybrid_13(x: np.ndarray) -> float:
    part1, part2 = _divide_functions(x, 0.25)
    f1 = so11_extended_f10(part1)
    f2 = so11_shifted_rosenbrock(part2) - float(_bias_vector()[2])
    return float(f1 + f2)


def so11_hybrid_14(x: np.ndarray) -> float:
    part1, part2 = _divide_functions(x, 0.25)
    f1 = so11_extended_f10(part1)
    f2 = so11_shifted_rastrigin(part2) - float(_bias_vector()[3])
    return float(f1 + f2)


def so11_hybrid_15(x: np.ndarray) -> float:
    sv = _slice_shift(_shift_vec("f15"), x.shape[0])
    desp = x - sv
    part1, part2 = _divide_functions(desp, 0.25)
    f1 = so11_bohachevsky(part1)
    f2 = so11_schwefel2_22(part2)
    return float(f1 + f2)


def so11_hybrid_16(x: np.ndarray) -> float:
    part1, part2 = _divide_functions(x, 0.5)
    f1 = so11_extended_f10(part1)
    f2 = so11_shifted_sphere(part2) - float(_bias_vector()[0])
    return float(f1 + f2)


def so11_hybrid_17(x: np.ndarray) -> float:
    part1, part2 = _divide_functions(x, 0.75)
    f1 = so11_extended_f10(part1)
    f2 = so11_shifted_rosenbrock(part2) - float(_bias_vector()[2])
    return float(f1 + f2)


def so11_hybrid_18(x: np.ndarray) -> float:
    part1, part2 = _divide_functions(x, 0.75)
    f1 = so11_extended_f10(part1)
    f2 = so11_shifted_rastrigin(part2) - float(_bias_vector()[3])
    return float(f1 + f2)


def so11_hybrid_19(x: np.ndarray) -> float:
    sv = _slice_shift(_shift_vec("f19"), x.shape[0])
    desp = x - sv
    part1, part2 = _divide_functions(desp, 0.75)
    f1 = so11_bohachevsky(part1)
    f2 = so11_schwefel2_22(part2)
    return float(f1 + f2)


# ========================================================= #
# DEFINE THE OFFICIAL 19 SOCO11 FUNCTIONS                   #
# ========================================================= #
soco_funcs: list["ExperimentFunction"] = [
    {
        "name": "SOCO11 Shifted Sphere",
        "call": so11_shifted_sphere,
        "domain": (-100.0, 100.0),
        "optimal_x_value": [0.0],
        "optimal_value": 0.0,
        "dimension": 500,
    },
    {
        "name": "SOCO11 Schwefel Problem (max-abs)",
        "call": so11_schwefel_problem,
        "domain": (-100.0, 100.0),
        "optimal_x_value": [0.0],
        "optimal_value": 0.0,
        "dimension": 500,
    },
    {
        "name": "SOCO11 Shifted Rosenbrock",
        "call": so11_shifted_rosenbrock,
        "domain": (-30.0, 30.0),
        "optimal_x_value": [1.0],
        "optimal_value": 0.0,
        "dimension": 500,
    },
    {
        "name": "SOCO11 Shifted Rastrigin",
        "call": so11_shifted_rastrigin,
        "domain": (-5.12, 5.12),
        "optimal_x_value": [0.0],
        "optimal_value": 0.0,
        "dimension": 500,
    },
    {
        "name": "SOCO11 Shifted Griewank",
        "call": so11_shifted_griewank,
        "domain": (-600.0, 600.0),
        "optimal_x_value": [0.0],
        "optimal_value": 0.0,
        "dimension": 500,
    },
    {
        "name": "SOCO11 Shifted Ackley",
        "call": so11_shifted_ackley,
        "domain": (-32.768, 32.768),
        "optimal_x_value": [0.0],
        "optimal_value": 0.0,
        "dimension": 500,
    },
    {
        "name": "SOCO11 Schwefel 2.22 (shifted)",
        "call": so11_schwefel2_22,
        "domain": (-100.0, 100.0),
        "optimal_x_value": [0.0],
        "optimal_value": 0.0,
        "dimension": 500,
    },
    {
        "name": "SOCO11 Schwefel 1.2 (shifted)",
        "call": so11_schwefel1_2,
        "domain": (-100.0, 100.0),
        "optimal_x_value": [0.0],
        "optimal_value": 0.0,
        "dimension": 500,
    },
    {
        "name": "SOCO11 Extended f10 (shifted)",
        "call": so11_extended_f10,
        "domain": (-100.0, 100.0),
        "optimal_x_value": [0.0],
        "optimal_value": 0.0,
        "dimension": 500,
    },
    {
        "name": "SOCO11 Bohachevsky (shifted)",
        "call": so11_bohachevsky,
        "domain": (-100.0, 100.0),
        "optimal_x_value": [0.0],
        "optimal_value": 0.0,
        "dimension": 500,
    },
    {
        "name": "SOCO11 Schaffer (shifted)",
        "call": so11_schaffer,
        "domain": (-100.0, 100.0),
        "optimal_x_value": [0.0],
        "optimal_value": 0.0,
        "dimension": 500,
    },
    {
        "name": "SOCO11 Hybrid 12",
        "call": so11_hybrid_12,
        "domain": (-100.0, 100.0),
        "optimal_x_value": [0.0],
        "optimal_value": 0.0,
        "dimension": 500,
    },
    {
        "name": "SOCO11 Hybrid 13",
        "call": so11_hybrid_13,
        "domain": (-100.0, 100.0),
        "optimal_x_value": [0.0],
        "optimal_value": 0.0,
        "dimension": 500,
    },
    {
        "name": "SOCO11 Hybrid 14",
        "call": so11_hybrid_14,
        "domain": (-100.0, 100.0),
        "optimal_x_value": [0.0],
        "optimal_value": 0.0,
        "dimension": 500,
    },
    {
        "name": "SOCO11 Hybrid 15",
        "call": so11_hybrid_15,
        "domain": (-100.0, 100.0),
        "optimal_x_value": [0.0],
        "optimal_value": 0.0,
        "dimension": 500,
    },
    {
        "name": "SOCO11 Hybrid 16",
        "call": so11_hybrid_16,
        "domain": (-100.0, 100.0),
        "optimal_x_value": [0.0],
        "optimal_value": 0.0,
        "dimension": 500,
    },
    {
        "name": "SOCO11 Hybrid 17",
        "call": so11_hybrid_17,
        "domain": (-100.0, 100.0),
        "optimal_x_value": [0.0],
        "optimal_value": 0.0,
        "dimension": 500,
    },
    {
        "name": "SOCO11 Hybrid 18",
        "call": so11_hybrid_18,
        "domain": (-100.0, 100.0),
        "optimal_x_value": [0.0],
        "optimal_value": 0.0,
        "dimension": 500,
    },
    {
        "name": "SOCO11 Hybrid 19",
        "call": so11_hybrid_19,
        "domain": (-100.0, 100.0),
        "optimal_x_value": [0.0],
        "optimal_value": 0.0,
        "dimension": 500,
    },
]
