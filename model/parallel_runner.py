"""
Parallel runner helpers placed in an importable module to support spawn start
method on macOS/Windows. Workers import functions from here instead of __main__.
"""

from __future__ import annotations

from functools import partial
import os
import random
import warnings
from typing import List, Tuple

import numpy as np

from model.functions.bench_funcs import bench_funcs
from model.functions.stoch_funcs import stoch_funcs
from model.functions.real_life_funcs import real_life_funcs
from model.functions.cec_funcs import cec_funcs
from model.functions.mpb_funcs import mpb_benchmarks
from model.solver import Solver, ExperimentFunction, BenchmarkResult
from model.soa.template import Algorithm
from model.sdao import SDAO
from model.soa.fractal import StochasticFractalSearch
from model.soa.algebraic_sgd import AlgebraicSGD
from model.soa.shade import SHADEwithILS
from model.soa.path_relinking import PathRelinking
from model.soa.amso import AMSO
from model.soa.tlpso import TLPSO
from model.soa.sfoa import SFOA
from model.soa.pade_pet import PaDE_PET


# Define the Cities for the VRP problem (duplicated here for worker isolation)
travel_times = np.array(
    [
        [0, 2550, 780, 1620, 2140, 120, 1700, 2490, 1540, 2600],
        [2550, 0, 2010, 1370, 380, 2570, 1200, 120, 1440, 340],
        [780, 2010, 0, 1090, 1730, 860, 1260, 1980, 970, 2100],
        [1620, 1370, 1090, 0, 1170, 1590, 200, 1300, 240, 1460],
        [2140, 380, 1730, 1170, 0, 2200, 980, 390, 1050, 650],
        [120, 2570, 860, 1590, 2200, 0, 1650, 2510, 1520, 2650],
        [1700, 1200, 1260, 200, 980, 1650, 0, 1280, 280, 1350],
        [2490, 120, 1980, 1300, 390, 2510, 1280, 0, 1380, 420],
        [1540, 1440, 970, 240, 1050, 1520, 280, 1380, 0, 1450],
        [2600, 340, 2100, 1460, 650, 2650, 1350, 420, 1450, 0],
    ]
)
deadlines = [2400, 3000, 1200, 1800, 2500, 2000, 1600, 2800, 1700, 3200]


def set_thread_env() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def functions_due_to_scenario(scenario: int) -> List[ExperimentFunction]:
    match scenario:
        case 0:
            return bench_funcs
        case 1:
            return stoch_funcs
        case 2:
            warnings.warn(
                "\033[93mThe dimension is going to be ignored for this scenario. "
                + "Each example has his own dimension set for the demo.\033[0m"
            )
            for func in real_life_funcs:
                match func["name"]:
                    case "Predictive Maintenance":
                        failure_probs = np.random.uniform(0, 1, size=30)
                        repair_costs = np.random.uniform(0, 100, size=30)
                        func["call"] = partial(
                            func["call"],
                            failure_probs=failure_probs,  # type: ignore
                            repair_costs=repair_costs,  # type: ignore
                            downtime_costs=10,  # type: ignore
                        )
                    case "VRP":
                        func["call"] = partial(
                            func["call"],
                            travel_times=travel_times,  # type: ignore
                            deadlines=deadlines,  # type: ignore
                            traffic_noise_std=0.5,  # type: ignore
                        )
                    case _:
                        pass
            return real_life_funcs
        case 3:
            return cec_funcs
        case 4:
            warnings.warn(
                "\033[93mThe dimension is going to be ignored for this scenario. "
                + "Each example has his own dimension set for the demo.\033[0m"
            )
            return mpb_benchmarks
        case _:
            raise NotImplementedError(f"Invalid scenario. Scenario: {scenario}")


def create_algorithm(key: str, iterations: int, verbose: bool) -> Algorithm:
    match key:
        case "sdao":
            return SDAO(
                num_iterations=iterations,
                num_particles=50,
                version=1,
                params={
                    "learning_rate": 0.08009472801977902,
                    "memory_coeff": 0.6462704921052449,
                    "diffusion_coeff": 3.952340863752026,
                    "density_radius": 2.2081413057223624,
                    "decay_rate": 0.07500855901413722,
                    "contract_every": 10,
                },
                verbose=verbose,
            )
        case "sfs":
            return StochasticFractalSearch(
                n_population=50,
                n_iterations=iterations,
                fractal_factor=0.9,
                verbose=verbose,
            )
        case "sgd":
            return AlgebraicSGD(n_iterations=iterations, verbose=verbose)
        case "shade":
            return SHADEwithILS(
                n_population=50,
                n_iterations=iterations,
                memory_size=10,
                verbose=verbose,
            )
        case "path_relinking":
            return PathRelinking(
                n_population=50,
                n_iterations=iterations,
                elite_ratio=0.2,
                verbose=verbose,
            )
        case "amso":
            return AMSO(num_swarms=5, swarm_size=10, n_iterations=iterations, verbose=verbose)
        case "tlpso":
            return TLPSO(
                global_swarm_size=5,
                local_swarm_size=50,
                max_iterations=iterations,
                verbose=verbose,
            )
        case "sfoa":
            return SFOA(n_population=50, n_iterations=iterations, verbose=verbose)
        case "pade_pet":
            return PaDE_PET(n_population=50, n_iterations=iterations, verbose=verbose)
        case _:
            raise ValueError(f"Invalid algorithm key: {key}")


def run_algorithm_job(
    scenario: int,
    algo_key: str,
    iterations: int,
    dimension: int,
    experiments: int,
    verbose: bool,
    seed: int,
) -> Tuple[str, List[BenchmarkResult]]:
    import os
    print(f"[PID {os.getpid()}] Starting {algo_key} with seed {seed}")
    set_thread_env()
    try:
        random.seed(seed)
        np.random.seed(seed)
    except Exception:
        pass
    solver = Solver(num_experiments=experiments, functions=functions_due_to_scenario(scenario))
    alg = create_algorithm(algo_key, iterations, verbose)
    name = alg.__class__.__name__
    print(f"[PID {os.getpid()}] Running {name} benchmark...")
    results = solver.benchmark(dimension=dimension, model=alg.optimize, trajectory=alg.trajectory)
    print(f"[PID {os.getpid()}] Completed {name}")
    return name, results


