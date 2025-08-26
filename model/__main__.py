"""
Run the algorithm...
"""

from argparse import ArgumentParser
import warnings
import pickle
from functools import partial
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import matplotlib.pyplot as plt  # pylint: disable=E0401
import pydash as _py
import pandas as pd
import numpy as np

# Local imports
from model.functions.bench_funcs import bench_funcs
from model.functions.stoch_funcs import stoch_funcs
from model.functions.real_life_funcs import real_life_funcs
from model.functions.cec_funcs import cec_funcs
from model.functions.mpb_funcs import mpb_benchmarks
from model.solver import Solver, ExperimentFunction, BenchmarkResult
from model.utils import statistical_tests

# Model imports
from model.sdao import SDAO
from model.soa.fractal import StochasticFractalSearch
from model.soa.algebraic_sgd import AlgebraicSGD
from model.soa.shade import SHADEwithILS
from model.soa.path_relinking import PathRelinking
from model.soa.amso import AMSO
from model.soa.tlpso import TLPSO
from model.soa.sfoa import SFOA
from model.soa.pade_pet import PaDE_PET
from model.parallel_runner import (
    set_thread_env as pr_set_thread_env,
    create_algorithm as pr_create_algorithm,
    run_algorithm_job as pr_run_algorithm_job,
)

# Define the Cities for the VRP problem
# 0 => New York, 1 => LA, 2 => Chicago, 3 => Houston, 4 => Phoenix
# 5 => Philadelphia, 6 => San Antonio, 7 => San Diego, 8 => Dallas, 9 => San Jose
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


# Some changes for the Matplotlib import
plt.rcParams["svg.fonttype"] = "none"
# set_matplotlib_formats('svg')


def plot_bar_results(benchmarks: dict[str, list[BenchmarkResult]]) -> None:
    """Plot the results in a horizontal bar chart"""
    _, ax = plt.subplots()
    # Define the y values (functions)
    y_values = {res["function"] for res in benchmarks["SDAO"]}
    # Define the height of each bar
    bar_height = 0.35
    # Define the positions of the bars
    r1 = np.arange(len(y_values))

    for alg_name, bench in benchmarks.items():
        # Extract the mean best values for each algorithm
        values = [
            # * Note: The log is to convert the values to a better scale
            # * for visualization purposes.
            # * This is going to standardize the values to a log scale.
            np.log10(1 + np.mean([res["best_value"] for res in results]))
            for results in _py.group_by(bench, lambda x: x["function"]).values()
        ]
        # Add the bar plot
        ax.barh(r1, values, height=bar_height, edgecolor="grey", label=alg_name)
        # Update the r1 values
        r1 = [x + bar_height for x in r1]
    # Add labels
    ax.set_ylabel("Functions", fontweight="bold")
    ax.set_xlabel("Best Value", fontweight="bold")
    ax.set_title("Best Value for each Function and Algorithm")
    ax.set_yticks([r + bar_height / 2 for r in range(len(y_values))])
    ax.set_yticklabels(y_values)

    # Add legend
    ax.legend()

    # Show the plot
    plt.show()


def show_results(benchmarks: dict[str, list[BenchmarkResult]]) -> None:
    """Show results as a table, as a csv"""
    # Print the results for the best value.
    # The columns should be the test functions, and the rows should be the algorithms.
    # The values should be the mean best value for each function.
    bench_results = {}
    for alg_name, bench in benchmarks.items():
        # Extract the mean best values for each algorithm
        values = {
            func: {
                # The percentual error
                "e": round(
                    np.mean(
                        [
                            res["error"] if res["error"] else res["best_value"]
                            for res in results
                        ]
                    ),
                    2,
                ),
                # The time taken
                "t": round(np.mean([res["time"] for res in results]), 4),
                # The memory usage
                "m": round(np.mean([res["memory"] for res in results]), 4),
            }
            # * Note: The log is to convert the values to a better scale
            # * for visualization purposes.
            # * This is going to standardize the values to a log scale.
            # func: np.log10(np.mean([res["best_value"] for res in results]))
            for func, results in _py.group_by(bench, lambda x: x["function"]).items()
        }
        bench_results[alg_name] = values
    # From this, plot it as a table
    dataframe = pd.DataFrame(bench_results)
    print(dataframe)


def functions_due_to_scenario(scenario: int) -> list[ExperimentFunction]:
    """Due to the scenario, return the experiment functions to use!"""
    match scenario:
        case 0:
            print("Using Scenario 0: Normal benchmark functions.")
            return bench_funcs
        case 1:
            print("Using Scenario 1: Stochastic benchmark functions.")
            return stoch_funcs
        case 2:
            print("Using Scenario 2: Real life functions.")
            # * Here, using the dimension, we're going to generate the variables for
            # * each function....
            warnings.warn(
                "\033[93mThe dimension is going to be ignored for this scenario. "
                + "Each example has his own dimension set for the demo.\033[0m"
            )
            for func in real_life_funcs:
                match func["name"]:
                    case "Predictive Maintenance":
                        # In this scenario, generate the
                        # failure probabilities to use in the function.
                        # Also, generate the repair costs!
                        failure_probs = np.random.uniform(0, 1, size=30)
                        repair_costs = np.random.uniform(0, 100, size=30)
                        # And include the maintenance costs
                        func["call"] = partial(
                            func["call"],
                            failure_probs=failure_probs,  # type: ignore
                            repair_costs=repair_costs,  # type: ignore
                            downtime_costs=10,  # type: ignore
                        )
                    case "VRP":
                        # And include the travel time matrix
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
            # Run the CEC benchmark functions
            print("Using Scenario 3: CEC benchmark functions.")
            return cec_funcs
        case 4:
            # Include the MPB benchmark functions
            print("Using Scenario 4: Moving Peaks Benchmark functions.")
            warnings.warn(
                "\033[93mThe dimension is going to be ignored for this scenario. "
                + "Each example has his own dimension set for the demo.\033[0m"
            )
            return mpb_benchmarks
        case _:
            raise NotImplementedError(f"Invalid scenario. Scenario: {scenario}")

    # helpers moved to model.parallel_runner


if __name__ == "__main__":
    # Parse the arguments
    parser = ArgumentParser(description="Run the SDAO algorithm.")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Increase output verbosity."
    )
    parser.add_argument(
        "-e",
        "--experiments",
        type=int,
        default=100,
        help="Number of experiments (for statistical analysis).",
    )
    parser.add_argument(
        "-i", "--iterations", type=int, default=100, help="Number of iterations."
    )
    parser.add_argument(
        "-d", "--dimension", type=int, default=2, help="Number of iterations."
    )
    parser.add_argument(
        "-s",
        "--scenario",
        type=str,
        default="0",
        help="Scenario to run. Use 'all' to run them all at once",
    )
    parser.add_argument(
        "-a", "--algorithm", type=str, default="all", help="Algorithm to run."
    )
    parser.add_argument(
        "-ltx",
        "--latex",
        type=str,
        default=True,
        help="Store the results in a LaTex friendly way.",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run algorithms in parallel using separate processes.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=os.cpu_count() or 1,
        help="Maximum number of parallel worker processes.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Base seed for reproducible parallel runs.",
    )

    args = parser.parse_args()

    # Check the scenarios that we have available... if we have  "all" as scenario
    # then we should
    scenarios = [0, 1, 2, 3] if args.scenario == "all" else [int(args.scenario)]
    # ====================================== #
    #                Models                  #
    # ====================================== #
    sdao = SDAO(
        num_iterations=args.iterations,
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
        verbose=args.verbose,
    )

    sfs = StochasticFractalSearch(
        n_population=50,
        n_iterations=args.iterations,
        fractal_factor=0.9,
        verbose=args.verbose,
    )

    sgd = AlgebraicSGD(n_iterations=args.iterations, verbose=args.verbose)

    shade = SHADEwithILS(
        n_population=50,
        n_iterations=args.iterations,
        memory_size=10,
        verbose=args.verbose,
    )

    path_relinking = PathRelinking(
        n_population=50,
        n_iterations=args.iterations,
        elite_ratio=0.2,
        verbose=args.verbose,
    )

    amso = AMSO(
        num_swarms=5, swarm_size=10, n_iterations=args.iterations, verbose=args.verbose
    )

    tlpso = TLPSO(
        global_swarm_size=5,
        local_swarm_size=50,
        max_iterations=args.iterations,
        verbose=args.verbose,
    )

    sfoa = SFOA(
        n_population=50,
        n_iterations=args.iterations,
        verbose=args.verbose,
    )

    pade_pet = PaDE_PET(
        n_population=50,
        n_iterations=args.iterations,
        verbose=args.verbose,
    )

    # Define which algorithms you'll run (as keys)
    algorithms_keys: list[str] = []
    match args.algorithm:
        case "all":
            algorithms_keys = [
                "sdao",
                "sfs",
                "sgd",
                "shade",
                "path_relinking",
                "amso",
                "tlpso",
                "sfoa",
                "pade_pet",
            ]
        case "sdao":
            algorithms_keys = ["sdao"]
        case "sfs":
            algorithms_keys = ["sfs"]
        case "sgd":
            algorithms_keys = ["sgd"]
        case "shade":
            algorithms_keys = ["shade"]
        case "path_relinking":
            algorithms_keys = ["path_relinking"]
        case "amso":
            algorithms_keys = ["amso"]
        case "tlpso":
            algorithms_keys = ["tlpso"]
        case "sfoa":
            algorithms_keys = ["sdao", "sfoa"]
        case "pade_pet":
            algorithms_keys = ["sdao", "pade_pet"]
        case _:
            raise ValueError(f"Invalid algorithm: {args.algorithm}")

    # Instance the algorithm(s) and run them for each scenario
    for scenario in scenarios:
        benchmarks_results: dict[str, list[BenchmarkResult]] = {}
        if args.parallel:
            pr_set_thread_env()
            try:
                mp.set_start_method("spawn", force=True)
            except RuntimeError:
                pass
            base_seed = args.seed
            seed_seq = np.random.SeedSequence(base_seed)
            child_seeds = seed_seq.spawn(len(algorithms_keys))
            max_workers = max(1, int(args.max_workers))
            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                print(
                    f"Running {len(algorithms_keys)} algorithms in parallel with {max_workers} workers..."
                )
                print(f"Algorithms: {algorithms_keys}")
                futures = []
                for algo_key, ss in zip(algorithms_keys, child_seeds):
                    seed = int(ss.generate_state(1)[0])
                    print(f"Submitting {algo_key} with seed {seed}")
                    futures.append(
                        ex.submit(
                            pr_run_algorithm_job,
                            scenario,
                            algo_key,
                            args.iterations,
                            args.dimension,
                            args.experiments,
                            args.verbose,
                            seed,
                        )
                    )
                print(f"Submitted {len(futures)} jobs, waiting for completion...")
                for i, fut in enumerate(as_completed(futures), 1):
                    name, results = fut.result()
                    benchmarks_results[name] = results
                    print(f"✓ Completed {name} ({i}/{len(futures)})")
        else:
            solver = Solver(
                num_experiments=args.experiments,
                functions=functions_due_to_scenario(scenario),
            )
            for i, algo_key in enumerate(algorithms_keys, start=1):
                alg = pr_create_algorithm(algo_key, args.iterations, args.verbose)
                NAME = alg.__class__.__name__
                print(
                    f"Running the {NAME} algorithm... \033[50mRunning {i}/{len(algorithms_keys)}\033[0m"
                )
                print(32 * "=")
                results = solver.benchmark(
                    dimension=args.dimension,
                    model=alg.optimize,
                    trajectory=alg.trajectory,
                )
                benchmarks_results[NAME] = results
                print("\n")

        # Store the results for this scenario
        with open(f"scenario_{scenario}_d_{args.dimension}.pkl", "wb") as f:
            pickle.dump(benchmarks_results, f)
        # Plot the results as a bar chart
        print("Plotting the results...")
        # ====================================== #
        #                Plotting                #
        # ====================================== #
        # plot_bar_results(benchmarks_results)
        show_results(benchmarks_results)
        # ====================================== #
        #                Analysis                #
        # ====================================== #
        # Perform an statistical test on the results
        if len(benchmarks_results) > 1:
            # ! ONLY IF WE HAVE MORE THAN 1 ALGORITHM RUNNING, we can
            # ! perform this statistical test.
            if args.experiments <= 3:
                warnings.warn(
                    "\033[93mNot enough experiments to perform a statistical test. "
                    + "Include more experiments using the `-e` flag.\033[0m"
                )
            else:
                print("Performing statistical test on the results...")
                try:
                    statistical_tests(benchmarks_results, args.latex)
                    # In this case, if len(scenarios) > 1, we should rename the files to
                    # include the scenario number.
                    if args.latex is True and len(scenarios) > 1:
                        for test in ("anova_results", "tukey_results"):
                            with open(f"{test}.tex", "r") as f:
                                content = f.read()
                                # Write the content to a new file
                                with open(f"{test}_scenario_{scenario}.tex", "w") as f2:
                                    f2.write(content)

                except Exception:  # pylint: disable=W0718
                    warnings.warn(
                        "\033[93mAn error occurred while performing the statistical test.\033[0m"
                    )
        else:
            # In this case, we don't have enough algorithms to perform a statistical test.
            warnings.warn(
                "\033[93mNot enough algorithms to perform a statistical test.\033[0m"
            )
