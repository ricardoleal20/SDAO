"""
Run the algorithm...
"""
from typing import Literal
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import pydash as _py
import numpy as np
# Local imports
from model.functions.bench_funcs import bench_funcs
from model.solver import Solver, ExperimentFunction, BenchmarkResult
# Model imports
from model.soa.template import Algorithm
from model.sdao import SDAO
from model.soa.fractal import StochasticFractalSearch


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
            np.log10(1+np.mean([res["best_value"] for res in results]))
            for results in _py.group_by(bench, lambda x: x["function"]).values()
        ]
        # Add the bar plot
        ax.barh(r1, values, height=bar_height,
                edgecolor='grey', label=alg_name)
        # Update the r1 values
        r1 = [x + bar_height for x in r1]
    # Add labels
    ax.set_ylabel('Functions', fontweight='bold')
    ax.set_xlabel('Best Value', fontweight='bold')
    ax.set_title('Best Value for each Function and Algorithm')
    ax.set_yticks([r + bar_height / 2 for r in range(len(y_values))])
    ax.set_yticklabels(y_values)

    # Add legend
    ax.legend()

    # Show the plot
    plt.show()

def functions_due_to_scenario(scenario: Literal[0, 1, 2]) -> list[ExperimentFunction]:
    """Due to the scenario, return the experiment functions to use!"""
    match scenario:
        case 0:
            print("Using Scenario 0: Normal benchmark functions.")
            return bench_funcs
        case 1:
            raise NotImplementedError(
                "Scenario 1 not implemented yet... [STOCHASTIC_FUNCS]")
        case 2:
            raise NotImplementedError(
                "Scenario 2 not implemented yet... [REAL_WORLD]")
        case _:
            raise NotImplementedError(
                f"Invalid scenario. Scenario: {scenario}")


if __name__ == "__main__":
    # Parse the arguments
    parser = ArgumentParser(description="Run the SDAO algorithm.")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Increase output verbosity."
    )
    parser.add_argument(
        "-i", "--iterations", type=int, default=100, help="Number of iterations."
    )
    parser.add_argument(
        "-d", "--dimension", type=int, default=2, help="Number of iterations."
    )
    parser.add_argument(
        "-s", "--scenario", type=int, default=0, help="Scenario to run."
    )
    parser.add_argument(
        "-a", "--algorithm", type=str, default="all", help="Algorithm to run."
    )
    args = parser.parse_args()

    # Create the solver
    solver = Solver(
        num_experiments=args.iterations,
        functions=functions_due_to_scenario(args.scenario)
    )
    # ====================================== #
    #                Models                  #
    # ====================================== #
    sdao = SDAO(
        num_iterations=300,
        num_particles=50,
        version=1,
        params={
            "learning_rate": 0.01,
            "memory_coeff": 0.5,
            "decay_rate": 0.01,
            "diffusion_coeff": 1
        },
        verbose=args.verbose
    )

    sfs = StochasticFractalSearch(
        n_population=50,
        n_iterations=300,
        fractal_factor=0.9,
        verbose=args.verbose
    )

    # Define which algorithms you'll run
    algorithms: list[Algorithm] = []
    match args.algorithm:
        case "all":
            algorithms = [sdao, sfs]
        case "sdao":
            algorithms = [sdao]
        case "sfs":
            algorithms = [sfs]
        case _:
            raise ValueError(f"Invalid algorithm: {args.algorithm}")

    # Run the benchmark
    benchmarks_results: dict[str, list[BenchmarkResult]] = {}
    for alg in algorithms:
        NAME = alg.__class__.__name__
        print(f"Running the {NAME} algorithm...")
        print(32*"=")
        results = solver.benchmark(
            dimension=args.dimension,
            model=alg.optimize
        )
        # Print the results
        benchmarks_results[NAME] = results
        # New line
        print("\n")

    # Plot the results as a bar chart
    print("Plotting the results...")
    # ====================================== #
    #                Plotting                #
    # ====================================== #
    plot_bar_results(benchmarks_results)
