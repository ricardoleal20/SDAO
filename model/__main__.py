"""
Run the algorithm...
"""
from typing import Literal
from argparse import ArgumentParser
# Local imports
from model.functions.bench_funcs import stoch_funcs
from model.solver import Solver, ExperimentFunction
# Model imports
from model.soa.template import Algorithm
from model.sdao import SDAO
from model.soa.fractal import StochasticFractalSearch


def functions_due_to_scenario(scenario: Literal[0, 1, 2]) -> list[ExperimentFunction]:
    """Due to the scenario, return the experiment functions to use!"""
    match scenario:
        case 0:
            print("Using Scenario 0: Normal benchmark functions.")
            return stoch_funcs
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
    for alg in algorithms:
        print(f"Running the {alg.__class__.__name__} algorithm...")
        results = solver.benchmark(
            dimension=args.dimension,
            model=alg.optimize
        )
        # Print the results
        print("Results:")
        for res in results:
            print(
                f"[Experiment={res['iteration']}] || Best value: {res['best_value']}")
        # New line
        print("\n")
