"""
Script to optimize SDAO parameters using Optuna.
"""

from functools import partial
from itertools import repeat
import optuna
import numpy as np
from model.sdao import SDAO, SDAOParams
from model.solver import ExperimentFunction
from model.functions.bench_funcs import bench_funcs
from model.functions.stoch_funcs import stoch_funcs
from model.functions.cec_funcs import cec_funcs
from model.functions.real_life_funcs import real_life_funcs


# 1. Define the objective function
def objective(
    trial: optuna.Trial,
    dimension: int,
    num_experiments: int,
    objective_functions: list[list[ExperimentFunction]],
) -> float:
    """
    Objective function to optimize SDAO parameters.

    Args:
        trial: Optuna Trial object for suggesting parameter values.
        dimension: Dimension of the optimization problem.
        num_experiments: Number of experiments to average the results.
        objective_fn: Objective function to optimize (e.g., sphere_function).
        bounds: Search space boundaries.

    Returns:
        The average objective value (best value found by SDAO).
    """

    # 2. Suggest parameter values using Trial
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1)
    memory_coeff = trial.suggest_float("memory_coeff", 0.1, 0.9)
    diffusion_coeff = trial.suggest_float("diffusion_coeff", 0.1, 5.0)
    density_radius = trial.suggest_float("density_radius", 0.1, 5.0)
    decay_rate = trial.suggest_float("decay_rate", 1e-4, 1e-1)
    contract_every = trial.suggest_int("contract_every", 1, 50)

    # 3. Configure SDAO parameters
    sdao_params: SDAOParams = {
        "learning_rate": learning_rate,
        "memory_coeff": memory_coeff,
        "diffusion_coeff": diffusion_coeff,
        "density_radius": density_radius,
        "decay_rate": decay_rate,
        "contract_every": contract_every,
    }

    # 4. Initialize SDAO with the suggested parameters
    sdao = SDAO(
        num_particles=30,
        num_iterations=100,
        params=sdao_params,
        verbose=False,
    )

    # 5. Run SDAO multiple times and average the results
    results = []
    for _ in repeat(None, num_experiments):
        func_results = []
        for functions in objective_functions:
            exp_results = []
            for experiment in functions:
                # Get the objective function and bounds from here
                objective_fn = experiment["call"]
                bounds = experiment["domain"]
                dimension = experiment.get("dimension", dimension)
                # Get the optimal value
                optimal_value = experiment.get("optimal_value", 0)
                # Then, run the optimization
                best_value, _ = sdao.optimize(objective_fn, bounds, dimension)  # type: ignore
                # Append the absolute error obtained to the results
                exp_results.append(abs(best_value - optimal_value))
            # Append the average result from the experiment
            func_results.append(np.mean(exp_results))
        # Append the average result from the experimental obtained
        results.append(np.mean(exp_results))

    # 6. Return the average objective value
    return np.mean(results)  # type: ignore


def optimize_parameters():
    """Main function to execute the SDAO parameter optimization."""
    # Define problem parameters
    dimension = 50  # Dimension for the optimization problems
    num_experiments = 15  # Number of experiments to average
    obj_fncs = [
        bench_funcs,
        stoch_funcs,
        real_life_funcs,
        cec_funcs,
    ]  # Objective functions to optimize
    # Create the Optuna study
    study = optuna.create_study(direction="minimize")
    # * Wrap the objective function with the necessary arguments
    sdao_obj = partial(
        objective,
        dimension=dimension,
        num_experiments=num_experiments,
        objective_functions=obj_fncs,  # type: ignore
    )
    # Execute the optimization
    study.optimize(sdao_obj, n_trials=25, show_progress_bar=True)

    # Display the results
    print(32 * "=")
    print("\033[93mBest parameters found:\033[0m")
    print(study.best_params)
