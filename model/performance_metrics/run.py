"""
Run and get the performance metrics for a given scenario.

Args:
    scenario (str): The scenario to run.
    args (argparse.Namespace): The command line arguments.

Returns:
    dict: The performance metrics for the scenario.
"""

import typing as tp
import pickle
import argparse

# Local imports
from model.performance_metrics.metrics import METRICS
from model.performance_metrics.plotting import (
    plot_absolute_error,
    plot_general_absolute_error,
    convergence_general_plot,
)
from model.performance_metrics.tables import generate_mean_table


def __open_scenario_data(scenario: str, dimension: str) -> dict:
    """Open the scenario pkl file if it exists"""
    try:
        # Search for a dimension-specific file
        with open(f"scenario_{scenario}_d_{dimension}.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return {}


def __run_metrics(scenario: str, dimension: str) -> tuple[dict[str, tp.Any], dict]:
    """Run and get the performance metrics for a given scenario.

    Args:
        scenario (str): The scenario to use.
        dimension (str): The dimension to use.

    Returns:
        tuple[dict[str, tp.Any], dict]: The performance metrics for the scenario and the data.
    """
    # Open the scenario pkl file if it exists
    data = __open_scenario_data(scenario, dimension)
    metrics = {}
    for metric in METRICS:
        metrics[metric.__name__] = metric(data, int(scenario))
    return metrics, data


def main():
    """Main function to run the performance metrics."""
    parser = argparse.ArgumentParser(description="Run performance metrics.")
    parser.add_argument(
        "-s", "--scenario", type=str, help="The scenario to use.", default="0"
    )
    parser.add_argument(
        "-d", "--dimension", type=str, help="The dimension to use.", default="10"
    )
    parser.add_argument(
        "-pdf",
        "--pdf",
        action="store_true",
        help="Save the plot as a PDF.",
        default=False,
    )
    args = parser.parse_args()
    print(
        f"Running performance metrics for scenario {args.scenario} and dimension {args.dimension}"
    )
    if args.scenario == "all":
        all_metrics = {}
        data = {}
        for scenario in range(0, 4):
            all_metrics[scenario], data[scenario] = __run_metrics(
                str(scenario), args.dimension
            )
        # Generate some things as the general table
        generate_mean_table(
            {s: d["best_solution"] for s, d in all_metrics.items()},
            {s: d["stability"] for s, d in all_metrics.items()},
        )
        # Generate the convergence plot
        convergence_general_plot(data, store_as_pdf=args.pdf)
    else:
        metrics, _ = __run_metrics(args.scenario, args.dimension)
        # Then, using the metrics, run some post process
        print("Running post-processing for data...")
        print("Plots")
        plot_absolute_error(
            metrics["best_solution"], metrics["stability"], store_as_pdf=args.pdf
        )
        plot_general_absolute_error(
            metrics["best_solution"], metrics["stability"], store_as_pdf=args.pdf
        )


if __name__ == "__main__":
    main()
