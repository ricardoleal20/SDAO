"""
Define different metric plots, such as the
"""

from typing import Sequence
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Local imports
from model.__main__ import functions_due_to_scenario


_INPUT_DATA = dict[str, dict[str, float]]

# Configure the PLT style
plt.rcParams["font.family"] = "Times New Roman"  # Use Times New Roman
plt.rcParams["font.size"] = 36  # This is an appropriate size for Springer
plt.rcParams["axes.labelsize"] = 24  # Size of labels for axes
plt.rcParams["xtick.labelsize"] = 36  # Size of labels on x-axis
plt.rcParams["ytick.labelsize"] = 36  # Size of labels on y-axis
plt.rcParams["legend.fontsize"] = 16  # Size of legend


def __smooth_curve(curve: Sequence[float], window=5) -> np.ndarray:
    """Smooth the curve using a moving average"""
    return np.convolve(curve, np.ones(window) / window, mode="same")


def plot_absolute_error(
    average_error: _INPUT_DATA,
    standard_deviation: _INPUT_DATA,
    store_as_pdf: bool = False,
) -> None:
    """Based on the data provided, create a plot showing
    the average error for each algorith and each function,
    along with their standard deviation as a parameter (to simulate a Box Plot)
    """
    # First of all, get all the algorithms
    algorithms = list(average_error.keys())
    # Get also all the functions. For that, only get the keys for SDAO.
    # Since the functions should be all the same
    functions = list(average_error["SDAO"].keys())
    # Create the Plot and defien the colors using sns
    fig, ax = plt.subplots(figsize=(30, 12))
    # Define color palette (ColorBlind Friendly)
    colors = sns.color_palette("Greys", len(algorithms))
    x = np.arange(len(functions))  # Position of each group
    width = 0.50  # Width of each bar
    # Iterate over the algorithms to create a bar plot for each function
    for i in range(len(algorithms)):
        ax.bar(
            # Get the position of the bar
            x + i * width,
            # Get the average error for each function
            [average_error[algorithms[i]][func] for func in functions],
            width,
            label=algorithms[i],
            yerr=[standard_deviation[algorithms[i]][func] for func in functions],
            capsize=3,
            alpha=0.9,
            color=colors[i],  # Custom color for each algorithm
            edgecolor="black",  # Edge color to improve contrast
        )
    # Add some style to the plot
    # ax.set_xlabel("Benchmark Functions")
    ax.set_ylabel("Mean Absolute Error with Standard Deviation")
    ax.set_title(
        "Absolute Error Comparison of Optimization Algorithms Over Benchmark Functions"
    )
    ax.set_xticks(x + width * (len(algorithms) / 2))
    ax.set_xticklabels(functions, rotation=45, ha="right")
    ax.legend(
        title="Optimization Algorithms",
        frameon=True,
        bbox_to_anchor=(0.65, 0.95),
        loc="upper left",
        fontsize=16,
    )
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    # Add a limit to the y-axis
    ax.set_ylim(bottom=0, top=1000)
    plt.tight_layout()
    # And show it if store_as_pdf is False
    if not store_as_pdf:
        plt.show()
    else:
        # Otherwise...
        plt.savefig("comparison_plot.pdf", format="pdf", dpi=300)


def plot_general_absolute_error(
    average_error: _INPUT_DATA,
    standard_deviation: _INPUT_DATA,
    store_as_pdf: bool = False,
) -> None:
    """Plot the general absolute error with standard deviation.
    The general absolute error is calculated as the mean of the absolute errors.
    along with the mean of the standard deviations.

    Args:
        average_error: The average error for each algorithm.
        standard_deviation: The standard deviation for each algorithm.
        store_as_pdf: Whether to store the plot as a PDF file.
    """
    # First of all, get all the algorithms
    algorithms = []
    for algorithm in average_error.keys():
        if algorithm != "StochasticFractalSearch":
            algorithms.append(algorithm)
        else:
            algorithms.append("SFS")
    # Get the mean absolute error per algorithm
    absolute_error_per_algorithm = [
        np.mean(list(errors.values())) for errors in average_error.values()
    ]
    # Get the mean standard deviation per algorithm
    standard_deviation_per_algorithm = [
        np.mean(list(deviations.values())) for deviations in standard_deviation.values()
    ]
    # Plot the mean absolute error along with the mean standard deviation
    fig, ax = plt.subplots(figsize=(30, 12))
    ax.bar(
        algorithms,
        absolute_error_per_algorithm,
        yerr=standard_deviation_per_algorithm,
        capsize=2,
        color="black",
        alpha=0.7,
    )
    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Mean Absolute Error")
    ax.set_title("Mean Absolute Error with Standard Deviation")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    # Add a limit to the y-axis
    ax.set_ylim(bottom=0, top=5000)
    plt.tight_layout()
    # And show it if store_as_pdf is False
    if not store_as_pdf:
        plt.show()
    else:
        # Otherwise...
        plt.savefig("general_comparison_plot.pdf", format="pdf", dpi=300)


def convergence_general_plot(
    data: dict[int, dict[str, list[dict]]], store_as_pdf: bool = False
) -> None:
    """Create and save a PDF with convergence plots for each scenario coming from functions_due_to_scenario.

    Args:
        data: Dictionary with the data of the experiments, where each key is a scenario (0,1,2,3)
              and its value is a dictionary that maps the name of the algorithm to a list of results.
              Each result is a dictionary that must contain at least:
                  - "function": name of the function evaluated
                  - "trajectory": list of tuples (iteration, value)
                  - (Optional) "optimal_value": optimal value (although it is recommended to obtain it from functions_due_to_scenario)
    """
    # Define the colors and styles for the plots (you can modify or expand according to the number of algorithms)
    colors = (
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
    )
    linestyles = ("-", "--", "-.", ":", (0, (5, 2)), (0, (3, 1, 1, 1)), "-")

    # Iterate over each scenario present in the data
    for scenario in sorted(data.keys()):
        # Get the functions associated with the scenario
        functions_scenario = functions_due_to_scenario(scenario)
        num_functions = len(functions_scenario)
        if num_functions == 0:
            continue  # If there are no functions for this scenario, skip it

        # Configure the grid of subplots (for example, 3 columns)
        num_cols = 3
        num_rows = (
            num_functions + num_cols - 1
        ) // num_cols  # Integer division rounded up
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 6 * num_rows))

        # Ensure we have an array of axes (flatten for easy iteration)
        if isinstance(axes, np.ndarray):
            axes = axes.flatten()
        else:
            axes = [axes]

        # Iterate over each function defined for the scenario
        for idx, bench_func in enumerate(functions_scenario):
            function_name = bench_func["name"]
            optimal_value = bench_func.get("optimal_value", 0)
            ax = axes[idx]
            ax.set_title(function_name)
            ax.set_xlabel("Iterations")
            ax.set_ylabel("Absolute Error (log scale)")
            ax.set_yscale("log")
            ax.grid(True, linestyle="--", alpha=0.5)

            # Extract the data corresponding to this scenario
            data_scenario = data[scenario]

            # Iterate over each algorithm present in the data
            for i, (algorithm, data_alg) in enumerate(data_scenario.items()):
                # Filter the executions corresponding to the current function
                data_func = [d for d in data_alg if d.get("function") == function_name]
                if not data_func:
                    continue

                # Collect the absolute errors for each iteration
                errors_per_iter = {}
                for d in data_func:
                    # If the optimal value comes in the data, you could use it; in this example we use the one obtained from bench_func
                    trajectory = d["trajectory"]
                    for iteration, value in trajectory:
                        error = abs(value - optimal_value)
                        errors_per_iter.setdefault(iteration, []).append(error)

                if not errors_per_iter:
                    continue

                # Sort the iterations and calculate the mean and standard deviation
                iterations = sorted(errors_per_iter.keys())
                mean_curve = [np.mean(errors_per_iter[it]) for it in iterations]
                std_curve = [np.std(errors_per_iter[it]) for it in iterations]

                # Optional: smooth the curves using the __smooth_curve function
                # mean_curve = __smooth_curve(mean_curve)
                # std_curve = __smooth_curve(std_curve)

                # Plot the curve and the uncertainty area
                ax.plot(
                    iterations,
                    mean_curve,
                    label=algorithm,
                    color=colors[i % len(colors)],
                    linestyle=linestyles[i % len(linestyles)],
                    linewidth=2,
                )
                ax.fill_between(
                    iterations,
                    np.array(mean_curve) - np.array(std_curve),
                    np.array(mean_curve) + np.array(std_curve),
                    color=colors[i % len(colors)],
                    alpha=0.15,
                )

            # Determine the y-axis limits for the zoomed-in view
            all_errors = [
                error for errors in errors_per_iter.values() for error in errors
            ]
            if all_errors:
                min_error = max(min(all_errors), 1e-10)  # Avoid log(0) issues
                max_error = min(
                    max(all_errors), 1e3
                )  # Limit to a reasonable upper bound
                ax.set_ylim(bottom=min_error, top=max_error)

            ax.legend(loc="upper right", fontsize=9)

        # Delete empty subplots
        for j in range(idx + 1, len(axes)):
            fig.delaxes(axes[j])

        # plt.suptitle(
        #     f"Convergence in Functions of Benchmark for Scenario {scenario}",
        #     fontsize=16,
        # )
        plt.tight_layout()

        # Store the PDF file for the current scenario
        pdf_filename = f"convergence_plot_scenario_{scenario}.pdf"
        plt.savefig(pdf_filename, format="pdf", dpi=300)
        plt.close(fig)
