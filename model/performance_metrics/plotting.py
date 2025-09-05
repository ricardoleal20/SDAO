"""
Define different metric plots, such as the
"""

from typing import Sequence
import numpy as np
import matplotlib.pyplot as plt

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


# Unified color palette and line styles across all plots
COLORS = (
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # olive
    "#17becf",  # cyan
    "#aec7e8",  # light blue
)
LINESTYLES = (
    "-",
    "--",
    "-.",
    ":",
    (0, (5, 2)),
    (0, (3, 1, 1, 1)),
    "-",
)


def _order_algorithms(algorithms: Sequence[str]) -> list[str]:
    """Return algorithms with 'SDAO' first, followed by others alphabetically."""
    others = sorted([a for a in algorithms if a != "SDAO"])
    return (["SDAO"] if "SDAO" in algorithms else []) + others


def _display_name(algorithm: str) -> str:
    """Map internal algorithm names to display labels (e.g., abbreviations)."""
    return "SFS" if algorithm == "StochasticFractalSearch" else algorithm


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
    # First of all, get all the algorithms, SDAO first
    algorithms = _order_algorithms(list(average_error.keys()))
    # Get also all the functions. For that, only get the keys for SDAO.
    # Since the functions should be all the same
    functions = list(average_error["SDAO"].keys())
    # Create the Plot and defien the colors using unified palette
    fig, ax = plt.subplots(figsize=(30, 12))
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
            label=_display_name(algorithms[i]),
            yerr=[standard_deviation[algorithms[i]][func] for func in functions],
            capsize=3,
            alpha=0.9,
            color=COLORS[i % len(COLORS)],
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
    # Order algorithms with SDAO first and map display names
    algorithms = _order_algorithms(list(average_error.keys()))
    display_labels = [_display_name(a) for a in algorithms]

    # Compute metrics in that order
    absolute_error_per_algorithm = [
        np.mean(list(average_error[alg].values())) for alg in algorithms
    ]
    standard_deviation_per_algorithm = [
        np.mean(list(standard_deviation[alg].values())) for alg in algorithms
    ]

    # Plot the mean absolute error along with the mean standard deviation
    fig, ax = plt.subplots(figsize=(30, 12))
    ax.bar(
        display_labels,
        absolute_error_per_algorithm,
        yerr=standard_deviation_per_algorithm,
        capsize=2,
        color=[COLORS[i % len(COLORS)] for i in range(len(algorithms))],
        alpha=0.8,
        edgecolor="black",
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
    data: dict[int, dict[str, list[dict]]],
    store_as_pdf: bool = False,
    dimension: int = 10,
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
    # Use unified COLORS and LINESTYLES

    legend_position_per_scenario = {
        0: (0.85, 0.2),
        1: (0.5, 0.03),
        2: (0.5, 0.53),
        3: (0.85, 0.2),
    }

    # Iterate over each scenario present in the data
    for scenario in sorted(data.keys()):
        # Get the functions associated with the scenario
        functions_scenario = functions_due_to_scenario(scenario)
        num_functions = len(functions_scenario)
        if num_functions == 0:
            continue  # If there are no functions for this scenario, skip it

        # For scenario 3 (CEC), split across multiple pages to avoid overly tight subplots
        if scenario == 3:
            page_size = 12
            num_cols = 3
            for page_start in range(0, num_functions, page_size):
                sub_funcs = functions_scenario[page_start : page_start + page_size]
                num_rows = (len(sub_funcs) + num_cols - 1) // num_cols
                fig, axes = plt.subplots(
                    num_rows, num_cols, figsize=(17, 4.8 * num_rows)
                )

                if isinstance(axes, np.ndarray):
                    axes = axes.flatten()
                else:
                    axes = [axes]

                for idx, bench_func in enumerate(sub_funcs):
                    function_name = bench_func["name"]
                    optimal_value = bench_func.get("optimal_value", 0)
                    ax = axes[idx]
                    ax.set_title(
                        function_name
                        if "CEC" not in function_name
                        else function_name.replace("CEC", ""),
                        fontsize=20,
                    )
                    ax.set_xlabel("Iterations")
                    ax.set_ylabel("Absolute Error")
                    ax.set_yscale("log")
                    ax.grid(True, linestyle="--", alpha=0.5)

                    data_scenario = data[scenario]
                    ordered_algs = _order_algorithms(list(data_scenario.keys()))
                    for i, algorithm in enumerate(ordered_algs):
                        data_alg = data_scenario[algorithm]
                        data_func = [
                            d for d in data_alg if d.get("function") == function_name
                        ]
                        if not data_func:
                            continue

                        errors_per_iter = {}
                        for d in data_func:
                            trajectory = d["trajectory"]
                            for iteration, value in trajectory:
                                error = abs(value - optimal_value)
                                errors_per_iter.setdefault(iteration, []).append(error)

                        if not errors_per_iter:
                            continue

                        iterations = sorted(errors_per_iter.keys())
                        mean_curve = [np.mean(errors_per_iter[it]) for it in iterations]
                        std_curve = [np.std(errors_per_iter[it]) for it in iterations]

                        ax.plot(
                            iterations,
                            mean_curve,
                            label=_display_name(algorithm),
                            color=COLORS[i % len(COLORS)],
                            linestyle=LINESTYLES[i % len(LINESTYLES)],
                            linewidth=2,
                        )
                        ax.fill_between(
                            iterations,
                            np.array(mean_curve) - np.array(std_curve),
                            np.array(mean_curve) + np.array(std_curve),
                            color=COLORS[i % len(COLORS)],
                            alpha=0.15,
                        )

                    if dimension > 25 and function_name != "Xin-She Yang 1":
                        ax.set_ylim(bottom=1, top=1e10)
                    ax.set_xlim(left=0, right=300)
                    ax.legend().remove()

                # Add legend: if the page has fewer subplots than columns (e.g., last page with 2),
                # use the empty rightmost axis as a dedicated legend panel. Otherwise, place it below.
                handles, labels = ax.get_legend_handles_labels()
                if len(sub_funcs) < num_cols:
                    legend_ax = axes[-1]
                    legend_ax.axis("off")
                    legend_ax.legend(
                        handles,
                        labels,
                        loc="center",
                        ncol=1,
                        fontsize=14,
                        frameon=True,
                    )
                else:
                    fig.legend(
                        handles,
                        labels,
                        loc="lower center",
                        bbox_to_anchor=(0.5, 0.02),
                        ncol=7,
                        fontsize=16,
                    )

                if len(sub_funcs) >= num_cols:
                    for j in range(idx + 1, len(axes)):
                        fig.delaxes(axes[j])

                plt.tight_layout()
                # Leave extra space at the bottom for the legend to avoid overlapping.
                # If legend goes into a dedicated axis, we can use a smaller bottom margin.
                if len(sub_funcs) < num_cols:
                    plt.subplots_adjust(bottom=0.10)
                else:
                    if num_rows <= 1:
                        plt.subplots_adjust(bottom=0.24)
                    elif num_rows == 2:
                        plt.subplots_adjust(bottom=0.18)
                    else:
                        plt.subplots_adjust(bottom=0.14)
                pdf_filename = f"convergence_plot_scenario_{scenario}_d{dimension}_p{page_start // page_size + 1}.pdf"
                plt.savefig(
                    pdf_filename,
                    format="pdf",
                    dpi=300,
                    bbox_inches="tight",
                    pad_inches=0.1,
                )
                plt.close(fig)
            # Continue to next scenario since pages already saved
            continue

        # Default behavior for other scenarios: single-page grid
        num_cols = 3
        num_rows = (num_functions + num_cols - 1) // num_cols
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(17, 5 * num_rows))

        if isinstance(axes, np.ndarray):
            axes = axes.flatten()
        else:
            axes = [axes]

        # Iterate over each function defined for the scenario
        for idx, bench_func in enumerate(functions_scenario):
            function_name = bench_func["name"]
            optimal_value = bench_func.get("optimal_value", 0)
            ax = axes[idx]
            ax.set_title(
                function_name
                if "CEC" not in function_name
                else function_name.replace("CEC", ""),
                fontsize=20,
            )
            ax.set_xlabel("Iterations")
            ax.set_ylabel("Absolute Error")
            ax.set_yscale("log")
            ax.grid(True, linestyle="--", alpha=0.5)

            # Extract the data corresponding to this scenario
            data_scenario = data[scenario]

            # Iterate over each algorithm present in the data
            ordered_algs = _order_algorithms(list(data_scenario.keys()))
            for i, algorithm in enumerate(ordered_algs):
                data_alg = data_scenario[algorithm]
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
                    label=_display_name(algorithm),
                    color=COLORS[i % len(COLORS)],
                    linestyle=LINESTYLES[i % len(LINESTYLES)],
                    linewidth=2,
                )
                ax.fill_between(
                    iterations,
                    np.array(mean_curve) - np.array(std_curve),
                    np.array(mean_curve) + np.array(std_curve),
                    color=COLORS[i % len(COLORS)],
                    alpha=0.15,
                )

            # Determine the y-axis limits for the zoomed-in view
            if dimension > 25 and function_name != "Xin-She Yang 1":
                ax.set_ylim(bottom=1, top=1e10)
            ax.set_xlim(left=0, right=300)

            # Remove individual legends
            ax.legend().remove()

        # Add a single legend with scenario-specific positioning
        handles, labels = ax.get_legend_handles_labels()
        if scenario == 3:
            # Place legend at the very bottom-right edge
            fig.legend(
                handles,
                labels,
                loc="lower right",
                bbox_to_anchor=(0.98, 0.02),
                ncol=1,
                fontsize=20,
            )
        else:
            # Specific adjustment for SCENARIO 1: anchor the bottom of the legend
            # slightly above the bottom border to avoid cuts.
            legend_loc = (
                "lower center" if scenario == 1 else "upper center"
            )
            # For SCENARIO 2 move the legend outside, in the top,
            # to avoid covering the central function titles. This is manually done by now, probably
            # I can automate it later for all the scenarios...
            legend_anchor = (
                (0.5, 0.06)
                if scenario == 1
                else ((0.5, 1.02) if scenario == 2 else legend_position_per_scenario[scenario])
            )
            fig.legend(
                handles,
                labels,
                loc=legend_loc,
                bbox_to_anchor=legend_anchor,
                ncol=1 if scenario not in [1, 2] else 7,
                fontsize=20 if scenario not in [1, 2] else 16,
            )

        # Delete empty subplots
        for j in range(idx + 1, len(axes)):
            fig.delaxes(axes[j])

        # plt.suptitle(
        #     f"Convergence in Functions of Benchmark for Scenario {scenario}",
        #     fontsize=16,
        # )
        plt.tight_layout()
        # For different scenarios, we need to adjust the margins
        if scenario == 1:
            # Reduce the bottom margin to bring the legend closer to the subplots
            plt.subplots_adjust(bottom=0.15)
        elif scenario == 2:
            # Leave extra space above for the external legend
            plt.subplots_adjust(top=0.88)
        # Add extra vertical spacing between rows for scenario 2
        if scenario == 2:
            plt.subplots_adjust(hspace=0.6)

        # Store the PDF file for the current scenario
        pdf_filename = f"convergence_plot_scenario_{scenario}_d{dimension}.pdf"
        plt.savefig(pdf_filename, format="pdf", dpi=300, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)


def exploration_in_algorithm(data: dict[int, dict[str, list[dict]]]) -> None:
    """Plot the standard deviation per iteration for each function in the data"""
    # Use unified COLORS and LINESTYLES

    legend_position_per_scenario = {
        0: (0.85, 0.2),
        1: (0.5, 0.03),
        2: (0.5, 0.53),
        3: (0.85, 0.2),
    }
    # Get the data ONLY for scenario 1
    scenario = 1
    # Iterate over here to create the figure
    # Get the functions associated with the scenario
    functions_scenario = functions_due_to_scenario(scenario)
    if scenario != 1:
        return
    num_functions = len(functions_scenario)

    # Configure the grid of subplots (for example, 3 columns)
    num_cols = 3
    num_rows = (num_functions + num_cols - 1) // num_cols  # Integer division rounded up
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(17, 5 * num_rows))

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
        ax.set_title(
            function_name
            if "CEC" not in function_name
            else function_name.replace("CEC", ""),
            fontsize=20,
        )
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Absolute Error")
        ax.set_yscale("log")
        ax.grid(True, linestyle="--", alpha=0.5)

        # Extract the data corresponding to this scenario
        data_scenario = data[scenario]

        # Iterate over each algorithm present in the data
        ordered_algs = _order_algorithms(list(data_scenario.keys()))
        for i, algorithm in enumerate(ordered_algs):
            data_alg = data_scenario[algorithm]
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
            std_curve = [np.std(errors_per_iter[it]) for it in iterations]

            # Optional: smooth the curves using the __smooth_curve function
            # mean_curve = __smooth_curve(mean_curve)
            # std_curve = __smooth_curve(std_curve)

            # Plot the curve and the uncertainty area
            ax.plot(
                iterations,
                std_curve,
                label=_display_name(algorithm),
                color=COLORS[i % len(COLORS)],
                linestyle=LINESTYLES[i % len(LINESTYLES)],
                linewidth=2,
            )

        # Determine the y-axis limits for the zoomed-in view
        if function_name == "Rosenbrock":
            ax.set_ylim(bottom=0, top=1e6)
        else:
            ax.set_ylim(bottom=0, top=1e4)
        ax.set_xlim(left=0, right=300)

        # Remove individual legends
        ax.legend().remove()

    # Add a single legend outside the plot
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=legend_position_per_scenario[scenario],
        ncol=1 if scenario not in [1, 2] else 7,
        fontsize=20 if scenario not in [1, 2] else 16,
    )

    # Delete empty subplots
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    # plt.suptitle(
    #     f"Convergence in Functions of Benchmark for Scenario {scenario}",
    #     fontsize=16,
    # )
    plt.tight_layout()
    # Store the figure
    fig.savefig(
        "diversity_per_iteration.pdf", dpi=300, bbox_inches="tight", format="pdf"
    )


def convergence_summary_plot_paper(
    data: dict[int, dict[str, list[dict]]],
    dimension: int = 50,
    store_as_pdf: bool = True,
    output_name: str = "convergence_summary_main.pdf",
):
    """
    Generates a 2x2 summary convergence plot for the main paper.
    Each subplot corresponds to one benchmark category.

    Parameters:
    - data: Dictionary with all algorithm results.
    - dimension: Dimension for which the results were taken.
    - store_as_pdf: Save the figure to disk.
    - output_name: File name for the exported figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    # Settings
    selected = {
        0: "Rastrigin",
        1: "Ackley",
        3: "CEC Shifted Rastrigin",
        2: "Supply Chain Network Design",
    }
    limits_per_func = {
        "Rastrigin": (1e-0, 1.25e3),
        "Ackley": (1e-0, 4e1),
        "CEC Shifted Rastrigin": (1e-0, 1.25e3),
        "Supply Chain Network Design": (1e-0, 1e4),
    }
    # Use unified COLORS and LINESTYLES

    for idx, (scenario, func_name) in enumerate(selected.items()):
        ax = axes[idx]
        data_scenario = data.get(scenario, {})
        if not data_scenario:
            continue

        ax.set_title(func_name.replace("CEC", "").strip(), fontsize=20)
        ax.set_xlabel("Iterations", fontsize=16)
        ax.set_ylabel("Absolute Error", fontsize=16)
        ax.set_yscale("log")
        ax.grid(True, linestyle="--", alpha=0.5)

        ordered_algs = _order_algorithms(list(data_scenario.keys()))
        for i, algorithm in enumerate(ordered_algs):
            data_alg = data_scenario[algorithm]
            # Extract data for the selected function
            runs = [d for d in data_alg if d["function"] == func_name]
            if not runs:
                continue

            errors_per_iter = {}
            optimal_value = runs[0].get("optimal_value", 0)

            for run in runs:
                for it, val in run["trajectory"]:
                    err = abs(val - optimal_value)
                    errors_per_iter.setdefault(it, []).append(err)

            if not errors_per_iter:
                continue

            iterations = sorted(errors_per_iter.keys())
            mean_curve = [np.mean(errors_per_iter[it]) for it in iterations]
            std_curve = [np.std(errors_per_iter[it]) for it in iterations]

            ax.plot(
                iterations,
                mean_curve,
                label=_display_name(algorithm),
                color=COLORS[i % len(COLORS)],
                linestyle=LINESTYLES[i % len(LINESTYLES)],
                linewidth=2,
            )
            ax.fill_between(
                iterations,
                np.array(mean_curve) - np.array(std_curve),
                np.array(mean_curve) + np.array(std_curve),
                color=COLORS[i % len(COLORS)],
                alpha=0.15,
            )

        ax.set_xlim(left=0, right=300)
        if dimension > 25:
            ax.set_ylim(*limits_per_func[func_name])

    # Global legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=7, fontsize=16)
    plt.tight_layout()
    # fig.suptitle(
    #     f"Convergence Curves on Representative Functions (d = {dimension})",
    #     fontsize=16,
    # )
    plt.subplots_adjust(bottom=0.18)

    if store_as_pdf:
        plt.savefig(output_name, dpi=300, bbox_inches="tight", format="pdf")
        print(f"✅ Saved: {output_name}")
    else:
        plt.show()
