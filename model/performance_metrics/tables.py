"""
Generate the tables of different data processes
"""

from typing import Sequence
import numpy as np

_INPUT_DATA = dict[str, dict[str, float]]

SCENARIO_NAMES = {
    0: "Standard Benchmarks",
    1: "Stochastic Benchmarks",
    2: "Real-world Benchmarks",
    3: "CEC Benchmarks",
}


def __get_valid_data(data: Sequence[float]) -> list[float]:
    """Get the valid data from the array of dict.values()"""
    return [v for v in data if not np.isnan(v) or np.isinf(v) or np.isclose(v, 9e99)]


def __format_number(number: float | np.floating) -> str:
    """Format the number to a maximum of four characters"""
    if number < 1e3:
        return (
            f"{number:.0f}"
            if number >= 100
            else f"{number:.1f}"
            if number >= 10
            else f"{number:.2f}"
        )
    elif number < 1e6:
        number /= 1e3
        return (
            f"{number:.0f}k"
            if number >= 100
            else f"{number:.1f}k"
            if number >= 10
            else f"{number:.2f}k"
        )
    elif number < 1e9:
        number /= 1e6
        return (
            f"{number:.0f}M"
            if number >= 100
            else f"{number:.1f}M"
            if number >= 10
            else f"{number:.2f}M"
        )
    elif number < 1e12:
        number /= 1e9
        return (
            f"{number:.0f}G"
            if number >= 100
            else f"{number:.1f}G"
            if number >= 10
            else f"{number:.2f}G"
        )
    else:
        return f"{number:.2e}"


def generate_mean_table(
    average_error: dict[int, _INPUT_DATA],
    standard_deviation: dict[int, _INPUT_DATA],
) -> None:
    """Generate the mean table of different algorithms"""
    # First of all, get all the algorithms
    algorithms = list(average_error[0].keys())
    # Get the scenarios based on the number of keys
    scenarios = average_error.keys()

    # Create LaTeX table
    table = (
        "\\begin{table*}[h!]\n"
        "\\centering\n"
        "\\caption{Comparison of Absolute Error Across Benchmark Sets}\n"
        "\\label{tab:absolute_error}\n"
        "\\resizebox{\\textwidth}{!}{"
        "\\begin{tabular}{|l|" + "c|" * len(scenarios) + "}\n"
        "\\hline\n"
        "\\textbf{Algorithm} & "
        + " & ".join(f"\\textbf{{{SCENARIO_NAMES[i]}}}" for i in range(len(scenarios)))
        + " \\\\\n"
        " & " + " & ".join("$\\mu \\pm \\sigma$" for _ in scenarios) + " \\\\\n"
        "\\hline\n"
    )

    # Populate the table with data
    for algorithm in algorithms:
        table += algorithm
        for scenario in scenarios:
            mean_error = np.mean(
                __get_valid_data(average_error[scenario][algorithm].values())  # type: ignore
            )
            std_dev = np.mean(
                __get_valid_data(standard_deviation[scenario][algorithm].values())  # type: ignore
            )
            table += (
                f" & ${__format_number(mean_error)} \\pm {__format_number(std_dev)}$"
            )
        table += " \\\\\n"

    table += "\\hline\n\\end{tabular}\n}\n\\end{table*}"

    # Print the table
    print("Mean Table for Algorithms...")
    print(table)
    print("\n")
