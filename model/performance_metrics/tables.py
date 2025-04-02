"""
Generate the tables of different data processes
"""

from typing import Sequence
import numpy as np
from scipy.stats import f_oneway, wilcoxon

# Local imports
from model.utils.statistical_utils import _post_hoc_test

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


def generate_anova_result(data: dict[int, dict[str, list[dict]]]) -> None:
    """Generate an ANOVA results for different"""
    # Then, create s single array with the data for all the errors obtained, in all the functions for all the algorithms
    for scenario, scenario_data in data.items():
        # Create the data array
        anova_data = [[e["error"] for e in d] for d in scenario_data.values()]
        # Using this, get the anova result
        f_result, p_result = f_oneway(*anova_data)
        print(f"ANOVA Result for Scenario {scenario}:")
        print(f"F-Value: {f_result}")
        print(f"P-Value: {p_result}")
        print("\n")


def generate_post_hoc_test(data: dict[int, dict[str, list[dict]]]) -> None:
    """Generate a post-hoc test for different"""
    # Then, create s single array with the data for all the errors obtained, in all the functions for all the algorithms
    for scenario, scenario_data in data.items():
        # Create the data array
        anova_data = [[e["error"] for e in d] for d in scenario_data.values()]
        # Perform post-hoc test
        post_hoc_result = _post_hoc_test(scenario_data, anova_data)  # type: ignore
        print(f"Post-Hoc Result for Scenario {scenario}:")
        print(post_hoc_result)
        print("\n")


def generate_wilcoxon_test(data: dict[int, dict[str, list[dict]]]) -> None:
    """Generate a Wilcoxon test for different"""
    # Then, create s single array with the data for all the errors obtained, in all the functions for all the algorithms
    for scenario, scenario_data in data.items():
        # Get the grouped data per function
        grouped_data = {}
        for alg, alg_data in scenario_data.items():
            # Then, iterate over the algorithm data
            func_data = {}
            for d in alg_data:
                if d["function"] not in func_data:
                    func_data[d["function"]] = []
                func_data[d["function"]].append(d["error"])
            grouped_data[alg] = func_data
        # Perform Wilcoxon test for SDAO against algorithm and each function
        sdao_data = grouped_data["SDAO"]
        for alg, func_errors in grouped_data.items():
            if alg == "SDAO":
                continue
            for func, errors in func_errors.items():
                # Perform Wilcoxon test for SDAO against algorithm and each function
                wilcoxon_result = wilcoxon(errors, sdao_data[func])
                print(
                    f"Wilcoxon Result for Scenario {scenario}, Algorithm {alg}, Function {func}:"
                )
                print(wilcoxon_result)
                print("\n")
