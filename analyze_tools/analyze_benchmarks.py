"""
Simple graphic tool to analyze the results of the benchmarks.

It would generate three things:
    - A csv for the results
    - A line plot for the results

What we need to run this script, is:
    - The name of the benchmark to run. Those options can be:
        * alpha
        * beta
        * gamma
        * diff
        * n_of_particles
        * all
"""
# Built In imports
import typing as tp
import json
from argparse import ArgumentParser
# External imports
import numpy as np


class BenchMark(tp.TypedDict):
    """The benchmark dictionary type."""
    function: str  # The function to optimize
    alpha: float  # The alpha value
    gamma: float  # The gamma value
    beta: float  # The beta value
    diff_coeff: float  # The diffusion coefficient
    num_particles: int  # The number of particles
    num_threads: int  # The number of threads
    avg_best_value: float  # The average value
    avg_iterations: int  # The average number of iterations
    avg_time: float  # The average time taken to run the algorithm


# * Define the LaTex equivalent for each parameter
latex_equivalent = {
    "alpha": r"$\alpha$",
    "beta": r"$\beta$",
    "gamma": r"$\gamma$",
    "diff_coeff": r"$D^0$",
    "single_test": r"SDAO"
}

# * Define the expected values for each one of the benchmarks
expected_values = {
    "Sphere": 0.0,
    "Rastrigin": 0.0,
    "Rosenbrock": 0.0,
    "Ackley": 0.0,
    "Schwefel": 0.0,
}


def read_json_file(param: str) -> list[BenchMark]:
    """From the JSON file, we'll try to read the benchmark solution.
    It would also return a formatted dictionary
    
    Args:
        param (str): The name of the benchmark to read.
    
    Returns:
        dict: The benchmark information.
    """
    with open(f"benchmark_results_{param}.json", "r", encoding="utf-8") as file:
        data = json.load(file)
    # Then, generate the dictionary of solutions
    benchmark_data: list[BenchMark] = []
    for solution in data:
        benchmark_data.append({
            "function": solution["function"],
            "alpha": solution["alpha"],
            "gamma": solution["gamma"],
            "beta": solution["beta"],
            "diff_coeff": solution["diff_coeff"],
            "num_particles": solution["num_particles"],
            "num_threads": solution["num_threads"],
            "avg_best_value": np.mean(solution["best_values"]),
            "avg_iterations": np.mean(solution["iterations"]),
            "avg_time": np.mean(solution["time_seconds"])
        })
    return benchmark_data


def generate_csv_for_table(param: str, data: list[BenchMark]) -> None:
    """Generate the CSV for the data provided
    
    The columns are going to be:
        - Parameter (in LaTex)
        - Each one of the functions

    
    """
    # Convert to a CSV the data.
    csv_content = f"{latex_equivalent[param]}, {','.join(key for key in expected_values)}\n"
    for value in data:
        csv_content += f"{value[param]}, {value['avg_best_value']}," +\
            f" {value['avg_iterations']}, {value['avg_time']}\n"
    # At the end, write the CSV file
    with open(f"table_data_{param}.csv", "w", encoding="utf-8") as file:
        file.write(csv_content)


def generate_latex_table(param: str, data: list[BenchMark]) -> None:
    """Generate the CSV for the data provided
    
    The columns are going to be:
        - Parameter (in LaTex)
        - Each one of the functions
    """
    # First of all, group the data for his function to solve.
    # We wanto have the param as the key and then, each one of the functions
    param_values = []
    grouped_data = {}
    for value in data:
        obj_value = value[param]
        function = value["function"]
        # Set the param value only if it's not in the existing param values
        if obj_value not in param_values:
            param_values.append(obj_value)

        # Then, evaluate if the function is not in the grouped data
        if function not in grouped_data:
            grouped_data[function] = {}
        # Then, add the average value for this function
        grouped_data[function][obj_value] = value["avg_best_value"]
    # Sort the param values
    param_values = list(sorted(param_values))
    # Then, initialize the table defining their columns
    latex_table: list[str] = [r"\begin{tabular}{|c|c|c|c|} \hline"]

    # Get the column input
    column_data = " & ".join(
        [f"{latex_equivalent[param]}"] + [f"{val}" for val in param_values]
    )
    latex_table.append(column_data[0:-2] + r" \\ \hline\hline")
    # Then, iterate over the grouped data
    for function, function_values in grouped_data.items():
        # Then, depending on the function, we'll know
        row_data = r"\textbf{" + function + "} & "
        # Then, iterating over the function values, get their inputs
        for param_value in param_values:
            row_data += f"{function_values.get(param_value, 0.0):.2e} & "
        # And, at the end, append this row data to the latex table
        latex_table.append(row_data[0:-2] + r"\\")
    # Then, append the final lines
    latex_table.extend([r"\hline", r"\end{tabular}"])

    # At the end, print the table input in the terminal
    print(f"For the parameter {param}, here is the table input:\n")
    print("\n".join(latex_table))
    print(32 * "=")


def analyze_benchmark(param: str) -> None:
    """Analyze the benchmark results.
    
    Args:
        param (str): The name of the benchmark to analyze.
    """
    # Read the JSON file
    data = read_json_file(param)
    # Generate the CSV file for the LaTex table
    generate_latex_table(param, data)


def analyze_single_test(param: str) -> None:
    """Analyze the benchmark results for a single test using the data provided."""
    # Read the JSON file
    data = read_json_file(param)
    # Then, generate the latex table
    latex_table: list[str] = [
        r"\begin{tabular}{|c|c|c|c|} \hline",
        r" & \textbf{Value} & \textbf{Iterations} & \textbf{Time (s)} \\" +
        r" \hline\hline",
    ]
    # Iterate over the data to see the results
    for datum in data:
        # From the datum, get the function and the needed values
        function = datum["function"]
        best_value = datum["avg_best_value"]
        iterations = int(round(datum["avg_iterations"]))
        time = datum["avg_time"]
        # With this, generate the column for the latex table
        row_data = f"{function} & {best_value:.2e} & {iterations} & {time:.2e} \\\\"
        # Append the row data to the latex table
        latex_table.append(row_data)
    # Then, append the final lines
    latex_table.extend([r"\hline", r"\end{tabular}"])
    # At the end, print the table input in the terminal
    print("\n".join(latex_table))


if __name__ == "__main__":
    # Define the parser
    parser = ArgumentParser(description="Analyze the benchmarks results")
    parser.add_argument(
        "-b", "--benchmark", type=str,
        help="The benchmark to analyze", default="all"
    )
    # Parde the arguments
    args = parser.parse_args()

    # Available options
    available_options = ["single_test", "alpha", "beta",
                         "gamma", "diff_coeff", "all"]

    if args.benchmark not in available_options:
        raise ValueError(
            "Invalid benchmark. The only available options are:" +
            "[single_test, alpha, beta, gamma, diff_coeff, all]"
        )
    # Then, we'll only check if we have 'all'. If we do, we'll run all the benchmarks
    # iterating over the available options. If not, we'll only run it for the normal
    if args.benchmark == "all":
        for option in available_options[:-1]:
            analyze_benchmark(option)
    elif args.benchmark == "single_test":
        analyze_single_test("single_test")
    else:
        analyze_benchmark(args.benchmark)
