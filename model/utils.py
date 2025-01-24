"""
Include several utilities for the project, as functions to generate
tables for LaTex, plots for the results or to generate statistical analysis.
"""
from typing import TypedDict
# External imports
import pydash as _py
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import numpy as np
import pandas as pd
# Local imports
from model.solver import BenchmarkResult

# Classes


class AnovaResult(TypedDict):
    """Result of the ANOVA test."""
    function: str
    f_value: float
    p_value: float
    Significance: str

# Functions!


def statistical_tests(
    benchmark_results: dict[str, list[BenchmarkResult]],
    to_latex: bool = False
) -> None:
    """Perform an ANOVA and Post Hoc test on the given data.

    We're going to test the null hypothesis that all algorithms have the same
    performance on the benchmark functions.

    **Arguments**:
        - results: dict[str, list[BenchmarkResult]] - The results of the
          benchmark functions for each algorithm.

    **Returns**:
    """
    functions: set[str] = set()

    def _get_values(result: BenchmarkResult) -> str:
        """Get the value of the result."""
        # If the function name is not on functions, add it
        if result["function"] not in functions:
            functions.add(result["function"])
        # At the end, just return the value
        return result["function"]

    # Let's create the dictionary with the F values and p-values
    # for each function. For that' let's group all the benchmark results
    # by function
    statistical_data = {
        # First, in the tuple, set the algorithm name
        # Then, secondly, add the group by for each function
        arg: _py.group_by(
            arg_results,
            _get_values
        )
        for arg, arg_results in benchmark_results.items()
    }
    # Now, let's iterate over each algorithm and function...
    anova_tests: list[AnovaResult] = []
    turkeys_test: dict[str, pd.DataFrame] = {}

    for function in functions:
        # Get the results for each algorithm
        function_results = {
            alg_name: results[function]
            for alg_name, results in statistical_data.items()
        }
        # Now, let's get the values for each algorithm
        values = [
            np.array([res["best_value"] for res in results])
            for results in function_results.values()
        ]
        # Based on this, generate the anova test
        anova_tests.append(_anova_test(values, function))
        turkeys_test[function] = _post_hoc_test(function_results, values)
        # Then, print each one of the results
        if to_latex is False:
            print(64*"=")
            print("DATA\n")
            print({
                alg: np.array([res["best_value"] for res in results])
                for alg, results in function_results.items()
            })
            print(f"\n[Function: {function}]")
            print(turkeys_test[function].to_string(index=False))
            print("\n")
    if to_latex is False:
        print(64*"/")
        print("ANOVA RESULTS:\n")
        print(pd.DataFrame(anova_tests).to_string(index=False))
    else:
        # Store this in `anova_results.tex`
        with open("anova_results.tex", "w", encoding="utf-8") as file:
            file.write(pd.DataFrame(anova_tests).to_latex(index=False))
        # Join all the Turkey tests, adding the function as the initial column
        # and store it in `turkey_results.tex`
        with open("turkey_results.tex", "w", encoding="utf-8") as file:
            file.write(
                pd.concat(
                    [df.assign(Function=function).reindex(columns=[
                        # Select the columns to show...
                        # ignore the lower and upper
                        "Function", "Algorithm 1", "Algorithm 2",
                        "Median Difference", "p", "Significant"
                    ])
                        for function, df in turkeys_test.items()]
                ).to_latex(index=False)
            )
    # Print the general media of the results... for each function and each algorithm


def _anova_test(
    statistical_values: list[np.ndarray],
    function: str
) -> AnovaResult:
    """Perform the ANOVA test for the given data."""
    # Now, let's perform the ANOVA test
    f_value, p_value = f_oneway(*statistical_values)
    # At the end, return it
    return {
        "function": function,
        "f_value": f_value,
        "p_value": p_value,
        "Significance": "Significant" if p_value < 0.05 else "Not significant"
    }


def _post_hoc_test(
    statistical_data: dict[str, list[BenchmarkResult]],
    statistical_values: list[np.ndarray],
) -> pd.DataFrame:
    """Perform the Post-Hoc test for the given data."""

    data = pd.DataFrame({
        "values": np.concatenate(statistical_values),
        "groups": np.concatenate([
            np.repeat(group, len(results))
            for group, results in statistical_data.items()
        ])
    })
    # Run Tukey's HSD test
    tukey = pairwise_tukeyhsd(
        data["values"],
        data["groups"],
    )
    # Filter and reorder comparisons to keep SDAO in the first column
    comparisons = tukey.summary().data[1:]  # Skip the header row
    sdao_comparisons = [
        # A1, A2, Median Difference, p, Significant, lower, upper
        (comp[0], comp[1], comp[2], comp[3], comp[6], comp[4], comp[5])
        if comp[0] == 'SDAO'
        # Move the SDAO to be the first column...
        else (
            # A1, A2, Median Difference, p, Significant, lower, upper
            comp[1], comp[0], -comp[2], comp[3], comp[6], -comp[4], -comp[5]
        )
        for comp in comparisons
        if 'SDAO' in comp[0] or 'SDAO' in comp[1]
    ]

    # Drop the columns lower and upper
    return pd.DataFrame(
        sdao_comparisons,
        columns=["Algorithm 1", "Algorithm 2",
                 "Median Difference", "p", "Significant", "lower", "upper"]
    )
