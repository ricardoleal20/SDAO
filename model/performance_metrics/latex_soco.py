"""
Generate LaTeX tables for SOCO11 (scenario 5, d=500).

Outputs:
1) winners_soco.tex: Best mean absolute error per function and the winning algorithm.
2) soco_convergence_iterations.tex: Average iteration of numerical stability (<1e-4) per algorithm.
3) soco_mean_std.tex: Mean ± std (absolute error) per algorithm and function.

Input: scenario_5_d_500.pkl in the current working directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import pickle

from model.performance_metrics.metrics import (
    best_solution,
    stability,
    convergence_rate,
)
from model.functions.soco_funcs import soco_funcs
from model.performance_metrics.plotting import _display_name  # type: ignore
from model.solver import BenchmarkResult


BENCH_DATA = Dict[str, List[BenchmarkResult]]


def _load_data() -> BENCH_DATA:
    p = Path.cwd() / "scenario_5_d_500.pkl"
    if not p.exists():
        raise FileNotFoundError(
            "Expected scenario_5_d_500.pkl in the current directory."
        )
    with p.open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise ValueError("Pickle file has unexpected format (expected dict).")
    return data  # {algorithm: [runs...]}


def _latex_escape(text: str) -> str:
    return (
        text.replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
    )


def _fmt_num(x: float) -> str:
    # Use compact scientific notation if very small/large
    if x == 0:
        return "0"
    ax = abs(x)
    if ax < 1e-3 or ax >= 1e4:
        return f"{x:.2e}"
    return f"{x:.4f}"


def _fmt_pm(mean: float, std: float) -> str:
    return f"{_fmt_num(mean)} $\\pm$ {_fmt_num(std)}"


def _get_all_algorithms(data: BENCH_DATA) -> List[str]:
    return sorted(list(data.keys()))


def _get_all_functions() -> List[str]:
    return [f["name"] for f in soco_funcs]


def _write_file(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    print(f"✅ Saved: {path.name}")


def generate_winners_table(data: BENCH_DATA) -> None:
    mean_err = best_solution(data)
    std_err = stability(data)
    algorithms = _get_all_algorithms(data)
    functions = _get_all_functions()

    rows = []
    for func in functions:
        # Identify winner across algorithms
        best_alg = None
        best_val = float("inf")
        for alg in algorithms:
            val = mean_err.get(alg, {}).get(func, None)
            if val is None:
                continue
            if val < best_val:
                best_val = float(val)
                best_alg = alg

        if best_alg is None:
            continue
        std_val = float(std_err.get(best_alg, {}).get(func, 0.0))
        rows.append(
            (
                _latex_escape(func),
                _latex_escape(_display_name(best_alg)),
                _fmt_pm(best_val, std_val),
            )
        )

    header = (
        "\\begin{tabular}{l l r}\n"
        "\\toprule\n"
        "Function & Algorithm & Best (mean ± std) \\ \\midrule\n"
    )
    body = "\n".join([f"{f} & {a} & {v} \\" for f, a, v in rows])
    footer = "\n\\bottomrule\n\\end{tabular}\n"
    content = (
        "\\begin{table}[!ht]\\centering\n" + header + body + footer + "\\end{table}\n"
    )
    _write_file(Path("winners_soco.tex"), content)


def generate_convergence_table(data: BENCH_DATA) -> None:
    conv = convergence_rate(data, scenario=5)
    algorithms = _get_all_algorithms(data)
    functions = _get_all_functions()

    # Build LaTeX table with functions as rows and algorithms as columns
    col_header = " & ".join(_latex_escape(_display_name(a)) for a in algorithms)
    col_spec = ("l " + ("c " * len(algorithms))).strip()
    header = (
        "\\begin{tabular}{" + col_spec + "}\n"
        "\\toprule\n"
        f"Function & {col_header} \\ \\midrule\n"
    )

    lines = []
    for func in functions:
        cells = []
        for alg in algorithms:
            it = conv.get(alg, {}).get(func, None)
            cells.append("-" if it is None else f"{int(round(float(it)))}")
        line = f"{_latex_escape(func)} & " + " & ".join(cells) + " \\\\"
        lines.append(line)

    footer = "\n\\bottomrule\n\\end{tabular}\n"
    content = (
        "\\begin{table}[!ht]\\centering\n"
        "\\caption{Average iteration of numerical stability (< $10^{-4}$).}\n"
        + header
        + "\n".join(lines)
        + footer
        + "\\end{table}\n"
    )
    _write_file(Path("soco_convergence_iterations.tex"), content)


def generate_mean_std_table(data: BENCH_DATA) -> None:
    mean_err = best_solution(data)
    std_err = stability(data)
    algorithms = _get_all_algorithms(data)
    functions = _get_all_functions()

    # Functions as rows, algorithms as columns, entries are mean ± std
    col_header = " & ".join(_latex_escape(_display_name(a)) for a in algorithms)
    col_spec = ("l " + ("c " * len(algorithms))).strip()
    header = (
        "\\begin{tabular}{" + col_spec + "}\n"
        "\\toprule\n"
        f"Function & {col_header} \\ \\midrule\n"
    )

    lines = []
    for func in functions:
        cells = []
        for alg in algorithms:
            m = mean_err.get(alg, {}).get(func, None)
            s = std_err.get(alg, {}).get(func, None)
            cells.append("-" if (m is None or s is None) else _fmt_pm(float(m), float(s)))
        line = f"{_latex_escape(func)} & " + " & ".join(cells) + " \\\\"
        lines.append(line)

    footer = "\n\\bottomrule\n\\end{tabular}\n"
    content = (
        "\\begin{table}[!ht]\\centering\n"
        "\\caption{Mean ± std (absolute error).}\n"
        + header
        + "\n".join(lines)
        + footer
        + "\\end{table}\n"
    )
    _write_file(Path("soco_mean_std.tex"), content)


def main() -> None:
    data = _load_data()
    generate_winners_table(data)
    generate_convergence_table(data)
    generate_mean_std_table(data)


if __name__ == "__main__":
    main()


