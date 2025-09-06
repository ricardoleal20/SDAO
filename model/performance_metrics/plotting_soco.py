"""
Convergence plots for SOCO (scenario 5, d=500) only.

This script auto-detects data files for scenario_5_d_500 and produces a
single-page PDF with convergence curves for all SOCO functions, following
the visual style of the existing plotting utilities.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import pickle

import numpy as np
import matplotlib.pyplot as plt

# Reuse style and helpers from the main plotting module to keep consistency
from model.performance_metrics.plotting import (  # type: ignore
    COLORS,
    LINESTYLES,
    _order_algorithms,
    _display_name,
)
from model.functions.soco_funcs import soco_funcs


# Configure the PLT style to match the rest of the project
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 36
plt.rcParams["axes.labelsize"] = 24
plt.rcParams["xtick.labelsize"] = 36
plt.rcParams["ytick.labelsize"] = 36
plt.rcParams["legend.fontsize"] = 16


AlgorithmRuns = List[dict]
ScenarioData = Dict[str, AlgorithmRuns]


def _discover_data_files() -> List[Path]:
    """Return only the exact file scenario_5_d_500.pkl if present in CWD."""
    p = Path.cwd() / "scenario_5_d_500.pkl"
    return [p] if p.exists() else []


def _load_pickle(path: Path) -> ScenarioData | None:
    try:
        with path.open("rb") as f:
            data = pickle.load(f)
        if isinstance(data, dict):
            return data  # Expecting {algorithm: [runs...]}
        return None
    except Exception:
        return None


def _merge_data(datasets: List[ScenarioData]) -> ScenarioData:
    """Merge multiple scenario datasets by algorithm name."""
    merged: ScenarioData = {}
    for ds in datasets:
        for algorithm, runs in ds.items():
            merged.setdefault(algorithm, []).extend(runs)
    return merged


def _errors_per_iteration(runs: AlgorithmRuns) -> Tuple[List[int], List[float], List[float]]:
    """Compute mean/std absolute error per iteration across runs.

    Each run dict is expected to contain:
      - "trajectory": list[tuple[int, float]]
      - optionally "optimal_value" (if absent, 0 is used)
    """
    if not runs:
        return [], [], []
    optimal_value = runs[0].get("optimal_value", 0)
    errors_by_iter: Dict[int, List[float]] = {}
    for run in runs:
        trajectory = run.get("trajectory", [])
        for iteration, value in trajectory:
            err = abs(float(value) - float(optimal_value))
            errors_by_iter.setdefault(int(iteration), []).append(err)
    if not errors_by_iter:
        return [], [], []
    iterations = sorted(errors_by_iter.keys())
    mean_curve = [float(np.mean(errors_by_iter[it])) for it in iterations]
    std_curve = [float(np.std(errors_by_iter[it])) for it in iterations]
    return iterations, mean_curve, std_curve


def plot_soco_convergence(data: ScenarioData, store_as_pdf: bool = True) -> None:
    """Generate a convergence grid for all SOCO functions (scenario 5, d=500)."""
    num_functions = len(soco_funcs)
    if num_functions == 0:
        return

    num_cols = 3
    num_rows = (num_functions + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(17, 5 * num_rows))

    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]

    ordered_algs = _order_algorithms(list(data.keys()))

    for idx, bench_func in enumerate(soco_funcs):
        function_name = bench_func["name"]
        optimal_value = bench_func.get("optimal_value", 0)
        ax = axes[idx]
        ax.set_title(function_name, fontsize=20)
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Absolute Error")
        ax.set_yscale("log")
        ax.grid(True, linestyle="--", alpha=0.5)

        for i, algorithm in enumerate(ordered_algs):
            runs_for_func = [
                r
                for r in data.get(algorithm, [])
                if r.get("function") == function_name
            ]
            # If runs do not include optimal, inject from definition
            for r in runs_for_func:
                r.setdefault("optimal_value", optimal_value)
            iterations, mean_curve, std_curve = _errors_per_iteration(runs_for_func)
            if not iterations:
                continue

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
        ax.legend().remove()

    # Remove unused axes if any
    last_used = idx  # type: ignore[name-defined]
    for j in range(last_used + 1, len(axes)):
        fig.delaxes(axes[j])

    # Single legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=7, fontsize=16)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    output = "convergence_plot_scenario_5_d500.pdf"
    if store_as_pdf:
        plt.savefig(output, format="pdf", dpi=300, bbox_inches="tight", pad_inches=0.1)
        print(f"✅ Saved: {output}")
    else:
        plt.show()


def main() -> None:
    files = _discover_data_files()
    datasets: List[ScenarioData] = []
    for f in files:
        loaded = _load_pickle(f)
        if loaded:
            datasets.append(loaded)

    if not datasets:
        print("No data found for scenario_5_d_500. Expected: scenario_5_d_500.pkl in current directory.")
        return

    merged = _merge_data(datasets)
    plot_soco_convergence(merged, store_as_pdf=True)


if __name__ == "__main__":
    main()


