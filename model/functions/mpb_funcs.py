"""
Create a class that shows the MPB benchmark and a list of MPB benchmark
instances with different parameters. This benchmark would let us now the
behavior of how our algorithms behaves in a dynamic environment.
"""

from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from model.solver import ExperimentFunction


class MovingPeaksBenchmark:
    """Moving Peaks Benchmark (MPB) for dynamic optimization.

    This benchmark simulates a dynamic landscape with multiple peaks that change over time.
    Each peak is characterized by its position, height, and width.
    The objective function is defined as the maximum value over all peaks:

        f(x) = max_{i=1,...,n_peaks} (height_i - width_i * ||x - position_i||)

    Attributes:
        dimension (int): Dimensionality of the search space.
        n_peaks (int): Number of peaks.
        bounds (Tuple[float, float]): Lower and upper bounds for each dimension.
        peaks (List[dict]): List of peaks, each a dictionary with keys 'position',
                        'height', and 'width'.
        shift_severity (float): Maximum change in peak position per update.
        height_severity (float): Maximum change in peak height per update.
        width_severity (float): Maximum change in peak width per update.
    """

    def __init__(
        self,
        dimension: int,
        n_peaks: int,
        bounds: tuple[float, float] = (-100.0, 100.0),
        shift_severity: float = 1.0,
        height_severity: float = 7.0,
        width_severity: float = 0.01,
        seed: Optional[int] = None,
    ):
        """
        Initialize the Moving Peaks Benchmark.

        Args:
            dimension (int): Dimensionality of the search space.
            n_peaks (int): Number of peaks.
            bounds (Tuple[float, float], optional): Lower and upper bounds for each dimension.
                Default is (-100, 100).
            shift_severity (float, optional): Maximum change for peak positions per update.
            height_severity (float, optional): Maximum change in peak height per update.
            width_severity (float, optional): Maximum change in peak width per update.
            seed (int, optional): Random seed for reproducibility.
        """
        self.dimension = dimension
        self.n_peaks = n_peaks
        self.bounds = bounds
        self.shift_severity = shift_severity
        self.height_severity = height_severity
        self.width_severity = width_severity
        if seed is not None:
            np.random.seed(seed)
        self._initialize_peaks()

    def _initialize_peaks(self):
        """Initialize peaks using vectorized sampling.

        Positions are sampled uniformly from bounds with shape (n_peaks, dimension).
        Heights ~ U[30, 70] and widths ~ U[1, 5].
        """
        lower, upper = self.bounds
        self.positions = np.random.uniform(
            lower, upper, size=(self.n_peaks, self.dimension)
        )
        self.heights = np.random.uniform(30.0, 70.0, size=(self.n_peaks,))
        self.widths = np.random.uniform(1.0, 5.0, size=(self.n_peaks,))

    def update_peaks(self):
        """Vectorized dynamic update of peak positions, heights and widths."""
        lower, upper = self.bounds
        # Update positions
        shifts = np.random.uniform(
            -self.shift_severity, self.shift_severity, size=self.positions.shape
        )
        self.positions = np.clip(self.positions + shifts, lower, upper)
        # Update heights and widths
        h_delta = np.random.uniform(
            -self.height_severity, self.height_severity, size=self.heights.shape
        )
        w_delta = np.random.uniform(
            -self.width_severity, self.width_severity, size=self.widths.shape
        )
        self.heights = np.clip(self.heights + h_delta, 30.0, 70.0)
        self.widths = np.clip(self.widths + w_delta, 1.0, 5.0)

    def __call__(self, x: np.ndarray) -> float:
        """Evaluate f(x) = max_i height_i - width_i * ||x - position_i|| (vectorized)."""
        # Broadcast x against all positions: (n_peaks, dim) - (dim,) -> (n_peaks, dim)
        diffs = self.positions - x
        dists = np.linalg.norm(diffs, axis=1)
        values = self.heights - self.widths * dists
        return float(np.max(values))


# ----------------------------------------------------------------
# Create a list of MPB benchmark instances with different parameters.
# Each dictionary in the list follows a similar structure as your other benchmarks.
# ----------------------------------------------------------------
mpb_benchmarks: list["ExperimentFunction"] = [  # type: ignore
    {
        "name": "MPB_5Peaks_d10",
        "call": MovingPeaksBenchmark(
            dimension=10,
            n_peaks=5,
            bounds=(-100, 100),
            shift_severity=1.0,
            height_severity=7.0,
            width_severity=0.01,
            seed=42,
        ),
        "domain": (-100, 100),
        "dimension": 10,
        "description": "Moving Peaks Benchmark with 5 peaks in 10 dimensions.",
    },
    {
        "name": "MPB_10Peaks_d10",
        "call": MovingPeaksBenchmark(
            dimension=10,
            n_peaks=10,
            bounds=(-100, 100),
            shift_severity=1.0,
            height_severity=7.0,
            width_severity=0.01,
            seed=42,
        ),
        "domain": (-100, 100),
        "dimension": 10,
        "description": "Moving Peaks Benchmark with 10 peaks in 10 dimensions.",
    },
    {
        "name": "MPB_10Peaks_d30",
        "call": MovingPeaksBenchmark(
            dimension=30,
            n_peaks=10,
            bounds=(-100, 100),
            shift_severity=1.0,
            height_severity=7.0,
            width_severity=0.01,
            seed=42,
        ),
        "domain": (-100, 100),
        "dimension": 30,
        "description": "Moving Peaks Benchmark with 10 peaks in 30 dimensions.",
    },
    {
        "name": "MPB_15Peaks_d50",
        "call": MovingPeaksBenchmark(
            dimension=50,
            n_peaks=15,
            bounds=(-100, 100),
            shift_severity=1.0,
            height_severity=7.0,
            width_severity=0.01,
            seed=42,
        ),
        "domain": (-100, 100),
        "dimension": 50,
        "description": "Moving Peaks Benchmark with 15 peaks in 50 dimensions.",
    },
    {
        "name": "MPB_20Peaks_d100",
        "call": MovingPeaksBenchmark(
            dimension=100,
            n_peaks=20,
            bounds=(-100, 100),
            shift_severity=1.0,
            height_severity=7.0,
            width_severity=0.01,
            seed=42,
        ),
        "domain": (-100, 100),
        "dimension": 100,
        "description": "Moving Peaks Benchmark with 20 peaks in 100 dimensions.",
    },
]
