# Stochastic Diffusion Adaptive Optimization (SDAO)

## Description

The Stochastic Diffusion Adaptive Optimization (SDAO) is a stochastic optimization algorithm inspired by adaptive diffusion. This algorithm is used to solve optimization problems in high-dimensional search spaces. SDAO is compared against various state-of-the-art (SoA) optimization algorithms across different benchmark scenarios.

## Benchmark Functions

The SDAO model has been tested against a variety of benchmark functions, which are divided into the following categories:

1. **Standard Benchmark Functions**: These are classical test functions commonly used in optimization literature.
2. **Stochastic Benchmark Functions**: These functions include stochastic noise to simulate uncertainty in function evaluations.
3. **Real-World Functions**: These functions represent real-world optimization problems, such as Vehicle Routing Problem (VRP), predictive maintenance, and more.
4. **CEC Functions**: These are test functions from the Competitions on Evolutionary Computation (CEC), which are widely used to evaluate optimization algorithms.

## Compared Algorithms

SDAO is compared against several state-of-the-art optimization algorithms, including:

- **Stochastic Fractal Search (SFS)**
- **Algebraic Stochastic Gradient Descent (ASGD)**
- **SHADE with Iterative Local Search (SHADEwithILS)**
- **Path Relinking**
- **Adaptive Multi-Swarm Optimization (AMSO)**
- **Two-Level Particle Swarm Optimization (TLPSO)**

## Requirements

To run the model and tests, you will need to have the following Python packages installed:

- `numpy`
- `scipy`
- `pandas`
- `matplotlib`
- `seaborn`
- `optuna`
- `tqdm`
- `pydash`
- `statsmodels`

You can install these packages using `poetry`:

```bash
poetry install
```

## Project Structure

The project is organized as follows:

- `SDAO/model/`: Contains the implementation of the SDAO model and other optimization algorithms.
  - `__init__.py`: Initializes the module.
  - `__main__.py`: Runs the SDAO algorithm.
  - `functions/`: Contains the benchmark functions.
  - `performance_metrics/`: Contains performance metrics and scripts to generate plots and tables.
  - `soa/`: Contains the implementation of state-of-the-art optimization algorithms.
  - `solver.py`: Defines the solver interface for testing and comparing models.
  - `utils/`: Contains utilities for parameter optimization and statistical analysis.

## Running the Model

### Running the SDAO Algorithm

To run the SDAO algorithm, you can use the following command:

```bash
poetry run python -m model
```

This command will execute the SDAO algorithm with the default parameters and generate the benchmark results.

### Parameter Optimization

To optimize the SDAO parameters using Optuna, you can run the following script:

```bash
poetry run python -m model.utils.optimize_params
```

This script will perform a hyperparameter search using Optuna and display the best parameters found.

### Performance Metrics

To generate performance metrics and plots, you can run the following script:

```bash
poetry run python -m model.performance_metrics.run -s all -d 50 -pdf
```

This command will generate performance metrics for all scenarios and specified dimensions, and save the plots in PDF format.

## License

This project is licensed under the MIT License. You can view the license file for more details.
