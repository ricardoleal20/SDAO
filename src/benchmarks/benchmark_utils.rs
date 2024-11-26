// External crates imports
use itertools::Itertools;
use ndarray::Array1;
use serde::Serialize;
use std::fs::File;
// Local imports
use sdao::{Status, SDAO};

// Structure to store the benchmark results. This structure will be serialized to JSON.
#[derive(Serialize)]
pub struct BenchmarkResult {
    pub function: String,
    pub alpha: f64,
    pub gamma: f64,
    pub diff_coeff: f64,
    pub beta: f64,
    pub num_particles: usize,
    pub num_threads: usize,
    pub best_values: Vec<f64>,
    pub iterations: Vec<usize>,
    pub time_seconds: Vec<f64>,
    pub status: Vec<String>,
}

// Function to execute the benchmark multiple times for a set of parameters.
pub fn run_benchmark_multiple(
    objective_fn: fn(&Array1<f64>) -> f64,
    function_name: &str,
    range_values: Vec<f64>,
    alpha: f64,
    gamma: f64,
    diff_coeff: f64,
    beta: f64,
    num_particles: usize,
    num_threads: usize,
    max_iterations: usize,
    verbose: bool,
    repetitions: usize,
    dimension: Option<usize>,
) -> BenchmarkResult {
    let dimension = dimension.unwrap_or(1);
    let mut best_values = Vec::with_capacity(repetitions);
    let mut iterations = Vec::with_capacity(repetitions);
    let mut time_seconds = Vec::with_capacity(repetitions);
    let mut status = Vec::with_capacity(repetitions);

    for _ in 0..repetitions {
        // Initialize and run the optimizer
        let mut optimizer = SDAO::new(
            num_particles,
            dimension,
            range_values.clone(),
            alpha,
            gamma,
            diff_coeff,
            beta,
            max_iterations,
            objective_fn,
            Some(1e-6),
            Some(num_threads),
            Some(verbose),
        );
        let (run_status, best_particle) = optimizer.run();
        // Match the status of the run to store it as a string
        let (status_str, best_value) = match run_status {
            Status::Optimal => (
                "Optimal".to_string(),
                best_particle.map_or(f64::INFINITY, |p| p.best_value),
            ),
            Status::Feasible => (
                "Feasible".to_string(),
                best_particle.map_or(f64::INFINITY, |p| p.best_value),
            ),
            Status::Infeasible => ("Infeasible".to_string(), f64::INFINITY),
        };

        // Set the parameters to store, such as the values, iterations, time (in seconds) and status.
        best_values.push(best_value);
        iterations.push(optimizer.used_iterations);
        time_seconds.push(optimizer.execution_time);
        status.push(status_str);
    }

    BenchmarkResult {
        function: function_name.to_string(),
        alpha,
        gamma,
        diff_coeff,
        beta,
        num_particles,
        num_threads,
        best_values,
        iterations,
        time_seconds,
        status,
    }
}

/// Function to generate a different group of parameter combinations to improve the benchmarks
pub fn generate_parameter_combinations(
    alpha_values: &[f64],
    gamma_values: &[f64],
    diff_coeffs: &[f64],
    beta_values: &[f64],
) -> Vec<(f64, f64, f64, f64)> {
    alpha_values
        .iter()
        .cartesian_product(gamma_values.iter())
        .cartesian_product(diff_coeffs.iter())
        .cartesian_product(beta_values.iter())
        .map(|(((a, g), d), b)| (*a, *g, *d, *b))
        .collect()
}

/// Function to store the results from the benchmarks tests in Json Format
pub fn save_benchmark_results(results: &[BenchmarkResult], filename: &str) -> std::io::Result<()> {
    let file = File::create(filename)?;
    serde_json::to_writer_pretty(file, results)?;
    Ok(())
}
