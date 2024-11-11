// External crates imports
use rayon::prelude::*;
// Instance the benchmark module
mod benchmarks;
// Import the be benchmarks functions
use benchmarks::functions::{
    ackley_function, rastrigin_function, rosenbrock_function, schwefel_function, sphere_function,
};
use benchmarks::{
    generate_parameter_combinations, run_benchmark_multiple, save_benchmark_results,
    BenchmarkResult,
};

fn main() {
    // * General configuration for the parameters

    // let alpha_values = vec![0.001, 0.01, 0.1];
    // let gamma_values = vec![0.1, 0.5, 1.0];
    // let diff_values = vec![0.1, 1.0, 2.0];
    // let beta_values = vec![0.001, 0.01, 0.1];
    // let num_particles_values = vec![20, 100, 200];
    // let num_threads_values = vec![1, 4, 8];
    // let max_iterations = 500;
    // let verbose = false;
    // let repetitions = 10; // No. of repetitions to make with each test

    //let alpha_values = vec![0.001, 0.01, 0.1];
    // let gamma_values = vec![0.1, 0.5, 1.0];
    //let diff_values = vec![0.1, 1.0, 2.0];
    let beta_values = vec![0.001, 0.01, 0.1];
    //let num_particles_values = vec![20, 100, 200];
    let num_threads_values = vec![1];
    let max_iterations = 500;
    let verbose = false;
    let repetitions = 100; // No. of repetitions to make with each test

    // Define the default parameters for the combinations
    let alpha_values = vec![0.01];
    let gamma_values = vec![0.5];
    let diff_values = vec![1.0];
    //let beta_values = vec![0.01];
    let num_particles_values = vec![100];
    // // Get the sets of parameters
    // let parameter_sets = vec![
    //     ("alpha", &alpha_values, &gamma_val, &diff_val, &beta_val, &num_particles_val),
    //     ("gamma", &alpha_val, &gamma_values, &diff_val, &beta_val, &num_particles_val),
    //     ("diff", &alpha_val, &gamma_val, &diff_values, &beta_val, &num_particles_val),
    //     ("beta", &alpha_val, &gamma_val, &diff_val, &beta_values, &num_particles_val),
    //     ("num_particles", &alpha_val, &gamma_val, &diff_val, &beta_val, &num_particles_values),
    // ];

    // List all our test objective functions to test with them
    let benchmark_functions: Vec<(&str, fn(&ndarray::Array1<f64>) -> f64, Vec<f64>)> = vec![
        ("Sphere", sphere_function, vec![-10.0, 10.0]),
        ("Rosenbrock", rosenbrock_function, vec![-500.0, 500.0]),
        ("Rastrigin", rastrigin_function, vec![-5.12, 5.12]),
        ("Ackley", ackley_function, vec![-10.0, 10.0]),
        ("Schwefel", schwefel_function, vec![-500.0, 500.0]),
    ];

    let mut results: Vec<BenchmarkResult> = Vec::new();

    // Iterate over our combinations of parameters and objective functions
    for (func_name, func, range_values) in benchmark_functions.iter() {
        let parameter_combinations = generate_parameter_combinations(
            &alpha_values,
            &gamma_values,
            &diff_values,
            &beta_values,
            &num_particles_values,
            &num_threads_values,
        );

        let func_results: Vec<BenchmarkResult> = parameter_combinations
            .par_iter()
            .map(
                |(alpha, gamma, diff_coeff, beta, num_particles, num_threads)| {
                    run_benchmark_multiple(
                        *func,
                        func_name,
                        range_values.clone(),
                        *alpha,
                        *gamma,
                        *diff_coeff,
                        *beta,
                        *num_particles,
                        *num_threads,
                        max_iterations,
                        verbose,
                        repetitions,
                    )
                },
            )
            .collect();

        results.extend(func_results);
    }

    // Store the results in a JSON file, to analyze them better with Python
    if let Err(e) = save_benchmark_results(&results, "benchmark_results_beta.json") {
        eprintln!("Error saving results: {}", e);
    } else {
        println!(
            "Benchmarking completed. Results saved in 'benchmark_results_{}.json'.",
            "beta"
        );
    }
}
