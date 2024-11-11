// External crates imports
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;
// Import the benchmarks functions and utilities
mod benchmark_utils;
mod functions;

use benchmark_utils::{
    generate_parameter_combinations, run_benchmark_multiple, save_benchmark_results,
    BenchmarkResult,
};
use functions::{
    ackley_function, rastrigin_function, rosenbrock_function, schwefel_function, sphere_function,
};

/// Perform the benchmarks for the sensitivity analysis of the parameters
///
/// This function will perform the benchmarks for the sensitivity analysis of the parameters
/// of the SDAO. It will test the parameters `alpha`, `gamma`, `diff_coeff` and `beta`
/// against the functions `Sphere`, `Rosenbrock`, `Rastrigin`, `Ackley` and `Schwefel`.
///
/// It will store the results of each benchmark on their corresponding JSON file.
pub fn sensitibility_benchmarks() {
    // * General configuration for the parameters
    let alpha_values = vec![0.001, 0.01, 0.1];
    let gamma_values = vec![0.1, 0.5, 1.0];
    let diff_values = vec![0.1, 1.0, 2.0];
    let beta_values = vec![0.001, 0.01, 0.1];

    // Define the default values for the parameters
    let alpha_default_value = vec![0.01];
    let gamma_default_value = vec![0.5];
    let diff_default_value = vec![1.0];
    let beta_default_value = vec![0.01];
    // Include extra configuration
    let num_particles_values = 100;
    let num_threads_values: usize = 1;
    let max_iterations = 500;
    let verbose = false;
    let repetitions = 100; // No. of repetitions to make with each test

    // // Get the sets of parameters
    let parameter_sets = vec![
        (
            "alpha",
            &alpha_values,
            &gamma_default_value,
            &diff_default_value,
            &beta_default_value,
        ),
        (
            "gamma",
            &alpha_default_value,
            &gamma_values,
            &diff_default_value,
            &beta_default_value,
        ),
        (
            "diff_coeff",
            &alpha_default_value,
            &gamma_default_value,
            &diff_values,
            &beta_default_value,
        ),
        (
            "beta",
            &alpha_default_value,
            &gamma_default_value,
            &diff_default_value,
            &beta_values,
        ),
    ];

    // List all our test objective functions to test with them
    let benchmark_functions: Vec<(&str, fn(&ndarray::Array1<f64>) -> f64, Vec<f64>)> = vec![
        ("Sphere", sphere_function, vec![-5.12, 5.12]),
        ("Rosenbrock", rosenbrock_function, vec![-5.0, 10.0]),
        ("Rastrigin", rastrigin_function, vec![-5.12, 5.12]),
        ("Ackley", ackley_function, vec![-32.768, 32.768]),
        ("Schwefel", schwefel_function, vec![-500.0, 500.0]),
    ];

    let mut results: Vec<BenchmarkResult> = Vec::new();

    // Iterate over the combinations of functions
    for (param, alpha_values, gamma_values, diff_values, beta_values) in parameter_sets.iter() {
        // Here, iterate over the parameter sets
        for (func_name, func, range_values) in benchmark_functions.iter() {
            // Here, create the combinations for the parameter that contains the extra values on it
            let parameter_combinations = generate_parameter_combinations(
                &alpha_values,
                &gamma_values,
                &diff_values,
                &beta_values,
            );
            // Then, get the results from the function
            let func_results: Vec<BenchmarkResult> = parameter_combinations
                .par_iter()
                .map(|(alpha, gamma, diff_coeff, beta)| {
                    run_benchmark_multiple(
                        *func,
                        func_name,
                        range_values.clone(),
                        *alpha,
                        *gamma,
                        *diff_coeff,
                        *beta,
                        num_particles_values,
                        num_threads_values,
                        max_iterations,
                        verbose,
                        repetitions,
                    )
                })
                .collect();

            // Extend the results with the new results
            results.extend(func_results);
        }
        // Once we finished the benchmarking, we save the results in a JSON file for this parameter
        let filename = format!("benchmark_results_{}.json", param);
        // Store the results in a JSON file, to analyze them better with Python
        if let Err(e) = save_benchmark_results(&results, &filename) {
            eprintln!("Error saving results: {}", e);
        } else {
            println!("Benchmarking completed. Results saved in '{}'.", filename);
        }
    }
}

/// Perform a single test benchmark on the benchmark functions
///
/// This function would use a default set of parameters just to test the SDAO behaviour.
/// It would also store the results in a JSON file for analyze them later.
pub fn single_test_benchmark() {
    // Define the default values for the parameters. I will use the same vectorized values
    // as it was used in the sensitivity analysis, since it is easier to reuse the other functions as the
    // one that store the data
    let alpha_value = 0.01;
    let gamma_value = 0.5;
    let diff_value = 1.0;
    let beta_value = 0.01;
    let num_particles_value = 100;
    let num_threads_value: usize = 1;
    let max_iterations = 500;

    // Define the functions with their search-space parameters
    let benchmark_functions: Vec<(&str, fn(&ndarray::Array1<f64>) -> f64, Vec<f64>)> = vec![
        ("Sphere", sphere_function, vec![-5.12, 5.12]),
        ("Rosenbrock", rosenbrock_function, vec![-5.0, 10.0]),
        ("Rastrigin", rastrigin_function, vec![-5.12, 5.12]),
        ("Ackley", ackley_function, vec![-32.768, 32.768]),
        ("Schwefel", schwefel_function, vec![-500.0, 500.0]),
    ];
    // Create the vecto for the results
    let mut results: Vec<BenchmarkResult> = Vec::new();

    // Then, iterate over it to test the SDAO
    for (func_name, func, range_values) in benchmark_functions.iter() {
        // Run the benchmark for the function
        let result = run_benchmark_multiple(
            *func,
            func_name,
            range_values.clone(),
            alpha_value,
            gamma_value,
            diff_value,
            beta_value,
            num_particles_value,
            num_threads_value,
            max_iterations,
            false,
            100,
        );
        // Extend the vector with the solution
        results.push(result);
        println!("Benchmark for function '{}' completed.", func_name);
    }
    // Store the results in a JSON file, to analyze them better with Python
    let filename = format!("benchmark_results_single_test.json",);
    if let Err(e) = save_benchmark_results(&results, &filename) {
        eprintln!("Error saving results: {}", e);
    } else {
        println!("Benchmarking completed. Results saved in '{}'.", filename);
    }
}
