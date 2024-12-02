// External crates imports
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;
use sysinfo::System;
// Import the benchmarks functions and utilities
mod benchmark_utils;
mod functions;
mod other_algorithms;

use benchmark_utils::{
    generate_parameter_combinations, run_benchmark_multiple, save_benchmark_results,
    BenchmarkResult,
};
use functions::{
    ackley_function,
    beale_function,
    booth_function,
    drop_wave_function,
    expanded_schaffer_f6_function,
    griewank_function,
    happy_cat_function,
    rastrigin_function,
    // Noisy functions
    rastrigin_noisy_function,
    rosenbrock_function,
    salomon_function,
    schaffer_f7_function,
    schwefel_function,
    sphere_function,
    weierstrass_function,
    xin_she_yang_1_function,
};
// Algorithms imports
use other_algorithms::{fick_law_algorithm, gradient_descent, pso, simulated_annealing};
use sdao::SDAO;

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
    let benchmark_functions: Vec<(&str, fn(&ndarray::Array1<f64>) -> f64, Vec<f64>, usize)> = vec![
        ("Sphere", sphere_function, vec![-5.12, 5.12], 1),
        ("Rosenbrock", rosenbrock_function, vec![-5.0, 10.0], 2),
        ("Rastrigin", rastrigin_function, vec![-5.12, 5.12], 1),
        ("Ackley", ackley_function, vec![-32.768, 32.768], 2),
        ("Schwefel", schwefel_function, vec![-500.0, 500.0], 1),
    ];

    let mut results: Vec<BenchmarkResult> = Vec::new();

    // Iterate over the combinations of functions
    for (param, alpha_values, gamma_values, diff_values, beta_values) in parameter_sets.iter() {
        // Here, iterate over the parameter sets
        for (func_name, func, range_values, dim) in benchmark_functions.iter() {
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
                        Some(*dim),
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
    let max_iterations: usize = 500;

    // Define the functions with their search-space parameters
    let benchmark_functions: Vec<(&str, fn(&ndarray::Array1<f64>) -> f64, Vec<f64>, usize)> = vec![
        ("Sphere", sphere_function, vec![-5.12, 5.12], 1),
        ("Rosenbrock", rosenbrock_function, vec![-5.0, 10.0], 2),
        ("Rastrigin", rastrigin_function, vec![-5.12, 5.12], 1),
        ("Ackley", ackley_function, vec![-32.768, 32.768], 2),
        ("Schwefel", schwefel_function, vec![-500.0, 500.0], 1),
        ("HappyCat", happy_cat_function, vec![-20.0, 20.0], 1),
        ("DropWave", drop_wave_function, vec![-5.12, 5.12], 2),
        ("Salomon", salomon_function, vec![-100.0, 100.0], 1),
        ("SchafferF7", schaffer_f7_function, vec![-100.0, 100.0], 2),
        ("Weierstrass", weierstrass_function, vec![-0.5, 0.5], 1),
        ("XinSheYang1", xin_she_yang_1_function, vec![-10.0, 10.0], 1),
        ("Booth", booth_function, vec![-10.0, 10.0], 2),
        ("Beale", beale_function, vec![-4.5, 4.5], 2),
        ("Griewank", griewank_function, vec![-600.0, 600.0], 1),
        (
            "ExpandedSchafferF6",
            expanded_schaffer_f6_function,
            vec![-100.0, 100.0],
            2,
        ),
    ];
    // Create the vecto for the results
    let mut results: Vec<BenchmarkResult> = Vec::new();

    // Then, iterate over it to test the SDAO
    for (func_name, func, range_values, dim) in benchmark_functions.iter() {
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
            Some(*dim),
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

/// Test the other algorithms agaisnt the SDAO.
///
/// This would test the FLA, GD, SA against the SDAO algorithm to see how they perform
/// with the same functions.
pub fn compare_algorithms_benchmark() {
    // General variables
    let max_iterations = 500;
    let repetitions = 1000; // No. of repetitions to make with each test

    let benchmark_functions: Vec<(&str, fn(&ndarray::Array1<f64>) -> f64, Vec<f64>)> = vec![
        ("Sphere", sphere_function, vec![-5.12, 5.12]),
        ("Rosenbrock", rosenbrock_function, vec![-5.0, 10.0]),
        ("Rastrigin", rastrigin_function, vec![-5.12, 5.12]),
        ("Ackley", ackley_function, vec![-32.768, 32.768]),
        ("Schwefel", schwefel_function, vec![-500.0, 500.0]),
        (
            "Rastrigin with noise",
            rastrigin_noisy_function,
            vec![-5.12, 5.12],
        ),
    ];
    // Initialize the system info object
    let mut system = System::new_all();

    // * SDAO
    // Define the default values for the SDAO.
    let alpha_value = 0.01;
    let gamma_value = 0.5;
    let diff_value = 1.0;
    let beta_value = 0.01;
    let num_particles_value = 100;

    // Run the SDAO for all the functions
    println!("Running the SDAO algorithm for the functions...");
    print!("");
    for (func_name, func, range_values) in benchmark_functions.iter() {
        let mut results: Vec<(f64, u64, f64)> = Vec::new();
        // Run the experiments
        for _ in 0..repetitions {
            // Record initial memory usage
            system.refresh_memory();
            let initial_memory = system.used_memory();

            // Set the optimizer
            let mut optimizer = SDAO::new(
                num_particles_value,
                1,
                range_values.clone(),
                alpha_value,
                gamma_value,
                diff_value,
                beta_value,
                max_iterations,
                *func,
                // Optional ones
                None,
                None,
                None,
            );
            // Using the optimizer, run it and get the result
            let result = optimizer.run();

            // Record final memory usage
            system.refresh_memory();
            let final_memory = system.used_memory();
            let memory_used = final_memory.saturating_sub(initial_memory) / 1000; // Memory used during the algorithm (in KB)

            let _iterations = optimizer.used_iterations;
            let time = optimizer.execution_time;
            if let Some(best_particle) = result.1 {
                results.push((best_particle.best_value, memory_used, time));
            }
        }
        // Get the mean values and print them
        calculate_mean_values(func_name, results);
    }

    // * FLA
    println!("=====================================");
    println!("Running the FLA algorithm for the functions...");
    print!("");
    for (func_name, func, range_values) in benchmark_functions.iter() {
        let mut results: Vec<(f64, u64, f64)> = Vec::new();
        // Run the experiments
        for _ in 0..repetitions {
            // Record initial memory usage
            system.refresh_memory();
            let initial_memory = system.used_memory();

            let (best_value, _, time) = fick_law_algorithm(
                num_particles_value,
                max_iterations,
                diff_value,
                range_values.clone(),
                *func,
            );
            // Record final memory usage
            system.refresh_memory();
            let final_memory = system.used_memory();
            let memory_used = final_memory.saturating_sub(initial_memory) / 1000; // Memory used during the algorithm (in KB)
                                                                                  // At the end, push the results
            results.push((best_value, memory_used, time));
        }
        // Get the mean values and print them
        calculate_mean_values(func_name, results);
    }
    // For FLA, I'll use the same number of particles and the same diff value.
    // * SA
    // I'll use the temperature and cooling rate inside the SA implementation
    println!("=====================================");
    println!("Running the SA algorithm for the functions...");
    print!("");
    for (func_name, func, range_values) in benchmark_functions.iter() {
        let mut results: Vec<(f64, u64, f64)> = Vec::new();
        // Run the experiments
        for _ in 0..repetitions {
            // Record initial memory usage
            system.refresh_memory();
            let initial_memory = system.used_memory();

            let (best_value, _, time) = simulated_annealing(range_values.clone(), *func);
            // Record final memory usage
            system.refresh_memory();
            let final_memory = system.used_memory();
            let memory_used = final_memory.saturating_sub(initial_memory) / 1000; // Memory used during the algorithm (in KB)

            // At the end, push the results
            results.push((best_value, memory_used, time));
        }
        // Get the mean values and print them
        calculate_mean_values(func_name, results);
    }
    // * GD
    println!("=====================================");
    println!("Running the GD algorithm for the functions...");
    print!("");
    for (func_name, func, range_values) in benchmark_functions.iter() {
        let mut results: Vec<(f64, u64, f64)> = Vec::new();
        // Run the experiments
        for _ in 0..repetitions {
            // Record initial memory usage
            system.refresh_memory();
            let initial_memory = system.used_memory();

            let (best_value, _, time) =
                gradient_descent(range_values.clone(), alpha_value, max_iterations, *func);
            // Record final memory usage
            system.refresh_memory();
            let final_memory = system.used_memory();
            let memory_used = final_memory.saturating_sub(initial_memory) / 1000; // Memory used during the algorithm (in KB)

            // At the end, push the results
            results.push((best_value, memory_used, time));
        }
        // Get the mean values and print them
        calculate_mean_values(func_name, results);
    }
    // * PSO
    println!("=====================================");
    println!("Running the PSO algorithm for the functions...");
    print!("");
    for (func_name, func, range_values) in benchmark_functions.iter() {
        let mut results: Vec<(f64, u64, f64)> = Vec::new();
        // Run the experiments
        for _ in 0..repetitions {
            // Record initial memory usage
            system.refresh_memory();
            let initial_memory = system.used_memory();

            let (best_value, time) = pso(
                range_values.clone(),
                *func,
                num_particles_value,
                1,
                max_iterations,
            );
            // Record final memory usage
            system.refresh_memory();
            let final_memory = system.used_memory();
            let memory_used = final_memory.saturating_sub(initial_memory) / 1000; // Memory used during the algorithm (in KB)

            // At the end, push the results
            results.push((best_value, memory_used, time));
        }
        // Get the mean values and print them
        calculate_mean_values(func_name, results);
    }
}

fn calculate_mean_values(func_name: &&str, values: Vec<(f64, u64, f64)>) {
    // Init the variables to sum the positions
    let mut sum_best_value = 0.0;
    let mut sum_memory = 0;
    let mut sum_time = 0.0;

    // Iterate over the values to sum them
    for (best_value, memory, time) in &values {
        sum_best_value += best_value;
        sum_memory += *memory;
        sum_time += time;
    }
    let len_values = values.len() as f64;
    // Get the average
    let avg_best_value = sum_best_value / len_values;
    let avg_memory = (sum_memory as f64 / len_values).round() as usize;
    let avg_time = sum_time / len_values;
    // Get the info for the BoxPlot
    let mut best_values: Vec<f64> = values
        .iter()
        .map(|(best_value, _, _)| *best_value)
        .collect();
    best_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let first_quarter = best_values[(len_values * 0.25).floor() as usize];
    let second_quarter = best_values[(len_values * 0.75).floor() as usize];
    // Get the lower and upper bounds
    let lower_bound = best_values[0];
    let upper_bound = best_values[(len_values - 1.0) as usize];

    // Print the valued
    println!(
        "Function: '{}'. Best value: |LB={}, FQ={},M={},SQ={}, UB={}|. Memory (in KB): {}. Time (ms): {}",
        func_name,
        (lower_bound + 1.0).log10(),
        (first_quarter + 1.0).log10(),
        (avg_best_value.abs() + 1.0).log10(), // I perform a log10 +1 to normalize the data,
        (second_quarter + 1.0).log10(),
        (upper_bound + 1.0).log10(),
        avg_memory,
        avg_time * 1000.0 // To return the time in ms. 1 s = 1000 ms
    );
}
