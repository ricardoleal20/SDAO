/// FLA: Fick's Law Algorithm
///
/// Algorithm based on the Fick's Law, developed by Hashim and company.
/// This algorithm uses a simple view of Fick's 2nd law to optimize
/// the diffusion of particle to reduce the gradint of concentration, looking
/// for its minimum
///
/// DOI: https://doi.org/10.1016/j.knosys.2022.110146
///
use ndarray::Array1;
use rand::Rng;

// Calculate the gradient of the objective function using finite differences
fn calculate_gradient(objective_fn: fn(&Array1<f64>) -> f64, position: f64) -> f64 {
    // Define the finite step
    let h = 1e-8;
    // Get the previous and next position reference. Then, increment and decrement the
    // the position of the particle by the finite step.
    let mut pos_plus = position.clone();
    let mut pos_minus = position.clone();
    pos_plus += h;
    pos_minus -= h;
    // Evaluate the objective function in the new positions
    let f_plus = (objective_fn)(&Array1::from(vec![pos_plus]));
    let f_minus = (objective_fn)(&Array1::from(vec![pos_minus]));
    // Return the gradient calculation
    (f_plus - f_minus) / (2.0 * h)
}

/// FLA implementation
///
/// This function implements the Fick's Law Algorithm (FLA) to solve optimization problems.
///
/// Arguments:
///     - `pop_size`: usize - The population size to use in the algorithm
///     - `max_iter`: usize - The maximum number of iterations to run the algorithm
///     - `diffusion_coeff`: f64 - The diffusion coefficient to use in the algorithm
///     - `search_space`: (f64, f64) - The search space to use in the algorithm
///
/// Returns:
///     - The best value obtained by the algorithm
///     - The iterations used by the algorithm
pub fn fick_law_algorithm(
    pop_size: usize,
    max_iter: usize,
    diffusion_coeff: f64,
    search_space: Vec<f64>,
    objective_fn: fn(&Array1<f64>) -> f64,
) -> (f64, f64, f64) {
    // Init the random number generator
    let mut rng = rand::thread_rng();

    // Initialize the poblation with random values ranging the search space
    let mut population: Vec<f64> = (0..pop_size)
        .map(|_| rng.gen_range(search_space[0]..search_space[1]))
        .collect();

    let mut iterations: usize = 0;
    let start_time = std::time::Instant::now();

    // Start the optimization process. Iterate over the iterations
    for _ in 0..max_iter {
        for i in 0..pop_size {
            let x = population[i];
            let grad = calculate_gradient(objective_fn, x);
            let movement = -diffusion_coeff * grad; // Movement according to Fick's law
            population[i] += movement;
        }
        iterations += 1;
    }
    // Get the total time
    let elapsed_time = start_time.elapsed().as_secs_f64();

    // Get the best value of the population
    let best_value = population
        .into_iter()
        .reduce(|a, b| {
            if objective_fn(&Array1::from(vec![a])) < objective_fn(&Array1::from(vec![b])) {
                a
            } else {
                b
            }
        })
        .unwrap();

    (
        objective_fn(&Array1::from(vec![best_value])),
        iterations as f64,
        elapsed_time,
    )
}
