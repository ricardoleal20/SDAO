/// GD: Gradient Descent
///
/// Gradient Descent is an optimization algorithm used
/// to minimize some function by iteratively moving in
/// the direction of steepest descent as defined by the
/// negative of the gradient.
///
/// DOI: https://doi.org/10.48550/arXiv.1609.04747
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

/// Gradient Descent implementation
///
/// This function implements the Gradient Descent algorithm to solve optimization problems.
///
/// Arguments:
///    - `initial_position`: f64 - The initial position to start the optimization process
///    
pub fn gradient_descent(
    search_space: Vec<f64>,
    learning_rate: f64,
    max_iterations: usize,
    objective_fn: fn(&Array1<f64>) -> f64,
) -> (f64, f64, f64) {
    // Init the random number generator
    let mut rng = rand::thread_rng();
    // Get the initial position for x
    let mut x_position = rng.gen_range(search_space[0]..search_space[1]);
    // Init the other parameters such as the learning rate
    let precision = 0.00001;
    let mut iterations = 0;

    // Start the optimization process and the timing
    let start_time = std::time::Instant::now();
    loop {
        let grad = calculate_gradient(objective_fn, x_position);
        let x_next = x_position - learning_rate * grad;

        let step_size = (x_next - x_position).abs();

        x_position = x_next;
        iterations += 1;

        if step_size < precision || iterations >= max_iterations {
            break;
        }
    }
    // Return the minimum obtained and the number of iterations
    (
        objective_fn(&Array1::from(vec![x_position])),
        iterations as f64,
        start_time.elapsed().as_secs_f64(),
    )
}
