/// SA: Simulated Annealing
///
/// This algorithm is well know for his simple implementation and impressive results.
/// It is based on the annealing process of metals, where the temperature is reduced
/// to obtain a solid state. In this case, the temperature is used to accept or reject
/// new solutions in the optimization process.
///
/// DOI: https://doi.org/10.1126/science.220.4598.671
///
use ndarray::Array1;
use rand::Rng;

/// Simulated Annealing implementation
///
/// This function implements the Simulated Annealing algorithm to solve optimization problems.
///
/// Arguments:
///    - `search_space`: Vec<f64> - The search space to use in the algorithm
///    - `objective_fn`: fn(&Array1<f64>) -> f64 - The objective function to optimize
///
/// Returns:
///   - The best value obtained by the algorithm
///  - The iterations used by the algorithm
pub fn simulated_annealing(
    search_space: Vec<f64>,
    objective_fn: fn(&Array1<f64>) -> f64,
) -> (f64, f64, f64) {
    let mut rng = rand::thread_rng();

    // Initialize the current solution and the best solution
    let mut current_solution = rng.gen_range(search_space[0]..search_space[1]);
    let mut best_solution = current_solution;
    let mut current_energy = objective_fn(&Array1::from(vec![current_solution]));
    let mut best_energy = current_energy;
    // Get the temperature and cooling rate
    let mut temperature = 1000.0;
    let cooling_rate = 0.003;

    // Start the optimization process
    let mut iterations = 0;
    let start_time = std::time::Instant::now();
    while temperature > 1.0 {
        // Generate a neighborhood solution
        let neighbor = current_solution + rng.gen_range(-1.0..1.0);
        // Get the energy of the neighbor solution
        let neighbor_energy = objective_fn(&Array1::from(vec![neighbor]));
        // Get the difference in energy
        let delta_energy = neighbor_energy - current_energy;
        // Decide if we take (or not) the new solution
        if delta_energy < 0.0 || rng.gen::<f64>() < (-delta_energy / temperature).exp() {
            current_solution = neighbor;
            current_energy = neighbor_energy;
        }
        // Update the best solution found
        if current_energy < best_energy {
            best_solution = current_solution;
            best_energy = current_energy;
        }
        // Cool the temperature
        temperature *= 1.0 - cooling_rate;
        // Increase the iterations
        iterations += 1;
    }

    // Get the total time
    let elapsed_time = start_time.elapsed().as_secs_f64();

    // Return the best solution
    (
        objective_fn(&Array1::from(vec![best_solution])),
        iterations as f64,
        elapsed_time,
    )
}
