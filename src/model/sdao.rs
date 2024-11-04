// Module imports
// Imports from external crates
use ndarray::Array1;
use rand::distributions::{Distribution, Uniform};
use rand_distr::Normal;
use rayon::prelude::*;
// Local imports
use super::particle::Particle;
use crate::enums::Status;

// * Equivalent for the user input vs the algorithm parameters
// learning_rate = alpha
// memory_coeff = gamma
// decay_rate = beta

/// Create a SDAO instance with the provided parameters.
///
/// **Arguments**:
///     - num_particles: Number of particles to use.
///     - dimension: Dimension of the search space.
///     - learning_rate: Learning rate of the algorithm.
///     - memory_coeff: Memory coefficient of the algorithm.
///     - diffusion_coeff: Diffusion coefficient of the algorithm.
///     - decay_rate: Decay rate of the algorithm.
///     - max_iterations: Maximum number of iterations.
///     - objective_fn: Objective function to optimize.
///     - threshold: Threshold to consider the algorithm has converged. Default to 1e-6.
///     - num_threads: Number of threads to use. Default to 1.
///     - verbose: Whether to print information during the run. Default to false.
#[derive(Debug)]
pub struct SDAO {
    particles: Vec<Particle>,
    alpha: f64,
    gamma: f64,
    diffusion_coeff: f64,
    beta: f64,
    max_iterations: usize,
    objective_fn: fn(&Array1<f64>) -> f64,
    // num_threads: usize,
    threshold: f64,
    verbose: bool,
}

impl SDAO {
    /// Create a SDAO instance with the provided parameters.
    ///
    /// Args:
    ///     - num_particles: Number of particles to use.
    ///     - dimension: Dimension of the search space.
    ///     - learning_rate: Learning rate of the algorithm.
    ///     - memory_coeff: Memory coefficient of the algorithm.
    ///     - diffusion_coeff: Diffusion coefficient of the algorithm.
    ///     - decay_rate: Decay rate of the algorithm.
    ///     - max_iterations: Maximum number of iterations.
    ///     - objective_fn: Objective function to optimize.
    ///     - threshold: Threshold to consider the algorithm has converged. Default to 1e-6.
    ///     - num_threads: Number of threads to use. Default to 1.
    ///     - verbose: Whether to print information during the run. Default to false.
    pub fn new(
        num_particles: usize,
        dimension: usize,
        learning_rate: f64,
        memory_coeff: f64,
        diffusion_coeff: f64,
        decay_rate: f64,
        max_iterations: usize,
        objective_fn: fn(&Array1<f64>) -> f64,
        // Optional parameters
        threshold: Option<f64>,
        num_threads: Option<usize>,
        verbose: Option<bool>,
    ) -> Self {
        // Get the optional parameters. In this case, the verbose flag and the number of threads
        let verbose: bool = verbose.unwrap_or(false);
        let num_threads = num_threads.unwrap_or(1);
        let threshold = threshold.unwrap_or(1e-6);

        // * Configure Rayon to utiliz the specified thread numbers. By default is 1
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build_global()
            .unwrap();

        // Initialize the random number generator
        let mut rng = rand::thread_rng();
        let uniform = Uniform::new(-10.0, 10.0); // Search range

        // Initialize the number of particles randomly. Depending on which particles we do want to use
        // we'll create a new particle with a random position and a value of 0.
        let particles = (0..num_particles)
            .map(|_| {
                // Get the position of the particle in the search space. This position
                // is going to be a random place in the search space.
                let position = Array1::from_iter((0..dimension).map(|_| uniform.sample(&mut rng)));
                let value = 50.0; // The value is 0.0 since it's the one we want.
                Particle {
                    position: position.clone(), // This clone is necessary to avoid borrowing the position
                    best_position: position,
                    best_value: value,
                }
            })
            .collect();

        // And then, initialize the SDAO struct with the particles, the learning rate, the memory coefficient,
        SDAO {
            particles,
            alpha: learning_rate,
            gamma: memory_coeff,
            diffusion_coeff,
            beta: decay_rate,
            max_iterations,
            objective_fn,
            threshold,
            verbose,
        }
    }

    /// Run the SDAO algorithm.
    ///
    /// Returns:
    ///     - Status: Enum indicating the result of the run.
    ///     - Option<Particle>: The best particle found if any.
    pub fn run(&mut self) -> (Status, Option<Particle>) {
        if self.verbose {
            println!("Starting the optimization...");
            println!("* =================================== *");
            println!("Number of particles: {}", self.particles.len());
            println!("Number of dimensions: {}", self.particles[0].position.len());
            println!("Learning rate: {}", self.alpha);
            println!("Memory coefficient: {}", self.gamma);
            println!("Diffusion coefficient: {}", self.diffusion_coeff);
            println!("Decay rate: {}", self.beta);
            println!("Maximum number of iterations: {}", self.max_iterations);
            println!("* =================================== *");
        }
        // Initialize the flag for the optimal and the feasible solution
        let mut optimal_found = false;
        let mut feasible_solution: Option<Particle> = None;

        for k in 0..self.max_iterations {
            // Process each particle in parallel
            // Calculate the gradients before the parallel iteration
            let gradients: Vec<Array1<f64>> = self
                .particles
                .iter()
                .map(|particle| self.calculate_gradient(&particle.position))
                .collect();

            // Get the length of the position
            let position_len = self.particles[0].position.len();

            // Iterate over the particles in parallel to obtain the new positions
            self.particles
                .par_iter_mut()
                .zip(gradients.into_par_iter())
                .for_each(|(particle, gradient)| {
                    // Evaluate the objective function in the current position
                    let current_value = (self.objective_fn)(&particle.position);

                    // Update the personal best position if necessary
                    if current_value < particle.best_value {
                        particle.best_value = current_value;
                        particle.best_position = particle.position.clone();
                    }

                    // Generate the stochastic perturbation
                    let mut rng = rand::thread_rng();
                    let normal_dist = Normal::new(0.0, 1.0).unwrap();
                    let eta: Array1<f64> =
                        Array1::from_shape_fn(position_len, |_| normal_dist.sample(&mut rng));

                    // Update the particle position
                    particle.position = &particle.position - &(self.alpha * &gradient)
                        + &(self.gamma * (&particle.best_position - &particle.position))
                        + &((self.diffusion_coeff * 2.0).sqrt() * eta);
                });

            // Adapt the diffusion coefficient
            self.diffusion_coeff *= (-self.beta * k as f64).exp();

            // Verify if we have reach the optimal
            if self.check_optimality() {
                optimal_found = true;
                feasible_solution = Some(self.get_best_particle());
                if self.verbose {
                    println!("Optimal found in iteration {}", k + 1);
                }
                break;
            }

            // Log the model information if the verbose flag is set
            if self.verbose && k % 10 == 0 {
                if let Some(best) = self.get_best_particle_opt() {
                    println!("[Iter={},T=]: {}", k + 1, best.best_value);
                }
            }
        }

        if optimal_found {
            (Status::Optimal, feasible_solution)
        } else {
            // Determine if a feasible solution was found
            let best = self.get_best_particle_opt();
            if let Some(best_particle) = best {
                (Status::Feasible, Some(best_particle.clone()))
            } else {
                (Status::Infeasible, None)
            }
        }
    }

    /// Check if the optimum has been found (depends on the definition of the optimum).
    fn check_optimality(&self) -> bool {
        // This method is to evaluate if any particles had a value below the threshold.
        // If it has, then we can say that the optimal has been found.
        self.particles.iter().any(|p| p.best_value < self.threshold)
    }

    /// Obtiene la mejor partícula encontrada
    /// Obtain the best particle found.
    /// This is done by getting the particle with the lowest value.
    fn get_best_particle(&self) -> Particle {
        self.particles
            .iter()
            .min_by(|a, b| a.best_value.partial_cmp(&b.best_value).unwrap())
            .unwrap()
            .clone()
    }

    /// Obtain the best particle found as an Option.
    fn get_best_particle_opt(&self) -> Option<&Particle> {
        self.particles
            .iter()
            .min_by(|a, b| a.best_value.partial_cmp(&b.best_value).unwrap())
    }

    // Calculate the gradient of the objective function using finite differences
    fn calculate_gradient(&self, position: &Array1<f64>) -> Array1<f64> {
        // Define the finite step
        let h = 1e-8;
        let mut gradient = Array1::<f64>::zeros(position.len());

        for i in 0..position.len() {
            // Get the previous and next position reference. Then, increment and decrement the
            // positions by the finite step.
            let mut pos_plus = position.clone();
            let mut pos_minus = position.clone();
            pos_plus[i] += h;
            pos_minus[i] -= h;
            // Evaluate the objective function in the new positions
            let f_plus = (self.objective_fn)(&pos_plus);
            let f_minus = (self.objective_fn)(&pos_minus);
            // Calculate the gradient using the finite differences
            gradient[i] = (f_plus - f_minus) / (2.0 * h);
        }
        gradient
    }
}
