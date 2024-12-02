/// PSO algorithm implementation
///
/// This algorithm (Particle Swarm Optimization) is a
/// population-based optimization algorithm that is
/// inspired by the social behavior of birds.
///
/// DOI: https://doi.org/10.1109/4235.985692
use ndarray::Array1;
use rand::prelude::*;
use std::f64;

// Recommendedn parameters
const W: f64 = 0.5; // Inercia
const C1: f64 = 1.5; // Coeficiente cognitivo
const C2: f64 = 1.5; // Coeficiente social

#[derive(Debug, Clone)]
struct Particle {
    position: Array1<f64>,
    velocity: Vec<f64>,
    best_position: Array1<f64>,
    best_value: f64,
}

pub fn pso(
    search_space: Vec<f64>,
    objective_fn: fn(&Array1<f64>) -> f64,
    num_particles: usize,
    dimension: usize,
    iterations: usize,
) -> (f64, f64) {
    // Initialize the random generator and the vector of particles
    let mut rng = thread_rng();
    let mut particles = Vec::with_capacity(num_particles);

    // Initialize each particle using the provided range of positions
    for _ in 0..num_particles {
        let position: Array1<f64> = (0..dimension)
            .map(|_| rng.gen_range(search_space[0]..search_space[1]))
            .collect();
        let velocity: Vec<f64> = (0..dimension).map(|_| rng.gen_range(-1.0..1.0)).collect();

        let value = objective_fn(&position);

        particles.push(Particle {
            position: position.clone(),
            velocity,
            best_position: position,
            best_value: value,
        });
    }

    // Find the best initial particle
    let (best_particle, _) = particles
        .iter()
        .enumerate()
        .min_by(|(_, p1), (_, p2)| p1.best_value.partial_cmp(&p2.best_value).unwrap())
        .unwrap();
    // From here, get the best position
    let mut global_best_position = particles[best_particle].position.clone();
    let mut global_best_value = particles[best_particle].best_value;

    // Start to calculate the time
    let start_time = std::time::Instant::now();
    // Iterate over the particles and the dimensions to search iteratively the best solution
    for _i in 0..iterations {
        // Check each particle
        for particle in &mut particles {
            for i in 0..dimension {
                // Update the velocity. For this, use two random parameters
                let r1: f64 = rng.gen();
                let r2: f64 = rng.gen();

                // Get the new velocity
                particle.velocity[i] = W * particle.velocity[i]
                    + C1 * r1 * (particle.best_position[i] - particle.position[i])
                    + C2 * r2 * (global_best_position[i] - particle.position[i]);
                // Update the position
                particle.position[i] += particle.velocity[i];
            }

            // Evaluate the new position obtianed
            let current_value = objective_fn(&particle.position);

            // Update agains the best personal value for this particle.
            // If this solution has better results, then just update the new value
            // and new position.
            if current_value < particle.best_value {
                particle.best_value = current_value;
                particle.best_position = particle.position.clone();
            }

            // And then, check this against the global best particle.
            if current_value < global_best_value {
                global_best_value = current_value;
                global_best_position = particle.position.clone();
            }
        }
        // println!(
        //     "Iteration: {} :: Best value: {} :: Best Position {}",
        //     i, global_best_value, global_best_position
        // );
    }
    // Get the total time
    let elapsed_time = start_time.elapsed().as_secs_f64();
    // At the end, return the best value obtained
    (global_best_value, elapsed_time)
}
