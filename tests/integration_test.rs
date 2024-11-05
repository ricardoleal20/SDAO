mod functions;
// External crate imports
use ndarray::Array1;
// Local imports
use functions::{
    ackley_function, rastrigin_function, rosenbrock_function, schwefel_function, sphere_function,
};
use sdao::{Status, SDAO};

fn create_sdao(objective_func: fn(&Array1<f64>) -> f64, search_space: Vec<f64>) -> SDAO {
    SDAO::new(
        100,
        1,
        search_space,
        0.01,
        0.5,
        1.0,
        0.01,
        500,
        objective_func,
        None,
        None,
        None,
    )
}

fn run_sdao(mut sdao: SDAO, expected_value: f64) -> bool {
    // Instance the success variable
    let mut success = false;

    for _ in 0..5 {
        let (status, best_particle) = sdao.run();
        if matches!(status, Status::Optimal | Status::Feasible) {
            if let Some(particle) = best_particle {
                if particle.best_value < expected_value {
                    success = true;
                    break;
                }
            }
        }
    }
    // Return the success execution
    success
}

/// ============================================ ///
///                    TESTS                     ///
/// ============================================ ///

/// SDAO initializer test. This test ensures that the SDAO can be initialized
/// with the given parameters and that the particles are correctly created.
#[test]
fn test_init_sdao() {
    // Just init the SDAO. Use the sphere function as a test
    create_sdao(sphere_function, vec![-10.0, 10.0]);
}

/// Simple test to ensure that the SDAO is working and is able
/// to find the optimal solution for the sphere function.
#[test]
fn test_sdao_optimal_sphere() {
    let optimizer = create_sdao(sphere_function, vec![-10.0, 10.0]);
    // Evaluate if the algorithm run successfully or not
    assert!(
        run_sdao(optimizer, 1e-6),
        "Failed to find an optimal solution within 5 retries for the Sphere function"
    );
}

/// Simple test to ensure that the SDAO can solve the rosenbrock function
#[test]
fn test_sdao_optimal_rosenbrock() {
    let optimizer = create_sdao(rosenbrock_function, vec![-500.0, 500.0]);
    // Evaluate if the algorithm run successfully or not
    assert!(
        run_sdao(optimizer, 1e-6),
        "Failed to find an optimal solution within 5 retries for the Rosenbrock function"
    );
}

/// Simple test to ensure that the SDAO can solve the rastrigin function
#[test]
fn test_sdao_optimal_rastrigin() {
    let optimizer = create_sdao(rastrigin_function, vec![-5.12, 5.12]);
    // Evaluate if the algorithm run successfully or not
    assert!(
        run_sdao(optimizer, 1e-6),
        "Failed to find an optimal solution within 5 retries for the Rastrigin function"
    );
}

/// Simple test to ensure that the SDAO can solve the rastrigin function
#[test]
fn test_sdao_optimal_ackley() {
    let optimizer = create_sdao(ackley_function, vec![-10.0, 10.0]);
    // Evaluate if the algorithm run successfully or not
    assert!(
        run_sdao(optimizer, 1e-2),
        "Failed to find an optimal solution within 5 retries for the Ackley function"
    );
}

/// Simple test to ensure that the SDAO can solve the schwefel function
#[test]
fn test_sdao_optimal_schwefel() {
    let optimizer = create_sdao(schwefel_function, vec![-500.0, 500.0]);
    // Evaluate if the algorithm run successfully or not
    assert!(
        run_sdao(optimizer, 1e-6),
        "Failed to find an optimal solution within 5 retries for the Schwefel function"
    );
}
