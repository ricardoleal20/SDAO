mod functions;
// Use imports
use functions::sphere_function;
use sdao::{Status, SDAO};

/// Simple test to ensure that the SDAO is working and is able
/// to find the optimal solution for the sphere function.
#[test]
fn test_sdao_optimal_sphere() {
    let mut optimizer = SDAO::new(
        50,
        2,
        0.01,
        0.5,
        1.0,
        0.01,
        1000,
        sphere_function,
        None,
        None,
        None,
    );

    // Get the best particle and the status of the optimization
    let (status, best_particle) = optimizer.run();
    assert!(matches!(status, Status::Optimal | Status::Feasible));

    // Check if the value for the best particle fits the expected threshold
    if let Some(particle) = best_particle {
        assert!(particle.best_value < 1e-6);
    }
}
