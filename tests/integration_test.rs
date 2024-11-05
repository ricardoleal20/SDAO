mod functions;
// External crate imports
use ndarray::Array1;
// Local imports
use functions::{
    ackley_function, rastrigin_function, rosenbrock_function, schwefel_function, sphere_function,
};
use sdao::{Status, SDAO};

fn create_sdao(objective_func: fn(&Array1<f64>) -> f64, dimensions: usize) -> SDAO {
    SDAO::new(
        50,
        dimensions,
        [-10.0, 10.0].to_vec(),
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

/// SDAO initializer test. This test ensures that the SDAO can be initialized
/// with the given parameters and that the particles are correctly created.
#[test]
fn test_init_sdao() {
    // Just init the SDAO. Use the sphere function as a test
    create_sdao(sphere_function, 1);
}

/// Simple test to ensure that the SDAO is working and is able
/// to find the optimal solution for the sphere function.
#[test]
fn test_sdao_optimal_sphere() {
    let mut optimizer = create_sdao(sphere_function, 1);
    // Get the best particle and the status of the optimization
    let (status, best_particle) = optimizer.run();
    assert!(matches!(status, Status::Optimal | Status::Feasible));
    // Check if the value for the best particle fits the expected threshold
    if let Some(particle) = best_particle {
        assert!(particle.best_value < 1e-6);
    }
}

/// Simple test to ensure that the SDAO can solve the rosenbrock function
#[test]
fn test_sdao_optimal_rosenbrock() {
    //let mut optimizer = create_sdao(rosenbrock_function, 2);
    let mut optimizer = SDAO::new(
        50,
        1,
        [-500.0, 500.0].to_vec(),
        0.01,
        0.5,
        1.0,
        0.01,
        500,
        rosenbrock_function,
        None,
        None,
        None,
    );
    // Get the best particle and the status of the optimization
    let (status, best_particle) = optimizer.run();
    assert!(matches!(status, Status::Optimal | Status::Feasible));
    println!("Status: {:?}", status);
    // Check if the value for the best particle fits the expected threshold
    if let Some(particle) = best_particle {
        assert!(particle.best_value < 1e-6);
    }
}

/// Simple test to ensure that the SDAO can solve the rastrigin function
#[test]
fn test_sdao_optimal_rastrigin() {
    //let mut optimizer = create_sdao(rastrigin_function, 1);
    let mut optimizer = SDAO::new(
        50,
        1,
        [-5.12, 5.12].to_vec(),
        0.01,
        0.5,
        1.0,
        0.01,
        500,
        rastrigin_function,
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

/// Simple test to ensure that the SDAO can solve the rastrigin function
#[test]
fn test_sdao_optimal_ackley() {
    // let mut optimizer = create_sdao(ackley_function, 1);
    let mut optimizer = SDAO::new(
        50,
        1,
        [-10.0, 10.0].to_vec(),
        0.01,
        0.5,
        1.0,
        0.01,
        500,
        ackley_function,
        None,
        None,
        None,
    );
    // Get the best particle and the status of the optimization
    let (status, best_particle) = optimizer.run();
    assert!(matches!(status, Status::Optimal | Status::Feasible));
    println!("Status: {:?}", status);
    // Check if the value for the best particle fits the expected threshold
    if let Some(particle) = best_particle {
        assert!(particle.best_value < 1e-3);
    }
}

/// Simple test to ensure that the SDAO can solve the schwefel function
#[test]
fn test_sdao_optimal_schwefel() {
    let mut optimizer = SDAO::new(
        50,
        1,
        [-500.0, 500.0].to_vec(),
        0.01,
        0.5,
        1.0,
        0.01,
        500,
        schwefel_function,
        None,
        None,
        None,
    );
    // Get the best particle and the status of the optimization
    let (status, best_particle) = optimizer.run();
    assert!(matches!(status, Status::Optimal | Status::Feasible));
    println!("Status: {:?}", status);
    // Check if the value for the best particle fits the expected threshold
    if let Some(particle) = best_particle {
        assert!(particle.best_value < 1e-6);
    }
}
