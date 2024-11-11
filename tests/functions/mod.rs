/// Distinct functions to test the algorithm behavior.
///
use ndarray::Array1;

/// Sphere Function: f(x) = sum(x_i^2)
///
/// **Arguments**:
///     - x: A reference to an Array1<f64> representing the position vector.
///
/// **Returns**:
///     - `f64`: The value of the sphere function at the given position.
pub fn sphere_function(x: &Array1<f64>) -> f64 {
    x.iter().map(|xi| xi.powi(2)).sum()
}

/// Rosenbrock Function, using a=1 and b=100:
///     f(x) = sum_{i=1}^{d-1} [100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2]
///
/// **Arguments**:
///     -x: A reference to an Array1<f64> representing the position vector.
///
/// **Returns**:
///     - `f64``: The value of the Rosenbrock function at the given position.
pub fn rosenbrock_function(x: &Array1<f64>) -> f64 {
    x.windows(2)
        .into_iter()
        .map(|w| 100.0 * (w[1] - w[0].powi(2)).powi(2) + (1.0 - w[0]).powi(2))
        .sum()
}

/// Rastrigin Function: f(x) = 10d + sum_{i=1}^{d} [x_i^2 - 10*cos(2*pi*x_i)]
///
/// **Arguments**:
///     - x: A reference to an Array1<f64> representing the position vector.
///
/// **Returns**:
///     - `f64`: The value of the Rastrigin function at the given position.
pub fn rastrigin_function(x: &Array1<f64>) -> f64 {
    let d = x.len() as f64;
    10.0 * d
        + x.iter()
            .map(|xi| xi.powi(2) - 10.0 * (2.0 * std::f64::consts::PI * xi).cos())
            .sum::<f64>()
}

/// Ackley Function:
/// f(x) = -20*exp(-0.2*sqrt(1/d * sum(x_i^2))) - exp(1/d * sum(cos(2*pi*x_i))) + 20 + e
///
/// **Arguments**:
///     - x: A reference to an Array1<f64> representing the position vector.
///
/// **Returns**:
///     - `f64`: The value of the Ackley function at the given position.
pub fn ackley_function(x: &Array1<f64>) -> f64 {
    let d = x.len() as f64;
    let sum_sq = x.iter().map(|xi| xi.powi(2)).sum::<f64>();
    let sum_cos = x
        .iter()
        .map(|xi| (2.0 * std::f64::consts::PI * xi).cos())
        .sum::<f64>();

    -20.0 * (-0.2 * (sum_sq / d).sqrt()).exp() - (sum_cos / d).exp() + 20.0 + std::f64::consts::E
}

/// Schwefel Function: f(x) = 418.9829*d - sum_{i=1}^{d} [x_i * sin(sqrt(|x_i|))]
///
/// **Arguments**:
///     - x: A reference to an Array1<f64> representing the position vector.
///
/// **Returns**:
///     - f64`:The value of the Schwefel function at the given position.
pub fn schwefel_function(x: &Array1<f64>) -> f64 {
    let d = x.len() as f64;
    let sum = x.iter().map(|xi| xi * (xi.abs().sqrt()).sin()).sum::<f64>();
    418.9829 * d - sum
}
