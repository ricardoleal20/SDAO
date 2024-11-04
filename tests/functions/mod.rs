/// Distinct functions to test the algorithm behavior.
///
use ndarray::Array1;

/// Sphere Function: f(x) = sum(x_i^2)
///
/// **Arguments**:
///     - `x`: A reference to an Array1<f64> representing the position vector.
///
/// **Returns**:
///     - `f64`: The value of the sphere function at the given position.
pub fn sphere_function(x: &Array1<f64>) -> f64 {
    x.iter().map(|xi| xi.powi(2)).sum()
}
