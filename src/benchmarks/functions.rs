/// Distinct functions to test the algorithm behavior.
///
/// REFERENCE: https://www.mdpi.com/2306-5729/7/4/46
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
    let mut sum = 0.0;
    for i in 0..x.len() {
        sum += (1.0 - x[i]).powi(2) + 100.0 * (x[i].powi(2) - 1.0).powi(2);
    }
    sum
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

/// Drop-Wave Function: f(x) = - (1 + cos(12*sqrt(x^2 + y^2))) / (0.5*(x^2 + y^2) + 2)
/// where x and y are the two dimensions of the position vector.
///
/// **Arguments**:
///    - x: A reference to an Array1<f64> representing the position vector.
///
/// **Returns**:
///   - `f64`: The value of the Drop-Wave function at the given position.
///
/// **Note**:
/// - The Drop-Wave function is a multimodal function with a global minimum at -1.
/// - The function is defined for x and y in [-5.12, 5.12].
pub fn drop_wave_function(x: &Array1<f64>) -> f64 {
    let x0 = x[0];
    let y = x[1];
    let numerator = 1.0 + (12.0 * (x0.powi(2) + y.powi(2)).sqrt()).cos();
    let denominator = 0.5 * (x0.powi(2) + y.powi(2)) + 2.0;
    1.0 - (numerator / denominator)
}

/// Booth Function: f(x, y) = (x + 2y - 7)^2 + (2x + y - 5)^2
/// where x and y are the two dimensions of the position vector.
///
/// **Arguments**:
///   - x: A reference to an Array1<f64> representing the position vector.
///
/// **Returns**:
///  - `f64`: The value of the Booth function at the given position.
///
/// **Note**:
/// - The Booth function is a multimodal function with a global minimum at f(1, 3) = 0.
/// - The function is defined for x and y in [-10, 10].
pub fn booth_function(x: &Array1<f64>) -> f64 {
    let x0 = x[0];
    let y = x[1];
    (x0 + 2.0 * y - 7.0).powi(2) + (2.0 * x0 + y - 5.0).powi(2)
}

/// Beale Function: f(x, y) = (1.5 - x + xy)^2 + (2.25 - x + xy^2)^2 + (2.625 - x + xy^3)^2
/// where x and y are the two dimensions of the position vector.
///
/// **Arguments**:
///  - x: A reference to an Array1<f64> representing the position vector.
///
/// **Returns**:
/// - `f64`: The value of the Beale function at the given position.
///
/// **Note**:
/// - The Beale function is a multimodal function with a global minimum at f(3, 0.5) = 0.
/// - The function is defined for x and y in [-4.5, 4.5].
pub fn beale_function(x: &Array1<f64>) -> f64 {
    let x0 = x[0];
    let y = x[1];
    (1.5 - x0 + x0 * y).powi(2)
        + (2.25 - x0 + x0 * y.powi(2)).powi(2)
        + (2.625 - x0 + x0 * y.powi(3)).powi(2)
}

/// Weierstrass Function: f(x) = sum_{i=1}^{d} sum_{k=0}^{20} [a^k * cos(2*pi*b^k*(x_i + 0.5))] - d*sum_{k=0}^{20} a^k * cos(pi*b^k)
/// where a = 0.5 and b = 3.
///
/// **Arguments**:
///     - x: A reference to an Array1<f64> representing the position vector.
///
/// **Returns**:
///    - `f64`: The value of the Weierstrass function at the given position.
///
/// **Note**:
/// - The Weierstrass function is a multimodal function with a global minimum at 0.
/// - The function is defined for x_i in [-0.5, 0.5] for all i in [1, d].
pub fn weierstrass_function(x: &Array1<f64>) -> f64 {
    let a: f64 = 0.5;
    let b: f64 = 3.0;
    let d = x.len() as f64;
    let mut sum = 0.0;
    for xi in x.iter() {
        for k in 0..21 {
            sum += a.powi(k) * (2.0 * std::f64::consts::PI * b.powi(k) * (xi + 0.5)).cos();
        }
    }
    // Get the dimension sum coponent
    let dim_sum = (0..21)
        .map(|k| a.powi(k) * (std::f64::consts::PI * b.powi(k)).cos())
        .sum::<f64>();
    sum - d * dim_sum
}

/// Griewank’s Function: f(x) = sum_{i=1}^{d} [x_i^2 / 4000] - prod_{i=1}^{d} cos(x_i / sqrt(i)) + 1
///
/// **Arguments**:
///   - x: A reference to an Array1<f64> representing the position vector.
///
/// **Returns**:
///  - `f64`: The value of the Griewank’s function at the given position.
///
/// **Note**:
/// - The Griewank’s function is a multimodal function with a global minimum at 0.
/// - The function is defined for x_i in [-600, 600] for all i in [1, d].
pub fn griewank_function(x: &Array1<f64>) -> f64 {
    let sum_sq = x.iter().map(|xi| xi.powi(2) / 4000.0).sum::<f64>();
    let prod_cos = x
        .iter()
        .enumerate()
        .map(|(i, xi)| (xi / ((i + 1) as f64).sqrt()).cos())
        .product::<f64>();
    sum_sq - prod_cos + 1.0
}

/// HappyCat Function: f(x) = (|x^2 - 4|^0.25 + 0.5*(x^2 - 4) + 0.5) + sum_{i=1}^{d} [1/(8*i) * (x_i^2 - 1)^2]
///
/// **Arguments**:
///  - x: A reference to an Array1<f64> representing the position vector.
///
/// **Returns**:
/// - `f64`: The value of the HappyCat function at the given position.
///
/// **Note**:
/// - The HappyCat function is a multimodal function with a global minimum at 0.
/// - The function is defined for x_i in [-2, 2] for all i in [1, d].
pub fn happy_cat_function(x: &Array1<f64>) -> f64 {
    let sum_sq = x.iter().map(|xi| xi.powi(2)).sum::<f64>();
    let sum = x
        .iter()
        .enumerate()
        .map(|(i, xi)| (1.0 / (8.0 * (i + 1) as f64)) * (xi.powi(2) - 1.0).powi(2))
        .sum::<f64>();
    (sum_sq - 4.0).abs().powf(0.25) + 0.5 * (sum_sq - 4.0) + 0.5 + sum
}

/// Schaffer’s F7 Function: f(x, y) = 0.5 + (sin(sqrt(x^2 + y^2))^2 - 0.5) / (1 + 0.001*(x^2 + y^2))^2
/// where x and y are the two dimensions of the position vector.
///
/// **Arguments**:
///     - x: A reference to an Array1<f64> representing the position vector.
///
/// **Returns**:
///    - `f64`: The value of the Schaffer’s F7 function at the given position.
///
/// **Note**:
/// - The Schaffer’s F7 function is a multimodal function with a global minimum at f(0, 0) = 0.
/// - The function is defined for x and y in [-100, 100].
pub fn schaffer_f7_function(x: &Array1<f64>) -> f64 {
    let x0 = x[0];
    let y = x[1];
    let sq_sum = x0.powi(2) + y.powi(2);
    0.5 + (x0.sin().powi(2) - 0.5) / (1.0 + 0.001 * sq_sum).powi(2)
}

/// Expanded Schaffer’s F6 Function: f(x, y) = 0.5 + (sin(sqrt(x^2 + y^2))^2 - 0.5) / (1 + 0.001*(x^2 + y^2))^2
/// where x and y are the two dimensions of the position vector.
///
/// **Arguments**:
///    - x: A reference to an Array1<f64> representing the position vector.
///
/// **Returns**:
///   - `f64`: The value of the Expanded Schaffer’s F6 function at the given position.
///
/// **Note**:
/// - The Expanded Schaffer’s F6 function is a multimodal function with a global minimum at f(0, 0) = 0.
/// - The function is defined for x and y in [-100, 100].
pub fn expanded_schaffer_f6_function(x: &Array1<f64>) -> f64 {
    let x0 = x[0];
    let y = x[1];
    let sq_sum = x0.powi(2) + y.powi(2);
    0.5 + (x0.sin().powi(2) - 0.5) / (1.0 + 0.001 * sq_sum).powi(2)
}

/// Xin-She Yang’s 1 Function: f(x) = sum_{i=1}^{d} |x_i|^i
/// where d is the number of dimensions of the position vector.
///
/// **Arguments**:
///   - x: A reference to an Array1<f64> representing the position vector.
///
/// **Returns**:
///  - `f64`: The value of the Xin-She Yang’s 1 function at the given position.
///
/// **Note**:
/// - The Xin-She Yang’s 1 function is a multimodal function with a global minimum at 0.
/// - The function is defined for x_i in [-5, 5] for all i in [1, d].
pub fn xin_she_yang_1_function(x: &Array1<f64>) -> f64 {
    x.iter()
        .enumerate()
        .map(|(i, xi)| xi.abs().powi(i as i32 + 1))
        .sum()
}

/// Salomon Function: f(x) = 1 - cos(2*pi*sqrt(sum_{i=1}^{d} x_i^2)) + 0.1*sqrt(sum_{i=1}^{d} x_i^2)
/// where d is the number of dimensions of the position vector.
///
/// **Arguments**:
///     - x: A reference to an Array1<f64> representing the position vector.
///
/// **Returns**:
///     - `f64`: The value of the Salomon function at the given position.
///
/// **Note**:
/// - The Salomon function is a multimodal function with a global minimum at 0.
/// - The function is defined for x_i in [-100, 100] for all i in [1, d].
pub fn salomon_function(x: &Array1<f64>) -> f64 {
    let sum_sq = x.iter().map(|xi| xi.powi(2)).sum::<f64>();
    1.0 - (2.0 * std::f64::consts::PI * sum_sq.sqrt()).cos() + 0.1 * sum_sq.sqrt()
}
