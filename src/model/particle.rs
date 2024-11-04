use ndarray::Array1;

#[derive(Clone, Debug)]
pub struct Particle {
    pub position: Array1<f64>,
    pub best_position: Array1<f64>,
    pub best_value: f64,
}
