/// Other algorithms to test and benchmark them.
///
/// This algorithms work with different approaches to solve optimization problems.
/// and it would be great to compare them with the SDAO development.
// Import the modules here
mod fla;
mod gd;
mod pso;
mod sa;
// Import their algorithms here
pub use fla::fick_law_algorithm;
pub use gd::gradient_descent;
pub use pso::pso;
pub use sa::simulated_annealing;
