//! This module defines the enums to use in the library.

/// Represents the status of a execution
#[derive(Debug)]
pub enum Status {
    /// Indicates that the run has found an optimal solution.
    Optimal,
    /// Indicates that the run has found a feasible solution.
    Feasible,
    /// Indicates that the run has found an infeasible solution.
    Infeasible,
}
