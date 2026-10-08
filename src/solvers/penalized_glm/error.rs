//! Error type of the penalized GLM engine and the AFT model.

use thiserror::Error;

/// Errors reported by the penalized GLM engine ([`super`]) and the AFT survival
/// model ([`crate::solvers::aft`]).
///
/// The variants separate *input* problems (invalid values, dimension mismatches,
/// too little data) from *data-dependent numerical breakdowns*
/// ([`GlmEngineError::NumericalFailure`], [`GlmEngineError::SingularMatrix`]), so a
/// caller such as a SQL front end can report the former as an error and the latter
/// as a degenerate (NULL) result.
#[derive(Error, Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum GlmEngineError {
    /// Too few usable rows for the number of parameters.
    #[error("Insufficient data: {rows} rows, {cols} features (need rows > features)")]
    InsufficientData {
        /// Usable rows.
        rows: usize,
        /// Number of parameters / features.
        cols: usize,
    },

    /// Too few usable rows (free-form message).
    #[error("Insufficient data: {0}")]
    InsufficientDataMsg(String),

    /// Every row was dropped because of NULL/NaN values.
    #[error("All rows filtered due to NULL/NaN values")]
    NoValidData,

    /// Lengths of the response and the design disagree.
    #[error("Dimension mismatch: y has {y_len} elements, X has {x_rows} rows")]
    DimensionMismatch {
        /// Length of the response.
        y_len: usize,
        /// Rows of the design.
        x_rows: usize,
    },

    /// A required input is empty.
    #[error("Empty input: {field} cannot be empty")]
    EmptyInput {
        /// Name of the empty input.
        field: &'static str,
    },

    /// Invalid input (free-form message).
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// An option or data value outside its domain.
    #[error("Invalid value for {field}: {message}")]
    InvalidValue {
        /// Name of the offending option or input.
        field: &'static str,
        /// What is wrong with it.
        message: String,
    },

    /// A matrix that must be inverted is singular or near-singular.
    #[error("Matrix is singular or near-singular")]
    SingularMatrix,

    /// A data-dependent numerical breakdown (non-finite linear predictor or
    /// likelihood, etc.).
    #[error("Numerical failure: {0}")]
    NumericalFailure(String),
}

/// Result alias for the penalized GLM engine.
pub type GlmEngineResult<T> = Result<T, GlmEngineError>;

impl From<GlmEngineError> for crate::solvers::RegressionError {
    fn from(e: GlmEngineError) -> Self {
        use crate::solvers::RegressionError;
        match e {
            GlmEngineError::SingularMatrix => RegressionError::SingularMatrix,
            GlmEngineError::DimensionMismatch { y_len, x_rows } => {
                RegressionError::DimensionMismatch { x_rows, y_len }
            }
            GlmEngineError::InsufficientData { rows, cols } => {
                RegressionError::InsufficientObservations {
                    needed: cols + 1,
                    got: rows,
                }
            }
            other => RegressionError::NumericalError(other.to_string()),
        }
    }
}
