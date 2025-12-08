//! Core types for regression analysis.

mod options;
mod prediction;
mod result;

pub use options::{
    LambdaScaling, OptionsError, RegressionOptions, RegressionOptionsBuilder, SolverType,
};
pub use prediction::{IntervalType, PredictionResult};
pub use result::RegressionResult;
