//! Core types for regression analysis.

mod binomial;
mod family;
mod link;
mod na_action;
mod options;
mod prediction;
mod result;

pub use binomial::BinomialFamily;
pub use family::{GlmFamily, TweedieFamily};
pub use link::BinomialLink;
pub use na_action::{NaAction, NaError, NaHandler, NaInfo, NaResult};
pub use options::{
    LambdaScaling, OptionsError, RegressionOptions, RegressionOptionsBuilder, SolverType,
};
pub use prediction::{IntervalType, PredictionResult, PredictionType};
pub use result::RegressionResult;
