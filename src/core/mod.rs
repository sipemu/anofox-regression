//! Core types for regression analysis.

mod binomial;
mod family;
mod link;
mod na_action;
mod negative_binomial;
mod options;
mod poisson;
mod poisson_link;
mod prediction;
mod result;

pub use binomial::BinomialFamily;
pub use family::{GlmFamily, TweedieFamily};
pub use link::BinomialLink;
pub use na_action::{NaAction, NaError, NaHandler, NaInfo, NaResult};
pub use negative_binomial::{estimate_theta_ml, estimate_theta_moments, NegativeBinomialFamily};
pub use options::{
    LambdaScaling, OptionsError, RegressionOptions, RegressionOptionsBuilder, SolverType,
};
pub use poisson::PoissonFamily;
pub use poisson_link::PoissonLink;
pub use prediction::{IntervalType, PredictionResult, PredictionType};
pub use result::RegressionResult;
