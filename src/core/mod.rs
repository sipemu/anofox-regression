//! Core types for regression analysis.

mod binomial;
mod family;
mod link;
mod model_info;
mod na_action;
mod negative_binomial;
mod options;
mod poisson;
mod poisson_link;
mod prediction;
mod result;
mod validate;

pub use binomial::BinomialFamily;
pub(crate) use family::glm_null_deviance;
pub use family::{GlmFamily, TweedieFamily};
pub use link::BinomialLink;
pub use model_info::{HasModelInfo, ModelInfo};
pub(crate) use na_action::apply_na_action;
pub use na_action::{NaAction, NaError, NaHandler, NaInfo, NaResult};
pub(crate) use negative_binomial::distinct_values as negative_binomial_distinct_values;
pub use negative_binomial::{estimate_theta_ml, estimate_theta_moments, NegativeBinomialFamily};
pub use options::{
    LambdaScaling, OptionsError, RegressionOptions, RegressionOptionsBuilder, SolverType,
};
pub use poisson::PoissonFamily;
pub use poisson_link::PoissonLink;
pub use prediction::{IntervalType, PredictionResult, PredictionType};
pub use result::RegressionResult;
pub(crate) use validate::{validate_finite, validate_finite_extra, validate_xy};
