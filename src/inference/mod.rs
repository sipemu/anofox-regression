//! Statistical inference (standard errors, p-values, confidence intervals).

mod coefficient;
mod prediction;

pub use coefficient::CoefficientInference;
pub use prediction::{
    compute_prediction_intervals, compute_xtwx_inverse_augmented, compute_xtx_inverse,
    compute_xtx_inverse_augmented,
};
