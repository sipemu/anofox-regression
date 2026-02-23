//! Statistical inference (standard errors, p-values, confidence intervals).

mod coefficient;
mod prediction;
mod robust_covariance;

pub use coefficient::CoefficientInference;
pub use prediction::{
    compute_prediction_intervals, compute_xtwx_inverse_augmented,
    compute_xtwx_inverse_augmented_reduced, compute_xtwx_inverse_reduced, compute_xtx_inverse,
    compute_xtx_inverse_augmented, compute_xtx_inverse_augmented_reduced,
    compute_xtx_inverse_reduced,
};
pub use robust_covariance::{
    compute_hc_inference, compute_hc_standard_errors, HcInference, HcInterceptInference, HcResult,
    HcType,
};
