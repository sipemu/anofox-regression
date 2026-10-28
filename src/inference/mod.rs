//! Statistical inference (standard errors, p-values, confidence intervals).

pub mod anova;
mod coefficient;
mod predict_se;
mod prediction;
mod robust_covariance;
mod variance_factor;
mod wald;
pub(crate) use variance_factor::expand_reduced_factor;
pub(crate) use wald::{fill_wald_inference, WaldReference};

pub use anova::{
    anova, drop_intercept, wald_chunk, wald_test, AnovaOptions, AnovaRow, AnovaRowKind, AnovaTable,
    WaldDistribution, WaldError, WaldTest,
};
pub use coefficient::CoefficientInference;
pub(crate) use predict_se::{glm_prediction, quadratic_se};
pub use predict_se::{PredictSeError, PredictSeOptions, PredictWithSe, PredictionSe};
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
pub use variance_factor::{
    compute_ridge_variance_factor, compute_variance_factor, intervals_from_variance_factor,
    leverage_new,
};
