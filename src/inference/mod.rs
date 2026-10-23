//! Statistical inference (standard errors, p-values, confidence intervals).

mod anova_table;
mod coefficient;
mod nested_f;
mod predict_se;
mod prediction;
mod robust_covariance;
mod variance_factor;
mod wald;
pub(crate) use variance_factor::expand_reduced_factor;
pub(crate) use wald::{fill_wald_inference, WaldReference};

pub use anova_table::{
    anova_table, AnovaTable, AnovaTableError, AnovaTableOptions, AnovaTerm, AnovaType,
};
pub use coefficient::CoefficientInference;
pub use nested_f::{
    nested_f_test, nested_f_test_indices, NestedFTest, NestedFTestError, NestedFTestOptions,
};
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
