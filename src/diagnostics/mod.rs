//! Regression diagnostics (leverage, Cook's distance, VIF, etc.).
//!
//! This module provides tools for diagnosing regression models:
//!
//! - **Leverage**: Identifies observations with unusual predictor values
//! - **Residuals**: Standardized and studentized residuals for outlier detection
//! - **GLM Residuals**: Pearson, deviance, and working residuals for GLMs
//! - **Influence**: Cook's distance and DFFITS for influential point detection
//! - **VIF**: Variance Inflation Factor for multicollinearity detection
//! - **Quasi-separation**: Detection of separation issues in GLMs
//! - **Condition Number**: Matrix condition and collinearity diagnostics
//!
//! # Pre-fitting Diagnostics
//!
//! These diagnostics help identify potential issues before fitting a model:
//!
//! ```rust
//! # use faer::{Col, Mat};
//! # let x = Mat::from_fn(50, 2, |i, j| if j == 0 { i as f64 / 10.0 } else { ((i * 7) % 11) as f64 / 10.0 });
//! # let y = Col::from_fn(50, |i| if (i * 7) % 3 == 0 { 1.0 } else { 0.0 });
//! use anofox_regression::diagnostics::{
//!     check_binary_separation, condition_diagnostic, variance_inflation_factor,
//! };
//!
//! // Check matrix conditioning
//! let cond = condition_diagnostic(&x, true);
//! if let Some(warning) = &cond.warning {
//!     eprintln!("Warning: {}", warning);
//! }
//!
//! // Check for separation in logistic regression
//! let sep = check_binary_separation(&x, &y);
//! if sep.has_separation {
//!     eprintln!("Warning: {}", sep.warning_message.unwrap());
//! }
//!
//! // Check for multicollinearity
//! let vif = variance_inflation_factor(&x);
//! # let _ = vif;
//! ```
//!
//! # Post-fitting Diagnostics
//!
//! These diagnostics help assess model fit and identify problematic observations:
//!
//! ```rust
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! # use faer::{Col, Mat};
//! # let x = Mat::from_fn(50, 2, |i, j| if j == 0 { i as f64 / 10.0 } else { ((i * 7) % 11) as f64 / 10.0 });
//! # let y = Col::from_fn(50, |i| 1.0 + 2.0 * x[(i, 0)] - x[(i, 1)] + ((i * 13) % 7) as f64 / 20.0);
//! # use anofox_regression::solvers::{FittedRegressor, OlsRegressor, Regressor};
//! use anofox_regression::diagnostics::{
//!     compute_leverage, cooks_distance, high_leverage_points, influential_cooks,
//! };
//!
//! // After fitting a model
//! let fit = OlsRegressor::builder().build().fit(&x, &y)?;
//! let (residuals, mse, n_params) = (&fit.result().residuals, fit.result().mse, 3);
//! let leverage = compute_leverage(&x, true);
//! let cooks = cooks_distance(residuals, &leverage, mse, n_params);
//!
//! // Identify problematic observations
//! let high_leverage = high_leverage_points(&leverage, n_params, None);
//! let influential = influential_cooks(&cooks, None);
//! # let _ = (high_leverage, influential);
//! # Ok(())
//! # }
//! ```

pub mod accuracy;
pub(crate) mod augment;
mod condition_number;
mod glm_residuals;
mod heteroskedasticity;
mod influence;
mod leverage;
mod partial_residuals;
mod quasi_separation;
mod residuals;
mod vif;

// Re-export main functions
pub use augment::{
    augment, augment_glm, augment_glm_with, augment_with, press, Augment, AugmentSpec, Augmentable,
    Press, ResidualType,
};
pub use condition_number::{
    classify_condition_number, condition_diagnostic, condition_number,
    variance_decomposition_proportions, ConditionDiagnostic, ConditionSeverity,
};
pub use glm_residuals::{
    deviance_residuals, estimate_dispersion_deviance, estimate_dispersion_pearson,
    pearson_chi_squared, pearson_residuals, response_residuals, standardized_deviance_residuals,
    standardized_pearson_residuals, working_residuals,
};
pub use heteroskedasticity::{breusch_pagan, white_test, HeteroskedasticityTest};
pub use influence::{cooks_distance, dffits, influential_cooks, influential_dffits};
pub use leverage::{compute_leverage, compute_leverage_with_aliased, high_leverage_points};
pub use partial_residuals::{
    added_variable, partial_residuals, partial_residuals_glm, AddedVariable, PartialResiduals,
};
pub use quasi_separation::{
    check_binary_separation, check_count_sparsity, detect_separation, LinearSeparation,
    SeparationCheck, SeparationType,
};
pub use residuals::{
    externally_studentized_residuals, residual_outliers, standardized_residuals,
    studentized_residuals,
};
pub use vif::{generalized_vif, high_vif_predictors, variance_inflation_factor};
