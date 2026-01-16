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
//! ```rust,ignore
//! use anofox_regression::diagnostics::{
//!     condition_diagnostic, check_binary_separation, variance_inflation_factor
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
//! ```
//!
//! # Post-fitting Diagnostics
//!
//! These diagnostics help assess model fit and identify problematic observations:
//!
//! ```rust,ignore
//! use anofox_regression::diagnostics::{compute_leverage, cooks_distance, pearson_residuals};
//!
//! // After fitting a model
//! let leverage = compute_leverage(&x, true);
//! let cooks = cooks_distance(&residuals, &leverage, mse, n_params);
//!
//! // Identify problematic observations
//! let high_leverage = high_leverage_points(&leverage, n_params, None);
//! let influential = influential_cooks(&cooks, None);
//! ```

mod condition_number;
mod glm_residuals;
mod influence;
mod leverage;
mod quasi_separation;
mod residuals;
mod vif;

// Re-export main functions
pub use condition_number::{
    classify_condition_number, condition_diagnostic, condition_number,
    variance_decomposition_proportions, ConditionDiagnostic, ConditionSeverity,
};
pub use glm_residuals::{
    deviance_residuals, estimate_dispersion_deviance, estimate_dispersion_pearson,
    pearson_chi_squared, pearson_residuals, response_residuals, standardized_deviance_residuals,
    standardized_pearson_residuals, working_residuals,
};
pub use influence::{cooks_distance, dffits, influential_cooks, influential_dffits};
pub use leverage::{compute_leverage, compute_leverage_with_aliased, high_leverage_points};
pub use quasi_separation::{
    check_binary_separation, check_count_sparsity, SeparationCheck, SeparationType,
};
pub use residuals::{
    externally_studentized_residuals, residual_outliers, standardized_residuals,
    studentized_residuals,
};
pub use vif::{generalized_vif, high_vif_predictors, variance_inflation_factor};
