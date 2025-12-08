//! Regression diagnostics (leverage, Cook's distance, VIF, etc.).
//!
//! This module provides tools for diagnosing regression models:
//!
//! - **Leverage**: Identifies observations with unusual predictor values
//! - **Residuals**: Standardized and studentized residuals for outlier detection
//! - **GLM Residuals**: Pearson, deviance, and working residuals for GLMs
//! - **Influence**: Cook's distance and DFFITS for influential point detection
//! - **VIF**: Variance Inflation Factor for multicollinearity detection
//!
//! # Example
//!
//! ```rust,ignore
//! use statistics::diagnostics::{compute_leverage, cooks_distance, variance_inflation_factor};
//!
//! // After fitting a model
//! let leverage = compute_leverage(&x, true);
//! let cooks = cooks_distance(&residuals, &leverage, mse, n_params);
//! let vif = variance_inflation_factor(&x);
//!
//! // Identify problematic observations
//! let high_leverage = high_leverage_points(&leverage, n_params, None);
//! let influential = influential_cooks(&cooks, None);
//! let collinear = high_vif_predictors(&vif, 5.0);
//! ```

mod glm_residuals;
mod influence;
mod leverage;
mod residuals;
mod vif;

// Re-export main functions
pub use glm_residuals::{
    deviance_residuals, estimate_dispersion_deviance, estimate_dispersion_pearson,
    pearson_chi_squared, pearson_residuals, response_residuals, standardized_deviance_residuals,
    standardized_pearson_residuals, working_residuals,
};
pub use influence::{cooks_distance, dffits, influential_cooks, influential_dffits};
pub use leverage::{compute_leverage, high_leverage_points};
pub use residuals::{
    externally_studentized_residuals, residual_outliers, standardized_residuals,
    studentized_residuals,
};
pub use vif::{generalized_vif, high_vif_predictors, variance_inflation_factor};
