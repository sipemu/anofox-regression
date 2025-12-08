//! A robust statistics library for regression analysis.
//!
//! This library provides sklearn-style regression estimators with full statistical
//! inference support including standard errors, t-statistics, p-values, and
//! confidence intervals.
//!
//! # Example
//!
//! ```rust,ignore
//! use statistics::prelude::*;
//!
//! // Create and fit an OLS model
//! let fitted = OlsRegressor::builder()
//!     .with_intercept(true)
//!     .confidence_level(0.95)
//!     .build()
//!     .fit(&x, &y)?;
//!
//! // Make predictions
//! let predictions = fitted.predict(&x_new);
//!
//! // Access statistics
//! let stats = fitted.result();
//! println!("R² = {}", stats.r_squared);
//! ```

pub mod core;
pub mod diagnostics;
pub mod distributions;
pub mod inference;
pub mod solvers;
pub mod utils;

/// Prelude module for convenient imports.
pub mod prelude {
    pub use crate::core::{
        IntervalType, LambdaScaling, PredictionResult, RegressionOptions, RegressionOptionsBuilder,
        RegressionResult,
    };
    pub use crate::diagnostics::{
        compute_leverage, cooks_distance, high_leverage_points, influential_cooks,
        standardized_residuals, studentized_residuals, variance_inflation_factor,
    };
    pub use crate::solvers::{
        ElasticNetRegressor, FittedRegressor, OlsRegressor, Regressor, RidgeRegressor,
        RlsRegressor, WlsRegressor,
    };
}

pub use crate::core::{
    IntervalType, LambdaScaling, PredictionResult, RegressionOptions, RegressionOptionsBuilder,
    RegressionResult,
};
pub use crate::solvers::{FittedRegressor, Regressor};
