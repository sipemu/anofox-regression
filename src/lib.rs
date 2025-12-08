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
        BinomialFamily, BinomialLink, GlmFamily, IntervalType, LambdaScaling, NaAction, NaError,
        NaHandler, NaInfo, PredictionResult, PredictionType, RegressionOptions,
        RegressionOptionsBuilder, RegressionResult, TweedieFamily,
    };
    pub use crate::diagnostics::{
        compute_leverage, cooks_distance, deviance_residuals, high_leverage_points,
        influential_cooks, pearson_residuals, standardized_residuals, studentized_residuals,
        variance_inflation_factor, working_residuals,
    };
    pub use crate::solvers::{
        BinomialRegressor, BlsRegressor, ElasticNetRegressor, FittedBinomial, FittedRegressor,
        FittedTweedie, OlsRegressor, Regressor, RidgeRegressor, RlsRegressor, TweedieRegressor,
        WlsRegressor,
    };
}

pub use crate::core::{
    BinomialFamily, BinomialLink, GlmFamily, IntervalType, LambdaScaling, NaAction, NaError,
    NaHandler, NaInfo, PredictionResult, PredictionType, RegressionOptions,
    RegressionOptionsBuilder, RegressionResult, TweedieFamily,
};
pub use crate::solvers::{BinomialRegressor, FittedBinomial, FittedRegressor, Regressor};
