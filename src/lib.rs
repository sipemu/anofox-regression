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
        NaHandler, NaInfo, NegativeBinomialFamily, PoissonFamily, PoissonLink, PredictionResult,
        PredictionType, RegressionOptions, RegressionOptionsBuilder, RegressionResult,
        TweedieFamily,
    };
    pub use crate::diagnostics::{
        check_binary_separation, check_count_sparsity, classify_condition_number, compute_leverage,
        condition_diagnostic, condition_number, cooks_distance, deviance_residuals,
        high_leverage_points, influential_cooks, pearson_residuals, standardized_residuals,
        studentized_residuals, variance_decomposition_proportions, variance_inflation_factor,
        working_residuals, ConditionDiagnostic, ConditionSeverity, SeparationCheck, SeparationType,
    };
    pub use crate::solvers::{
        AlmDistribution, AlmRegressor, BinomialRegressor, BlsRegressor, ElasticNetRegressor,
        FittedAlm, FittedBinomial, FittedIsotonic, FittedNegativeBinomial, FittedPls,
        FittedPoisson, FittedQuantile, FittedRegressor, FittedTweedie, IsotonicRegressor,
        LinkFunction, NegativeBinomialRegressor, OlsRegressor, OutOfBounds, PlsRegressor,
        PoissonRegressor, QuantileRegressor, Regressor, RidgeRegressor, RlsRegressor,
        TweedieRegressor, WlsRegressor,
    };
}

pub use crate::core::{
    BinomialFamily, BinomialLink, GlmFamily, IntervalType, LambdaScaling, NaAction, NaError,
    NaHandler, NaInfo, NegativeBinomialFamily, PoissonFamily, PoissonLink, PredictionResult,
    PredictionType, RegressionOptions, RegressionOptionsBuilder, RegressionResult, TweedieFamily,
};
pub use crate::solvers::{
    BinomialRegressor, FittedBinomial, FittedIsotonic, FittedNegativeBinomial, FittedPoisson,
    FittedQuantile, FittedRegressor, IsotonicRegressor, NegativeBinomialRegressor, OutOfBounds,
    PoissonRegressor, QuantileRegressor, Regressor,
};
