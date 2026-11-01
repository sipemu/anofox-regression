//! A robust statistics library for regression analysis, validated against R
//! and scikit-learn.
//!
//! Estimators follow a builder / `fit` / fitted-model pattern with full
//! statistical inference: standard errors, t or z statistics, p-values,
//! confidence and prediction intervals. Most are checked against their R
//! equivalent (`lm`, `glm`, `glmnet`, `MASS`, `quantreg`, `lme4`, `nlme`,
//! `survival`, `rms`, `Hmisc`, `greybox`, …); see `validation/VALIDATION.md`
//! in the repository.
//!
//! # Example
//!
//! ```rust
//! use anofox_regression::prelude::*;
//! use faer::{Col, Mat};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // y = 1 + 2·x₁ − 0.5·x₂ plus a little deterministic noise
//! let x = Mat::from_fn(30, 2, |i, j| if j == 0 { i as f64 } else { (i as f64 * 0.7).sin() });
//! let y = Col::from_fn(30, |i| {
//!     1.0 + 2.0 * x[(i, 0)] - 0.5 * x[(i, 1)] + 0.1 * ((i * 7) % 5) as f64
//! });
//!
//! let fitted = OlsRegressor::builder()
//!     .with_intercept(true)
//!     .compute_inference(true)
//!     .confidence_level(0.95)
//!     .build()
//!     .fit(&x, &y)?;
//!
//! let result = fitted.result();
//! assert!((result.coefficients[0] - 2.0).abs() < 0.01);
//! assert!(result.r_squared > 0.99);
//! let se = result.std_errors.as_ref().expect("inference was requested");
//! assert!(se[0] > 0.0);
//!
//! // Predictions with 95 % prediction intervals
//! let x_new = Mat::from_fn(2, 2, |i, j| if j == 0 { 31.0 + i as f64 } else { 0.0 });
//! let pred = fitted.predict_with_interval(&x_new, Some(IntervalType::Prediction), 0.95);
//! assert!(pred.lower[0] < pred.fit[0] && pred.fit[0] < pred.upper[0]);
//! # Ok(())
//! # }
//! ```
#![forbid(unsafe_code)]
#![deny(missing_docs)]

/// The linear-algebra crate whose types (`Mat`, `Col`, …) the API uses,
/// re-exported so callers depend on the matching version (see
/// `STABILITY.md`: a faer upgrade that changes these types is a major
/// release).
pub use faer;

pub mod core;
pub mod design;
pub mod diagnostics;
pub mod distributions;
pub mod effects;
pub mod imputation;
pub mod inference;
pub mod reduction;
pub mod solvers;
pub mod splines;
pub mod survival;
pub mod transform;
pub mod utils;
pub mod validation;

/// Prelude module for convenient imports.
pub mod prelude {
    pub use crate::core::{
        BinomialFamily, BinomialLink, GlmFamily, HasModelInfo, IntervalType, LambdaScaling,
        ModelInfo, NaAction, NaError, NaHandler, NaInfo, NegativeBinomialFamily, PoissonFamily,
        PoissonLink, PredictionResult, PredictionType, RegressionOptions, RegressionOptionsBuilder,
        RegressionResult, SolverType, TweedieFamily,
    };
    pub use crate::diagnostics::{
        augment, augment_with, check_binary_separation, check_count_sparsity,
        classify_condition_number, compute_leverage, condition_diagnostic, condition_number,
        cooks_distance, detect_separation, deviance_residuals, high_leverage_points,
        influential_cooks, pearson_residuals, standardized_residuals, studentized_residuals,
        variance_decomposition_proportions, variance_inflation_factor, working_residuals, Augment,
        Augmentable, ConditionDiagnostic, ConditionSeverity, LinearSeparation, ResidualType,
        SeparationCheck, SeparationType,
    };
    pub use crate::inference::{
        anova_table, nested_f_test, nested_f_test_indices, AnovaTable, AnovaTableError,
        AnovaTableOptions, AnovaTerm, AnovaType, HcInference, HcType, NestedFTest,
        NestedFTestError, NestedFTestOptions, PredictSeOptions, PredictWithSe,
    };
    pub use crate::solvers::aft::{AftDistribution, AftRegressor, FittedAft};
    pub use crate::solvers::eb_shrink::{eb_shrink, EbShrinkOptions, EbShrinkResult, TauMethod};
    pub use crate::solvers::penalized_glm::{
        FittedPenalizedGlm, PenalizedGlmFamily, PenalizedGlmRegressor, PriorSpec, VcovType,
    };
    pub use crate::solvers::{
        AlmDistribution, AlmRegressor, BinomialRegressor, BlsRegressor, ElasticNetRegressor,
        FactorSummary, FittedAlm, FittedBinomial, FittedGamma, FittedGlmm, FittedHuber,
        FittedIsotonic, FittedLogistic, FittedNegativeBinomial, FittedPSpline, FittedPls,
        FittedPoisson, FittedQuantile, FittedRegressor, FittedTweedie, GammaRegressor, GlmmDdf,
        GlmmInference, GlmmRegressor, GlmmRegressorBuilder, HuberRegressor, IsotonicRegressor,
        LinkFunction, LogisticRegression, NegativeBinomialRegressor, OlsRegressor, OutOfBounds,
        PSplineRegressor, Penalty, PlsRegressor, PoissonRegressor, QuantileRegressor, Regressor,
        RidgeRegressor, RlsRegressor, TweedieRegressor, WlsRegressor,
    };
}

pub use crate::core::{
    BinomialFamily, BinomialLink, GlmFamily, HasModelInfo, IntervalType, LambdaScaling, ModelInfo,
    NaAction, NaError, NaHandler, NaInfo, NegativeBinomialFamily, PoissonFamily, PoissonLink,
    PredictionResult, PredictionType, RegressionOptions, RegressionOptionsBuilder,
    RegressionResult, SolverType, TweedieFamily,
};
pub use crate::inference::{
    anova_table, nested_f_test, nested_f_test_indices, AnovaTable, AnovaTableError,
    AnovaTableOptions, AnovaTerm, AnovaType, HcInference, HcType, NestedFTest, NestedFTestError,
    NestedFTestOptions,
};
pub use crate::solvers::{
    BinomialRegressor, FactorSummary, FittedBinomial, FittedGlmm, FittedIsotonic,
    FittedNegativeBinomial, FittedPSpline, FittedPoisson, FittedQuantile, FittedRegressor, GlmmDdf,
    GlmmInference, GlmmRegressor, GlmmRegressorBuilder, IsotonicRegressor,
    NegativeBinomialRegressor, OutOfBounds, PSplineRegressor, PoissonRegressor, QuantileRegressor,
    Regressor,
};

// The README's code blocks run as doctests, so they cannot go stale.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;
