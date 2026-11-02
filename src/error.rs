//! The crate-level error type.
//!
//! Each module reports its own error type (most estimators
//! [`RegressionError`]); [`Error`](enum@Error) wraps all of them with `From`
//! conversions, so an application can use `?` on any call of the crate and
//! return one `anofox_regression::Result`.
//!
//! ```rust
//! use anofox_regression::prelude::*;
//! use anofox_regression::splines::Rcs;
//! use faer::{Col, Mat};
//!
//! fn fit() -> anofox_regression::Result<f64> {
//!     let z: Vec<f64> = (0..40).map(|i| i as f64).collect();
//!     let basis = Rcs::from_data(&z, 4)?; // SplineError
//!     let x = basis.basis(&z, "z").columns;
//!     let y = Col::from_fn(40, |i| (i as f64 / 6.0).sin());
//!     let fitted = OlsRegressor::builder().build().fit(&x, &y)?; // RegressionError
//!     Ok(fitted.result().r_squared)
//! }
//! # assert!(fit().unwrap() > 0.5);
//! ```

use thiserror::Error;

pub use crate::solvers::RegressionError;

/// Any error of this crate (see the [module docs](self)).
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Error {
    /// An estimator, solver or inference routine.
    #[error(transparent)]
    Regression(#[from] RegressionError),
    /// Regression options.
    #[error(transparent)]
    Options(#[from] crate::core::OptionsError),
    /// Missing-value handling.
    #[error(transparent)]
    Na(#[from] crate::core::NaError),
    /// The penalized GLM / AFT engine.
    #[error(transparent)]
    GlmEngine(#[from] crate::solvers::penalized_glm::GlmEngineError),
    /// Designs with a term map.
    #[error(transparent)]
    Design(#[from] crate::design::DesignError),
    /// Spline bases.
    #[error(transparent)]
    Spline(#[from] crate::splines::SplineError),
    /// Resampling validation and calibration.
    #[error(transparent)]
    Validation(#[from] crate::validation::ValidationError),
    /// Predictive accuracy indexes.
    #[error(transparent)]
    Accuracy(#[from] crate::diagnostics::accuracy::AccuracyError),
    /// Cluster-robust covariance.
    #[error(transparent)]
    Cluster(#[from] crate::inference::cluster::ClusterError),
    /// Wald tests and the ANOVA of a design.
    #[error(transparent)]
    Wald(#[from] crate::inference::anova::WaldError),
    /// ANOVA tables.
    #[error(transparent)]
    AnovaTable(#[from] crate::inference::AnovaTableError),
    /// Nested-model F tests.
    #[error(transparent)]
    NestedFTest(#[from] crate::inference::NestedFTestError),
    /// Prediction standard errors.
    #[error(transparent)]
    PredictSe(#[from] crate::inference::PredictSeError),
    /// Model approximation.
    #[error(transparent)]
    Approximation(#[from] crate::inference::approximation::ApproximationError),
    /// Likelihood-ratio, score and profile-likelihood inference.
    #[error(transparent)]
    Likelihood(#[from] crate::inference::likelihood::LikelihoodError),
    /// Survival analysis.
    #[error(transparent)]
    Survival(#[from] crate::survival::SurvivalError),
    /// Empirical-Bayes shrinkage.
    #[error(transparent)]
    EbShrink(#[from] crate::solvers::eb_shrink::EbShrinkError),
    /// Causal mediation.
    #[error(transparent)]
    Mediation(#[from] crate::solvers::mediation::MediationError),
    /// Demand classification (AID).
    #[error(transparent)]
    Aid(#[from] crate::solvers::aid::heuristic::AidError),
}

/// `Result` with the crate-level [`Error`](enum@Error).
pub type Result<T, E = Error> = std::result::Result<T, E>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn module_errors_convert_and_keep_their_message() {
        let e: Error = RegressionError::invalid_parameter("tau", "must be in (0, 1)").into();
        assert_eq!(e.to_string(), "invalid parameter `tau`: must be in (0, 1)");
        assert!(matches!(
            e,
            Error::Regression(RegressionError::InvalidParameter { name: "tau", .. })
        ));
        let e: Error = crate::solvers::penalized_glm::GlmEngineError::SingularMatrix.into();
        assert!(matches!(e, Error::GlmEngine(_)));
        let r: RegressionError = crate::solvers::penalized_glm::GlmEngineError::InvalidValue {
            field: "alpha",
            message: "must be positive".into(),
        }
        .into();
        assert!(matches!(
            r,
            RegressionError::InvalidParameter { name: "alpha", .. }
        ));
    }
}
