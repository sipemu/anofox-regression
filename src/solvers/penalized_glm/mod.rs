//! Family-generic penalized GLM engine (issue #44).
//!
//! One IRLS loop, generic over [`GlmFamily`](crate::core::GlmFamily), with:
//!
//! * explicit per-coefficient priors ([`PriorSpec`]: flat, Gaussian, Laplace) and a
//!   legacy uniform ridge (`lambda`), entering the normal equations as `X'WX + P`
//!   (Laplace/L1 priors switch the inner solve to proximal coordinate descent);
//! * a selectable covariance at the mode ([`VcovType`]: Laplace `(X'WX+P)^-1`,
//!   sandwich `(X'WX+P)^-1 X'WX (X'WX+P)^-1`, naive `(X'WX)^-1`);
//! * offsets, a constant-column policy and aliased-column handling (`NaN` in the
//!   expanded results);
//! * real per-family log-likelihoods, AIC/BIC, deviance, null deviance, pseudo-R²
//!   and a `converged` flag (the last iterate is returned on non-convergence).
//!
//! Supported families: Poisson (log / identity / sqrt), Binomial (logit / probit /
//! cloglog), Logistic (binomial-logit plus training accuracy), Negative Binomial
//! (fixed `theta`, or `theta` estimated by the `MASS::glm.nb` alternation), Gamma
//! (log link) and Tweedie (`1 <= p <= 2`, log link).
//!
//! Two entry points:
//!
//! * [`PenalizedGlmRegressor`] — an idiomatic builder taking a faer `Mat`/`Col`;
//! * the per-family functions ([`fit_poisson`], [`fit_binomial`], [`fit_logistic`],
//!   [`fit_negbinomial`], [`fit_gamma`], [`fit_tweedie`]) taking a column-major
//!   `&[Vec<f64>]` design (rows with a non-finite value are dropped) and the
//!   per-family option structs. These are what a SQL / FFI front end calls.
//!
//! The lower-level building blocks ([`engine`], [`irls`], [`laplace`], [`loglik`],
//! [`normal_eq`], [`penalty`], [`design`]) are public as well so other models (e.g.
//! the AFT model in [`crate::solvers::aft`], or mixed models with Gaussian random
//! effects as quadratic precision blocks) can reuse them.
//!
//! # Example
//!
//! ```
//! use anofox_regression::solvers::penalized_glm::{
//!     PenalizedGlmFamily, PenalizedGlmRegressor, PriorSpec, VcovType,
//! };
//! use faer::{Col, Mat};
//!
//! let x = Mat::from_fn(8, 1, |i, _| i as f64);
//! let y = Col::from_fn(8, |i| [1.0, 0.0, 2.0, 3.0, 2.0, 5.0, 6.0, 9.0][i]);
//! let fit = PenalizedGlmRegressor::builder()
//!     .family(PenalizedGlmFamily::poisson())
//!     .priors(vec![PriorSpec::flat(), PriorSpec::normal(0.0, 1.0)])
//!     .vcov_type(VcovType::Laplace)
//!     .compute_inference(true)
//!     .build()
//!     .fit(&x, &y)
//!     .unwrap();
//! assert!(fit.converged());
//! assert!(fit.coefficients()[0] > 0.0);
//! assert!(fit.std_errors().unwrap()[0] > 0.0);
//! ```

pub mod design;
pub mod engine;
pub mod error;
mod families;
pub mod irls;
pub mod laplace;
pub mod loglik;
pub mod normal_eq;
pub mod penalty;
mod regressor;
pub mod types;

#[cfg(test)]
mod parity;

pub use design::ConstantColumnPolicy;
pub use engine::{fit, DispersionRule, EngineFit, EngineOptions};
pub use error::{GlmEngineError, GlmEngineResult};
pub use families::{
    fit_binomial, fit_gamma, fit_logistic, fit_negbinomial, fit_poisson, fit_tweedie, GlmResult,
    LogisticResult,
};
pub use loglik::LogLikKind;
pub use regressor::{
    FittedPenalizedGlm, PenalizedGlmFamily, PenalizedGlmRegressor, PenalizedGlmRegressorBuilder,
};
pub use types::{
    BinomialOptions, GammaOptions, GlmFitResult, GlmInferenceResult, GlmPriorOptions,
    LogisticOptions, NegBinomialOptions, PoissonOptions, PriorKind, PriorSpec, TweedieOptions,
    VcovType,
};

/// Relative range tolerance for treating a column as constant when an intercept
/// is fitted.
const CONSTANT_RANGE_RTOL: f64 = 1e-10;

/// True when column `col` restricted to `rows` should be dropped as aliased:
/// constant (scale-relative test) with an intercept, all-zero without one.
pub(crate) fn is_droppable_column(col: &[f64], rows: &[usize], fit_intercept: bool) -> bool {
    let m = faer::Mat::from_fn(rows.len(), 1, |i, _| col[rows[i]]);
    if fit_intercept {
        crate::utils::detect_constant_columns_relative(&m, CONSTANT_RANGE_RTOL)[0]
    } else {
        crate::utils::detect_zero_columns(&m)[0]
    }
}
