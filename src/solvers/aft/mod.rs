//! Accelerated failure time (AFT) survival regression (issue #45).
//!
//! `log T = x'beta + sigma * W` with right censoring, where `W` follows a standard
//! extreme-value (Weibull, exponential), normal (log-normal) or logistic
//! (log-logistic) distribution — the parametrisation of R's `survival::survreg`.
//! Fitted by Newton–Raphson on `(beta, log sigma)` with step halving, optional
//! Gaussian/Laplace priors ([`PriorSpec`](crate::solvers::penalized_glm::PriorSpec))
//! and Laplace / sandwich / naive covariance
//! ([`VcovType`](crate::solvers::penalized_glm::VcovType)).
//!
//! Entry points: [`AftRegressor`] (builder over faer `Mat`/`Col` with a boolean
//! event indicator) and [`fit_aft`] (column-major `&[Vec<f64>]` design, event
//! indicator as `f64`, as a SQL / FFI front end calls it). [`AftDistribution`]
//! exposes the standardized and time-scale distribution functions
//! (`cdf_time`, `survival_time`, `quantile_time`).
//!
//! # Example
//!
//! ```
//! use anofox_regression::solvers::aft::{AftDistribution, AftRegressor};
//! use faer::{Col, Mat};
//!
//! let x = Mat::from_fn(10, 1, |i, _| (i % 2) as f64);
//! let t = Col::from_fn(10, |i| [5.0, 9.0, 6.0, 14.0, 4.0, 11.0, 7.0, 16.0, 5.5, 12.0][i]);
//! let event = vec![true, true, true, false, true, true, true, true, false, true];
//! let fit = AftRegressor::builder()
//!     .distribution(AftDistribution::Weibull)
//!     .compute_inference(true)
//!     .build()
//!     .fit(&x, &t, &event)
//!     .unwrap();
//! assert!(fit.converged());
//! assert!(fit.coefficients()[0] > 0.0); // group 1 survives longer
//! let median = fit.predict_quantile(&x, 0.5);
//! assert!(median[1] > median[0]);
//! ```

pub mod distribution;
mod model;
mod regressor;

pub use distribution::AftDistribution;
pub use model::{fit_aft, AftFitResult, AftInference, AftOptions, AftResult};
pub use regressor::{AftRegressor, AftRegressorBuilder, FittedAft};
