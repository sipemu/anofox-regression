//! Gamma GLM regression — sklearn-style convenience wrapper around
//! [`TweedieRegressor`] with `var_power = 2.0`.
//!
//! For modeling positive continuous outcomes (costs, durations, claim sizes)
//! where the mean–variance relationship is `Var[Y] = φ * μ^2`. The canonical
//! link is the log link, which is also what scikit-learn's
//! `GammaRegressor` uses.
//!
//! This is intentionally a thin re-skin of [`TweedieRegressor`]: the
//! underlying numerics, IRLS loop, inference and prediction code are shared.
//! See `TweedieRegressor::gamma()` for the long-form factory.
//!
//! # Example
//!
//! ```rust,ignore
//! use anofox_regression::solvers::{GammaRegressor, Regressor, FittedRegressor};
//!
//! let fitted = GammaRegressor::builder()
//!     .with_intercept(true)
//!     .max_iterations(50)
//!     .build()
//!     .fit(&x, &y)?;
//!
//! let mu_hat = fitted.predict(&x_new);
//! ```

use crate::core::{IntervalType, PredictionResult, RegressionResult};
use crate::solvers::traits::{FittedRegressor, RegressionError, Regressor};
use crate::solvers::tweedie::{FittedTweedie, TweedieRegressor};
use faer::{Col, Mat};

/// Gamma GLM regression with log link.
///
/// Equivalent to `TweedieRegressor::gamma().build()` — see
/// [`crate::solvers::TweedieRegressor`] for the full numerical contract.
#[derive(Debug, Clone)]
pub struct GammaRegressor {
    inner: TweedieRegressor,
}

impl GammaRegressor {
    /// Start configuring a Gamma regressor.
    pub fn builder() -> GammaRegressorBuilder {
        GammaRegressorBuilder::default()
    }
}

impl Regressor for GammaRegressor {
    type Fitted = FittedGamma;

    fn fit(&self, x: &Mat<f64>, y: &Col<f64>) -> Result<Self::Fitted, RegressionError> {
        let inner = self.inner.fit(x, y)?;
        Ok(FittedGamma { inner })
    }
}

/// A fitted Gamma GLM model.
#[derive(Debug)]
pub struct FittedGamma {
    inner: FittedTweedie,
}

impl FittedGamma {
    /// Mean predictions on the response scale (i.e. `exp(η)` for the log link).
    pub fn predict_mu(&self, x: &Mat<f64>) -> Col<f64> {
        self.inner.predict_mu(x)
    }

    /// Linear predictor `η = Xβ + intercept` on the link scale.
    pub fn predict_eta(&self, x: &Mat<f64>) -> Col<f64> {
        self.inner.predict_eta(x)
    }

    /// Access the wrapped [`FittedTweedie`] for full GLM diagnostics.
    pub fn inner(&self) -> &FittedTweedie {
        &self.inner
    }
}

impl FittedRegressor for FittedGamma {
    fn predict(&self, x: &Mat<f64>) -> Col<f64> {
        self.inner.predict(x)
    }

    fn result(&self) -> &RegressionResult {
        self.inner.result()
    }

    fn predict_with_interval(
        &self,
        x: &Mat<f64>,
        interval: Option<IntervalType>,
        level: f64,
    ) -> PredictionResult {
        self.inner.predict_with_interval(x, interval, level)
    }
}

/// Builder for [`GammaRegressor`].
///
/// Mirrors the subset of `TweedieRegressorBuilder` relevant for Gamma
/// regression (var_power is fixed to `2.0`).
#[derive(Debug, Clone)]
pub struct GammaRegressorBuilder {
    with_intercept: bool,
    compute_inference: bool,
    confidence_level: f64,
    max_iterations: usize,
    tolerance: f64,
    lambda: f64,
    link_power: Option<f64>,
    offset: Option<Col<f64>>,
}

impl Default for GammaRegressorBuilder {
    fn default() -> Self {
        Self {
            with_intercept: true,
            compute_inference: true,
            confidence_level: 0.95,
            max_iterations: 25,
            tolerance: 1e-8,
            lambda: 0.0,
            link_power: None,
            offset: None,
        }
    }
}

impl GammaRegressorBuilder {
    /// Whether to include an intercept term (default: `true`).
    pub fn with_intercept(mut self, include: bool) -> Self {
        self.with_intercept = include;
        self
    }

    /// Whether to compute coefficient inference (default: `true`).
    pub fn compute_inference(mut self, compute: bool) -> Self {
        self.compute_inference = compute;
        self
    }

    /// Confidence level for coefficient intervals (default: `0.95`).
    pub fn confidence_level(mut self, level: f64) -> Self {
        self.confidence_level = level;
        self
    }

    /// Maximum IRLS iterations (default: `25`).
    pub fn max_iterations(mut self, max_iter: usize) -> Self {
        self.max_iterations = max_iter;
        self
    }

    /// IRLS convergence tolerance (default: `1e-8`).
    pub fn tolerance(mut self, tol: f64) -> Self {
        self.tolerance = tol;
        self
    }

    /// L2 (ridge) penalty strength applied to non-intercept coefficients
    /// (default: `0.0`, i.e. unpenalised).
    pub fn lambda(mut self, lambda: f64) -> Self {
        self.lambda = lambda;
        self
    }

    /// Override the link power. Defaults to the canonical link `log` for
    /// Gamma (which corresponds to `link_power = -1`).
    pub fn link_power(mut self, q: f64) -> Self {
        self.link_power = Some(q);
        self
    }

    /// Offset on the linear predictor (one entry per observation).
    pub fn offset(mut self, offset: Col<f64>) -> Self {
        self.offset = Some(offset);
        self
    }

    /// Build the [`GammaRegressor`].
    pub fn build(self) -> GammaRegressor {
        let mut tw = TweedieRegressor::gamma()
            .with_intercept(self.with_intercept)
            .compute_inference(self.compute_inference)
            .confidence_level(self.confidence_level)
            .max_iterations(self.max_iterations)
            .tolerance(self.tolerance)
            .lambda(self.lambda);

        if let Some(q) = self.link_power {
            tw = tw.link_power(q);
        }
        if let Some(offset) = self.offset {
            tw = tw.offset(offset);
        }

        GammaRegressor { inner: tw.build() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dataset() -> (Mat<f64>, Col<f64>) {
        // Reproduce a small Gamma-shaped sample with mean = exp(0.5 + 0.4 x).
        let xs = [0.5_f64, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0];
        let n = xs.len();
        let x = Mat::from_fn(n, 1, |i, _| xs[i]);
        let mut y = Col::zeros(n);
        // Deterministic Gamma-ish values; exact values don't matter — we
        // only check that GammaRegressor matches TweedieRegressor::gamma().
        let raws = [1.7, 2.3, 3.4, 5.0, 7.6, 11.2, 16.0, 24.5];
        for i in 0..n {
            y[i] = raws[i];
        }
        (x, y)
    }

    #[test]
    fn matches_tweedie_gamma_factory() {
        let (x, y) = dataset();

        let tweedie_fit = TweedieRegressor::gamma()
            .with_intercept(true)
            .compute_inference(false)
            .max_iterations(50)
            .tolerance(1e-10)
            .build()
            .fit(&x, &y)
            .unwrap();

        let gamma_fit = GammaRegressor::builder()
            .with_intercept(true)
            .compute_inference(false)
            .max_iterations(50)
            .tolerance(1e-10)
            .build()
            .fit(&x, &y)
            .unwrap();

        // Coefficients should match exactly: same numerics, same options.
        let a = tweedie_fit.result();
        let b = gamma_fit.result();
        assert!((a.intercept.unwrap() - b.intercept.unwrap()).abs() < 1e-12);
        assert_eq!(a.coefficients.nrows(), b.coefficients.nrows());
        for i in 0..a.coefficients.nrows() {
            assert!((a.coefficients[i] - b.coefficients[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn predict_delegates_to_tweedie() {
        let (x, y) = dataset();
        let gamma_fit = GammaRegressor::builder()
            .compute_inference(false)
            .build()
            .fit(&x, &y)
            .unwrap();

        let mu = gamma_fit.predict(&x);
        let mu_link = gamma_fit.predict_mu(&x);
        for i in 0..x.nrows() {
            assert!((mu[i] - mu_link[i]).abs() < 1e-12);
            assert!(mu[i] > 0.0, "Gamma predictions must be positive");
        }
    }
}
