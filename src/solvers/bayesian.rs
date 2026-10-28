//! Bayesian linear regression: empirical-Bayes / type-II ML estimators.
//!
//! [`BayesianRidge`] places a Gaussian prior `β | λ ~ N(0, λ⁻¹ I)` and
//! tunes the noise precision `α` and weight precision `λ` jointly via the
//! evidence approximation (a.k.a. ML-II). Closed-form updates use the
//! singular-value decomposition of the centered design matrix.
//!
//! [`ArdRegression`] is the same model with a *per-feature* precision
//! `λ_j`; large `λ_j` drives the corresponding `β_j` toward zero, giving
//! automatic feature relevance pruning.
//!
//! Both estimators follow `sklearn.linear_model.BayesianRidge` and
//! `sklearn.linear_model.ARDRegression` step for step (initialisation,
//! update order, convergence test and the final refit with the converged
//! hyperparameters), with the same defaults (gamma-prior shapes
//! `alpha_1 = alpha_2 = lambda_1 = lambda_2 = 1e-6`).
//!
//! Both keep the posterior covariance `Σ = (λ I + α XᵀX)⁻¹` of the
//! coefficients (centered design; for ARD, `diag(λ_kept)` over the kept
//! features). The posterior predictive standard deviation of a new row `x`
//! is `sqrt((x − x̄)ᵀ Σ (x − x̄) + 1/α)`, as sklearn's
//! `predict(X, return_std=True)`; [`FittedRegressor::predict_with_interval`]
//! turns it into Gaussian credible intervals (`IntervalType::Prediction`)
//! or, without the `1/α` noise term, intervals for the mean
//! (`IntervalType::Confidence`).
//!
//! References:
//!  - Tipping, M. E. (2001). "Sparse Bayesian Learning and the Relevance
//!    Vector Machine". JMLR 1: 211–244.
//!  - MacKay, D. J. C. (1992). "Bayesian Interpolation". Neural Computation.
//!
//! # Example
//!
//! ```rust,ignore
//! use anofox_regression::solvers::{BayesianRidge, Regressor, FittedRegressor};
//!
//! let fitted = BayesianRidge::builder()
//!     .max_iter(300)
//!     .tolerance(1e-3)
//!     .build()
//!     .fit(&x, &y)?;
//! let std = fitted.predict_std(&x_new);
//! ```

#![allow(clippy::needless_range_loop)]

use crate::core::{HasModelInfo, ModelInfo};
use crate::core::{IntervalType, PredictionResult, RegressionResult};
use crate::solvers::traits::{FittedRegressor, RegressionError, Regressor};
use faer::prelude::Solve;
use faer::{Col, Mat};
use statrs::distribution::{ContinuousCDF, Normal};

// ----------------------------------------------------------------------------
// Shared helpers
// ----------------------------------------------------------------------------

/// Design and response centered on their means (when fitting an intercept),
/// as sklearn's `_preprocess_data`.
struct Centered {
    xc: Mat<f64>,
    yc: Col<f64>,
    x_mean: Vec<f64>,
    y_mean: f64,
}

fn center(x: &Mat<f64>, y: &Col<f64>, fit_intercept: bool) -> Centered {
    let n = x.nrows();
    let p = x.ncols();
    let mut x_mean = vec![0.0_f64; p];
    let mut y_mean = 0.0_f64;
    if fit_intercept {
        for j in 0..p {
            x_mean[j] = (0..n).map(|i| x[(i, j)]).sum::<f64>() / n as f64;
        }
        y_mean = (0..n).map(|i| y[i]).sum::<f64>() / n as f64;
    }
    Centered {
        xc: Mat::from_fn(n, p, |i, j| x[(i, j)] - x_mean[j]),
        yc: Col::from_fn(n, |i| y[i] - y_mean),
        x_mean,
        y_mean,
    }
}

/// `np.var(y)`: the (1/n) variance about the mean.
fn variance(y: &Col<f64>) -> f64 {
    let n = y.nrows();
    let mean = (0..n).map(|i| y[i]).sum::<f64>() / n as f64;
    (0..n).map(|i| (y[i] - mean).powi(2)).sum::<f64>() / n as f64
}

fn check_dims(x: &Mat<f64>, y: &Col<f64>, min_obs: usize) -> Result<(), RegressionError> {
    if x.nrows() != y.nrows() {
        return Err(RegressionError::DimensionMismatch {
            x_rows: x.nrows(),
            y_len: y.nrows(),
        });
    }
    if x.nrows() < min_obs {
        return Err(RegressionError::InsufficientObservations {
            needed: min_obs,
            got: x.nrows(),
        });
    }
    Ok(())
}

/// Inverse of a symmetric positive-definite matrix via Cholesky.
fn spd_inverse(a: &Mat<f64>) -> Result<Mat<f64>, RegressionError> {
    let k = a.nrows();
    let llt = a
        .llt(faer::Side::Lower)
        .map_err(|_| RegressionError::SingularMatrix)?;
    Ok(llt.solve(Mat::<f64>::identity(k, k)))
}

/// Builds the regression result shared by both estimators. Bayesian fits have
/// no classical residual degrees of freedom or likelihood-based fit statistics:
/// `adj_r_squared`, `rmse`, the F test, the log-likelihood and the
/// information criteria are NaN rather than the `RegressionResult::empty`
/// zeros (which would read as a perfect fit).
fn bayesian_result(
    x: &Mat<f64>,
    y: &Col<f64>,
    coef: &[f64],
    intercept: Option<f64>,
) -> RegressionResult {
    let n = x.nrows();
    let p = x.ncols();
    let fitted_values = Col::from_fn(n, |i| {
        let mut v = intercept.unwrap_or(0.0);
        for j in 0..p {
            v += x[(i, j)] * coef[j];
        }
        v
    });
    let residuals = Col::from_fn(n, |i| y[i] - fitted_values[i]);
    let n_params = p + usize::from(intercept.is_some());
    let mut result = RegressionResult::empty(p, n);
    result.coefficients = Col::from_fn(p, |i| coef[i]);
    result.intercept = intercept;
    result.r_squared = compute_r_squared(y, &residuals);
    result.residuals = residuals;
    result.fitted_values = fitted_values;
    result.rank = n_params;
    result.n_parameters = n_params;
    result.n_observations = n;
    result.adj_r_squared = f64::NAN;
    result.rmse = f64::NAN;
    result.mse = f64::NAN;
    result.f_statistic = f64::NAN;
    result.f_pvalue = f64::NAN;
    result.log_likelihood = f64::NAN;
    result.aic = f64::NAN;
    result.aicc = f64::NAN;
    result.bic = f64::NAN;
    result
}

/// Posterior predictive mean and standard deviation of the rows of `x`:
/// `sqrt(x̃ᵀ Σ x̃ + noise_var)` with `x̃` the centered row restricted to
/// `columns` (the features `sigma` is defined over).
fn posterior_std(
    x: &Mat<f64>,
    x_mean: &[f64],
    columns: &[usize],
    sigma: &Mat<f64>,
    noise_var: f64,
) -> Col<f64> {
    let k = columns.len();
    Col::from_fn(x.nrows(), |i| {
        let xc: Vec<f64> = columns.iter().map(|&j| x[(i, j)] - x_mean[j]).collect();
        let mut q = 0.0_f64;
        for a in 0..k {
            let mut s = 0.0;
            for b in 0..k {
                s += sigma[(a, b)] * xc[b];
            }
            q += xc[a] * s;
        }
        (q.max(0.0) + noise_var).sqrt()
    })
}

/// What the posterior predictive needs: feature means, the features the
/// posterior covariance is defined over, the covariance and the noise
/// precision.
struct Posterior<'a> {
    x_mean: &'a [f64],
    columns: &'a [usize],
    sigma: &'a Mat<f64>,
    alpha: f64,
}

impl Posterior<'_> {
    /// Point predictions with Gaussian credible bounds `fit ± z · sd`, where
    /// `sd` includes the noise variance `1/α` for prediction intervals and not
    /// for confidence intervals (the posterior of the mean).
    fn interval(
        &self,
        fit: Col<f64>,
        x: &Mat<f64>,
        interval: Option<IntervalType>,
        level: f64,
    ) -> PredictionResult {
        let Some(kind) = interval else {
            return PredictionResult::point_only(fit);
        };
        let noise_var = match kind {
            IntervalType::Prediction => 1.0 / self.alpha,
            IntervalType::Confidence => 0.0,
        };
        let sd = posterior_std(x, self.x_mean, self.columns, self.sigma, noise_var);
        let z = Normal::new(0.0, 1.0)
            .map(|d| d.inverse_cdf(0.5 + level / 2.0))
            .unwrap_or(f64::NAN);
        let n = fit.nrows();
        let lower = Col::from_fn(n, |i| fit[i] - z * sd[i]);
        let upper = Col::from_fn(n, |i| fit[i] + z * sd[i]);
        PredictionResult::with_intervals(fit, lower, upper, sd)
    }
}

fn linear_predict(x: &Mat<f64>, result: &RegressionResult) -> Col<f64> {
    let p = x.ncols();
    let intercept = result.intercept.unwrap_or(0.0);
    Col::from_fn(x.nrows(), |i| {
        let mut v = intercept;
        for j in 0..p {
            v += x[(i, j)] * result.coefficients[j];
        }
        v
    })
}

// ----------------------------------------------------------------------------
// BayesianRidge
// ----------------------------------------------------------------------------

/// Bayesian Ridge regression with evidence-maximisation hyperparameter
/// updates.
#[derive(Debug, Clone)]
pub struct BayesianRidge {
    fit_intercept: bool,
    max_iter: usize,
    tol: f64,
    alpha_1: f64,
    alpha_2: f64,
    lambda_1: f64,
    lambda_2: f64,
    alpha_init: Option<f64>,
    lambda_init: Option<f64>,
}

impl Default for BayesianRidge {
    fn default() -> Self {
        Self {
            fit_intercept: true,
            max_iter: 300,
            tol: 1e-3,
            alpha_1: 1e-6,
            alpha_2: 1e-6,
            lambda_1: 1e-6,
            lambda_2: 1e-6,
            alpha_init: None,
            lambda_init: None,
        }
    }
}

impl BayesianRidge {
    pub fn builder() -> BayesianRidgeBuilder {
        BayesianRidgeBuilder::default()
    }
}

/// The SVD pieces of the centered design used by the BayesianRidge updates.
struct RidgeSvd {
    v: Mat<f64>,
    s: Vec<f64>,
    eig: Vec<f64>,
    uty: Vec<f64>,
}

impl RidgeSvd {
    /// Posterior mean `β = V diag(s / (s² + λ/α)) Uᵀy` and its residual sum of
    /// squares on the centered data (sklearn's `_update_coef_`).
    fn coef(&self, xc: &Mat<f64>, yc: &Col<f64>, alpha: f64, lambda: f64) -> (Vec<f64>, f64) {
        let k = self.s.len();
        let p = xc.ncols();
        let d = Col::from_fn(k, |i| {
            let denom = self.eig[i] + lambda / alpha;
            if denom > 0.0 {
                self.s[i] * self.uty[i] / denom
            } else {
                0.0
            }
        });
        let beta_col = &self.v * d;
        let beta: Vec<f64> = (0..p).map(|j| beta_col[j]).collect();
        let xb = xc * &beta_col;
        let sse = (0..xc.nrows()).map(|i| (yc[i] - xb[i]).powi(2)).sum();
        (beta, sse)
    }
}

impl Regressor for BayesianRidge {
    type Fitted = FittedBayesianRidge;

    fn fit(&self, x: &Mat<f64>, y: &Col<f64>) -> Result<Self::Fitted, RegressionError> {
        check_dims(x, y, 1)?;
        let n = x.nrows();
        let p = x.ncols();
        let c = center(x, y, self.fit_intercept);

        let svd =
            c.xc.thin_svd()
                .map_err(|_| RegressionError::SingularMatrix)?;
        let u = svd.U();
        let s_col = svd.S().column_vector();
        let k = s_col.nrows();
        let s: Vec<f64> = (0..k).map(|i| s_col[i]).collect();
        let uty_col = u.transpose() * &c.yc;
        let parts = RidgeSvd {
            v: svd.V().to_owned(),
            eig: s.iter().map(|v| v * v).collect(),
            uty: (0..k).map(|i| uty_col[i]).collect(),
            s,
        };

        let mut alpha = self
            .alpha_init
            .unwrap_or_else(|| 1.0 / (variance(y) + f64::EPSILON));
        let mut lambda = self.lambda_init.unwrap_or(1.0);

        let mut coef_old: Option<Vec<f64>> = None;
        let mut n_iter = 0;
        for iter in 0..self.max_iter {
            n_iter = iter + 1;
            let (coef, sse) = parts.coef(&c.xc, &c.yc, alpha, lambda);

            // MacKay (1992) updates.
            let gamma: f64 = parts
                .eig
                .iter()
                .map(|e| alpha * e / (lambda + alpha * e))
                .sum();
            let beta_sq: f64 = coef.iter().map(|b| b * b).sum();
            lambda = (gamma + 2.0 * self.lambda_1) / (beta_sq + 2.0 * self.lambda_2);
            alpha = (n as f64 - gamma + 2.0 * self.alpha_1) / (sse + 2.0 * self.alpha_2);

            if let Some(old) = &coef_old {
                let change: f64 = old.iter().zip(&coef).map(|(a, b)| (a - b).abs()).sum();
                if change < self.tol {
                    break;
                }
            }
            coef_old = Some(coef);
        }

        // Posterior mean with the converged hyperparameters.
        let (coef, sse) = parts.coef(&c.xc, &c.yc, alpha, lambda);

        // Posterior covariance Σ = (λ I + α XᵀX)⁻¹.
        let xtx = c.xc.transpose() * &c.xc;
        let precision = Mat::from_fn(p, p, |i, j| {
            alpha * xtx[(i, j)] + if i == j { lambda } else { 0.0 }
        });
        let sigma = spd_inverse(&precision)?;

        let log_marginal_likelihood = self.log_marginal_likelihood(
            n,
            p,
            &parts.eig,
            alpha,
            lambda,
            coef.iter().map(|b| b * b).sum(),
            sse,
        );

        let intercept = self
            .fit_intercept
            .then(|| c.y_mean - c.x_mean.iter().zip(&coef).map(|(m, b)| m * b).sum::<f64>());
        let result = bayesian_result(x, y, &coef, intercept);

        Ok(FittedBayesianRidge {
            result,
            alpha,
            lambda,
            sigma,
            x_mean: c.x_mean,
            n_iter,
            log_marginal_likelihood,
        })
    }
}

impl BayesianRidge {
    /// sklearn's `_log_marginal_likelihood` (the last entry of `scores_`).
    #[allow(clippy::too_many_arguments)]
    fn log_marginal_likelihood(
        &self,
        n: usize,
        p: usize,
        eig: &[f64],
        alpha: f64,
        lambda: f64,
        beta_sq: f64,
        sse: f64,
    ) -> f64 {
        // log |Σ| over all p directions: eigenvalues of XᵀX beyond the SVD
        // rank are zero.
        let mut logdet_sigma = 0.0;
        for j in 0..p {
            let e = eig.get(j).copied().unwrap_or(0.0);
            logdet_sigma -= (lambda + alpha * e).ln();
        }
        let mut score = self.lambda_1 * lambda.ln() - self.lambda_2 * lambda;
        score += self.alpha_1 * alpha.ln() - self.alpha_2 * alpha;
        score += 0.5
            * (p as f64 * lambda.ln() + n as f64 * alpha.ln() - alpha * sse - lambda * beta_sq
                + logdet_sigma
                - n as f64 * (2.0 * std::f64::consts::PI).ln());
        score
    }
}

/// A fitted Bayesian Ridge model.
#[derive(Debug, Clone)]
pub struct FittedBayesianRidge {
    result: RegressionResult,
    alpha: f64,
    lambda: f64,
    sigma: Mat<f64>,
    x_mean: Vec<f64>,
    n_iter: usize,
    log_marginal_likelihood: f64,
}

impl HasModelInfo for FittedBayesianRidge {
    fn model_info(&self) -> ModelInfo {
        ModelInfo::gaussian("bayesian_ridge")
    }
}

impl FittedBayesianRidge {
    /// Estimated noise precision `α` (sklearn `alpha_`).
    pub fn alpha(&self) -> f64 {
        self.alpha
    }
    /// Estimated weight precision `λ` (sklearn `lambda_`).
    pub fn lambda(&self) -> f64 {
        self.lambda
    }
    /// Posterior covariance of the coefficients, `p × p` (sklearn `sigma_`).
    pub fn sigma(&self) -> &Mat<f64> {
        &self.sigma
    }
    /// Diagonal of the posterior covariance: the posterior variances.
    pub fn sigma_diag(&self) -> Vec<f64> {
        (0..self.sigma.nrows())
            .map(|j| self.sigma[(j, j)])
            .collect()
    }
    /// Number of evidence iterations run (sklearn `n_iter_`).
    pub fn n_iter(&self) -> usize {
        self.n_iter
    }
    /// Log marginal likelihood (evidence) at the converged hyperparameters,
    /// sklearn's final `scores_` entry with `compute_score=True`.
    pub fn log_marginal_likelihood(&self) -> f64 {
        self.log_marginal_likelihood
    }
    /// Posterior predictive standard deviation of the rows of `x`, as
    /// sklearn's `predict(X, return_std=True)`.
    pub fn predict_std(&self, x: &Mat<f64>) -> Col<f64> {
        let columns: Vec<usize> = (0..x.ncols()).collect();
        posterior_std(x, &self.x_mean, &columns, &self.sigma, 1.0 / self.alpha)
    }
}

impl FittedRegressor for FittedBayesianRidge {
    fn predict(&self, x: &Mat<f64>) -> Col<f64> {
        linear_predict(x, &self.result)
    }
    fn result(&self) -> &RegressionResult {
        &self.result
    }
    fn predict_with_interval(
        &self,
        x: &Mat<f64>,
        interval: Option<IntervalType>,
        level: f64,
    ) -> PredictionResult {
        let columns: Vec<usize> = (0..x.ncols()).collect();
        let posterior = Posterior {
            x_mean: &self.x_mean,
            columns: &columns,
            sigma: &self.sigma,
            alpha: self.alpha,
        };
        posterior.interval(self.predict(x), x, interval, level)
    }
}

/// Builder for [`BayesianRidge`].
#[derive(Debug, Clone, Default)]
pub struct BayesianRidgeBuilder {
    inner: BayesianRidge,
}
impl BayesianRidgeBuilder {
    pub fn fit_intercept(mut self, v: bool) -> Self {
        self.inner.fit_intercept = v;
        self
    }
    pub fn max_iter(mut self, v: usize) -> Self {
        self.inner.max_iter = v;
        self
    }
    pub fn tolerance(mut self, v: f64) -> Self {
        self.inner.tol = v;
        self
    }
    pub fn alpha_1(mut self, v: f64) -> Self {
        self.inner.alpha_1 = v;
        self
    }
    pub fn alpha_2(mut self, v: f64) -> Self {
        self.inner.alpha_2 = v;
        self
    }
    pub fn lambda_1(mut self, v: f64) -> Self {
        self.inner.lambda_1 = v;
        self
    }
    pub fn lambda_2(mut self, v: f64) -> Self {
        self.inner.lambda_2 = v;
        self
    }
    pub fn alpha_init(mut self, v: f64) -> Self {
        self.inner.alpha_init = Some(v);
        self
    }
    pub fn lambda_init(mut self, v: f64) -> Self {
        self.inner.lambda_init = Some(v);
        self
    }
    pub fn build(self) -> BayesianRidge {
        self.inner
    }
}

// ----------------------------------------------------------------------------
// ARDRegression
// ----------------------------------------------------------------------------

/// Automatic Relevance Determination regression.
#[derive(Debug, Clone)]
pub struct ArdRegression {
    fit_intercept: bool,
    max_iter: usize,
    tol: f64,
    alpha_1: f64,
    alpha_2: f64,
    lambda_1: f64,
    lambda_2: f64,
    /// Features with `λ_j >= threshold_lambda` are pruned (β_j set to 0).
    threshold_lambda: f64,
}

impl Default for ArdRegression {
    fn default() -> Self {
        Self {
            fit_intercept: true,
            max_iter: 300,
            tol: 1e-3,
            alpha_1: 1e-6,
            alpha_2: 1e-6,
            lambda_1: 1e-6,
            lambda_2: 1e-6,
            threshold_lambda: 10_000.0,
        }
    }
}

impl ArdRegression {
    pub fn builder() -> ArdRegressionBuilder {
        ArdRegressionBuilder::default()
    }

    /// `Σ = (diag(λ_kept) + α X_keptᵀ X_kept)⁻¹` over the kept features.
    fn sigma(
        xtx: &Mat<f64>,
        alpha: f64,
        lambdas: &[f64],
        kept: &[usize],
    ) -> Result<Mat<f64>, RegressionError> {
        let k = kept.len();
        let a = Mat::from_fn(k, k, |i, j| {
            alpha * xtx[(kept[i], kept[j])] + if i == j { lambdas[kept[i]] } else { 0.0 }
        });
        spd_inverse(&a)
    }

    /// `β_kept = α Σ X_keptᵀ y`; the other coefficients keep their value.
    fn update_coef(coef: &mut [f64], sigma: &Mat<f64>, xty: &[f64], alpha: f64, kept: &[usize]) {
        let k = kept.len();
        for a in 0..k {
            let mut s = 0.0;
            for b in 0..k {
                s += sigma[(a, b)] * xty[kept[b]];
            }
            coef[kept[a]] = alpha * s;
        }
    }
}

impl Regressor for ArdRegression {
    type Fitted = FittedArd;

    fn fit(&self, x: &Mat<f64>, y: &Col<f64>) -> Result<Self::Fitted, RegressionError> {
        check_dims(x, y, 2)?;
        let n = x.nrows();
        let p = x.ncols();
        let c = center(x, y, self.fit_intercept);

        let mut alpha = 1.0 / (variance(&c.yc) + f64::EPSILON);
        let mut lambdas = vec![1.0_f64; p];
        let mut coef = vec![0.0_f64; p];
        let mut kept: Vec<usize> = (0..p).collect();

        let xtx = c.xc.transpose() * &c.xc;
        let xty_col = c.xc.transpose() * &c.yc;
        let xty: Vec<f64> = (0..p).map(|i| xty_col[i]).collect();
        let sse_of = |coef: &[f64]| -> f64 {
            let xb = &c.xc * Col::from_fn(p, |j| coef[j]);
            (0..n).map(|i| (c.yc[i] - xb[i]).powi(2)).sum()
        };

        let mut coef_old: Option<Vec<f64>> = None;
        let mut n_iter = 0;
        for iter in 0..self.max_iter {
            n_iter = iter + 1;
            let sigma = Self::sigma(&xtx, alpha, &lambdas, &kept)?;
            Self::update_coef(&mut coef, &sigma, &xty, alpha, &kept);
            let sse = sse_of(&coef);

            // γ_j = 1 − λ_j Σ_jj, then the λ_j and α updates (Tipping 2001).
            let mut gamma_sum = 0.0;
            for (i, &j) in kept.iter().enumerate() {
                let gamma = 1.0 - lambdas[j] * sigma[(i, i)];
                gamma_sum += gamma;
                lambdas[j] =
                    (gamma + 2.0 * self.lambda_1) / (coef[j] * coef[j] + 2.0 * self.lambda_2);
            }
            alpha = (n as f64 - gamma_sum + 2.0 * self.alpha_1) / (sse + 2.0 * self.alpha_2);

            // Prune features whose precision crossed the threshold.
            kept = (0..p)
                .filter(|&j| lambdas[j] < self.threshold_lambda)
                .collect();
            for j in 0..p {
                if lambdas[j] >= self.threshold_lambda {
                    coef[j] = 0.0;
                }
            }

            if let Some(old) = &coef_old {
                let change: f64 = old.iter().zip(&coef).map(|(a, b)| (a - b).abs()).sum();
                if change < self.tol {
                    break;
                }
            }
            coef_old = Some(coef.clone());
            if kept.is_empty() {
                break;
            }
        }

        // Posterior with the converged hyperparameters.
        let sigma = if kept.is_empty() {
            Mat::zeros(0, 0)
        } else {
            let sigma = Self::sigma(&xtx, alpha, &lambdas, &kept)?;
            Self::update_coef(&mut coef, &sigma, &xty, alpha, &kept);
            sigma
        };

        let intercept = self
            .fit_intercept
            .then(|| c.y_mean - c.x_mean.iter().zip(&coef).map(|(m, b)| m * b).sum::<f64>());
        let result = bayesian_result(x, y, &coef, intercept);

        Ok(FittedArd {
            result,
            alpha,
            lambdas,
            sigma,
            kept,
            x_mean: c.x_mean,
            n_iter,
        })
    }
}

/// A fitted ARD regression model.
#[derive(Debug, Clone)]
pub struct FittedArd {
    result: RegressionResult,
    alpha: f64,
    lambdas: Vec<f64>,
    sigma: Mat<f64>,
    kept: Vec<usize>,
    x_mean: Vec<f64>,
    n_iter: usize,
}

impl HasModelInfo for FittedArd {
    fn model_info(&self) -> ModelInfo {
        ModelInfo::gaussian("ard")
    }
}

impl FittedArd {
    /// Estimated noise precision `α` (sklearn `alpha_`).
    pub fn alpha(&self) -> f64 {
        self.alpha
    }
    /// Estimated per-feature weight precisions `λ_j` (sklearn `lambda_`).
    pub fn lambdas(&self) -> &[f64] {
        &self.lambdas
    }
    /// Indices of the features kept (`λ_j < threshold_lambda`).
    pub fn kept_features(&self) -> &[usize] {
        &self.kept
    }
    /// Posterior covariance over the kept features (sklearn `sigma_`).
    pub fn sigma(&self) -> &Mat<f64> {
        &self.sigma
    }
    /// Posterior variance of every coefficient; 0 for pruned features.
    pub fn sigma_diag(&self) -> Vec<f64> {
        let mut d = vec![0.0; self.lambdas.len()];
        for (i, &j) in self.kept.iter().enumerate() {
            d[j] = self.sigma[(i, i)];
        }
        d
    }
    /// Number of iterations run (sklearn `n_iter_`).
    pub fn n_iter(&self) -> usize {
        self.n_iter
    }
    /// Posterior predictive standard deviation of the rows of `x`, as
    /// sklearn's `predict(X, return_std=True)`.
    pub fn predict_std(&self, x: &Mat<f64>) -> Col<f64> {
        posterior_std(x, &self.x_mean, &self.kept, &self.sigma, 1.0 / self.alpha)
    }
}

impl FittedRegressor for FittedArd {
    fn predict(&self, x: &Mat<f64>) -> Col<f64> {
        linear_predict(x, &self.result)
    }
    fn result(&self) -> &RegressionResult {
        &self.result
    }
    fn predict_with_interval(
        &self,
        x: &Mat<f64>,
        interval: Option<IntervalType>,
        level: f64,
    ) -> PredictionResult {
        let posterior = Posterior {
            x_mean: &self.x_mean,
            columns: &self.kept,
            sigma: &self.sigma,
            alpha: self.alpha,
        };
        posterior.interval(self.predict(x), x, interval, level)
    }
}

/// Builder for [`ArdRegression`].
#[derive(Debug, Clone, Default)]
pub struct ArdRegressionBuilder {
    inner: ArdRegression,
}
impl ArdRegressionBuilder {
    pub fn fit_intercept(mut self, v: bool) -> Self {
        self.inner.fit_intercept = v;
        self
    }
    pub fn max_iter(mut self, v: usize) -> Self {
        self.inner.max_iter = v;
        self
    }
    pub fn tolerance(mut self, v: f64) -> Self {
        self.inner.tol = v;
        self
    }
    pub fn alpha_1(mut self, v: f64) -> Self {
        self.inner.alpha_1 = v;
        self
    }
    pub fn alpha_2(mut self, v: f64) -> Self {
        self.inner.alpha_2 = v;
        self
    }
    pub fn lambda_1(mut self, v: f64) -> Self {
        self.inner.lambda_1 = v;
        self
    }
    pub fn lambda_2(mut self, v: f64) -> Self {
        self.inner.lambda_2 = v;
        self
    }
    pub fn threshold_lambda(mut self, v: f64) -> Self {
        self.inner.threshold_lambda = v;
        self
    }
    pub fn build(self) -> ArdRegression {
        self.inner
    }
}

fn compute_r_squared(y: &Col<f64>, residuals: &Col<f64>) -> f64 {
    let n = y.nrows();
    let y_mean: f64 = (0..n).map(|i| y[i]).sum::<f64>() / n as f64;
    let tss: f64 = (0..n).map(|i| (y[i] - y_mean).powi(2)).sum();
    let rss: f64 = (0..n).map(|i| residuals[i].powi(2)).sum();
    if tss == 0.0 {
        if rss == 0.0 {
            1.0
        } else {
            0.0
        }
    } else {
        1.0 - rss / tss
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bayesian_ridge_recovers_clean_signal() {
        let n = 100;
        let p = 3;
        let x = Mat::from_fn(n, p, |i, j| {
            let t = i as f64 / 10.0;
            match j {
                0 => t,
                1 => (t * 0.3).sin(),
                _ => (t * 0.7).cos(),
            }
        });
        let true_beta = [1.0, 2.0, -0.5];
        let y = Col::from_fn(n, |i| {
            let mut v = 0.5;
            for j in 0..p {
                v += x[(i, j)] * true_beta[j];
            }
            v
        });

        let fitted = BayesianRidge::builder().build().fit(&x, &y).unwrap();
        for j in 0..p {
            assert!(
                (fitted.result.coefficients[j] - true_beta[j]).abs() < 0.1,
                "coef[{}] = {}",
                j,
                fitted.result.coefficients[j]
            );
        }
    }

    #[test]
    fn ard_zeros_irrelevant_features() {
        // p=4, only first two relevant.
        let n = 60;
        let p = 4;
        let x = Mat::from_fn(n, p, |i, j| {
            let t = i as f64 / 5.0;
            match j {
                0 => t,
                1 => (t * 0.3).sin(),
                _ => 0.001 * (t + j as f64), // tiny / nearly constant
            }
        });
        let true_beta = [2.0, -1.0, 0.0, 0.0];
        let y = Col::from_fn(n, |i| {
            let mut v = 0.0;
            for j in 0..p {
                v += x[(i, j)] * true_beta[j];
            }
            v
        });

        let fitted = ArdRegression::builder()
            .threshold_lambda(10_000.0)
            .build()
            .fit(&x, &y)
            .unwrap();
        // ARD should drive λ for the last two features → β ≈ 0.
        assert!(fitted.result.coefficients[2].abs() < 0.5);
        assert!(fitted.result.coefficients[3].abs() < 0.5);
    }
}
