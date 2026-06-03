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
//! Both estimators match `sklearn.linear_model.BayesianRidge` and
//! `sklearn.linear_model.ARDRegression` with default hyperparameters
//! (gamma-prior shapes `alpha_1 = alpha_2 = lambda_1 = lambda_2 = 1e-6`).
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
//! ```

#![allow(clippy::needless_range_loop)]

use crate::core::{IntervalType, PredictionResult, RegressionResult};
use crate::solvers::traits::{FittedRegressor, RegressionError, Regressor};
use faer::prelude::Solve;
use faer::{Col, Mat};

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

impl Regressor for BayesianRidge {
    type Fitted = FittedBayesianRidge;

    fn fit(&self, x: &Mat<f64>, y: &Col<f64>) -> Result<Self::Fitted, RegressionError> {
        let n = x.nrows();
        let p = x.ncols();
        if n != y.nrows() {
            return Err(RegressionError::DimensionMismatch {
                x_rows: n,
                y_len: y.nrows(),
            });
        }

        // Center X and y when fit_intercept=True.
        let mut x_mean = vec![0.0_f64; p];
        let mut y_mean = 0.0_f64;
        if self.fit_intercept {
            for j in 0..p {
                let mut s = 0.0;
                for i in 0..n {
                    s += x[(i, j)];
                }
                x_mean[j] = s / n as f64;
            }
            let s: f64 = (0..n).map(|i| y[i]).sum();
            y_mean = s / n as f64;
        }
        let xc = Mat::from_fn(n, p, |i, j| x[(i, j)] - x_mean[j]);
        let yc = Col::from_fn(n, |i| y[i] - y_mean);

        // SVD: xc = U diag(s) V'.
        let svd = xc.svd().map_err(|_| RegressionError::SingularMatrix)?;
        let u_mat = svd.U().to_owned();
        let v_mat = svd.V().to_owned();
        let s_col_view = svd.S().column_vector().to_owned();
        let k = s_col_view.nrows();
        let s_vec: Vec<f64> = (0..k).map(|i| s_col_view[i]).collect();
        let eig: Vec<f64> = s_vec.iter().map(|sv| sv * sv).collect();

        // Project y onto left singular vectors: u'y has length k.
        let uty_mat = u_mat.transpose() * &yc;
        let uty: Vec<f64> = (0..k).map(|i| uty_mat[i]).collect();
        let ssq_y: f64 = (0..n).map(|i| yc[i] * yc[i]).sum();

        let mut alpha = self.alpha_init.unwrap_or_else(|| {
            // sklearn init: 1 / var(y) (when fit_intercept=True, var of centered y).
            let var = ssq_y / n as f64;
            if var > 0.0 {
                1.0 / var
            } else {
                1.0
            }
        });
        let mut lambda = self.lambda_init.unwrap_or(1.0);

        let mut coef = vec![0.0_f64; p];
        let mut sigma_diag = vec![0.0_f64; p];

        for _it in 0..self.max_iter {
            // β = α V diag(1 / (α s² + λ)) U' y truncated to k singular
            // components.
            let coef_new = {
                let mut d = vec![0.0_f64; k];
                for i in 0..k {
                    let denom = alpha * eig[i] + lambda;
                    if denom > 0.0 {
                        d[i] = alpha * s_vec[i] * uty[i] / denom;
                    }
                }
                // β = V d
                let d_col = Col::from_fn(k, |i| d[i]);
                let beta_mat = &v_mat * d_col;
                let mut beta = vec![0.0_f64; p];
                for j in 0..p {
                    beta[j] = beta_mat[j];
                }
                beta
            };

            // Effective number of parameters γ.
            let mut gamma = 0.0_f64;
            for i in 0..k {
                let denom = alpha * eig[i] + lambda;
                if denom > 0.0 {
                    gamma += alpha * eig[i] / denom;
                }
            }

            // ||β||²
            let beta_sq: f64 = coef_new.iter().map(|b| b * b).sum();

            // residual ||y - Xβ||²
            let xb = &xc * Col::from_fn(p, |i| coef_new[i]);
            let mut rss = 0.0_f64;
            for i in 0..n {
                let r = yc[i] - xb[i];
                rss += r * r;
            }

            // Updates.
            let lambda_new = (gamma + 2.0 * self.lambda_1) / (beta_sq + 2.0 * self.lambda_2);
            let alpha_new = (n as f64 - gamma + 2.0 * self.alpha_1) / (rss + 2.0 * self.alpha_2);

            // Posterior covariance diagonal (used downstream for prediction
            // intervals): Σ_jj = sum_i v[j,i]² / (α s² + λ).
            for j in 0..p {
                let mut s = 0.0;
                for i in 0..k {
                    let denom = alpha_new * eig[i] + lambda_new;
                    if denom > 0.0 {
                        let vij = v_mat[(j, i)];
                        s += vij * vij / denom;
                    }
                }
                sigma_diag[j] = s;
            }

            // Convergence test (sklearn uses sum|β_new - β_old| < tol).
            let coef_change: f64 = coef_new
                .iter()
                .zip(coef.iter())
                .map(|(a, b)| (a - b).abs())
                .sum();
            coef = coef_new;
            alpha = alpha_new;
            lambda = lambda_new;
            if coef_change < self.tol {
                break;
            }
        }

        // Intercept = mean(y) - mean(X) · β
        let intercept = if self.fit_intercept {
            let mut b0 = y_mean;
            for j in 0..p {
                b0 -= x_mean[j] * coef[j];
            }
            Some(b0)
        } else {
            None
        };

        let coefficients = Col::from_fn(p, |i| coef[i]);
        let fitted_values = Col::from_fn(n, |i| {
            let mut v = intercept.unwrap_or(0.0);
            for j in 0..p {
                v += x[(i, j)] * coef[j];
            }
            v
        });
        let residuals = Col::from_fn(n, |i| y[i] - fitted_values[i]);

        let n_params = p + if self.fit_intercept { 1 } else { 0 };
        let mut result = RegressionResult::empty(p, n);
        result.coefficients = coefficients;
        result.intercept = intercept;
        result.residuals = residuals;
        result.fitted_values = fitted_values;
        result.rank = n_params;
        result.n_parameters = n_params;
        result.n_observations = n;
        result.r_squared = compute_r_squared(y, &result.residuals);

        Ok(FittedBayesianRidge {
            result,
            alpha,
            lambda,
            sigma_diag,
        })
    }
}

/// A fitted Bayesian Ridge model.
#[derive(Debug, Clone)]
pub struct FittedBayesianRidge {
    result: RegressionResult,
    alpha: f64,
    lambda: f64,
    sigma_diag: Vec<f64>,
}

impl FittedBayesianRidge {
    pub fn alpha(&self) -> f64 {
        self.alpha
    }
    pub fn lambda(&self) -> f64 {
        self.lambda
    }
    pub fn sigma_diag(&self) -> &[f64] {
        &self.sigma_diag
    }
}

impl FittedRegressor for FittedBayesianRidge {
    fn predict(&self, x: &Mat<f64>) -> Col<f64> {
        let n = x.nrows();
        let p = x.ncols();
        let intercept = self.result.intercept.unwrap_or(0.0);
        Col::from_fn(n, |i| {
            let mut v = intercept;
            for j in 0..p {
                v += x[(i, j)] * self.result.coefficients[j];
            }
            v
        })
    }
    fn result(&self) -> &RegressionResult {
        &self.result
    }
    fn predict_with_interval(
        &self,
        x: &Mat<f64>,
        _interval: Option<IntervalType>,
        _level: f64,
    ) -> PredictionResult {
        PredictionResult::point_only(self.predict(x))
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
    /// Features with `λ_j > threshold_lambda` are pruned (β_j set to 0).
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
}

impl Regressor for ArdRegression {
    type Fitted = FittedArd;

    fn fit(&self, x: &Mat<f64>, y: &Col<f64>) -> Result<Self::Fitted, RegressionError> {
        let n = x.nrows();
        let p = x.ncols();
        if n != y.nrows() {
            return Err(RegressionError::DimensionMismatch {
                x_rows: n,
                y_len: y.nrows(),
            });
        }

        let mut x_mean = vec![0.0_f64; p];
        let mut y_mean = 0.0_f64;
        if self.fit_intercept {
            for j in 0..p {
                let mut s = 0.0;
                for i in 0..n {
                    s += x[(i, j)];
                }
                x_mean[j] = s / n as f64;
            }
            let s: f64 = (0..n).map(|i| y[i]).sum();
            y_mean = s / n as f64;
        }
        let xc = Mat::from_fn(n, p, |i, j| x[(i, j)] - x_mean[j]);
        let yc = Col::from_fn(n, |i| y[i] - y_mean);

        let var_y: f64 = (0..n).map(|i| yc[i] * yc[i]).sum::<f64>() / n as f64;
        let mut alpha = if var_y > 0.0 { 1.0 / var_y } else { 1.0 };
        let mut lambdas = vec![1.0_f64; p];
        let mut coef = vec![0.0_f64; p];
        let mut kept: Vec<usize> = (0..p).collect();

        // Precompute XᵀX (full) and Xᵀy — we'll subset them by `kept`.
        let xtx_full = xc.transpose() * &xc;
        let xty_full_col = xc.transpose() * &yc;
        let xty_full: Vec<f64> = (0..p).map(|i| xty_full_col[i]).collect();

        for _it in 0..self.max_iter {
            // A = α XᵀX[kept,kept] + diag(λ[kept])
            let k = kept.len();
            if k == 0 {
                break;
            }
            let a = Mat::from_fn(k, k, |i, j| {
                let mut v = alpha * xtx_full[(kept[i], kept[j])];
                if i == j {
                    v += lambdas[kept[i]];
                }
                v
            });
            let llt = a
                .llt(faer::Side::Lower)
                .map_err(|_| RegressionError::SingularMatrix)?;

            // β[kept] = α A⁻¹ Xᵀy[kept]
            let rhs = Col::from_fn(k, |i| alpha * xty_full[kept[i]]);
            let beta_kept = llt.solve(&rhs);

            // Σ = A⁻¹ — diagonal via solving columns of I.
            let identity = Mat::<f64>::identity(k, k);
            let inv_a = llt.solve(&identity);

            // γ_j = 1 − λ_j Σ_jj
            let mut gamma_vec = vec![0.0_f64; k];
            for i in 0..k {
                let sigma_ii = inv_a[(i, i)];
                gamma_vec[i] = 1.0 - lambdas[kept[i]] * sigma_ii;
            }
            let gamma_sum: f64 = gamma_vec.iter().sum();

            // residual ||y - Xβ||²
            let mut beta_full = vec![0.0_f64; p];
            for (i, &j) in kept.iter().enumerate() {
                beta_full[j] = beta_kept[i];
            }
            let xb = &xc * Col::from_fn(p, |i| beta_full[i]);
            let mut rss = 0.0_f64;
            for i in 0..n {
                let r = yc[i] - xb[i];
                rss += r * r;
            }

            // λ_j updates
            for (i, &j) in kept.iter().enumerate() {
                let b = beta_kept[i];
                let denom = b * b + 2.0 * self.lambda_2;
                if denom > 0.0 {
                    lambdas[j] = (gamma_vec[i] + 2.0 * self.lambda_1) / denom;
                }
            }
            // α update
            let alpha_new =
                (n as f64 - gamma_sum + 2.0 * self.alpha_1) / (rss + 2.0 * self.alpha_2);

            // Convergence on coefficient change.
            let mut change: f64 = 0.0;
            for (i, &j) in kept.iter().enumerate() {
                change += (beta_kept[i] - coef[j]).abs();
                coef[j] = beta_kept[i];
            }
            // Reset pruned coefficients to 0 in `coef`.
            for j in 0..p {
                if !kept.contains(&j) {
                    coef[j] = 0.0;
                }
            }
            alpha = alpha_new;

            // Prune features with λ > threshold.
            let new_kept: Vec<usize> = kept
                .iter()
                .copied()
                .filter(|&j| lambdas[j] < self.threshold_lambda)
                .collect();
            kept = new_kept;

            if change < self.tol {
                break;
            }
        }

        let intercept = if self.fit_intercept {
            let mut b0 = y_mean;
            for j in 0..p {
                b0 -= x_mean[j] * coef[j];
            }
            Some(b0)
        } else {
            None
        };

        let coefficients = Col::from_fn(p, |i| coef[i]);
        let fitted_values = Col::from_fn(n, |i| {
            let mut v = intercept.unwrap_or(0.0);
            for j in 0..p {
                v += x[(i, j)] * coef[j];
            }
            v
        });
        let residuals = Col::from_fn(n, |i| y[i] - fitted_values[i]);

        let n_params = p + if self.fit_intercept { 1 } else { 0 };
        let mut result = RegressionResult::empty(p, n);
        result.coefficients = coefficients;
        result.intercept = intercept;
        result.residuals = residuals;
        result.fitted_values = fitted_values;
        result.rank = n_params;
        result.n_parameters = n_params;
        result.n_observations = n;
        result.r_squared = compute_r_squared(y, &result.residuals);

        Ok(FittedArd {
            result,
            alpha,
            lambdas,
        })
    }
}

/// A fitted ARD regression model.
#[derive(Debug, Clone)]
pub struct FittedArd {
    result: RegressionResult,
    alpha: f64,
    lambdas: Vec<f64>,
}

impl FittedArd {
    pub fn alpha(&self) -> f64 {
        self.alpha
    }
    pub fn lambdas(&self) -> &[f64] {
        &self.lambdas
    }
}

impl FittedRegressor for FittedArd {
    fn predict(&self, x: &Mat<f64>) -> Col<f64> {
        let n = x.nrows();
        let p = x.ncols();
        let intercept = self.result.intercept.unwrap_or(0.0);
        Col::from_fn(n, |i| {
            let mut v = intercept;
            for j in 0..p {
                v += x[(i, j)] * self.result.coefficients[j];
            }
            v
        })
    }
    fn result(&self) -> &RegressionResult {
        &self.result
    }
    fn predict_with_interval(
        &self,
        x: &Mat<f64>,
        _interval: Option<IntervalType>,
        _level: f64,
    ) -> PredictionResult {
        PredictionResult::point_only(self.predict(x))
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
