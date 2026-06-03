//! Least Angle Regression (LARS) and its Lasso variant (LassoLars).
//!
//! Computes the full regularization path in `O(n * p * min(n, p))` time —
//! the same cost as a single OLS fit — making it competitive with
//! coordinate descent for problems with many features. Implements the
//! algorithm of Efron, Hastie, Johnstone & Tibshirani (2004), with the
//! Lasso modification of dropping a predictor from the active set as soon
//! as its coefficient crosses zero.
//!
//! Reference: Efron, B., Hastie, T., Johnstone, I., & Tibshirani, R.
//! (2004). "Least Angle Regression". Annals of Statistics 32 (2): 407-499.
//!
//! Matches scikit-learn's `lars_path` (with `method='lar'` or
//! `method='lasso'`) and the convenience estimators
//! `sklearn.linear_model.Lars` and `LassoLars`.
//!
//! # Example
//!
//! ```rust,ignore
//! use anofox_regression::solvers::{LarsRegressor, LarsMethod, Regressor, FittedRegressor};
//!
//! let fitted = LarsRegressor::builder()
//!     .method(LarsMethod::Lasso)
//!     .alpha(0.1)
//!     .build()
//!     .fit(&x, &y)?;
//! ```

#![allow(clippy::needless_range_loop)]

use crate::core::{IntervalType, PredictionResult, RegressionResult};
use crate::solvers::traits::{FittedRegressor, RegressionError, Regressor};
use faer::prelude::Solve;
use faer::{Col, Mat};

/// LARS variant: plain LARS (greedy forward stagewise approximation) or
/// the Lasso modification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LarsMethod {
    /// Plain LARS — variables only enter the active set.
    #[default]
    Lar,
    /// LassoLars — variables can also leave the active set when their
    /// coefficient passes through zero, reproducing the exact Lasso path.
    Lasso,
}

/// Least Angle Regression estimator.
#[derive(Debug, Clone)]
pub struct LarsRegressor {
    method: LarsMethod,
    fit_intercept: bool,
    /// Stop when this many active predictors are reached (`None` ⇒ run to
    /// `min(n - intercept, p)`).
    n_nonzero_coefs: Option<usize>,
    /// LassoLars-only stop: terminate as soon as the path reaches an alpha
    /// at or below this value. Plain LARS ignores this.
    alpha: f64,
    /// Whether to standardise predictors to unit norm before fitting (the
    /// LARS literature assumes this). The output coefficients are
    /// transformed back to the original predictor scale.
    standardize: bool,
    eps: f64,
}

impl Default for LarsRegressor {
    fn default() -> Self {
        Self {
            method: LarsMethod::Lar,
            fit_intercept: true,
            n_nonzero_coefs: None,
            alpha: 0.0,
            standardize: false,
            eps: f64::EPSILON,
        }
    }
}

impl LarsRegressor {
    pub fn builder() -> LarsRegressorBuilder {
        LarsRegressorBuilder::default()
    }
}

impl Regressor for LarsRegressor {
    type Fitted = FittedLars;

    fn fit(&self, x: &Mat<f64>, y: &Col<f64>) -> Result<Self::Fitted, RegressionError> {
        let n = x.nrows();
        let p = x.ncols();
        if n != y.nrows() {
            return Err(RegressionError::DimensionMismatch {
                x_rows: n,
                y_len: y.nrows(),
            });
        }

        // 1. Centering + (optional) column standardisation. sklearn's
        // lars_path always centers when fit_intercept=True; column-norm
        // normalisation is controlled by `Xy=None & normalize=True/False`.
        let (xs, x_mean, x_norm, ys, y_mean) =
            preprocess(x, y, self.fit_intercept, self.standardize);

        // 2. Path computation.
        let max_active = self.n_nonzero_coefs.unwrap_or(p.min(n.saturating_sub(1)));
        let max_active = max_active.min(p);
        let path = lars_path(&xs, &ys, self.method, self.alpha, max_active, self.eps)?;
        let alphas = path.alphas.clone();
        let coefs_path = path.coefs.clone();

        // For LassoLars: linearly interpolate the path so the returned
        // coefficients correspond to exactly the requested alpha. (sklearn
        // does the same.) For plain LARS or alpha == 0 we just take the
        // last path step.
        let beta_scaled =
            if self.method == LarsMethod::Lasso && self.alpha > 0.0 && coefs_path.len() >= 2 {
                interpolate_lasso_path(&coefs_path, &alphas, self.alpha)
            } else {
                path.last_coefs()
            };

        // 3. Undo the column scaling so coefficients live on the original
        // input scale.
        let mut beta = vec![0.0_f64; p];
        for j in 0..p {
            beta[j] = beta_scaled[j] / x_norm[j];
        }

        let intercept = if self.fit_intercept {
            let mut acc = y_mean;
            for j in 0..p {
                acc -= beta[j] * x_mean[j];
            }
            Some(acc)
        } else {
            None
        };

        let coefficients = Col::from_fn(p, |i| beta[i]);
        let fitted_values = Col::from_fn(n, |i| {
            let mut v = intercept.unwrap_or(0.0);
            for j in 0..p {
                v += x[(i, j)] * beta[j];
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

        Ok(FittedLars {
            result,
            alphas,
            coefs_path,
            x_mean,
            x_norm,
            y_mean,
            method: self.method,
            fit_intercept: self.fit_intercept,
        })
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

fn preprocess(
    x: &Mat<f64>,
    y: &Col<f64>,
    fit_intercept: bool,
    standardize: bool,
) -> (Mat<f64>, Vec<f64>, Vec<f64>, Col<f64>, f64) {
    let n = x.nrows();
    let p = x.ncols();
    let mut x_mean = vec![0.0_f64; p];
    if fit_intercept {
        for j in 0..p {
            let mut s = 0.0;
            for i in 0..n {
                s += x[(i, j)];
            }
            x_mean[j] = s / n as f64;
        }
    }
    let y_mean = if fit_intercept {
        let s: f64 = (0..n).map(|i| y[i]).sum();
        s / n as f64
    } else {
        0.0
    };
    let mut xs = Mat::from_fn(n, p, |i, j| x[(i, j)] - x_mean[j]);
    let ys = Col::from_fn(n, |i| y[i] - y_mean);
    let mut x_norm = vec![1.0_f64; p];
    if standardize {
        for j in 0..p {
            let mut s = 0.0;
            for i in 0..n {
                s += xs[(i, j)] * xs[(i, j)];
            }
            let nrm = s.sqrt();
            if nrm > 0.0 {
                x_norm[j] = nrm;
                for i in 0..n {
                    xs[(i, j)] /= nrm;
                }
            }
        }
    }
    (xs, x_mean, x_norm, ys, y_mean)
}

#[derive(Debug, Clone)]
struct LarsPath {
    /// Coefficients at each step of the path. `coefs[k][j]` is the
    /// coefficient of variable `j` after step `k`.
    coefs: Vec<Vec<f64>>,
    /// Maximum absolute correlation at each step (= alpha-on-the-path).
    alphas: Vec<f64>,
}

impl LarsPath {
    fn last_coefs(&self) -> Vec<f64> {
        self.coefs.last().cloned().expect("path is empty")
    }
}

fn interpolate_lasso_path(coefs: &[Vec<f64>], alphas: &[f64], target: f64) -> Vec<f64> {
    let p = coefs[0].len();
    // alphas decrease monotonically along the path. Find consecutive
    // (alpha_k, alpha_{k+1}) bracketing target.
    let n = alphas.len();
    if target >= alphas[0] {
        return coefs[0].clone();
    }
    if target <= alphas[n - 1] {
        return coefs[n - 1].clone();
    }
    for k in 0..(n - 1) {
        let a_hi = alphas[k];
        let a_lo = alphas[k + 1];
        if a_lo <= target && target <= a_hi {
            let denom = a_hi - a_lo;
            if denom.abs() < f64::EPSILON {
                return coefs[k + 1].clone();
            }
            let t = (a_hi - target) / denom;
            let mut out = vec![0.0; p];
            for j in 0..p {
                out[j] = (1.0 - t) * coefs[k][j] + t * coefs[k + 1][j];
            }
            return out;
        }
    }
    coefs[n - 1].clone()
}

fn lars_path(
    x: &Mat<f64>,
    y: &Col<f64>,
    method: LarsMethod,
    target_alpha: f64,
    max_active: usize,
    eps: f64,
) -> Result<LarsPath, RegressionError> {
    let n = x.nrows();
    let p = x.ncols();

    // c_j = X_j' (y - μ) — start with μ=0 so c_j = X_j' y.
    let mut c: Vec<f64> = (0..p)
        .map(|j| {
            let mut s = 0.0;
            for i in 0..n {
                s += x[(i, j)] * y[i];
            }
            s
        })
        .collect();
    let mut mu = Col::<f64>::zeros(n);
    let mut beta = vec![0.0_f64; p];
    let mut active: Vec<usize> = Vec::with_capacity(p);
    let mut sign: Vec<f64> = vec![0.0_f64; p];

    let mut path = LarsPath {
        coefs: vec![beta.clone()],
        alphas: vec![c.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()))],
    };

    let max_steps = (max_active * 4).max(2 * p);
    let mut step = 0;
    while step < max_steps {
        // Identify next candidate to enter (largest abs correlation).
        let mut best_j = usize::MAX;
        let mut best_c = 0.0_f64;
        for j in 0..p {
            if active.contains(&j) {
                continue;
            }
            let aj = c[j].abs();
            if aj > best_c {
                best_c = aj;
                best_j = j;
            }
        }
        if best_j == usize::MAX || best_c < eps {
            break;
        }
        // Add it (sign matches sign of correlation).
        active.push(best_j);
        sign[best_j] = c[best_j].signum();

        // Build X_A (with signed columns) and solve for w_A.
        let k = active.len();
        let xa = Mat::from_fn(n, k, |i, jj| sign[active[jj]] * x[(i, active[jj])]);
        let g = xa.transpose() * &xa;

        // Solve G_A * inv_g_1 = 1_A.
        let ones = Col::from_fn(k, |_| 1.0);
        let g_lu = g.partial_piv_lu();
        let inv_g_1 = g_lu.solve(&ones);
        let s_inv_g_1: f64 = (0..k).map(|i| inv_g_1[i]).sum();
        if !s_inv_g_1.is_finite() || s_inv_g_1 <= 0.0 {
            return Err(RegressionError::SingularMatrix);
        }
        let a_a = 1.0 / s_inv_g_1.sqrt();
        let w_a: Vec<f64> = (0..k).map(|i| a_a * inv_g_1[i]).collect();
        let u_a = &xa * Col::from_fn(k, |i| w_a[i]);

        // a = X' u_A
        let mut a = vec![0.0_f64; p];
        for j in 0..p {
            let mut s = 0.0;
            for i in 0..n {
                s += x[(i, j)] * u_a[i];
            }
            a[j] = s;
        }

        let cmax = best_c; // current max correlation magnitude

        // γ from incoming-variable conditions.
        let mut gamma = if active.len() < p {
            let mut g_min = f64::INFINITY;
            for j in 0..p {
                if active.contains(&j) {
                    continue;
                }
                let denom_plus = a_a - a[j];
                if denom_plus.abs() > eps {
                    let v = (cmax - c[j]) / denom_plus;
                    if v > eps && v < g_min {
                        g_min = v;
                    }
                }
                let denom_minus = a_a + a[j];
                if denom_minus.abs() > eps {
                    let v = (cmax + c[j]) / denom_minus;
                    if v > eps && v < g_min {
                        g_min = v;
                    }
                }
            }
            g_min
        } else {
            // All variables active → step until correlation reaches 0,
            // i.e. project residual onto u_A's coefficient.
            cmax / a_a
        };

        if !gamma.is_finite() {
            // All denominators degenerate.
            break;
        }

        // For LassoLars also check the drop condition: any active β
        // crossing zero.
        let mut drop_index: Option<usize> = None;
        if method == LarsMethod::Lasso {
            for (idx, &j) in active.iter().enumerate() {
                let dir = sign[j] * w_a[idx];
                if dir.abs() > eps {
                    let gj = -beta[j] / dir;
                    if gj > eps && gj < gamma {
                        gamma = gj;
                        drop_index = Some(idx);
                    }
                }
            }
        }

        // Update.
        for (idx, &j) in active.iter().enumerate() {
            beta[j] += gamma * sign[j] * w_a[idx];
        }
        for i in 0..n {
            mu[i] += gamma * u_a[i];
        }
        for j in 0..p {
            c[j] -= gamma * a[j];
        }

        // Record path step.
        let alpha = c.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
        path.coefs.push(beta.clone());
        path.alphas.push(alpha);

        // Drop (for Lasso) — also revert that variable's β to exactly 0.
        if let Some(idx) = drop_index {
            let dropped = active.remove(idx);
            sign[dropped] = 0.0;
            beta[dropped] = 0.0;
            *path.coefs.last_mut().unwrap() = beta.clone();
        }

        step += 1;
        if active.len() >= max_active && drop_index.is_none() {
            break;
        }
        if method == LarsMethod::Lasso && target_alpha > 0.0 && alpha <= target_alpha {
            break;
        }
    }

    Ok(path)
}

/// A fitted LARS / LassoLars model.
#[derive(Debug, Clone)]
pub struct FittedLars {
    result: RegressionResult,
    alphas: Vec<f64>,
    coefs_path: Vec<Vec<f64>>,
    /// Column means used to undo centering when generating the path on the
    /// original predictor scale. Kept for future prediction-interval support.
    #[allow(dead_code)]
    x_mean: Vec<f64>,
    x_norm: Vec<f64>,
    #[allow(dead_code)]
    y_mean: f64,
    method: LarsMethod,
    #[allow(dead_code)]
    fit_intercept: bool,
}

impl FittedLars {
    pub fn alphas(&self) -> &[f64] {
        &self.alphas
    }
    /// Path of coefficients, transformed back to the original predictor
    /// scale (one row per path step, length `p` per row).
    pub fn coefs_path(&self) -> Vec<Vec<f64>> {
        self.coefs_path
            .iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .map(|(j, v)| v / self.x_norm[j])
                    .collect()
            })
            .collect()
    }
    pub fn method(&self) -> LarsMethod {
        self.method
    }
}

impl FittedRegressor for FittedLars {
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

/// Builder for [`LarsRegressor`].
#[derive(Debug, Clone, Default)]
pub struct LarsRegressorBuilder {
    inner: LarsRegressor,
}

impl LarsRegressorBuilder {
    pub fn method(mut self, method: LarsMethod) -> Self {
        self.inner.method = method;
        self
    }
    pub fn fit_intercept(mut self, include: bool) -> Self {
        self.inner.fit_intercept = include;
        self
    }
    pub fn n_nonzero_coefs(mut self, value: usize) -> Self {
        self.inner.n_nonzero_coefs = Some(value);
        self
    }
    /// LassoLars target alpha (path terminates at this regularisation
    /// level). Plain LARS ignores this.
    pub fn alpha(mut self, value: f64) -> Self {
        self.inner.alpha = value;
        self
    }
    pub fn standardize(mut self, value: bool) -> Self {
        self.inner.standardize = value;
        self
    }
    pub fn build(self) -> LarsRegressor {
        self.inner
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lars_recovers_ols_on_full_path() {
        // Three linearly-independent features, plain LARS → coefficients
        // should equal OLS at the final step.
        let n = 50;
        let p = 3;
        // Deterministic but non-collinear features.
        let x = Mat::from_fn(n, p, |i, j| {
            let t = i as f64 / 5.0;
            match j {
                0 => t.sin(),
                1 => (t * 0.5).cos(),
                _ => (t * 0.7).exp().ln() + 0.1 * t, // ≈ 0.7t + 0.1t
            }
        });
        let true_beta = [1.0, -2.0, 0.5];
        let y = Col::from_fn(n, |i| {
            let mut s = 0.5;
            for j in 0..p {
                s += x[(i, j)] * true_beta[j];
            }
            s
        });

        let fitted = LarsRegressor::builder()
            .method(LarsMethod::Lar)
            .build()
            .fit(&x, &y)
            .unwrap();
        for j in 0..p {
            assert!(
                (fitted.result.coefficients[j] - true_beta[j]).abs() < 1e-6,
                "coef[{}] = {}",
                j,
                fitted.result.coefficients[j]
            );
        }
        assert!((fitted.result.intercept.unwrap() - 0.5).abs() < 1e-6);
    }
}
