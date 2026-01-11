//! Quantile Regression solver.
//!
//! Implements quantile regression using Iteratively Reweighted Least Squares (IRLS)
//! with the asymmetric Huber loss approximation for numerical stability.
//!
//! Quantile regression estimates conditional quantiles of the response variable,
//! providing a more complete picture of the conditional distribution than OLS,
//! which only estimates the conditional mean.
//!
//! # Algorithm
//!
//! The algorithm minimizes the asymmetric loss function:
//! L(u) = Σ ρ_τ(y_i - x_i'β)
//!
//! where ρ_τ(u) = u(τ - I(u < 0)) is the check function.
//!
//! We use IRLS with smoothed weights to handle the non-differentiability at zero:
//! w_i = τ / max(ε, |r_i|) if r_i >= 0
//! w_i = (1-τ) / max(ε, |r_i|) if r_i < 0
//!
//! # References
//!
//! - Koenker, R., & Bassett, G. (1978). Regression quantiles. Econometrica, 33-50.
//! - Validated against R's `quantreg` package: <https://cran.r-project.org/package=quantreg>

use crate::core::{IntervalType, PredictionResult, RegressionResult};
use crate::solvers::traits::{FittedRegressor, RegressionError, Regressor};
use faer::{Col, Mat};

/// Quantile Regression estimator.
///
/// Estimates conditional quantiles of the response variable distribution.
///
/// # Example
///
/// ```rust,ignore
/// use statistics::solvers::{QuantileRegressor, Regressor, FittedRegressor};
/// use faer::{Mat, Col};
///
/// let x = Mat::from_fn(100, 2, |i, j| (i + j) as f64);
/// let y = Col::from_fn(100, |i| 1.0 + 2.0 * i as f64 + rand::random::<f64>());
///
/// // Fit median regression (tau = 0.5)
/// let fitted = QuantileRegressor::builder()
///     .tau(0.5)
///     .build()
///     .fit(&x, &y)?;
///
/// println!("Coefficients: {:?}", fitted.coefficients());
/// ```
#[derive(Debug, Clone)]
pub struct QuantileRegressor {
    /// Quantile to estimate (0 < tau < 1)
    tau: f64,
    /// Whether to include an intercept term
    with_intercept: bool,
    /// Maximum iterations for IRLS
    max_iterations: usize,
    /// Convergence tolerance
    tolerance: f64,
    /// Smoothing parameter for weights
    epsilon: f64,
    /// Optional observation weights
    weights: Option<Col<f64>>,
}

impl QuantileRegressor {
    /// Create a new quantile regressor with the given quantile.
    ///
    /// # Arguments
    /// * `tau` - The quantile to estimate (must be between 0 and 1)
    pub fn new(tau: f64) -> Self {
        Self {
            tau,
            with_intercept: true,
            max_iterations: 100,
            tolerance: 1e-6,
            epsilon: 1e-6,
            weights: None,
        }
    }

    /// Create a builder for configuring the regressor.
    pub fn builder() -> QuantileRegressorBuilder {
        QuantileRegressorBuilder::default()
    }

    /// Fit the model using IRLS algorithm.
    fn fit_irls(
        &self,
        x: &Mat<f64>,
        y: &Col<f64>,
    ) -> Result<(Col<f64>, Option<f64>), RegressionError> {
        let n = x.nrows();
        let p = x.ncols();

        // Add intercept column if needed
        let x_aug = if self.with_intercept {
            let mut x_new = Mat::zeros(n, p + 1);
            for i in 0..n {
                x_new[(i, 0)] = 1.0;
                for j in 0..p {
                    x_new[(i, j + 1)] = x[(i, j)];
                }
            }
            x_new
        } else {
            x.clone()
        };
        let p_aug = x_aug.ncols();

        // Initialize with OLS solution
        let mut beta = self.ols_solve(&x_aug, y)?;

        // IRLS iterations
        for _iter in 0..self.max_iterations {
            // Compute residuals
            let mut residuals = Col::zeros(n);
            for i in 0..n {
                let mut pred = 0.0;
                for j in 0..p_aug {
                    pred += x_aug[(i, j)] * beta[j];
                }
                residuals[i] = y[i] - pred;
            }

            // Compute asymmetric weights
            let mut w = Col::zeros(n);
            for i in 0..n {
                let abs_r = residuals[i].abs().max(self.epsilon);
                if residuals[i] >= 0.0 {
                    w[i] = self.tau / abs_r;
                } else {
                    w[i] = (1.0 - self.tau) / abs_r;
                }

                // Apply observation weights if provided
                if let Some(ref obs_weights) = self.weights {
                    w[i] *= obs_weights[i];
                }
            }

            // Solve weighted least squares: (X'WX)^{-1} X'Wy
            let beta_new = self.wls_solve(&x_aug, y, &w)?;

            // Check convergence
            let mut max_change = 0.0f64;
            for j in 0..p_aug {
                let change = (beta_new[j] - beta[j]).abs();
                max_change = max_change.max(change);
            }

            beta = beta_new;

            if max_change < self.tolerance {
                break;
            }
        }

        // Extract intercept and coefficients
        if self.with_intercept {
            let intercept = beta[0];
            let mut coefficients = Col::zeros(p);
            for j in 0..p {
                coefficients[j] = beta[j + 1];
            }
            Ok((coefficients, Some(intercept)))
        } else {
            Ok((beta, None))
        }
    }

    /// Solve OLS using QR decomposition.
    fn ols_solve(&self, x: &Mat<f64>, y: &Col<f64>) -> Result<Col<f64>, RegressionError> {
        let n = x.nrows();
        let p = x.ncols();

        // Compute X'X
        let mut xtx = Mat::zeros(p, p);
        for i in 0..p {
            for j in 0..p {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += x[(k, i)] * x[(k, j)];
                }
                xtx[(i, j)] = sum;
            }
        }

        // Compute X'y
        let mut xty = Col::zeros(p);
        for j in 0..p {
            let mut sum = 0.0;
            for i in 0..n {
                sum += x[(i, j)] * y[i];
            }
            xty[j] = sum;
        }

        // Solve using Cholesky decomposition
        self.solve_symmetric(&xtx, &xty)
    }

    /// Solve weighted least squares.
    fn wls_solve(
        &self,
        x: &Mat<f64>,
        y: &Col<f64>,
        w: &Col<f64>,
    ) -> Result<Col<f64>, RegressionError> {
        let n = x.nrows();
        let p = x.ncols();

        // Compute X'WX
        let mut xtwx = Mat::zeros(p, p);
        for i in 0..p {
            for j in 0..p {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += x[(k, i)] * w[k] * x[(k, j)];
                }
                xtwx[(i, j)] = sum;
            }
        }

        // Compute X'Wy
        let mut xtwy = Col::zeros(p);
        for j in 0..p {
            let mut sum = 0.0;
            for i in 0..n {
                sum += x[(i, j)] * w[i] * y[i];
            }
            xtwy[j] = sum;
        }

        // Solve using Cholesky decomposition
        self.solve_symmetric(&xtwx, &xtwy)
    }

    /// Solve symmetric positive definite system Ax = b using Cholesky decomposition.
    fn solve_symmetric(&self, a: &Mat<f64>, b: &Col<f64>) -> Result<Col<f64>, RegressionError> {
        let n = a.nrows();

        // Cholesky decomposition: A = LL'
        let mut l: Mat<f64> = Mat::zeros(n, n);
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..j {
                sum += l[(j, k)].powi(2);
            }
            let diag = a[(j, j)] - sum;
            if diag <= 0.0 {
                // Add small regularization if needed
                l[(j, j)] = (a[(j, j)] + 1e-10).sqrt();
            } else {
                l[(j, j)] = diag.sqrt();
            }

            for i in (j + 1)..n {
                let mut sum = 0.0;
                for k in 0..j {
                    sum += l[(i, k)] * l[(j, k)];
                }
                l[(i, j)] = (a[(i, j)] - sum) / l[(j, j)];
            }
        }

        // Forward substitution: Ly = b
        let mut y_sol = Col::zeros(n);
        for i in 0..n {
            let mut sum = b[i];
            for j in 0..i {
                sum -= l[(i, j)] * y_sol[j];
            }
            y_sol[i] = sum / l[(i, i)];
        }

        // Backward substitution: L'x = y
        let mut x = Col::zeros(n);
        for i in (0..n).rev() {
            let mut sum = y_sol[i];
            for j in (i + 1)..n {
                sum -= l[(j, i)] * x[j];
            }
            x[i] = sum / l[(i, i)];
        }

        Ok(x)
    }

    /// Compute the check function (quantile loss).
    fn check_loss(&self, residuals: &Col<f64>) -> f64 {
        let mut loss = 0.0;
        for i in 0..residuals.nrows() {
            let r = residuals[i];
            if r >= 0.0 {
                loss += self.tau * r;
            } else {
                loss += (self.tau - 1.0) * r;
            }
        }
        loss
    }
}

impl Regressor for QuantileRegressor {
    type Fitted = FittedQuantile;

    fn fit(&self, x: &Mat<f64>, y: &Col<f64>) -> Result<Self::Fitted, RegressionError> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        // Validate dimensions
        if x.nrows() != y.nrows() {
            return Err(RegressionError::DimensionMismatch {
                x_rows: x.nrows(),
                y_len: y.nrows(),
            });
        }

        // Need at least p+1 observations
        let min_obs = if self.with_intercept {
            n_features + 2
        } else {
            n_features + 1
        };
        if n_samples < min_obs {
            return Err(RegressionError::InsufficientObservations {
                needed: min_obs,
                got: n_samples,
            });
        }

        // Validate tau
        if self.tau <= 0.0 || self.tau >= 1.0 {
            return Err(RegressionError::NumericalError(format!(
                "tau must be between 0 and 1, got {}",
                self.tau
            )));
        }

        // Validate weights if provided
        if let Some(ref w) = self.weights {
            if w.nrows() != n_samples {
                return Err(RegressionError::DimensionMismatch {
                    x_rows: n_samples,
                    y_len: w.nrows(),
                });
            }
            if w.iter().any(|&wi| wi < 0.0) {
                return Err(RegressionError::InvalidWeights);
            }
        }

        // Fit using IRLS
        let (coefficients, intercept) = self.fit_irls(x, y)?;

        // Compute fitted values and residuals
        let mut fitted_values = Col::zeros(n_samples);
        let mut residuals = Col::zeros(n_samples);

        for i in 0..n_samples {
            let mut pred = intercept.unwrap_or(0.0);
            for j in 0..n_features {
                pred += x[(i, j)] * coefficients[j];
            }
            fitted_values[i] = pred;
            residuals[i] = y[i] - pred;
        }

        // Compute pseudo R-squared (Koenker-Machado)
        // R1 = 1 - V(full) / V(null) where V is the quantile loss
        let full_loss = self.check_loss(&residuals);

        // Null model: just the quantile of y
        let y_quantile = self.compute_quantile(y);
        let null_residuals = Col::from_fn(n_samples, |i| y[i] - y_quantile);
        let null_loss = self.check_loss(&null_residuals);

        let pseudo_r_squared = if null_loss > 0.0 {
            1.0 - full_loss / null_loss
        } else {
            0.0
        };

        // Build result
        let n_params = if intercept.is_some() {
            n_features + 1
        } else {
            n_features
        };

        let mut result = RegressionResult::empty(n_features, n_samples);
        result.coefficients = coefficients.clone();
        result.intercept = intercept;
        result.residuals = residuals;
        result.fitted_values = fitted_values;
        result.n_parameters = n_params;
        result.n_observations = n_samples;
        result.r_squared = pseudo_r_squared;
        result.adj_r_squared = f64::NAN; // Not well-defined for quantile regression
        result.aliased = vec![false; n_features];

        Ok(FittedQuantile {
            tau: self.tau,
            with_intercept: self.with_intercept,
            result,
            check_loss: full_loss,
        })
    }
}

impl QuantileRegressor {
    /// Compute the tau-th sample quantile of y.
    fn compute_quantile(&self, y: &Col<f64>) -> f64 {
        let mut sorted: Vec<f64> = y.iter().copied().collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n = sorted.len();
        let index = (self.tau * (n - 1) as f64).floor() as usize;
        let frac = self.tau * (n - 1) as f64 - index as f64;

        if index + 1 < n {
            sorted[index] * (1.0 - frac) + sorted[index + 1] * frac
        } else {
            sorted[index]
        }
    }
}

/// A fitted Quantile Regression model.
#[derive(Debug, Clone)]
pub struct FittedQuantile {
    /// The quantile that was estimated
    tau: f64,
    /// Whether model includes intercept
    #[allow(dead_code)]
    with_intercept: bool,
    /// Regression result
    result: RegressionResult,
    /// Check function loss (quantile loss)
    check_loss: f64,
}

impl FittedQuantile {
    /// Get the quantile (tau) that was estimated.
    pub fn tau(&self) -> f64 {
        self.tau
    }

    /// Get the check function loss (quantile loss).
    pub fn check_loss(&self) -> f64 {
        self.check_loss
    }

    /// Get the pseudo R-squared (Koenker-Machado).
    pub fn pseudo_r_squared(&self) -> f64 {
        self.result.r_squared
    }
}

impl FittedRegressor for FittedQuantile {
    fn predict(&self, x: &Mat<f64>) -> Col<f64> {
        let n_samples = x.nrows();
        let n_features = x.ncols();
        let mut predictions = Col::zeros(n_samples);

        let intercept = self.result.intercept.unwrap_or(0.0);

        for i in 0..n_samples {
            let mut pred = intercept;
            for j in 0..n_features {
                pred += x[(i, j)] * self.result.coefficients[j];
            }
            predictions[i] = pred;
        }

        predictions
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
        let predictions = self.predict(x);

        match interval {
            None => PredictionResult::point_only(predictions),
            Some(_interval_type) => {
                // Quantile regression intervals require bootstrap or rank-based methods
                // For now, return NaN intervals
                let n = x.nrows();
                let mut lower = Col::zeros(n);
                let mut upper = Col::zeros(n);
                let mut se = Col::zeros(n);
                for i in 0..n {
                    lower[i] = f64::NAN;
                    upper[i] = f64::NAN;
                    se[i] = f64::NAN;
                }
                let _ = level;
                PredictionResult::with_intervals(predictions, lower, upper, se)
            }
        }
    }
}

/// Builder for `QuantileRegressor`.
#[derive(Debug, Clone)]
pub struct QuantileRegressorBuilder {
    tau: f64,
    with_intercept: bool,
    max_iterations: usize,
    tolerance: f64,
    epsilon: f64,
    weights: Option<Col<f64>>,
}

impl Default for QuantileRegressorBuilder {
    fn default() -> Self {
        Self {
            tau: 0.5,
            with_intercept: true,
            max_iterations: 100,
            tolerance: 1e-6,
            epsilon: 1e-6,
            weights: None,
        }
    }
}

impl QuantileRegressorBuilder {
    /// Create a new builder with default options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the quantile to estimate.
    ///
    /// Default is 0.5 (median). Must be between 0 and 1.
    pub fn tau(mut self, tau: f64) -> Self {
        self.tau = tau;
        self
    }

    /// Set whether to include an intercept term.
    ///
    /// Default is true.
    pub fn with_intercept(mut self, include: bool) -> Self {
        self.with_intercept = include;
        self
    }

    /// Set the maximum number of IRLS iterations.
    ///
    /// Default is 100.
    pub fn max_iterations(mut self, max_iter: usize) -> Self {
        self.max_iterations = max_iter;
        self
    }

    /// Set the convergence tolerance.
    ///
    /// Default is 1e-6.
    pub fn tolerance(mut self, tol: f64) -> Self {
        self.tolerance = tol;
        self
    }

    /// Set the smoothing parameter for weight computation.
    ///
    /// Default is 1e-6. Smaller values give more exact quantile regression
    /// but may cause numerical instability.
    pub fn epsilon(mut self, eps: f64) -> Self {
        self.epsilon = eps;
        self
    }

    /// Set observation weights.
    pub fn weights(mut self, w: Col<f64>) -> Self {
        self.weights = Some(w);
        self
    }

    /// Build the quantile regressor.
    pub fn build(self) -> QuantileRegressor {
        QuantileRegressor {
            tau: self.tau,
            with_intercept: self.with_intercept,
            max_iterations: self.max_iterations,
            tolerance: self.tolerance,
            epsilon: self.epsilon,
            weights: self.weights,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_median_regression() {
        // Simple linear relationship
        let x = Mat::from_fn(50, 1, |i, _| i as f64);
        let y = Col::from_fn(50, |i| 2.0 + 1.5 * i as f64);

        let model = QuantileRegressor::builder().tau(0.5).build();
        let fitted = model.fit(&x, &y).expect("model should fit");

        // Coefficients should be close to [1.5] with intercept ~2.0
        assert!((fitted.intercept().unwrap() - 2.0).abs() < 0.5);
        assert!((fitted.coefficients()[0] - 1.5).abs() < 0.1);
    }

    #[test]
    fn test_different_quantiles() {
        let x = Mat::from_fn(100, 1, |i, _| i as f64);
        let y = Col::from_fn(100, |i| 1.0 + 2.0 * i as f64);

        // Lower quantile
        let model_25 = QuantileRegressor::builder().tau(0.25).build();
        let fitted_25 = model_25.fit(&x, &y).expect("model should fit");

        // Upper quantile
        let model_75 = QuantileRegressor::builder().tau(0.75).build();
        let fitted_75 = model_75.fit(&x, &y).expect("model should fit");

        // Both should have similar slopes (data is homoscedastic)
        assert!((fitted_25.coefficients()[0] - 2.0).abs() < 0.2);
        assert!((fitted_75.coefficients()[0] - 2.0).abs() < 0.2);
    }

    #[test]
    fn test_no_intercept() {
        let x = Mat::from_fn(50, 1, |i, _| (i + 1) as f64);
        let y = Col::from_fn(50, |i| 2.5 * (i + 1) as f64);

        let model = QuantileRegressor::builder()
            .tau(0.5)
            .with_intercept(false)
            .build();
        let fitted = model.fit(&x, &y).expect("model should fit");

        assert!(fitted.intercept().is_none());
        assert!((fitted.coefficients()[0] - 2.5).abs() < 0.1);
    }

    #[test]
    fn test_predict() {
        let x = Mat::from_fn(50, 1, |i, _| i as f64);
        let y = Col::from_fn(50, |i| 1.0 + 2.0 * i as f64);

        let model = QuantileRegressor::builder().tau(0.5).build();
        let fitted = model.fit(&x, &y).expect("model should fit");

        let x_new = Mat::from_fn(5, 1, |i, _| (i + 100) as f64);
        let preds = fitted.predict(&x_new);

        // Check predictions
        for i in 0..5 {
            let expected = 1.0 + 2.0 * (i + 100) as f64;
            assert!((preds[i] - expected).abs() < 1.0);
        }
    }

    #[test]
    fn test_invalid_tau() {
        let x = Mat::from_fn(20, 1, |i, _| i as f64);
        let y = Col::from_fn(20, |i| i as f64);

        let model = QuantileRegressor::builder().tau(1.5).build();
        let result = model.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_dimension_mismatch() {
        let x = Mat::from_fn(10, 2, |i, j| (i + j) as f64);
        let y = Col::from_fn(5, |i| i as f64);

        let model = QuantileRegressor::builder().build();
        let result = model.fit(&x, &y);
        assert!(result.is_err());
    }
}
