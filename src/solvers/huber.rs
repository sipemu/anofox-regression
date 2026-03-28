//! Huber M-estimator regression solver.
//!
//! Implements robust regression using Iteratively Reweighted Least Squares (IRLS)
//! with Huber's M-estimator. This approach is robust to outliers in the response
//! variable while remaining efficient for clean data.
//!
//! # Algorithm
//!
//! The Huber loss function is:
//! L(r) = r²/2               if |r| ≤ ε·σ
//! L(r) = ε·σ·|r| - (ε·σ)²/2  otherwise
//!
//! The IRLS weights are:
//! w_i = 1.0                 if |r_i / (ε·σ)| ≤ 1
//! w_i = 1.0 / |r_i / (ε·σ)|  otherwise
//!
//! Scale σ is estimated via MAD: median(|r_i|) / 0.6745.
//!
//! L2 regularization is applied via the `alpha` parameter.
//!
//! # References
//!
//! - Huber, P. J. (1964). Robust estimation of a location parameter. Ann. Math. Stat.
//! - Default epsilon of 1.35 matches sklearn's `HuberRegressor`.

use crate::core::{IntervalType, PredictionResult, RegressionResult};
use crate::solvers::traits::{FittedRegressor, RegressionError, Regressor};
use faer::{Col, Mat};
use statrs::distribution::{ContinuousCDF, FisherSnedecor};

/// Huber M-estimator regression.
///
/// Robust linear regression that downweights outliers using the Huber loss.
///
/// # Example
///
/// ```rust,ignore
/// use statistics::solvers::{HuberRegressor, Regressor, FittedRegressor};
/// use faer::{Mat, Col};
///
/// let x = Mat::from_fn(100, 2, |i, j| (i + j) as f64);
/// let y = Col::from_fn(100, |i| 1.0 + 2.0 * i as f64);
///
/// let fitted = HuberRegressor::builder()
///     .epsilon(1.35)
///     .build()
///     .fit(&x, &y)?;
///
/// println!("Coefficients: {:?}", fitted.coefficients());
/// println!("Outliers: {}", fitted.n_outliers());
/// ```
#[derive(Debug, Clone)]
pub struct HuberRegressor {
    /// Huber threshold parameter. Points with |r_i / sigma| > epsilon are
    /// downweighted. Default is 1.35 (matching sklearn).
    epsilon: f64,
    /// L2 regularization parameter. Default is 0.0001.
    alpha: f64,
    /// Whether to include an intercept term.
    with_intercept: bool,
    /// Maximum iterations for IRLS.
    max_iterations: usize,
    /// Convergence tolerance.
    tolerance: f64,
}

impl HuberRegressor {
    /// Create a new Huber regressor with default parameters.
    pub fn new() -> Self {
        Self {
            epsilon: 1.35,
            alpha: 0.0001,
            with_intercept: true,
            max_iterations: 100,
            tolerance: 1e-5,
        }
    }

    /// Create a builder for configuring the regressor.
    pub fn builder() -> HuberRegressorBuilder {
        HuberRegressorBuilder::default()
    }

    /// Fit the model using IRLS algorithm.
    fn fit_irls(
        &self,
        x: &Mat<f64>,
        y: &Col<f64>,
    ) -> Result<(Col<f64>, Option<f64>, f64), RegressionError> {
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

        let mut sigma = 1.0;

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

            // Estimate scale via MAD: median(|r_i|) / 0.6745
            sigma = self.estimate_scale(&residuals);

            // Compute Huber weights
            let mut w = Col::zeros(n);
            for i in 0..n {
                let scaled_r = (residuals[i] / sigma).abs();
                if scaled_r <= self.epsilon {
                    w[i] = 1.0;
                } else {
                    w[i] = self.epsilon / scaled_r;
                }
            }

            // Solve weighted ridge: (X'WX + alpha*I) * beta = X'Wy
            // where I does NOT penalize the intercept position
            let beta_new = self.wls_ridge_solve(&x_aug, y, &w)?;

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
            Ok((coefficients, Some(intercept), sigma))
        } else {
            Ok((beta, None, sigma))
        }
    }

    /// Estimate the scale (sigma) via MAD of residuals.
    fn estimate_scale(&self, residuals: &Col<f64>) -> f64 {
        let n = residuals.nrows();
        let mut abs_residuals: Vec<f64> = Vec::with_capacity(n);
        for i in 0..n {
            abs_residuals.push(residuals[i].abs());
        }
        abs_residuals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let median_abs = if n.is_multiple_of(2) {
            (abs_residuals[n / 2 - 1] + abs_residuals[n / 2]) / 2.0
        } else {
            abs_residuals[n / 2]
        };

        // MAD-based scale estimate with floor to avoid division by zero
        (median_abs / 0.6745).max(1e-10)
    }

    /// Solve OLS using Cholesky decomposition for initialization.
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

    /// Solve weighted ridge regression: (X'WX + alpha*I) * beta = X'Wy.
    ///
    /// The intercept position (index 0 when with_intercept=true) is NOT penalized.
    fn wls_ridge_solve(
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

        // Add alpha to diagonal (but not to intercept position)
        let start = if self.with_intercept { 1 } else { 0 };
        for j in start..p {
            xtwx[(j, j)] += self.alpha;
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

    /// Compute fit statistics (following ridge.rs pattern).
    #[allow(clippy::too_many_arguments)]
    fn compute_statistics(
        &self,
        y: &Col<f64>,
        coefficients: &Col<f64>,
        intercept: Option<f64>,
        residuals: &Col<f64>,
        fitted_values: &Col<f64>,
        aliased: &[bool],
        rank: usize,
        n_params: usize,
    ) -> RegressionResult {
        let n = y.nrows();
        let n_features = coefficients.nrows();

        // Compute y mean
        let y_mean: f64 = y.iter().sum::<f64>() / n as f64;

        // Compute TSS (total sum of squares)
        let tss: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();

        // Compute RSS (residual sum of squares)
        let rss: f64 = residuals.iter().map(|&r| r.powi(2)).sum();

        // R-squared
        let r_squared = if tss > 0.0 {
            (1.0 - rss / tss).clamp(0.0, 1.0)
        } else if rss < 1e-10 {
            1.0
        } else {
            0.0
        };

        // Adjusted R-squared
        let df_total = (n - 1) as f64;
        let df_resid = (n - n_params) as f64;
        let adj_r_squared = if df_resid > 0.0 && df_total > 0.0 {
            1.0 - (1.0 - r_squared) * df_total / df_resid
        } else {
            f64::NAN
        };

        // MSE and RMSE
        let mse = if df_resid > 0.0 {
            rss / df_resid
        } else {
            f64::NAN
        };
        let rmse = mse.sqrt();

        // F-statistic
        let ess = tss - rss;
        let df_model = (n_params - if intercept.is_some() { 1 } else { 0 }) as f64;
        let f_statistic = if df_model > 0.0 && df_resid > 0.0 && mse > 0.0 {
            (ess / df_model) / mse
        } else {
            f64::NAN
        };

        // F p-value
        let f_pvalue = if f_statistic.is_finite() && df_model > 0.0 && df_resid > 0.0 {
            let f_dist = FisherSnedecor::new(df_model, df_resid).ok();
            f_dist.map_or(f64::NAN, |d| 1.0 - d.cdf(f_statistic))
        } else {
            f64::NAN
        };

        // Information criteria
        let log_likelihood = if mse > 0.0 {
            -0.5 * n as f64 * (1.0 + (2.0 * std::f64::consts::PI).ln() + mse.ln())
        } else {
            f64::NAN
        };

        let k = n_params as f64;
        let aic = if log_likelihood.is_finite() {
            2.0 * k - 2.0 * log_likelihood
        } else {
            f64::NAN
        };

        let aicc = if log_likelihood.is_finite() && (n as f64 - k - 1.0) > 0.0 {
            aic + 2.0 * k * (k + 1.0) / (n as f64 - k - 1.0)
        } else {
            f64::NAN
        };

        let bic = if log_likelihood.is_finite() {
            k * (n as f64).ln() - 2.0 * log_likelihood
        } else {
            f64::NAN
        };

        let mut result = RegressionResult::empty(n_features, n);
        result.coefficients = coefficients.clone();
        result.intercept = intercept;
        result.residuals = residuals.clone();
        result.fitted_values = fitted_values.clone();
        result.rank = rank;
        result.n_parameters = n_params;
        result.n_observations = n;
        result.aliased = aliased.to_vec();
        result.r_squared = r_squared;
        result.adj_r_squared = adj_r_squared;
        result.mse = mse;
        result.rmse = rmse;
        result.f_statistic = f_statistic;
        result.f_pvalue = f_pvalue;
        result.aic = aic;
        result.aicc = aicc;
        result.bic = bic;
        result.log_likelihood = log_likelihood;
        result.confidence_level = 0.95;

        result
    }
}

impl Default for HuberRegressor {
    fn default() -> Self {
        Self::new()
    }
}

impl Regressor for HuberRegressor {
    type Fitted = FittedHuber;

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

        // Validate epsilon
        if self.epsilon <= 1.0 {
            return Err(RegressionError::NumericalError(format!(
                "epsilon must be greater than 1.0, got {}",
                self.epsilon
            )));
        }

        // Fit using IRLS
        let (coefficients, intercept, sigma) = self.fit_irls(x, y)?;

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

        // Build outlier mask: |r_i| > epsilon * scale
        let threshold = self.epsilon * sigma;
        let outlier_mask: Vec<bool> = (0..n_samples)
            .map(|i| residuals[i].abs() > threshold)
            .collect();

        // Compute statistics
        let n_params = if intercept.is_some() {
            n_features + 1
        } else {
            n_features
        };
        let aliased = vec![false; n_features];
        let rank = n_features;

        let result = self.compute_statistics(
            y,
            &coefficients,
            intercept,
            &residuals,
            &fitted_values,
            &aliased,
            rank,
            n_params,
        );

        Ok(FittedHuber {
            result,
            scale: sigma,
            epsilon: self.epsilon,
            outlier_mask,
        })
    }
}

/// A fitted Huber M-estimator regression model.
#[derive(Debug, Clone)]
pub struct FittedHuber {
    /// Regression result with fit statistics.
    result: RegressionResult,
    /// Estimated scale (sigma) from MAD.
    scale: f64,
    /// Huber threshold parameter used during fitting.
    epsilon: f64,
    /// Boolean mask: true if |r_i| > epsilon * scale (outlier).
    outlier_mask: Vec<bool>,
}

impl FittedHuber {
    /// Get the estimated scale (sigma) from the MAD estimator.
    pub fn scale(&self) -> f64 {
        self.scale
    }

    /// Get the outlier mask (true for observations with |r_i| > epsilon * scale).
    pub fn outliers(&self) -> &[bool] {
        &self.outlier_mask
    }

    /// Get the number of detected outliers.
    pub fn n_outliers(&self) -> usize {
        self.outlier_mask.iter().filter(|&&o| o).count()
    }

    /// Get the epsilon parameter used during fitting.
    pub fn epsilon(&self) -> f64 {
        self.epsilon
    }
}

impl FittedRegressor for FittedHuber {
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
                // Huber regression intervals are not standard; return NaN for now
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

/// Builder for `HuberRegressor`.
#[derive(Debug, Clone)]
pub struct HuberRegressorBuilder {
    epsilon: f64,
    alpha: f64,
    with_intercept: bool,
    max_iterations: usize,
    tolerance: f64,
}

impl Default for HuberRegressorBuilder {
    fn default() -> Self {
        Self {
            epsilon: 1.35,
            alpha: 0.0001,
            with_intercept: true,
            max_iterations: 100,
            tolerance: 1e-5,
        }
    }
}

impl HuberRegressorBuilder {
    /// Create a new builder with default options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the Huber threshold parameter (epsilon).
    ///
    /// Default is 1.35 (matching sklearn). Must be greater than 1.0.
    /// Smaller values make the estimator more robust but less efficient.
    pub fn epsilon(mut self, epsilon: f64) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Set the L2 regularization parameter (alpha).
    ///
    /// Default is 0.0001. Larger values shrink coefficients toward zero.
    pub fn alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
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
    /// Default is 1e-5.
    pub fn tolerance(mut self, tol: f64) -> Self {
        self.tolerance = tol;
        self
    }

    /// Build the Huber regressor.
    pub fn build(self) -> HuberRegressor {
        HuberRegressor {
            epsilon: self.epsilon,
            alpha: self.alpha,
            with_intercept: self.with_intercept,
            max_iterations: self.max_iterations,
            tolerance: self.tolerance,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_huber_defaults() {
        let builder = HuberRegressorBuilder::default();
        assert!((builder.epsilon - 1.35).abs() < 1e-10);
        assert!((builder.alpha - 0.0001).abs() < 1e-10);
        assert!(builder.with_intercept);
        assert_eq!(builder.max_iterations, 100);
        assert!((builder.tolerance - 1e-5).abs() < 1e-15);

        let model = HuberRegressor::builder().build();
        assert!((model.epsilon - 1.35).abs() < 1e-10);
        assert!((model.alpha - 0.0001).abs() < 1e-10);
    }

    #[test]
    fn test_huber_no_outliers() {
        // Clean linear data: y = 2.0 + 1.5*x
        let n = 100;
        let x = Mat::from_fn(n, 1, |i, _| i as f64);
        let y = Col::from_fn(n, |i| 2.0 + 1.5 * i as f64);

        let huber = HuberRegressor::builder().build();
        let fitted = huber.fit(&x, &y).expect("model should fit");

        // On clean data, Huber should closely match OLS
        let intercept = fitted.intercept().unwrap();
        let slope = fitted.coefficients()[0];

        assert!(
            (intercept - 2.0).abs() < 0.5,
            "intercept {intercept} should be close to 2.0"
        );
        assert!(
            (slope - 1.5).abs() < 0.1,
            "slope {slope} should be close to 1.5"
        );
        assert!(
            fitted.r_squared() > 0.99,
            "R² should be very high for clean data"
        );
    }

    #[test]
    fn test_huber_with_outliers() {
        // Linear data with outliers
        let n = 100;
        let x = Mat::from_fn(n, 1, |i, _| i as f64);
        let mut y_vec: Vec<f64> = (0..n).map(|i| 2.0 + 1.5 * i as f64).collect();

        // Contaminate ~10% with large outliers
        for i in [5, 15, 25, 35, 45, 55, 65, 75, 85, 95] {
            y_vec[i] += 500.0;
        }
        let y = Col::from_fn(n, |i| y_vec[i]);

        let huber = HuberRegressor::builder().build();
        let fitted = huber.fit(&x, &y).expect("model should fit");

        // Huber should still recover the slope reasonably well despite outliers
        let slope = fitted.coefficients()[0];
        assert!(
            (slope - 1.5).abs() < 2.0,
            "slope {slope} should be reasonable despite outliers"
        );

        // Should detect some outliers
        assert!(
            fitted.n_outliers() > 0,
            "should detect at least some outliers"
        );
    }

    #[test]
    fn test_huber_outlier_detection() {
        // Create data where some points are clearly outliers
        let n = 50;
        let x = Mat::from_fn(n, 1, |i, _| i as f64);
        let mut y_vec: Vec<f64> = (0..n).map(|i| 1.0 + 2.0 * i as f64).collect();

        // Add extreme outliers at known positions
        y_vec[10] += 1000.0;
        y_vec[20] += 1000.0;
        y_vec[30] += 1000.0;

        let y = Col::from_fn(n, |i| y_vec[i]);

        let huber = HuberRegressor::builder().build();
        let fitted = huber.fit(&x, &y).expect("model should fit");

        let outliers = fitted.outliers();
        assert_eq!(outliers.len(), n);

        // The three contaminated points should be flagged as outliers
        assert!(
            fitted.n_outliers() >= 3,
            "should detect at least the 3 injected outliers, got {}",
            fitted.n_outliers()
        );
    }

    #[test]
    fn test_huber_scale_estimation() {
        // Clean linear data: scale should be small
        let n = 100;
        let x = Mat::from_fn(n, 1, |i, _| i as f64);
        let y = Col::from_fn(n, |i| 2.0 + 1.5 * i as f64);

        let huber = HuberRegressor::builder().build();
        let fitted = huber.fit(&x, &y).expect("model should fit");

        // For exact linear data, the scale should be very small
        let scale = fitted.scale();
        assert!(
            scale < 1.0,
            "scale {scale} should be small for clean linear data"
        );
        assert!(scale > 0.0, "scale should be positive");
    }

    #[test]
    fn test_huber_regularization() {
        // Test that higher alpha shrinks coefficients
        let n = 50;
        let x = Mat::from_fn(n, 1, |i, _| (i + 1) as f64);
        let y = Col::from_fn(n, |i| 1.0 + 3.0 * (i + 1) as f64);

        let fitted_low = HuberRegressor::builder()
            .alpha(0.0001)
            .build()
            .fit(&x, &y)
            .expect("model should fit");

        let fitted_high = HuberRegressor::builder()
            .alpha(100.0)
            .build()
            .fit(&x, &y)
            .expect("model should fit");

        let coef_low = fitted_low.coefficients()[0].abs();
        let coef_high = fitted_high.coefficients()[0].abs();

        assert!(
            coef_high < coef_low,
            "higher alpha should shrink coefficients: |{}| should be < |{}|",
            coef_high,
            coef_low
        );
    }

    #[test]
    fn test_huber_predict() {
        let n = 50;
        let x = Mat::from_fn(n, 1, |i, _| i as f64);
        let y = Col::from_fn(n, |i| 1.0 + 2.0 * i as f64);

        let huber = HuberRegressor::builder().build();
        let fitted = huber.fit(&x, &y).expect("model should fit");

        let x_new = Mat::from_fn(5, 1, |i, _| (i + 100) as f64);
        let preds = fitted.predict(&x_new);

        // Predictions should be close to true values
        for i in 0..5 {
            let expected = 1.0 + 2.0 * (i + 100) as f64;
            assert!(
                (preds[i] - expected).abs() < 2.0,
                "prediction {} should be close to expected {}",
                preds[i],
                expected
            );
        }
    }

    #[test]
    fn test_huber_dimension_mismatch() {
        let x = Mat::from_fn(10, 2, |i, j| (i + j) as f64);
        let y = Col::from_fn(5, |i| i as f64);

        let model = HuberRegressor::builder().build();
        let result = model.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_huber_no_intercept() {
        let n = 50;
        let x = Mat::from_fn(n, 1, |i, _| (i + 1) as f64);
        let y = Col::from_fn(n, |i| 2.5 * (i + 1) as f64);

        let model = HuberRegressor::builder().with_intercept(false).build();
        let fitted = model.fit(&x, &y).expect("model should fit");

        assert!(fitted.intercept().is_none());
        assert!(
            (fitted.coefficients()[0] - 2.5).abs() < 0.2,
            "slope {} should be close to 2.5",
            fitted.coefficients()[0]
        );
    }
}
