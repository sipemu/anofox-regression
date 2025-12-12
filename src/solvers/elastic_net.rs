//! Elastic Net solver (combined L1 and L2 regularization).

use crate::core::{
    IntervalType, LambdaScaling, PredictionResult, RegressionOptions, RegressionOptionsBuilder,
    RegressionResult,
};
use crate::inference::{
    compute_prediction_intervals, compute_xtx_inverse_augmented_reduced,
    compute_xtx_inverse_reduced,
};
use crate::solvers::ridge::RidgeRegressor;
use crate::solvers::traits::{FittedRegressor, RegressionError, Regressor};
use crate::utils::{center_columns, center_vector, detect_constant_columns};
use argmin::core::{CostFunction, Executor, Gradient, State};
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::quasinewton::LBFGS;
use faer::{Col, Mat};
use statrs::distribution::{ContinuousCDF, FisherSnedecor};

/// Elastic Net regression estimator using L-BFGS optimization.
///
/// Minimizes: ||y - Xβ||² + λ(α||β||₁ + (1-α)||β||₂²)
///
/// Where:
/// - α = 1 is pure Lasso (L1)
/// - α = 0 is pure Ridge (L2)
/// - 0 < α < 1 is a mix of both
///
/// # Example
///
/// ```rust,ignore
/// use statistics::solvers::{ElasticNetRegressor, Regressor, FittedRegressor};
/// use faer::{Mat, Col};
///
/// let x = Mat::from_fn(100, 5, |i, j| (i + j) as f64);
/// let y = Col::from_fn(100, |i| 1.0 + 2.0 * i as f64);
///
/// let fitted = ElasticNetRegressor::builder()
///     .with_intercept(true)
///     .lambda(1.0)
///     .alpha(0.5)  // 50% L1, 50% L2
///     .build()
///     .fit(&x, &y)?;
///
/// println!("Non-zero coefficients: {:?}", fitted.coefficients());
/// ```
#[derive(Debug, Clone)]
pub struct ElasticNetRegressor {
    options: RegressionOptions,
}

impl ElasticNetRegressor {
    /// Create a new Elastic Net regressor with the given options.
    pub fn new(options: RegressionOptions) -> Self {
        Self { options }
    }

    /// Create a builder for configuring the regressor.
    pub fn builder() -> ElasticNetRegressorBuilder {
        ElasticNetRegressorBuilder::default()
    }

    /// Soft thresholding operator: S(z, γ) = sign(z) * max(|z| - γ, 0)
    fn soft_threshold(z: f64, gamma: f64) -> f64 {
        if z > gamma {
            z - gamma
        } else if z < -gamma {
            z + gamma
        } else {
            0.0
        }
    }
}

/// Cost function for Elastic Net optimization using argmin.
/// Computes the smooth part of the objective: ||y - Xβ||² + λ(1-α)||β||₂²
#[derive(Clone)]
struct ElasticNetCost<'a> {
    x: &'a Mat<f64>,
    y: &'a Col<f64>,
    lambda: f64,
    alpha: f64,
}

impl ElasticNetCost<'_> {
    /// Compute residuals: y - Xβ
    fn compute_residuals(&self, beta: &[f64]) -> Vec<f64> {
        let n = self.x.nrows();
        let p = self.x.ncols();
        (0..n)
            .map(|i| {
                let pred: f64 = (0..p).map(|j| self.x[(i, j)] * beta[j]).sum();
                self.y[i] - pred
            })
            .collect()
    }
}

impl CostFunction for ElasticNetCost<'_> {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, beta: &Self::Param) -> Result<Self::Output, argmin::core::Error> {
        // ||y - Xβ||² + λ(1-α)||β||₂²
        let residuals = self.compute_residuals(beta);
        let rss: f64 = residuals.iter().map(|r| r * r).sum();
        let l2_penalty: f64 = beta.iter().map(|b| b * b).sum();
        Ok(rss + self.lambda * (1.0 - self.alpha) * l2_penalty)
    }
}

impl Gradient for ElasticNetCost<'_> {
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;

    fn gradient(&self, beta: &Self::Param) -> Result<Self::Gradient, argmin::core::Error> {
        // ∇f = -2X'(y - Xβ) + 2λ(1-α)β
        let residuals = self.compute_residuals(beta);
        let p = self.x.ncols();
        let n = self.x.nrows();
        let grad: Vec<f64> = (0..p)
            .map(|j| {
                let xtresid: f64 = (0..n).map(|i| self.x[(i, j)] * residuals[i]).sum();
                -2.0 * xtresid + 2.0 * self.lambda * (1.0 - self.alpha) * beta[j]
            })
            .collect();
        Ok(grad)
    }
}

impl Regressor for ElasticNetRegressor {
    type Fitted = FittedElasticNet;

    fn fit(&self, x: &Mat<f64>, y: &Col<f64>) -> Result<Self::Fitted, RegressionError> {
        // When alpha = 0, delegate to Ridge
        if self.options.alpha == 0.0 {
            let ridge = RidgeRegressor::new(self.options.clone());
            let ridge_fitted = ridge.fit(x, y)?;
            // For pure Ridge, use the same aliased mask from result
            let aliased = ridge_fitted.result().aliased.clone();
            let xtx_inverse = compute_xtx_inverse_augmented_reduced(x, &aliased).ok();
            return Ok(FittedElasticNet {
                options: self.options.clone(),
                result: ridge_fitted.result().clone(),
                xtx_inverse,
                aliased,
            });
        }

        let n_samples = x.nrows();
        let n_features = x.ncols();

        // Validate dimensions
        if x.nrows() != y.nrows() {
            return Err(RegressionError::DimensionMismatch {
                x_rows: x.nrows(),
                y_len: y.nrows(),
            });
        }

        if n_samples < 2 {
            return Err(RegressionError::InsufficientObservations {
                needed: 2,
                got: n_samples,
            });
        }

        if self.options.with_intercept {
            // Detect constant columns (aliased with intercept)
            let constant_cols = detect_constant_columns(x, self.options.rank_tolerance);

            // Center the data
            let (x_centered, x_means) = center_columns(x);
            let (y_centered, y_mean) = center_vector(y);

            // Solve Elastic Net using L-BFGS
            let coefficients = self.fit_lbfgs(&x_centered, &y_centered)?;

            // Compute intercept: intercept = y_mean - x_means' * coefficients
            let mut intercept = y_mean;
            for j in 0..n_features {
                if !constant_cols[j] {
                    intercept -= x_means[j] * coefficients[j];
                }
            }

            // Mark aliased columns (constant or zero coefficient due to L1)
            let mut aliased = constant_cols.clone();
            for j in 0..n_features {
                if coefficients[j].abs() < 1e-10 {
                    aliased[j] = true;
                }
            }

            // Compute fitted values and residuals
            let mut fitted_values = Col::zeros(n_samples);
            let mut residuals = Col::zeros(n_samples);

            for i in 0..n_samples {
                let mut pred = intercept;
                for j in 0..n_features {
                    if !constant_cols[j] {
                        pred += x[(i, j)] * coefficients[j];
                    }
                }
                fitted_values[i] = pred;
                residuals[i] = y[i] - pred;
            }

            // Count non-zero coefficients for effective rank
            let n_nonzero = coefficients
                .iter()
                .enumerate()
                .filter(|(j, &c)| !constant_cols[*j] && c.abs() > 1e-10)
                .count();
            let n_params = n_nonzero + 1; // +1 for intercept

            let result = self.compute_statistics(
                x,
                y,
                &coefficients,
                Some(intercept),
                &residuals,
                &fitted_values,
                &constant_cols,
                n_nonzero,
                n_params,
            )?;

            // Compute (X_aug'X_aug)⁻¹ for prediction intervals (using non-aliased columns)
            // Note: This is approximate for L1-penalized models
            let xtx_inverse = compute_xtx_inverse_augmented_reduced(x, &constant_cols).ok();

            Ok(FittedElasticNet {
                options: self.options.clone(),
                result,
                xtx_inverse,
                aliased: constant_cols,
            })
        } else {
            // No intercept case - detect zero-variance columns
            let mut aliased = vec![false; n_features];
            for j in 0..n_features {
                let mut col_sq = 0.0;
                for i in 0..n_samples {
                    col_sq += x[(i, j)] * x[(i, j)];
                }
                if col_sq < self.options.rank_tolerance {
                    aliased[j] = true;
                }
            }

            let coefficients = self.fit_lbfgs(x, y)?;

            // Mark zero coefficients as aliased
            for j in 0..n_features {
                if coefficients[j].abs() < 1e-10 {
                    aliased[j] = true;
                }
            }

            // Compute fitted values and residuals
            let mut fitted_values = Col::zeros(n_samples);
            let mut residuals = Col::zeros(n_samples);

            for i in 0..n_samples {
                let mut pred = 0.0;
                for j in 0..n_features {
                    pred += x[(i, j)] * coefficients[j];
                }
                fitted_values[i] = pred;
                residuals[i] = y[i] - pred;
            }

            let n_nonzero = coefficients
                .iter()
                .enumerate()
                .filter(|(j, &c)| !aliased[*j] && c.abs() > 1e-10)
                .count();
            let n_params = n_nonzero;

            let result = self.compute_statistics(
                x,
                y,
                &coefficients,
                None,
                &residuals,
                &fitted_values,
                &aliased,
                n_nonzero,
                n_params,
            )?;

            // Compute (X'X)⁻¹ for prediction intervals (using non-aliased columns)
            // Note: This is approximate for L1-penalized models
            let xtx_inverse = compute_xtx_inverse_reduced(x, &aliased).ok();

            Ok(FittedElasticNet {
                options: self.options.clone(),
                result,
                xtx_inverse,
                aliased,
            })
        }
    }
}

impl ElasticNetRegressor {
    /// Solve Elastic Net using L-BFGS optimization with proximal soft thresholding.
    fn fit_lbfgs(&self, x: &Mat<f64>, y: &Col<f64>) -> Result<Col<f64>, RegressionError> {
        let n_samples = x.nrows();
        let n_features = x.ncols();
        let alpha = self.options.alpha;

        // Apply lambda scaling based on convention
        let lambda = match self.options.lambda_scaling {
            LambdaScaling::Raw => self.options.lambda,
            LambdaScaling::Glmnet => self.options.lambda * n_samples as f64,
        };

        // Create cost function (smooth part: RSS + L2 penalty)
        let cost = ElasticNetCost {
            x,
            y,
            lambda,
            alpha,
        };

        // Set up L-BFGS solver
        let linesearch = MoreThuenteLineSearch::new();
        let solver = LBFGS::new(linesearch, 7);

        // Initialize coefficients to zero
        let init = vec![0.0; n_features];

        // Run optimization
        let result = Executor::new(cost, solver)
            .configure(|state| {
                state
                    .param(init)
                    .max_iters(self.options.max_iterations as u64)
            })
            .run()
            .map_err(|e| RegressionError::NumericalError(format!("L-BFGS failed: {}", e)))?;

        // Get optimized coefficients
        let beta_smooth = result
            .state()
            .get_best_param()
            .ok_or_else(|| {
                RegressionError::NumericalError("L-BFGS did not return parameters".to_string())
            })?
            .clone();

        // Apply soft thresholding for L1 penalty (proximal step)
        let l1_threshold = lambda * alpha;
        let coefficients = Col::from_fn(n_features, |j| {
            Self::soft_threshold(beta_smooth[j], l1_threshold)
        });

        Ok(coefficients)
    }

    /// Compute fit statistics.
    #[allow(clippy::too_many_arguments)]
    fn compute_statistics(
        &self,
        x: &Mat<f64>,
        y: &Col<f64>,
        coefficients: &Col<f64>,
        intercept: Option<f64>,
        residuals: &Col<f64>,
        fitted_values: &Col<f64>,
        aliased: &[bool],
        rank: usize,
        n_params: usize,
    ) -> Result<RegressionResult, RegressionError> {
        let n = y.nrows();
        let n_features = x.ncols();

        // Compute y mean
        let y_mean: f64 = y.iter().sum::<f64>() / n as f64;

        // Compute TSS
        let tss: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();

        // Compute RSS
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
        let df_resid = (n.saturating_sub(n_params.max(1))) as f64;
        let adj_r_squared = if df_resid > 0.0 && df_total > 0.0 {
            1.0 - (1.0 - r_squared) * df_total / df_resid
        } else {
            f64::NAN
        };

        // MSE and RMSE
        let mse = if df_resid > 0.0 {
            rss / df_resid
        } else {
            rss / n as f64
        };
        let rmse = mse.sqrt();

        // F-statistic (approximate for elastic net)
        let ess = tss - rss;
        let df_model = (n_params.saturating_sub(if intercept.is_some() { 1 } else { 0 })) as f64;
        let f_statistic = if df_model > 0.0 && df_resid > 0.0 && mse > 0.0 {
            (ess / df_model.max(1.0)) / mse
        } else {
            f64::NAN
        };

        let f_pvalue = if f_statistic.is_finite() && df_model > 0.0 && df_resid > 0.0 {
            let f_dist = FisherSnedecor::new(df_model.max(1.0), df_resid).ok();
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
        result.rank_tolerance = self.options.rank_tolerance;
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
        result.confidence_level = self.options.confidence_level;

        // Elastic Net doesn't have simple standard errors
        // (would require bootstrap or other methods)

        Ok(result)
    }
}

/// A fitted Elastic Net model.
#[derive(Debug, Clone)]
pub struct FittedElasticNet {
    options: RegressionOptions,
    result: RegressionResult,
    /// (X'X)⁻¹ or (X_aug'X_aug)⁻¹ for prediction intervals (reduced to non-aliased columns)
    /// Note: Standard errors for L1-penalized models are approximate
    xtx_inverse: Option<Mat<f64>>,
    /// Which columns are aliased (constant or zero coefficient)
    aliased: Vec<bool>,
}

impl FittedElasticNet {
    /// Get the options used to fit this model.
    pub fn options(&self) -> &RegressionOptions {
        &self.options
    }

    /// Get the lambda (regularization) parameter.
    pub fn lambda(&self) -> f64 {
        self.options.lambda
    }

    /// Get the alpha (L1/L2 mixing) parameter.
    pub fn alpha(&self) -> f64 {
        self.options.alpha
    }

    /// Count non-zero coefficients (sparsity).
    pub fn n_nonzero(&self) -> usize {
        self.result
            .coefficients
            .iter()
            .filter(|&&c| c.abs() > 1e-10)
            .count()
    }
}

impl FittedRegressor for FittedElasticNet {
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
            Some(interval_type) => {
                // Note: Prediction intervals for L1-penalized models are approximate
                match &self.xtx_inverse {
                    Some(xtx_inv) => {
                        let df = self.result.residual_df() as f64;
                        let has_intercept = self.result.intercept.is_some();

                        // Filter x to only non-aliased columns (to match xtx_inverse dimensions)
                        let x_reduced = self.reduce_to_non_aliased(x);

                        compute_prediction_intervals(
                            &x_reduced,
                            xtx_inv,
                            &predictions,
                            self.result.mse,
                            df,
                            level,
                            interval_type,
                            has_intercept,
                        )
                    }
                    None => {
                        let n = x.nrows();
                        let mut lower = Col::zeros(n);
                        let mut upper = Col::zeros(n);
                        let mut se = Col::zeros(n);
                        for i in 0..n {
                            lower[i] = f64::NAN;
                            upper[i] = f64::NAN;
                            se[i] = f64::NAN;
                        }
                        PredictionResult::with_intervals(predictions, lower, upper, se)
                    }
                }
            }
        }
    }
}

impl FittedElasticNet {
    /// Reduce x to only non-aliased columns.
    fn reduce_to_non_aliased(&self, x: &Mat<f64>) -> Mat<f64> {
        let n_samples = x.nrows();
        let non_aliased_cols: Vec<usize> = self
            .aliased
            .iter()
            .enumerate()
            .filter(|(_, &is_aliased)| !is_aliased)
            .map(|(j, _)| j)
            .collect();

        let n_reduced = non_aliased_cols.len();
        let mut x_reduced = Mat::zeros(n_samples, n_reduced);

        for i in 0..n_samples {
            for (k, &j) in non_aliased_cols.iter().enumerate() {
                x_reduced[(i, k)] = x[(i, j)];
            }
        }

        x_reduced
    }
}

/// Builder for `ElasticNetRegressor`.
#[derive(Debug, Clone, Default)]
pub struct ElasticNetRegressorBuilder {
    builder: RegressionOptionsBuilder,
}

impl ElasticNetRegressorBuilder {
    /// Create a new builder with default options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set whether to include an intercept term.
    pub fn with_intercept(mut self, include: bool) -> Self {
        self.builder = self.builder.with_intercept(include);
        self
    }

    /// Set the regularization parameter (lambda).
    pub fn lambda(mut self, lambda: f64) -> Self {
        self.builder = self.builder.lambda(lambda);
        self
    }

    /// Set the L1/L2 mixing parameter (alpha).
    /// - alpha = 1: pure Lasso (L1)
    /// - alpha = 0: pure Ridge (L2)
    pub fn alpha(mut self, alpha: f64) -> Self {
        self.builder = self.builder.alpha(alpha);
        self
    }

    /// Set the lambda scaling convention.
    ///
    /// Use `LambdaScaling::Glmnet` to match R's glmnet package behavior.
    pub fn lambda_scaling(mut self, scaling: LambdaScaling) -> Self {
        self.builder = self.builder.lambda_scaling(scaling);
        self
    }

    /// Set maximum iterations for coordinate descent.
    pub fn max_iterations(mut self, max_iter: usize) -> Self {
        self.builder = self.builder.max_iterations(max_iter);
        self
    }

    /// Set convergence tolerance.
    pub fn tolerance(mut self, tol: f64) -> Self {
        self.builder = self.builder.tolerance(tol);
        self
    }

    /// Build the Elastic Net regressor.
    pub fn build(self) -> ElasticNetRegressor {
        ElasticNetRegressor::new(self.builder.build_unchecked())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_elastic_net_basic() {
        let x = Mat::from_fn(20, 2, |i, j| ((i + j) as f64) * 0.1);
        let mut y = Col::zeros(20);
        for i in 0..20 {
            y[i] = 1.0 + 2.0 * x[(i, 0)] + 3.0 * x[(i, 1)];
        }

        let model = ElasticNetRegressor::builder()
            .with_intercept(true)
            .lambda(0.01)
            .alpha(0.5)
            .build();

        let fitted = model.fit(&x, &y).expect("model should fit");

        assert!(fitted.r_squared() > 0.9);
    }

    #[test]
    fn test_lasso_sparsity() {
        // With high lambda and alpha=1 (Lasso), should get sparse solution
        let x = Mat::from_fn(50, 5, |i, j| ((i + j) as f64) * 0.1);
        let mut y = Col::zeros(50);
        // Only first 2 features matter
        for i in 0..50 {
            y[i] = 1.0 + 2.0 * x[(i, 0)] + 3.0 * x[(i, 1)];
        }

        let model = ElasticNetRegressor::builder()
            .with_intercept(true)
            .lambda(1.0)
            .alpha(1.0) // Pure Lasso
            .build();

        let fitted = model.fit(&x, &y).expect("model should fit");

        // Some coefficients should be zero (sparse)
        let n_zero = fitted
            .coefficients()
            .iter()
            .filter(|&&c| c.abs() < 1e-10)
            .count();

        // At least some sparsity expected
        assert!(
            n_zero >= 1,
            "Expected some zero coefficients, got {}",
            n_zero
        );
    }
}
