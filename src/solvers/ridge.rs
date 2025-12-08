//! Ridge regression solver (L2 regularization).

use crate::core::{
    IntervalType, LambdaScaling, PredictionResult, RegressionOptions, RegressionOptionsBuilder,
    RegressionResult,
};
use crate::inference::{compute_prediction_intervals, CoefficientInference};
use crate::solvers::ols::OlsRegressor;
use crate::solvers::traits::{FittedRegressor, RegressionError, Regressor};
use crate::utils::{center_columns, center_vector, detect_constant_columns};
use faer::{Col, Mat};
use statrs::distribution::{ContinuousCDF, FisherSnedecor, StudentsT};

/// Ridge regression estimator with L2 regularization.
///
/// Minimizes: ||y - Xβ||² + λ||β||²
///
/// The solution is: β = (X'X + λI)^(-1) X'y
///
/// When λ = 0, this reduces to OLS.
///
/// # Example
///
/// ```rust,ignore
/// use statistics::solvers::{RidgeRegressor, Regressor, FittedRegressor};
/// use faer::{Mat, Col};
///
/// let x = Mat::from_fn(100, 2, |i, j| (i + j) as f64);
/// let y = Col::from_fn(100, |i| 1.0 + 2.0 * i as f64);
///
/// let fitted = RidgeRegressor::builder()
///     .with_intercept(true)
///     .lambda(0.1)
///     .build()
///     .fit(&x, &y)?;
///
/// println!("R² = {}", fitted.r_squared());
/// ```
#[derive(Debug, Clone)]
pub struct RidgeRegressor {
    options: RegressionOptions,
}

impl RidgeRegressor {
    /// Create a new Ridge regressor with the given options.
    pub fn new(options: RegressionOptions) -> Self {
        Self { options }
    }

    /// Create a builder for configuring the regressor.
    pub fn builder() -> RidgeRegressorBuilder {
        RidgeRegressorBuilder::default()
    }
}

impl Regressor for RidgeRegressor {
    type Fitted = FittedRidge;

    fn fit(&self, x: &Mat<f64>, y: &Col<f64>) -> Result<Self::Fitted, RegressionError> {
        // When lambda = 0, delegate to OLS
        if self.options.lambda == 0.0 {
            let ols = OlsRegressor::new(self.options.clone());
            let ols_fitted = ols.fit(x, y)?;
            // Compute (X'X)^-1 for prediction intervals
            let xtx_inverse = crate::inference::compute_xtx_inverse_augmented(x).ok();
            return Ok(FittedRidge {
                options: self.options.clone(),
                result: ols_fitted.result().clone(),
                xtx_reg_inverse: xtx_inverse,
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

        // Detect constant columns (but Ridge handles them better than OLS)
        let _constant_cols = detect_constant_columns(x, self.options.rank_tolerance);

        if self.options.with_intercept {
            // Center the data
            let (x_centered, x_means) = center_columns(x);
            let (y_centered, y_mean) = center_vector(y);

            // Solve Ridge regression on centered data
            let coefficients = self.solve_ridge(&x_centered, &y_centered)?;

            // Compute intercept: intercept = y_mean - x_means' * coefficients
            let mut intercept = y_mean;
            for j in 0..n_features {
                intercept -= x_means[j] * coefficients[j];
            }

            // Compute fitted values and residuals
            let mut fitted_values = Col::zeros(n_samples);
            let mut residuals = Col::zeros(n_samples);

            for i in 0..n_samples {
                let mut pred = intercept;
                for j in 0..n_features {
                    pred += x[(i, j)] * coefficients[j];
                }
                fitted_values[i] = pred;
                residuals[i] = y[i] - pred;
            }

            // Compute statistics
            let aliased = vec![false; n_features]; // Ridge has no aliased coefficients
            let rank = n_features; // Ridge is always full rank
            let n_params = n_features + 1; // +1 for intercept

            let result = self.compute_statistics(
                x,
                y,
                &coefficients,
                Some(intercept),
                &residuals,
                &fitted_values,
                &aliased,
                rank,
                n_params,
            )?;

            // Compute (X_aug'X_aug + λI_aug)⁻¹ for prediction intervals
            let xtx_reg_inverse = self.compute_xtx_reg_inverse_augmented(x);

            Ok(FittedRidge {
                options: self.options.clone(),
                result,
                xtx_reg_inverse,
            })
        } else {
            // No intercept case
            let coefficients = self.solve_ridge(x, y)?;

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

            let aliased = vec![false; n_features];
            let rank = n_features;
            let n_params = n_features;

            let result = self.compute_statistics(
                x,
                y,
                &coefficients,
                None,
                &residuals,
                &fitted_values,
                &aliased,
                rank,
                n_params,
            )?;

            // Compute (X'X + λI)⁻¹ for prediction intervals
            let xtx_reg_inverse = self.compute_xtx_reg_inverse(x);

            Ok(FittedRidge {
                options: self.options.clone(),
                result,
                xtx_reg_inverse,
            })
        }
    }
}

impl RidgeRegressor {
    /// Compute (X_aug'X_aug + λI_aug)⁻¹ for models with intercept.
    fn compute_xtx_reg_inverse_augmented(&self, x: &Mat<f64>) -> Option<Mat<f64>> {
        let n_samples = x.nrows();
        let n_features = x.ncols();
        let aug_size = n_features + 1;

        let lambda = match self.options.lambda_scaling {
            LambdaScaling::Raw => self.options.lambda,
            LambdaScaling::Glmnet => self.options.lambda * n_samples as f64,
        };

        // Build X_aug'X_aug
        let mut xtx_aug: Mat<f64> = Mat::zeros(aug_size, aug_size);
        for i in 0..n_samples {
            // (0,0): n (sum of 1*1)
            xtx_aug[(0, 0)] += 1.0;
            for j in 0..n_features {
                // (0,j+1) and (j+1,0): sum of x_j
                xtx_aug[(0, j + 1)] += x[(i, j)];
                xtx_aug[(j + 1, 0)] += x[(i, j)];
                // (j+1,k+1): sum of x_j * x_k
                for k in 0..n_features {
                    xtx_aug[(j + 1, k + 1)] += x[(i, j)] * x[(i, k)];
                }
            }
        }

        // Add λI (but don't penalize intercept term)
        for j in 1..aug_size {
            xtx_aug[(j, j)] += lambda;
        }

        // Invert using QR
        let qr: faer::linalg::solvers::Qr<f64> = xtx_aug.qr();
        let q = qr.compute_q();
        let r = qr.compute_r();

        for i in 0..aug_size {
            if r[(i, i)].abs() < 1e-14 {
                return None;
            }
        }

        let mut inv = Mat::zeros(aug_size, aug_size);
        let qt = q.transpose();

        for col in 0..aug_size {
            for i in (0..aug_size).rev() {
                let mut sum = qt[(i, col)];
                for j in (i + 1)..aug_size {
                    sum -= r[(i, j)] * inv[(j, col)];
                }
                inv[(i, col)] = sum / r[(i, i)];
            }
        }

        Some(inv)
    }

    /// Compute (X'X + λI)⁻¹ for models without intercept.
    fn compute_xtx_reg_inverse(&self, x: &Mat<f64>) -> Option<Mat<f64>> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        let lambda = match self.options.lambda_scaling {
            LambdaScaling::Raw => self.options.lambda,
            LambdaScaling::Glmnet => self.options.lambda * n_samples as f64,
        };

        // Compute X'X + λI
        let xtx = x.transpose() * x;
        let mut xtx_reg = xtx.clone();
        for i in 0..n_features {
            xtx_reg[(i, i)] += lambda;
        }

        // Invert using QR
        let qr: faer::linalg::solvers::Qr<f64> = xtx_reg.qr();
        let q = qr.compute_q();
        let r = qr.compute_r();

        for i in 0..n_features {
            if r[(i, i)].abs() < 1e-14 {
                return None;
            }
        }

        let mut inv = Mat::zeros(n_features, n_features);
        let qt = q.transpose();

        for col in 0..n_features {
            for i in (0..n_features).rev() {
                let mut sum = qt[(i, col)];
                for j in (i + 1)..n_features {
                    sum -= r[(i, j)] * inv[(j, col)];
                }
                inv[(i, col)] = sum / r[(i, i)];
            }
        }

        Some(inv)
    }

    /// Solve Ridge regression: β = (X'X + λI)^(-1) X'y
    fn solve_ridge(&self, x: &Mat<f64>, y: &Col<f64>) -> Result<Col<f64>, RegressionError> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        // Apply lambda scaling based on convention
        let lambda = match self.options.lambda_scaling {
            LambdaScaling::Raw => self.options.lambda,
            LambdaScaling::Glmnet => self.options.lambda * n_samples as f64,
        };

        // Compute X'X
        let xtx = x.transpose() * x;

        // Add λI to the diagonal (X'X + λI)
        let mut xtx_reg = xtx.clone();
        for i in 0..n_features {
            xtx_reg[(i, i)] += lambda;
        }

        // Compute X'y
        let xty = x.transpose() * y;

        // Solve (X'X + λI) β = X'y using QR decomposition
        let qr = xtx_reg.qr();
        let q = qr.compute_q();
        let r = qr.compute_r();

        // Check if R is singular
        for i in 0..n_features {
            if r[(i, i)].abs() < 1e-14 {
                return Err(RegressionError::SingularMatrix);
            }
        }

        // Solve R β = Q' (X'y)
        let qty = q.transpose() * &xty;

        // Back-substitution
        let mut coefficients = Col::zeros(n_features);
        for i in (0..n_features).rev() {
            let mut sum = qty[i];
            for j in (i + 1)..n_features {
                sum -= r[(i, j)] * coefficients[j];
            }
            coefficients[i] = sum / r[(i, i)];
        }

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

        // Compute inference statistics if requested
        if self.options.compute_inference {
            self.compute_inference(x, &mut result)?;
        }

        Ok(result)
    }

    /// Compute inference statistics for Ridge regression.
    fn compute_inference(
        &self,
        x: &Mat<f64>,
        result: &mut RegressionResult,
    ) -> Result<(), RegressionError> {
        let n_samples = x.nrows();
        let n_features = x.ncols();
        let df = result.residual_df() as f64;

        // Apply lambda scaling based on convention
        let lambda = match self.options.lambda_scaling {
            LambdaScaling::Raw => self.options.lambda,
            LambdaScaling::Glmnet => self.options.lambda * n_samples as f64,
        };

        if df <= 0.0 || !result.mse.is_finite() {
            return Ok(());
        }

        // For Ridge, SE(β) ≈ sqrt(MSE * diag((X'X + λI)^(-1)))
        // This is an approximation; true Ridge SE is more complex

        // Compute (X'X + λI)^(-1)
        let xtx = x.transpose() * x;
        let mut xtx_reg = xtx.clone();
        for i in 0..n_features {
            xtx_reg[(i, i)] += lambda;
        }

        // Invert using QR
        let qr = xtx_reg.qr();
        let q = qr.compute_q();
        let r = qr.compute_r();

        let mut xtx_inv = Mat::zeros(n_features, n_features);
        let qt = q.transpose();

        for col in 0..n_features {
            for i in (0..n_features).rev() {
                let mut sum = qt[(i, col)];
                for j in (i + 1)..n_features {
                    sum -= r[(i, j)] * xtx_inv[(j, col)];
                }
                xtx_inv[(i, col)] = sum / r[(i, i)];
            }
        }

        // Compute standard errors
        let mut std_errors = Col::zeros(n_features);
        for j in 0..n_features {
            let var = result.mse * xtx_inv[(j, j)];
            std_errors[j] = if var >= 0.0 { var.sqrt() } else { f64::NAN };
        }

        // t-statistics
        let t_stats = CoefficientInference::t_statistics(&result.coefficients, &std_errors);

        // p-values
        let p_vals = CoefficientInference::p_values(&t_stats, df);

        // Confidence intervals
        let (ci_lower, ci_upper) = CoefficientInference::confidence_intervals(
            &result.coefficients,
            &std_errors,
            df,
            self.options.confidence_level,
        );

        result.std_errors = Some(std_errors);
        result.t_statistics = Some(t_stats);
        result.p_values = Some(p_vals);
        result.conf_interval_lower = Some(ci_lower);
        result.conf_interval_upper = Some(ci_upper);

        // Intercept inference
        if let Some(intercept) = result.intercept {
            let se_int = (result.mse / result.n_observations as f64).sqrt();
            let t_int = intercept / se_int;

            let t_dist = StudentsT::new(0.0, 1.0, df).ok();
            let p_int = t_dist.map_or(f64::NAN, |d| 2.0 * (1.0 - d.cdf(t_int.abs())));

            let t_crit = t_dist.map_or(f64::NAN, |d| {
                d.inverse_cdf(1.0 - (1.0 - self.options.confidence_level) / 2.0)
            });
            let ci_int = (intercept - t_crit * se_int, intercept + t_crit * se_int);

            result.intercept_std_error = Some(se_int);
            result.intercept_t_statistic = Some(t_int);
            result.intercept_p_value = Some(p_int);
            result.intercept_conf_interval = Some(ci_int);
        }

        Ok(())
    }
}

/// A fitted Ridge regression model.
#[derive(Debug, Clone)]
pub struct FittedRidge {
    options: RegressionOptions,
    result: RegressionResult,
    /// (X'X + λI)⁻¹ for prediction intervals (augmented if with_intercept)
    xtx_reg_inverse: Option<Mat<f64>>,
}

impl FittedRidge {
    /// Get the options used to fit this model.
    pub fn options(&self) -> &RegressionOptions {
        &self.options
    }

    /// Get the lambda (regularization) parameter.
    pub fn lambda(&self) -> f64 {
        self.options.lambda
    }
}

impl FittedRegressor for FittedRidge {
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
            Some(interval_type) => match &self.xtx_reg_inverse {
                Some(xtx_inv) => {
                    let df = self.result.residual_df() as f64;
                    let has_intercept = self.result.intercept.is_some();

                    compute_prediction_intervals(
                        x,
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
            },
        }
    }
}

/// Builder for `RidgeRegressor`.
#[derive(Debug, Clone, Default)]
pub struct RidgeRegressorBuilder {
    builder: RegressionOptionsBuilder,
}

impl RidgeRegressorBuilder {
    /// Create a new builder with default options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set whether to include an intercept term.
    pub fn with_intercept(mut self, include: bool) -> Self {
        self.builder = self.builder.with_intercept(include);
        self
    }

    /// Set the L2 regularization parameter (lambda).
    pub fn lambda(mut self, lambda: f64) -> Self {
        self.builder = self.builder.lambda(lambda);
        self
    }

    /// Set the lambda scaling convention.
    ///
    /// Use `LambdaScaling::Glmnet` to match R's glmnet package behavior.
    pub fn lambda_scaling(mut self, scaling: LambdaScaling) -> Self {
        self.builder = self.builder.lambda_scaling(scaling);
        self
    }

    /// Set whether to compute inference statistics.
    pub fn compute_inference(mut self, compute: bool) -> Self {
        self.builder = self.builder.compute_inference(compute);
        self
    }

    /// Set the confidence level for confidence intervals.
    pub fn confidence_level(mut self, level: f64) -> Self {
        self.builder = self.builder.confidence_level(level);
        self
    }

    /// Build the Ridge regressor.
    pub fn build(self) -> RidgeRegressor {
        RidgeRegressor::new(self.builder.build_unchecked())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ridge_basic() {
        let x = Mat::from_fn(10, 1, |i, _| i as f64);
        let y = Col::from_fn(10, |i| 2.0 + 3.0 * i as f64);

        let model = RidgeRegressor::builder()
            .with_intercept(true)
            .lambda(0.01)
            .build();

        let fitted = model.fit(&x, &y).unwrap();

        assert!(fitted.r_squared() > 0.99);
    }
}
