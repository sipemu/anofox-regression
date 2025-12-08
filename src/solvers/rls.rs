//! Recursive Least Squares solver.

use crate::core::{
    IntervalType, PredictionResult, RegressionOptions, RegressionOptionsBuilder, RegressionResult,
};
use crate::inference::compute_prediction_intervals;
use crate::solvers::traits::{FittedRegressor, RegressionError, Regressor};
use faer::{Col, Mat};
use statrs::distribution::{ContinuousCDF, FisherSnedecor};

/// Recursive Least Squares regression estimator.
///
/// RLS is an online learning algorithm that updates coefficients incrementally
/// as new observations arrive. It uses a forgetting factor (λ) to weight
/// recent observations more heavily.
///
/// The update equations are:
/// - P = (1/λ) * (P - P*x*x'*P / (λ + x'*P*x))
/// - K = P*x / (λ + x'*P*x)  (Kalman gain)
/// - β = β + K*(y - x'*β)
///
/// When λ = 1, all observations are weighted equally (converges to batch OLS).
/// When λ < 1, recent observations have more influence.
///
/// # Example
///
/// ```rust,ignore
/// use statistics::solvers::{RlsRegressor, Regressor, FittedRegressor};
/// use faer::{Mat, Col};
///
/// let x = Mat::from_fn(100, 2, |i, j| (i + j) as f64);
/// let y = Col::from_fn(100, |i| 1.0 + 2.0 * i as f64);
///
/// let fitted = RlsRegressor::builder()
///     .with_intercept(true)
///     .forgetting_factor(0.99)  // Recent data weighted more
///     .build()
///     .fit(&x, &y)?;
///
/// println!("R² = {}", fitted.r_squared());
/// ```
#[derive(Debug, Clone)]
pub struct RlsRegressor {
    options: RegressionOptions,
}

impl RlsRegressor {
    /// Create a new RLS regressor with the given options.
    pub fn new(options: RegressionOptions) -> Self {
        Self { options }
    }

    /// Create a builder for configuring the regressor.
    pub fn builder() -> RlsRegressorBuilder {
        RlsRegressorBuilder::default()
    }
}

impl Regressor for RlsRegressor {
    type Fitted = FittedRls;

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

        if n_samples < 2 {
            return Err(RegressionError::InsufficientObservations {
                needed: 2,
                got: n_samples,
            });
        }

        let forgetting_factor = self.options.forgetting_factor;
        let n_params = if self.options.with_intercept {
            n_features + 1
        } else {
            n_features
        };

        // Initialize P matrix (inverse covariance) with large diagonal
        let init_scale = 1e6;
        let mut p = Mat::zeros(n_params, n_params);
        for i in 0..n_params {
            p[(i, i)] = init_scale;
        }

        // Initialize coefficients to zero
        let mut beta = Col::zeros(n_params);

        // Process each observation
        for i in 0..n_samples {
            // Build augmented feature vector
            let mut xi = Col::zeros(n_params);
            if self.options.with_intercept {
                xi[0] = 1.0;
                for j in 0..n_features {
                    xi[j + 1] = x[(i, j)];
                }
            } else {
                for j in 0..n_features {
                    xi[j] = x[(i, j)];
                }
            }

            // Compute prediction error
            let y_pred: f64 = xi.iter().zip(beta.iter()).map(|(&x, &b)| x * b).sum();
            let error = y[i] - y_pred;

            // Compute P * x
            let mut px: Col<f64> = Col::zeros(n_params);
            for j in 0..n_params {
                for k in 0..n_params {
                    px[j] += p[(j, k)] * xi[k];
                }
            }

            // Compute x' * P * x
            let xpx: f64 = (0..n_params).map(|j| xi[j] * px[j]).sum();

            // Compute gain: K = P*x / (λ + x'*P*x)
            let denom = forgetting_factor + xpx;
            let mut k: Col<f64> = Col::zeros(n_params);
            for j in 0..n_params {
                k[j] = px[j] / denom;
            }

            // Update coefficients: β = β + K * error
            for j in 0..n_params {
                let k_j: f64 = k[j];
                beta[j] += k_j * error;
            }

            // Update P matrix: P = (1/λ) * (P - K * x' * P)
            // Equivalent to: P = (1/λ) * (P - P*x*x'*P / (λ + x'*P*x))
            let mut p_new: Mat<f64> = Mat::zeros(n_params, n_params);
            for j in 0..n_params {
                for l in 0..n_params {
                    let k_j: f64 = k[j];
                    let px_l: f64 = px[l];
                    p_new[(j, l)] = (p[(j, l)] - k_j * px_l) / forgetting_factor;
                }
            }
            p = p_new;
        }

        // Extract intercept and coefficients
        let (intercept, coefficients) = if self.options.with_intercept {
            let intercept = Some(beta[0]);
            let mut coef = Col::zeros(n_features);
            for j in 0..n_features {
                coef[j] = beta[j + 1];
            }
            (intercept, coef)
        } else {
            let mut coef = Col::zeros(n_features);
            for j in 0..n_features {
                coef[j] = beta[j];
            }
            (None, coef)
        };

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

        // Compute statistics
        let aliased = vec![false; n_features];
        let rank = n_features;
        let n_params_actual = if self.options.with_intercept {
            n_features + 1
        } else {
            n_features
        };

        let result = self.compute_statistics(
            y,
            &coefficients,
            intercept,
            &residuals,
            &fitted_values,
            &aliased,
            rank,
            n_params_actual,
            n_features,
        )?;

        Ok(FittedRls {
            options: self.options.clone(),
            p_matrix: p,
            result,
        })
    }
}

impl RlsRegressor {
    /// Compute fit statistics.
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
        n_features: usize,
    ) -> Result<RegressionResult, RegressionError> {
        let n = y.nrows();

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

        let f_pvalue = if f_statistic.is_finite() && df_model > 0.0 && df_resid > 0.0 {
            FisherSnedecor::new(df_model, df_resid)
                .ok()
                .map_or(f64::NAN, |d| 1.0 - d.cdf(f_statistic))
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

        // Note: RLS doesn't provide standard errors in the same way as OLS
        // The P matrix can be used, but interpretation differs

        Ok(result)
    }
}

/// A fitted RLS model.
#[derive(Debug, Clone)]
pub struct FittedRls {
    options: RegressionOptions,
    /// The P matrix (inverse covariance estimate) after fitting.
    p_matrix: Mat<f64>,
    result: RegressionResult,
}

impl FittedRls {
    /// Get the options used to fit this model.
    pub fn options(&self) -> &RegressionOptions {
        &self.options
    }

    /// Get the forgetting factor used.
    pub fn forgetting_factor(&self) -> f64 {
        self.options.forgetting_factor
    }

    /// Get the P matrix (inverse covariance estimate).
    pub fn p_matrix(&self) -> &Mat<f64> {
        &self.p_matrix
    }

    /// Update the model with a new observation (online learning).
    ///
    /// Returns the prediction for the new observation before updating.
    pub fn update(&mut self, x_new: &Col<f64>, y_new: f64) -> f64 {
        let n_features = self.result.coefficients.nrows();
        let n_params = self.p_matrix.nrows();
        let forgetting_factor = self.options.forgetting_factor;

        // Build augmented feature vector
        let mut xi = Col::zeros(n_params);
        if self.options.with_intercept {
            xi[0] = 1.0;
            for j in 0..n_features {
                xi[j + 1] = x_new[j];
            }
        } else {
            for j in 0..n_features {
                xi[j] = x_new[j];
            }
        }

        // Get current coefficients as augmented vector
        let mut beta = Col::zeros(n_params);
        if self.options.with_intercept {
            beta[0] = self.result.intercept.unwrap_or(0.0);
            for j in 0..n_features {
                beta[j + 1] = self.result.coefficients[j];
            }
        } else {
            for j in 0..n_features {
                beta[j] = self.result.coefficients[j];
            }
        }

        // Compute prediction before update
        let y_pred: f64 = xi.iter().zip(beta.iter()).map(|(&x, &b)| x * b).sum();
        let error = y_new - y_pred;

        // Compute P * x
        let mut px: Col<f64> = Col::zeros(n_params);
        for j in 0..n_params {
            for k in 0..n_params {
                px[j] += self.p_matrix[(j, k)] * xi[k];
            }
        }

        // Compute x' * P * x
        let xpx: f64 = (0..n_params).map(|j| xi[j] * px[j]).sum();

        // Compute gain
        let denom = forgetting_factor + xpx;
        let mut k: Col<f64> = Col::zeros(n_params);
        for j in 0..n_params {
            k[j] = px[j] / denom;
        }

        // Update coefficients
        for j in 0..n_params {
            let k_j: f64 = k[j];
            beta[j] += k_j * error;
        }

        // Update P matrix
        let mut p_new: Mat<f64> = Mat::zeros(n_params, n_params);
        for j in 0..n_params {
            for l in 0..n_params {
                let k_j: f64 = k[j];
                let px_l: f64 = px[l];
                p_new[(j, l)] = (self.p_matrix[(j, l)] - k_j * px_l) / forgetting_factor;
            }
        }
        self.p_matrix = p_new;

        // Store updated coefficients
        if self.options.with_intercept {
            self.result.intercept = Some(beta[0]);
            for j in 0..n_features {
                self.result.coefficients[j] = beta[j + 1];
            }
        } else {
            for j in 0..n_features {
                self.result.coefficients[j] = beta[j];
            }
        }

        y_pred
    }
}

impl FittedRegressor for FittedRls {
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
                // RLS stores p_matrix which is the estimate of (X'X)^-1
                // Note: For RLS, this is already augmented if with_intercept=true
                let df = self.result.residual_df() as f64;
                let has_intercept = self.result.intercept.is_some();

                compute_prediction_intervals(
                    x,
                    &self.p_matrix,
                    &predictions,
                    self.result.mse,
                    df,
                    level,
                    interval_type,
                    has_intercept,
                )
            }
        }
    }
}

/// Builder for `RlsRegressor`.
#[derive(Debug, Clone, Default)]
pub struct RlsRegressorBuilder {
    builder: RegressionOptionsBuilder,
}

impl RlsRegressorBuilder {
    /// Create a new builder with default options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set whether to include an intercept term.
    pub fn with_intercept(mut self, include: bool) -> Self {
        self.builder = self.builder.with_intercept(include);
        self
    }

    /// Set the forgetting factor (0 < λ ≤ 1).
    ///
    /// - λ = 1: All observations weighted equally (standard RLS)
    /// - λ < 1: Recent observations weighted more heavily
    pub fn forgetting_factor(mut self, factor: f64) -> Self {
        self.builder = self.builder.forgetting_factor(factor);
        self
    }

    /// Build the RLS regressor.
    pub fn build(self) -> RlsRegressor {
        RlsRegressor::new(self.builder.build_unchecked())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solvers::ols::OlsRegressor;

    #[test]
    fn test_rls_converges_to_ols() {
        // With forgetting_factor = 1, RLS should converge to batch OLS
        let x = Mat::from_fn(50, 1, |i, _| i as f64);
        let y = Col::from_fn(50, |i| 2.0 + 3.0 * i as f64);

        let rls = RlsRegressor::builder()
            .with_intercept(true)
            .forgetting_factor(1.0)
            .build();
        let ols = OlsRegressor::builder().with_intercept(true).build();

        let rls_fit = rls.fit(&x, &y).unwrap();
        let ols_fit = ols.fit(&x, &y).unwrap();

        // Should be close (not exact due to initialization)
        assert!(
            (rls_fit.coefficients()[0] - ols_fit.coefficients()[0]).abs() < 0.1,
            "RLS coef: {}, OLS coef: {}",
            rls_fit.coefficients()[0],
            ols_fit.coefficients()[0]
        );
    }

    #[test]
    fn test_rls_basic() {
        let x = Mat::from_fn(30, 1, |i, _| i as f64);
        let y = Col::from_fn(30, |i| 1.0 + 2.0 * i as f64);

        let model = RlsRegressor::builder()
            .with_intercept(true)
            .forgetting_factor(0.99)
            .build();

        let fitted = model.fit(&x, &y).unwrap();

        assert!(fitted.r_squared() > 0.9);
    }

    #[test]
    fn test_rls_online_update() {
        let x = Mat::from_fn(20, 1, |i, _| i as f64);
        let y = Col::from_fn(20, |i| 1.0 + 2.0 * i as f64);

        let model = RlsRegressor::builder()
            .with_intercept(true)
            .forgetting_factor(1.0)
            .build();

        let mut fitted = model.fit(&x, &y).unwrap();

        // Update with a new observation
        let x_new = Col::from_fn(1, |_| 20.0);
        let y_new = 1.0 + 2.0 * 20.0;

        let pred_before = fitted.update(&x_new, y_new);

        // Prediction should be close to actual
        assert!(
            (pred_before - y_new).abs() < 1.0,
            "Prediction {} should be close to {}",
            pred_before,
            y_new
        );
    }
}
