//! Weighted Least Squares solver.

use crate::core::{
    IntervalType, PredictionResult, RegressionOptions, RegressionOptionsBuilder, RegressionResult,
};
use crate::inference::{
    compute_prediction_intervals, compute_xtwx_inverse_augmented_reduced,
    compute_xtwx_inverse_reduced, compute_xtx_inverse_augmented_reduced, CoefficientInference,
};
use crate::solvers::ols::OlsRegressor;
use crate::solvers::traits::{FittedRegressor, RegressionError, Regressor};
use crate::utils::detect_constant_columns;
use faer::linalg::solvers::Qr;
use faer::{Col, Mat};
use statrs::distribution::{ContinuousCDF, FisherSnedecor, StudentsT};

/// Weighted Least Squares regression estimator.
///
/// Minimizes: Σ w_i (y_i - x_i'β)²
///
/// This is equivalent to transforming the problem:
/// X → W^(1/2)X, y → W^(1/2)y, then applying OLS.
///
/// When all weights are equal, this reduces to OLS.
///
/// # Example
///
/// ```rust,ignore
/// use statistics::solvers::{WlsRegressor, Regressor, FittedRegressor};
/// use faer::{Mat, Col};
///
/// let x = Mat::from_fn(100, 2, |i, j| (i + j) as f64);
/// let y = Col::from_fn(100, |i| 1.0 + 2.0 * i as f64);
/// let weights = Col::from_fn(100, |i| 1.0 / (i + 1) as f64);  // Inverse variance weights
///
/// let fitted = WlsRegressor::builder()
///     .with_intercept(true)
///     .weights(weights)
///     .build()
///     .fit(&x, &y)?;
///
/// println!("R² = {}", fitted.r_squared());
/// ```
#[derive(Debug, Clone)]
pub struct WlsRegressor {
    options: RegressionOptions,
    weights: Option<Col<f64>>,
}

impl WlsRegressor {
    /// Create a new WLS regressor with the given options.
    pub fn new(options: RegressionOptions) -> Self {
        Self {
            options,
            weights: None,
        }
    }

    /// Set the observation weights.
    pub fn with_weights(mut self, weights: Col<f64>) -> Self {
        self.weights = Some(weights);
        self
    }

    /// Create a builder for configuring the regressor.
    pub fn builder() -> WlsRegressorBuilder {
        WlsRegressorBuilder::default()
    }
}

impl Regressor for WlsRegressor {
    type Fitted = FittedWls;

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

        // Get weights or use unit weights
        let weights = match &self.weights {
            Some(w) => {
                if w.nrows() != n_samples {
                    return Err(RegressionError::DimensionMismatch {
                        x_rows: n_samples,
                        y_len: w.nrows(),
                    });
                }
                // Validate weights are non-negative
                for i in 0..n_samples {
                    if w[i] < 0.0 {
                        return Err(RegressionError::InvalidWeights);
                    }
                }
                w.clone()
            }
            None => Col::from_fn(n_samples, |_| 1.0),
        };

        // Check that not all weights are zero
        let weight_sum: f64 = weights.iter().sum();
        if weight_sum < 1e-14 {
            return Err(RegressionError::InvalidWeights);
        }

        // Count effective observations (non-zero weights)
        let n_effective: usize = weights.iter().filter(|&&w| w > 1e-14).count();
        let n_params = if self.options.with_intercept {
            n_features + 1
        } else {
            n_features
        };

        if n_effective < n_params {
            return Err(RegressionError::InsufficientObservations {
                needed: n_params,
                got: n_effective,
            });
        }

        // Check if all weights are equal - if so, delegate to OLS
        let first_weight = weights[0];
        let all_equal = weights.iter().all(|&w| (w - first_weight).abs() < 1e-14);

        if all_equal && first_weight > 0.0 {
            let ols = OlsRegressor::new(self.options.clone());
            let ols_fitted = ols.fit(x, y)?;
            // For uniform weights, use unweighted (X'X)⁻¹ with aliased columns filtered
            let aliased = ols_fitted.result().aliased.clone();
            let xtx_inverse = compute_xtx_inverse_augmented_reduced(x, &aliased).ok();
            return Ok(FittedWls {
                options: self.options.clone(),
                weights: weights.clone(),
                result: ols_fitted.result().clone(),
                xtwx_inverse: xtx_inverse,
                aliased,
            });
        }

        // Compute sqrt(weights) for transformation
        let sqrt_weights = Col::from_fn(n_samples, |i| weights[i].sqrt());

        // Transform data: X_w = W^(1/2) X, y_w = W^(1/2) y
        let mut x_weighted = Mat::zeros(n_samples, n_features);
        let mut y_weighted = Col::zeros(n_samples);

        for i in 0..n_samples {
            let sw = sqrt_weights[i];
            y_weighted[i] = y[i] * sw;
            for j in 0..n_features {
                x_weighted[(i, j)] = x[(i, j)] * sw;
            }
        }

        if self.options.with_intercept {
            // For WLS with intercept, we need weighted centering
            // Pass ORIGINAL (unweighted) x, y and weights - centering happens inside
            let (x_centered, y_centered, x_means, y_mean) = self.weighted_center(x, y, &weights);

            // Detect constant columns in CENTERED weighted data
            // This is important for extreme weights like 1/x² where x*sqrt(1/x²)=1 is constant
            // but the centered weighted data has variation
            let constant_cols = detect_constant_columns(&x_centered, self.options.rank_tolerance);

            // Solve using QR decomposition
            let (coefficients, aliased, rank) =
                self.solve_with_qr(&x_centered, &y_centered, &constant_cols)?;

            // Compute intercept in original (unweighted) space
            let mut intercept = y_mean;
            for j in 0..n_features {
                if !aliased[j] && !coefficients[j].is_nan() {
                    intercept -= x_means[j] * coefficients[j];
                }
            }

            // Compute fitted values and residuals in original space
            let mut fitted_values = Col::zeros(n_samples);
            let mut residuals = Col::zeros(n_samples);

            for i in 0..n_samples {
                let mut pred = intercept;
                for j in 0..n_features {
                    if !aliased[j] && !coefficients[j].is_nan() {
                        pred += x[(i, j)] * coefficients[j];
                    }
                }
                fitted_values[i] = pred;
                residuals[i] = y[i] - pred;
            }

            let n_params = rank + 1;
            let result = self.compute_statistics(
                x,
                y,
                &weights,
                &coefficients,
                Some(intercept),
                &residuals,
                &fitted_values,
                &aliased,
                rank,
                n_params,
            )?;

            // Compute (X_aug'WX_aug)⁻¹ for prediction intervals (using non-aliased columns)
            let xtwx_inverse = compute_xtwx_inverse_augmented_reduced(x, &weights, &aliased).ok();

            Ok(FittedWls {
                options: self.options.clone(),
                weights,
                result,
                xtwx_inverse,
                aliased,
            })
        } else {
            // No intercept case - detect constant columns in weighted data
            let constant_cols = detect_constant_columns(&x_weighted, self.options.rank_tolerance);
            let (coefficients, aliased, rank) =
                self.solve_with_qr(&x_weighted, &y_weighted, &constant_cols)?;

            // Compute fitted values and residuals
            let mut fitted_values = Col::zeros(n_samples);
            let mut residuals = Col::zeros(n_samples);

            for i in 0..n_samples {
                let mut pred = 0.0;
                for j in 0..n_features {
                    if !aliased[j] && !coefficients[j].is_nan() {
                        pred += x[(i, j)] * coefficients[j];
                    }
                }
                fitted_values[i] = pred;
                residuals[i] = y[i] - pred;
            }

            let n_params = rank;
            let result = self.compute_statistics(
                x,
                y,
                &weights,
                &coefficients,
                None,
                &residuals,
                &fitted_values,
                &aliased,
                rank,
                n_params,
            )?;

            // Compute (X'WX)⁻¹ for prediction intervals (using non-aliased columns)
            let xtwx_inverse = compute_xtwx_inverse_reduced(x, &weights, &aliased).ok();

            Ok(FittedWls {
                options: self.options.clone(),
                weights,
                result,
                xtwx_inverse,
                aliased,
            })
        }
    }
}

impl WlsRegressor {
    /// Weighted centering of data.
    ///
    /// Takes ORIGINAL (unweighted) x, y and the weights.
    /// Returns centered-and-weighted data plus the weighted means.
    fn weighted_center(
        &self,
        x_orig: &Mat<f64>,
        y_orig: &Col<f64>,
        weights: &Col<f64>,
    ) -> (Mat<f64>, Col<f64>, Col<f64>, f64) {
        let n_samples = x_orig.nrows();
        let n_features = x_orig.ncols();

        // Compute sum of weights
        let sum_w: f64 = weights.iter().sum();

        // Compute weighted means of ORIGINAL (unweighted) data
        let mut x_means = Col::zeros(n_features);
        for j in 0..n_features {
            let mut sum = 0.0;
            for i in 0..n_samples {
                sum += weights[i] * x_orig[(i, j)];
            }
            x_means[j] = sum / sum_w;
        }

        let y_mean: f64 = {
            let mut sum = 0.0;
            for i in 0..n_samples {
                sum += weights[i] * y_orig[i];
            }
            sum / sum_w
        };

        // Center THEN weight in one step
        let mut x_centered_weighted = Mat::zeros(n_samples, n_features);
        let mut y_centered_weighted = Col::zeros(n_samples);

        for i in 0..n_samples {
            let sqrt_w = weights[i].sqrt();
            y_centered_weighted[i] = sqrt_w * (y_orig[i] - y_mean);
            for j in 0..n_features {
                x_centered_weighted[(i, j)] = sqrt_w * (x_orig[(i, j)] - x_means[j]);
            }
        }

        (x_centered_weighted, y_centered_weighted, x_means, y_mean)
    }

    /// Solve using QR decomposition with column pivoting.
    fn solve_with_qr(
        &self,
        x: &Mat<f64>,
        y: &Col<f64>,
        constant_cols: &[bool],
    ) -> Result<(Col<f64>, Vec<bool>, usize), RegressionError> {
        let n_features = x.ncols();
        let n_samples = x.nrows();

        let mut aliased = constant_cols.to_vec();

        let qr = x.col_piv_qr();
        let q = qr.compute_Q();
        let r = qr.R();
        let perm = qr.P();

        let perm_arr = perm.arrays().0;
        let mut perm_inv: Vec<usize> = vec![0; n_features];
        perm_inv[..n_features].copy_from_slice(&perm_arr[..n_features]);

        // Determine rank
        let mut rank = 0;
        for i in 0..n_features.min(n_samples) {
            if r[(i, i)].abs() > self.options.rank_tolerance {
                rank += 1;
            } else {
                break;
            }
        }

        if rank == 0 {
            let mut coefficients = Col::zeros(n_features);
            for j in 0..n_features {
                coefficients[j] = f64::NAN;
                aliased[j] = true;
            }
            return Ok((coefficients, aliased, 0));
        }

        // Mark aliased columns
        for j in 0..n_features {
            if constant_cols[j] || perm_inv[j] >= rank {
                aliased[j] = true;
            }
        }

        // Solve
        let qty = q.transpose() * y;
        let mut beta_reduced = Col::zeros(rank);
        for i in (0..rank).rev() {
            let mut sum = qty[i];
            for j in (i + 1)..rank {
                sum -= r[(i, j)] * beta_reduced[j];
            }
            beta_reduced[i] = sum / r[(i, i)];
        }

        // Map back to original order
        let mut coefficients = Col::zeros(n_features);
        for j in 0..n_features {
            if aliased[j] {
                coefficients[j] = f64::NAN;
            } else {
                coefficients[j] = beta_reduced[perm_inv[j]];
            }
        }

        Ok((coefficients, aliased, rank))
    }

    /// Compute fit statistics.
    #[allow(clippy::too_many_arguments)]
    fn compute_statistics(
        &self,
        x: &Mat<f64>,
        y: &Col<f64>,
        weights: &Col<f64>,
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

        // Compute weighted mean of y
        let sum_w: f64 = weights.iter().sum();
        let y_mean: f64 = y
            .iter()
            .zip(weights.iter())
            .map(|(&yi, &wi)| wi * yi)
            .sum::<f64>()
            / sum_w;

        // Compute weighted TSS
        let tss: f64 = y
            .iter()
            .zip(weights.iter())
            .map(|(&yi, &wi)| wi * (yi - y_mean).powi(2))
            .sum();

        // Compute weighted RSS
        let rss: f64 = residuals
            .iter()
            .zip(weights.iter())
            .map(|(&ri, &wi)| wi * ri.powi(2))
            .sum();

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

        // MSE and RMSE (weighted)
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

        // Compute inference if requested
        if self.options.compute_inference {
            self.compute_inference(x, weights, &mut result)?;
        }

        Ok(result)
    }

    /// Compute inference statistics for WLS.
    fn compute_inference(
        &self,
        x: &Mat<f64>,
        weights: &Col<f64>,
        result: &mut RegressionResult,
    ) -> Result<(), RegressionError> {
        let df = result.residual_df() as f64;

        if df <= 0.0 || !result.mse.is_finite() {
            return Ok(());
        }

        // Use the weighted augmented design matrix method for models with intercept
        // This computes SE for both intercept and coefficients correctly
        if let Some(intercept) = result.intercept {
            match CoefficientInference::standard_errors_wls_with_intercept(
                x,
                weights,
                result.mse,
                &result.aliased,
            ) {
                Ok((std_errors, se_int)) => {
                    let t_stats =
                        CoefficientInference::t_statistics(&result.coefficients, &std_errors);
                    let p_vals = CoefficientInference::p_values(&t_stats, df);
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
                    let t_int = if se_int > 0.0 {
                        intercept / se_int
                    } else {
                        f64::NAN
                    };

                    let t_dist = StudentsT::new(0.0, 1.0, df).ok();
                    let p_int = if t_int.is_finite() {
                        t_dist.map_or(f64::NAN, |d| 2.0 * (1.0 - d.cdf(t_int.abs())))
                    } else {
                        f64::NAN
                    };
                    let t_crit = t_dist.map_or(f64::NAN, |d| {
                        d.inverse_cdf(1.0 - (1.0 - self.options.confidence_level) / 2.0)
                    });

                    result.intercept_std_error = Some(se_int);
                    result.intercept_t_statistic = Some(t_int);
                    result.intercept_p_value = Some(p_int);
                    result.intercept_conf_interval =
                        Some((intercept - t_crit * se_int, intercept + t_crit * se_int));
                }
                Err(_) => {
                    // Failed to compute standard errors, leave as None
                }
            }
        } else {
            // No intercept case - use X'WX directly
            let n_features = x.ncols();
            let n_samples = x.nrows();

            let mut xtwx: Mat<f64> = Mat::zeros(n_features, n_features);
            for i in 0..n_samples {
                for j in 0..n_features {
                    for k in 0..n_features {
                        xtwx[(j, k)] += weights[i] * x[(i, j)] * x[(i, k)];
                    }
                }
            }

            // Invert X'WX using QR
            let qr: Qr<f64> = xtwx.qr();
            let q = qr.compute_Q();
            let r = qr.R();

            let mut xtwx_inv: Mat<f64> = Mat::zeros(n_features, n_features);
            let qt = q.transpose();

            for col in 0..n_features {
                for i in (0..n_features).rev() {
                    if r[(i, i)].abs() < 1e-14 {
                        continue;
                    }
                    let mut sum = qt[(i, col)];
                    for j in (i + 1)..n_features {
                        sum -= r[(i, j)] * xtwx_inv[(j, col)];
                    }
                    xtwx_inv[(i, col)] = sum / r[(i, i)];
                }
            }

            // Compute standard errors
            let mut std_errors = Col::zeros(n_features);
            for j in 0..n_features {
                if result.aliased[j] {
                    std_errors[j] = f64::NAN;
                } else {
                    let var = result.mse * xtwx_inv[(j, j)];
                    std_errors[j] = if var >= 0.0 { var.sqrt() } else { f64::NAN };
                }
            }

            let t_stats = CoefficientInference::t_statistics(&result.coefficients, &std_errors);
            let p_vals = CoefficientInference::p_values(&t_stats, df);
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
        }

        Ok(())
    }
}

/// A fitted WLS model.
#[derive(Debug, Clone)]
pub struct FittedWls {
    options: RegressionOptions,
    weights: Col<f64>,
    result: RegressionResult,
    /// (X'WX)⁻¹ or (X_aug'WX_aug)⁻¹ for prediction intervals (reduced to non-aliased columns)
    xtwx_inverse: Option<Mat<f64>>,
    /// Which columns are aliased (collinear or constant)
    aliased: Vec<bool>,
}

impl FittedWls {
    /// Get the options used to fit this model.
    pub fn options(&self) -> &RegressionOptions {
        &self.options
    }

    /// Get the weights used for fitting.
    pub fn weights(&self) -> &Col<f64> {
        &self.weights
    }
}

impl FittedRegressor for FittedWls {
    fn predict(&self, x: &Mat<f64>) -> Col<f64> {
        let n_samples = x.nrows();
        let n_features = x.ncols();
        let mut predictions = Col::zeros(n_samples);

        let intercept = self.result.intercept.unwrap_or(0.0);

        for i in 0..n_samples {
            let mut pred = intercept;
            for j in 0..n_features {
                if !self.result.aliased[j] && !self.result.coefficients[j].is_nan() {
                    pred += x[(i, j)] * self.result.coefficients[j];
                }
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
                match &self.xtwx_inverse {
                    Some(xtwx_inv) => {
                        let df = self.result.residual_df() as f64;
                        let has_intercept = self.result.intercept.is_some();

                        // Filter x to only non-aliased columns (to match xtwx_inverse dimensions)
                        let x_reduced = self.reduce_to_non_aliased(x);

                        // For WLS prediction on new data, we use the weighted inverse
                        // but predictions on new data are unweighted
                        compute_prediction_intervals(
                            &x_reduced,
                            xtwx_inv,
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

impl FittedWls {
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

/// Builder for `WlsRegressor`.
#[derive(Debug, Clone, Default)]
pub struct WlsRegressorBuilder {
    builder: RegressionOptionsBuilder,
    weights: Option<Col<f64>>,
}

impl WlsRegressorBuilder {
    /// Create a new builder with default options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set whether to include an intercept term.
    pub fn with_intercept(mut self, include: bool) -> Self {
        self.builder = self.builder.with_intercept(include);
        self
    }

    /// Set the observation weights.
    pub fn weights(mut self, weights: Col<f64>) -> Self {
        self.weights = Some(weights);
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

    /// Build the WLS regressor.
    pub fn build(self) -> WlsRegressor {
        let mut regressor = WlsRegressor::new(self.builder.build_unchecked());
        if let Some(w) = self.weights {
            regressor = regressor.with_weights(w);
        }
        regressor
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wls_equal_weights_equals_ols() {
        let x = Mat::from_fn(10, 1, |i, _| i as f64);
        let y = Col::from_fn(10, |i| 2.0 + 3.0 * i as f64);
        let weights = Col::from_fn(10, |_| 1.0);

        let wls = WlsRegressor::builder()
            .with_intercept(true)
            .weights(weights)
            .build();
        let ols = OlsRegressor::builder().with_intercept(true).build();

        let wls_fit = wls.fit(&x, &y).expect("WLS model should fit");
        let ols_fit = ols.fit(&x, &y).expect("OLS model should fit");

        assert!((wls_fit.coefficients()[0] - ols_fit.coefficients()[0]).abs() < 1e-10);
        assert!(
            (wls_fit.intercept().expect("intercept exists")
                - ols_fit.intercept().expect("intercept exists"))
            .abs()
                < 1e-10
        );
    }

    #[test]
    fn test_wls_basic() {
        let x = Mat::from_fn(20, 1, |i, _| i as f64);
        let y = Col::from_fn(20, |i| 1.0 + 2.0 * i as f64 + (i as f64) * 0.1);
        let weights = Col::from_fn(20, |i| 1.0 / ((i + 1) as f64)); // Higher weight for early obs

        let model = WlsRegressor::builder()
            .with_intercept(true)
            .weights(weights)
            .build();

        let fitted = model.fit(&x, &y).expect("model should fit");

        assert!(fitted.r_squared() > 0.9);
    }
}
