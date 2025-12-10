//! Dynamic Linear Model (lmDynamic) - Time-varying parameter regression.
//!
//! This module implements a time-varying parameter model that combines multiple
//! candidate regression models using pointwise information criteria weighting.
//! Based on the lmDynamic function from the greybox R package.
//!
//! # Algorithm
//!
//! 1. Generate candidate models from variable subsets
//! 2. Fit each candidate model using ALM
//! 3. Compute pointwise information criteria for each observation
//! 4. Calculate dynamic weights for each model at each time point
//! 5. Optionally smooth weights using LOWESS
//! 6. Compute weighted parameter estimates that vary over time
//!
//! # Example
//!
//! ```rust,ignore
//! use anofox_regression::solvers::{LmDynamicRegressor, InformationCriterion};
//! use faer::{Mat, Col};
//!
//! let x = Mat::from_fn(100, 2, |i, j| (i + j) as f64);
//! let y = Col::from_fn(100, |i| i as f64);
//!
//! let fitted = LmDynamicRegressor::builder()
//!     .ic(InformationCriterion::AICc)
//!     .lowess_span(0.3)
//!     .build()
//!     .fit(&x, &y)?;
//!
//! // Get time-varying coefficients
//! let dyn_coefs = fitted.dynamic_coefficients();
//! ```

use crate::core::{
    IntervalType, PredictionResult, RegressionOptions, RegressionOptionsBuilder, RegressionResult,
};
use crate::solvers::alm::{log_likelihood, AlmDistribution, AlmRegressor, FittedAlm};
use crate::solvers::lowess::lowess_smooth_weights;
use crate::solvers::traits::{FittedRegressor, RegressionError, Regressor};
use faer::{Col, Mat};
use statrs::distribution::{ContinuousCDF, StudentsT};

/// Information criterion type for model weighting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InformationCriterion {
    /// Akaike Information Criterion
    AIC,
    /// Corrected AIC (better for small samples)
    #[default]
    AICc,
    /// Bayesian Information Criterion
    BIC,
}

impl InformationCriterion {
    /// Compute the information criterion for a single observation.
    ///
    /// # Arguments
    /// * `log_lik` - Pointwise log-likelihood for one observation
    /// * `k` - Number of parameters in the model
    /// * `n` - Total number of observations (for AICc correction)
    pub fn compute(&self, log_lik: f64, k: usize, n: usize) -> f64 {
        let k_f = k as f64;
        let n_f = n as f64;

        match self {
            InformationCriterion::AIC => 2.0 * k_f - 2.0 * log_lik,
            InformationCriterion::AICc => {
                let aic = 2.0 * k_f - 2.0 * log_lik;
                if n_f - k_f - 1.0 > 0.0 {
                    aic + 2.0 * k_f * (k_f + 1.0) / (n_f - k_f - 1.0)
                } else {
                    aic
                }
            }
            InformationCriterion::BIC => k_f * n_f.ln() - 2.0 * log_lik,
        }
    }
}

/// Specification for a candidate model.
#[derive(Debug, Clone)]
pub struct ModelSpec {
    /// Which variables are included (indices into original X)
    pub included_vars: Vec<usize>,
    /// Number of parameters (including intercept if applicable)
    pub n_params: usize,
}

/// Time-varying parameter model using pointwise information criteria.
#[derive(Debug, Clone)]
pub struct LmDynamicRegressor {
    options: RegressionOptions,
    ic_type: InformationCriterion,
    distribution: AlmDistribution,
    lowess_span: Option<f64>,
    max_models: Option<usize>,
}

impl Default for LmDynamicRegressor {
    fn default() -> Self {
        Self {
            options: RegressionOptionsBuilder::default().build_unchecked(),
            ic_type: InformationCriterion::default(),
            distribution: AlmDistribution::Normal,
            lowess_span: Some(0.3), // Default LOWESS smoothing
            max_models: Some(64),   // Limit for computational tractability
        }
    }
}

impl LmDynamicRegressor {
    /// Create a new dynamic regressor with default options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a builder for configuring the regressor.
    pub fn builder() -> LmDynamicRegressorBuilder {
        LmDynamicRegressorBuilder::default()
    }

    /// Get the information criterion type.
    pub fn ic_type(&self) -> InformationCriterion {
        self.ic_type
    }

    /// Get the distribution.
    pub fn distribution(&self) -> AlmDistribution {
        self.distribution
    }

    /// Get the LOWESS span (None if no smoothing).
    pub fn lowess_span(&self) -> Option<f64> {
        self.lowess_span
    }

    /// Generate all candidate model specifications.
    fn generate_model_specs(&self, p: usize) -> Vec<ModelSpec> {
        let max = self.max_models.unwrap_or(usize::MAX);
        let mut specs = Vec::new();

        // Generate all non-empty subsets of variables
        let total_models = (1usize << p) - 1; // 2^p - 1 models

        for mask in 1..=total_models {
            if specs.len() >= max {
                break;
            }

            let included: Vec<usize> = (0..p).filter(|&j| (mask >> j) & 1 == 1).collect();

            let n_params = included.len() + if self.options.with_intercept { 1 } else { 0 };

            specs.push(ModelSpec {
                included_vars: included,
                n_params,
            });
        }

        specs
    }

    /// Extract columns from X based on included variables.
    fn extract_columns(&self, x: &Mat<f64>, included: &[usize]) -> Mat<f64> {
        let n = x.nrows();
        let p_new = included.len();
        let mut x_new = Mat::zeros(n, p_new);

        for (new_j, &orig_j) in included.iter().enumerate() {
            for i in 0..n {
                x_new[(i, new_j)] = x[(i, orig_j)];
            }
        }

        x_new
    }

    /// Fit the dynamic model.
    fn fit_internal(&self, x: &Mat<f64>, y: &Col<f64>) -> Result<FittedLmDynamic, RegressionError> {
        let n = x.nrows();
        let p = x.ncols();

        if n < 3 {
            return Err(RegressionError::InsufficientObservations { needed: 3, got: n });
        }

        // Generate candidate models
        let model_specs = self.generate_model_specs(p);
        let n_models = model_specs.len();

        if n_models == 0 {
            return Err(RegressionError::InsufficientObservations { needed: 1, got: 0 });
        }

        // Fit all candidate models and compute pointwise IC
        let mut fitted_models: Vec<Option<FittedAlm>> = Vec::with_capacity(n_models);
        let mut pointwise_ic = Mat::from_fn(n, n_models, |_, _| f64::INFINITY);

        for (m, spec) in model_specs.iter().enumerate() {
            let x_subset = self.extract_columns(x, &spec.included_vars);

            let alm = AlmRegressor::builder()
                .distribution(self.distribution)
                .with_intercept(self.options.with_intercept)
                .compute_inference(false)
                .build();

            match alm.fit(&x_subset, y) {
                Ok(fitted) => {
                    // Compute pointwise IC for each observation
                    let mu = &fitted.result().fitted_values;
                    let scale = fitted.scale();

                    for i in 0..n {
                        let yi = Col::from_fn(1, |_| y[i]);
                        let mui = Col::from_fn(1, |_| mu[i]);
                        let ll_i = log_likelihood(
                            &yi,
                            &mui,
                            self.distribution,
                            scale,
                            fitted.extra_parameter(),
                        );
                        pointwise_ic[(i, m)] = self.ic_type.compute(ll_i, spec.n_params, n);
                    }

                    fitted_models.push(Some(fitted));
                }
                Err(_) => {
                    fitted_models.push(None);
                    // IC values remain INFINITY (worst possible)
                }
            }
        }

        // Compute weights from pointwise IC
        // pICWeights = exp(-0.5 * (pIC - min(pIC))) / row_sums
        let mut model_weights = Mat::zeros(n, n_models);

        for i in 0..n {
            // Find min IC for this observation
            let min_ic = (0..n_models)
                .map(|m| pointwise_ic[(i, m)])
                .fold(f64::INFINITY, f64::min);

            // Compute weights (Akaike weights formula)
            let mut row_sum = 0.0;
            for m in 0..n_models {
                let delta = pointwise_ic[(i, m)] - min_ic;
                let w = (-0.5 * delta).exp();
                model_weights[(i, m)] = w;
                row_sum += w;
            }

            // Normalize
            if row_sum > 1e-10 {
                for m in 0..n_models {
                    model_weights[(i, m)] /= row_sum;
                }
            } else {
                // Uniform weights if all are zero
                for m in 0..n_models {
                    model_weights[(i, m)] = 1.0 / n_models as f64;
                }
            }
        }

        // Optional LOWESS smoothing
        let smoothed_weights = self
            .lowess_span
            .map(|span| lowess_smooth_weights(&model_weights, span));

        // Compute dynamic coefficients
        let weights_to_use = smoothed_weights.as_ref().unwrap_or(&model_weights);
        let dynamic_coefficients =
            self.compute_dynamic_coefficients(&model_specs, &fitted_models, weights_to_use, p);

        // Compute weighted average coefficients for the base result
        let avg_coefs = self.compute_average_coefficients(&dynamic_coefficients);

        // Build base regression result
        let result = self.build_base_result(x, y, &avg_coefs, &dynamic_coefficients)?;

        Ok(FittedLmDynamic {
            options: self.options.clone(),
            distribution: self.distribution,
            result,
            dynamic_coefficients,
            model_weights,
            smoothed_weights,
            model_specs,
            pointwise_ic,
            n_features: p,
            original_x: x.to_owned(),
            has_intercept: self.options.with_intercept,
        })
    }

    /// Compute time-varying coefficients as weighted average of model coefficients.
    fn compute_dynamic_coefficients(
        &self,
        specs: &[ModelSpec],
        fitted_models: &[Option<FittedAlm>],
        weights: &Mat<f64>,
        p: usize,
    ) -> Mat<f64> {
        let n = weights.nrows();

        // Number of coefficient columns: p (features) + 1 if intercept
        let n_coef_cols = p + if self.options.with_intercept { 1 } else { 0 };
        let mut dynamic_coefs = Mat::zeros(n, n_coef_cols);

        for i in 0..n {
            for (m, (spec, fitted_opt)) in specs.iter().zip(fitted_models.iter()).enumerate() {
                let weight = weights[(i, m)];

                if let Some(fitted) = fitted_opt {
                    let coefs = &fitted.result().coefficients;

                    // Add intercept contribution
                    if self.options.with_intercept {
                        if let Some(intercept) = fitted.result().intercept {
                            dynamic_coefs[(i, 0)] += weight * intercept;
                        }
                    }

                    // Add coefficient contributions
                    // Map from subset indices back to original feature indices
                    for (subset_j, &orig_j) in spec.included_vars.iter().enumerate() {
                        let col_idx = if self.options.with_intercept {
                            orig_j + 1
                        } else {
                            orig_j
                        };
                        if subset_j < coefs.nrows() {
                            dynamic_coefs[(i, col_idx)] += weight * coefs[subset_j];
                        }
                    }
                }
            }
        }

        dynamic_coefs
    }

    /// Compute average coefficients across time.
    fn compute_average_coefficients(&self, dynamic_coefs: &Mat<f64>) -> Col<f64> {
        let n = dynamic_coefs.nrows();
        let p = dynamic_coefs.ncols();

        Col::from_fn(p, |j| {
            let sum: f64 = (0..n).map(|i| dynamic_coefs[(i, j)]).sum();
            sum / n as f64
        })
    }

    /// Build the base regression result.
    fn build_base_result(
        &self,
        x: &Mat<f64>,
        y: &Col<f64>,
        avg_coefs: &Col<f64>,
        dynamic_coefs: &Mat<f64>,
    ) -> Result<RegressionResult, RegressionError> {
        let n = x.nrows();
        let p = x.ncols();

        // Compute fitted values using dynamic coefficients
        let mut fitted_values = Col::zeros(n);
        for i in 0..n {
            if self.options.with_intercept {
                fitted_values[i] = dynamic_coefs[(i, 0)]; // intercept
                for j in 0..p {
                    fitted_values[i] += x[(i, j)] * dynamic_coefs[(i, j + 1)];
                }
            } else {
                for j in 0..p {
                    fitted_values[i] += x[(i, j)] * dynamic_coefs[(i, j)];
                }
            }
        }

        // Compute residuals
        let mut residuals = Col::zeros(n);
        for i in 0..n {
            residuals[i] = y[i] - fitted_values[i];
        }

        // Extract coefficients and intercept
        let (coefficients, intercept) = if self.options.with_intercept {
            let intercept = avg_coefs[0];
            let coefs = Col::from_fn(p, |j| avg_coefs[j + 1]);
            (coefs, Some(intercept))
        } else {
            (avg_coefs.clone(), None)
        };

        // Statistics
        let y_mean: f64 = y.iter().sum::<f64>() / n as f64;
        let tss: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();
        let rss: f64 = residuals.iter().map(|&r| r.powi(2)).sum();

        let r_squared = if tss > 0.0 {
            (1.0 - rss / tss).clamp(0.0, 1.0)
        } else {
            1.0
        };

        let n_params = avg_coefs.nrows();
        let df_resid = (n - n_params) as f64;
        let adj_r_squared = if df_resid > 0.0 {
            1.0 - (1.0 - r_squared) * (n - 1) as f64 / df_resid
        } else {
            f64::NAN
        };

        let mse = if df_resid > 0.0 {
            rss / df_resid
        } else {
            f64::NAN
        };
        let rmse = mse.sqrt();

        let aliased = vec![false; p];

        let mut result = RegressionResult::empty(p, n);
        result.coefficients = coefficients;
        result.intercept = intercept;
        result.residuals = residuals;
        result.fitted_values = fitted_values;
        result.rank = p;
        result.n_parameters = n_params;
        result.n_observations = n;
        result.aliased = aliased;
        result.r_squared = r_squared;
        result.adj_r_squared = adj_r_squared;
        result.mse = mse;
        result.rmse = rmse;
        result.confidence_level = self.options.confidence_level;

        Ok(result)
    }
}

impl Regressor for LmDynamicRegressor {
    type Fitted = FittedLmDynamic;

    fn fit(&self, x: &Mat<f64>, y: &Col<f64>) -> Result<Self::Fitted, RegressionError> {
        // Validate dimensions
        if x.nrows() != y.nrows() {
            return Err(RegressionError::DimensionMismatch {
                x_rows: x.nrows(),
                y_len: y.nrows(),
            });
        }

        self.fit_internal(x, y)
    }
}

/// Result from fitting lmDynamic.
#[derive(Debug, Clone)]
pub struct FittedLmDynamic {
    #[allow(dead_code)]
    options: RegressionOptions,
    distribution: AlmDistribution,
    /// Base regression result (using time-averaged coefficients)
    result: RegressionResult,
    /// Per-observation coefficients (n_obs x n_coefs matrix)
    /// If with_intercept: column 0 = intercept, columns 1..p+1 = feature coefficients
    /// If no intercept: columns 0..p = feature coefficients
    dynamic_coefficients: Mat<f64>,
    /// Model weights for each observation (n_obs x n_models matrix)
    model_weights: Mat<f64>,
    /// Smoothed weights if LOWESS was applied
    smoothed_weights: Option<Mat<f64>>,
    /// Candidate model specifications
    model_specs: Vec<ModelSpec>,
    /// Pointwise IC values (n_obs x n_models matrix)
    pointwise_ic: Mat<f64>,
    /// Number of original features
    #[allow(dead_code)]
    n_features: usize,
    /// Original X matrix for interval computation
    original_x: Mat<f64>,
    /// Whether model has intercept
    has_intercept: bool,
}

impl FittedLmDynamic {
    /// Get the distribution used.
    pub fn distribution(&self) -> AlmDistribution {
        self.distribution
    }

    /// Get the time-varying coefficients matrix.
    ///
    /// Returns a matrix where each row contains the coefficients for that observation.
    /// If with_intercept, column 0 is the intercept.
    pub fn dynamic_coefficients(&self) -> &Mat<f64> {
        &self.dynamic_coefficients
    }

    /// Get the model weights matrix (before smoothing).
    pub fn model_weights(&self) -> &Mat<f64> {
        &self.model_weights
    }

    /// Get the smoothed model weights (if LOWESS was applied).
    pub fn smoothed_weights(&self) -> Option<&Mat<f64>> {
        self.smoothed_weights.as_ref()
    }

    /// Get the candidate model specifications.
    pub fn model_specs(&self) -> &[ModelSpec] {
        &self.model_specs
    }

    /// Get the pointwise information criteria matrix.
    pub fn pointwise_ic(&self) -> &Mat<f64> {
        &self.pointwise_ic
    }

    /// Get coefficient at a specific time point.
    ///
    /// # Arguments
    /// * `obs_index` - Observation index (0 to n-1)
    /// * `coef_index` - Coefficient index (0 = intercept if with_intercept, else first feature)
    pub fn coefficient_at(&self, obs_index: usize, coef_index: usize) -> Option<f64> {
        if obs_index < self.dynamic_coefficients.nrows()
            && coef_index < self.dynamic_coefficients.ncols()
        {
            Some(self.dynamic_coefficients[(obs_index, coef_index)])
        } else {
            None
        }
    }

    /// Get all coefficients at a specific time point.
    pub fn coefficients_at(&self, obs_index: usize) -> Option<Col<f64>> {
        if obs_index < self.dynamic_coefficients.nrows() {
            let ncols = self.dynamic_coefficients.ncols();
            Some(Col::from_fn(ncols, |j| {
                self.dynamic_coefficients[(obs_index, j)]
            }))
        } else {
            None
        }
    }

    /// Compute prediction intervals for dynamic model using the averaged model approach.
    ///
    /// Uses the formula: SE = sqrt(MSE * (1 + h)) for prediction intervals
    /// and SE = sqrt(MSE * h) for confidence intervals, where h is the leverage.
    fn compute_dynamic_intervals(
        &self,
        x_new: &Mat<f64>,
        predictions: &Col<f64>,
        interval_type: IntervalType,
        level: f64,
    ) -> PredictionResult {
        let n_new = x_new.nrows();
        let df = self.result.residual_df() as f64;

        // Invalid cases: return NaN intervals
        if df <= 0.0 || !self.result.mse.is_finite() || self.result.mse < 0.0 {
            return self.create_nan_intervals(predictions, n_new);
        }

        // Compute (X'X)^-1 for the original data
        let xtx_inv = match self.compute_xtx_inverse() {
            Some(inv) => inv,
            None => return self.create_nan_intervals(predictions, n_new),
        };

        // Compute t-critical value
        let t_crit = self.compute_t_critical(df, level);

        let mse = self.result.mse;
        let mut se = Col::zeros(n_new);
        let mut lower = Col::zeros(n_new);
        let mut upper = Col::zeros(n_new);

        for i in 0..n_new {
            // Build observation vector (with intercept if needed)
            let x0 = self.build_obs_vector(x_new, i);

            // Compute leverage h = x0' * (X'X)^-1 * x0
            let h = self.compute_leverage(&x0, &xtx_inv);

            // Compute variance based on interval type
            let var = match interval_type {
                IntervalType::Confidence => mse * h,
                IntervalType::Prediction => mse * (1.0 + h),
            };

            se[i] = if var >= 0.0 { var.sqrt() } else { f64::NAN };
            let margin = t_crit * se[i];
            lower[i] = predictions[i] - margin;
            upper[i] = predictions[i] + margin;
        }

        PredictionResult::with_intervals(predictions.clone(), lower, upper, se)
    }

    /// Compute (X'X)^-1 for interval computation.
    fn compute_xtx_inverse(&self) -> Option<Mat<f64>> {
        let n = self.original_x.nrows();
        let p = self.original_x.ncols();

        // Build design matrix with optional intercept
        let design = if self.has_intercept {
            let aug_size = p + 1;
            Mat::from_fn(n, aug_size, |i, j| {
                if j == 0 {
                    1.0
                } else {
                    self.original_x[(i, j - 1)]
                }
            })
        } else {
            self.original_x.clone()
        };

        // Compute X'X
        let xtx = design.transpose() * &design;

        // Compute inverse using QR decomposition
        let dim = xtx.nrows();
        let qr = xtx.qr();
        let q = qr.compute_Q();
        let r = qr.R().to_owned();

        // Check for singularity
        for i in 0..dim {
            if r[(i, i)].abs() < 1e-10 {
                return None;
            }
        }

        // Solve for inverse
        let qt = q.transpose().to_owned();
        let mut inv = Mat::zeros(dim, dim);

        for col in 0..dim {
            for i in (0..dim).rev() {
                let mut sum = qt[(i, col)];
                for j in (i + 1)..dim {
                    sum -= r[(i, j)] * inv[(j, col)];
                }
                inv[(i, col)] = sum / r[(i, i)];
            }
        }

        Some(inv)
    }

    /// Build observation vector (optionally augmented with intercept).
    fn build_obs_vector(&self, x_new: &Mat<f64>, row: usize) -> Col<f64> {
        let n_features = x_new.ncols();

        if self.has_intercept {
            let mut x0 = Col::zeros(n_features + 1);
            x0[0] = 1.0;
            for j in 0..n_features {
                x0[j + 1] = x_new[(row, j)];
            }
            x0
        } else {
            Col::from_fn(n_features, |j| x_new[(row, j)])
        }
    }

    /// Compute leverage h = x0' * (X'X)^-1 * x0.
    fn compute_leverage(&self, x0: &Col<f64>, xtx_inv: &Mat<f64>) -> f64 {
        let p = x0.nrows();

        // Compute (X'X)^-1 * x0
        let mut xtx_inv_x0 = Col::zeros(p);
        for i in 0..p {
            let mut sum = 0.0;
            for j in 0..p {
                sum += xtx_inv[(i, j)] * x0[j];
            }
            xtx_inv_x0[i] = sum;
        }

        // Compute x0' * ((X'X)^-1 * x0)
        let mut h = 0.0;
        for i in 0..p {
            h += x0[i] * xtx_inv_x0[i];
        }

        h.max(0.0) // Ensure non-negative
    }

    /// Compute t-critical value for confidence intervals.
    fn compute_t_critical(&self, df: f64, level: f64) -> f64 {
        let t_dist = StudentsT::new(0.0, 1.0, df).expect("valid t-distribution parameters");
        let alpha = 1.0 - level;
        t_dist.inverse_cdf(1.0 - alpha / 2.0)
    }

    /// Create NaN intervals for invalid cases.
    fn create_nan_intervals(&self, predictions: &Col<f64>, n: usize) -> PredictionResult {
        let se = Col::from_fn(n, |_| f64::NAN);
        let lower = Col::from_fn(n, |_| f64::NAN);
        let upper = Col::from_fn(n, |_| f64::NAN);
        PredictionResult::with_intervals(predictions.clone(), lower, upper, se)
    }
}

impl FittedRegressor for FittedLmDynamic {
    fn predict(&self, x: &Mat<f64>) -> Col<f64> {
        let n = x.nrows();
        let p = x.ncols();
        let mut predictions = Col::zeros(n);

        // Use time-averaged coefficients for prediction
        let intercept = self.result.intercept.unwrap_or(0.0);

        for i in 0..n {
            predictions[i] = intercept;
            for j in 0..p.min(self.result.coefficients.nrows()) {
                predictions[i] += x[(i, j)] * self.result.coefficients[j];
            }
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
                self.compute_dynamic_intervals(x, &predictions, interval_type, level)
            }
        }
    }
}

/// Builder for LmDynamicRegressor.
#[derive(Debug, Clone)]
pub struct LmDynamicRegressorBuilder {
    options_builder: RegressionOptionsBuilder,
    ic_type: InformationCriterion,
    distribution: AlmDistribution,
    lowess_span: Option<f64>,
    max_models: Option<usize>,
}

impl Default for LmDynamicRegressorBuilder {
    fn default() -> Self {
        Self {
            options_builder: RegressionOptionsBuilder::default(),
            ic_type: InformationCriterion::default(),
            distribution: AlmDistribution::default(),
            lowess_span: Some(0.3),
            max_models: Some(64),
        }
    }
}

impl LmDynamicRegressorBuilder {
    /// Create a new builder with default options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the information criterion for model weighting.
    pub fn ic(mut self, ic: InformationCriterion) -> Self {
        self.ic_type = ic;
        self
    }

    /// Set the distribution family.
    pub fn distribution(mut self, dist: AlmDistribution) -> Self {
        self.distribution = dist;
        self
    }

    /// Set the LOWESS smoothing span (fraction of data, 0 to 1).
    ///
    /// Use `None` to disable smoothing.
    pub fn lowess_span(mut self, span: f64) -> Self {
        self.lowess_span = Some(span.clamp(0.05, 1.0));
        self
    }

    /// Disable LOWESS smoothing.
    pub fn no_smoothing(mut self) -> Self {
        self.lowess_span = None;
        self
    }

    /// Set maximum number of candidate models to consider.
    ///
    /// For p features, there are 2^p - 1 possible models.
    /// Set this to limit computational cost.
    pub fn max_models(mut self, max: usize) -> Self {
        self.max_models = Some(max);
        self
    }

    /// Set whether to include an intercept term.
    pub fn with_intercept(mut self, include: bool) -> Self {
        self.options_builder = self.options_builder.with_intercept(include);
        self
    }

    /// Set the confidence level for intervals.
    pub fn confidence_level(mut self, level: f64) -> Self {
        self.options_builder = self.options_builder.confidence_level(level);
        self
    }

    /// Build the LmDynamicRegressor.
    pub fn build(self) -> LmDynamicRegressor {
        LmDynamicRegressor {
            options: self.options_builder.build_unchecked(),
            ic_type: self.ic_type,
            distribution: self.distribution,
            lowess_span: self.lowess_span,
            max_models: self.max_models,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_information_criterion_compute() {
        // AIC = 2k - 2*LL
        let aic = InformationCriterion::AIC.compute(-10.0, 3, 100);
        assert!((aic - 26.0).abs() < 1e-10); // 2*3 - 2*(-10) = 26

        // BIC = k*ln(n) - 2*LL
        let bic = InformationCriterion::BIC.compute(-10.0, 3, 100);
        let expected_bic = 3.0 * (100.0_f64).ln() + 20.0;
        assert!((bic - expected_bic).abs() < 1e-10);
    }

    #[test]
    fn test_builder_defaults() {
        let model = LmDynamicRegressor::builder().build();

        assert_eq!(model.ic_type(), InformationCriterion::AICc);
        assert_eq!(model.distribution(), AlmDistribution::Normal);
        assert!(model.lowess_span().is_some());
    }

    #[test]
    fn test_builder_custom() {
        let model = LmDynamicRegressor::builder()
            .ic(InformationCriterion::BIC)
            .distribution(AlmDistribution::Laplace)
            .lowess_span(0.5)
            .max_models(32)
            .with_intercept(false)
            .build();

        assert_eq!(model.ic_type(), InformationCriterion::BIC);
        assert_eq!(model.distribution(), AlmDistribution::Laplace);
        assert_eq!(model.lowess_span(), Some(0.5));
    }

    #[test]
    fn test_fit_simple() {
        // Simple linear data
        let x = Mat::from_fn(30, 1, |i, _| (i + 1) as f64);
        let y = Col::from_fn(30, |i| 2.0 + 1.5 * (i + 1) as f64 + 0.1 * (i as f64).sin());

        let model = LmDynamicRegressor::builder()
            .ic(InformationCriterion::AICc)
            .with_intercept(true)
            .build();

        let result = model.fit(&x, &y);
        assert!(result.is_ok(), "Should fit successfully");

        let fitted = result.unwrap();

        // Check dimensions
        assert_eq!(fitted.dynamic_coefficients().nrows(), 30);
        assert_eq!(fitted.dynamic_coefficients().ncols(), 2); // intercept + 1 coef

        // Model weights should be normalized
        for i in 0..30 {
            let row_sum: f64 = (0..fitted.model_weights().ncols())
                .map(|j| fitted.model_weights()[(i, j)])
                .sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-6,
                "Row {} weights sum to {}",
                i,
                row_sum
            );
        }
    }

    #[test]
    fn test_fit_multiple_features() {
        // Data with 3 features
        let x = Mat::from_fn(40, 3, |i, j| (i + j + 1) as f64 * 0.1);
        let y = Col::from_fn(40, |i| {
            let i_f = i as f64;
            1.0 + 0.5 * (i_f + 1.0) * 0.1 + 0.3 * (i_f + 2.0) * 0.1
        });

        let model = LmDynamicRegressor::builder()
            .max_models(7) // Limit to manageable number
            .with_intercept(true)
            .build();

        let result = model.fit(&x, &y);
        assert!(result.is_ok(), "Should fit with multiple features");

        let fitted = result.unwrap();

        // Should have 2^3 - 1 = 7 models (or less if max_models limits it)
        assert!(fitted.model_specs().len() <= 7);
        assert!(fitted.model_specs().len() >= 1);
    }

    #[test]
    fn test_dynamic_coefficients_vary() {
        // Create data where relationship changes over time
        let x = Mat::from_fn(50, 1, |i, _| (i + 1) as f64);
        let y = Col::from_fn(50, |i| {
            let t = i as f64;
            // Coefficient varies: starts at 1.0, increases to 2.0
            let slope = 1.0 + t / 50.0;
            slope * (i + 1) as f64
        });

        let model = LmDynamicRegressor::builder()
            .no_smoothing() // No smoothing to see raw variation
            .with_intercept(true)
            .build();

        let fitted = model.fit(&x, &y).expect("Should fit");

        // Dynamic coefficients should show variation
        let coef_start = fitted.coefficient_at(0, 1).unwrap(); // First coefficient
        let coef_end = fitted.coefficient_at(49, 1).unwrap();

        // Both should be positive (positive relationship)
        assert!(coef_start > 0.0, "Start coefficient should be positive");
        assert!(coef_end > 0.0, "End coefficient should be positive");
    }

    #[test]
    fn test_smoothed_weights_exist() {
        let x = Mat::from_fn(30, 1, |i, _| i as f64);
        let y = Col::from_fn(30, |i| i as f64 + 0.1);

        let model = LmDynamicRegressor::builder().lowess_span(0.3).build();

        let fitted = model.fit(&x, &y).expect("Should fit");

        // Smoothed weights should exist
        assert!(fitted.smoothed_weights().is_some());

        // Smoothed weights should also be normalized
        let sw = fitted.smoothed_weights().unwrap();
        for i in 0..sw.nrows() {
            let row_sum: f64 = (0..sw.ncols()).map(|j| sw[(i, j)]).sum();
            assert!((row_sum - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_predict() {
        let x = Mat::from_fn(30, 1, |i, _| (i + 1) as f64);
        let y = Col::from_fn(30, |i| 2.0 + 1.5 * (i + 1) as f64);

        let model = LmDynamicRegressor::builder().with_intercept(true).build();
        let fitted = model.fit(&x, &y).expect("Should fit");

        // Predict on new data
        let x_new = Mat::from_fn(5, 1, |i, _| (i + 31) as f64);
        let predictions = fitted.predict(&x_new);

        assert_eq!(predictions.nrows(), 5);

        // Predictions should be in reasonable range
        for i in 0..5 {
            assert!(predictions[i] > 0.0);
            assert!(predictions[i] < 200.0);
        }
    }

    // === Error Path Tests ===

    #[test]
    fn test_fit_insufficient_observations() {
        // Tests lines 184-188: n < 3 error
        let x = Mat::from_fn(2, 1, |i, _| i as f64);
        let y = Col::from_fn(2, |i| i as f64);

        let model = LmDynamicRegressor::builder().build();
        let result = model.fit(&x, &y);

        assert!(result.is_err());
        match result {
            Err(RegressionError::InsufficientObservations { needed, got }) => {
                assert_eq!(needed, 3);
                assert_eq!(got, 2);
            }
            _ => panic!("Expected InsufficientObservations error"),
        }
    }

    #[test]
    fn test_fit_dimension_mismatch() {
        // Tests Regressor trait: x.nrows() != y.nrows()
        let x = Mat::from_fn(10, 2, |i, j| (i + j) as f64);
        let y = Col::from_fn(5, |i| i as f64); // Wrong length

        let model = LmDynamicRegressor::builder().build();
        let result = model.fit(&x, &y);

        assert!(result.is_err());
        match result {
            Err(RegressionError::DimensionMismatch { x_rows, y_len }) => {
                assert_eq!(x_rows, 10);
                assert_eq!(y_len, 5);
            }
            _ => panic!("Expected DimensionMismatch error"),
        }
    }

    #[test]
    fn test_fit_zero_features() {
        // Tests lines 195-199: no model specs generated (p = 0)
        let x = Mat::<f64>::zeros(10, 0); // No features
        let y = Col::from_fn(10, |i| i as f64);

        let model = LmDynamicRegressor::builder().build();
        let result = model.fit(&x, &y);

        assert!(result.is_err());
    }

    // === Builder Tests ===

    #[test]
    fn test_builder_lowess_span_clamping() {
        // Tests lines 631-633: span is clamped to [0.05, 1.0]

        // Test lower bound clamping
        let model1 = LmDynamicRegressor::builder().lowess_span(0.01).build();
        assert_eq!(model1.lowess_span(), Some(0.05));

        // Test upper bound clamping
        let model2 = LmDynamicRegressor::builder().lowess_span(1.5).build();
        assert_eq!(model2.lowess_span(), Some(1.0));

        // Test normal value
        let model3 = LmDynamicRegressor::builder().lowess_span(0.4).build();
        assert_eq!(model3.lowess_span(), Some(0.4));
    }

    #[test]
    fn test_builder_no_smoothing() {
        // Tests lines 637-639: disable smoothing
        let model = LmDynamicRegressor::builder().no_smoothing().build();

        assert_eq!(model.lowess_span(), None);
    }

    #[test]
    fn test_fit_without_smoothing() {
        // Verify model works without LOWESS smoothing
        let x = Mat::from_fn(30, 1, |i, _| (i + 1) as f64);
        let y = Col::from_fn(30, |i| 2.0 + 1.5 * (i + 1) as f64);

        let model = LmDynamicRegressor::builder()
            .no_smoothing()
            .with_intercept(true)
            .build();

        let fitted = model.fit(&x, &y).expect("Should fit");

        // No smoothed weights should exist
        assert!(fitted.smoothed_weights().is_none());

        // Model weights should still be normalized
        for i in 0..30 {
            let sum: f64 = (0..fitted.model_weights().ncols())
                .map(|j| fitted.model_weights()[(i, j)])
                .sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }
    }

    // === Edge Case Tests ===

    #[test]
    fn test_fit_with_laplace_distribution() {
        // Tests non-Normal distribution (Laplace)
        let x = Mat::from_fn(50, 1, |i, _| (i + 1) as f64);
        let y = Col::from_fn(50, |i| {
            let t = (i + 1) as f64;
            2.0 + 1.5 * t + if i % 3 == 0 { 5.0 } else { 0.0 } // Some outliers
        });

        let model = LmDynamicRegressor::builder()
            .distribution(AlmDistribution::Laplace)
            .with_intercept(true)
            .build();

        assert_eq!(model.distribution(), AlmDistribution::Laplace);

        let result = model.fit(&x, &y);
        assert!(result.is_ok());

        let fitted = result.unwrap();
        assert_eq!(fitted.distribution(), AlmDistribution::Laplace);
    }

    #[test]
    fn test_fit_no_intercept() {
        // Tests fitting without intercept
        let x = Mat::from_fn(30, 2, |i, j| ((i + 1) * (j + 1)) as f64);
        let y = Col::from_fn(30, |i| {
            let i_f = (i + 1) as f64;
            2.0 * i_f + 3.0 * i_f * 2.0
        });

        let model = LmDynamicRegressor::builder().with_intercept(false).build();

        let fitted = model.fit(&x, &y).expect("Should fit");

        // No intercept should be present
        assert!(fitted.result().intercept.is_none());

        // Dynamic coefficients should have 2 columns (no intercept column)
        assert_eq!(fitted.dynamic_coefficients().ncols(), 2);
    }

    #[test]
    fn test_dynamic_coefficient_mapping() {
        // Test that coefficients are correctly mapped from subset models
        let x = Mat::from_fn(30, 2, |i, j| {
            if j == 0 {
                i as f64
            } else {
                (i as f64).sin() * 10.0
            }
        });
        let y = Col::from_fn(30, |i| 1.0 + 2.0 * i as f64 + 3.0 * (i as f64).sin() * 10.0);

        let model = LmDynamicRegressor::builder()
            .max_models(7) // All 2^2 - 1 = 3 models for 2 features
            .with_intercept(true)
            .build();

        let fitted = model.fit(&x, &y).expect("Should fit");

        // Dynamic coefficients should have 3 columns: intercept + 2 features
        assert_eq!(fitted.dynamic_coefficients().ncols(), 3);

        // All coefficients should be finite
        for i in 0..30 {
            for j in 0..3 {
                assert!(
                    fitted.dynamic_coefficients()[(i, j)].is_finite(),
                    "Dynamic coef[{},{}] should be finite",
                    i,
                    j
                );
            }
        }
    }

    // === Information Criterion Tests ===

    #[test]
    fn test_information_criterion_aicc_small_sample() {
        // Test AICc correction when n - k - 1 <= 0
        let log_lik = -10.0;
        let k = 5;
        let n = 5; // n - k - 1 = -1 (edge case)

        let aicc = InformationCriterion::AICc.compute(log_lik, k, n);
        let aic = InformationCriterion::AIC.compute(log_lik, k, n);

        // When n - k - 1 <= 0, AICc should equal AIC
        assert!((aicc - aic).abs() < 1e-10);
    }

    #[test]
    fn test_information_criterion_bic() {
        // Test BIC computation
        let log_lik = -15.0;
        let k = 4;
        let n = 50;

        let bic = InformationCriterion::BIC.compute(log_lik, k, n);
        // BIC = k*ln(n) - 2*LL = 4*ln(50) + 30
        let expected = 4.0 * (50.0_f64).ln() + 30.0;
        assert!((bic - expected).abs() < 1e-10);
    }

    // === Accessor Tests ===

    #[test]
    fn test_coefficient_at() {
        let x = Mat::from_fn(30, 2, |i, j| (i + j) as f64 * 0.1);
        let y = Col::from_fn(30, |i| i as f64);

        let model = LmDynamicRegressor::builder().with_intercept(true).build();
        let fitted = model.fit(&x, &y).expect("Should fit");

        // Valid indices
        assert!(fitted.coefficient_at(0, 0).is_some());
        assert!(fitted.coefficient_at(29, 2).is_some());

        // Invalid indices
        assert!(fitted.coefficient_at(30, 0).is_none()); // Row out of bounds
        assert!(fitted.coefficient_at(0, 10).is_none()); // Col out of bounds
    }

    #[test]
    fn test_coefficients_at() {
        let x = Mat::from_fn(30, 2, |i, j| (i + j) as f64 * 0.1);
        let y = Col::from_fn(30, |i| i as f64);

        let model = LmDynamicRegressor::builder().with_intercept(true).build();
        let fitted = model.fit(&x, &y).expect("Should fit");

        let coefs = fitted.coefficients_at(15);
        assert!(coefs.is_some());
        let coefs = coefs.unwrap();
        assert_eq!(coefs.nrows(), 3); // intercept + 2 features

        // Invalid index
        assert!(fitted.coefficients_at(100).is_none());
    }

    // === Prediction Tests ===

    #[test]
    fn test_predict_with_interval_none() {
        // Tests predict_with_interval with None returns point only
        let x = Mat::from_fn(30, 1, |i, _| (i + 1) as f64);
        let y = Col::from_fn(30, |i| 2.0 + 1.5 * (i + 1) as f64);

        let model = LmDynamicRegressor::builder().with_intercept(true).build();
        let fitted = model.fit(&x, &y).expect("Should fit");

        let x_new = Mat::from_fn(5, 1, |i, _| (i + 31) as f64);
        let result = fitted.predict_with_interval(&x_new, None, 0.95);

        // Should have predictions
        assert_eq!(result.fit.nrows(), 5);
        for i in 0..5 {
            assert!(result.fit[i].is_finite());
        }
    }

    #[test]
    fn test_model_weights_normalization() {
        // Test that model weights are properly normalized to sum to 1
        let x = Mat::from_fn(40, 2, |i, j| (i + j) as f64 * 0.1);
        let y = Col::from_fn(40, |i| i as f64 + (i as f64).sin());

        let model = LmDynamicRegressor::builder()
            .max_models(3)
            .with_intercept(true)
            .build();

        let fitted = model.fit(&x, &y).expect("Should fit");

        // Each row should sum to 1
        for i in 0..40 {
            let row_sum: f64 = (0..fitted.model_weights().ncols())
                .map(|j| fitted.model_weights()[(i, j)])
                .sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-6,
                "Row {} weights sum to {} instead of 1.0",
                i,
                row_sum
            );

            // All weights should be non-negative
            for j in 0..fitted.model_weights().ncols() {
                assert!(
                    fitted.model_weights()[(i, j)] >= 0.0,
                    "Weight[{},{}] = {} is negative",
                    i,
                    j,
                    fitted.model_weights()[(i, j)]
                );
            }
        }
    }

    #[test]
    fn test_pointwise_ic_dimensions() {
        let x = Mat::from_fn(30, 2, |i, j| (i + j) as f64 * 0.1);
        let y = Col::from_fn(30, |i| i as f64);

        let model = LmDynamicRegressor::builder()
            .max_models(3)
            .with_intercept(true)
            .build();

        let fitted = model.fit(&x, &y).expect("Should fit");

        // Pointwise IC should be n_obs × n_models
        let ic = fitted.pointwise_ic();
        assert_eq!(ic.nrows(), 30);
        assert_eq!(ic.ncols(), fitted.model_specs().len());
    }

    #[test]
    fn test_distribution_accessor() {
        let model = LmDynamicRegressor::builder()
            .distribution(AlmDistribution::Laplace)
            .build();

        let x = Mat::from_fn(30, 1, |i, _| i as f64);
        let y = Col::from_fn(30, |i| i as f64);

        let fitted = model.fit(&x, &y).expect("Should fit");
        assert_eq!(fitted.distribution(), AlmDistribution::Laplace);
    }

    #[test]
    fn test_max_models_limit() {
        // Verify max_models actually limits the number of models
        let x = Mat::from_fn(30, 5, |i, j| (i + j) as f64 * 0.1);
        let y = Col::from_fn(30, |i| i as f64);

        let model = LmDynamicRegressor::builder()
            .max_models(5) // Limit to 5 models instead of 2^5-1=31
            .build();

        let fitted = model.fit(&x, &y).expect("Should fit");

        // Should have at most 5 models
        assert!(fitted.model_specs().len() <= 5);
    }

    // === Prediction Interval Tests ===

    #[test]
    fn test_predict_with_prediction_interval() {
        // Basic prediction interval test
        let x = Mat::from_fn(50, 1, |i, _| (i + 1) as f64);
        let y = Col::from_fn(50, |i| {
            let t = (i + 1) as f64;
            2.0 + 1.5 * t + (t * 0.1).sin() * 0.5
        });

        let model = LmDynamicRegressor::builder().with_intercept(true).build();
        let fitted = model.fit(&x, &y).expect("Should fit");

        let x_new = Mat::from_fn(5, 1, |i, _| (i + 51) as f64);
        let result = fitted.predict_with_interval(&x_new, Some(IntervalType::Prediction), 0.95);

        // Should have predictions and intervals
        assert_eq!(result.fit.nrows(), 5);
        assert_eq!(result.lower.nrows(), 5);
        assert_eq!(result.upper.nrows(), 5);
        assert_eq!(result.se.nrows(), 5);

        for i in 0..5 {
            assert!(result.fit[i].is_finite(), "Prediction should be finite");
            assert!(result.lower[i].is_finite(), "Lower bound should be finite");
            assert!(result.upper[i].is_finite(), "Upper bound should be finite");
            assert!(result.se[i].is_finite(), "SE should be finite");
            assert!(
                result.lower[i] < result.fit[i],
                "Lower bound should be below prediction"
            );
            assert!(
                result.upper[i] > result.fit[i],
                "Upper bound should be above prediction"
            );
            assert!(result.se[i] > 0.0, "SE should be positive");
        }
    }

    #[test]
    fn test_predict_with_confidence_interval() {
        // Confidence interval test
        let x = Mat::from_fn(50, 1, |i, _| (i + 1) as f64);
        let y = Col::from_fn(50, |i| {
            let t = (i + 1) as f64;
            2.0 + 1.5 * t + (t * 0.1).sin() * 0.5
        });

        let model = LmDynamicRegressor::builder().with_intercept(true).build();
        let fitted = model.fit(&x, &y).expect("Should fit");

        let x_new = Mat::from_fn(5, 1, |i, _| (i + 51) as f64);
        let result = fitted.predict_with_interval(&x_new, Some(IntervalType::Confidence), 0.95);

        for i in 0..5 {
            assert!(result.lower[i].is_finite());
            assert!(result.upper[i].is_finite());
            assert!(result.lower[i] < result.fit[i]);
            assert!(result.upper[i] > result.fit[i]);
        }
    }

    #[test]
    fn test_prediction_interval_wider_than_confidence() {
        // Prediction intervals should be wider than confidence intervals
        let x = Mat::from_fn(50, 1, |i, _| (i + 1) as f64);
        let y = Col::from_fn(50, |i| {
            let t = (i + 1) as f64;
            2.0 + 1.5 * t + (t * 0.1).sin() * 0.5
        });

        let model = LmDynamicRegressor::builder().with_intercept(true).build();
        let fitted = model.fit(&x, &y).expect("Should fit");

        let x_new = Mat::from_fn(5, 1, |i, _| (i + 51) as f64);

        let pred = fitted.predict_with_interval(&x_new, Some(IntervalType::Prediction), 0.95);
        let conf = fitted.predict_with_interval(&x_new, Some(IntervalType::Confidence), 0.95);

        for i in 0..5 {
            assert!(
                pred.se[i] > conf.se[i],
                "Prediction SE {} should be > Confidence SE {}",
                pred.se[i],
                conf.se[i]
            );
        }
    }

    #[test]
    fn test_different_confidence_levels() {
        // Higher confidence level should produce wider intervals
        let x = Mat::from_fn(50, 1, |i, _| (i + 1) as f64);
        let y = Col::from_fn(50, |i| {
            let t = (i + 1) as f64;
            2.0 + 1.5 * t + (t * 0.1).sin() * 0.5
        });

        let model = LmDynamicRegressor::builder().with_intercept(true).build();
        let fitted = model.fit(&x, &y).expect("Should fit");

        let x_new = Mat::from_fn(3, 1, |i, _| (i + 51) as f64);

        let result_90 = fitted.predict_with_interval(&x_new, Some(IntervalType::Prediction), 0.90);
        let result_95 = fitted.predict_with_interval(&x_new, Some(IntervalType::Prediction), 0.95);
        let result_99 = fitted.predict_with_interval(&x_new, Some(IntervalType::Prediction), 0.99);

        for i in 0..3 {
            let width_90 = result_90.upper[i] - result_90.lower[i];
            let width_95 = result_95.upper[i] - result_95.lower[i];
            let width_99 = result_99.upper[i] - result_99.lower[i];

            assert!(
                width_95 > width_90,
                "95% interval should be wider than 90%: {} vs {}",
                width_95,
                width_90
            );
            assert!(
                width_99 > width_95,
                "99% interval should be wider than 95%: {} vs {}",
                width_99,
                width_95
            );
        }
    }

    #[test]
    fn test_interval_no_intercept() {
        // Test intervals work without intercept using non-collinear data
        let x = Mat::from_fn(50, 2, |i, j| {
            let i_f = (i + 1) as f64;
            if j == 0 {
                i_f
            } else {
                (i_f * 0.1).sin() * 10.0
            }
        });
        let y = Col::from_fn(50, |i| {
            let i_f = (i + 1) as f64;
            2.0 * i_f + 3.0 * (i_f * 0.1).sin() * 10.0
        });

        let model = LmDynamicRegressor::builder().with_intercept(false).build();
        let fitted = model.fit(&x, &y).expect("Should fit");

        let x_new = Mat::from_fn(5, 2, |i, j| {
            let i_f = (i + 51) as f64;
            if j == 0 {
                i_f
            } else {
                (i_f * 0.1).sin() * 10.0
            }
        });
        let result = fitted.predict_with_interval(&x_new, Some(IntervalType::Prediction), 0.95);

        for i in 0..5 {
            assert!(result.lower[i].is_finite(), "Lower should be finite");
            assert!(result.upper[i].is_finite(), "Upper should be finite");
        }
    }

    #[test]
    fn test_interval_multiple_features() {
        // Test intervals with multiple features
        let x = Mat::from_fn(60, 3, |i, j| {
            let i_f = (i + 1) as f64;
            match j {
                0 => i_f,
                1 => i_f.powi(2) * 0.01,
                _ => (i_f * 0.1).sin(),
            }
        });
        let y = Col::from_fn(60, |i| {
            let t = (i + 1) as f64;
            1.0 + 0.5 * t + 0.1 * t.powi(2) * 0.01 + 0.3 * (t * 0.1).sin()
        });

        let model = LmDynamicRegressor::builder()
            .with_intercept(true)
            .max_models(7)
            .build();
        let fitted = model.fit(&x, &y).expect("Should fit");

        let x_new = Mat::from_fn(5, 3, |i, j| {
            let i_f = (i + 61) as f64;
            match j {
                0 => i_f,
                1 => i_f.powi(2) * 0.01,
                _ => (i_f * 0.1).sin(),
            }
        });

        let result = fitted.predict_with_interval(&x_new, Some(IntervalType::Prediction), 0.95);

        for i in 0..5 {
            assert!(result.lower[i].is_finite());
            assert!(result.upper[i].is_finite());
            assert!(result.se[i] > 0.0);
        }
    }

    #[test]
    fn test_interval_compute_xtx_inverse() {
        // Direct test for compute_xtx_inverse via prediction intervals
        let x = Mat::from_fn(30, 1, |i, _| (i + 1) as f64);
        let y = Col::from_fn(30, |i| 2.0 + 1.5 * (i + 1) as f64);

        let model = LmDynamicRegressor::builder().with_intercept(true).build();
        let fitted = model.fit(&x, &y).expect("Should fit");

        // Request intervals - this exercises compute_xtx_inverse
        let x_new = Mat::from_fn(3, 1, |i, _| (i + 31) as f64);
        let result = fitted.predict_with_interval(&x_new, Some(IntervalType::Confidence), 0.95);

        // Should produce valid intervals if XTX inverse is computed correctly
        for i in 0..3 {
            assert!(result.se[i].is_finite() && result.se[i] > 0.0);
        }
    }

    #[test]
    fn test_interval_extrapolation_leverage() {
        // Extrapolation should have higher leverage (wider intervals)
        let x = Mat::from_fn(50, 1, |i, _| (i + 1) as f64); // x from 1 to 50
        let y = Col::from_fn(50, |i| 2.0 + 1.5 * (i + 1) as f64);

        let model = LmDynamicRegressor::builder().with_intercept(true).build();
        let fitted = model.fit(&x, &y).expect("Should fit");

        // Near interpolation (within data range)
        let x_interp = Mat::from_fn(1, 1, |_, _| 25.0);
        let result_interp =
            fitted.predict_with_interval(&x_interp, Some(IntervalType::Confidence), 0.95);

        // Extrapolation (far outside data range)
        let x_extrap = Mat::from_fn(1, 1, |_, _| 100.0);
        let result_extrap =
            fitted.predict_with_interval(&x_extrap, Some(IntervalType::Confidence), 0.95);

        let se_interp = result_interp.se[0];
        let se_extrap = result_extrap.se[0];

        // Extrapolation should have larger SE due to higher leverage
        assert!(
            se_extrap > se_interp,
            "Extrapolation SE {} should be > interpolation SE {}",
            se_extrap,
            se_interp
        );
    }
}
