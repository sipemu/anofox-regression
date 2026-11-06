//! Accelerated failure time (AFT) regression wrapper for WebAssembly.

use anofox_regression::solvers::aft::{
    AftDistribution, AftRegressor as RustAftRegressor, FittedAft as RustFittedAft,
};
use anofox_regression::solvers::penalized_glm::{PriorSpec, VcovType};
use faer::Mat;
use serde::Serialize;
use wasm_bindgen::prelude::*;

use super::{
    col_vec, fit_error, mat_flat, mat_rows, parse_priors, parse_vcov_type, to_col, to_js, to_mat,
};

/// Wald inference of an AFT fit.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AftInferenceResult {
    pub std_errors: Vec<f64>,
    pub z_values: Vec<f64>,
    pub p_values: Vec<f64>,
    pub ci_lower: Vec<f64>,
    pub ci_upper: Vec<f64>,
    pub confidence_level: f64,
    pub intercept_std_error: Option<f64>,
    pub intercept_z_value: Option<f64>,
    pub intercept_p_value: Option<f64>,
    pub intercept_ci_lower: Option<f64>,
    pub intercept_ci_upper: Option<f64>,
    /// Standard error of `log(scale)`; absent when the scale is fixed.
    pub log_scale_std_error: Option<f64>,
    /// Covariance of (intercept, coefficients, log scale), as nested rows.
    pub vcov: Option<Vec<Vec<f64>>>,
}

/// Result from fitting an AFT model.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AftResult {
    pub coefficients: Vec<f64>,
    pub intercept: Option<f64>,
    pub scale: f64,
    pub distribution: String,
    pub log_likelihood: f64,
    pub null_log_likelihood: f64,
    pub aic: f64,
    pub bic: f64,
    pub n_observations: usize,
    pub n_events: usize,
    pub n_censored: usize,
    pub iterations: usize,
    pub converged: bool,
    pub inference: Option<AftInferenceResult>,
}

/// A fitted AFT model.
#[wasm_bindgen]
pub struct FittedAft {
    fitted: RustFittedAft,
    n_features: usize,
}

impl FittedAft {
    fn mat(&self, x: &[f64], n_rows: usize) -> Result<Mat<f64>, JsError> {
        to_mat(x, n_rows, self.n_features)
    }
}

#[wasm_bindgen]
impl FittedAft {
    /// Rebuild a fitted model from stored parts: the distribution name, the
    /// intercept (undefined without one), the coefficients, the scale and
    /// optionally the row-major covariance (intercept, coefficients, then
    /// `log σ` when the scale is estimated). It predicts as the fit.
    #[wasm_bindgen(js_name = fromParts)]
    pub fn from_parts(
        distribution: &str,
        intercept: Option<f64>,
        coefficients: Vec<f64>,
        scale: f64,
        vcov: Option<Vec<f64>>,
    ) -> Result<FittedAft, JsError> {
        let dist = AftDistribution::from_name(distribution)
            .ok_or_else(|| JsError::new(&format!("Unknown distribution '{distribution}'")))?;
        let n_features = coefficients.len();
        let k = n_features + usize::from(intercept.is_some()) + usize::from(!dist.scale_is_fixed());
        let v = vcov.map(|v| to_mat(&v, k, k)).transpose()?;
        let fitted = RustFittedAft::from_parts(dist, intercept, coefficients, scale, v.as_ref())
            .map_err(fit_error)?;
        Ok(Self { fitted, n_features })
    }

    /// Get the full result as a JavaScript object.
    #[wasm_bindgen(js_name = getResult)]
    pub fn get_result(&self) -> Result<JsValue, JsError> {
        let core = self.fitted.fit_result();
        let inference = self.fitted.inference().map(|inf| AftInferenceResult {
            std_errors: inf.std_errors.clone(),
            z_values: inf.z_values.clone(),
            p_values: inf.p_values.clone(),
            ci_lower: inf.ci_lower.clone(),
            ci_upper: inf.ci_upper.clone(),
            confidence_level: inf.confidence_level,
            intercept_std_error: inf.intercept_std_error,
            intercept_z_value: inf.intercept_z_value,
            intercept_p_value: inf.intercept_p_value,
            intercept_ci_lower: inf.intercept_ci_lower,
            intercept_ci_upper: inf.intercept_ci_upper,
            log_scale_std_error: inf.log_scale_std_error,
            vcov: inf.vcov.as_ref().map(mat_rows),
        });
        to_js(&AftResult {
            coefficients: core.coefficients.clone(),
            intercept: core.intercept,
            scale: core.scale,
            distribution: self.fitted.distribution().as_str().to_string(),
            log_likelihood: core.log_likelihood,
            null_log_likelihood: core.null_log_likelihood,
            aic: core.aic,
            bic: core.bic,
            n_observations: core.n_observations,
            n_events: core.n_events,
            n_censored: core.n_censored,
            iterations: core.iterations as usize,
            converged: core.converged,
            inference,
        })
    }

    /// Coefficients on the log-time scale (excluding the intercept).
    #[wasm_bindgen(js_name = getCoefficients)]
    pub fn get_coefficients(&self) -> Vec<f64> {
        col_vec(self.fitted.coefficients().as_ref())
    }

    /// The intercept on the log-time scale.
    #[wasm_bindgen(js_name = getIntercept)]
    pub fn get_intercept(&self) -> Option<f64> {
        self.fitted.intercept()
    }

    /// The scale σ (1 for the exponential distribution).
    #[wasm_bindgen(js_name = getScale)]
    pub fn get_scale(&self) -> f64 {
        self.fitted.scale()
    }

    /// The log-likelihood.
    #[wasm_bindgen(js_name = getLogLikelihood)]
    pub fn get_log_likelihood(&self) -> f64 {
        self.fitted.log_likelihood()
    }

    /// Linear predictor (location of log T).
    #[wasm_bindgen(js_name = predictLinear)]
    pub fn predict_linear(&self, x: &[f64], n_rows: usize) -> Result<Vec<f64>, JsError> {
        Ok(col_vec(
            self.fitted.predict_linear(&self.mat(x, n_rows)?).as_ref(),
        ))
    }

    /// Predicted median survival time.
    #[wasm_bindgen(js_name = predictMedian)]
    pub fn predict_median(&self, x: &[f64], n_rows: usize) -> Result<Vec<f64>, JsError> {
        Ok(col_vec(
            self.fitted.predict_median(&self.mat(x, n_rows)?).as_ref(),
        ))
    }

    /// Predicted `p`-quantile of the survival time.
    #[wasm_bindgen(js_name = predictQuantile)]
    pub fn predict_quantile(&self, x: &[f64], n_rows: usize, p: f64) -> Result<Vec<f64>, JsError> {
        Ok(col_vec(
            self.fitted
                .predict_quantile(&self.mat(x, n_rows)?, p)
                .as_ref(),
        ))
    }

    /// Predicted survival probability `S(t | x)`.
    #[wasm_bindgen(js_name = predictSurvival)]
    pub fn predict_survival(&self, x: &[f64], n_rows: usize, t: f64) -> Result<Vec<f64>, JsError> {
        Ok(col_vec(
            self.fitted
                .predict_survival(&self.mat(x, n_rows)?, t)
                .as_ref(),
        ))
    }

    /// Predicted failure probability `F(t | x)`.
    #[wasm_bindgen(js_name = predictCdf)]
    pub fn predict_cdf(&self, x: &[f64], n_rows: usize, t: f64) -> Result<Vec<f64>, JsError> {
        Ok(col_vec(
            self.fitted.predict_cdf(&self.mat(x, n_rows)?, t).as_ref(),
        ))
    }

    /// Standardized residuals `(log t − η) / σ`.
    #[wasm_bindgen(js_name = standardizedResiduals)]
    pub fn standardized_residuals(
        &self,
        x: &[f64],
        n_rows: usize,
        time: &[f64],
    ) -> Result<Vec<f64>, JsError> {
        let x_mat = self.mat(x, n_rows)?;
        let time = to_col(time, n_rows, "time")?;
        Ok(col_vec(
            self.fitted.standardized_residuals(&x_mat, &time).as_ref(),
        ))
    }

    /// Cox–Snell residuals `−log S(t | x)`.
    #[wasm_bindgen(js_name = coxSnellResiduals)]
    pub fn cox_snell_residuals(
        &self,
        x: &[f64],
        n_rows: usize,
        time: &[f64],
    ) -> Result<Vec<f64>, JsError> {
        let x_mat = self.mat(x, n_rows)?;
        let time = to_col(time, n_rows, "time")?;
        Ok(col_vec(
            self.fitted.cox_snell_residuals(&x_mat, &time).as_ref(),
        ))
    }

    /// Cluster-robust covariance of (intercept, coefficients, log scale)
    /// (`survreg(..., cluster = id)` with the "robcov" preset). Needs a fit
    /// with inference.
    #[wasm_bindgen(js_name = clusterVcov)]
    pub fn cluster_vcov(
        &self,
        x: &[f64],
        n_rows: usize,
        time: &[f64],
        event: &[u8],
        clusters: JsValue,
        options: JsValue,
    ) -> Result<JsValue, JsError> {
        let x_mat = self.mat(x, n_rows)?;
        let time = to_col(time, n_rows, "time")?;
        let event: Vec<bool> = event.iter().map(|&e| e != 0).collect();
        let c = crate::inference::parse_clusters(clusters, n_rows)?;
        let v = self
            .fitted
            .cluster_vcov(
                &x_mat,
                &time,
                &event,
                &crate::inference::cluster_refs(&c),
                &crate::inference::parse_cluster_options(options)?,
            )
            .map_err(fit_error)?;
        Ok(crate::convert::ToJs::to_js(&v))
    }

    /// Per-observation score contributions, row-major `n_rows × k` with `k`
    /// = intercept + coefficients + log scale (when estimated).
    #[wasm_bindgen(js_name = scoreContributions)]
    pub fn score_contributions(
        &self,
        x: &[f64],
        n_rows: usize,
        time: &[f64],
        event: &[u8],
    ) -> Result<Vec<f64>, JsError> {
        let x_mat = self.mat(x, n_rows)?;
        let time = to_col(time, n_rows, "time")?;
        let event: Vec<bool> = event.iter().map(|&e| e != 0).collect();
        let s = self
            .fitted
            .score_contributions(&x_mat, &time, &event)
            .map_err(fit_error)?;
        Ok(mat_flat(&s))
    }
}

/// Accelerated failure time regression with right censoring
/// (`survival::survreg`).
#[wasm_bindgen]
pub struct AftRegressor {
    distribution: AftDistribution,
    with_intercept: bool,
    max_iterations: u32,
    tolerance: f64,
    compute_inference: bool,
    confidence_level: f64,
    priors: Vec<PriorSpec>,
    vcov: VcovType,
}

#[wasm_bindgen]
impl AftRegressor {
    /// Create an AFT regressor (Weibull, intercept, no inference).
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            distribution: AftDistribution::Weibull,
            with_intercept: true,
            max_iterations: 100,
            tolerance: 1e-9,
            compute_inference: false,
            confidence_level: 0.95,
            priors: Vec::new(),
            vcov: VcovType::default(),
        }
    }

    /// Set the survival-time distribution: "weibull" (default),
    /// "lognormal", "loglogistic" or "exponential".
    #[wasm_bindgen(js_name = setDistribution)]
    pub fn set_distribution(&mut self, name: &str) -> Result<(), JsError> {
        self.distribution = AftDistribution::from_name(name).ok_or_else(|| {
            JsError::new(&format!(
                "Unknown distribution '{}': expected weibull, lognormal, loglogistic or exponential",
                name
            ))
        })?;
        Ok(())
    }

    /// Set whether to include an intercept term.
    #[wasm_bindgen(js_name = setWithIntercept)]
    pub fn set_with_intercept(&mut self, include: bool) {
        self.with_intercept = include;
    }

    /// Set the maximum Newton iterations (default 100).
    #[wasm_bindgen(js_name = setMaxIterations)]
    pub fn set_max_iterations(&mut self, max_iter: u32) {
        self.max_iterations = max_iter;
    }

    /// Set the convergence tolerance on the log-likelihood (default 1e-9).
    #[wasm_bindgen(js_name = setTolerance)]
    pub fn set_tolerance(&mut self, tol: f64) {
        self.tolerance = tol;
    }

    /// Set whether to compute standard errors, tests and intervals.
    #[wasm_bindgen(js_name = setComputeInference)]
    pub fn set_compute_inference(&mut self, compute: bool) {
        self.compute_inference = compute;
    }

    /// Set the confidence level (default 0.95).
    #[wasm_bindgen(js_name = setConfidenceLevel)]
    pub fn set_confidence_level(&mut self, level: f64) {
        self.confidence_level = level;
    }

    /// Set per-coefficient priors, an array of
    /// `{ kind: "flat" | "normal" | "laplace", loc, scale }` in feature
    /// order, optionally with a leading entry for the intercept.
    #[wasm_bindgen(js_name = setPriors)]
    pub fn set_priors(&mut self, priors: JsValue) -> Result<(), JsError> {
        self.priors = parse_priors(priors)?;
        Ok(())
    }

    /// Set the covariance at the mode: "laplace" (default), "sandwich" or
    /// "naive".
    #[wasm_bindgen(js_name = setVcovType)]
    pub fn set_vcov_type(&mut self, vcov: &str) -> Result<(), JsError> {
        self.vcov = parse_vcov_type(vcov)?;
        Ok(())
    }

    /// Fit to survival times `time` (> 0) with `event[i]` = 1 for an
    /// observed event and 0 for a right-censored time.
    #[wasm_bindgen]
    pub fn fit(
        &self,
        x: &[f64],
        n_rows: usize,
        n_cols: usize,
        time: &[f64],
        event: &[u8],
    ) -> Result<FittedAft, JsError> {
        let x_mat = to_mat(x, n_rows, n_cols)?;
        let time = to_col(time, n_rows, "time")?;
        if event.len() != n_rows {
            return Err(JsError::new(&format!(
                "Expected {} event values, got {}",
                n_rows,
                event.len()
            )));
        }
        let event: Vec<bool> = event.iter().map(|&e| e != 0).collect();
        let fitted = RustAftRegressor::builder()
            .distribution(self.distribution)
            .with_intercept(self.with_intercept)
            .max_iterations(self.max_iterations)
            .tolerance(self.tolerance)
            .compute_inference(self.compute_inference)
            .confidence_level(self.confidence_level)
            .priors(self.priors.clone())
            .vcov_type(self.vcov)
            .build()
            .fit(&x_mat, &time, &event)
            .map_err(fit_error)?;
        Ok(FittedAft {
            fitted,
            n_features: n_cols,
        })
    }
}

impl Default for AftRegressor {
    fn default() -> Self {
        Self::new()
    }
}

iterative_fit!(FittedAft);
likelihood_methods!(FittedAft);

#[wasm_bindgen]
impl FittedAft {
    /// Survival-time quantiles at probabilities `probs` for one covariate
    /// row.
    #[wasm_bindgen]
    pub fn quantiles(&self, x_row: &[f64], probs: &[f64]) -> Result<Vec<f64>, JsError> {
        self.check_row(x_row)?;
        Ok(self.fitted.quantiles(x_row, probs))
    }

    /// Survival probabilities at `times` for one covariate row.
    #[wasm_bindgen(js_name = survivalCurve)]
    pub fn survival_curve(&self, x_row: &[f64], times: &[f64]) -> Result<Vec<f64>, JsError> {
        self.check_row(x_row)?;
        Ok(self.fitted.survival_curve(x_row, times))
    }
}

impl FittedAft {
    fn check_row(&self, x_row: &[f64]) -> Result<(), JsError> {
        if x_row.len() != self.n_features {
            return Err(JsError::new(&format!(
                "Expected {} values, got {}",
                self.n_features,
                x_row.len()
            )));
        }
        Ok(())
    }
}
