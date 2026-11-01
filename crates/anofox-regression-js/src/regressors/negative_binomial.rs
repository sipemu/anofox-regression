//! Negative Binomial GLM wrapper for WebAssembly.

use anofox_regression::core::NaAction;
use anofox_regression::solvers::{
    FittedNegativeBinomial as RustFittedNegativeBinomial, FittedRegressor,
    NegativeBinomialRegressor as RustNegativeBinomialRegressor, Regressor,
};
use faer::{Col, Mat};
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

use super::{parse_na_action, row_column};

/// Result from fitting a Negative Binomial regression model.
#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NegativeBinomialResult {
    pub coefficients: Vec<f64>,
    pub intercept: Option<f64>,
    pub deviance: f64,
    pub null_deviance: f64,
    pub theta: f64,
    pub iterations: usize,
    pub n_observations: usize,
    pub aic: f64,
    pub residuals: Vec<f64>,
    pub fitted_values: Vec<f64>,
    pub std_errors: Option<Vec<f64>>,
    pub p_values: Option<Vec<f64>>,
}

/// A fitted Negative Binomial regression model.
#[wasm_bindgen]
pub struct FittedNegativeBinomial {
    fitted: RustFittedNegativeBinomial,
    n_features: usize,
}

#[wasm_bindgen]
impl FittedNegativeBinomial {
    #[wasm_bindgen(js_name = getResult)]
    pub fn get_result(&self) -> Result<JsValue, JsError> {
        let result = self.fitted.result();
        let nb_result = NegativeBinomialResult {
            coefficients: result.coefficients.iter().copied().collect(),
            intercept: result.intercept,
            deviance: self.fitted.deviance,
            null_deviance: self.fitted.null_deviance,
            theta: self.fitted.theta,
            iterations: self.fitted.iterations,
            n_observations: result.n_observations,
            aic: result.aic,
            residuals: result.residuals_expanded().iter().copied().collect(),
            fitted_values: result.fitted_expanded().iter().copied().collect(),
            std_errors: result
                .std_errors
                .as_ref()
                .map(|v| v.iter().copied().collect()),
            p_values: result
                .p_values
                .as_ref()
                .map(|v| v.iter().copied().collect()),
        };
        serde_wasm_bindgen::to_value(&nb_result).map_err(|e| JsError::new(&e.to_string()))
    }

    #[wasm_bindgen(js_name = getCoefficients)]
    pub fn get_coefficients(&self) -> Vec<f64> {
        self.fitted.result().coefficients.iter().copied().collect()
    }

    #[wasm_bindgen(js_name = getIntercept)]
    pub fn get_intercept(&self) -> Option<f64> {
        self.fitted.result().intercept
    }

    #[wasm_bindgen(js_name = getTheta)]
    pub fn get_theta(&self) -> f64 {
        self.fitted.theta
    }

    #[wasm_bindgen(js_name = getDeviance)]
    pub fn get_deviance(&self) -> f64 {
        self.fitted.deviance
    }

    /// Make predictions on the response scale with a new per-row offset on
    /// the linear predictor.
    #[wasm_bindgen(js_name = predictWithOffset)]
    pub fn predict_with_offset(
        &self,
        x: &[f64],
        n_rows: usize,
        offset: &[f64],
    ) -> Result<Vec<f64>, JsError> {
        let n_cols = self.n_features;
        if x.len() != n_rows * n_cols || offset.len() != n_rows {
            return Err(JsError::new("Invalid x or offset dimensions"));
        }
        let x_mat = Mat::from_fn(n_rows, n_cols, |i, j| x[i * n_cols + j]);
        let offset = Col::from_fn(n_rows, |i| offset[i]);
        Ok(self
            .fitted
            .predict_with_offset(&x_mat, &offset)
            .iter()
            .copied()
            .collect())
    }

    #[wasm_bindgen]
    pub fn predict(&self, x: &[f64], n_rows: usize) -> Result<Vec<f64>, JsError> {
        let n_cols = self.n_features;
        if x.len() != n_rows * n_cols {
            return Err(JsError::new("Invalid dimensions"));
        }
        let x_mat = Mat::from_fn(n_rows, n_cols, |i, j| x[i * n_cols + j]);
        Ok(self.fitted.predict(&x_mat).iter().copied().collect())
    }
}

/// Negative Binomial GLM for overdispersed count data.
#[wasm_bindgen]
pub struct NegativeBinomialRegressor {
    na_action: NaAction,
    offset: Option<Vec<f64>>,
    theta: Option<f64>,
    estimate_theta: bool,
    with_intercept: bool,
    compute_inference: bool,
    max_iterations: usize,
    tolerance: f64,
}

#[wasm_bindgen]
impl NegativeBinomialRegressor {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            na_action: NaAction::default(),
            offset: None,
            theta: None,
            estimate_theta: true,
            with_intercept: true,
            compute_inference: true,
            max_iterations: 25,
            tolerance: 1e-8,
        }
    }

    /// Set fixed theta (dispersion parameter).
    #[wasm_bindgen(js_name = setTheta)]
    pub fn set_theta(&mut self, theta: f64) {
        self.theta = Some(theta);
        self.estimate_theta = false;
    }

    /// Set whether to estimate theta from data.
    #[wasm_bindgen(js_name = setEstimateTheta)]
    pub fn set_estimate_theta(&mut self, estimate: bool) {
        self.estimate_theta = estimate;
        if estimate {
            self.theta = None;
        }
    }

    #[wasm_bindgen(js_name = setWithIntercept)]
    pub fn set_with_intercept(&mut self, include: bool) {
        self.with_intercept = include;
    }

    #[wasm_bindgen(js_name = setComputeInference)]
    pub fn set_compute_inference(&mut self, compute: bool) {
        self.compute_inference = compute;
    }

    #[wasm_bindgen(js_name = setMaxIterations)]
    pub fn set_max_iterations(&mut self, max_iter: usize) {
        self.max_iterations = max_iter;
    }

    /// Set the handling of rows with NaN in `x` or `y` (R's `na.action`):
    /// "omit" (default), "exclude" (residuals and fitted values padded with
    /// NaN at the dropped rows), "fail" or "pass".
    #[wasm_bindgen(js_name = setNaAction)]
    pub fn set_na_action(&mut self, action: &str) -> Result<(), JsError> {
        self.na_action = parse_na_action(action)?;
        Ok(())
    }

    /// Set an offset on the linear predictor, one value per row (e.g.
    /// `log(exposure)`), as `glm(offset = )`.
    #[wasm_bindgen(js_name = setOffset)]
    pub fn set_offset(&mut self, offset: Vec<f64>) {
        self.offset = Some(offset);
    }

    /// Set the convergence tolerance for IRLS.
    #[wasm_bindgen(js_name = setTolerance)]
    pub fn set_tolerance(&mut self, tol: f64) {
        self.tolerance = tol;
    }

    #[wasm_bindgen]
    pub fn fit(
        &self,
        x: &[f64],
        n_rows: usize,
        n_cols: usize,
        y: &[f64],
    ) -> Result<FittedNegativeBinomial, JsError> {
        if x.len() != n_rows * n_cols {
            return Err(JsError::new("Invalid x dimensions"));
        }
        if y.len() != n_rows {
            return Err(JsError::new("Invalid y length"));
        }

        // Validate non-negative counts
        if y.iter().any(|&yi| yi < 0.0) {
            return Err(JsError::new("Response must be non-negative counts"));
        }

        let x_mat = Mat::from_fn(n_rows, n_cols, |i, j| x[i * n_cols + j]);
        let y_col = Col::from_fn(n_rows, |i| y[i]);

        let mut builder = RustNegativeBinomialRegressor::builder()
            .na_action(self.na_action)
            .with_intercept(self.with_intercept)
            .compute_inference(self.compute_inference)
            .max_iterations(self.max_iterations)
            .tolerance(self.tolerance)
            .estimate_theta(self.estimate_theta);

        if let Some(theta) = self.theta {
            builder = builder.theta(theta);
        }

        if let Some(offset) = row_column(&self.offset, n_rows, "offset")? {
            builder = builder.offset(offset);
        }

        let model = builder.build();

        let fitted = model
            .fit(&x_mat, &y_col)
            .map_err(|e| JsError::new(&format!("Regression failed: {}", e)))?;

        Ok(FittedNegativeBinomial {
            fitted,
            n_features: n_cols,
        })
    }
}

impl Default for NegativeBinomialRegressor {
    fn default() -> Self {
        Self::new()
    }
}
