//! Dynamic Linear Model (lmDynamic) wrapper for WebAssembly.

use anofox_regression::solvers::{
    FittedLmDynamic as RustFittedLmDynamic, FittedRegressor,
    InformationCriterion as RustInformationCriterion, LmDynamicRegressor as RustLmDynamicRegressor,
    Regressor,
};
use faer::{Col, Mat};
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

/// Result from fitting a Dynamic Linear Model.
#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LmDynamicResult {
    pub coefficients: Vec<f64>,
    pub intercept: Option<f64>,
    pub r_squared: f64,
    pub adj_r_squared: f64,
    pub mse: f64,
    pub rmse: f64,
    pub n_observations: usize,
    pub residuals: Vec<f64>,
    pub fitted_values: Vec<f64>,
}

/// A fitted Dynamic Linear Model.
#[wasm_bindgen]
pub struct FittedLmDynamic {
    fitted: RustFittedLmDynamic,
    n_features: usize,
}

#[wasm_bindgen]
impl FittedLmDynamic {
    #[wasm_bindgen(js_name = getResult)]
    pub fn get_result(&self) -> Result<JsValue, JsError> {
        let result = self.fitted.result();
        let lm_result = LmDynamicResult {
            coefficients: result.coefficients.iter().copied().collect(),
            intercept: result.intercept,
            r_squared: result.r_squared,
            adj_r_squared: result.adj_r_squared,
            mse: result.mse,
            rmse: result.rmse,
            n_observations: result.n_observations,
            residuals: result.residuals.iter().copied().collect(),
            fitted_values: result.fitted_values.iter().copied().collect(),
        };
        serde_wasm_bindgen::to_value(&lm_result).map_err(|e| JsError::new(&e.to_string()))
    }

    #[wasm_bindgen(js_name = getCoefficients)]
    pub fn get_coefficients(&self) -> Vec<f64> {
        self.fitted.result().coefficients.iter().copied().collect()
    }

    #[wasm_bindgen(js_name = getIntercept)]
    pub fn get_intercept(&self) -> Option<f64> {
        self.fitted.result().intercept
    }

    #[wasm_bindgen(js_name = getRSquared)]
    pub fn get_r_squared(&self) -> f64 {
        self.fitted.result().r_squared
    }

    /// Get the time-varying (dynamic) coefficients matrix.
    /// Returns a flattened array in row-major order (n_observations x n_coefficients).
    #[wasm_bindgen(js_name = getDynamicCoefficients)]
    pub fn get_dynamic_coefficients(&self) -> Vec<f64> {
        let dyn_coefs = self.fitted.dynamic_coefficients();
        let n_rows = dyn_coefs.nrows();
        let n_cols = dyn_coefs.ncols();
        let mut result = Vec::with_capacity(n_rows * n_cols);
        for i in 0..n_rows {
            for j in 0..n_cols {
                result.push(dyn_coefs[(i, j)]);
            }
        }
        result
    }

    /// Get the number of rows in the dynamic coefficients matrix.
    #[wasm_bindgen(js_name = getDynamicCoefficientsRows)]
    pub fn get_dynamic_coefficients_rows(&self) -> usize {
        self.fitted.dynamic_coefficients().nrows()
    }

    /// Get the number of columns in the dynamic coefficients matrix.
    #[wasm_bindgen(js_name = getDynamicCoefficientsCols)]
    pub fn get_dynamic_coefficients_cols(&self) -> usize {
        self.fitted.dynamic_coefficients().ncols()
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

/// Dynamic Linear Model with time-varying parameters.
///
/// Combines multiple candidate models using pointwise information criteria
/// weighting. Optionally smooths weights using LOWESS.
#[wasm_bindgen]
pub struct LmDynamicRegressor {
    ic_type: String,
    with_intercept: bool,
    lowess_span: Option<f64>,
    max_models: Option<usize>,
}

#[wasm_bindgen]
impl LmDynamicRegressor {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            ic_type: "aicc".to_string(),
            with_intercept: true,
            lowess_span: Some(0.3),
            max_models: Some(64),
        }
    }

    /// Set the information criterion for model weighting.
    /// Options: "aic", "aicc" (default), "bic"
    #[wasm_bindgen(js_name = setIc)]
    pub fn set_ic(&mut self, ic: &str) {
        self.ic_type = ic.to_string();
    }

    #[wasm_bindgen(js_name = setWithIntercept)]
    pub fn set_with_intercept(&mut self, include: bool) {
        self.with_intercept = include;
    }

    /// Set the LOWESS smoothing span (fraction of data, 0 to 1).
    /// Set to null/undefined to disable smoothing.
    #[wasm_bindgen(js_name = setLowessSpan)]
    pub fn set_lowess_span(&mut self, span: Option<f64>) {
        self.lowess_span = span.map(|s| s.clamp(0.05, 1.0));
    }

    /// Set maximum number of candidate models to consider.
    #[wasm_bindgen(js_name = setMaxModels)]
    pub fn set_max_models(&mut self, max: usize) {
        self.max_models = Some(max);
    }

    fn parse_ic(&self) -> RustInformationCriterion {
        match self.ic_type.to_lowercase().as_str() {
            "aic" => RustInformationCriterion::AIC,
            "bic" => RustInformationCriterion::BIC,
            _ => RustInformationCriterion::AICc,
        }
    }

    #[wasm_bindgen]
    pub fn fit(
        &self,
        x: &[f64],
        n_rows: usize,
        n_cols: usize,
        y: &[f64],
    ) -> Result<FittedLmDynamic, JsError> {
        if x.len() != n_rows * n_cols {
            return Err(JsError::new("Invalid x dimensions"));
        }
        if y.len() != n_rows {
            return Err(JsError::new("Invalid y length"));
        }

        let x_mat = Mat::from_fn(n_rows, n_cols, |i, j| x[i * n_cols + j]);
        let y_col = Col::from_fn(n_rows, |i| y[i]);

        let mut builder = RustLmDynamicRegressor::builder()
            .ic(self.parse_ic())
            .with_intercept(self.with_intercept);

        if let Some(span) = self.lowess_span {
            builder = builder.lowess_span(span);
        } else {
            builder = builder.no_smoothing();
        }

        if let Some(max) = self.max_models {
            builder = builder.max_models(max);
        }

        let model = builder.build();

        let fitted = model
            .fit(&x_mat, &y_col)
            .map_err(|e| JsError::new(&format!("Regression failed: {}", e)))?;

        Ok(FittedLmDynamic {
            fitted,
            n_features: n_cols,
        })
    }
}

impl Default for LmDynamicRegressor {
    fn default() -> Self {
        Self::new()
    }
}
