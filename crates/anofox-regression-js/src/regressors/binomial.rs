//! Binomial (Logistic) Regression wrapper for WebAssembly.

use anofox_regression::core::BinomialLink;
use anofox_regression::solvers::{
    BinomialRegressor as RustBinomialRegressor, FittedBinomial as RustFittedBinomial,
    FittedRegressor, Regressor,
};
use faer::{Col, Mat};
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

/// Result from fitting a Binomial regression model.
#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BinomialResult {
    pub coefficients: Vec<f64>,
    pub intercept: Option<f64>,
    pub deviance: f64,
    pub null_deviance: f64,
    pub iterations: usize,
    pub n_observations: usize,
    pub aic: f64,
    pub bic: f64,
    pub residuals: Vec<f64>,
    pub fitted_values: Vec<f64>,
    pub std_errors: Option<Vec<f64>>,
    pub p_values: Option<Vec<f64>>,
}

/// A fitted Binomial regression model (logistic/probit).
#[wasm_bindgen]
pub struct FittedBinomial {
    fitted: RustFittedBinomial,
    n_features: usize,
}

#[wasm_bindgen]
impl FittedBinomial {
    #[wasm_bindgen(js_name = getResult)]
    pub fn get_result(&self) -> Result<JsValue, JsError> {
        let result = self.fitted.result();
        let bin_result = BinomialResult {
            coefficients: result.coefficients.iter().copied().collect(),
            intercept: result.intercept,
            deviance: self.fitted.deviance,
            null_deviance: self.fitted.null_deviance,
            iterations: self.fitted.iterations,
            n_observations: result.n_observations,
            aic: result.aic,
            bic: result.bic,
            residuals: result.residuals.iter().copied().collect(),
            fitted_values: result.fitted_values.iter().copied().collect(),
            std_errors: result
                .std_errors
                .as_ref()
                .map(|v| v.iter().copied().collect()),
            p_values: result
                .p_values
                .as_ref()
                .map(|v| v.iter().copied().collect()),
        };
        serde_wasm_bindgen::to_value(&bin_result).map_err(|e| JsError::new(&e.to_string()))
    }

    #[wasm_bindgen(js_name = getCoefficients)]
    pub fn get_coefficients(&self) -> Vec<f64> {
        self.fitted.result().coefficients.iter().copied().collect()
    }

    #[wasm_bindgen(js_name = getIntercept)]
    pub fn get_intercept(&self) -> Option<f64> {
        self.fitted.result().intercept
    }

    #[wasm_bindgen(js_name = getDeviance)]
    pub fn get_deviance(&self) -> f64 {
        self.fitted.deviance
    }

    /// Predict probabilities.
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

/// Binomial GLM (logistic/probit regression).
#[wasm_bindgen]
pub struct BinomialRegressor {
    link: String,
    with_intercept: bool,
    compute_inference: bool,
    max_iterations: usize,
    tolerance: f64,
}

#[wasm_bindgen]
impl BinomialRegressor {
    /// Create a new Binomial regressor (default: logit link).
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            link: "logit".to_string(),
            with_intercept: true,
            compute_inference: true,
            max_iterations: 25,
            tolerance: 1e-8,
        }
    }

    /// Set link function: "logit" (default), "probit", or "cloglog".
    #[wasm_bindgen(js_name = setLink)]
    pub fn set_link(&mut self, link: &str) {
        self.link = link.to_string();
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

    /// Fit logistic/probit regression to binary data.
    #[wasm_bindgen]
    pub fn fit(
        &self,
        x: &[f64],
        n_rows: usize,
        n_cols: usize,
        y: &[f64],
    ) -> Result<FittedBinomial, JsError> {
        if x.len() != n_rows * n_cols {
            return Err(JsError::new("Invalid x dimensions"));
        }
        if y.len() != n_rows {
            return Err(JsError::new("Invalid y length"));
        }

        // Validate binary response
        if y.iter().any(|&yi| !(0.0..=1.0).contains(&yi)) {
            return Err(JsError::new("Response must be between 0 and 1"));
        }

        let x_mat = Mat::from_fn(n_rows, n_cols, |i, j| x[i * n_cols + j]);
        let y_col = Col::from_fn(n_rows, |i| y[i]);

        let link = match self.link.as_str() {
            "probit" => BinomialLink::Probit,
            "cloglog" => BinomialLink::Cloglog,
            _ => BinomialLink::Logit,
        };

        let model = RustBinomialRegressor::builder()
            .link(link)
            .with_intercept(self.with_intercept)
            .compute_inference(self.compute_inference)
            .max_iterations(self.max_iterations)
            .tolerance(self.tolerance)
            .build();

        let fitted = model
            .fit(&x_mat, &y_col)
            .map_err(|e| JsError::new(&format!("Regression failed: {}", e)))?;

        Ok(FittedBinomial {
            fitted,
            n_features: n_cols,
        })
    }
}

impl Default for BinomialRegressor {
    fn default() -> Self {
        Self::new()
    }
}
