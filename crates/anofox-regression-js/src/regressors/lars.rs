//! LARS / LassoLars wrapper for WebAssembly.

use anofox_regression::solvers::{
    FittedLars as RustFittedLars, FittedRegressor, LarsMethod, LarsRegressor as RustLarsRegressor,
    Regressor,
};
use faer::{Col, Mat};
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LarsResult {
    pub coefficients: Vec<f64>,
    pub intercept: Option<f64>,
    pub r_squared: f64,
    pub n_observations: usize,
    pub residuals: Vec<f64>,
    pub fitted_values: Vec<f64>,
    pub alphas: Vec<f64>,
    pub coefs_path: Vec<Vec<f64>>,
}

#[wasm_bindgen]
pub struct FittedLars {
    fitted: RustFittedLars,
    n_features: usize,
}

#[wasm_bindgen]
impl FittedLars {
    #[wasm_bindgen(js_name = getResult)]
    pub fn get_result(&self) -> Result<JsValue, JsError> {
        let r = self.fitted.result();
        let out = LarsResult {
            coefficients: r.coefficients.iter().copied().collect(),
            intercept: r.intercept,
            r_squared: r.r_squared,
            n_observations: r.n_observations,
            residuals: r.residuals.iter().copied().collect(),
            fitted_values: r.fitted_values.iter().copied().collect(),
            alphas: self.fitted.alphas().to_vec(),
            coefs_path: self.fitted.coefs_path(),
        };
        serde_wasm_bindgen::to_value(&out).map_err(|e| JsError::new(&e.to_string()))
    }

    #[wasm_bindgen(js_name = getCoefficients)]
    pub fn get_coefficients(&self) -> Vec<f64> {
        self.fitted.result().coefficients.iter().copied().collect()
    }

    #[wasm_bindgen(js_name = getIntercept)]
    pub fn get_intercept(&self) -> Option<f64> {
        self.fitted.result().intercept
    }

    #[wasm_bindgen(js_name = getAlphas)]
    pub fn get_alphas(&self) -> Vec<f64> {
        self.fitted.alphas().to_vec()
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

/// Least Angle Regression (LARS) / LassoLars solver.
#[wasm_bindgen]
pub struct LarsRegressor {
    method: String,
    fit_intercept: bool,
    n_nonzero_coefs: Option<usize>,
    alpha: f64,
    standardize: bool,
}

#[wasm_bindgen]
impl LarsRegressor {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            method: "lar".to_string(),
            fit_intercept: true,
            n_nonzero_coefs: None,
            alpha: 0.0,
            standardize: false,
        }
    }

    /// `"lar"` (default, plain LARS) or `"lasso"` (LassoLars variant).
    #[wasm_bindgen(js_name = setMethod)]
    pub fn set_method(&mut self, value: &str) {
        self.method = value.to_string();
    }

    #[wasm_bindgen(js_name = setFitIntercept)]
    pub fn set_fit_intercept(&mut self, include: bool) {
        self.fit_intercept = include;
    }

    #[wasm_bindgen(js_name = setNNonzeroCoefs)]
    pub fn set_n_nonzero_coefs(&mut self, value: usize) {
        self.n_nonzero_coefs = Some(value);
    }

    /// LassoLars-only: terminate the path once `α` reaches this value.
    /// Plain LARS ignores this setting.
    #[wasm_bindgen(js_name = setAlpha)]
    pub fn set_alpha(&mut self, value: f64) {
        self.alpha = value;
    }

    #[wasm_bindgen(js_name = setStandardize)]
    pub fn set_standardize(&mut self, value: bool) {
        self.standardize = value;
    }

    #[wasm_bindgen]
    pub fn fit(
        &self,
        x: &[f64],
        n_rows: usize,
        n_cols: usize,
        y: &[f64],
    ) -> Result<FittedLars, JsError> {
        if x.len() != n_rows * n_cols {
            return Err(JsError::new("Invalid x dimensions"));
        }
        if y.len() != n_rows {
            return Err(JsError::new("Invalid y length"));
        }
        let method = match self.method.as_str() {
            "lasso" => LarsMethod::Lasso,
            _ => LarsMethod::Lar,
        };
        let x_mat = Mat::from_fn(n_rows, n_cols, |i, j| x[i * n_cols + j]);
        let y_col = Col::from_fn(n_rows, |i| y[i]);

        let mut builder = RustLarsRegressor::builder()
            .method(method)
            .fit_intercept(self.fit_intercept)
            .alpha(self.alpha)
            .standardize(self.standardize);
        if let Some(k) = self.n_nonzero_coefs {
            builder = builder.n_nonzero_coefs(k);
        }
        let model = builder.build();
        let fitted = model
            .fit(&x_mat, &y_col)
            .map_err(|e| JsError::new(&format!("Regression failed: {}", e)))?;

        Ok(FittedLars {
            fitted,
            n_features: n_cols,
        })
    }
}

impl Default for LarsRegressor {
    fn default() -> Self {
        Self::new()
    }
}
