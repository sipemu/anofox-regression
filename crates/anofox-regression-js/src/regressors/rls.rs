//! Recursive Least Squares (RLS) wrapper for WebAssembly.

use anofox_regression::solvers::{
    FittedRegressor, FittedRls as RustFittedRls, Regressor, RlsRegressor as RustRlsRegressor,
};
use faer::{Col, Mat};
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

/// Result from fitting a Recursive Least Squares model.
#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RlsResult {
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

/// A fitted Recursive Least Squares model.
#[wasm_bindgen]
pub struct FittedRls {
    fitted: RustFittedRls,
    n_features: usize,
}

#[wasm_bindgen]
impl FittedRls {
    #[wasm_bindgen(js_name = getResult)]
    pub fn get_result(&self) -> Result<JsValue, JsError> {
        let result = self.fitted.result();
        let rls_result = RlsResult {
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
        serde_wasm_bindgen::to_value(&rls_result).map_err(|e| JsError::new(&e.to_string()))
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

    #[wasm_bindgen(js_name = getForgettingFactor)]
    pub fn get_forgetting_factor(&self) -> f64 {
        self.fitted.forgetting_factor()
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

/// Recursive Least Squares regression for online learning.
///
/// RLS updates coefficients incrementally as new observations arrive.
/// Uses a forgetting factor to weight recent observations more heavily.
#[wasm_bindgen]
pub struct RlsRegressor {
    with_intercept: bool,
    forgetting_factor: f64,
}

#[wasm_bindgen]
impl RlsRegressor {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            with_intercept: true,
            forgetting_factor: 1.0,
        }
    }

    #[wasm_bindgen(js_name = setWithIntercept)]
    pub fn set_with_intercept(&mut self, include: bool) {
        self.with_intercept = include;
    }

    /// Set the forgetting factor (0 < λ ≤ 1).
    ///
    /// - λ = 1: All observations weighted equally (standard RLS)
    /// - λ < 1: Recent observations weighted more heavily
    #[wasm_bindgen(js_name = setForgettingFactor)]
    pub fn set_forgetting_factor(&mut self, factor: f64) {
        self.forgetting_factor = factor.clamp(0.01, 1.0);
    }

    #[wasm_bindgen]
    pub fn fit(
        &self,
        x: &[f64],
        n_rows: usize,
        n_cols: usize,
        y: &[f64],
    ) -> Result<FittedRls, JsError> {
        if x.len() != n_rows * n_cols {
            return Err(JsError::new("Invalid x dimensions"));
        }
        if y.len() != n_rows {
            return Err(JsError::new("Invalid y length"));
        }

        let x_mat = Mat::from_fn(n_rows, n_cols, |i, j| x[i * n_cols + j]);
        let y_col = Col::from_fn(n_rows, |i| y[i]);

        let model = RustRlsRegressor::builder()
            .with_intercept(self.with_intercept)
            .forgetting_factor(self.forgetting_factor)
            .build();

        let fitted = model
            .fit(&x_mat, &y_col)
            .map_err(|e| JsError::new(&format!("Regression failed: {}", e)))?;

        Ok(FittedRls {
            fitted,
            n_features: n_cols,
        })
    }
}

impl Default for RlsRegressor {
    fn default() -> Self {
        Self::new()
    }
}
