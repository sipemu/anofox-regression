//! Partial Least Squares (PLS) wrapper for WebAssembly.

use anofox_regression::solvers::{
    FittedPls as RustFittedPls, FittedRegressor, PlsRegressor as RustPlsRegressor, Regressor,
};
use faer::{Col, Mat};
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

/// Result from fitting a Partial Least Squares model.
#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PlsResult {
    pub coefficients: Vec<f64>,
    pub intercept: Option<f64>,
    pub r_squared: f64,
    pub adj_r_squared: f64,
    pub mse: f64,
    pub rmse: f64,
    pub n_observations: usize,
    pub n_components: usize,
    pub residuals: Vec<f64>,
    pub fitted_values: Vec<f64>,
}

/// A fitted Partial Least Squares model.
#[wasm_bindgen]
pub struct FittedPls {
    fitted: RustFittedPls,
    n_features: usize,
}

#[wasm_bindgen]
impl FittedPls {
    #[wasm_bindgen(js_name = getResult)]
    pub fn get_result(&self) -> Result<JsValue, JsError> {
        let result = self.fitted.result();
        let pls_result = PlsResult {
            coefficients: result.coefficients.iter().copied().collect(),
            intercept: result.intercept,
            r_squared: result.r_squared,
            adj_r_squared: result.adj_r_squared,
            mse: result.mse,
            rmse: result.rmse,
            n_observations: result.n_observations,
            n_components: self.fitted.n_components(),
            residuals: result.residuals.iter().copied().collect(),
            fitted_values: result.fitted_values.iter().copied().collect(),
        };
        serde_wasm_bindgen::to_value(&pls_result).map_err(|e| JsError::new(&e.to_string()))
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

    #[wasm_bindgen(js_name = getNComponents)]
    pub fn get_n_components(&self) -> usize {
        self.fitted.n_components()
    }

    /// Get the X means used for centering.
    #[wasm_bindgen(js_name = getXMeans)]
    pub fn get_x_means(&self) -> Vec<f64> {
        self.fitted.x_means().iter().copied().collect()
    }

    /// Get the Y mean used for centering.
    #[wasm_bindgen(js_name = getYMean)]
    pub fn get_y_mean(&self) -> f64 {
        self.fitted.y_mean()
    }

    /// Transform new X data into scores (latent variables).
    #[wasm_bindgen]
    pub fn transform(&self, x: &[f64], n_rows: usize) -> Result<Vec<f64>, JsError> {
        let n_cols = self.n_features;
        if x.len() != n_rows * n_cols {
            return Err(JsError::new("Invalid dimensions"));
        }
        let x_mat = Mat::from_fn(n_rows, n_cols, |i, j| x[i * n_cols + j]);
        let scores = self.fitted.transform(&x_mat);

        // Return flattened scores (row-major)
        let n_components = scores.ncols();
        let mut result = Vec::with_capacity(n_rows * n_components);
        for i in 0..n_rows {
            for j in 0..n_components {
                result.push(scores[(i, j)]);
            }
        }
        Ok(result)
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

/// Partial Least Squares regression.
///
/// Uses the SIMPLS algorithm for finding latent components that
/// maximize covariance between X scores and y. Useful for highly
/// collinear data or when there are more predictors than observations.
#[wasm_bindgen]
pub struct PlsRegressor {
    n_components: usize,
    with_intercept: bool,
    scale: bool,
}

#[wasm_bindgen]
impl PlsRegressor {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            n_components: 2,
            with_intercept: true,
            scale: false,
        }
    }

    /// Set the number of latent components to extract.
    #[wasm_bindgen(js_name = setNComponents)]
    pub fn set_n_components(&mut self, n: usize) {
        self.n_components = n.max(1);
    }

    #[wasm_bindgen(js_name = setWithIntercept)]
    pub fn set_with_intercept(&mut self, include: bool) {
        self.with_intercept = include;
    }

    /// Set whether to scale X to unit variance.
    #[wasm_bindgen(js_name = setScale)]
    pub fn set_scale(&mut self, scale: bool) {
        self.scale = scale;
    }

    #[wasm_bindgen]
    pub fn fit(
        &self,
        x: &[f64],
        n_rows: usize,
        n_cols: usize,
        y: &[f64],
    ) -> Result<FittedPls, JsError> {
        if x.len() != n_rows * n_cols {
            return Err(JsError::new("Invalid x dimensions"));
        }
        if y.len() != n_rows {
            return Err(JsError::new("Invalid y length"));
        }

        let x_mat = Mat::from_fn(n_rows, n_cols, |i, j| x[i * n_cols + j]);
        let y_col = Col::from_fn(n_rows, |i| y[i]);

        let model = RustPlsRegressor::builder()
            .n_components(self.n_components)
            .with_intercept(self.with_intercept)
            .scale(self.scale)
            .build();

        let fitted = model
            .fit(&x_mat, &y_col)
            .map_err(|e| JsError::new(&format!("Regression failed: {}", e)))?;

        Ok(FittedPls {
            fitted,
            n_features: n_cols,
        })
    }
}

impl Default for PlsRegressor {
    fn default() -> Self {
        Self::new()
    }
}
