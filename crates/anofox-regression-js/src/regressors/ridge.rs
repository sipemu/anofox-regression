//! Ridge Regression wrapper for WebAssembly.

use anofox_regression::solvers::{
    FittedRegressor, Regressor, RidgeRegressor as RustRidgeRegressor,
};
use faer::{Col, Mat};
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

/// Result from fitting a Ridge regression model.
#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RidgeResult {
    pub coefficients: Vec<f64>,
    pub intercept: Option<f64>,
    pub r_squared: f64,
    pub adj_r_squared: f64,
    pub rmse: f64,
    pub mse: f64,
    pub lambda: f64,
    pub n_observations: usize,
    pub n_parameters: usize,
    pub residuals: Vec<f64>,
    pub fitted_values: Vec<f64>,
    pub std_errors: Option<Vec<f64>>,
    pub t_statistics: Option<Vec<f64>>,
    pub p_values: Option<Vec<f64>>,
    pub confidence_level: f64,
}

/// A fitted Ridge regression model.
#[wasm_bindgen]
pub struct FittedRidge {
    fitted: anofox_regression::solvers::FittedRidge,
    n_features: usize,
    lambda: f64,
}

#[wasm_bindgen]
impl FittedRidge {
    /// Get the regression result as a JavaScript object.
    #[wasm_bindgen(js_name = getResult)]
    pub fn get_result(&self) -> Result<JsValue, JsError> {
        let result = self.fitted.result();

        let ridge_result = RidgeResult {
            coefficients: result.coefficients.iter().copied().collect(),
            intercept: result.intercept,
            r_squared: result.r_squared,
            adj_r_squared: result.adj_r_squared,
            rmse: result.rmse,
            mse: result.mse,
            lambda: self.lambda,
            n_observations: result.n_observations,
            n_parameters: result.n_parameters,
            residuals: result.residuals.iter().copied().collect(),
            fitted_values: result.fitted_values.iter().copied().collect(),
            std_errors: result
                .std_errors
                .as_ref()
                .map(|v| v.iter().copied().collect()),
            t_statistics: result
                .t_statistics
                .as_ref()
                .map(|v| v.iter().copied().collect()),
            p_values: result
                .p_values
                .as_ref()
                .map(|v| v.iter().copied().collect()),
            confidence_level: result.confidence_level,
        };

        serde_wasm_bindgen::to_value(&ridge_result).map_err(|e| JsError::new(&e.to_string()))
    }

    /// Get the coefficients as a Float64Array.
    #[wasm_bindgen(js_name = getCoefficients)]
    pub fn get_coefficients(&self) -> Vec<f64> {
        self.fitted.result().coefficients.iter().copied().collect()
    }

    /// Get the intercept.
    #[wasm_bindgen(js_name = getIntercept)]
    pub fn get_intercept(&self) -> Option<f64> {
        self.fitted.result().intercept
    }

    /// Get R-squared.
    #[wasm_bindgen(js_name = getRSquared)]
    pub fn get_r_squared(&self) -> f64 {
        self.fitted.result().r_squared
    }

    /// Make predictions on new data.
    #[wasm_bindgen]
    pub fn predict(&self, x: &[f64], n_rows: usize) -> Result<Vec<f64>, JsError> {
        let n_cols = self.n_features;

        if x.len() != n_rows * n_cols {
            return Err(JsError::new(&format!(
                "Expected {} elements for {}x{} matrix, got {}",
                n_rows * n_cols,
                n_rows,
                n_cols,
                x.len()
            )));
        }

        let x_mat = Mat::from_fn(n_rows, n_cols, |i, j| x[i * n_cols + j]);
        let predictions = self.fitted.predict(&x_mat);

        Ok(predictions.iter().copied().collect())
    }
}

/// Ridge regression with L2 regularization.
///
/// Minimizes: ||y - Xb||^2 + lambda * ||b||^2
#[wasm_bindgen]
pub struct RidgeRegressor {
    with_intercept: bool,
    compute_inference: bool,
    confidence_level: f64,
    lambda: f64,
}

#[wasm_bindgen]
impl RidgeRegressor {
    /// Create a new Ridge regressor with default options.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            with_intercept: true,
            compute_inference: true,
            confidence_level: 0.95,
            lambda: 1.0,
        }
    }

    /// Set whether to include an intercept term.
    #[wasm_bindgen(js_name = setWithIntercept)]
    pub fn set_with_intercept(&mut self, include: bool) {
        self.with_intercept = include;
    }

    /// Set whether to compute inference statistics.
    #[wasm_bindgen(js_name = setComputeInference)]
    pub fn set_compute_inference(&mut self, compute: bool) {
        self.compute_inference = compute;
    }

    /// Set the confidence level for confidence intervals.
    #[wasm_bindgen(js_name = setConfidenceLevel)]
    pub fn set_confidence_level(&mut self, level: f64) {
        self.confidence_level = level;
    }

    /// Set the regularization parameter lambda.
    #[wasm_bindgen(js_name = setLambda)]
    pub fn set_lambda(&mut self, lambda: f64) {
        self.lambda = lambda;
    }

    /// Fit the Ridge model to the data.
    #[wasm_bindgen]
    pub fn fit(
        &self,
        x: &[f64],
        n_rows: usize,
        n_cols: usize,
        y: &[f64],
    ) -> Result<FittedRidge, JsError> {
        if x.len() != n_rows * n_cols {
            return Err(JsError::new(&format!(
                "Expected {} elements for {}x{} matrix, got {}",
                n_rows * n_cols,
                n_rows,
                n_cols,
                x.len()
            )));
        }

        if y.len() != n_rows {
            return Err(JsError::new(&format!(
                "Expected {} target values, got {}",
                n_rows,
                y.len()
            )));
        }

        let x_mat = Mat::from_fn(n_rows, n_cols, |i, j| x[i * n_cols + j]);
        let y_col = Col::from_fn(n_rows, |i| y[i]);

        let model = RustRidgeRegressor::builder()
            .with_intercept(self.with_intercept)
            .compute_inference(self.compute_inference)
            .confidence_level(self.confidence_level)
            .lambda(self.lambda)
            .build();

        let fitted = model
            .fit(&x_mat, &y_col)
            .map_err(|e| JsError::new(&format!("Regression failed: {}", e)))?;

        Ok(FittedRidge {
            fitted,
            n_features: n_cols,
            lambda: self.lambda,
        })
    }
}

impl Default for RidgeRegressor {
    fn default() -> Self {
        Self::new()
    }
}
