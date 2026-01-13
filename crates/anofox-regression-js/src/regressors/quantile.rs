//! Quantile Regression wrapper for WebAssembly.

use anofox_regression::solvers::{
    FittedQuantile as RustFittedQuantile, FittedRegressor,
    QuantileRegressor as RustQuantileRegressor, Regressor,
};
use faer::{Col, Mat};
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

/// Result from fitting a Quantile regression model.
#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct QuantileResult {
    pub coefficients: Vec<f64>,
    pub intercept: Option<f64>,
    pub tau: f64,
    pub n_observations: usize,
    pub residuals: Vec<f64>,
    pub fitted_values: Vec<f64>,
}

/// A fitted Quantile regression model.
#[wasm_bindgen]
pub struct FittedQuantile {
    fitted: RustFittedQuantile,
    n_features: usize,
    tau: f64,
}

#[wasm_bindgen]
impl FittedQuantile {
    /// Get the regression result as a JavaScript object.
    #[wasm_bindgen(js_name = getResult)]
    pub fn get_result(&self) -> Result<JsValue, JsError> {
        let result = self.fitted.result();

        let quantile_result = QuantileResult {
            coefficients: result.coefficients.iter().copied().collect(),
            intercept: result.intercept,
            tau: self.tau,
            n_observations: result.n_observations,
            residuals: result.residuals.iter().copied().collect(),
            fitted_values: result.fitted_values.iter().copied().collect(),
        };

        serde_wasm_bindgen::to_value(&quantile_result).map_err(|e| JsError::new(&e.to_string()))
    }

    /// Get the coefficients.
    #[wasm_bindgen(js_name = getCoefficients)]
    pub fn get_coefficients(&self) -> Vec<f64> {
        self.fitted.result().coefficients.iter().copied().collect()
    }

    /// Get the intercept.
    #[wasm_bindgen(js_name = getIntercept)]
    pub fn get_intercept(&self) -> Option<f64> {
        self.fitted.result().intercept
    }

    /// Get the quantile (tau).
    #[wasm_bindgen(js_name = getTau)]
    pub fn get_tau(&self) -> f64 {
        self.tau
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

/// Quantile Regression estimator.
///
/// Estimates conditional quantiles of the response variable.
/// Use tau=0.5 for median regression.
#[wasm_bindgen]
pub struct QuantileRegressor {
    tau: f64,
    with_intercept: bool,
    max_iterations: usize,
    tolerance: f64,
}

#[wasm_bindgen]
impl QuantileRegressor {
    /// Create a new Quantile regressor with default options (median regression).
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            tau: 0.5,
            with_intercept: true,
            max_iterations: 100,
            tolerance: 1e-6,
        }
    }

    /// Set the quantile to estimate (0 < tau < 1).
    /// tau=0.5 for median, tau=0.25 for first quartile, etc.
    #[wasm_bindgen(js_name = setTau)]
    pub fn set_tau(&mut self, tau: f64) {
        self.tau = tau;
    }

    /// Set whether to include an intercept term.
    #[wasm_bindgen(js_name = setWithIntercept)]
    pub fn set_with_intercept(&mut self, include: bool) {
        self.with_intercept = include;
    }

    /// Set the maximum iterations for IRLS algorithm.
    #[wasm_bindgen(js_name = setMaxIterations)]
    pub fn set_max_iterations(&mut self, max_iter: usize) {
        self.max_iterations = max_iter;
    }

    /// Set the convergence tolerance.
    #[wasm_bindgen(js_name = setTolerance)]
    pub fn set_tolerance(&mut self, tol: f64) {
        self.tolerance = tol;
    }

    /// Fit the Quantile model to the data.
    #[wasm_bindgen]
    pub fn fit(
        &self,
        x: &[f64],
        n_rows: usize,
        n_cols: usize,
        y: &[f64],
    ) -> Result<FittedQuantile, JsError> {
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

        if self.tau <= 0.0 || self.tau >= 1.0 {
            return Err(JsError::new("tau must be between 0 and 1 (exclusive)"));
        }

        let x_mat = Mat::from_fn(n_rows, n_cols, |i, j| x[i * n_cols + j]);
        let y_col = Col::from_fn(n_rows, |i| y[i]);

        let model = RustQuantileRegressor::builder()
            .tau(self.tau)
            .with_intercept(self.with_intercept)
            .max_iterations(self.max_iterations)
            .tolerance(self.tolerance)
            .build();

        let fitted = model
            .fit(&x_mat, &y_col)
            .map_err(|e| JsError::new(&format!("Regression failed: {}", e)))?;

        Ok(FittedQuantile {
            fitted,
            n_features: n_cols,
            tau: self.tau,
        })
    }
}

impl Default for QuantileRegressor {
    fn default() -> Self {
        Self::new()
    }
}
