//! Isotonic Regression wrapper for WebAssembly.

use anofox_regression::solvers::{
    FittedIsotonic as RustFittedIsotonic, FittedRegressor,
    IsotonicRegressor as RustIsotonicRegressor, OutOfBounds,
};
use faer::Col;
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

/// Result from fitting an Isotonic regression model.
#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct IsotonicResult {
    pub x_thresholds: Vec<f64>,
    pub y_values: Vec<f64>,
    pub increasing: bool,
    pub r_squared: f64,
    pub n_observations: usize,
    pub residuals: Vec<f64>,
    pub fitted_values: Vec<f64>,
}

/// A fitted Isotonic regression model.
#[wasm_bindgen]
pub struct FittedIsotonic {
    fitted: RustFittedIsotonic,
}

#[wasm_bindgen]
impl FittedIsotonic {
    /// Get the regression result as a JavaScript object.
    #[wasm_bindgen(js_name = getResult)]
    pub fn get_result(&self) -> Result<JsValue, JsError> {
        let result = self.fitted.result();

        let isotonic_result = IsotonicResult {
            x_thresholds: self.fitted.x_thresholds().iter().copied().collect(),
            y_values: self.fitted.y_values().iter().copied().collect(),
            increasing: self.fitted.is_increasing(),
            r_squared: result.r_squared,
            n_observations: result.n_observations,
            residuals: result.residuals.iter().copied().collect(),
            fitted_values: result.fitted_values.iter().copied().collect(),
        };

        serde_wasm_bindgen::to_value(&isotonic_result).map_err(|e| JsError::new(&e.to_string()))
    }

    /// Get R-squared.
    #[wasm_bindgen(js_name = getRSquared)]
    pub fn get_r_squared(&self) -> f64 {
        self.fitted.result().r_squared
    }

    /// Get whether the fit is increasing or decreasing.
    #[wasm_bindgen(js_name = isIncreasing)]
    pub fn is_increasing(&self) -> bool {
        self.fitted.is_increasing()
    }

    /// Get the fitted values for the training data.
    #[wasm_bindgen(js_name = getFittedValues)]
    pub fn get_fitted_values(&self) -> Vec<f64> {
        self.fitted.fitted_values().iter().copied().collect()
    }

    /// Make predictions on new data.
    /// x should be a 1D array of feature values.
    #[wasm_bindgen]
    pub fn predict(&self, x: &[f64]) -> Vec<f64> {
        // Use predict_single for each x value
        x.iter().map(|&xi| self.fitted.predict_single(xi)).collect()
    }
}

/// Isotonic (Monotonic) Regression estimator.
///
/// Fits a monotonic (non-decreasing or non-increasing) step function to the data
/// using the Pool Adjacent Violators Algorithm (PAVA).
#[wasm_bindgen]
pub struct IsotonicRegressor {
    increasing: bool,
    out_of_bounds: String,
}

#[wasm_bindgen]
impl IsotonicRegressor {
    /// Create a new Isotonic regressor with default options.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            increasing: true,
            out_of_bounds: "clip".to_string(),
        }
    }

    /// Set whether to fit an increasing (true) or decreasing (false) function.
    #[wasm_bindgen(js_name = setIncreasing)]
    pub fn set_increasing(&mut self, increasing: bool) {
        self.increasing = increasing;
    }

    /// Set how to handle out-of-bounds predictions.
    /// Options: "clip" (default), "nan", "extrapolate"
    #[wasm_bindgen(js_name = setOutOfBounds)]
    pub fn set_out_of_bounds(&mut self, mode: &str) {
        self.out_of_bounds = mode.to_string();
    }

    /// Fit the Isotonic model to 1D data.
    ///
    /// # Arguments
    /// * `x` - Feature values (1D array)
    /// * `y` - Target values
    #[wasm_bindgen]
    pub fn fit(&self, x: &[f64], y: &[f64]) -> Result<FittedIsotonic, JsError> {
        if x.len() != y.len() {
            return Err(JsError::new(&format!(
                "x and y must have the same length: {} vs {}",
                x.len(),
                y.len()
            )));
        }

        if x.len() < 2 {
            return Err(JsError::new("Need at least 2 observations"));
        }

        let out_of_bounds = match self.out_of_bounds.as_str() {
            "nan" => OutOfBounds::Nan,
            "extrapolate" => OutOfBounds::Extrapolate,
            _ => OutOfBounds::Clip,
        };

        let x_col = Col::from_fn(x.len(), |i| x[i]);
        let y_col = Col::from_fn(y.len(), |i| y[i]);

        let model = RustIsotonicRegressor::builder()
            .increasing(self.increasing)
            .out_of_bounds(out_of_bounds)
            .build();

        let fitted = model
            .fit_1d(&x_col, &y_col)
            .map_err(|e| JsError::new(&format!("Regression failed: {}", e)))?;

        Ok(FittedIsotonic { fitted })
    }
}

impl Default for IsotonicRegressor {
    fn default() -> Self {
        Self::new()
    }
}
