//! Bounded Least Squares (BLS) wrapper for WebAssembly.

use anofox_regression::solvers::{
    BlsRegressor as RustBlsRegressor, FittedBls as RustFittedBls, FittedRegressor, Regressor,
};
use faer::{Col, Mat};
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

/// Result from fitting a Bounded Least Squares model.
#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BlsResult {
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

/// A fitted Bounded Least Squares model.
#[wasm_bindgen]
pub struct FittedBls {
    fitted: RustFittedBls,
    n_features: usize,
}

#[wasm_bindgen]
impl FittedBls {
    #[wasm_bindgen(js_name = getResult)]
    pub fn get_result(&self) -> Result<JsValue, JsError> {
        let result = self.fitted.result();
        let bls_result = BlsResult {
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
        serde_wasm_bindgen::to_value(&bls_result).map_err(|e| JsError::new(&e.to_string()))
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

/// Bounded Least Squares regression (NNLS or box-constrained).
///
/// Uses the Lawson-Hanson active set algorithm for solving
/// least squares problems with coefficient bounds.
#[wasm_bindgen]
pub struct BlsRegressor {
    with_intercept: bool,
    lower_bound_all: Option<f64>,
    upper_bound_all: Option<f64>,
    lower_bounds: Option<Vec<f64>>,
    upper_bounds: Option<Vec<f64>>,
}

#[wasm_bindgen]
impl BlsRegressor {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            with_intercept: true,
            lower_bound_all: None,
            upper_bound_all: None,
            lower_bounds: None,
            upper_bounds: None,
        }
    }

    /// Create a Non-Negative Least Squares (NNLS) regressor.
    #[wasm_bindgen]
    pub fn nnls() -> Self {
        Self {
            with_intercept: true,
            lower_bound_all: Some(0.0),
            upper_bound_all: None,
            lower_bounds: None,
            upper_bounds: None,
        }
    }

    #[wasm_bindgen(js_name = setWithIntercept)]
    pub fn set_with_intercept(&mut self, include: bool) {
        self.with_intercept = include;
    }

    /// Set the same lower bound for all coefficients.
    #[wasm_bindgen(js_name = setLowerBoundAll)]
    pub fn set_lower_bound_all(&mut self, bound: f64) {
        self.lower_bound_all = Some(bound);
        self.lower_bounds = None;
    }

    /// Set the same upper bound for all coefficients.
    #[wasm_bindgen(js_name = setUpperBoundAll)]
    pub fn set_upper_bound_all(&mut self, bound: f64) {
        self.upper_bound_all = Some(bound);
        self.upper_bounds = None;
    }

    /// Set per-variable lower bounds.
    #[wasm_bindgen(js_name = setLowerBounds)]
    pub fn set_lower_bounds(&mut self, bounds: Vec<f64>) {
        self.lower_bounds = Some(bounds);
        self.lower_bound_all = None;
    }

    /// Set per-variable upper bounds.
    #[wasm_bindgen(js_name = setUpperBounds)]
    pub fn set_upper_bounds(&mut self, bounds: Vec<f64>) {
        self.upper_bounds = Some(bounds);
        self.upper_bound_all = None;
    }

    #[wasm_bindgen]
    pub fn fit(
        &self,
        x: &[f64],
        n_rows: usize,
        n_cols: usize,
        y: &[f64],
    ) -> Result<FittedBls, JsError> {
        if x.len() != n_rows * n_cols {
            return Err(JsError::new("Invalid x dimensions"));
        }
        if y.len() != n_rows {
            return Err(JsError::new("Invalid y length"));
        }

        let x_mat = Mat::from_fn(n_rows, n_cols, |i, j| x[i * n_cols + j]);
        let y_col = Col::from_fn(n_rows, |i| y[i]);

        let mut builder = RustBlsRegressor::builder().with_intercept(self.with_intercept);

        if let Some(bound) = self.lower_bound_all {
            builder = builder.lower_bound_all(bound);
        }
        if let Some(bound) = self.upper_bound_all {
            builder = builder.upper_bound_all(bound);
        }
        if let Some(ref bounds) = self.lower_bounds {
            builder = builder.lower_bounds(bounds.clone());
        }
        if let Some(ref bounds) = self.upper_bounds {
            builder = builder.upper_bounds(bounds.clone());
        }

        let model = builder.build();

        let fitted = model
            .fit(&x_mat, &y_col)
            .map_err(|e| JsError::new(&format!("Regression failed: {}", e)))?;

        Ok(FittedBls {
            fitted,
            n_features: n_cols,
        })
    }
}

impl Default for BlsRegressor {
    fn default() -> Self {
        Self::new()
    }
}
