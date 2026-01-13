//! Elastic Net wrapper for WebAssembly.

use anofox_regression::solvers::{
    ElasticNetRegressor as RustElasticNetRegressor, FittedElasticNet as RustFittedElasticNet,
    FittedRegressor, Regressor,
};
use faer::{Col, Mat};
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

/// Result from fitting an Elastic Net model.
#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ElasticNetResult {
    pub coefficients: Vec<f64>,
    pub intercept: Option<f64>,
    pub r_squared: f64,
    pub lambda: f64,
    pub alpha: f64,
    pub n_observations: usize,
    pub residuals: Vec<f64>,
    pub fitted_values: Vec<f64>,
}

/// A fitted Elastic Net model.
#[wasm_bindgen]
pub struct FittedElasticNet {
    fitted: RustFittedElasticNet,
    n_features: usize,
    lambda: f64,
    alpha: f64,
}

#[wasm_bindgen]
impl FittedElasticNet {
    #[wasm_bindgen(js_name = getResult)]
    pub fn get_result(&self) -> Result<JsValue, JsError> {
        let result = self.fitted.result();
        let en_result = ElasticNetResult {
            coefficients: result.coefficients.iter().copied().collect(),
            intercept: result.intercept,
            r_squared: result.r_squared,
            lambda: self.lambda,
            alpha: self.alpha,
            n_observations: result.n_observations,
            residuals: result.residuals.iter().copied().collect(),
            fitted_values: result.fitted_values.iter().copied().collect(),
        };
        serde_wasm_bindgen::to_value(&en_result).map_err(|e| JsError::new(&e.to_string()))
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

/// Elastic Net regression (L1 + L2 regularization).
///
/// alpha=1.0 is Lasso, alpha=0.0 is Ridge
#[wasm_bindgen]
pub struct ElasticNetRegressor {
    with_intercept: bool,
    lambda: f64,
    alpha: f64,
    max_iterations: usize,
    tolerance: f64,
}

#[wasm_bindgen]
impl ElasticNetRegressor {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            with_intercept: true,
            lambda: 1.0,
            alpha: 0.5,
            max_iterations: 1000,
            tolerance: 1e-6,
        }
    }

    #[wasm_bindgen(js_name = setWithIntercept)]
    pub fn set_with_intercept(&mut self, include: bool) {
        self.with_intercept = include;
    }

    /// Set regularization strength.
    #[wasm_bindgen(js_name = setLambda)]
    pub fn set_lambda(&mut self, lambda: f64) {
        self.lambda = lambda;
    }

    /// Set L1/L2 mix ratio (0=Ridge, 1=Lasso).
    #[wasm_bindgen(js_name = setAlpha)]
    pub fn set_alpha(&mut self, alpha: f64) {
        self.alpha = alpha;
    }

    #[wasm_bindgen(js_name = setMaxIterations)]
    pub fn set_max_iterations(&mut self, max_iter: usize) {
        self.max_iterations = max_iter;
    }

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
    ) -> Result<FittedElasticNet, JsError> {
        if x.len() != n_rows * n_cols {
            return Err(JsError::new("Invalid x dimensions"));
        }
        if y.len() != n_rows {
            return Err(JsError::new("Invalid y length"));
        }

        let x_mat = Mat::from_fn(n_rows, n_cols, |i, j| x[i * n_cols + j]);
        let y_col = Col::from_fn(n_rows, |i| y[i]);

        let model = RustElasticNetRegressor::builder()
            .with_intercept(self.with_intercept)
            .lambda(self.lambda)
            .alpha(self.alpha)
            .max_iterations(self.max_iterations)
            .tolerance(self.tolerance)
            .build();

        let fitted = model
            .fit(&x_mat, &y_col)
            .map_err(|e| JsError::new(&format!("Regression failed: {}", e)))?;

        Ok(FittedElasticNet {
            fitted,
            n_features: n_cols,
            lambda: self.lambda,
            alpha: self.alpha,
        })
    }
}

impl Default for ElasticNetRegressor {
    fn default() -> Self {
        Self::new()
    }
}
