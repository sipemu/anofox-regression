//! Tweedie GLM wrapper for WebAssembly.

use anofox_regression::solvers::{
    FittedRegressor, FittedTweedie as RustFittedTweedie, Regressor,
    TweedieRegressor as RustTweedieRegressor,
};
use faer::{Col, Mat};
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

/// Result from fitting a Tweedie regression model.
#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TweedieResult {
    pub coefficients: Vec<f64>,
    pub intercept: Option<f64>,
    pub deviance: f64,
    pub null_deviance: f64,
    pub var_power: f64,
    pub link_power: f64,
    pub dispersion: f64,
    pub iterations: usize,
    pub n_observations: usize,
    pub aic: f64,
    pub residuals: Vec<f64>,
    pub fitted_values: Vec<f64>,
    pub std_errors: Option<Vec<f64>>,
    pub p_values: Option<Vec<f64>>,
}

/// A fitted Tweedie regression model.
#[wasm_bindgen]
pub struct FittedTweedie {
    fitted: RustFittedTweedie,
    n_features: usize,
}

#[wasm_bindgen]
impl FittedTweedie {
    #[wasm_bindgen(js_name = getResult)]
    pub fn get_result(&self) -> Result<JsValue, JsError> {
        let result = self.fitted.result();
        let family = self.fitted.family();
        let tweedie_result = TweedieResult {
            coefficients: result.coefficients.iter().copied().collect(),
            intercept: result.intercept,
            deviance: self.fitted.deviance,
            null_deviance: self.fitted.null_deviance,
            var_power: family.var_power,
            link_power: family.link_power,
            dispersion: self.fitted.dispersion,
            iterations: self.fitted.iterations,
            n_observations: result.n_observations,
            aic: result.aic,
            residuals: result.residuals.iter().copied().collect(),
            fitted_values: result.fitted_values.iter().copied().collect(),
            std_errors: result.std_errors.as_ref().map(|v| v.iter().copied().collect()),
            p_values: result.p_values.as_ref().map(|v| v.iter().copied().collect()),
        };
        serde_wasm_bindgen::to_value(&tweedie_result).map_err(|e| JsError::new(&e.to_string()))
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

/// Tweedie GLM (flexible variance function).
///
/// Common var_power values:
/// - 0: Gaussian (normal)
/// - 1: Poisson
/// - 2: Gamma
/// - 1 < p < 2: Compound Poisson-Gamma (insurance claims)
/// - 3: Inverse Gaussian
#[wasm_bindgen]
pub struct TweedieRegressor {
    var_power: f64,
    link_power: f64,
    with_intercept: bool,
    compute_inference: bool,
    max_iterations: usize,
    tolerance: f64,
}

#[wasm_bindgen]
impl TweedieRegressor {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            var_power: 1.5,  // Compound Poisson-Gamma
            link_power: 0.0, // Log link
            with_intercept: true,
            compute_inference: true,
            max_iterations: 25,
            tolerance: 1e-8,
        }
    }

    /// Create a Gamma regressor (var_power=2, log link).
    #[wasm_bindgen]
    pub fn gamma() -> Self {
        Self {
            var_power: 2.0,
            link_power: 0.0,
            with_intercept: true,
            compute_inference: true,
            max_iterations: 25,
            tolerance: 1e-8,
        }
    }

    /// Set variance power (0=Gaussian, 1=Poisson, 2=Gamma, 3=InverseGaussian).
    #[wasm_bindgen(js_name = setVarPower)]
    pub fn set_var_power(&mut self, var_power: f64) {
        self.var_power = var_power;
    }

    /// Set link power (0=log, 1=identity).
    #[wasm_bindgen(js_name = setLinkPower)]
    pub fn set_link_power(&mut self, link_power: f64) {
        self.link_power = link_power;
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

    #[wasm_bindgen]
    pub fn fit(
        &self,
        x: &[f64],
        n_rows: usize,
        n_cols: usize,
        y: &[f64],
    ) -> Result<FittedTweedie, JsError> {
        if x.len() != n_rows * n_cols {
            return Err(JsError::new("Invalid x dimensions"));
        }
        if y.len() != n_rows {
            return Err(JsError::new("Invalid y length"));
        }

        let x_mat = Mat::from_fn(n_rows, n_cols, |i, j| x[i * n_cols + j]);
        let y_col = Col::from_fn(n_rows, |i| y[i]);

        let model = RustTweedieRegressor::builder()
            .var_power(self.var_power)
            .link_power(self.link_power)
            .with_intercept(self.with_intercept)
            .compute_inference(self.compute_inference)
            .max_iterations(self.max_iterations)
            .tolerance(self.tolerance)
            .build();

        let fitted = model
            .fit(&x_mat, &y_col)
            .map_err(|e| JsError::new(&format!("Regression failed: {}", e)))?;

        Ok(FittedTweedie {
            fitted,
            n_features: n_cols,
        })
    }
}

impl Default for TweedieRegressor {
    fn default() -> Self {
        Self::new()
    }
}
