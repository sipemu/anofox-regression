//! Gamma GLM Regression wrapper for WebAssembly.

use anofox_regression::solvers::{
    FittedGamma as RustFittedGamma, FittedRegressor, GammaRegressor as RustGammaRegressor,
    Regressor,
};
use faer::{Col, Mat};
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GammaResult {
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

/// A fitted Gamma GLM regression model.
#[wasm_bindgen]
pub struct FittedGamma {
    fitted: RustFittedGamma,
    n_features: usize,
}

#[wasm_bindgen]
impl FittedGamma {
    #[wasm_bindgen(js_name = getResult)]
    pub fn get_result(&self) -> Result<JsValue, JsError> {
        let inner = self.fitted.inner();
        let result = self.fitted.result();
        let r = GammaResult {
            coefficients: result.coefficients.iter().copied().collect(),
            intercept: result.intercept,
            deviance: inner.deviance,
            null_deviance: inner.null_deviance,
            iterations: inner.iterations,
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
        serde_wasm_bindgen::to_value(&r).map_err(|e| JsError::new(&e.to_string()))
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
        self.fitted.inner().deviance
    }

    /// Predict mean values on the response scale.
    #[wasm_bindgen]
    pub fn predict(&self, x: &[f64], n_rows: usize) -> Result<Vec<f64>, JsError> {
        let x_mat = self.build_matrix(x, n_rows)?;
        Ok(self.fitted.predict(&x_mat).iter().copied().collect())
    }

    /// Predict on the linear-predictor (link) scale: `η = Xβ + intercept`.
    #[wasm_bindgen(js_name = predictEta)]
    pub fn predict_eta(&self, x: &[f64], n_rows: usize) -> Result<Vec<f64>, JsError> {
        let x_mat = self.build_matrix(x, n_rows)?;
        Ok(self.fitted.predict_eta(&x_mat).iter().copied().collect())
    }

    fn build_matrix(&self, x: &[f64], n_rows: usize) -> Result<Mat<f64>, JsError> {
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
        Ok(Mat::from_fn(n_rows, n_cols, |i, j| x[i * n_cols + j]))
    }
}

/// Gamma GLM regression with log link.
///
/// Convenience wrapper around `TweedieRegressor` with `var_power = 2`.
/// Use this for positive continuous responses where the variance scales as
/// the square of the mean.
///
/// # Example (JavaScript)
/// ```javascript
/// const regressor = new GammaRegressor();
/// regressor.setWithIntercept(true);
/// const fitted = regressor.fit(x, n_rows, n_cols, y);
/// const mu = fitted.predict(x_new, n_rows_new);
/// ```
#[wasm_bindgen]
pub struct GammaRegressor {
    with_intercept: bool,
    compute_inference: bool,
    confidence_level: f64,
    max_iterations: usize,
    tolerance: f64,
    lambda: f64,
}

#[wasm_bindgen]
impl GammaRegressor {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            with_intercept: true,
            compute_inference: true,
            confidence_level: 0.95,
            max_iterations: 25,
            tolerance: 1e-8,
            lambda: 0.0,
        }
    }

    #[wasm_bindgen(js_name = setWithIntercept)]
    pub fn set_with_intercept(&mut self, include: bool) {
        self.with_intercept = include;
    }

    #[wasm_bindgen(js_name = setComputeInference)]
    pub fn set_compute_inference(&mut self, compute: bool) {
        self.compute_inference = compute;
    }

    #[wasm_bindgen(js_name = setConfidenceLevel)]
    pub fn set_confidence_level(&mut self, level: f64) {
        self.confidence_level = level;
    }

    #[wasm_bindgen(js_name = setMaxIterations)]
    pub fn set_max_iterations(&mut self, max_iter: usize) {
        self.max_iterations = max_iter;
    }

    #[wasm_bindgen(js_name = setTolerance)]
    pub fn set_tolerance(&mut self, tol: f64) {
        self.tolerance = tol;
    }

    /// L2 (ridge) penalty applied to non-intercept coefficients.
    #[wasm_bindgen(js_name = setLambda)]
    pub fn set_lambda(&mut self, lambda: f64) {
        self.lambda = lambda;
    }

    #[wasm_bindgen]
    pub fn fit(
        &self,
        x: &[f64],
        n_rows: usize,
        n_cols: usize,
        y: &[f64],
    ) -> Result<FittedGamma, JsError> {
        if x.len() != n_rows * n_cols {
            return Err(JsError::new("Invalid x dimensions"));
        }
        if y.len() != n_rows {
            return Err(JsError::new("Invalid y length"));
        }
        if y.iter().any(|&yi| yi <= 0.0) {
            return Err(JsError::new("Gamma regression requires y > 0"));
        }

        let x_mat = Mat::from_fn(n_rows, n_cols, |i, j| x[i * n_cols + j]);
        let y_col = Col::from_fn(n_rows, |i| y[i]);

        let model = RustGammaRegressor::builder()
            .with_intercept(self.with_intercept)
            .compute_inference(self.compute_inference)
            .confidence_level(self.confidence_level)
            .max_iterations(self.max_iterations)
            .tolerance(self.tolerance)
            .lambda(self.lambda)
            .build();

        let fitted = model
            .fit(&x_mat, &y_col)
            .map_err(|e| JsError::new(&format!("Regression failed: {}", e)))?;

        Ok(FittedGamma {
            fitted,
            n_features: n_cols,
        })
    }
}

impl Default for GammaRegressor {
    fn default() -> Self {
        Self::new()
    }
}
