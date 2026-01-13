//! Poisson Regression wrapper for WebAssembly.

use anofox_regression::core::PoissonLink;
use anofox_regression::solvers::{
    FittedPoisson as RustFittedPoisson, FittedRegressor, PoissonRegressor as RustPoissonRegressor,
    Regressor,
};
use faer::{Col, Mat};
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

/// Result from fitting a Poisson regression model.
#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PoissonResult {
    pub coefficients: Vec<f64>,
    pub intercept: Option<f64>,
    pub deviance: f64,
    pub null_deviance: f64,
    pub dispersion: f64,
    pub iterations: usize,
    pub n_observations: usize,
    pub n_parameters: usize,
    pub aic: f64,
    pub bic: f64,
    pub residuals: Vec<f64>,
    pub fitted_values: Vec<f64>,
    pub std_errors: Option<Vec<f64>>,
    pub p_values: Option<Vec<f64>>,
}

/// A fitted Poisson regression model.
#[wasm_bindgen]
pub struct FittedPoisson {
    fitted: RustFittedPoisson,
    n_features: usize,
}

#[wasm_bindgen]
impl FittedPoisson {
    /// Get the regression result as a JavaScript object.
    #[wasm_bindgen(js_name = getResult)]
    pub fn get_result(&self) -> Result<JsValue, JsError> {
        let result = self.fitted.result();

        let poisson_result = PoissonResult {
            coefficients: result.coefficients.iter().copied().collect(),
            intercept: result.intercept,
            deviance: self.fitted.deviance,
            null_deviance: self.fitted.null_deviance,
            dispersion: self.fitted.dispersion,
            iterations: self.fitted.iterations,
            n_observations: result.n_observations,
            n_parameters: result.n_parameters,
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

        serde_wasm_bindgen::to_value(&poisson_result).map_err(|e| JsError::new(&e.to_string()))
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

    /// Get the deviance.
    #[wasm_bindgen(js_name = getDeviance)]
    pub fn get_deviance(&self) -> f64 {
        self.fitted.deviance
    }

    /// Make predictions (predicted counts).
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

/// Poisson GLM regression estimator.
///
/// Models count data using Poisson distribution with configurable link function.
#[wasm_bindgen]
pub struct PoissonRegressor {
    link: String,
    with_intercept: bool,
    compute_inference: bool,
    max_iterations: usize,
    tolerance: f64,
}

#[wasm_bindgen]
impl PoissonRegressor {
    /// Create a new Poisson regressor with default options (log link).
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            link: "log".to_string(),
            with_intercept: true,
            compute_inference: true,
            max_iterations: 25,
            tolerance: 1e-8,
        }
    }

    /// Set the link function: "log" (default), "identity", or "sqrt".
    #[wasm_bindgen(js_name = setLink)]
    pub fn set_link(&mut self, link: &str) {
        self.link = link.to_string();
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

    /// Set the maximum iterations for IRLS.
    #[wasm_bindgen(js_name = setMaxIterations)]
    pub fn set_max_iterations(&mut self, max_iter: usize) {
        self.max_iterations = max_iter;
    }

    /// Fit the Poisson model to count data.
    #[wasm_bindgen]
    pub fn fit(
        &self,
        x: &[f64],
        n_rows: usize,
        n_cols: usize,
        y: &[f64],
    ) -> Result<FittedPoisson, JsError> {
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

        // Check for negative counts
        if y.iter().any(|&yi| yi < 0.0) {
            return Err(JsError::new(
                "Poisson regression requires non-negative count data",
            ));
        }

        let x_mat = Mat::from_fn(n_rows, n_cols, |i, j| x[i * n_cols + j]);
        let y_col = Col::from_fn(n_rows, |i| y[i]);

        let link = match self.link.as_str() {
            "identity" => PoissonLink::Identity,
            "sqrt" => PoissonLink::Sqrt,
            _ => PoissonLink::Log,
        };

        let model = RustPoissonRegressor::builder()
            .link(link)
            .with_intercept(self.with_intercept)
            .compute_inference(self.compute_inference)
            .max_iterations(self.max_iterations)
            .tolerance(self.tolerance)
            .build();

        let fitted = model
            .fit(&x_mat, &y_col)
            .map_err(|e| JsError::new(&format!("Regression failed: {}", e)))?;

        Ok(FittedPoisson {
            fitted,
            n_features: n_cols,
        })
    }
}

impl Default for PoissonRegressor {
    fn default() -> Self {
        Self::new()
    }
}
