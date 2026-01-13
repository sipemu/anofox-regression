//! Augmented Linear Model (ALM) wrapper for WebAssembly.

use anofox_regression::solvers::{
    AlmDistribution as RustAlmDistribution, AlmRegressor as RustAlmRegressor,
    FittedAlm as RustFittedAlm, FittedRegressor, Regressor,
};
use faer::{Col, Mat};
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

/// Result from fitting an Augmented Linear Model.
#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AlmResult {
    pub coefficients: Vec<f64>,
    pub intercept: Option<f64>,
    pub r_squared: f64,
    pub adj_r_squared: f64,
    pub mse: f64,
    pub rmse: f64,
    pub log_likelihood: f64,
    pub aic: f64,
    pub bic: f64,
    pub n_observations: usize,
    pub residuals: Vec<f64>,
    pub fitted_values: Vec<f64>,
}

/// A fitted Augmented Linear Model.
#[wasm_bindgen]
pub struct FittedAlm {
    fitted: RustFittedAlm,
    n_features: usize,
}

#[wasm_bindgen]
impl FittedAlm {
    #[wasm_bindgen(js_name = getResult)]
    pub fn get_result(&self) -> Result<JsValue, JsError> {
        let result = self.fitted.result();
        let alm_result = AlmResult {
            coefficients: result.coefficients.iter().copied().collect(),
            intercept: result.intercept,
            r_squared: result.r_squared,
            adj_r_squared: result.adj_r_squared,
            mse: result.mse,
            rmse: result.rmse,
            log_likelihood: result.log_likelihood,
            aic: result.aic,
            bic: result.bic,
            n_observations: result.n_observations,
            residuals: result.residuals.iter().copied().collect(),
            fitted_values: result.fitted_values.iter().copied().collect(),
        };
        serde_wasm_bindgen::to_value(&alm_result).map_err(|e| JsError::new(&e.to_string()))
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

    #[wasm_bindgen(js_name = getScale)]
    pub fn get_scale(&self) -> f64 {
        self.fitted.scale()
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

/// Augmented Linear Model with various error distributions.
///
/// Supports: Normal, Laplace, Student-t, Logistic, AsymmetricLaplace,
/// GeneralisedNormal, LogNormal, Gamma, InverseGaussian, Exponential,
/// Poisson, NegativeBinomial, and more.
#[wasm_bindgen]
pub struct AlmRegressor {
    distribution: String,
    with_intercept: bool,
    compute_inference: bool,
    max_iterations: usize,
    tolerance: f64,
}

#[wasm_bindgen]
impl AlmRegressor {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            distribution: "normal".to_string(),
            with_intercept: true,
            compute_inference: true,
            max_iterations: 100,
            tolerance: 1e-8,
        }
    }

    /// Set the error distribution.
    ///
    /// Options: "normal", "laplace", "student_t", "logistic", "asymmetric_laplace",
    /// "generalised_normal", "log_normal", "log_laplace", "gamma", "inverse_gaussian",
    /// "exponential", "poisson", "negative_binomial", "beta"
    #[wasm_bindgen(js_name = setDistribution)]
    pub fn set_distribution(&mut self, dist: &str) {
        self.distribution = dist.to_string();
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

    fn parse_distribution(&self) -> RustAlmDistribution {
        match self.distribution.to_lowercase().as_str() {
            "laplace" => RustAlmDistribution::Laplace,
            "student_t" | "studentt" | "t" => RustAlmDistribution::StudentT,
            "logistic" => RustAlmDistribution::Logistic,
            "asymmetric_laplace" | "asymmetriclaplace" => RustAlmDistribution::AsymmetricLaplace,
            "generalised_normal" | "generalisednormal" | "gnormal" => {
                RustAlmDistribution::GeneralisedNormal
            }
            "log_normal" | "lognormal" => RustAlmDistribution::LogNormal,
            "log_laplace" | "loglaplace" => RustAlmDistribution::LogLaplace,
            "gamma" => RustAlmDistribution::Gamma,
            "inverse_gaussian" | "inversegaussian" | "invgaussian" => {
                RustAlmDistribution::InverseGaussian
            }
            "exponential" | "exp" => RustAlmDistribution::Exponential,
            "poisson" => RustAlmDistribution::Poisson,
            "negative_binomial" | "negativebinomial" | "negbin" => {
                RustAlmDistribution::NegativeBinomial
            }
            "beta" => RustAlmDistribution::Beta,
            "folded_normal" | "foldednormal" => RustAlmDistribution::FoldedNormal,
            "s" => RustAlmDistribution::S,
            _ => RustAlmDistribution::Normal,
        }
    }

    #[wasm_bindgen]
    pub fn fit(
        &self,
        x: &[f64],
        n_rows: usize,
        n_cols: usize,
        y: &[f64],
    ) -> Result<FittedAlm, JsError> {
        if x.len() != n_rows * n_cols {
            return Err(JsError::new("Invalid x dimensions"));
        }
        if y.len() != n_rows {
            return Err(JsError::new("Invalid y length"));
        }

        let x_mat = Mat::from_fn(n_rows, n_cols, |i, j| x[i * n_cols + j]);
        let y_col = Col::from_fn(n_rows, |i| y[i]);

        let distribution = self.parse_distribution();

        let model = RustAlmRegressor::builder()
            .distribution(distribution)
            .with_intercept(self.with_intercept)
            .compute_inference(self.compute_inference)
            .max_iterations(self.max_iterations)
            .tolerance(self.tolerance)
            .build();

        let fitted = model
            .fit(&x_mat, &y_col)
            .map_err(|e| JsError::new(&format!("Regression failed: {}", e)))?;

        Ok(FittedAlm {
            fitted,
            n_features: n_cols,
        })
    }
}

impl Default for AlmRegressor {
    fn default() -> Self {
        Self::new()
    }
}
