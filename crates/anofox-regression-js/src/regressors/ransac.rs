//! RANSAC Regression wrapper for WebAssembly.

use anofox_regression::solvers::{
    FittedRansac as RustFittedRansac, FittedRegressor, RansacRegressor as RustRansacRegressor,
    Regressor,
};
use faer::{Col, Mat};
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RansacResult {
    pub coefficients: Vec<f64>,
    pub intercept: Option<f64>,
    pub r_squared: f64,
    pub n_observations: usize,
    pub n_inliers: usize,
    pub n_trials: usize,
    pub residual_threshold: f64,
    pub inlier_mask: Vec<u8>,
    pub residuals: Vec<f64>,
    pub fitted_values: Vec<f64>,
}

#[wasm_bindgen]
pub struct FittedRansac {
    fitted: RustFittedRansac,
    n_features: usize,
}

#[wasm_bindgen]
impl FittedRansac {
    #[wasm_bindgen(js_name = getResult)]
    pub fn get_result(&self) -> Result<JsValue, JsError> {
        let r = self.fitted.result();
        let out = RansacResult {
            coefficients: r.coefficients.iter().copied().collect(),
            intercept: r.intercept,
            r_squared: r.r_squared,
            n_observations: r.n_observations,
            n_inliers: self.fitted.n_inliers(),
            n_trials: self.fitted.n_trials(),
            residual_threshold: self.fitted.residual_threshold(),
            inlier_mask: self
                .fitted
                .inlier_mask()
                .iter()
                .map(|&b| if b { 1 } else { 0 })
                .collect(),
            residuals: r.residuals.iter().copied().collect(),
            fitted_values: r.fitted_values.iter().copied().collect(),
        };
        serde_wasm_bindgen::to_value(&out).map_err(|e| JsError::new(&e.to_string()))
    }

    #[wasm_bindgen(js_name = getCoefficients)]
    pub fn get_coefficients(&self) -> Vec<f64> {
        self.fitted.result().coefficients.iter().copied().collect()
    }

    #[wasm_bindgen(js_name = getIntercept)]
    pub fn get_intercept(&self) -> Option<f64> {
        self.fitted.result().intercept
    }

    #[wasm_bindgen(js_name = getInlierMask)]
    pub fn get_inlier_mask(&self) -> Vec<u8> {
        self.fitted
            .inlier_mask()
            .iter()
            .map(|&b| if b { 1 } else { 0 })
            .collect()
    }

    #[wasm_bindgen(js_name = getNInliers)]
    pub fn get_n_inliers(&self) -> usize {
        self.fitted.n_inliers()
    }

    #[wasm_bindgen(js_name = getNTrials)]
    pub fn get_n_trials(&self) -> usize {
        self.fitted.n_trials()
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

/// RANSAC (RAndom SAmple Consensus) regression.
#[wasm_bindgen]
pub struct RansacRegressor {
    with_intercept: bool,
    min_samples: Option<usize>,
    residual_threshold: Option<f64>,
    max_trials: usize,
    stop_probability: f64,
    stop_n_inliers: Option<usize>,
    random_state: u64,
}

#[wasm_bindgen]
impl RansacRegressor {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            with_intercept: true,
            min_samples: None,
            residual_threshold: None,
            max_trials: 100,
            stop_probability: 0.99,
            stop_n_inliers: None,
            random_state: 0,
        }
    }

    #[wasm_bindgen(js_name = setWithIntercept)]
    pub fn set_with_intercept(&mut self, include: bool) {
        self.with_intercept = include;
    }

    #[wasm_bindgen(js_name = setMinSamples)]
    pub fn set_min_samples(&mut self, value: usize) {
        self.min_samples = Some(value);
    }

    #[wasm_bindgen(js_name = setResidualThreshold)]
    pub fn set_residual_threshold(&mut self, value: f64) {
        self.residual_threshold = Some(value);
    }

    #[wasm_bindgen(js_name = setMaxTrials)]
    pub fn set_max_trials(&mut self, value: usize) {
        self.max_trials = value;
    }

    #[wasm_bindgen(js_name = setStopProbability)]
    pub fn set_stop_probability(&mut self, value: f64) {
        self.stop_probability = value;
    }

    #[wasm_bindgen(js_name = setStopNInliers)]
    pub fn set_stop_n_inliers(&mut self, value: usize) {
        self.stop_n_inliers = Some(value);
    }

    #[wasm_bindgen(js_name = setRandomState)]
    pub fn set_random_state(&mut self, seed: u64) {
        self.random_state = seed;
    }

    #[wasm_bindgen]
    pub fn fit(
        &self,
        x: &[f64],
        n_rows: usize,
        n_cols: usize,
        y: &[f64],
    ) -> Result<FittedRansac, JsError> {
        if x.len() != n_rows * n_cols {
            return Err(JsError::new("Invalid x dimensions"));
        }
        if y.len() != n_rows {
            return Err(JsError::new("Invalid y length"));
        }
        let x_mat = Mat::from_fn(n_rows, n_cols, |i, j| x[i * n_cols + j]);
        let y_col = Col::from_fn(n_rows, |i| y[i]);

        let mut builder = RustRansacRegressor::builder()
            .with_intercept(self.with_intercept)
            .max_trials(self.max_trials)
            .stop_probability(self.stop_probability)
            .random_state(self.random_state);
        if let Some(k) = self.min_samples {
            builder = builder.min_samples(k);
        }
        if let Some(t) = self.residual_threshold {
            builder = builder.residual_threshold(t);
        }
        if let Some(s) = self.stop_n_inliers {
            builder = builder.stop_n_inliers(s);
        }
        let model = builder.build();

        let fitted = model
            .fit(&x_mat, &y_col)
            .map_err(|e| JsError::new(&format!("Regression failed: {}", e)))?;

        Ok(FittedRansac {
            fitted,
            n_features: n_cols,
        })
    }
}

impl Default for RansacRegressor {
    fn default() -> Self {
        Self::new()
    }
}
