//! Passive-Aggressive Regression wrapper for WebAssembly.

use anofox_regression::solvers::{
    FittedPassiveAggressive as RustFittedPa, FittedRegressor, PaLoss,
    PassiveAggressiveRegressor as RustPaRegressor, Regressor,
};
use faer::{Col, Mat};
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PaResult {
    pub coefficients: Vec<f64>,
    pub intercept: Option<f64>,
    pub n_iter: usize,
    pub r_squared: f64,
    pub n_observations: usize,
    pub residuals: Vec<f64>,
    pub fitted_values: Vec<f64>,
}

#[wasm_bindgen]
pub struct FittedPassiveAggressive {
    fitted: RustFittedPa,
    n_features: usize,
}

#[wasm_bindgen]
impl FittedPassiveAggressive {
    #[wasm_bindgen(js_name = getResult)]
    pub fn get_result(&self) -> Result<JsValue, JsError> {
        let r = self.fitted.result();
        let out = PaResult {
            coefficients: r.coefficients.iter().copied().collect(),
            intercept: r.intercept,
            n_iter: self.fitted.n_iter(),
            r_squared: r.r_squared,
            n_observations: r.n_observations,
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

    #[wasm_bindgen(js_name = getNIter)]
    pub fn get_n_iter(&self) -> usize {
        self.fitted.n_iter()
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

/// Passive-Aggressive online regression.
#[wasm_bindgen]
pub struct PassiveAggressiveRegressor {
    c: f64,
    epsilon: f64,
    with_intercept: bool,
    max_iter: usize,
    tolerance: f64,
    shuffle: bool,
    loss: String,
    random_state: u64,
}

#[wasm_bindgen]
impl PassiveAggressiveRegressor {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            c: 1.0,
            epsilon: 0.1,
            with_intercept: true,
            max_iter: 1000,
            tolerance: 1e-3,
            shuffle: true,
            loss: "epsilon_insensitive".to_string(),
            random_state: 0,
        }
    }

    /// Regularisation constant; larger `C` ⇒ more aggressive updates.
    #[wasm_bindgen(js_name = setC)]
    pub fn set_c(&mut self, value: f64) {
        self.c = value;
    }

    #[wasm_bindgen(js_name = setEpsilon)]
    pub fn set_epsilon(&mut self, value: f64) {
        self.epsilon = value;
    }

    #[wasm_bindgen(js_name = setWithIntercept)]
    pub fn set_with_intercept(&mut self, include: bool) {
        self.with_intercept = include;
    }

    #[wasm_bindgen(js_name = setMaxIter)]
    pub fn set_max_iter(&mut self, value: usize) {
        self.max_iter = value;
    }

    #[wasm_bindgen(js_name = setTolerance)]
    pub fn set_tolerance(&mut self, value: f64) {
        self.tolerance = value;
    }

    #[wasm_bindgen(js_name = setShuffle)]
    pub fn set_shuffle(&mut self, value: bool) {
        self.shuffle = value;
    }

    /// Loss variant: `"epsilon_insensitive"` (PA-I, default) or
    /// `"squared_epsilon_insensitive"` (PA-II).
    #[wasm_bindgen(js_name = setLoss)]
    pub fn set_loss(&mut self, value: &str) {
        self.loss = value.to_string();
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
    ) -> Result<FittedPassiveAggressive, JsError> {
        if x.len() != n_rows * n_cols {
            return Err(JsError::new("Invalid x dimensions"));
        }
        if y.len() != n_rows {
            return Err(JsError::new("Invalid y length"));
        }
        let x_mat = Mat::from_fn(n_rows, n_cols, |i, j| x[i * n_cols + j]);
        let y_col = Col::from_fn(n_rows, |i| y[i]);

        let loss = match self.loss.as_str() {
            "squared_epsilon_insensitive" | "squared" => PaLoss::SquaredEpsilonInsensitive,
            _ => PaLoss::EpsilonInsensitive,
        };

        let model = RustPaRegressor::builder()
            .c(self.c)
            .epsilon(self.epsilon)
            .with_intercept(self.with_intercept)
            .max_iter(self.max_iter)
            .tolerance(self.tolerance)
            .shuffle(self.shuffle)
            .loss(loss)
            .random_state(self.random_state)
            .build();

        let fitted = model
            .fit(&x_mat, &y_col)
            .map_err(|e| JsError::new(&format!("Regression failed: {}", e)))?;

        Ok(FittedPassiveAggressive {
            fitted,
            n_features: n_cols,
        })
    }
}

impl Default for PassiveAggressiveRegressor {
    fn default() -> Self {
        Self::new()
    }
}
