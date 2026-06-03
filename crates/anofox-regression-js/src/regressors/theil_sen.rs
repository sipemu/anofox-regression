//! Theil-Sen Regression wrapper for WebAssembly.

use anofox_regression::solvers::{
    FittedRegressor, FittedTheilSen as RustFittedTheilSen, Regressor,
    TheilSenRegressor as RustTheilSenRegressor,
};
use faer::{Col, Mat};
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TheilSenResult {
    pub coefficients: Vec<f64>,
    pub intercept: Option<f64>,
    pub r_squared: f64,
    pub adj_r_squared: f64,
    pub rmse: f64,
    pub mse: f64,
    pub n_observations: usize,
    pub n_parameters: usize,
    pub residuals: Vec<f64>,
    pub fitted_values: Vec<f64>,
}

#[wasm_bindgen]
pub struct FittedTheilSen {
    fitted: RustFittedTheilSen,
    n_features: usize,
}

#[wasm_bindgen]
impl FittedTheilSen {
    #[wasm_bindgen(js_name = getResult)]
    pub fn get_result(&self) -> Result<JsValue, JsError> {
        let r = self.fitted.result();
        let out = TheilSenResult {
            coefficients: r.coefficients.iter().copied().collect(),
            intercept: r.intercept,
            r_squared: r.r_squared,
            adj_r_squared: r.adj_r_squared,
            rmse: r.rmse,
            mse: r.mse,
            n_observations: r.n_observations,
            n_parameters: r.n_parameters,
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

/// Theil-Sen robust regression.
///
/// Computes the spatial (L1) median of OLS coefficient vectors fit on every
/// (or up to `maxSubpopulation`) subsample of size `nFeatures + 1`. Highly
/// robust to outliers (breakdown point ~29.3%).
#[wasm_bindgen]
pub struct TheilSenRegressor {
    with_intercept: bool,
    max_subpopulation: usize,
    n_subsamples: Option<usize>,
    max_iter: usize,
    tolerance: f64,
    random_state: u64,
}

#[wasm_bindgen]
impl TheilSenRegressor {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            with_intercept: true,
            max_subpopulation: 10_000,
            n_subsamples: None,
            max_iter: 300,
            tolerance: 1e-3,
            random_state: 0,
        }
    }

    #[wasm_bindgen(js_name = setWithIntercept)]
    pub fn set_with_intercept(&mut self, include: bool) {
        self.with_intercept = include;
    }

    #[wasm_bindgen(js_name = setMaxSubpopulation)]
    pub fn set_max_subpopulation(&mut self, value: usize) {
        self.max_subpopulation = value;
    }

    /// Sets the subsample size (default = `n_features + 1`).
    #[wasm_bindgen(js_name = setNSubsamples)]
    pub fn set_n_subsamples(&mut self, value: usize) {
        self.n_subsamples = Some(value);
    }

    #[wasm_bindgen(js_name = setMaxIter)]
    pub fn set_max_iter(&mut self, value: usize) {
        self.max_iter = value;
    }

    #[wasm_bindgen(js_name = setTolerance)]
    pub fn set_tolerance(&mut self, value: f64) {
        self.tolerance = value;
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
    ) -> Result<FittedTheilSen, JsError> {
        if x.len() != n_rows * n_cols {
            return Err(JsError::new("Invalid x dimensions"));
        }
        if y.len() != n_rows {
            return Err(JsError::new("Invalid y length"));
        }
        let x_mat = Mat::from_fn(n_rows, n_cols, |i, j| x[i * n_cols + j]);
        let y_col = Col::from_fn(n_rows, |i| y[i]);

        let mut builder = RustTheilSenRegressor::builder()
            .with_intercept(self.with_intercept)
            .max_subpopulation(self.max_subpopulation)
            .max_iter(self.max_iter)
            .tolerance(self.tolerance)
            .random_state(self.random_state);
        if let Some(k) = self.n_subsamples {
            builder = builder.n_subsamples(k);
        }
        let model = builder.build();

        let fitted = model
            .fit(&x_mat, &y_col)
            .map_err(|e| JsError::new(&format!("Regression failed: {}", e)))?;

        Ok(FittedTheilSen {
            fitted,
            n_features: n_cols,
        })
    }
}

impl Default for TheilSenRegressor {
    fn default() -> Self {
        Self::new()
    }
}
