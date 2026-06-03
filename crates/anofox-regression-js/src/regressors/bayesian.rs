//! Bayesian Ridge and ARD wrappers for WebAssembly.

use anofox_regression::solvers::{
    ArdRegression as RustArd, BayesianRidge as RustBayesianRidge, FittedArd as RustFittedArd,
    FittedBayesianRidge as RustFittedBayesianRidge, FittedRegressor, Regressor,
};
use faer::{Col, Mat};
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

// --------- BayesianRidge ----------

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BayesianRidgeResult {
    pub coefficients: Vec<f64>,
    pub intercept: Option<f64>,
    pub alpha: f64,
    pub lambda: f64,
    pub sigma_diag: Vec<f64>,
    pub r_squared: f64,
    pub n_observations: usize,
    pub residuals: Vec<f64>,
    pub fitted_values: Vec<f64>,
}

#[wasm_bindgen]
pub struct FittedBayesianRidge {
    fitted: RustFittedBayesianRidge,
    n_features: usize,
}

#[wasm_bindgen]
impl FittedBayesianRidge {
    #[wasm_bindgen(js_name = getResult)]
    pub fn get_result(&self) -> Result<JsValue, JsError> {
        let r = self.fitted.result();
        let out = BayesianRidgeResult {
            coefficients: r.coefficients.iter().copied().collect(),
            intercept: r.intercept,
            alpha: self.fitted.alpha(),
            lambda: self.fitted.lambda(),
            sigma_diag: self.fitted.sigma_diag().to_vec(),
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

    #[wasm_bindgen(js_name = getAlpha)]
    pub fn get_alpha(&self) -> f64 {
        self.fitted.alpha()
    }

    #[wasm_bindgen(js_name = getLambda)]
    pub fn get_lambda(&self) -> f64 {
        self.fitted.lambda()
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

/// Bayesian Ridge regression with evidence-maximisation hyperparameters.
#[wasm_bindgen]
pub struct BayesianRidge {
    fit_intercept: bool,
    max_iter: usize,
    tol: f64,
    alpha_1: f64,
    alpha_2: f64,
    lambda_1: f64,
    lambda_2: f64,
}

#[wasm_bindgen]
impl BayesianRidge {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            fit_intercept: true,
            max_iter: 300,
            tol: 1e-3,
            alpha_1: 1e-6,
            alpha_2: 1e-6,
            lambda_1: 1e-6,
            lambda_2: 1e-6,
        }
    }

    #[wasm_bindgen(js_name = setFitIntercept)]
    pub fn set_fit_intercept(&mut self, v: bool) {
        self.fit_intercept = v;
    }
    #[wasm_bindgen(js_name = setMaxIter)]
    pub fn set_max_iter(&mut self, v: usize) {
        self.max_iter = v;
    }
    #[wasm_bindgen(js_name = setTolerance)]
    pub fn set_tolerance(&mut self, v: f64) {
        self.tol = v;
    }
    #[wasm_bindgen(js_name = setAlpha1)]
    pub fn set_alpha_1(&mut self, v: f64) {
        self.alpha_1 = v;
    }
    #[wasm_bindgen(js_name = setAlpha2)]
    pub fn set_alpha_2(&mut self, v: f64) {
        self.alpha_2 = v;
    }
    #[wasm_bindgen(js_name = setLambda1)]
    pub fn set_lambda_1(&mut self, v: f64) {
        self.lambda_1 = v;
    }
    #[wasm_bindgen(js_name = setLambda2)]
    pub fn set_lambda_2(&mut self, v: f64) {
        self.lambda_2 = v;
    }

    #[wasm_bindgen]
    pub fn fit(
        &self,
        x: &[f64],
        n_rows: usize,
        n_cols: usize,
        y: &[f64],
    ) -> Result<FittedBayesianRidge, JsError> {
        if x.len() != n_rows * n_cols {
            return Err(JsError::new("Invalid x dimensions"));
        }
        if y.len() != n_rows {
            return Err(JsError::new("Invalid y length"));
        }
        let x_mat = Mat::from_fn(n_rows, n_cols, |i, j| x[i * n_cols + j]);
        let y_col = Col::from_fn(n_rows, |i| y[i]);
        let model = RustBayesianRidge::builder()
            .fit_intercept(self.fit_intercept)
            .max_iter(self.max_iter)
            .tolerance(self.tol)
            .alpha_1(self.alpha_1)
            .alpha_2(self.alpha_2)
            .lambda_1(self.lambda_1)
            .lambda_2(self.lambda_2)
            .build();
        let fitted = model
            .fit(&x_mat, &y_col)
            .map_err(|e| JsError::new(&format!("Regression failed: {}", e)))?;
        Ok(FittedBayesianRidge {
            fitted,
            n_features: n_cols,
        })
    }
}

impl Default for BayesianRidge {
    fn default() -> Self {
        Self::new()
    }
}

// --------- ARDRegression ----------

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ArdResult {
    pub coefficients: Vec<f64>,
    pub intercept: Option<f64>,
    pub alpha: f64,
    pub lambdas: Vec<f64>,
    pub r_squared: f64,
    pub n_observations: usize,
    pub residuals: Vec<f64>,
    pub fitted_values: Vec<f64>,
}

#[wasm_bindgen]
pub struct FittedArd {
    fitted: RustFittedArd,
    n_features: usize,
}

#[wasm_bindgen]
impl FittedArd {
    #[wasm_bindgen(js_name = getResult)]
    pub fn get_result(&self) -> Result<JsValue, JsError> {
        let r = self.fitted.result();
        let out = ArdResult {
            coefficients: r.coefficients.iter().copied().collect(),
            intercept: r.intercept,
            alpha: self.fitted.alpha(),
            lambdas: self.fitted.lambdas().to_vec(),
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

    #[wasm_bindgen(js_name = getLambdas)]
    pub fn get_lambdas(&self) -> Vec<f64> {
        self.fitted.lambdas().to_vec()
    }

    #[wasm_bindgen(js_name = getAlpha)]
    pub fn get_alpha(&self) -> f64 {
        self.fitted.alpha()
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

/// Automatic Relevance Determination regression.
#[wasm_bindgen]
pub struct ArdRegression {
    fit_intercept: bool,
    max_iter: usize,
    tol: f64,
    alpha_1: f64,
    alpha_2: f64,
    lambda_1: f64,
    lambda_2: f64,
    threshold_lambda: f64,
}

#[wasm_bindgen]
impl ArdRegression {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            fit_intercept: true,
            max_iter: 300,
            tol: 1e-3,
            alpha_1: 1e-6,
            alpha_2: 1e-6,
            lambda_1: 1e-6,
            lambda_2: 1e-6,
            threshold_lambda: 10_000.0,
        }
    }

    #[wasm_bindgen(js_name = setFitIntercept)]
    pub fn set_fit_intercept(&mut self, v: bool) {
        self.fit_intercept = v;
    }
    #[wasm_bindgen(js_name = setMaxIter)]
    pub fn set_max_iter(&mut self, v: usize) {
        self.max_iter = v;
    }
    #[wasm_bindgen(js_name = setTolerance)]
    pub fn set_tolerance(&mut self, v: f64) {
        self.tol = v;
    }
    #[wasm_bindgen(js_name = setAlpha1)]
    pub fn set_alpha_1(&mut self, v: f64) {
        self.alpha_1 = v;
    }
    #[wasm_bindgen(js_name = setAlpha2)]
    pub fn set_alpha_2(&mut self, v: f64) {
        self.alpha_2 = v;
    }
    #[wasm_bindgen(js_name = setLambda1)]
    pub fn set_lambda_1(&mut self, v: f64) {
        self.lambda_1 = v;
    }
    #[wasm_bindgen(js_name = setLambda2)]
    pub fn set_lambda_2(&mut self, v: f64) {
        self.lambda_2 = v;
    }
    #[wasm_bindgen(js_name = setThresholdLambda)]
    pub fn set_threshold_lambda(&mut self, v: f64) {
        self.threshold_lambda = v;
    }

    #[wasm_bindgen]
    pub fn fit(
        &self,
        x: &[f64],
        n_rows: usize,
        n_cols: usize,
        y: &[f64],
    ) -> Result<FittedArd, JsError> {
        if x.len() != n_rows * n_cols {
            return Err(JsError::new("Invalid x dimensions"));
        }
        if y.len() != n_rows {
            return Err(JsError::new("Invalid y length"));
        }
        let x_mat = Mat::from_fn(n_rows, n_cols, |i, j| x[i * n_cols + j]);
        let y_col = Col::from_fn(n_rows, |i| y[i]);
        let model = RustArd::builder()
            .fit_intercept(self.fit_intercept)
            .max_iter(self.max_iter)
            .tolerance(self.tol)
            .alpha_1(self.alpha_1)
            .alpha_2(self.alpha_2)
            .lambda_1(self.lambda_1)
            .lambda_2(self.lambda_2)
            .threshold_lambda(self.threshold_lambda)
            .build();
        let fitted = model
            .fit(&x_mat, &y_col)
            .map_err(|e| JsError::new(&format!("Regression failed: {}", e)))?;
        Ok(FittedArd {
            fitted,
            n_features: n_cols,
        })
    }
}

impl Default for ArdRegression {
    fn default() -> Self {
        Self::new()
    }
}
