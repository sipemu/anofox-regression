//! Huber Regression wrapper for WebAssembly.

use anofox_regression::solvers::{
    FittedHuber as RustFittedHuber, FittedRegressor, HuberRegressor as RustHuberRegressor,
    Regressor,
};
use faer::{Col, Mat};
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

/// Result from fitting a Huber regression model.
#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HuberResult {
    /// Estimated coefficients (excluding intercept).
    pub coefficients: Vec<f64>,
    /// Intercept term (null if model was fit without intercept).
    pub intercept: Option<f64>,
    /// R-squared (coefficient of determination).
    pub r_squared: f64,
    /// Adjusted R-squared.
    pub adj_r_squared: f64,
    /// Root mean squared error.
    pub rmse: f64,
    /// Mean squared error.
    pub mse: f64,
    /// Number of observations.
    pub n_observations: usize,
    /// Number of parameters (including intercept if present).
    pub n_parameters: usize,
    /// Residual degrees of freedom.
    pub residual_df: usize,
    /// Akaike Information Criterion.
    pub aic: f64,
    /// Bayesian Information Criterion.
    pub bic: f64,
    /// Residuals (y - fitted_values).
    pub residuals: Vec<f64>,
    /// Fitted values (predictions on training data).
    pub fitted_values: Vec<f64>,
    /// Estimated scale (sigma) from the MAD estimator.
    pub scale: f64,
    /// Huber threshold parameter used during fitting.
    pub epsilon: f64,
    /// Boolean mask: true if observation was flagged as an outlier.
    pub outliers: Vec<bool>,
    /// Number of detected outliers.
    pub n_outliers: usize,
}

/// A fitted Huber regression model.
#[wasm_bindgen]
pub struct FittedHuber {
    fitted: RustFittedHuber,
    n_features: usize,
}

#[wasm_bindgen]
impl FittedHuber {
    /// Get the regression result as a JavaScript object.
    #[wasm_bindgen(js_name = getResult)]
    pub fn get_result(&self) -> Result<JsValue, JsError> {
        let result = self.fitted.result();

        let huber_result = HuberResult {
            coefficients: result.coefficients.iter().copied().collect(),
            intercept: result.intercept,
            r_squared: result.r_squared,
            adj_r_squared: result.adj_r_squared,
            rmse: result.rmse,
            mse: result.mse,
            n_observations: result.n_observations,
            n_parameters: result.n_parameters,
            residual_df: result.residual_df(),
            aic: result.aic,
            bic: result.bic,
            residuals: result.residuals.iter().copied().collect(),
            fitted_values: result.fitted_values.iter().copied().collect(),
            scale: self.fitted.scale(),
            epsilon: self.fitted.epsilon(),
            outliers: self.fitted.outliers().to_vec(),
            n_outliers: self.fitted.n_outliers(),
        };

        serde_wasm_bindgen::to_value(&huber_result).map_err(|e| JsError::new(&e.to_string()))
    }

    /// Get the coefficients as a Float64Array.
    #[wasm_bindgen(js_name = getCoefficients)]
    pub fn get_coefficients(&self) -> Vec<f64> {
        self.fitted.result().coefficients.iter().copied().collect()
    }

    /// Get the intercept (returns null if model was fit without intercept).
    #[wasm_bindgen(js_name = getIntercept)]
    pub fn get_intercept(&self) -> Option<f64> {
        self.fitted.result().intercept
    }

    /// Get the estimated scale (sigma) from the MAD estimator.
    #[wasm_bindgen(js_name = getScale)]
    pub fn get_scale(&self) -> f64 {
        self.fitted.scale()
    }

    /// Get the epsilon parameter used during fitting.
    #[wasm_bindgen(js_name = getEpsilon)]
    pub fn get_epsilon(&self) -> f64 {
        self.fitted.epsilon()
    }

    /// Get the number of detected outliers.
    #[wasm_bindgen(js_name = getNOutliers)]
    pub fn get_n_outliers(&self) -> usize {
        self.fitted.n_outliers()
    }

    /// Get the outlier mask as a boolean array (Uint8Array of 0/1 in JS).
    #[wasm_bindgen(js_name = getOutliers)]
    pub fn get_outliers(&self) -> Vec<u8> {
        self.fitted
            .outliers()
            .iter()
            .map(|&b| if b { 1u8 } else { 0u8 })
            .collect()
    }

    /// Make predictions on new data.
    ///
    /// # Arguments
    /// * `x` - Feature matrix as a flat array (row-major order)
    /// * `n_rows` - Number of rows in the matrix
    ///
    /// # Returns
    /// Predictions as a Float64Array
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

/// Huber regression — a robust regression estimator that is resilient to outliers.
///
/// Uses the Huber loss function which behaves like squared loss for small residuals
/// and absolute loss for large residuals, fit via IRLS with optional L2 regularization.
///
/// # Example (JavaScript)
/// ```javascript
/// const regressor = new HuberRegressor();
/// regressor.setEpsilon(1.35);
/// regressor.setAlpha(0.0001);
///
/// const fitted = regressor.fit(x, n_rows, n_cols, y);
/// console.log(fitted.getNOutliers());
/// ```
#[wasm_bindgen]
pub struct HuberRegressor {
    epsilon: f64,
    alpha: f64,
    with_intercept: bool,
    max_iterations: usize,
    tolerance: f64,
}

#[wasm_bindgen]
impl HuberRegressor {
    /// Create a new Huber regressor with default options.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            epsilon: 1.35,
            alpha: 0.0001,
            with_intercept: true,
            max_iterations: 100,
            tolerance: 1e-5,
        }
    }

    /// Set the Huber threshold parameter (default: 1.35, must be > 1.0).
    #[wasm_bindgen(js_name = setEpsilon)]
    pub fn set_epsilon(&mut self, epsilon: f64) {
        self.epsilon = epsilon;
    }

    /// Set the L2 regularization parameter (default: 0.0001).
    #[wasm_bindgen(js_name = setAlpha)]
    pub fn set_alpha(&mut self, alpha: f64) {
        self.alpha = alpha;
    }

    /// Set whether to include an intercept term (default: true).
    #[wasm_bindgen(js_name = setWithIntercept)]
    pub fn set_with_intercept(&mut self, include: bool) {
        self.with_intercept = include;
    }

    /// Set the maximum number of IRLS iterations (default: 100).
    #[wasm_bindgen(js_name = setMaxIterations)]
    pub fn set_max_iterations(&mut self, max_iter: usize) {
        self.max_iterations = max_iter;
    }

    /// Set the convergence tolerance (default: 1e-5).
    #[wasm_bindgen(js_name = setTolerance)]
    pub fn set_tolerance(&mut self, tol: f64) {
        self.tolerance = tol;
    }

    /// Fit the Huber model to the data.
    ///
    /// # Arguments
    /// * `x` - Feature matrix as a flat array (row-major order)
    /// * `n_rows` - Number of rows (observations)
    /// * `n_cols` - Number of columns (features)
    /// * `y` - Target values
    #[wasm_bindgen]
    pub fn fit(
        &self,
        x: &[f64],
        n_rows: usize,
        n_cols: usize,
        y: &[f64],
    ) -> Result<FittedHuber, JsError> {
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
                "Expected {} target values for {} observations, got {}",
                n_rows,
                n_rows,
                y.len()
            )));
        }

        let x_mat = Mat::from_fn(n_rows, n_cols, |i, j| x[i * n_cols + j]);
        let y_col = Col::from_fn(n_rows, |i| y[i]);

        let model = RustHuberRegressor::builder()
            .epsilon(self.epsilon)
            .alpha(self.alpha)
            .with_intercept(self.with_intercept)
            .max_iterations(self.max_iterations)
            .tolerance(self.tolerance)
            .build();

        let fitted = model
            .fit(&x_mat, &y_col)
            .map_err(|e| JsError::new(&format!("Regression failed: {}", e)))?;

        Ok(FittedHuber {
            fitted,
            n_features: n_cols,
        })
    }
}

impl Default for HuberRegressor {
    fn default() -> Self {
        Self::new()
    }
}
