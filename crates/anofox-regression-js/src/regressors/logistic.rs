//! Logistic Regression wrapper for WebAssembly.

use anofox_regression::solvers::{
    FittedLogistic as RustFittedLogistic, FittedRegressor,
    LogisticRegression as RustLogisticRegression, Penalty,
};
use faer::{Col, Mat};
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

/// Result from fitting a Logistic regression classifier.
#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LogisticResult {
    /// Estimated coefficients (excluding intercept).
    pub coefficients: Vec<f64>,
    /// Intercept term (null if model was fit without intercept).
    pub intercept: Option<f64>,
    /// Residual deviance.
    pub deviance: f64,
    /// Null deviance (intercept-only model).
    pub null_deviance: f64,
    /// Number of IRLS iterations used to fit the model.
    pub iterations: usize,
    /// Number of observations.
    pub n_observations: usize,
    /// Akaike Information Criterion.
    pub aic: f64,
    /// Bayesian Information Criterion.
    pub bic: f64,
    /// Residuals from the underlying GLM.
    pub residuals: Vec<f64>,
    /// Fitted probabilities P(Y=1|X) on the training data.
    pub fitted_values: Vec<f64>,
    /// Standard errors of coefficients (if inference was computed).
    pub std_errors: Option<Vec<f64>>,
    /// P-values for coefficient significance tests (if inference was computed).
    pub p_values: Option<Vec<f64>>,
}

/// A fitted Logistic regression classifier.
#[wasm_bindgen]
pub struct FittedLogistic {
    fitted: RustFittedLogistic,
    n_features: usize,
}

#[wasm_bindgen]
impl FittedLogistic {
    /// Get the regression result as a JavaScript object.
    #[wasm_bindgen(js_name = getResult)]
    pub fn get_result(&self) -> Result<JsValue, JsError> {
        let inner = self.fitted.inner();
        let result = inner.result();

        let log_result = LogisticResult {
            coefficients: result.coefficients.iter().copied().collect(),
            intercept: result.intercept,
            deviance: inner.deviance,
            null_deviance: inner.null_deviance,
            iterations: self.fitted.n_iter(),
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

        serde_wasm_bindgen::to_value(&log_result).map_err(|e| JsError::new(&e.to_string()))
    }

    /// Get the coefficients as a Float64Array.
    #[wasm_bindgen(js_name = getCoefficients)]
    pub fn get_coefficients(&self) -> Vec<f64> {
        self.fitted.coefficients().iter().copied().collect()
    }

    /// Get the intercept (returns null if model was fit without intercept).
    #[wasm_bindgen(js_name = getIntercept)]
    pub fn get_intercept(&self) -> Option<f64> {
        self.fitted.intercept()
    }

    /// Get the number of IRLS iterations used to fit the model.
    #[wasm_bindgen(js_name = getNIter)]
    pub fn get_n_iter(&self) -> usize {
        self.fitted.n_iter()
    }

    /// Predict class labels (0.0 or 1.0) using the configured threshold.
    #[wasm_bindgen]
    pub fn predict(&self, x: &[f64], n_rows: usize) -> Result<Vec<f64>, JsError> {
        let x_mat = self.build_matrix(x, n_rows)?;
        Ok(self.fitted.predict(&x_mat).iter().copied().collect())
    }

    /// Predict probabilities P(Y=1|X) for the given data.
    #[wasm_bindgen(js_name = predictProba)]
    pub fn predict_proba(&self, x: &[f64], n_rows: usize) -> Result<Vec<f64>, JsError> {
        let x_mat = self.build_matrix(x, n_rows)?;
        Ok(self.fitted.predict_proba(&x_mat).iter().copied().collect())
    }

    /// Compute the decision function (log-odds / linear predictor).
    #[wasm_bindgen(js_name = decisionFunction)]
    pub fn decision_function(&self, x: &[f64], n_rows: usize) -> Result<Vec<f64>, JsError> {
        let x_mat = self.build_matrix(x, n_rows)?;
        Ok(self
            .fitted
            .decision_function(&x_mat)
            .iter()
            .copied()
            .collect())
    }

    /// Compute classification accuracy on the given data.
    #[wasm_bindgen]
    pub fn score(&self, x: &[f64], n_rows: usize, y: &[f64]) -> Result<f64, JsError> {
        let x_mat = self.build_matrix(x, n_rows)?;
        if y.len() != n_rows {
            return Err(JsError::new(&format!(
                "Expected {} target values, got {}",
                n_rows,
                y.len()
            )));
        }
        let y_col = Col::from_fn(n_rows, |i| y[i]);
        Ok(self.fitted.score(&x_mat, &y_col))
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

/// Logistic regression classifier (binary classification).
///
/// # Example (JavaScript)
/// ```javascript
/// const regressor = new LogisticRegression();
/// regressor.setL2(1.0);
/// regressor.setThreshold(0.5);
///
/// const fitted = regressor.fit(x, n_rows, n_cols, y);
/// const labels = fitted.predict(x_new, n_rows_new);
/// const probs = fitted.predictProba(x_new, n_rows_new);
/// ```
#[wasm_bindgen]
pub struct LogisticRegression {
    l2_lambda: Option<f64>,
    with_intercept: bool,
    threshold: f64,
    max_iterations: usize,
    tolerance: f64,
    compute_inference: bool,
    confidence_level: f64,
}

#[wasm_bindgen]
impl LogisticRegression {
    /// Create a new Logistic regression model with default options.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            l2_lambda: None,
            with_intercept: true,
            threshold: 0.5,
            max_iterations: 100,
            tolerance: 1e-8,
            compute_inference: true,
            confidence_level: 0.95,
        }
    }

    /// Enable L2 regularization with the given lambda.
    ///
    /// Pass any non-negative value; `0.0` disables regularization. The
    /// scikit-learn–style `C` inverse-strength parameter equals `1.0 / lambda`.
    #[wasm_bindgen(js_name = setL2)]
    pub fn set_l2(&mut self, lambda: f64) {
        if lambda > 0.0 {
            self.l2_lambda = Some(lambda);
        } else {
            self.l2_lambda = None;
        }
    }

    /// Set L2 strength using sklearn's `C` parameter (lambda = 1/C).
    #[wasm_bindgen(js_name = setC)]
    pub fn set_c(&mut self, c_value: f64) {
        if c_value > 0.0 {
            self.l2_lambda = Some(1.0 / c_value);
        } else {
            self.l2_lambda = None;
        }
    }

    /// Set whether to include an intercept term (default: true).
    #[wasm_bindgen(js_name = setWithIntercept)]
    pub fn set_with_intercept(&mut self, include: bool) {
        self.with_intercept = include;
    }

    /// Set the classification threshold (default: 0.5).
    #[wasm_bindgen(js_name = setThreshold)]
    pub fn set_threshold(&mut self, threshold: f64) {
        self.threshold = threshold;
    }

    /// Set the maximum number of IRLS iterations (default: 100).
    #[wasm_bindgen(js_name = setMaxIterations)]
    pub fn set_max_iterations(&mut self, max_iter: usize) {
        self.max_iterations = max_iter;
    }

    /// Set the convergence tolerance (default: 1e-8).
    #[wasm_bindgen(js_name = setTolerance)]
    pub fn set_tolerance(&mut self, tol: f64) {
        self.tolerance = tol;
    }

    /// Set whether to compute inference statistics (default: true).
    #[wasm_bindgen(js_name = setComputeInference)]
    pub fn set_compute_inference(&mut self, compute: bool) {
        self.compute_inference = compute;
    }

    /// Set the confidence level for confidence intervals (default: 0.95).
    #[wasm_bindgen(js_name = setConfidenceLevel)]
    pub fn set_confidence_level(&mut self, level: f64) {
        self.confidence_level = level;
    }

    /// Fit the logistic regression model to binary target data.
    ///
    /// # Arguments
    /// * `x` - Feature matrix as a flat array (row-major order)
    /// * `n_rows` - Number of rows (observations)
    /// * `n_cols` - Number of columns (features)
    /// * `y` - Binary target values (0.0 or 1.0)
    #[wasm_bindgen]
    pub fn fit(
        &self,
        x: &[f64],
        n_rows: usize,
        n_cols: usize,
        y: &[f64],
    ) -> Result<FittedLogistic, JsError> {
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

        let penalty = match self.l2_lambda {
            Some(lambda) => Penalty::L2(lambda),
            None => Penalty::None,
        };

        let model = RustLogisticRegression::builder()
            .penalty(penalty)
            .with_intercept(self.with_intercept)
            .threshold(self.threshold)
            .max_iterations(self.max_iterations)
            .tolerance(self.tolerance)
            .compute_inference(self.compute_inference)
            .confidence_level(self.confidence_level)
            .build();

        let fitted = model
            .fit(&x_mat, &y_col)
            .map_err(|e| JsError::new(&format!("Regression failed: {}", e)))?;

        Ok(FittedLogistic {
            fitted,
            n_features: n_cols,
        })
    }
}

impl Default for LogisticRegression {
    fn default() -> Self {
        Self::new()
    }
}
