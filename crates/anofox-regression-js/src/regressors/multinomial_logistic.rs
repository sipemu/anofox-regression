//! Multinomial / binary logistic regression with an elastic-net penalty
//! (sklearn `LogisticRegression`) for WebAssembly.

use anofox_regression::solvers::{
    FittedMultinomialLogistic as RustFitted, MultinomialLogisticRegressor as RustRegressor,
};
use faer::{Col, Mat};
use wasm_bindgen::prelude::*;

use crate::convert::ToJs;

/// A fitted multinomial / binary logistic regression.
#[wasm_bindgen]
pub struct FittedMultinomialLogistic {
    fitted: RustFitted,
    n_features: usize,
}

#[wasm_bindgen]
impl FittedMultinomialLogistic {
    /// Rebuild a fitted model from `classes()`, `coefficients()` (row-major
    /// `rows × nFeatures`, one row for two classes, `K` otherwise) and
    /// `intercepts()` of an earlier fit; it predicts as that fit.
    #[wasm_bindgen(js_name = fromParts)]
    pub fn from_parts(
        classes: Vec<f64>,
        coefficients: &[f64],
        intercepts: Vec<f64>,
        n_features: usize,
    ) -> Result<FittedMultinomialLogistic, JsError> {
        let rows = intercepts.len();
        let b = crate::regressors::to_mat(coefficients, rows, n_features)?;
        let fitted = RustFitted::from_parts(classes, b, Col::from_fn(rows, |k| intercepts[k]))
            .map_err(crate::regressors::fit_error)?;
        Ok(Self { fitted, n_features })
    }

    /// The sorted distinct class labels.
    #[wasm_bindgen]
    pub fn classes(&self) -> Vec<f64> {
        self.fitted.classes().to_vec()
    }

    /// Coefficients, one array per linear predictor (one for two classes,
    /// `K` otherwise).
    #[wasm_bindgen]
    pub fn coefficients(&self) -> JsValue {
        self.fitted.coefficients().to_js()
    }

    /// Intercepts, one per linear predictor.
    #[wasm_bindgen]
    pub fn intercepts(&self) -> Vec<f64> {
        self.fitted.intercepts().iter().copied().collect()
    }

    /// Linear predictors, one array per row.
    #[wasm_bindgen(js_name = decisionFunction)]
    pub fn decision_function(&self, x: &[f64], n_rows: usize) -> Result<JsValue, JsError> {
        Ok(self
            .fitted
            .decision_function(&self.matrix(x, n_rows)?)
            .to_js())
    }

    /// Class probabilities, one array of `K` per row (classes order).
    #[wasm_bindgen(js_name = predictProba)]
    pub fn predict_proba(&self, x: &[f64], n_rows: usize) -> Result<JsValue, JsError> {
        Ok(self.fitted.predict_proba(&self.matrix(x, n_rows)?).to_js())
    }

    /// The most probable class of each row.
    #[wasm_bindgen]
    pub fn predict(&self, x: &[f64], n_rows: usize) -> Result<Vec<f64>, JsError> {
        let p = self.fitted.predict(&self.matrix(x, n_rows)?);
        Ok(p.iter().copied().collect())
    }

    /// Classification accuracy.
    #[wasm_bindgen]
    pub fn score(&self, x: &[f64], n_rows: usize, y: &[f64]) -> Result<f64, JsError> {
        if y.len() != n_rows {
            return Err(JsError::new("y length must equal n_rows"));
        }
        let xm = self.matrix(x, n_rows)?;
        Ok(self.fitted.score(&xm, &Col::from_fn(n_rows, |i| y[i])))
    }

    /// Weighted log-likelihood at the fit.
    #[wasm_bindgen(js_name = logLikelihood)]
    pub fn log_likelihood(&self) -> f64 {
        self.fitted.log_likelihood()
    }

    /// The minimised penalized objective.
    #[wasm_bindgen]
    pub fn objective(&self) -> f64 {
        self.fitted.objective()
    }

    fn matrix(&self, x: &[f64], n_rows: usize) -> Result<Mat<f64>, JsError> {
        let p = self.n_features;
        if x.len() != n_rows * p {
            return Err(JsError::new("Invalid dimensions"));
        }
        Ok(Mat::from_fn(n_rows, p, |i, j| x[i * p + j]))
    }
}

iterative_fit!(FittedMultinomialLogistic);

/// Multinomial (softmax) / binary logistic regression with an elastic-net
/// penalty, as sklearn's `LogisticRegression` (`lambda = 1 / C`).
#[wasm_bindgen]
pub struct MultinomialLogisticRegressor {
    lambda: f64,
    l1_ratio: f64,
    with_intercept: bool,
    weights: Option<Vec<f64>>,
    max_iterations: usize,
    tolerance: f64,
    error_on_non_convergence: bool,
}

impl Default for MultinomialLogisticRegressor {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen]
impl MultinomialLogisticRegressor {
    /// sklearn's defaults: `C = 1`, ridge, an intercept.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            lambda: 1.0,
            l1_ratio: 0.0,
            with_intercept: true,
            weights: None,
            max_iterations: 100,
            tolerance: 1e-8,
            error_on_non_convergence: true,
        }
    }

    /// Penalty strength `λ ≥ 0` (default 1).
    #[wasm_bindgen(js_name = setLambda)]
    pub fn set_lambda(&mut self, lambda: f64) {
        self.lambda = lambda;
    }

    /// sklearn's `C` (`λ = 1 / C`).
    #[wasm_bindgen(js_name = setC)]
    pub fn set_c(&mut self, c: f64) {
        self.lambda = 1.0 / c;
    }

    /// Elastic-net mixing in `[0, 1]` (0 ridge, 1 lasso).
    #[wasm_bindgen(js_name = setL1Ratio)]
    pub fn set_l1_ratio(&mut self, l1_ratio: f64) {
        self.l1_ratio = l1_ratio;
    }

    /// Fit intercepts (default true).
    #[wasm_bindgen(js_name = setWithIntercept)]
    pub fn set_with_intercept(&mut self, include: bool) {
        self.with_intercept = include;
    }

    /// Non-negative sample weights.
    #[wasm_bindgen(js_name = setWeights)]
    pub fn set_weights(&mut self, weights: Vec<f64>) {
        self.weights = Some(weights);
    }

    /// Maximum proximal Newton iterations (default 100).
    #[wasm_bindgen(js_name = setMaxIterations)]
    pub fn set_max_iterations(&mut self, max_iterations: usize) {
        self.max_iterations = max_iterations;
    }

    /// Step tolerance (default 1e-8).
    #[wasm_bindgen(js_name = setTolerance)]
    pub fn set_tolerance(&mut self, tolerance: f64) {
        self.tolerance = tolerance;
    }

    /// Throw when `maxIterations` is exhausted (default true).
    #[wasm_bindgen(js_name = setErrorOnNonConvergence)]
    pub fn set_error_on_non_convergence(&mut self, error: bool) {
        self.error_on_non_convergence = error;
    }

    /// Fit to class labels `y` (row-major `x`).
    #[wasm_bindgen]
    pub fn fit(
        &self,
        x: &[f64],
        n_rows: usize,
        n_cols: usize,
        y: &[f64],
    ) -> Result<FittedMultinomialLogistic, JsError> {
        if x.len() != n_rows * n_cols || y.len() != n_rows {
            return Err(JsError::new("Invalid dimensions"));
        }
        let xm = Mat::from_fn(n_rows, n_cols, |i, j| x[i * n_cols + j]);
        let ym = Col::from_fn(n_rows, |i| y[i]);
        let mut b = RustRegressor::builder()
            .lambda(self.lambda)
            .l1_ratio(self.l1_ratio)
            .with_intercept(self.with_intercept)
            .max_iterations(self.max_iterations)
            .tolerance(self.tolerance)
            .error_on_non_convergence(self.error_on_non_convergence);
        if let Some(w) = &self.weights {
            b = b.weights(Col::from_fn(w.len(), |i| w[i]));
        }
        let fitted = b
            .build()
            .fit(&xm, &ym)
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(FittedMultinomialLogistic {
            fitted,
            n_features: n_cols,
        })
    }
}
