//! Continuation-ratio ordinal regression wrapper for WebAssembly.

use anofox_regression::solvers::{
    cr_setup, ContinuationRatioRegressor as RustContinuationRatioRegressor,
    FittedContinuationRatio as RustFittedContinuationRatio,
};
use faer::Mat;
use serde::Serialize;
use wasm_bindgen::prelude::*;

use super::{col_vec, fit_error, mat_flat, mat_rows, to_col, to_js, to_mat};

/// Result from fitting a continuation-ratio model.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ContinuationRatioResult {
    /// The distinct response values.
    pub levels: Vec<f64>,
    /// Intercept followed by the cohort effects, the slopes and the
    /// cohort-specific slope differences (columns of the expanded design).
    pub coefficients: Vec<f64>,
    /// Standard errors of `coefficients`.
    pub std_errors: Vec<f64>,
    /// Covariance of `coefficients`, as nested rows.
    pub vcov: Vec<Vec<f64>>,
    pub log_likelihood: f64,
    pub iterations: usize,
    pub converged: bool,
}

/// The `rms::cr.setup` expansion of an ordered response.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CrSetupResult {
    /// Binary response of the expanded rows (1 = stops at this cohort).
    pub y: Vec<f64>,
    /// Cohort (0-based) of each expanded row.
    pub cohort: Vec<usize>,
    /// Original row of each expanded row.
    pub subs: Vec<usize>,
    /// Number of expanded rows per original row.
    pub reps: Vec<usize>,
    /// The distinct response values.
    pub levels: Vec<f64>,
}

/// The `rms::cr.setup` expansion of the ordered response `y`:
/// `{ y, cohort, subs, reps, levels }`.
#[wasm_bindgen(js_name = crSetup)]
pub fn cr_setup_js(y: &[f64]) -> Result<JsValue, JsError> {
    let y = to_col(y, y.len(), "target")?;
    let d = cr_setup(&y).map_err(fit_error)?;
    to_js(&CrSetupResult {
        y: col_vec(d.y.as_ref()),
        cohort: d.cohort,
        subs: d.subs,
        reps: d.reps,
        levels: d.levels,
    })
}

/// A fitted continuation-ratio model.
#[wasm_bindgen]
pub struct FittedContinuationRatio {
    fitted: RustFittedContinuationRatio,
}

impl FittedContinuationRatio {
    fn mat(&self, x: &[f64], n_rows: usize) -> Result<Mat<f64>, JsError> {
        to_mat(x, n_rows, self.fitted.n_features())
    }
}

#[wasm_bindgen]
impl FittedContinuationRatio {
    /// Get the full result as a JavaScript object.
    #[wasm_bindgen(js_name = getResult)]
    pub fn get_result(&self) -> Result<JsValue, JsError> {
        use anofox_regression::solvers::IterativeFit;
        let f = &self.fitted;
        let vcov = f.vcov();
        to_js(&ContinuationRatioResult {
            levels: f.levels().to_vec(),
            coefficients: col_vec(f.coefficients().as_ref()),
            std_errors: (0..vcov.nrows()).map(|j| vcov[(j, j)].sqrt()).collect(),
            vcov: mat_rows(&vcov),
            log_likelihood: f.log_likelihood(),
            iterations: f.iterations(),
            converged: f.converged(),
        })
    }

    /// Intercept followed by the coefficients of the expanded design.
    #[wasm_bindgen(js_name = getCoefficients)]
    pub fn get_coefficients(&self) -> Vec<f64> {
        col_vec(self.fitted.coefficients().as_ref())
    }

    /// The distinct response values.
    #[wasm_bindgen(js_name = getLevels)]
    pub fn get_levels(&self) -> Vec<f64> {
        self.fitted.levels().to_vec()
    }

    /// The log-likelihood.
    #[wasm_bindgen(js_name = getLogLikelihood)]
    pub fn get_log_likelihood(&self) -> f64 {
        self.fitted.log_likelihood()
    }

    /// Covariance of the coefficients, row-major.
    #[wasm_bindgen]
    pub fn vcov(&self) -> Vec<f64> {
        mat_flat(&self.fitted.vcov())
    }

    /// `P(Y = y_j | Y ≥ y_j, x)`, row-major `n_rows × (levels − 1)`.
    #[wasm_bindgen(js_name = conditionalProbabilities)]
    pub fn conditional_probabilities(&self, x: &[f64], n_rows: usize) -> Result<Vec<f64>, JsError> {
        Ok(mat_flat(
            &self.fitted.conditional_probabilities(&self.mat(x, n_rows)?),
        ))
    }

    /// `P(Y = y_j | x)`, row-major `n_rows × levels`.
    #[wasm_bindgen(js_name = categoryProbabilities)]
    pub fn category_probabilities(&self, x: &[f64], n_rows: usize) -> Result<Vec<f64>, JsError> {
        Ok(mat_flat(
            &self.fitted.category_probabilities(&self.mat(x, n_rows)?),
        ))
    }
}

/// Continuation-ratio model for an ordered response, fitted as a logistic
/// regression on the `rms::cr.setup` expansion.
#[wasm_bindgen]
pub struct ContinuationRatioRegressor {
    cohort_specific: Vec<usize>,
    weights: Option<Vec<f64>>,
}

#[wasm_bindgen]
impl ContinuationRatioRegressor {
    /// Create a regressor with equal slopes in every cohort.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            cohort_specific: Vec::new(),
            weights: None,
        }
    }

    /// Columns of `x` whose slope may differ by cohort.
    #[wasm_bindgen(js_name = setCohortSpecific)]
    pub fn set_cohort_specific(&mut self, columns: &[u32]) {
        self.cohort_specific = columns.iter().map(|&c| c as usize).collect();
    }

    /// Case weights of the subjects, one per row.
    #[wasm_bindgen(js_name = setWeights)]
    pub fn set_weights(&mut self, weights: Vec<f64>) {
        self.weights = Some(weights);
    }

    /// Fit to predictors `x` (no intercept column) and the ordered response
    /// `y` (at least three distinct values).
    #[wasm_bindgen]
    pub fn fit(
        &self,
        x: &[f64],
        n_rows: usize,
        n_cols: usize,
        y: &[f64],
    ) -> Result<FittedContinuationRatio, JsError> {
        let x_mat = to_mat(x, n_rows, n_cols)?;
        let y_col = to_col(y, n_rows, "target")?;
        let mut builder =
            RustContinuationRatioRegressor::builder().cohort_specific(&self.cohort_specific);
        if let Some(w) = &self.weights {
            builder = builder.weights(w);
        }
        let fitted = builder.build().fit(&x_mat, &y_col).map_err(fit_error)?;
        Ok(FittedContinuationRatio { fitted })
    }
}

impl Default for ContinuationRatioRegressor {
    fn default() -> Self {
        Self::new()
    }
}

iterative_fit!(FittedContinuationRatio);
