//! Ridge Regression wrapper for WebAssembly.

use anofox_regression::core::NaAction;
use anofox_regression::solvers::{
    FittedRegressor, Regressor, RidgeRegressor as RustRidgeRegressor,
};
use faer::{Col, Mat};
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

use super::parse_na_action;

/// Result from fitting a Ridge regression model.
#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RidgeResult {
    pub coefficients: Vec<f64>,
    pub intercept: Option<f64>,
    pub r_squared: f64,
    pub adj_r_squared: f64,
    pub rmse: f64,
    pub mse: f64,
    pub lambda: f64,
    pub n_observations: usize,
    pub n_parameters: usize,
    pub residuals: Vec<f64>,
    pub fitted_values: Vec<f64>,
    pub std_errors: Option<Vec<f64>>,
    pub t_statistics: Option<Vec<f64>>,
    pub p_values: Option<Vec<f64>>,
    pub confidence_level: f64,
}

/// A fitted Ridge regression model.
#[wasm_bindgen]
pub struct FittedRidge {
    fitted: anofox_regression::solvers::FittedRidge,
    n_features: usize,
    lambda: f64,
}

#[wasm_bindgen]
impl FittedRidge {
    /// Get the regression result as a JavaScript object.
    #[wasm_bindgen(js_name = getResult)]
    pub fn get_result(&self) -> Result<JsValue, JsError> {
        let result = self.fitted.result();

        let ridge_result = RidgeResult {
            coefficients: result.coefficients.iter().copied().collect(),
            intercept: result.intercept,
            r_squared: result.r_squared,
            adj_r_squared: result.adj_r_squared,
            rmse: result.rmse,
            mse: result.mse,
            lambda: self.lambda,
            n_observations: result.n_observations,
            n_parameters: result.n_parameters,
            residuals: result.residuals_expanded().iter().copied().collect(),
            fitted_values: result.fitted_expanded().iter().copied().collect(),
            std_errors: result
                .std_errors
                .as_ref()
                .map(|v| v.iter().copied().collect()),
            t_statistics: result
                .t_statistics
                .as_ref()
                .map(|v| v.iter().copied().collect()),
            p_values: result
                .p_values
                .as_ref()
                .map(|v| v.iter().copied().collect()),
            confidence_level: result.confidence_level,
        };

        serde_wasm_bindgen::to_value(&ridge_result).map_err(|e| JsError::new(&e.to_string()))
    }

    /// Get the coefficients as a Float64Array.
    #[wasm_bindgen(js_name = getCoefficients)]
    pub fn get_coefficients(&self) -> Vec<f64> {
        self.fitted.result().coefficients.iter().copied().collect()
    }

    /// Get the intercept.
    #[wasm_bindgen(js_name = getIntercept)]
    pub fn get_intercept(&self) -> Option<f64> {
        self.fitted.result().intercept
    }

    /// Get R-squared.
    #[wasm_bindgen(js_name = getRSquared)]
    pub fn get_r_squared(&self) -> f64 {
        self.fitted.result().r_squared
    }

    /// Make predictions on new data.
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

/// Ridge regression with L2 regularization.
///
/// Minimizes: ||y - Xb||^2 + lambda * ||b||^2
#[wasm_bindgen]
pub struct RidgeRegressor {
    na_action: NaAction,
    with_intercept: bool,
    compute_inference: bool,
    confidence_level: f64,
    lambda: f64,
    /// `(lambda_small, cond_threshold)` of conditioning-aware λ selection.
    conditioning: Option<(f64, f64)>,
}

#[wasm_bindgen]
impl RidgeRegressor {
    /// Create a new Ridge regressor with default options.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            na_action: NaAction::default(),
            with_intercept: true,
            compute_inference: true,
            confidence_level: 0.95,
            lambda: 1.0,
            conditioning: None,
        }
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

    /// Set the confidence level for confidence intervals.
    #[wasm_bindgen(js_name = setConfidenceLevel)]
    pub fn set_confidence_level(&mut self, level: f64) {
        self.confidence_level = level;
    }

    /// Set the regularization parameter lambda.
    #[wasm_bindgen(js_name = setLambda)]
    pub fn set_lambda(&mut self, lambda: f64) {
        self.lambda = lambda;
    }

    /// Set the handling of rows with NaN in `x` or `y` (R's `na.action`):
    /// "omit" (default), "exclude" (residuals and fitted values padded with
    /// NaN at the dropped rows), "fail" or "pass".
    #[wasm_bindgen(js_name = setNaAction)]
    pub fn set_na_action(&mut self, action: &str) -> Result<(), JsError> {
        self.na_action = parse_na_action(action)?;
        Ok(())
    }

    /// Fit the Ridge model to the data.
    #[wasm_bindgen]
    pub fn fit(
        &self,
        x: &[f64],
        n_rows: usize,
        n_cols: usize,
        y: &[f64],
    ) -> Result<FittedRidge, JsError> {
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

        let x_mat = Mat::from_fn(n_rows, n_cols, |i, j| x[i * n_cols + j]);
        let y_col = Col::from_fn(n_rows, |i| y[i]);

        let model = self.rust();

        let fitted = model
            .fit(&x_mat, &y_col)
            .map_err(|e| JsError::new(&format!("Regression failed: {}", e)))?;

        Ok(self.wrap(fitted, n_cols))
    }
}

impl Default for RidgeRegressor {
    fn default() -> Self {
        Self::new()
    }
}
partial_residuals_method!(FittedRidge);

impl crate::convert::ToJs for anofox_regression::solvers::RidgeLoo {
    fn to_js(&self) -> JsValue {
        crate::convert::object(&[
            ("leverage", crate::convert::ToJs::to_js(&self.leverage)),
            (
                "loo_residual",
                crate::convert::ToJs::to_js(&self.loo_residual),
            ),
            ("loo_fitted", crate::convert::ToJs::to_js(&self.loo_fitted)),
            ("press", crate::convert::ToJs::to_js(&self.press)),
        ])
    }
}

#[wasm_bindgen]
impl FittedRidge {
    /// Exact leave-one-out diagnostics of the ridge fit (same λ):
    /// `{ leverage, looResidual, looFitted, press: { press, pressRmse,
    /// looRSquared, n } }`. `x` and `y` are the data of the fit.
    #[wasm_bindgen]
    pub fn loo(&self, x: &[f64], n_rows: usize, y: &[f64]) -> Result<JsValue, JsError> {
        let x_mat = super::to_mat(x, n_rows, self.n_features)?;
        let y_col = super::to_col(y, n_rows, "target")?;
        let loo = self.fitted.loo(&x_mat, &y_col).map_err(super::fit_error)?;
        Ok(crate::convert::ToJs::to_js(&loo))
    }
}
predict_se_method!(FittedRidge);

#[wasm_bindgen]
impl FittedRidge {
    /// Covariance of the coefficients `σ̂² M`, intercept first (nested
    /// rows); undefined without a variance factor.
    #[wasm_bindgen]
    pub fn vcov(&self) -> JsValue {
        crate::convert::ToJs::to_js(&self.fitted.vcov())
    }

    /// Variance factor `M` with `Var(β̂) = σ² M`, intercept first.
    #[wasm_bindgen(js_name = varianceFactor)]
    pub fn variance_factor(&self) -> JsValue {
        crate::convert::ToJs::to_js(&self.fitted.variance_factor().cloned())
    }

    /// Leverage `x₀ᵀ M x₀` of new rows.
    #[wasm_bindgen(js_name = leverageNew)]
    pub fn leverage_new(&self, x: &[f64], n_rows: usize) -> Result<Vec<f64>, JsError> {
        let x_mat = super::to_mat(x, n_rows, self.n_features)?;
        Ok(self.fitted.leverage_new(&x_mat).iter().copied().collect())
    }
}

impl RidgeRegressor {
    fn rust(&self) -> RustRidgeRegressor {
        let mut b = RustRidgeRegressor::builder()
            .na_action(self.na_action)
            .with_intercept(self.with_intercept)
            .compute_inference(self.compute_inference)
            .confidence_level(self.confidence_level)
            .lambda(self.lambda);
        if let Some((small, threshold)) = self.conditioning {
            b = b.conditioning_aware(small, threshold);
        }
        b.build()
    }

    /// The λ the fit used (with conditioning-aware selection, possibly the
    /// small λ).
    fn wrap(&self, fitted: anofox_regression::solvers::FittedRidge, p: usize) -> FittedRidge {
        let lambda = fitted.lambda();
        FittedRidge {
            fitted,
            n_features: p,
            lambda,
        }
    }
}

#[wasm_bindgen]
impl RidgeRegressor {
    /// Fit from a `SufficientStats` state (λ applied here, so one state
    /// serves any λ).
    #[wasm_bindgen(js_name = fitFromState)]
    pub fn fit_from_state(
        &self,
        state: &crate::streaming::SufficientStats,
    ) -> Result<FittedRidge, JsError> {
        let f = self
            .rust()
            .fit_from_state(&state.inner)
            .map_err(super::fit_error)?;
        Ok(self.wrap(f, state.inner.n_features()))
    }

    /// Fit from a `MomentAccumulator`.
    #[wasm_bindgen(js_name = fitFromAccumulator)]
    pub fn fit_from_accumulator(
        &self,
        acc: &crate::streaming::MomentAccumulator,
    ) -> Result<FittedRidge, JsError> {
        let f = self
            .rust()
            .fit_from_accumulator(&acc.inner)
            .map_err(super::fit_error)?;
        Ok(self.wrap(f, acc.inner.n_features()))
    }

    /// Fit from raw moments: `xtx` (row-major p × p), `xty`, `sumX`, `sumY`
    /// and `n`.
    #[wasm_bindgen(js_name = fitFromMoments)]
    pub fn fit_from_moments(
        &self,
        xtx: &[f64],
        xty: &[f64],
        sum_x: &[f64],
        sum_y: f64,
        n: usize,
    ) -> Result<FittedRidge, JsError> {
        let p = xty.len();
        let f = self
            .rust()
            .fit_from_moments(
                &crate::inference::square(xtx, p, "xtx")?,
                &super::to_col(xty, p, "xty")?,
                &super::to_col(sum_x, p, "sumX")?,
                sum_y,
                n,
            )
            .map_err(super::fit_error)?;
        Ok(self.wrap(f, p))
    }

    /// The ridge path (glmnet's objective, `alpha = 0`) at `lambdas`.
    #[wasm_bindgen]
    pub fn path(
        &self,
        x: &[f64],
        n_rows: usize,
        n_cols: usize,
        y: &[f64],
        lambdas: &[f64],
    ) -> Result<crate::path::RegPath, JsError> {
        let p = self
            .rust()
            .path(
                &super::to_mat(x, n_rows, n_cols)?,
                &super::to_col(y, n_rows, "y")?,
                lambdas,
            )
            .map_err(super::fit_error)?;
        Ok(crate::path::RegPath { inner: p })
    }

    /// K-fold CV of the ridge path with fold labels `foldIds`.
    #[allow(clippy::too_many_arguments)]
    #[wasm_bindgen(js_name = cvPath)]
    pub fn cv_path(
        &self,
        x: &[f64],
        n_rows: usize,
        n_cols: usize,
        y: &[f64],
        lambdas: &[f64],
        fold_ids: &[u32],
    ) -> Result<crate::path::CvPath, JsError> {
        let folds: Vec<usize> = fold_ids.iter().map(|&f| f as usize).collect();
        let c = self
            .rust()
            .cv_path(
                &super::to_mat(x, n_rows, n_cols)?,
                &super::to_col(y, n_rows, "y")?,
                lambdas,
                &folds,
            )
            .map_err(super::fit_error)?;
        Ok(crate::path::CvPath { inner: c })
    }
}

#[wasm_bindgen]
impl RidgeRegressor {
    /// Conditioning-aware λ selection: when the condition number `κ(XᵀX)`
    /// of the penalised design is at most `condThreshold` (default 1000),
    /// fit with `lambdaSmall` (default 0.1) instead of the configured λ,
    /// which would only over-shrink a well-conditioned design. The λ used
    /// is reported by the fit.
    #[wasm_bindgen(js_name = setConditioning)]
    pub fn set_conditioning(&mut self, lambda_small: Option<f64>, cond_threshold: Option<f64>) {
        use anofox_regression::solvers::{DEFAULT_CONDITION_THRESHOLD, DEFAULT_LAMBDA_SMALL};
        self.conditioning = Some((
            lambda_small.unwrap_or(DEFAULT_LAMBDA_SMALL),
            cond_threshold.unwrap_or(DEFAULT_CONDITION_THRESHOLD),
        ));
    }

    /// Turn conditioning-aware λ selection off (the default).
    #[wasm_bindgen(js_name = clearConditioning)]
    pub fn clear_conditioning(&mut self) {
        self.conditioning = None;
    }

    /// The λ a fit on the row-major `x` would use: the configured λ, or the
    /// small λ when conditioning-aware selection is on and the design is
    /// well-conditioned.
    #[wasm_bindgen(js_name = selectLambda)]
    pub fn select_lambda(&self, x: &[f64], n_rows: usize, n_cols: usize) -> Result<f64, JsError> {
        Ok(self
            .rust()
            .select_lambda(&super::to_mat(x, n_rows, n_cols)?))
    }
}
