//! P-spline smoother wrapper for WebAssembly.

use anofox_regression::core::IntervalType;
use anofox_regression::solvers::{
    FittedPSpline as RustFittedPSpline, FittedRegressor, PSplineRegressor as RustPSplineRegressor,
    Regressor,
};
use serde::Serialize;
use wasm_bindgen::prelude::*;

use super::{col_vec, fit_error, to_col, to_js, to_mat};

/// Result from fitting a P-spline smoother.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PSplineResult {
    /// B-spline basis coefficients.
    pub coefficients: Vec<f64>,
    /// Effective degrees of freedom (trace of the smoother matrix).
    pub edf: f64,
    /// Residual variance estimate.
    pub sigma2: f64,
    /// Smoothing parameter of the difference penalty.
    pub lambda: f64,
    /// R-squared of the smooth.
    pub r_squared: f64,
    pub n_observations: usize,
    pub residuals: Vec<f64>,
    pub fitted_values: Vec<f64>,
}

/// Point predictions with interval bounds and standard errors.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct IntervalPrediction {
    pub fit: Vec<f64>,
    pub lower: Vec<f64>,
    pub upper: Vec<f64>,
    pub se: Vec<f64>,
}

/// Parse an interval kind: "confidence" or "prediction".
pub(crate) fn parse_interval(kind: &str) -> Result<IntervalType, JsError> {
    match kind {
        "confidence" => Ok(IntervalType::Confidence),
        "prediction" => Ok(IntervalType::Prediction),
        other => Err(JsError::new(&format!(
            "Unknown interval '{}': expected confidence or prediction",
            other
        ))),
    }
}

/// A fitted P-spline smoother.
#[wasm_bindgen]
pub struct FittedPSpline {
    fitted: RustFittedPSpline,
}

#[wasm_bindgen]
impl FittedPSpline {
    /// Get the fit summary as a JavaScript object.
    #[wasm_bindgen(js_name = getResult)]
    pub fn get_result(&self) -> Result<JsValue, JsError> {
        let result = self.fitted.result();
        to_js(&PSplineResult {
            coefficients: col_vec(result.coefficients.as_ref()),
            edf: self.fitted.edf(),
            sigma2: self.fitted.sigma2(),
            lambda: self.fitted.lambda(),
            r_squared: result.r_squared,
            n_observations: result.n_observations,
            residuals: col_vec(result.residuals.as_ref()),
            fitted_values: col_vec(result.fitted_values.as_ref()),
        })
    }

    /// Effective degrees of freedom.
    #[wasm_bindgen(js_name = getEdf)]
    pub fn get_edf(&self) -> f64 {
        self.fitted.edf()
    }

    /// Residual variance estimate σ².
    #[wasm_bindgen(js_name = getSigma2)]
    pub fn get_sigma2(&self) -> f64 {
        self.fitted.sigma2()
    }

    /// The (fixed or GCV-selected) smoothing parameter λ.
    #[wasm_bindgen(js_name = getLambda)]
    pub fn get_lambda(&self) -> f64 {
        self.fitted.lambda()
    }

    /// Smoothed values at new predictor values `x`.
    #[wasm_bindgen]
    pub fn predict(&self, x: &[f64]) -> Result<Vec<f64>, JsError> {
        let x_mat = to_mat(x, x.len(), 1)?;
        Ok(col_vec(self.fitted.predict(&x_mat).as_ref()))
    }

    /// Smoothed values with `"confidence"` or `"prediction"` intervals at
    /// `level` (e.g. 0.95): `{ fit, lower, upper, se }`.
    #[wasm_bindgen(js_name = predictInterval)]
    pub fn predict_interval(
        &self,
        x: &[f64],
        interval: &str,
        level: f64,
    ) -> Result<JsValue, JsError> {
        let x_mat = to_mat(x, x.len(), 1)?;
        let pred =
            self.fitted
                .predict_with_interval(&x_mat, Some(parse_interval(interval)?), level);
        to_js(&IntervalPrediction {
            fit: col_vec(pred.fit.as_ref()),
            lower: col_vec(pred.lower.as_ref()),
            upper: col_vec(pred.upper.as_ref()),
            se: col_vec(pred.se.as_ref()),
        })
    }
}

/// Penalized B-spline (P-spline) smoother of one predictor, as
/// `mgcv::gam(y ~ s(x, bs = "ps"), method = "GCV.Cp")`.
#[wasm_bindgen]
pub struct PSplineRegressor {
    n_basis: usize,
    penalty_order: usize,
    lambda: Option<f64>,
}

#[wasm_bindgen]
impl PSplineRegressor {
    /// Create a P-spline smoother with automatic basis size and
    /// GCV-selected smoothing.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            n_basis: 0,
            penalty_order: 2,
            lambda: None,
        }
    }

    /// Set the number of basis functions (0 = automatic).
    #[wasm_bindgen(js_name = setNBasis)]
    pub fn set_n_basis(&mut self, k: usize) {
        self.n_basis = k;
    }

    /// Set the difference-penalty order (default 2).
    #[wasm_bindgen(js_name = setPenaltyOrder)]
    pub fn set_penalty_order(&mut self, order: usize) {
        self.penalty_order = order;
    }

    /// Use a fixed smoothing parameter instead of GCV.
    #[wasm_bindgen(js_name = setLambda)]
    pub fn set_lambda(&mut self, lambda: f64) {
        self.lambda = Some(lambda);
    }

    /// Fit the smoother to the predictor `x` and response `y`.
    #[wasm_bindgen]
    pub fn fit(&self, x: &[f64], y: &[f64]) -> Result<FittedPSpline, JsError> {
        let n = x.len();
        let x_mat = to_mat(x, n, 1)?;
        let y_col = to_col(y, n, "target")?;
        let mut model = RustPSplineRegressor::new()
            .with_n_basis(self.n_basis)
            .with_penalty_order(self.penalty_order);
        if let Some(lambda) = self.lambda {
            model = model.with_lambda(lambda);
        }
        let fitted = model.fit(&x_mat, &y_col).map_err(fit_error)?;
        Ok(FittedPSpline { fitted })
    }
}

impl Default for PSplineRegressor {
    fn default() -> Self {
        Self::new()
    }
}
