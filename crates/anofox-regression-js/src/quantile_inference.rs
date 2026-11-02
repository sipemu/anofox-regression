//! Quantile-regression inference (`solvers::quantile_inference`,
//! `quantreg::summary.rq` / `predict.rq`): iid, nid (Hendricks–Koenker
//! sandwich) and xy-bootstrap standard errors, the regression-quantile
//! process and confidence intervals for predicted quantiles.
//!
//! The standard-error type is "iid", "nid" or `{ type: "boot", seed?, r? }`
//! (defaults: seed 1, r 200).

use anofox_regression::solvers::quantile_inference::{
    hall_sheather_bandwidth, quantile_inference as rust_quantile_inference, quantile_process,
    QuantileInference as RustQuantileInference, QuantileSe,
};
use wasm_bindgen::prelude::*;

use crate::convert::{object, ToJs};
use crate::regressors::{fit_error, to_col, to_mat};

impl ToJs for QuantileSe {
    fn to_js(&self) -> JsValue {
        match *self {
            QuantileSe::Iid => JsValue::from_str("iid"),
            QuantileSe::Nid => JsValue::from_str("nid"),
            QuantileSe::Boot { seed, r } => object(&[
                ("type", "boot".to_js()),
                ("seed", seed.to_js()),
                ("r", r.to_js()),
            ]),
            _ => JsValue::UNDEFINED,
        }
    }
}

impl ToJs for RustQuantileInference {
    fn to_js(&self) -> JsValue {
        object(&[
            ("tau", self.tau.to_js()),
            ("se_type", self.se_type.to_js()),
            ("has_intercept", self.has_intercept.to_js()),
            ("estimate", self.estimate.to_js()),
            ("std_error", self.std_error.to_js()),
            ("statistic", self.statistic.to_js()),
            ("p_value", self.p_value.to_js()),
            ("conf_low", self.conf_low.to_js()),
            ("conf_high", self.conf_high.to_js()),
            ("conf_level", self.conf_level.to_js()),
            ("df_residual", self.df_residual.to_js()),
            ("bandwidth", self.bandwidth.to_js()),
            ("sparsity", self.sparsity.to_js()),
            ("covariance", self.covariance.to_js()),
            ("n_boot", self.n_boot.to_js()),
        ])
    }
}

/// "iid", "nid" or `{ type: "boot", seed?, r? }`.
pub(crate) fn parse_se(v: JsValue) -> Result<QuantileSe, JsError> {
    if let Some(s) = v.as_string() {
        return match s.as_str() {
            "iid" => Ok(QuantileSe::Iid),
            "nid" => Ok(QuantileSe::Nid),
            "boot" => Ok(QuantileSe::Boot { seed: 1, r: 200 }),
            other => Err(JsError::new(&format!(
                "Unknown se '{}': expected iid, nid or boot",
                other
            ))),
        };
    }
    let get = |k: &str| js_sys::Reflect::get(&v, &JsValue::from_str(k)).ok();
    match get("type").and_then(|t| t.as_string()).as_deref() {
        Some("boot") => {
            let num = |k: &str| get(k).and_then(|x| x.as_f64());
            let seed = num("seed").unwrap_or(1.0);
            if !(seed.is_finite() && seed >= 0.0 && seed.fract() == 0.0) {
                return Err(JsError::new("seed must be a non-negative integer"));
            }
            Ok(QuantileSe::Boot {
                seed: seed as u64,
                r: num("r").map_or(200, |r| r as usize),
            })
        }
        _ => Err(JsError::new(
            "se: expected \"iid\", \"nid\" or { type: \"boot\", seed?, r? }",
        )),
    }
}

/// Quantile-regression inference at one τ.
#[wasm_bindgen]
pub struct QuantileInference {
    pub(crate) inner: RustQuantileInference,
}

#[wasm_bindgen]
impl QuantileInference {
    /// `{ tau, seType, hasIntercept, estimate, stdError, statistic, pValue,
    /// confLow, confHigh, confLevel, dfResidual, bandwidth, sparsity,
    /// covariance, nBoot }` (intercept first when fitted).
    #[wasm_bindgen(js_name = getResult)]
    pub fn get_result(&self) -> JsValue {
        self.inner.to_js()
    }

    /// Confidence intervals of the predicted quantile at new rows
    /// (`predict.rq(interval = "confidence")`): `{ fit, lower, upper, se }`.
    #[wasm_bindgen(js_name = predictInterval)]
    pub fn predict_interval(
        &self,
        x: &[f64],
        n_rows: usize,
        level: f64,
    ) -> Result<JsValue, JsError> {
        let p = self.inner.estimate.len() - usize::from(self.inner.has_intercept);
        Ok(self
            .inner
            .predict_interval(&to_mat(x, n_rows, p)?, level)
            .map_err(fit_error)?
            .to_js())
    }
}

/// Fit the τ-quantile regression of `y` on `x` and its inference (`se` as
/// in the module docs, `level` the confidence level).
#[allow(clippy::too_many_arguments)]
#[wasm_bindgen(js_name = quantileInference)]
pub fn quantile_inference_js(
    x: &[f64],
    n_rows: usize,
    n_cols: usize,
    y: &[f64],
    tau: f64,
    with_intercept: bool,
    se: JsValue,
    level: f64,
) -> Result<QuantileInference, JsError> {
    Ok(QuantileInference {
        inner: rust_quantile_inference(
            &to_mat(x, n_rows, n_cols)?,
            &to_col(y, n_rows, "y")?,
            tau,
            with_intercept,
            parse_se(se)?,
            level,
        )
        .map_err(fit_error)?,
    })
}

/// The regression-quantile process: one inference per τ (standard errors
/// `NaN` when `se` is null or undefined).
#[allow(clippy::too_many_arguments)]
#[wasm_bindgen(js_name = quantileProcess)]
pub fn quantile_process_js(
    x: &[f64],
    n_rows: usize,
    n_cols: usize,
    y: &[f64],
    taus: &[f64],
    with_intercept: bool,
    se: JsValue,
    level: f64,
) -> Result<JsValue, JsError> {
    let se = if se.is_undefined() || se.is_null() {
        None
    } else {
        Some(parse_se(se)?)
    };
    let out = quantile_process(
        &to_mat(x, n_rows, n_cols)?,
        &to_col(y, n_rows, "y")?,
        taus,
        with_intercept,
        se,
        level,
    )
    .map_err(fit_error)?;
    Ok(out.to_js())
}

/// Hall–Sheather bandwidth for quantile `tau` and `n` observations.
#[wasm_bindgen(js_name = hallSheatherBandwidth)]
pub fn hall_sheather_bandwidth_js(tau: f64, n: usize) -> f64 {
    hall_sheather_bandwidth(tau, n)
}
