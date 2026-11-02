//! Small numerical helpers of the core crate: LOWESS, R's `pretty`, the
//! negative-binomial θ estimators and the ridge Gram condition number.

use anofox_regression::core::{estimate_theta_ml, estimate_theta_moments};
use anofox_regression::solvers::gram_condition_number;
use anofox_regression::solvers::lowess::{lowess as rust_lowess, LowessOptions};
use anofox_regression::utils::pretty as rust_pretty;
use serde::Deserialize;
use wasm_bindgen::prelude::*;

use crate::convert::{object, ToJs};
use crate::reduction::parse;
use crate::regressors::{fit_error, to_mat};

#[derive(Deserialize, Default)]
struct LowessOpts {
    frac: Option<f64>,
    iterations: Option<usize>,
    delta: Option<f64>,
}

/// LOWESS smoother of `y` on `x` (`stats::lowess`): options `{ frac?
/// (2/3), iterations? (3), delta? }`. Returns `{ x, fitted, order }` with
/// `x` sorted and `order` the permutation that sorts the input.
#[wasm_bindgen]
pub fn lowess(x: &[f64], y: &[f64], options: JsValue) -> Result<JsValue, JsError> {
    let o: LowessOpts = parse(options)?;
    let mut opts = LowessOptions::default();
    if let Some(f) = o.frac {
        opts = opts.frac(f);
    }
    if let Some(i) = o.iterations {
        opts = opts.iterations(i);
    }
    if o.delta.is_some() {
        opts = opts.delta(o.delta);
    }
    let f = rust_lowess(x, y, &opts).map_err(fit_error)?;
    Ok(object(&[
        ("x", f.x.to_js()),
        ("fitted", f.fitted.to_js()),
        ("order", f.order.to_js()),
    ]))
}

/// R's `pretty(c(lo, hi), n)`: about `n + 1` round tick values covering
/// `[lo, hi]`.
#[wasm_bindgen]
pub fn pretty(lo: f64, hi: f64, n: usize) -> Vec<f64> {
    rust_pretty(lo, hi, n)
}

/// Maximum-likelihood θ of a negative binomial with means `mu`
/// (`MASS::theta.ml`).
#[wasm_bindgen(js_name = estimateThetaMl)]
pub fn estimate_theta_ml_js(
    y: &[f64],
    mu: &[f64],
    max_iter: Option<usize>,
    tol: Option<f64>,
) -> Result<f64, JsError> {
    if y.len() != mu.len() {
        return Err(JsError::new("y and mu must have the same length"));
    }
    Ok(estimate_theta_ml(
        y,
        mu,
        max_iter.unwrap_or(25),
        tol.unwrap_or(f64::EPSILON.powf(0.25)),
    ))
}

/// Moment estimate of the negative-binomial θ (`MASS::theta.mm`).
#[wasm_bindgen(js_name = estimateThetaMoments)]
pub fn estimate_theta_moments_js(y: &[f64], mu: &[f64]) -> Result<f64, JsError> {
    if y.len() != mu.len() {
        return Err(JsError::new("y and mu must have the same length"));
    }
    Ok(estimate_theta_moments(y, mu))
}

/// Condition number of the (centred, with `withIntercept`) Gram matrix
/// `XᵀX` of the row-major `x`.
#[wasm_bindgen(js_name = gramConditionNumber)]
pub fn gram_condition_number_js(
    x: &[f64],
    n_rows: usize,
    n_cols: usize,
    with_intercept: bool,
) -> Result<f64, JsError> {
    Ok(gram_condition_number(
        &to_mat(x, n_rows, n_cols)?,
        with_intercept,
    ))
}
