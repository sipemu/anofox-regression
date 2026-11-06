//! Effects, contrasts and predictions on a model specification
//! (`anofox_regression::effects`): `datadist` limits, `summary.rms`,
//! `contrast.rms`, `Predict` (pointwise or simultaneous bands) and
//! `nomogram`.
//!
//! `beta` and `vcov` are the full coefficient vector and its covariance
//! (row-major `p × p`) in the design's column order, intercept first when
//! `intercept` is set. A *setting* is an object `{ name: value }` keyed by
//! predictor name, or an array of `[predictorIndex, value]` pairs;
//! predictors not given are held at their adjust-to values. Options are
//! `{ df?: number, level?: number }`: with `df` the critical values come
//! from Student's t with `df` residual degrees of freedom (OLS), otherwise
//! from the normal (GLMs, Cox, AFT); `level` defaults to 0.95.

use anofox_regression::effects::{
    simultaneous_critical, ContrastEstimate, Critical, EffectRow, Limits,
    ModelSpec as RustModelSpec, Nomogram as RustNomogram, NomogramAxis, Predictor, PredictorKind,
};
use faer::{Col, Mat};
use js_sys::{Array, Function};
use serde::Deserialize;
use wasm_bindgen::prelude::*;

use crate::convert::{impl_to_js, object, ToJs};
use crate::design::Design;
use crate::inference::square;
use crate::regressors::fit_error;
use crate::splines::{Lsp, Rcs};

impl_to_js!(Limits {
    low_effect,
    adjust_to,
    high_effect,
    low_prediction,
    high_prediction,
    low,
    high
});
impl_to_js!(ContrastEstimate {
    estimate,
    se,
    lower,
    upper,
    statistic,
    p_value
});
impl_to_js!(EffectRow {
    label,
    predictor,
    low,
    high,
    effect
});
impl_to_js!(NomogramAxis {
    name,
    values,
    xbeta,
    points
});
impl_to_js!(RustNomogram {
    axes,
    scale,
    intercept,
    total_points,
    lp_at,
    lp_points
});

impl ToJs for Predictor {
    fn to_js(&self) -> JsValue {
        let (kind, knots, levels) = match &self.kind {
            PredictorKind::Linear => ("linear", None, None),
            PredictorKind::Rcs(r) => ("rcs", Some(r.knots().to_vec()), None),
            PredictorKind::Lsp(l) => ("lsp", Some(l.knots().to_vec()), None),
            PredictorKind::Categorical(levels) => ("categorical", None, Some(levels.clone())),
            _ => ("other", None, None),
        };
        object(&[
            ("name", self.name.to_js()),
            ("kind", kind.to_js()),
            ("knots", knots.to_js()),
            ("levels", levels.to_js()),
            ("limits", self.limits.to_js()),
            ("distinct", self.distinct.to_js()),
        ])
    }
}

// ------------------------------------------------------------ parsing ----

#[derive(Deserialize, Default)]
struct CritOpts {
    df: Option<f64>,
    level: Option<f64>,
}

/// `{ df?, level? }` → (critical distribution, level).
fn parse_crit(options: JsValue) -> Result<(Critical, f64), JsError> {
    let o: CritOpts = if options.is_undefined() || options.is_null() {
        CritOpts::default()
    } else {
        serde_wasm_bindgen::from_value(options).map_err(|e| JsError::new(&e.to_string()))?
    };
    let crit = o.df.map_or(Critical::Normal, Critical::StudentT);
    Ok((crit, o.level.unwrap_or(0.95)))
}

fn col(v: &[f64]) -> Col<f64> {
    Col::from_fn(v.len(), |i| v[i])
}

fn coefs(beta: &[f64], vcov: &[f64]) -> Result<(Col<f64>, Mat<f64>), JsError> {
    Ok((col(beta), square(vcov, beta.len(), "vcov")?))
}

/// Number arrays from a JS array of arrays.
fn rows_of(v: JsValue, what: &str) -> Result<Vec<Vec<f64>>, JsError> {
    serde_wasm_bindgen::from_value(v).map_err(|e| {
        JsError::new(&format!(
            "{}: expected an array of number arrays ({})",
            what, e
        ))
    })
}

/// Stored `datadist` limits as a JS object.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct LimitsJs {
    low_effect: Option<f64>,
    adjust_to: f64,
    high_effect: Option<f64>,
    low_prediction: f64,
    high_prediction: f64,
    low: f64,
    high: f64,
}

// ---------------------------------------------------------- ModelSpec ----

/// Model specification with `datadist`-style limits: predictors (linear,
/// restricted cubic spline, linear spline or categorical) and interactions.
#[wasm_bindgen]
#[derive(Clone, Default)]
pub struct ModelSpec {
    inner: RustModelSpec,
}

impl ModelSpec {
    fn setting(&self, v: JsValue) -> Result<Vec<(usize, f64)>, JsError> {
        if v.is_undefined() || v.is_null() {
            return Ok(Vec::new());
        }
        if Array::is_array(&v) {
            let pairs: Vec<(usize, f64)> = serde_wasm_bindgen::from_value(v).map_err(|e| {
                JsError::new(&format!(
                    "setting: expected [predictorIndex, value] pairs ({})",
                    e
                ))
            })?;
            let n = self.inner.predictors().len();
            if let Some((k, _)) = pairs.iter().find(|(k, _)| *k >= n) {
                return Err(JsError::new(&format!("No predictor {}", k)));
            }
            return Ok(pairs);
        }
        let map: std::collections::BTreeMap<String, f64> = serde_wasm_bindgen::from_value(v)
            .map_err(|e| JsError::new(&format!("setting: expected {{ name: value }} ({})", e)))?;
        map.into_iter()
            .map(|(name, value)| {
                self.index(&name)
                    .map(|k| (k, value))
                    .ok_or_else(|| JsError::new(&format!("No predictor named '{}'", name)))
            })
            .collect()
    }

    fn index(&self, name: &str) -> Option<usize> {
        self.inner.predictors().iter().position(|p| p.name == name)
    }

    fn check(&self, a: usize, b: usize) -> Result<(), JsError> {
        let n = self.inner.predictors().len();
        match [a, b].into_iter().find(|&k| k >= n) {
            Some(k) => Err(JsError::new(&format!("No predictor {}", k))),
            None => Ok(()),
        }
    }

    fn check_beta(&self, beta: &[f64], intercept: bool) -> Result<(), JsError> {
        let want = self.inner.row(&[], intercept).map_err(fit_error)?.len();
        if beta.len() != want {
            return Err(JsError::new(&format!(
                "Expected {} coefficients, got {}",
                want,
                beta.len()
            )));
        }
        Ok(())
    }
}

#[wasm_bindgen]
impl ModelSpec {
    /// An empty specification.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self::default()
    }

    /// Linear predictor with limits from its data `x`; returns its index.
    #[wasm_bindgen]
    pub fn linear(&mut self, name: &str, x: &[f64]) -> usize {
        self.inner.linear(name, x)
    }

    /// Restricted-cubic-spline predictor; returns its index.
    #[wasm_bindgen]
    pub fn rcs(&mut self, name: &str, rcs: &Rcs, x: &[f64]) -> usize {
        self.inner.rcs(name, rcs.inner.clone(), x)
    }

    /// Linear-spline predictor; returns its index.
    #[wasm_bindgen]
    pub fn lsp(&mut self, name: &str, lsp: &Lsp, x: &[f64]) -> usize {
        self.inner.lsp(name, lsp.inner.clone(), x)
    }

    /// Categorical predictor (treatment contrasts, levels from `x`,
    /// adjusted to the most frequent level); returns its index.
    #[wasm_bindgen]
    pub fn categorical(&mut self, name: &str, x: &[f64]) -> usize {
        self.inner.categorical(name, x)
    }

    /// Categorical predictor with the given levels (the first is the
    /// reference and the adjust-to level); returns its index.
    #[wasm_bindgen(js_name = categoricalWithLevels)]
    pub fn categorical_with_levels(
        &mut self,
        name: &str,
        levels: &[f64],
    ) -> Result<usize, JsError> {
        self.inner
            .categorical_with_levels(name, levels)
            .map_err(fit_error)
    }

    /// Add a predictor with stored limits, e.g. from `predictors()` of an
    /// earlier specification: `kind` is "linear", "rcs", "lsp" or
    /// "categorical", `params` the knots (rcs / lsp) or levels
    /// (categorical), `limits` `{ lowEffect, adjustTo, highEffect,
    /// lowPrediction, highPrediction, low, high }` (null effect values for a
    /// categorical predictor are NaN) and `distinct` the nomogram ticks.
    /// Returns its index.
    #[wasm_bindgen(js_name = addPredictor)]
    pub fn add_predictor(
        &mut self,
        name: &str,
        kind: &str,
        params: Option<Vec<f64>>,
        limits: JsValue,
        distinct: Option<Vec<f64>>,
    ) -> Result<usize, JsError> {
        let p = params.unwrap_or_default();
        let spline_err = |e: anofox_regression::splines::SplineError| JsError::new(&e.to_string());
        let kind = match kind {
            "linear" => PredictorKind::Linear,
            "rcs" => {
                PredictorKind::Rcs(anofox_regression::splines::Rcs::new(&p).map_err(spline_err)?)
            }
            "lsp" => {
                PredictorKind::Lsp(anofox_regression::splines::Lsp::new(&p).map_err(spline_err)?)
            }
            "categorical" => PredictorKind::Categorical(p),
            other => {
                return Err(JsError::new(&format!(
                    "Unknown kind '{other}': expected linear, rcs, lsp or categorical"
                )))
            }
        };
        let l: LimitsJs = serde_wasm_bindgen::from_value(limits)
            .map_err(|e| JsError::new(&format!("limits: {e}")))?;
        let nan = |v: Option<f64>| v.unwrap_or(f64::NAN);
        let limits = Limits::new(
            nan(l.low_effect),
            l.adjust_to,
            nan(l.high_effect),
            l.low_prediction,
            l.high_prediction,
            l.low,
            l.high,
        );
        self.inner
            .add_predictor(name, kind, limits, distinct)
            .map_err(fit_error)
    }

    /// Full tensor-product interaction `a * b`.
    #[wasm_bindgen]
    pub fn interaction(&mut self, a: usize, b: usize) -> Result<(), JsError> {
        self.check(a, b)?;
        self.inner.interaction(a, b);
        Ok(())
    }

    /// Restricted interaction `a %ia% b`.
    #[wasm_bindgen(js_name = restrictedInteraction)]
    pub fn restricted_interaction(&mut self, a: usize, b: usize) -> Result<(), JsError> {
        self.check(a, b)?;
        self.inner.restricted_interaction(a, b);
        Ok(())
    }

    /// The predictors: `[{ name, kind: "linear" | "rcs" | "lsp" |
    /// "categorical", knots?, levels?, limits, distinct? }]` with `limits`
    /// `{ lowEffect, adjustTo, highEffect, lowPrediction, highPrediction,
    /// low, high }`.
    #[wasm_bindgen]
    pub fn predictors(&self) -> JsValue {
        self.inner.predictors().to_js()
    }

    /// Design (no intercept column) of the data, one array per predictor:
    /// main effects in predictor order, then the interactions.
    #[wasm_bindgen]
    pub fn design(&self, columns: JsValue) -> Result<Design, JsError> {
        let cols = rows_of(columns, "columns")?;
        if cols.len() != self.inner.predictors().len() {
            return Err(JsError::new(&format!(
                "Expected {} predictor columns, got {}",
                self.inner.predictors().len(),
                cols.len()
            )));
        }
        let refs: Vec<&[f64]> = cols.iter().map(Vec::as_slice).collect();
        Ok(Design {
            inner: self.inner.design(&refs).map_err(fit_error)?,
        })
    }

    /// Design row for a setting, with a leading 1 when `intercept`.
    #[wasm_bindgen]
    pub fn row(&self, setting: JsValue, intercept: bool) -> Result<Vec<f64>, JsError> {
        self.inner
            .row(&self.setting(setting)?, intercept)
            .map_err(fit_error)
    }

    /// Contrast `row(a) − row(b)` (`contrast.rms(fit, a, b)`): `{ estimate,
    /// se, lower, upper, statistic, pValue }`.
    #[allow(clippy::too_many_arguments)]
    #[wasm_bindgen]
    pub fn contrast(
        &self,
        beta: &[f64],
        vcov: &[f64],
        intercept: bool,
        a: JsValue,
        b: JsValue,
        options: JsValue,
    ) -> Result<JsValue, JsError> {
        self.check_beta(beta, intercept)?;
        let (beta, vcov) = coefs(beta, vcov)?;
        let (crit, level) = parse_crit(options)?;
        Ok(self
            .inner
            .contrast(
                &beta,
                &vcov,
                intercept,
                &self.setting(a)?,
                &self.setting(b)?,
                crit,
                level,
            )
            .map_err(fit_error)?
            .to_js())
    }

    /// Double-difference contrast `(row(a) − row(b)) − (row(a2) −
    /// row(b2))`, e.g. an interaction effect.
    #[allow(clippy::too_many_arguments)]
    #[wasm_bindgen(js_name = doubleContrast)]
    pub fn double_contrast(
        &self,
        beta: &[f64],
        vcov: &[f64],
        intercept: bool,
        a: JsValue,
        b: JsValue,
        a2: JsValue,
        b2: JsValue,
        options: JsValue,
    ) -> Result<JsValue, JsError> {
        self.check_beta(beta, intercept)?;
        let (beta, vcov) = coefs(beta, vcov)?;
        let (crit, level) = parse_crit(options)?;
        Ok(self
            .inner
            .double_contrast(
                &beta,
                &vcov,
                intercept,
                &self.setting(a)?,
                &self.setting(b)?,
                &self.setting(a2)?,
                &self.setting(b2)?,
                crit,
                level,
            )
            .map_err(fit_error)?
            .to_js())
    }

    /// `summary.rms`: the low → high effect of every numeric predictor and
    /// each level against the adjust-to level of a categorical one:
    /// `[{ label, predictor, low, high, effect }]`. `Math.exp` of an effect
    /// and its limits gives the odds / hazard / time ratio.
    #[wasm_bindgen]
    pub fn summary(
        &self,
        beta: &[f64],
        vcov: &[f64],
        intercept: bool,
        options: JsValue,
    ) -> Result<JsValue, JsError> {
        self.check_beta(beta, intercept)?;
        let (beta, vcov) = coefs(beta, vcov)?;
        let (crit, level) = parse_crit(options)?;
        Ok(self
            .inner
            .summary(&beta, &vcov, intercept, crit, level)
            .to_js())
    }

    /// `Predict(fit, x)`: the linear predictor over `values` of predictor
    /// `k` (default: 200 points between its prediction limits), others at
    /// their adjust-to values, with pointwise limits: `{ values, estimates }`.
    #[allow(clippy::too_many_arguments)]
    #[wasm_bindgen(js_name = predictGrid)]
    pub fn predict_grid(
        &self,
        beta: &[f64],
        vcov: &[f64],
        intercept: bool,
        k: usize,
        values: Option<Vec<f64>>,
        options: JsValue,
    ) -> Result<JsValue, JsError> {
        self.check_beta(beta, intercept)?;
        self.check(k, k)?;
        let (beta, vcov) = coefs(beta, vcov)?;
        let (crit, level) = parse_crit(options)?;
        let (v, est) = self
            .inner
            .predict_grid(&beta, &vcov, intercept, k, values, crit, level)
            .map_err(fit_error)?;
        Ok(object(&[("values", v.to_js()), ("estimates", est.to_js())]))
    }

    /// `predictGrid` with simultaneous confidence bands over the grid
    /// (`Predict(conf.type = "simultaneous")`): `{ values, estimates,
    /// critical }`.
    #[allow(clippy::too_many_arguments)]
    #[wasm_bindgen(js_name = predictGridSimultaneous)]
    pub fn predict_grid_simultaneous(
        &self,
        beta: &[f64],
        vcov: &[f64],
        intercept: bool,
        k: usize,
        values: Option<Vec<f64>>,
        options: JsValue,
    ) -> Result<JsValue, JsError> {
        self.check_beta(beta, intercept)?;
        self.check(k, k)?;
        let (beta, vcov) = coefs(beta, vcov)?;
        let (crit, level) = parse_crit(options)?;
        let (v, est, c) = self
            .inner
            .predict_grid_simultaneous(&beta, &vcov, intercept, k, values, crit, level)
            .map_err(fit_error)?;
        Ok(object(&[
            ("values", v.to_js()),
            ("estimates", est.to_js()),
            ("critical", c.to_js()),
        ]))
    }

    /// `rms::nomogram` for a main-effects model: `linearPredictors` are the
    /// fit's (intercept included); `values` optionally gives the ticks of
    /// each predictor (an array with `null` for the defaults); `maxscale`
    /// defaults to 100 and `nint` to 10.
    #[wasm_bindgen]
    pub fn nomogram(
        &self,
        beta: &[f64],
        intercept: bool,
        linear_predictors: &[f64],
        values: JsValue,
        maxscale: Option<f64>,
        nint: Option<usize>,
    ) -> Result<Nomogram, JsError> {
        let values: Vec<Option<Vec<f64>>> = if values.is_undefined() || values.is_null() {
            Vec::new()
        } else {
            serde_wasm_bindgen::from_value(values).map_err(|e| JsError::new(&e.to_string()))?
        };
        let inner = self
            .inner
            .nomogram(
                &col(beta),
                intercept,
                linear_predictors,
                &values,
                maxscale.unwrap_or(100.0),
                nint.unwrap_or(10),
            )
            .map_err(fit_error)?;
        Ok(Nomogram { inner })
    }
}

/// A nomogram: per-predictor point axes, the total-points axis and the
/// linear-predictor axis.
#[wasm_bindgen]
pub struct Nomogram {
    inner: RustNomogram,
}

#[wasm_bindgen]
impl Nomogram {
    /// `{ axes: [{ name, values, xbeta, points }], scale, intercept,
    /// totalPoints, lpAt, lpPoints }`.
    #[wasm_bindgen(js_name = getResult)]
    pub fn get_result(&self) -> JsValue {
        self.inner.to_js()
    }

    /// Linear predictor of a total score.
    #[wasm_bindgen(js_name = pointsToLp)]
    pub fn points_to_lp(&self, total: f64) -> f64 {
        self.inner.points_to_lp(total)
    }

    /// Total score of a linear predictor.
    #[wasm_bindgen(js_name = lpToPoints)]
    pub fn lp_to_points(&self, lp: f64) -> f64 {
        self.inner.lp_to_points(lp)
    }

    /// Ticks of a function axis, e.g. a risk axis with
    /// `fun = (lp) => 1 / (1 + Math.exp(-lp))`: `{ values, points }`.
    /// `nint` defaults to 10.
    #[wasm_bindgen(js_name = functionAxis)]
    pub fn function_axis(&self, fun: &Function, nint: Option<usize>) -> Result<JsValue, JsError> {
        // A throwing or non-numeric `fun` is reported after the call; the
        // placeholder keeps the Rust side finite meanwhile.
        let err = std::cell::RefCell::new(None::<String>);
        let fail = |msg: String| {
            err.borrow_mut().get_or_insert(msg);
            0.0
        };
        let call = |lp: f64| match fun.call1(&JsValue::NULL, &JsValue::from_f64(lp)) {
            Ok(v) => v
                .as_f64()
                .unwrap_or_else(|| fail("fun must return a number".into())),
            Err(e) => fail(format!(
                "fun threw: {}",
                String::from(js_sys::Error::from(e).message())
            )),
        };
        let lp = &self.inner.lp_at;
        if let (Some(&lo), Some(&hi)) = (lp.first(), lp.last()) {
            if !(call(lo).is_finite() && call(hi).is_finite()) {
                fail("fun must be finite over the lp axis".into());
            }
        }
        if let Some(e) = err.borrow_mut().take() {
            return Err(JsError::new(&e));
        }
        let (values, points) = self.inner.function_axis(call, nint.unwrap_or(10));
        if let Some(e) = err.into_inner() {
            return Err(JsError::new(&e));
        }
        Ok(object(&[
            ("values", values.to_js()),
            ("points", points.to_js()),
        ]))
    }
}

/// `datadist` limits of a numeric predictor: `{ lowEffect, adjustTo,
/// highEffect, lowPrediction, highPrediction, low, high }`.
#[wasm_bindgen(js_name = numericLimits)]
pub fn numeric_limits(x: &[f64]) -> Result<JsValue, JsError> {
    if !x.iter().any(|v| v.is_finite()) {
        return Err(JsError::new("x has no finite values"));
    }
    Ok(Limits::numeric(x).to_js())
}

/// Critical value of simultaneous (max-|t|) intervals for the linear
/// combinations `rows · β` (array of coefficient-length arrays) with
/// covariance `vcov` (row-major); options `{ df?, level? }`. NaN for bad
/// input.
#[wasm_bindgen(js_name = simultaneousCritical)]
pub fn simultaneous_critical_js(
    rows: JsValue,
    vcov: &[f64],
    options: JsValue,
) -> Result<f64, JsError> {
    let rows = rows_of(rows, "rows")?;
    let p = rows.first().map_or(0, Vec::len);
    let (crit, level) = parse_crit(options)?;
    Ok(simultaneous_critical(
        &rows,
        &square(vcov, p, "vcov")?,
        crit,
        level,
    ))
}
