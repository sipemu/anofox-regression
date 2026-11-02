//! Survival analysis (`anofox_regression::survival`): Kaplan–Meier /
//! Nelson–Aalen curves (`survfit`), the G-ρ tests (`survdiff`) and the Cox
//! model (`coxph`, `cox.zph`, `survfit.coxph`).
//!
//! Event indicators are `Uint8Array`s (1 = event, 0 = censored); stratum,
//! group and cluster codes are non-negative integers. Matrices in results
//! are arrays of rows.

use anofox_regression::survival::cox::{
    coxph as rust_coxph, CoxFit as RustCoxFit, CoxInput, CoxOptions, CoxSurvCurve, CoxZph, Ties,
    ZphTest, ZphTransform,
};
use anofox_regression::survival::{
    kaplan_meier as rust_kaplan_meier, survdiff as rust_survdiff, survfit as rust_survfit,
    ConfType, HazardType, SurvCurve as RustSurvCurve, SurvDiff, SurvInput, SurvQuantile,
    SurvSummary, SurvfitOptions,
};
use serde::Deserialize;
use wasm_bindgen::prelude::*;

use crate::convert::{impl_to_js, object, ToJs};
use crate::regressors::{fit_error, to_mat};

impl_to_js!(RustSurvCurve {
    stratum,
    records,
    time,
    n_risk,
    n_event,
    n_censor,
    surv,
    std_err,
    lower,
    upper,
    cumhaz,
    std_chaz
});
impl_to_js!(SurvQuantile {
    estimate,
    lower,
    upper
});
impl_to_js!(SurvSummary {
    records,
    n_max,
    n_start,
    events,
    rmean,
    se_rmean,
    median,
    median_lower,
    median_upper
});
impl_to_js!(SurvDiff {
    groups,
    n,
    observed,
    expected,
    variance,
    chisq,
    df,
    p_value
});
impl_to_js!(ZphTest { chisq, df, p_value });
impl_to_js!(CoxZph {
    terms,
    global,
    x,
    time,
    y
});
impl_to_js!(CoxSurvCurve {
    time,
    n_risk,
    cumhaz,
    surv,
    std_err,
    lower,
    upper
});

// ------------------------------------------------------------ parsing ----

fn parse<T: for<'de> Deserialize<'de> + Default>(v: JsValue) -> Result<T, JsError> {
    if v.is_undefined() || v.is_null() {
        return Ok(T::default());
    }
    serde_wasm_bindgen::from_value(v).map_err(|e| JsError::new(&e.to_string()))
}

fn events(event: &[u8], n: usize) -> Result<Vec<bool>, JsError> {
    if event.len() != n {
        return Err(JsError::new(&format!(
            "Expected {} event values, got {}",
            n,
            event.len()
        )));
    }
    Ok(event.iter().map(|&e| e != 0).collect())
}

fn usizes(v: &[u32]) -> Vec<usize> {
    v.iter().map(|&c| c as usize).collect()
}

/// `survfit` options: `{ confLevel?, confType?: "log" | "log-log" |
/// "plain", hazard?: "nelson-aalen" | "fleming-harrington", timefix?,
/// robust? }`.
#[derive(Deserialize, Default)]
#[serde(rename_all = "camelCase")]
struct SurvfitOpts {
    conf_level: Option<f64>,
    conf_type: Option<String>,
    hazard: Option<String>,
    timefix: Option<bool>,
    robust: Option<bool>,
}

fn survfit_options(options: JsValue) -> Result<SurvfitOptions, JsError> {
    let o: SurvfitOpts = parse(options)?;
    let mut out = SurvfitOptions::default();
    if let Some(level) = o.conf_level {
        out = out.conf_level(level);
    }
    if let Some(kind) = o.conf_type {
        out = out.conf_type(match kind.to_ascii_lowercase().as_str() {
            "log" => ConfType::Log,
            "log-log" | "loglog" => ConfType::LogLog,
            "plain" => ConfType::Plain,
            other => {
                return Err(JsError::new(&format!(
                    "Unknown confType '{}': expected log, log-log or plain",
                    other
                )))
            }
        });
    }
    if let Some(h) = o.hazard {
        out = out.hazard(match h.to_ascii_lowercase().as_str() {
            "nelson-aalen" => HazardType::NelsonAalen,
            "fleming-harrington" => HazardType::FlemingHarrington,
            other => {
                return Err(JsError::new(&format!(
                    "Unknown hazard '{}': expected nelson-aalen or fleming-harrington",
                    other
                )))
            }
        });
    }
    if let Some(t) = o.timefix {
        out = out.timefix(t);
    }
    if o.robust.is_some() {
        out = out.robust(o.robust);
    }
    Ok(out)
}

/// Optional per-row data of `survfit`.
#[derive(Deserialize, Default)]
struct SurvfitData {
    entry: Option<Vec<f64>>,
    weights: Option<Vec<f64>>,
    cluster: Option<Vec<usize>>,
    strata: Option<Vec<usize>>,
}

/// Optional per-row data of `coxph`.
#[derive(Deserialize, Default)]
struct CoxData {
    entry: Option<Vec<f64>>,
    weights: Option<Vec<f64>>,
    offset: Option<Vec<f64>>,
    strata: Option<Vec<usize>>,
}

/// `coxph` options: `{ ties?: "efron" | "breslow", maxIter?, eps?, ridge?,
/// confLevel?, timefix? }`.
#[derive(Deserialize, Default)]
#[serde(rename_all = "camelCase")]
struct CoxOpts {
    ties: Option<String>,
    max_iter: Option<usize>,
    eps: Option<f64>,
    ridge: Option<f64>,
    conf_level: Option<f64>,
    timefix: Option<bool>,
}

fn cox_options(options: JsValue) -> Result<CoxOptions, JsError> {
    let o: CoxOpts = parse(options)?;
    let mut out = CoxOptions::default();
    if let Some(t) = o.ties {
        out = out.ties(match t.to_ascii_lowercase().as_str() {
            "efron" => Ties::Efron,
            "breslow" => Ties::Breslow,
            other => {
                return Err(JsError::new(&format!(
                    "Unknown ties '{}': expected efron or breslow",
                    other
                )))
            }
        });
    }
    if let Some(v) = o.max_iter {
        out = out.max_iter(v);
    }
    if let Some(v) = o.eps {
        out = out.eps(v);
    }
    if let Some(v) = o.ridge {
        out = out.ridge(v);
    }
    if let Some(v) = o.conf_level {
        out = out.conf_level(v);
    }
    if let Some(v) = o.timefix {
        out = out.timefix(v);
    }
    Ok(out)
}

fn parse_zph_transform(name: Option<String>) -> Result<ZphTransform, JsError> {
    match name.as_deref().map(str::to_ascii_lowercase).as_deref() {
        None | Some("km") => Ok(ZphTransform::Km),
        Some("rank") => Ok(ZphTransform::Rank),
        Some("identity") => Ok(ZphTransform::Identity),
        Some("log") => Ok(ZphTransform::Log),
        Some(other) => Err(JsError::new(&format!(
            "Unknown transform '{}': expected km, rank, identity or log",
            other
        ))),
    }
}

// ---------------------------------------------------- survival curves ----

/// One Kaplan–Meier / Nelson–Aalen curve (one stratum).
#[wasm_bindgen]
pub struct SurvCurve {
    inner: RustSurvCurve,
}

#[wasm_bindgen]
impl SurvCurve {
    /// The curve: `{ stratum?, records, time, nRisk, nEvent, nCensor, surv,
    /// stdErr, lower, upper, cumhaz, stdChaz }` (one entry per distinct
    /// time; `stdErr` is the standard error of the cumulative hazard, as
    /// `survfit`'s `std.err`).
    #[wasm_bindgen(js_name = getResult)]
    pub fn get_result(&self) -> JsValue {
        self.inner.to_js()
    }

    /// Stratum code (undefined without strata).
    #[wasm_bindgen(js_name = getStratum)]
    pub fn get_stratum(&self) -> Option<usize> {
        self.inner.stratum
    }

    /// Median survival with its confidence limits: `{ estimate, lower,
    /// upper }` (NaN when not reached).
    #[wasm_bindgen]
    pub fn median(&self) -> JsValue {
        self.inner.median().to_js()
    }

    /// Restricted mean survival up to `endTime` and its standard error,
    /// `[rmean, se]`.
    #[wasm_bindgen(js_name = restrictedMean)]
    pub fn restricted_mean(&self, end_time: f64) -> Vec<f64> {
        let (m, se) = self.inner.restricted_mean(end_time);
        vec![m, se]
    }

    /// `summary(survfit)$table` with the restricted mean up to `endTime`:
    /// `{ records, nMax, nStart, events, rmean, seRmean, median,
    /// medianLower, medianUpper }`.
    #[wasm_bindgen]
    pub fn summary(&self, end_time: f64) -> JsValue {
        self.inner.summary(end_time).to_js()
    }

    /// Survival probability `S(t)` (right-continuous step function).
    #[wasm_bindgen(js_name = survivalAt)]
    pub fn survival_at(&self, t: f64) -> f64 {
        self.inner.survival_at(t)
    }
}

/// Kaplan–Meier curve of right-censored data without strata or weights.
/// `options` as in `survfit`.
#[wasm_bindgen(js_name = kaplanMeier)]
pub fn kaplan_meier_js(time: &[f64], event: &[u8], options: JsValue) -> Result<SurvCurve, JsError> {
    let event = events(event, time.len())?;
    let inner = rust_kaplan_meier(time, &event, &survfit_options(options)?).map_err(fit_error)?;
    Ok(SurvCurve { inner })
}

/// Kaplan–Meier / Nelson–Aalen curves per stratum (codes in increasing
/// order), as `survfit(Surv(...) ~ strata)`. `data` holds optional per-row
/// arrays `{ entry?, weights?, cluster?, strata? }` (entry times for
/// left-truncated data, case weights, cluster ids for the robust variance);
/// `options` is `{ confLevel?, confType?: "log" | "log-log" | "plain",
/// hazard?: "nelson-aalen" | "fleming-harrington", timefix?, robust? }`.
#[wasm_bindgen]
pub fn survfit(
    time: &[f64],
    event: &[u8],
    data: JsValue,
    options: JsValue,
) -> Result<Vec<SurvCurve>, JsError> {
    let event = events(event, time.len())?;
    let d: SurvfitData = parse(data)?;
    let mut input = SurvInput::new(time, &event);
    if let Some(e) = &d.entry {
        input = input.entry(e);
    }
    if let Some(w) = &d.weights {
        input = input.weights(w);
    }
    if let Some(c) = &d.cluster {
        input = input.cluster(c);
    }
    let curves =
        rust_survfit(&input, d.strata.as_deref(), &survfit_options(options)?).map_err(fit_error)?;
    Ok(curves
        .into_iter()
        .map(|inner| SurvCurve { inner })
        .collect())
}

/// k-sample G-ρ test (`survival::survdiff`): `rho = 0` (default) is the
/// log-rank test, `rho = 1` the Peto–Peto test; with `strata` the test is
/// stratified. Returns `{ groups, n, observed, expected, variance, chisq,
/// df, pValue }`.
#[wasm_bindgen]
pub fn survdiff(
    time: &[f64],
    event: &[u8],
    group: &[u32],
    strata: Option<Vec<u32>>,
    rho: Option<f64>,
) -> Result<JsValue, JsError> {
    let event = events(event, time.len())?;
    let strata = strata.map(|s| usizes(&s));
    Ok(rust_survdiff(
        time,
        &event,
        &usizes(group),
        strata.as_deref(),
        rho.unwrap_or(0.0),
    )
    .map_err(fit_error)?
    .to_js())
}

// ---------------------------------------------------------------- Cox ----

/// A fitted Cox proportional-hazards model.
#[wasm_bindgen]
pub struct CoxFit {
    fitted: RustCoxFit,
}

#[wasm_bindgen]
impl CoxFit {
    /// The fit: `{ coefficients, var, stdErrors, z, pValues, loglik:
    /// [null, final], lrTest, waldTest, scoreTest, df, n, nEvents,
    /// iterations, converged, concordance, dxy, r2 }`.
    #[wasm_bindgen(js_name = getResult)]
    pub fn get_result(&self) -> JsValue {
        let f = &self.fitted;
        object(&[
            ("coefficients", f.coefficients.to_js()),
            ("var", f.var.to_js()),
            ("std_errors", f.std_errors.to_js()),
            ("z", f.z.to_js()),
            ("p_values", f.p_values.to_js()),
            ("loglik", f.loglik.to_js()),
            ("lr_test", f.lr_test.to_js()),
            ("wald_test", f.wald_test.to_js()),
            ("score_test", f.score_test.to_js()),
            ("df", f.df.to_js()),
            ("n", f.n.to_js()),
            ("n_events", f.n_events.to_js()),
            ("iterations", f.iterations.to_js()),
            ("converged", f.converged.to_js()),
            ("concordance", f.concordance().to_js()),
            ("dxy", f.dxy().to_js()),
            ("r2", f.r2().to_js()),
        ])
    }

    /// The coefficients (log hazard ratios).
    #[wasm_bindgen(js_name = getCoefficients)]
    pub fn get_coefficients(&self) -> Vec<f64> {
        self.fitted.coefficients.clone()
    }

    /// Covariance of the coefficients, row-major `p × p`.
    #[wasm_bindgen(js_name = getVcov)]
    pub fn get_vcov(&self) -> Vec<f64> {
        crate::regressors::mat_flat(&self.fitted.var)
    }

    /// Whether the Newton iterations converged.
    #[wasm_bindgen(js_name = getConverged)]
    pub fn get_converged(&self) -> bool {
        self.fitted.converged
    }

    /// Newton iterations used.
    #[wasm_bindgen(js_name = getIterations)]
    pub fn get_iterations(&self) -> usize {
        self.fitted.iterations
    }

    /// p-value of a χ² statistic on the model's degrees of freedom.
    #[wasm_bindgen(js_name = testPValue)]
    pub fn test_p_value(&self, chisq: f64) -> f64 {
        self.fitted.test_p_value(chisq)
    }

    /// Linear predictors `xβ + offset` of the fitted rows (uncentred).
    #[wasm_bindgen(js_name = linearPredictors)]
    pub fn linear_predictors(&self) -> Vec<f64> {
        self.fitted.linear_predictors()
    }

    /// Harrell's C (`coxph`'s concordance).
    #[wasm_bindgen]
    pub fn concordance(&self) -> f64 {
        self.fitted.concordance()
    }

    /// Somers' Dxy `2(C − ½)`.
    #[wasm_bindgen]
    pub fn dxy(&self) -> f64 {
        self.fitted.dxy()
    }

    /// `rms::cph`'s R².
    #[wasm_bindgen]
    pub fn r2(&self) -> f64 {
        self.fitted.r2()
    }

    /// Score residuals, `n` rows of `p`.
    #[wasm_bindgen(js_name = scoreResiduals)]
    pub fn score_residuals(&self) -> JsValue {
        self.fitted.score_residuals().to_js()
    }

    /// Martingale residuals.
    #[wasm_bindgen(js_name = martingaleResiduals)]
    pub fn martingale_residuals(&self) -> Vec<f64> {
        self.fitted.martingale_residuals()
    }

    /// Deviance residuals.
    #[wasm_bindgen(js_name = devianceResiduals)]
    pub fn deviance_residuals(&self) -> Vec<f64> {
        self.fitted.deviance_residuals()
    }

    /// `dfbeta` residuals (approximate change in β̂ per left-out row), `n`
    /// rows of `p`.
    #[wasm_bindgen]
    pub fn dfbeta(&self) -> JsValue {
        self.fitted.dfbeta().to_js()
    }

    /// `dfbeta` divided by the standard errors.
    #[wasm_bindgen]
    pub fn dfbetas(&self) -> JsValue {
        self.fitted.dfbetas().to_js()
    }

    /// Schoenfeld residuals of the event rows, ordered by stratum and time:
    /// `{ rows, time, residuals }` with `residuals` one row of `p` per event
    /// (`scaled` gives `type = "scaledsch"`).
    #[wasm_bindgen(js_name = schoenfeldResiduals)]
    pub fn schoenfeld_residuals(&self, scaled: Option<bool>) -> JsValue {
        let (rows, time, r) = self.fitted.schoenfeld_residuals(scaled.unwrap_or(false));
        object(&[
            ("rows", rows.to_js()),
            ("time", time.to_js()),
            ("residuals", r.to_js()),
        ])
    }

    /// Cluster-robust covariance (`coxph(..., cluster = id)`), `p` rows of
    /// `p`; one cluster code per fitted row (each row its own cluster for
    /// `robust = TRUE`).
    #[wasm_bindgen(js_name = robustVcov)]
    pub fn robust_vcov(&self, clusters: &[u32]) -> Result<JsValue, JsError> {
        Ok(self
            .fitted
            .robust_vcov(&usizes(clusters))
            .map_err(fit_error)?
            .to_js())
    }

    /// Grambsch–Therneau test of proportional hazards (`cox.zph`):
    /// `transform` is "km" (default), "rank", "identity" or "log"; `terms`
    /// optionally groups columns into terms (array of column-index arrays).
    /// Returns `{ terms, global, x, time, y }` with tests `{ chisq, df,
    /// pValue }` and `y` the scaled Schoenfeld residuals.
    #[wasm_bindgen(js_name = coxZph)]
    pub fn cox_zph(&self, transform: Option<String>, terms: JsValue) -> Result<JsValue, JsError> {
        let terms: Option<Vec<Vec<usize>>> = parse(terms)?;
        Ok(self
            .fitted
            .cox_zph(parse_zph_transform(transform)?, terms.as_deref())
            .map_err(fit_error)?
            .to_js())
    }

    /// Predicted survival curve for the covariate row `x0` (uncentred) in
    /// `stratum` (omit for an unstratified fit), as `survfit(fit, newdata)`:
    /// `{ time, nRisk, cumhaz, surv, stdErr, lower, upper }`.
    #[wasm_bindgen(js_name = predictSurvival)]
    pub fn predict_survival(&self, x0: &[f64], stratum: Option<u32>) -> Result<JsValue, JsError> {
        Ok(self
            .fitted
            .predict_survival(x0, stratum.map(|s| s as usize))
            .map_err(fit_error)?
            .to_js())
    }

    /// What `lrTest` needs: `{ minusTwoLoglik, nullMinusTwoLoglik,
    /// lrDispersion, nParameters }`.
    #[wasm_bindgen]
    pub fn likelihood(&self) -> JsValue {
        crate::inference::likelihood_summary(&self.fitted)
    }

    /// Likelihood-ratio test against the null model: `{ statistic, df,
    /// pValue }`.
    #[wasm_bindgen(js_name = modelLrTest)]
    pub fn model_lr_test(&self) -> JsValue {
        crate::inference::model_lr(&self.fitted)
    }
}

/// Fit a Cox proportional-hazards model (`survival::coxph`) of `time` /
/// `event` on the row-major `nRows × nCols` covariates `x` (no intercept
/// column). `data` holds optional per-row arrays `{ entry?, weights?,
/// offset?, strata? }` (entry times give counting-process `(entry, time]`
/// data); `options` is `{ ties?: "efron" | "breslow", maxIter?, eps?,
/// ridge?, confLevel?, timefix? }`.
#[allow(clippy::too_many_arguments)]
#[wasm_bindgen]
pub fn coxph(
    x: &[f64],
    n_rows: usize,
    n_cols: usize,
    time: &[f64],
    event: &[u8],
    data: JsValue,
    options: JsValue,
) -> Result<CoxFit, JsError> {
    let x = to_mat(x, n_rows, n_cols)?;
    let event = events(event, n_rows)?;
    let d: CoxData = parse(data)?;
    let mut input = CoxInput::new(time, &event);
    if let Some(e) = &d.entry {
        input = input.entry(e);
    }
    if let Some(w) = &d.weights {
        input = input.weights(w);
    }
    if let Some(o) = &d.offset {
        input = input.offset(o);
    }
    if let Some(s) = &d.strata {
        input = input.strata(s);
    }
    let fitted = rust_coxph(&x, &input, &cox_options(options)?).map_err(fit_error)?;
    Ok(CoxFit { fitted })
}
