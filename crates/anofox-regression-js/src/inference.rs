//! Inference as free functions (`anofox_regression::inference`).
//!
//! Model-dependent inference (`predictSe`, `likelihood` / `modelLrTest`,
//! `clusterVcov`, `hcInference`, `vcov`) are methods on the fitted classes;
//! everything here works on arrays. Matrices are flat row-major arrays: a
//! covariance matrix is `p × p` for a coefficient vector of length `p`.

use anofox_regression::core::{IntervalType, PredictionResult, PredictionType};
use anofox_regression::inference::approximation::{
    approximate_model, approximation_path, heuristic_shrinkage, shrink_slopes,
};
use anofox_regression::inference::approximation::{ApproximationStep, ModelApproximation};
use anofox_regression::inference::pool::PooledWald;
use anofox_regression::inference::pool::{pool_rubin, pooled_wald_d1, PooledFit as RustPooledFit};
use anofox_regression::inference::{
    anova, anova_table, cluster_sandwich, compute_hc_inference, compute_hc_standard_errors,
    compute_prediction_intervals, compute_ridge_variance_factor, compute_variance_factor,
    compute_xtwx_inverse_augmented, compute_xtwx_inverse_augmented_reduced,
    compute_xtwx_inverse_reduced, compute_xtx_inverse, compute_xtx_inverse_augmented,
    compute_xtx_inverse_augmented_reduced, compute_xtx_inverse_reduced, drop_intercept, fastbw,
    fit_glm, glm_cluster_vcov, glm_score_test, intervals_from_variance_factor, leverage_new,
    lm_cluster_vcov, lr_test, model_lr_test, nested_f_test, nested_f_test_indices, profile_confint,
    wald_chunk, wald_test, AnovaOptions, AnovaRow, AnovaRowKind, AnovaTable, AnovaTableOptions,
    AnovaTerm, AnovaType, ChiSqTest, ClusterOptions, ClusterType, CoefficientInference, Dispersion,
    GlmData, GlmFit, HcInference, HcInterceptInference, HcResult, HcType, LikelihoodFit,
    NestedFTest, NestedFTestOptions, PredictSeOptions, PredictionSe, ScoreTest, StepDeletion,
    StepDown, StepRule, StepType, WaldAnovaTable, WaldDistribution, WaldTest,
};
use faer::{Col, Mat};
use serde::Deserialize;
use wasm_bindgen::prelude::*;

use crate::convert::{impl_to_js, impl_to_js_debug, ToJs};
use crate::design::TermMap;
use crate::diagnostics::parse_family;
use crate::regressors::{fit_error, to_col, to_mat};

impl_to_js_debug!(AnovaRowKind, AnovaType, PredictionType, IntervalType);

impl ToJs for HcType {
    fn to_js(&self) -> JsValue {
        JsValue::from_str(&format!("{:?}", self).to_lowercase())
    }
}

impl_to_js!(WaldTest { chi_square, df, f, df_residual, p_value }; chi_square_minus_df());
impl_to_js!(AnovaRow {
    label,
    kind,
    term,
    columns,
    test,
    partial_ss
});
impl_to_js!(WaldAnovaTable { rows, error });
impl_to_js!(AnovaTerm {
    term,
    columns,
    df,
    sum_sq,
    mean_sq,
    f_statistic,
    p_value
});
impl_to_js!(AnovaTable {
    anova_type,
    terms,
    residual_df,
    residual_sum_sq,
    residual_mean_sq,
    n_observations
});
impl_to_js!(ModelApproximation {
    kept_terms,
    columns,
    coefficients,
    vcov,
    r_squared
});
impl_to_js!(ApproximationStep { term, r_squared });
impl_to_js!(ChiSqTest {
    statistic,
    df,
    p_value
});
impl_to_js!(ScoreTest { test, z });
impl_to_js!(GlmFit {
    coefficients,
    vcov,
    deviance,
    null_deviance,
    dispersion,
    df_residual,
    mu,
    iterations
});
impl_to_js!(NestedFTest { f_statistic, df1, df2, p_value, rss_reduced, rss_full, adj_r_squared_reduced,
    adj_r_squared_full, n_parameters_reduced, n_parameters_full, n_observations }; delta_adj_r_squared());
impl_to_js!(RustPooledFit { m, coefficients, vcov, within, between, variance_inflation, riv, lambda, df_rubin,
    df_barnard_rubin, fmi, df_complete }; std_errors());
impl_to_js!(PooledWald {
    f,
    df1,
    df2,
    p_value,
    riv
});
impl_to_js!(PredictionSe {
    fit,
    se_fit,
    lower,
    upper,
    scale,
    interval,
    level,
    df,
    residual_scale
});
impl_to_js!(HcResult {
    std_errors,
    intercept_std_error
});
impl_to_js!(HcInterceptInference {
    std_error,
    t_statistic,
    p_value,
    conf_interval
});
impl_to_js!(HcInference {
    hc_type,
    std_errors,
    t_statistics,
    p_values,
    conf_interval_lower,
    conf_interval_upper,
    confidence_level,
    intercept
});
impl_to_js!(StepDeletion {
    term,
    chisq,
    df,
    p_value,
    residual_chisq,
    residual_df,
    residual_p,
    aic
});
impl_to_js!(StepDown {
    deleted,
    kept_terms,
    coefficients
});
impl_to_js!(PredictionResult {
    fit,
    lower,
    upper,
    se
});

// ------------------------------------------------------------ parsing ----

fn col(v: &[f64]) -> Col<f64> {
    Col::from_fn(v.len(), |i| v[i])
}

/// A `p × p` matrix from a flat row-major array.
pub(crate) fn square(v: &[f64], p: usize, what: &str) -> Result<Mat<f64>, JsError> {
    if v.len() != p * p {
        return Err(JsError::new(&format!(
            "{}: expected {} ({}x{}) values, got {}",
            what,
            p * p,
            p,
            p,
            v.len()
        )));
    }
    Ok(Mat::from_fn(p, p, |i, j| v[i * p + j]))
}

fn bools(v: &Option<Vec<u8>>, n: usize) -> Vec<bool> {
    match v {
        Some(v) => v.iter().map(|&a| a != 0).collect(),
        None => vec![false; n],
    }
}

fn usizes(v: &[u32]) -> Vec<usize> {
    v.iter().map(|&c| c as usize).collect()
}

fn parse_dist(df_residual: Option<f64>) -> WaldDistribution {
    match df_residual {
        Some(df_residual) => WaldDistribution::F { df_residual },
        None => WaldDistribution::ChiSquare,
    }
}

pub(crate) fn parse_hc_type(kind: &str) -> Result<HcType, JsError> {
    match kind.to_ascii_lowercase().as_str() {
        "hc0" => Ok(HcType::HC0),
        "hc1" => Ok(HcType::HC1),
        "hc2" => Ok(HcType::HC2),
        "hc3" => Ok(HcType::HC3),
        other => Err(JsError::new(&format!(
            "Unknown HC type '{}': expected hc0, hc1, hc2 or hc3",
            other
        ))),
    }
}

/// Cluster-robust covariance options as passed from JS: a preset
/// ("cr1" — `vcovCL`'s default, "hc0", "robcov" — `rms::robcov`) or
/// `{ type: "hc0" | "hc1" | "hc2" | "hc3", cadjust: boolean }`.
pub(crate) fn parse_cluster_options(options: JsValue) -> Result<ClusterOptions, JsError> {
    if options.is_undefined() || options.is_null() {
        return Ok(ClusterOptions::default());
    }
    if let Some(preset) = options.as_string() {
        return match preset.as_str() {
            "cr1" => Ok(ClusterOptions::cr1()),
            "hc0" => Ok(ClusterOptions::hc0()),
            "robcov" => Ok(ClusterOptions::robcov()),
            other => Err(JsError::new(&format!(
                "Unknown cluster preset '{}': expected cr1, hc0 or robcov",
                other
            ))),
        };
    }
    #[derive(Deserialize)]
    struct Opts {
        #[serde(rename = "type")]
        kind: Option<String>,
        cadjust: Option<bool>,
    }
    let o: Opts =
        serde_wasm_bindgen::from_value(options).map_err(|e| JsError::new(&e.to_string()))?;
    let mut out = ClusterOptions::default();
    if let Some(kind) = o.kind {
        out = out.kind(match parse_hc_type(&kind)? {
            HcType::HC0 => ClusterType::HC0,
            HcType::HC1 => ClusterType::HC1,
            HcType::HC2 => ClusterType::HC2,
            _ => ClusterType::HC3,
        });
    }
    if let Some(c) = o.cadjust {
        out = out.cadjust(c);
    }
    Ok(out)
}

/// One or two clustering dimensions, each one code per row.
pub(crate) fn parse_clusters(clusters: JsValue, n_rows: usize) -> Result<Vec<Vec<usize>>, JsError> {
    let c: Vec<Vec<usize>> =
        serde_wasm_bindgen::from_value(clusters).map_err(|e| JsError::new(&e.to_string()))?;
    if let Some(d) = c.iter().find(|d| d.len() != n_rows) {
        return Err(JsError::new(&format!(
            "Expected {} cluster codes per dimension, got {}",
            n_rows,
            d.len()
        )));
    }
    Ok(c)
}

pub(crate) fn cluster_refs(c: &[Vec<usize>]) -> Vec<&[usize]> {
    c.iter().map(Vec::as_slice).collect()
}

/// `predictSe` options: `{ scale?: "response" | "link", interval?:
/// "confidence" | "prediction", level?: number }`.
pub(crate) fn parse_predict_se_options(options: JsValue) -> Result<PredictSeOptions, JsError> {
    let mut out = PredictSeOptions::new();
    if options.is_undefined() || options.is_null() {
        return Ok(out);
    }
    #[derive(Deserialize)]
    struct Opts {
        scale: Option<String>,
        interval: Option<String>,
        level: Option<f64>,
    }
    let o: Opts =
        serde_wasm_bindgen::from_value(options).map_err(|e| JsError::new(&e.to_string()))?;
    match o.scale.as_deref() {
        None | Some("response") => out = out.response(),
        Some("link") => out = out.link(),
        Some(other) => {
            return Err(JsError::new(&format!(
                "Unknown scale '{}': expected response or link",
                other
            )))
        }
    }
    match o.interval.as_deref() {
        None | Some("confidence") => out = out.confidence(),
        Some("prediction") => out = out.prediction(),
        Some(other) => {
            return Err(JsError::new(&format!(
                "Unknown interval '{}': expected confidence or prediction",
                other
            )))
        }
    }
    if let Some(level) = o.level {
        out = out.level(level);
    }
    Ok(out)
}

// ------------------------------------------------- likelihood-ratio tests ----

/// What a likelihood-ratio test needs of a fit; the `likelihood()` method of
/// the fitted classes returns it.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct LikelihoodSummary {
    minus_two_loglik: f64,
    null_minus_two_loglik: Option<f64>,
    #[serde(default = "one")]
    lr_dispersion: f64,
    n_parameters: usize,
}

fn one() -> f64 {
    1.0
}

impl LikelihoodFit for LikelihoodSummary {
    fn minus_two_loglik(&self) -> f64 {
        self.minus_two_loglik
    }
    fn null_minus_two_loglik(&self) -> Option<f64> {
        self.null_minus_two_loglik
    }
    fn lr_dispersion(&self) -> f64 {
        self.lr_dispersion
    }
    fn n_parameters(&self) -> usize {
        self.n_parameters
    }
}

/// The likelihood summary of a fit:
/// `{ minusTwoLoglik, nullMinusTwoLoglik?, lrDispersion, nParameters }`.
pub(crate) fn likelihood_summary<F: LikelihoodFit + ?Sized>(fit: &F) -> JsValue {
    crate::convert::object(&[
        ("minus_two_loglik", fit.minus_two_loglik().to_js()),
        ("null_minus_two_loglik", fit.null_minus_two_loglik().to_js()),
        ("lr_dispersion", fit.lr_dispersion().to_js()),
        ("n_parameters", fit.n_parameters().to_js()),
    ])
}

/// The model likelihood-ratio test against the null model, if known.
pub(crate) fn model_lr<F: LikelihoodFit + ?Sized>(fit: &F) -> JsValue {
    model_lr_test(fit).to_js()
}

/// Likelihood-ratio test of nested fits `reduced ⊂ full`, from their
/// `likelihood()` summaries: `{ statistic, df, pValue }`.
#[wasm_bindgen(js_name = lrTest)]
pub fn lr_test_js(reduced: JsValue, full: JsValue) -> Result<JsValue, JsError> {
    let parse = |v: JsValue| -> Result<LikelihoodSummary, JsError> {
        serde_wasm_bindgen::from_value(v).map_err(|e| JsError::new(&e.to_string()))
    };
    Ok(lr_test(&parse(reduced)?, &parse(full)?).to_js())
}

// -------------------------------------------------------------- Wald, ANOVA ----

/// Wald test of `C β = r` (`r` = 0 by default) with the `q × p` contrast
/// matrix `c` (row-major); χ² by default, F with `dfResidual`.
#[wasm_bindgen(js_name = waldTest)]
pub fn wald_test_js(
    beta: &[f64],
    vcov: &[f64],
    c: &[f64],
    r: Option<Vec<f64>>,
    df_residual: Option<f64>,
) -> Result<JsValue, JsError> {
    let p = beta.len();
    if p == 0 || !c.len().is_multiple_of(p) {
        return Err(JsError::new(
            "c must have one row of length beta.length per constraint",
        ));
    }
    let c = to_mat(c, c.len() / p, p)?;
    let r = r.map(|r| col(&r));
    Ok(wald_test(
        &col(beta),
        &square(vcov, p, "vcov")?,
        &c,
        r.as_ref(),
        parse_dist(df_residual),
    )
    .map_err(fit_error)?
    .to_js())
}

/// Wald test that the coefficients `columns` are all zero.
#[wasm_bindgen(js_name = waldChunk)]
pub fn wald_chunk_js(
    beta: &[f64],
    vcov: &[f64],
    columns: &[u32],
    df_residual: Option<f64>,
) -> Result<JsValue, JsError> {
    let p = beta.len();
    Ok(wald_chunk(
        &col(beta),
        &square(vcov, p, "vcov")?,
        &usizes(columns),
        parse_dist(df_residual),
    )
    .map_err(fit_error)?
    .to_js())
}

/// Drop a leading intercept from `beta` and `vcov`: `{ beta, vcov }`.
#[wasm_bindgen(js_name = dropIntercept)]
pub fn drop_intercept_js(beta: &[f64], vcov: &[f64]) -> Result<JsValue, JsError> {
    let (b, v) = drop_intercept(&col(beta), &square(vcov, beta.len(), "vcov")?);
    Ok(crate::convert::object(&[
        ("beta", b.to_js()),
        ("vcov", v.to_js()),
    ]))
}

/// `rms::anova`: Wald tests of every term, its nonlinear and interaction
/// parts and the totals, from slope coefficients `beta` (no intercept) and
/// their covariance. F tests with `dfResidual`; `sigma2` adds partial sums
/// of squares.
#[wasm_bindgen(js_name = anova)]
pub fn anova_wald(
    beta: &[f64],
    vcov: &[f64],
    terms: &TermMap,
    df_residual: Option<f64>,
    sigma2: Option<f64>,
) -> Result<JsValue, JsError> {
    let mut options = AnovaOptions::default().distribution(parse_dist(df_residual));
    if let Some(s) = sigma2 {
        options = options.sigma2(Some(s));
    }
    Ok(anova(
        &col(beta),
        &square(vcov, beta.len(), "vcov")?,
        &terms.inner,
        &options,
    )
    .map_err(fit_error)?
    .to_js())
}

/// Sequential (type I) or type II ANOVA table of a linear model; `terms` is
/// an array of column-index arrays (one term per column by default).
#[wasm_bindgen(js_name = anovaTable)]
#[allow(clippy::too_many_arguments)]
pub fn anova_table_js(
    x: &[f64],
    n_rows: usize,
    n_cols: usize,
    y: &[f64],
    terms: JsValue,
    anova_type: Option<String>,
    with_intercept: Option<bool>,
) -> Result<JsValue, JsError> {
    let x_mat = to_mat(x, n_rows, n_cols)?;
    let y_col = to_col(y, n_rows, "target")?;
    let terms: Option<Vec<Vec<usize>>> = if terms.is_undefined() || terms.is_null() {
        None
    } else {
        Some(serde_wasm_bindgen::from_value(terms).map_err(|e| JsError::new(&e.to_string()))?)
    };
    let kind = match anova_type.as_deref() {
        None | Some("I") | Some("1") => AnovaType::I,
        Some("II") | Some("2") => AnovaType::II,
        Some(other) => {
            return Err(JsError::new(&format!(
                "Unknown ANOVA type '{}': expected I or II",
                other
            )))
        }
    };
    let options = AnovaTableOptions::default()
        .anova_type(kind)
        .with_intercept(with_intercept.unwrap_or(true));
    Ok(anova_table(&x_mat, &y_col, terms.as_deref(), &options)
        .map_err(fit_error)?
        .to_js())
}

/// `rms::fastbw`: fast backward step-down on the full-model `beta` / `vcov`
/// (the first `nIntercepts` entries are intercepts). `rule` is "aic"
/// (`threshold`, default 0, penalty `k` = 2) or "p" (`threshold` = sls,
/// default 0.05); `type` is "residual" (default) or "individual".
#[wasm_bindgen(js_name = fastbw)]
#[allow(clippy::too_many_arguments)]
pub fn fastbw_js(
    beta: &[f64],
    vcov: &[f64],
    n_intercepts: usize,
    terms: &TermMap,
    rule: Option<String>,
    threshold: Option<f64>,
    step_type: Option<String>,
    k: Option<f64>,
) -> Result<JsValue, JsError> {
    let rule = match rule.as_deref() {
        None | Some("aic") => StepRule::Aic {
            k: k.unwrap_or(2.0),
            threshold: threshold.unwrap_or(0.0),
        },
        Some("p") => StepRule::P {
            sls: threshold.unwrap_or(0.05),
        },
        Some(other) => {
            return Err(JsError::new(&format!(
                "Unknown rule '{}': expected aic or p",
                other
            )))
        }
    };
    let kind = match step_type.as_deref() {
        None | Some("residual") => StepType::Residual,
        Some("individual") => StepType::Individual,
        Some(other) => {
            return Err(JsError::new(&format!(
                "Unknown type '{}': expected residual or individual",
                other
            )))
        }
    };
    let p = beta.len();
    Ok(fastbw(
        &col(beta),
        &square(vcov, p, "vcov")?,
        n_intercepts,
        &terms.inner,
        rule,
        kind,
    )
    .to_js())
}

// ------------------------------------------------------ approximation ----

/// Van Houwelingen–Le Cessie heuristic shrinkage `(LR − df) / LR`.
#[wasm_bindgen(js_name = heuristicShrinkage)]
pub fn heuristic_shrinkage_js(lr_chisq: f64, df: f64) -> f64 {
    heuristic_shrinkage(lr_chisq, df)
}

/// Multiply the slopes (after `nIntercepts` intercepts) by `gamma`.
#[wasm_bindgen(js_name = shrinkSlopes)]
pub fn shrink_slopes_js(beta: &[f64], n_intercepts: usize, gamma: f64) -> Vec<f64> {
    shrink_slopes(&col(beta), n_intercepts, gamma)
        .iter()
        .copied()
        .collect()
}

/// Approximate the linear predictor `lp` by the terms `keep` (RMS §5.5);
/// `vcov` is the covariance of (intercept, slopes):
/// `{ keptTerms, columns, coefficients, vcov, rSquared }`.
#[wasm_bindgen(js_name = approximateModel)]
pub fn approximate_model_js(
    x: &[f64],
    n_rows: usize,
    lp: &[f64],
    vcov: &[f64],
    terms: &TermMap,
    keep: &[u32],
) -> Result<JsValue, JsError> {
    let p = terms.inner.n_columns();
    let x_mat = to_mat(x, n_rows, p)?;
    let lp = to_col(lp, n_rows, "linear predictor")?;
    let v = square(vcov, p + 1, "vcov (intercept first)")?;
    Ok(
        approximate_model(&x_mat, &lp, &v, &terms.inner, &usizes(keep))
            .map_err(fit_error)?
            .to_js(),
    )
}

/// Backward approximation path of `lp` by the terms of `x`: the R² after
/// dropping each term in turn, `[{ term, rSquared }]`.
#[wasm_bindgen(js_name = approximationPath)]
pub fn approximation_path_js(
    x: &[f64],
    n_rows: usize,
    lp: &[f64],
    terms: &TermMap,
) -> Result<JsValue, JsError> {
    let x_mat = to_mat(x, n_rows, terms.inner.n_columns())?;
    let lp = to_col(lp, n_rows, "linear predictor")?;
    Ok(approximation_path(&x_mat, &lp, &terms.inner)
        .map_err(fit_error)?
        .to_js())
}

// ------------------------------------------------------------- cluster ----

/// Cluster sandwich `A (Σ_g s_g s_gᵀ) A` from the `k × k` bread `A` and the
/// `n × k` per-observation scores; `clusters` is one or two arrays of
/// cluster codes.
#[wasm_bindgen(js_name = clusterSandwich)]
pub fn cluster_sandwich_js(
    bread: &[f64],
    scores: &[f64],
    n_rows: usize,
    clusters: JsValue,
    options: JsValue,
) -> Result<JsValue, JsError> {
    let k = (bread.len() as f64).sqrt() as usize;
    let bread = square(bread, k, "bread")?;
    let scores = to_mat(scores, n_rows, k)?;
    let c = parse_clusters(clusters, n_rows)?;
    Ok(cluster_sandwich(
        &bread,
        &scores,
        &cluster_refs(&c),
        &parse_cluster_options(options)?,
    )
    .map_err(fit_error)?
    .to_js())
}

/// Cluster-robust covariance of a linear model (`sandwich::vcovCL`),
/// intercept first.
#[wasm_bindgen(js_name = lmClusterVcov)]
#[allow(clippy::too_many_arguments)]
pub fn lm_cluster_vcov_js(
    x: &[f64],
    n_rows: usize,
    n_cols: usize,
    residuals: &[f64],
    weights: Option<Vec<f64>>,
    with_intercept: bool,
    aliased: Option<Vec<u8>>,
    clusters: JsValue,
    options: JsValue,
) -> Result<JsValue, JsError> {
    let x_mat = to_mat(x, n_rows, n_cols)?;
    let res = to_col(residuals, n_rows, "residual")?;
    let w = weights.map(|w| to_col(&w, n_rows, "weight")).transpose()?;
    let c = parse_clusters(clusters, n_rows)?;
    Ok(lm_cluster_vcov(
        &x_mat,
        &res,
        w.as_ref(),
        with_intercept,
        &bools(&aliased, n_cols),
        &cluster_refs(&c),
        &parse_cluster_options(options)?,
    )
    .map_err(fit_error)?
    .to_js())
}

/// Cluster-robust covariance of a GLM from its fitted means `mu`.
#[wasm_bindgen(js_name = glmClusterVcov)]
#[allow(clippy::too_many_arguments)]
pub fn glm_cluster_vcov_js(
    family: &str,
    x: &[f64],
    n_rows: usize,
    n_cols: usize,
    with_intercept: bool,
    y: &[f64],
    mu: &[f64],
    prior_weights: Option<Vec<f64>>,
    clusters: JsValue,
    options: JsValue,
    link: Option<String>,
    param: Option<f64>,
) -> Result<JsValue, JsError> {
    let fam = parse_family(family, link, param)?;
    let x_mat = to_mat(x, n_rows, n_cols)?;
    to_col(y, n_rows, "target")?;
    to_col(mu, n_rows, "mean")?;
    let c = parse_clusters(clusters, n_rows)?;
    Ok(glm_cluster_vcov(
        fam.as_dyn(),
        &x_mat,
        with_intercept,
        y,
        mu,
        prior_weights.as_deref(),
        &cluster_refs(&c),
        &parse_cluster_options(options)?,
    )
    .map_err(fit_error)?
    .to_js())
}

// ---------------------------------------------------------- GLM by IRLS ----

/// Options of `fitGlm` / `glmScoreTest` / `profileConfint`:
/// `{ withIntercept?, weights?, offset?, link?, param?, dispersion? }`
/// with `dispersion` a number (fixed) or "estimated" (default: fixed at 1).
#[derive(Deserialize, Default)]
#[serde(rename_all = "camelCase")]
struct GlmOpts {
    with_intercept: Option<bool>,
    weights: Option<Vec<f64>>,
    offset: Option<Vec<f64>>,
    link: Option<String>,
    param: Option<f64>,
    dispersion: Option<serde_json_like::Disp>,
}

mod serde_json_like {
    use serde::Deserialize;

    #[derive(Deserialize)]
    #[serde(untagged)]
    pub enum Disp {
        Fixed(f64),
        Named(String),
    }
}

fn glm_opts(options: JsValue) -> Result<GlmOpts, JsError> {
    if options.is_undefined() || options.is_null() {
        return Ok(GlmOpts::default());
    }
    serde_wasm_bindgen::from_value(options).map_err(|e| JsError::new(&e.to_string()))
}

fn dispersion(o: &GlmOpts) -> Result<Dispersion, JsError> {
    match &o.dispersion {
        None => Ok(Dispersion::Fixed(1.0)),
        Some(serde_json_like::Disp::Fixed(v)) => Ok(Dispersion::Fixed(*v)),
        Some(serde_json_like::Disp::Named(s)) if s == "estimated" => Ok(Dispersion::Estimated),
        Some(serde_json_like::Disp::Named(s)) => Err(JsError::new(&format!(
            "Unknown dispersion '{}': expected a number or \"estimated\"",
            s
        ))),
    }
}

fn glm_data<'a>(x: &'a Mat<f64>, y: &'a [f64], o: &'a GlmOpts) -> GlmData<'a> {
    GlmData::new(x, y)
        .with_intercept(o.with_intercept.unwrap_or(true))
        .weights(o.weights.as_deref())
        .offset(o.offset.as_deref())
}

/// A GLM fitted by IRLS with `fitGlm`.
#[wasm_bindgen]
pub struct FittedGlm {
    fitted: GlmFit,
}

#[wasm_bindgen]
impl FittedGlm {
    /// `{ coefficients, vcov, deviance, nullDeviance, dispersion,
    /// dfResidual, mu, iterations }` (coefficients intercept first).
    #[wasm_bindgen(js_name = getResult)]
    pub fn get_result(&self) -> JsValue {
        self.fitted.to_js()
    }

    /// The likelihood summary for `lrTest`.
    #[wasm_bindgen]
    pub fn likelihood(&self) -> JsValue {
        likelihood_summary(&self.fitted)
    }

    /// The model likelihood-ratio test against the null model.
    #[wasm_bindgen(js_name = modelLrTest)]
    pub fn model_lr_test(&self) -> JsValue {
        model_lr(&self.fitted)
    }
}

/// Fit a GLM of `family` ("poisson", "binomial", "negative_binomial",
/// "gamma", "tweedie") by IRLS (see `GlmOpts` for `options`).
#[wasm_bindgen(js_name = fitGlm)]
pub fn fit_glm_js(
    family: &str,
    x: &[f64],
    n_rows: usize,
    n_cols: usize,
    y: &[f64],
    options: JsValue,
) -> Result<FittedGlm, JsError> {
    let o = glm_opts(options)?;
    let fam = parse_family(family, o.link.clone(), o.param)?;
    let x_mat = to_mat(x, n_rows, n_cols)?;
    to_col(y, n_rows, "target")?;
    let fitted =
        fit_glm(fam.as_dyn(), &glm_data(&x_mat, y, &o), dispersion(&o)?).map_err(fit_error)?;
    Ok(FittedGlm { fitted })
}

/// Rao score test for adding the columns `xAdd` (`n × nAdd`) to the GLM
/// of `x`: `{ test: { statistic, df, pValue }, z }`.
#[wasm_bindgen(js_name = glmScoreTest)]
#[allow(clippy::too_many_arguments)]
pub fn glm_score_test_js(
    family: &str,
    x: &[f64],
    n_rows: usize,
    n_cols: usize,
    y: &[f64],
    x_add: &[f64],
    n_add: usize,
    dispersion: f64,
    options: JsValue,
) -> Result<JsValue, JsError> {
    let o = glm_opts(options)?;
    let fam = parse_family(family, o.link.clone(), o.param)?;
    let x_mat = to_mat(x, n_rows, n_cols)?;
    let x_add = to_mat(x_add, n_rows, n_add)?;
    to_col(y, n_rows, "target")?;
    Ok(
        glm_score_test(fam.as_dyn(), &glm_data(&x_mat, y, &o), &x_add, dispersion)
            .map_err(fit_error)?
            .to_js(),
    )
}

/// Profile-likelihood confidence intervals of the GLM coefficients
/// (intercept first): `[[lower, upper], …]` (`confint.glm`).
#[wasm_bindgen(js_name = profileConfint)]
pub fn profile_confint_js(
    family: &str,
    x: &[f64],
    n_rows: usize,
    n_cols: usize,
    y: &[f64],
    level: f64,
    options: JsValue,
) -> Result<JsValue, JsError> {
    let o = glm_opts(options)?;
    let fam = parse_family(family, o.link.clone(), o.param)?;
    let x_mat = to_mat(x, n_rows, n_cols)?;
    to_col(y, n_rows, "target")?;
    Ok(profile_confint(
        fam.as_dyn(),
        &glm_data(&x_mat, y, &o),
        dispersion(&o)?,
        level,
    )
    .map_err(fit_error)?
    .to_js())
}

// -------------------------------------------------------------- nested F ----

/// F test of the nested linear models `xReduced ⊂ xFull`.
#[wasm_bindgen(js_name = nestedFTest)]
#[allow(clippy::too_many_arguments)]
pub fn nested_f_test_js(
    x_reduced: &[f64],
    n_cols_reduced: usize,
    x_full: &[f64],
    n_cols_full: usize,
    y: &[f64],
    with_intercept: Option<bool>,
) -> Result<JsValue, JsError> {
    let n = y.len();
    let options = NestedFTestOptions::default().with_intercept(with_intercept.unwrap_or(true));
    Ok(nested_f_test(
        &to_mat(x_reduced, n, n_cols_reduced)?,
        &to_mat(x_full, n, n_cols_full)?,
        &col(y),
        &options,
    )
    .map_err(fit_error)?
    .to_js())
}

/// F test of the full model `x` against the model of its columns `reduced`.
#[wasm_bindgen(js_name = nestedFTestIndices)]
pub fn nested_f_test_indices_js(
    x_full: &[f64],
    n_cols: usize,
    y: &[f64],
    reduced: &[u32],
    with_intercept: Option<bool>,
) -> Result<JsValue, JsError> {
    let options = NestedFTestOptions::default().with_intercept(with_intercept.unwrap_or(true));
    Ok(nested_f_test_indices(
        &to_mat(x_full, y.len(), n_cols)?,
        &col(y),
        &usizes(reduced),
        &options,
    )
    .map_err(fit_error)?
    .to_js())
}

// ------------------------------------------------------- multiple imputation ----

/// Rubin's rules over `m` imputed fits.
#[wasm_bindgen]
pub struct PooledFit {
    inner: RustPooledFit,
}

#[wasm_bindgen]
impl PooledFit {
    /// The pooled estimates: `{ m, coefficients, stdErrors, vcov, within,
    /// between, varianceInflation, riv, lambda, dfRubin, dfBarnardRubin,
    /// fmi, dfComplete }`.
    #[wasm_bindgen(js_name = getResult)]
    pub fn get_result(&self) -> JsValue {
        self.inner.to_js()
    }

    /// The D1 multivariate Wald test that the coefficients `indices` are zero.
    #[wasm_bindgen(js_name = waldD1)]
    pub fn wald_d1(&self, indices: &[u32]) -> Result<JsValue, JsError> {
        Ok(pooled_wald_d1(&self.inner, &usizes(indices))
            .map_err(fit_error)?
            .to_js())
    }
}

/// Pool `m` imputed fits by Rubin's rules (`mice::pool`): `estimates` is
/// an array of coefficient arrays, `vcovs` an array of flat row-major
/// covariance matrices; `dfComplete` gives the Barnard–Rubin d.f.
#[wasm_bindgen(js_name = poolRubin)]
pub fn pool_rubin_js(
    estimates: JsValue,
    vcovs: JsValue,
    df_complete: Option<f64>,
) -> Result<PooledFit, JsError> {
    let est: Vec<Vec<f64>> =
        serde_wasm_bindgen::from_value(estimates).map_err(|e| JsError::new(&e.to_string()))?;
    let vc: Vec<Vec<f64>> =
        serde_wasm_bindgen::from_value(vcovs).map_err(|e| JsError::new(&e.to_string()))?;
    let est: Vec<Col<f64>> = est.iter().map(|e| col(e)).collect();
    let vc: Vec<Mat<f64>> = vc
        .iter()
        .zip(&est)
        .map(|(v, e)| square(v, e.nrows(), "vcov"))
        .collect::<Result<_, _>>()?;
    let inner = pool_rubin(&est, &vc, df_complete).map_err(fit_error)?;
    Ok(PooledFit { inner })
}

// ------------------------------------------- coefficient inference helpers ----

/// Standard errors `sqrt(mse · diag((XᵀX)⁻¹))` of the columns of `x`.
#[wasm_bindgen(js_name = coefficientStandardErrors)]
pub fn coefficient_standard_errors(
    x: &[f64],
    n_rows: usize,
    n_cols: usize,
    mse: f64,
    aliased: Option<Vec<u8>>,
) -> Result<Vec<f64>, JsError> {
    let se = CoefficientInference::standard_errors(
        &to_mat(x, n_rows, n_cols)?,
        mse,
        &bools(&aliased, n_cols),
    )
    .map_err(fit_error)?;
    Ok(se.iter().copied().collect())
}

/// Standard errors with an intercept column: `{ stdErrors, interceptStdError }`;
/// with `weights`, the WLS version.
#[wasm_bindgen(js_name = standardErrorsWithIntercept)]
pub fn standard_errors_with_intercept(
    x: &[f64],
    n_rows: usize,
    n_cols: usize,
    mse: f64,
    aliased: Option<Vec<u8>>,
    weights: Option<Vec<f64>>,
) -> Result<JsValue, JsError> {
    let x_mat = to_mat(x, n_rows, n_cols)?;
    let al = bools(&aliased, n_cols);
    let (se, se0) = match weights {
        Some(w) => CoefficientInference::standard_errors_wls_with_intercept(
            &x_mat,
            &to_col(&w, n_rows, "weight")?,
            mse,
            &al,
        ),
        None => CoefficientInference::standard_errors_with_intercept(&x_mat, mse, &al),
    }
    .map_err(fit_error)?;
    Ok(crate::convert::object(&[
        ("std_errors", se.to_js()),
        ("intercept_std_error", se0.to_js()),
    ]))
}

/// t statistics `β / SE`.
#[wasm_bindgen(js_name = tStatistics)]
pub fn t_statistics(coefficients: &[f64], std_errors: &[f64]) -> Vec<f64> {
    CoefficientInference::t_statistics(&col(coefficients), &col(std_errors))
        .iter()
        .copied()
        .collect()
}

/// Two-sided t p-values on `df` degrees of freedom.
#[wasm_bindgen(js_name = coefficientPValues)]
pub fn coefficient_p_values(t_statistics: &[f64], df: f64) -> Vec<f64> {
    CoefficientInference::p_values(&col(t_statistics), df)
        .iter()
        .copied()
        .collect()
}

/// t confidence intervals: `{ lower, upper }`.
#[wasm_bindgen(js_name = coefficientConfidenceIntervals)]
pub fn coefficient_confidence_intervals(
    coefficients: &[f64],
    std_errors: &[f64],
    df: f64,
    level: f64,
) -> JsValue {
    let (lo, hi) =
        CoefficientInference::confidence_intervals(&col(coefficients), &col(std_errors), df, level);
    crate::convert::object(&[("lower", lo.to_js()), ("upper", hi.to_js())])
}

// ------------------------------------------------------- robust covariance ----

/// Heteroskedasticity-consistent standard errors of an OLS fit from its
/// residuals: `{ stdErrors, interceptStdError? }`.
#[wasm_bindgen(js_name = computeHcStandardErrors)]
#[allow(clippy::too_many_arguments)]
pub fn compute_hc_standard_errors_js(
    x: &[f64],
    n_rows: usize,
    n_cols: usize,
    residuals: &[f64],
    with_intercept: bool,
    hc_type: &str,
    aliased: Option<Vec<u8>>,
) -> Result<JsValue, JsError> {
    Ok(compute_hc_standard_errors(
        &to_mat(x, n_rows, n_cols)?,
        &to_col(residuals, n_rows, "residual")?,
        &bools(&aliased, n_cols),
        with_intercept,
        parse_hc_type(hc_type)?,
    )
    .map_err(fit_error)?
    .to_js())
}

/// HC standard errors with t tests and intervals.
#[wasm_bindgen(js_name = computeHcInference)]
#[allow(clippy::too_many_arguments)]
pub fn compute_hc_inference_js(
    x: &[f64],
    n_rows: usize,
    n_cols: usize,
    coefficients: &[f64],
    intercept: Option<f64>,
    residuals: &[f64],
    hc_type: &str,
    confidence_level: f64,
    aliased: Option<Vec<u8>>,
) -> Result<JsValue, JsError> {
    Ok(compute_hc_inference(
        &to_mat(x, n_rows, n_cols)?,
        &col(coefficients),
        intercept,
        &to_col(residuals, n_rows, "residual")?,
        &bools(&aliased, n_cols),
        intercept.is_some(),
        parse_hc_type(hc_type)?,
        confidence_level,
    )
    .map_err(fit_error)?
    .to_js())
}

// ------------------------------------------------------- (XᵀX)⁻¹ helpers ----

/// `(XᵀX)⁻¹` (or `(XᵀWX)⁻¹` with `weights`), optionally with an intercept
/// column (`augmented`) and without the `aliased` columns (`reduced`):
/// nested rows.
#[wasm_bindgen(js_name = computeXtxInverse)]
pub fn compute_xtx_inverse_js(
    x: &[f64],
    n_rows: usize,
    n_cols: usize,
    augmented: bool,
    weights: Option<Vec<f64>>,
    aliased: Option<Vec<u8>>,
) -> Result<JsValue, JsError> {
    let x_mat = to_mat(x, n_rows, n_cols)?;
    let w = weights.map(|w| to_col(&w, n_rows, "weight")).transpose()?;
    let al = aliased.as_ref().map(|_| bools(&aliased, n_cols));
    let m = match (augmented, &w, &al) {
        (false, None, None) => compute_xtx_inverse(&x_mat),
        (true, None, None) => compute_xtx_inverse_augmented(&x_mat),
        (false, None, Some(a)) => compute_xtx_inverse_reduced(&x_mat, a),
        (true, None, Some(a)) => compute_xtx_inverse_augmented_reduced(&x_mat, a),
        (false, Some(w), None) => compute_xtwx_inverse_reduced(&x_mat, w, &vec![false; n_cols]),
        (true, Some(w), None) => compute_xtwx_inverse_augmented(&x_mat, w),
        (false, Some(w), Some(a)) => compute_xtwx_inverse_reduced(&x_mat, w, a),
        (true, Some(w), Some(a)) => compute_xtwx_inverse_augmented_reduced(&x_mat, w, a),
    }
    .map_err(fit_error)?;
    Ok(m.to_js())
}

fn interval_type(kind: &str) -> Result<IntervalType, JsError> {
    crate::regressors::parse_interval(kind)
}

/// Confidence or prediction intervals of linear-model predictions from
/// `(XᵀX)⁻¹` (`xtxInv`, flat, including the intercept when `hasIntercept`):
/// `{ fit, lower, upper, se }`.
#[wasm_bindgen(js_name = computePredictionIntervals)]
#[allow(clippy::too_many_arguments)]
pub fn compute_prediction_intervals_js(
    x_new: &[f64],
    n_rows: usize,
    n_cols: usize,
    xtx_inv: &[f64],
    predictions: &[f64],
    mse: f64,
    df: f64,
    level: f64,
    interval: &str,
    has_intercept: bool,
) -> Result<JsValue, JsError> {
    let k = n_cols + usize::from(has_intercept);
    Ok(compute_prediction_intervals(
        &to_mat(x_new, n_rows, n_cols)?,
        &square(xtx_inv, k, "xtxInv")?,
        &to_col(predictions, n_rows, "prediction")?,
        mse,
        df,
        level,
        interval_type(interval)?,
        has_intercept,
    )
    .to_js())
}

/// The variance factor `M` (`Var(β̂) = σ² M`, intercept first) of a
/// linear or ridge fit: nested rows.
#[wasm_bindgen(js_name = computeVarianceFactor)]
#[allow(clippy::too_many_arguments)]
pub fn compute_variance_factor_js(
    x: &[f64],
    n_rows: usize,
    n_cols: usize,
    weights: Option<Vec<f64>>,
    with_intercept: bool,
    excluded: Option<Vec<u8>>,
    ridge_lambda: f64,
) -> Result<JsValue, JsError> {
    let w = weights.map(|w| to_col(&w, n_rows, "weight")).transpose()?;
    Ok(compute_variance_factor(
        &to_mat(x, n_rows, n_cols)?,
        w.as_ref(),
        with_intercept,
        &bools(&excluded, n_cols),
        ridge_lambda,
    )
    .map_err(fit_error)?
    .to_js())
}

/// The ridge sandwich variance factor (`lambda` on the slopes).
#[wasm_bindgen(js_name = computeRidgeVarianceFactor)]
#[allow(clippy::too_many_arguments)]
pub fn compute_ridge_variance_factor_js(
    x: &[f64],
    n_rows: usize,
    n_cols: usize,
    weights: Option<Vec<f64>>,
    lambda: f64,
    with_intercept: bool,
    aliased: Option<Vec<u8>>,
) -> Result<JsValue, JsError> {
    let w = weights.map(|w| to_col(&w, n_rows, "weight")).transpose()?;
    Ok(compute_ridge_variance_factor(
        &to_mat(x, n_rows, n_cols)?,
        w.as_ref(),
        lambda,
        with_intercept,
        &bools(&aliased, n_cols),
    )
    .map_err(fit_error)?
    .to_js())
}

/// Leverage `x₀ᵀ M x₀` of new rows for a variance factor `M` (flat).
#[wasm_bindgen(js_name = leverageNew)]
pub fn leverage_new_js(
    m: &[f64],
    x_new: &[f64],
    n_rows: usize,
    n_cols: usize,
    with_intercept: bool,
) -> Result<Vec<f64>, JsError> {
    let k = n_cols + usize::from(with_intercept);
    Ok(leverage_new(
        &square(m, k, "M")?,
        &to_mat(x_new, n_rows, n_cols)?,
        with_intercept,
    )
    .iter()
    .copied()
    .collect())
}

/// Confidence or prediction intervals from a variance factor `M`:
/// `{ fit, lower, upper, se }`.
#[wasm_bindgen(js_name = intervalsFromVarianceFactor)]
#[allow(clippy::too_many_arguments)]
pub fn intervals_from_variance_factor_js(
    x_new: &[f64],
    n_rows: usize,
    n_cols: usize,
    m: &[f64],
    predictions: &[f64],
    sigma2: f64,
    df: f64,
    level: f64,
    interval: &str,
    with_intercept: bool,
) -> Result<JsValue, JsError> {
    let k = n_cols + usize::from(with_intercept);
    Ok(intervals_from_variance_factor(
        &to_mat(x_new, n_rows, n_cols)?,
        &square(m, k, "M")?,
        &to_col(predictions, n_rows, "prediction")?,
        sigma2,
        df,
        level,
        interval_type(interval)?,
        with_intercept,
    )
    .to_js())
}
