//! Causal mediation (`solvers::mediation`, R's `mediation::mediate`) and
//! empirical-Bayes shrinkage of group estimates (`solvers::eb_shrink`,
//! `metafor::rma(method = "DL")` + `blup`).

use anofox_regression::solvers::eb_shrink::{
    eb_shrink, EbShrinkOptions, EbShrinkResult, ShrunkenGroup, TauMethod,
};
use anofox_regression::solvers::mediation::{
    mediate as rust_mediate, BootCi, MediationEffect, MediationMethod, MediationOptions,
    MediationResult,
};
use serde::Deserialize;
use wasm_bindgen::prelude::*;

use crate::convert::{impl_to_js, impl_to_js_debug, ToJs};
use crate::reduction::parse;
use crate::regressors::fit_error;

impl_to_js_debug!(MediationMethod, BootCi);
impl_to_js!(MediationEffect {
    estimate,
    std_error,
    statistic,
    p_value,
    conf_low,
    conf_high
});
impl_to_js!(MediationResult {
    acme,
    ade,
    total,
    prop_mediated,
    a,
    b,
    c_prime,
    r_squared_mediator,
    r_squared_outcome,
    n_obs,
    n_sims,
    method,
    boot_ci,
    conf_level
});
impl_to_js!(ShrunkenGroup {
    estimate,
    se,
    shrunken,
    shrunken_se,
    weight
});
impl_to_js!(EbShrinkResult {
    mu,
    mu_se,
    tau_squared,
    i_squared,
    q,
    n_groups,
    groups
});

fn parse_boot_ci(v: Option<&str>) -> Result<BootCi, JsError> {
    match v {
        None | Some("percentile") => Ok(BootCi::Percentile),
        Some("bca") => Ok(BootCi::Bca),
        Some(other) => Err(JsError::new(&format!(
            "Unknown bootCi '{}': expected percentile or bca",
            other
        ))),
    }
}

#[derive(Deserialize, Default)]
#[serde(rename_all = "camelCase")]
struct MediationOpts {
    method: Option<String>,
    boot_ci: Option<String>,
    n_sims: Option<usize>,
    seed: Option<f64>,
    conf_level: Option<f64>,
    treat_value: Option<f64>,
    control_value: Option<f64>,
    covariates: Option<Vec<Vec<f64>>>,
}

/// Causal mediation of the effect of `x` on `y` through `m` (linear
/// mediator and outcome models, `mediation::mediate`). Options `{ method?:
/// "bootstrap" | "sobel" | "quasi-bayesian", bootCi?: "percentile" | "bca",
/// nSims? (1000), seed? (42), confLevel?, treatValue? (1), controlValue?
/// (0), covariates?: number[][] }`. Returns `{ acme, ade, total,
/// propMediated, a, b, cPrime, rSquaredMediator, rSquaredOutcome, nObs,
/// nSims, method, bootCi, confLevel }`, each effect `{ estimate, stdError,
/// statistic, pValue, confLow, confHigh }`.
#[wasm_bindgen]
pub fn mediate(y: &[f64], m: &[f64], x: &[f64], options: JsValue) -> Result<JsValue, JsError> {
    let o: MediationOpts = parse(options)?;
    let mut opts = MediationOptions::default();
    if let Some(method) = o.method.as_deref() {
        opts = opts.method(match method {
            "bootstrap" => MediationMethod::Bootstrap,
            "sobel" => MediationMethod::Sobel,
            "quasi-bayesian" => MediationMethod::QuasiBayesian,
            other => {
                return Err(JsError::new(&format!(
                    "Unknown method '{}': expected bootstrap, sobel or quasi-bayesian",
                    other
                )))
            }
        });
    }
    opts = opts.boot_ci(parse_boot_ci(o.boot_ci.as_deref())?);
    if let Some(v) = o.n_sims {
        opts = opts.n_sims(v);
    }
    if let Some(s) = o.seed {
        if !(s.is_finite() && s >= 0.0 && s.fract() == 0.0) {
            return Err(JsError::new("seed must be a non-negative integer"));
        }
        opts = opts.seed(s as u64);
    }
    if let Some(v) = o.conf_level {
        opts = opts.conf_level(v);
    }
    if let Some(v) = o.treat_value {
        opts = opts.treat_value(v);
    }
    if let Some(v) = o.control_value {
        opts = opts.control_value(v);
    }
    let covs = o.covariates.unwrap_or_default();
    Ok(rust_mediate(y, m, x, &covs, &opts)
        .map_err(fit_error)?
        .to_js())
}

/// Summary of resampling draws of an effect, as `mediate` reports it:
/// `{ estimate, stdError, statistic, pValue, confLow, confHigh }`; `ci` is
/// "percentile" (default) or "bca".
#[wasm_bindgen(js_name = mediationEffectFromDraws)]
pub fn mediation_effect_from_draws(
    estimate: f64,
    draws: &[f64],
    conf_level: f64,
    ci: Option<String>,
) -> Result<JsValue, JsError> {
    if draws.is_empty() {
        return Err(JsError::new("draws must not be empty"));
    }
    Ok(
        MediationEffect::from_draws(estimate, draws, conf_level, parse_boot_ci(ci.as_deref())?)
            .to_js(),
    )
}

#[derive(Deserialize, Default)]
#[serde(rename_all = "camelCase")]
struct EbOpts {
    method: Option<String>,
    tau_squared: Option<f64>,
}

/// Empirical-Bayes shrinkage of group estimates with standard errors:
/// options `{ method?: "dl" (DerSimonian–Laird, default) | "none" (complete
/// pooling), tauSquared? (fixed) }`. Returns `{ mu, muSe, tauSquared,
/// iSquared, q, nGroups, groups: [{ estimate, se, shrunken, shrunkenSe,
/// weight }] }`.
#[wasm_bindgen(js_name = ebShrink)]
pub fn eb_shrink_js(
    estimates: &[f64],
    standard_errors: &[f64],
    options: JsValue,
) -> Result<JsValue, JsError> {
    let o: EbOpts = parse(options)?;
    let mut opts = EbShrinkOptions::default();
    if let Some(m) = o.method.as_deref() {
        opts = opts.method(match m {
            "dl" => TauMethod::DerSimonianLaird,
            "none" => TauMethod::None,
            other => {
                return Err(JsError::new(&format!(
                    "Unknown method '{}': expected dl or none",
                    other
                )))
            }
        });
    }
    if o.tau_squared.is_some() {
        opts = opts.tau_squared(o.tau_squared);
    }
    Ok(eb_shrink(estimates, standard_errors, &opts)
        .map_err(fit_error)?
        .to_js())
}
