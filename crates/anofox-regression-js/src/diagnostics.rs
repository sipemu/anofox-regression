//! Model diagnostics as free functions (`anofox_regression::diagnostics`).
//!
//! Diagnostics that need a fitted model (`augment`, `press`,
//! `partialResiduals`) are methods on the fitted classes; everything here
//! works on arrays. Matrices are flat row-major arrays with their
//! dimensions, as in the estimators.

use anofox_regression::core::{
    BinomialFamily, BinomialLink, NegativeBinomialFamily, PoissonFamily, PoissonLink, TweedieFamily,
};
use anofox_regression::diagnostics::accuracy::{
    brier_score, concordance, effective_sample_size, gini_md, lrm_stats, nagelkerke_r2,
    r2_measures, rcorr_cens, somers2, val_prob, CensoredConcordance, Concordance, LrmStats,
    R2Measures, Somers2, ValProb,
};
use anofox_regression::diagnostics::{
    added_variable, breusch_pagan, check_binary_separation, check_count_sparsity,
    classify_condition_number, compute_leverage, compute_leverage_with_aliased,
    condition_diagnostic, condition_number, cooks_distance, detect_separation, deviance_residuals,
    dffits, estimate_dispersion_deviance, estimate_dispersion_pearson,
    externally_studentized_residuals, generalized_vif, high_leverage_points, high_vif_predictors,
    influential_cooks, influential_dffits, pearson_chi_squared, pearson_residuals,
    residual_outliers, response_residuals, standardized_deviance_residuals,
    standardized_pearson_residuals, standardized_residuals, studentized_residuals,
    variance_decomposition_proportions, variance_inflation_factor, white_test, working_residuals,
    AddedVariable, Augment, ConditionDiagnostic, ConditionSeverity, HeteroskedasticityTest,
    LinearSeparation, PartialResiduals, Press, ResidualType, SeparationCheck, SeparationType,
};
use faer::Col;
use wasm_bindgen::prelude::*;

use crate::convert::{impl_to_js, impl_to_js_debug, ToJs};
use crate::regressors::{fit_error, to_col, to_mat};

impl_to_js_debug!(ConditionSeverity, SeparationType);

impl ToJs for ResidualType {
    fn to_js(&self) -> JsValue {
        JsValue::from_str(self.as_str())
    }
}

impl_to_js!(Concordance { concordant, discordant, tied_x, tied_y, tied_xy };
    c(), dxy(), gamma(), tau_a(), comparable(), pairs());
impl_to_js!(Somers2 { c, dxy, n });
impl_to_js!(CensoredConcordance {
    c,
    dxy,
    sd,
    n,
    missing,
    uncensored,
    relevant_pairs,
    concordant,
    uncertain
});
impl_to_js!(R2Measures {
    r2,
    r2_adj,
    r2_ess,
    r2_adj_ess
});
impl_to_js!(LrmStats {
    obs,
    sum_weights,
    model_lr,
    df,
    p_value,
    c,
    dxy,
    gamma,
    tau_a,
    r2,
    r2_measures,
    brier,
    g,
    gr,
    gp
});
impl_to_js!(ValProb {
    dxy,
    c,
    r2,
    d,
    d_chisq,
    d_p,
    u,
    u_chisq,
    u_p,
    q,
    brier,
    intercept,
    slope,
    emax,
    e90,
    eavg,
    s_z,
    s_p
});
impl_to_js!(ConditionDiagnostic {
    condition_number,
    condition_number_xtx,
    singular_values,
    condition_indices,
    scaled_condition_number,
    severity,
    warning
});
impl_to_js!(HeteroskedasticityTest {
    statistic,
    df,
    p_value,
    method
});
impl_to_js!(AddedVariable {
    term,
    x_residual,
    y_residual,
    slope
});
impl_to_js!(PartialResiduals {
    term,
    x,
    component,
    partial_residual
});
impl_to_js!(SeparationCheck {
    has_separation,
    separated_predictors,
    separation_types,
    warning_message
});
impl_to_js!(LinearSeparation {
    separated,
    infinite,
    direction
});
impl_to_js!(Augment {
    fitted,
    residual,
    resid_type,
    std_residual,
    stud_residual,
    leverage,
    cooks_d,
    dffits,
    loo_residual,
    loo_fitted,
    approx_loo_fitted,
    approx_loo_residual
});
impl_to_js!(Press {
    press,
    press_rmse,
    loo_r_squared,
    n
});

/// Parse a residual type: "response", "pearson" or "deviance".
pub(crate) fn parse_residual_type(kind: &str) -> Result<ResidualType, JsError> {
    match kind {
        "response" => Ok(ResidualType::Response),
        "pearson" => Ok(ResidualType::Pearson),
        "deviance" => Ok(ResidualType::Deviance),
        other => Err(JsError::new(&format!(
            "Unknown residual type '{}': expected response, pearson or deviance",
            other
        ))),
    }
}

fn opt_slice(v: &Option<Vec<f64>>) -> Option<&[f64]> {
    v.as_deref()
}

fn col(v: &[f64]) -> Col<f64> {
    Col::from_fn(v.len(), |i| v[i])
}

fn vec_of(c: &Col<f64>) -> Vec<f64> {
    c.iter().copied().collect()
}

fn same_len(a: &[f64], b: &[f64], what: &str) -> Result<(), JsError> {
    if a.len() == b.len() {
        Ok(())
    } else {
        Err(JsError::new(&format!(
            "{}: lengths differ ({} vs {})",
            what,
            a.len(),
            b.len()
        )))
    }
}

// ------------------------------------------------------------ accuracy ----

/// Concordance of predictions `x` with outcomes `y` (`Hmisc::rcorr.cens`
/// without censoring): pair counts and C, Dxy, gamma, tau-a.
#[wasm_bindgen(js_name = concordance)]
pub fn concordance_js(x: &[f64], y: &[f64], weights: Option<Vec<f64>>) -> Result<JsValue, JsError> {
    Ok(concordance(x, y, opt_slice(&weights))
        .map_err(fit_error)?
        .to_js())
}

/// `Hmisc::somers2`: C index and Dxy of probabilities `p` for a 0/1 `y`.
#[wasm_bindgen(js_name = somers2)]
pub fn somers2_js(p: &[f64], y: &[f64], weights: Option<Vec<f64>>) -> Result<JsValue, JsError> {
    Ok(somers2(p, y, opt_slice(&weights))
        .map_err(fit_error)?
        .to_js())
}

/// `Hmisc::rcorr.cens`: Harrell's C for predictions `x` of censored
/// survival times (`event[i]` = 1 for an observed event).
#[wasm_bindgen(js_name = rcorrCens)]
pub fn rcorr_cens_js(x: &[f64], time: &[f64], event: &[u8]) -> Result<JsValue, JsError> {
    let event: Vec<bool> = event.iter().map(|&e| e != 0).collect();
    Ok(rcorr_cens(x, time, &event).map_err(fit_error)?.to_js())
}

/// Brier score of probabilities `p` for a 0/1 `y`.
#[wasm_bindgen(js_name = brierScore)]
pub fn brier_score_js(y: &[f64], p: &[f64], weights: Option<Vec<f64>>) -> Result<f64, JsError> {
    brier_score(y, p, opt_slice(&weights)).map_err(fit_error)
}

/// Gini's mean difference (`Hmisc::GiniMd`).
#[wasm_bindgen(js_name = giniMd)]
pub fn gini_md_js(x: &[f64]) -> f64 {
    gini_md(x)
}

/// Nagelkerke's R² from the model LR χ², the null deviance and `n`.
#[wasm_bindgen(js_name = nagelkerkeR2)]
pub fn nagelkerke_r2_js(lr: f64, null_deviance: f64, n: f64) -> f64 {
    nagelkerke_r2(lr, null_deviance, n)
}

/// `Hmisc::R2Measures`: LR-based R² and its adjusted forms for `p`
/// parameters and `n` observations (optionally an effective sample size).
#[wasm_bindgen(js_name = r2Measures)]
pub fn r2_measures_js(lr: f64, p: f64, n: f64, ess: Option<f64>) -> JsValue {
    r2_measures(lr, p, n, ess).to_js()
}

/// Effective sample size of an ordinal response with level frequencies
/// `freqs` (`n − Σ fᵢ³ / n²`).
#[wasm_bindgen(js_name = effectiveSampleSize)]
pub fn effective_sample_size_js(n: f64, freqs: &[f64]) -> f64 {
    effective_sample_size(n, freqs)
}

/// The `rms::lrm` statistics of a 0/1 `y` and linear predictor `lp` on
/// `df` degrees of freedom.
#[wasm_bindgen(js_name = lrmStats)]
pub fn lrm_stats_js(
    y: &[f64],
    lp: &[f64],
    weights: Option<Vec<f64>>,
    df: f64,
) -> Result<JsValue, JsError> {
    Ok(lrm_stats(y, lp, opt_slice(&weights), df)
        .map_err(fit_error)?
        .to_js())
}

/// `rms::val.prob`: validation statistics of probabilities `p` for a 0/1 `y`.
#[wasm_bindgen(js_name = valProb)]
pub fn val_prob_js(p: &[f64], y: &[f64]) -> Result<JsValue, JsError> {
    Ok(val_prob(p, y).map_err(fit_error)?.to_js())
}

// ------------------------------------------------ collinearity, leverage ----

/// Condition number of the (column-scaled) design.
#[wasm_bindgen(js_name = conditionNumber)]
pub fn condition_number_js(
    x: &[f64],
    n_rows: usize,
    n_cols: usize,
    with_intercept: bool,
) -> Result<f64, JsError> {
    Ok(condition_number(
        &to_mat(x, n_rows, n_cols)?,
        with_intercept,
    ))
}

/// Condition number, singular values, condition indices and severity.
#[wasm_bindgen(js_name = conditionDiagnostic)]
pub fn condition_diagnostic_js(
    x: &[f64],
    n_rows: usize,
    n_cols: usize,
    with_intercept: bool,
) -> Result<JsValue, JsError> {
    Ok(condition_diagnostic(&to_mat(x, n_rows, n_cols)?, with_intercept).to_js())
}

/// Severity of a condition number: "well_conditioned", "moderate", "high"
/// or "severe".
#[wasm_bindgen(js_name = classifyConditionNumber)]
pub fn classify_condition_number_js(cond: f64) -> JsValue {
    classify_condition_number(cond).to_js()
}

/// Belsley variance-decomposition proportions: row `j` is coefficient `j`
/// (intercept first), column `k` the component (decreasing singular value).
#[wasm_bindgen(js_name = varianceDecompositionProportions)]
pub fn variance_decomposition_proportions_js(
    x: &[f64],
    n_rows: usize,
    n_cols: usize,
    with_intercept: bool,
) -> Result<JsValue, JsError> {
    Ok(variance_decomposition_proportions(&to_mat(x, n_rows, n_cols)?, with_intercept).to_js())
}

/// Variance inflation factors of the columns of `x`.
#[wasm_bindgen(js_name = varianceInflationFactor)]
pub fn variance_inflation_factor_js(
    x: &[f64],
    n_rows: usize,
    n_cols: usize,
) -> Result<Vec<f64>, JsError> {
    Ok(vec_of(&variance_inflation_factor(&to_mat(
        x, n_rows, n_cols,
    )?)))
}

/// Generalized VIFs (Fox & Monette) of column groups of sizes `group_sizes`.
#[wasm_bindgen(js_name = generalizedVif)]
pub fn generalized_vif_js(
    x: &[f64],
    n_rows: usize,
    n_cols: usize,
    group_sizes: &[u32],
) -> Result<Vec<f64>, JsError> {
    let sizes: Vec<usize> = group_sizes.iter().map(|&s| s as usize).collect();
    Ok(generalized_vif(&to_mat(x, n_rows, n_cols)?, &sizes))
}

/// Indices of the predictors whose VIF exceeds `threshold`.
#[wasm_bindgen(js_name = highVifPredictors)]
pub fn high_vif_predictors_js(vif: &[f64], threshold: f64) -> Vec<usize> {
    high_vif_predictors(&col(vif), threshold)
}

/// Hat values of the design (with an intercept column if requested).
#[wasm_bindgen(js_name = computeLeverage)]
pub fn compute_leverage_js(
    x: &[f64],
    n_rows: usize,
    n_cols: usize,
    with_intercept: bool,
) -> Result<Vec<f64>, JsError> {
    Ok(vec_of(&compute_leverage(
        &to_mat(x, n_rows, n_cols)?,
        with_intercept,
    )))
}

/// Hat values ignoring the aliased columns (`aliased[j]` = 1).
#[wasm_bindgen(js_name = computeLeverageWithAliased)]
pub fn compute_leverage_with_aliased_js(
    x: &[f64],
    n_rows: usize,
    n_cols: usize,
    aliased: &[u8],
    with_intercept: bool,
) -> Result<Vec<f64>, JsError> {
    let aliased: Vec<bool> = aliased.iter().map(|&a| a != 0).collect();
    Ok(vec_of(&compute_leverage_with_aliased(
        &to_mat(x, n_rows, n_cols)?,
        &aliased,
        with_intercept,
    )))
}

/// Rows whose leverage exceeds `threshold` (default `2p/n`).
#[wasm_bindgen(js_name = highLeveragePoints)]
pub fn high_leverage_points_js(
    leverage: &[f64],
    n_params: usize,
    threshold: Option<f64>,
) -> Vec<usize> {
    high_leverage_points(&col(leverage), n_params, threshold)
}

// ------------------------------------------------- residuals, influence ----

/// Residuals divided by `sqrt(mse)`.
#[wasm_bindgen(js_name = standardizedResiduals)]
pub fn standardized_residuals_js(residuals: &[f64], mse: f64) -> Vec<f64> {
    vec_of(&standardized_residuals(&col(residuals), mse))
}

/// Internally studentized residuals `e / sqrt(mse (1 − h))`.
#[wasm_bindgen(js_name = studentizedResiduals)]
pub fn studentized_residuals_js(
    residuals: &[f64],
    leverage: &[f64],
    mse: f64,
) -> Result<Vec<f64>, JsError> {
    same_len(residuals, leverage, "residuals / leverage")?;
    Ok(vec_of(&studentized_residuals(
        &col(residuals),
        &col(leverage),
        mse,
    )))
}

/// Externally studentized (deleted) residuals (`rstudent`).
#[wasm_bindgen(js_name = externallyStudentizedResiduals)]
pub fn externally_studentized_residuals_js(
    residuals: &[f64],
    leverage: &[f64],
    mse: f64,
    n_params: usize,
) -> Result<Vec<f64>, JsError> {
    same_len(residuals, leverage, "residuals / leverage")?;
    Ok(vec_of(&externally_studentized_residuals(
        &col(residuals),
        &col(leverage),
        mse,
        n_params,
    )))
}

/// Rows whose |studentized residual| exceeds `threshold`.
#[wasm_bindgen(js_name = residualOutliers)]
pub fn residual_outliers_js(studentized: &[f64], threshold: f64) -> Vec<usize> {
    residual_outliers(&col(studentized), threshold)
}

/// Cook's distances.
#[wasm_bindgen(js_name = cooksDistance)]
pub fn cooks_distance_js(
    residuals: &[f64],
    leverage: &[f64],
    mse: f64,
    n_params: usize,
) -> Result<Vec<f64>, JsError> {
    same_len(residuals, leverage, "residuals / leverage")?;
    Ok(vec_of(&cooks_distance(
        &col(residuals),
        &col(leverage),
        mse,
        n_params,
    )))
}

/// DFFITS.
#[wasm_bindgen(js_name = dffits)]
pub fn dffits_js(
    residuals: &[f64],
    leverage: &[f64],
    mse: f64,
    n_params: usize,
) -> Result<Vec<f64>, JsError> {
    same_len(residuals, leverage, "residuals / leverage")?;
    Ok(vec_of(&dffits(
        &col(residuals),
        &col(leverage),
        mse,
        n_params,
    )))
}

/// Rows whose Cook's distance exceeds `threshold` (default `4/n`).
#[wasm_bindgen(js_name = influentialCooks)]
pub fn influential_cooks_js(cooks_d: &[f64], threshold: Option<f64>) -> Vec<usize> {
    influential_cooks(&col(cooks_d), threshold)
}

/// Rows whose |DFFITS| exceeds `threshold` (default `2 sqrt(p/n)`).
#[wasm_bindgen(js_name = influentialDffits)]
pub fn influential_dffits_js(
    dffits: &[f64],
    n_params: usize,
    threshold: Option<f64>,
) -> Vec<usize> {
    influential_dffits(&col(dffits), n_params, threshold)
}

// --------------------------------------------------------- GLM residuals ----

/// A GLM family for the residual functions: "poisson" (link "log" |
/// "identity" | "sqrt"), "binomial" (link "logit" | "probit" | "cloglog"),
/// "negative_binomial" (`param` = θ), "gamma" or "tweedie" (`param` = power).
enum Family {
    Poisson(PoissonFamily),
    Binomial(BinomialFamily),
    NegativeBinomial(NegativeBinomialFamily),
    Tweedie(TweedieFamily),
}

fn parse_family(family: &str, link: Option<String>, param: Option<f64>) -> Result<Family, JsError> {
    let bad_link = |l: &str| JsError::new(&format!("Unknown link '{}' for {}", l, family));
    Ok(match family {
        "poisson" => Family::Poisson(PoissonFamily::new(match link.as_deref() {
            None | Some("log") => PoissonLink::Log,
            Some("identity") => PoissonLink::Identity,
            Some("sqrt") => PoissonLink::Sqrt,
            Some(l) => return Err(bad_link(l)),
        })),
        "binomial" => Family::Binomial(BinomialFamily::new(match link.as_deref() {
            None | Some("logit") => BinomialLink::Logit,
            Some("probit") => BinomialLink::Probit,
            Some("cloglog") => BinomialLink::Cloglog,
            Some(l) => return Err(bad_link(l)),
        })),
        "negative_binomial" => Family::NegativeBinomial(NegativeBinomialFamily::new(
            param.ok_or_else(|| JsError::new("The negative_binomial family needs theta"))?,
        )),
        "gamma" => Family::Tweedie(TweedieFamily::gamma()),
        "tweedie" => Family::Tweedie(TweedieFamily::new(
            param.ok_or_else(|| JsError::new("The tweedie family needs a variance power"))?,
            0.0,
        )),
        other => {
            return Err(JsError::new(&format!(
            "Unknown family '{}': expected poisson, binomial, negative_binomial, gamma or tweedie",
            other
        )))
        }
    })
}

/// Call a generic `fn<F: GlmFamily>` with the concrete family.
macro_rules! with_family {
    ($fam:expr, |$f:ident| $body:expr) => {
        match &$fam {
            Family::Poisson($f) => $body,
            Family::Binomial($f) => $body,
            Family::NegativeBinomial($f) => $body,
            Family::Tweedie($f) => $body,
        }
    };
}

fn glm_family(
    family: &str,
    link: Option<String>,
    param: Option<f64>,
    y: &[f64],
    mu: &[f64],
) -> Result<Family, JsError> {
    same_len(y, mu, "y / mu")?;
    parse_family(family, link, param)
}

/// Response residuals `y − μ`.
#[wasm_bindgen(js_name = responseResiduals)]
pub fn response_residuals_js(y: &[f64], mu: &[f64]) -> Result<Vec<f64>, JsError> {
    same_len(y, mu, "y / mu")?;
    Ok(vec_of(&response_residuals(&col(y), &col(mu))))
}

/// Pearson residuals `(y − μ) / sqrt(V(μ))`.
#[wasm_bindgen(js_name = pearsonResiduals)]
pub fn pearson_residuals_js(
    y: &[f64],
    mu: &[f64],
    family: &str,
    link: Option<String>,
    param: Option<f64>,
) -> Result<Vec<f64>, JsError> {
    let fam = glm_family(family, link, param, y, mu)?;
    Ok(with_family!(fam, |f| vec_of(&pearson_residuals(
        &col(y),
        &col(mu),
        f
    ))))
}

/// Deviance residuals `sign(y − μ) sqrt(dᵢ)`.
#[wasm_bindgen(js_name = devianceResiduals)]
pub fn deviance_residuals_js(
    y: &[f64],
    mu: &[f64],
    family: &str,
    link: Option<String>,
    param: Option<f64>,
) -> Result<Vec<f64>, JsError> {
    let fam = glm_family(family, link, param, y, mu)?;
    Ok(with_family!(fam, |f| vec_of(&deviance_residuals(
        &col(y),
        &col(mu),
        f
    ))))
}

/// Working residuals `(y − μ) g'(μ)`.
#[wasm_bindgen(js_name = workingResiduals)]
pub fn working_residuals_js(
    y: &[f64],
    mu: &[f64],
    family: &str,
    link: Option<String>,
    param: Option<f64>,
) -> Result<Vec<f64>, JsError> {
    let fam = glm_family(family, link, param, y, mu)?;
    Ok(with_family!(fam, |f| vec_of(&working_residuals(
        &col(y),
        &col(mu),
        f
    ))))
}

/// Standardized Pearson residuals `r_P / sqrt(φ (1 − h))`.
#[wasm_bindgen(js_name = standardizedPearsonResiduals)]
#[allow(clippy::too_many_arguments)]
pub fn standardized_pearson_residuals_js(
    y: &[f64],
    mu: &[f64],
    leverage: &[f64],
    dispersion: f64,
    family: &str,
    link: Option<String>,
    param: Option<f64>,
) -> Result<Vec<f64>, JsError> {
    let fam = glm_family(family, link, param, y, mu)?;
    same_len(y, leverage, "y / leverage")?;
    Ok(with_family!(fam, |f| vec_of(
        &standardized_pearson_residuals(&col(y), &col(mu), f, &col(leverage), dispersion)
    )))
}

/// Standardized deviance residuals `r_D / sqrt(φ (1 − h))` (`rstandard`).
#[wasm_bindgen(js_name = standardizedDevianceResiduals)]
#[allow(clippy::too_many_arguments)]
pub fn standardized_deviance_residuals_js(
    y: &[f64],
    mu: &[f64],
    leverage: &[f64],
    dispersion: f64,
    family: &str,
    link: Option<String>,
    param: Option<f64>,
) -> Result<Vec<f64>, JsError> {
    let fam = glm_family(family, link, param, y, mu)?;
    same_len(y, leverage, "y / leverage")?;
    Ok(with_family!(fam, |f| vec_of(
        &standardized_deviance_residuals(&col(y), &col(mu), f, &col(leverage), dispersion)
    )))
}

/// Pearson χ² `Σ (y − μ)² / V(μ)`.
#[wasm_bindgen(js_name = pearsonChiSquared)]
pub fn pearson_chi_squared_js(
    y: &[f64],
    mu: &[f64],
    family: &str,
    link: Option<String>,
    param: Option<f64>,
) -> Result<f64, JsError> {
    let fam = glm_family(family, link, param, y, mu)?;
    Ok(with_family!(fam, |f| pearson_chi_squared(y, mu, f)))
}

/// Pearson dispersion estimate `χ²_P / (n − p)`.
#[wasm_bindgen(js_name = estimateDispersionPearson)]
pub fn estimate_dispersion_pearson_js(
    y: &[f64],
    mu: &[f64],
    n_params: usize,
    family: &str,
    link: Option<String>,
    param: Option<f64>,
) -> Result<f64, JsError> {
    let fam = glm_family(family, link, param, y, mu)?;
    Ok(with_family!(fam, |f| estimate_dispersion_pearson(
        y, mu, f, n_params
    )))
}

/// Deviance dispersion estimate `D / (n − p)`.
#[wasm_bindgen(js_name = estimateDispersionDeviance)]
pub fn estimate_dispersion_deviance_js(
    y: &[f64],
    mu: &[f64],
    n_params: usize,
    family: &str,
    link: Option<String>,
    param: Option<f64>,
) -> Result<f64, JsError> {
    let fam = glm_family(family, link, param, y, mu)?;
    Ok(with_family!(fam, |f| estimate_dispersion_deviance(
        y, mu, f, n_params
    )))
}

// ------------------------------------------ heteroskedasticity, others ----

/// Breusch–Pagan test of `residuals` on the regressors `z` (Koenker's
/// studentized form with `studentize`).
#[wasm_bindgen(js_name = breuschPagan)]
pub fn breusch_pagan_js(
    residuals: &[f64],
    z: &[f64],
    n_cols: usize,
    studentize: bool,
) -> Result<JsValue, JsError> {
    let n = residuals.len();
    Ok(
        breusch_pagan(&col(residuals), &to_mat(z, n, n_cols)?, studentize)
            .map_err(fit_error)?
            .to_js(),
    )
}

/// White's test of `residuals` on the regressors `x`.
#[wasm_bindgen(js_name = whiteTest)]
pub fn white_test_js(residuals: &[f64], x: &[f64], n_cols: usize) -> Result<JsValue, JsError> {
    let n = residuals.len();
    Ok(white_test(&col(residuals), &to_mat(x, n, n_cols)?)
        .map_err(fit_error)?
        .to_js())
}

/// Added-variable (partial regression) data for every column of `x`:
/// `[{ term, xResidual, yResidual, slope }]`.
#[wasm_bindgen(js_name = addedVariable)]
pub fn added_variable_js(
    x: &[f64],
    n_rows: usize,
    n_cols: usize,
    y: &[f64],
    weights: Option<Vec<f64>>,
    with_intercept: bool,
) -> Result<JsValue, JsError> {
    let x_mat = to_mat(x, n_rows, n_cols)?;
    let y_col = to_col(y, n_rows, "target")?;
    let w = weights.map(|w| to_col(&w, n_rows, "weight")).transpose()?;
    Ok(added_variable(&x_mat, &y_col, w.as_ref(), with_intercept)
        .map_err(fit_error)?
        .to_js())
}

/// Per-predictor check for complete / quasi separation of a 0/1 `y`.
#[wasm_bindgen(js_name = checkBinarySeparation)]
pub fn check_binary_separation_js(
    x: &[f64],
    n_rows: usize,
    n_cols: usize,
    y: &[f64],
) -> Result<JsValue, JsError> {
    let x_mat = to_mat(x, n_rows, n_cols)?;
    Ok(check_binary_separation(&x_mat, &to_col(y, n_rows, "target")?).to_js())
}

/// Linear-programming separation check (`detectseparation`):
/// `{ separated, infinite, direction }`.
#[wasm_bindgen(js_name = detectSeparation)]
pub fn detect_separation_js(
    x: &[f64],
    n_rows: usize,
    n_cols: usize,
    y: &[f64],
    with_intercept: bool,
) -> Result<JsValue, JsError> {
    let x_mat = to_mat(x, n_rows, n_cols)?;
    Ok(detect_separation(&x_mat, &to_col(y, n_rows, "target")?, with_intercept).to_js())
}

/// Per-predictor check for all-zero count cells.
#[wasm_bindgen(js_name = checkCountSparsity)]
pub fn check_count_sparsity_js(
    x: &[f64],
    n_rows: usize,
    n_cols: usize,
    y: &[f64],
) -> Result<JsValue, JsError> {
    let x_mat = to_mat(x, n_rows, n_cols)?;
    Ok(check_count_sparsity(&x_mat, &to_col(y, n_rows, "target")?).to_js())
}
