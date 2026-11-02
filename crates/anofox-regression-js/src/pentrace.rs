//! Penalized maximum likelihood à la rms (`solvers::pentrace`, RMS §9.10):
//! penalty matrices with differential penalties for nonlinear and
//! interaction terms, effective d.f., penalized least squares and the
//! `pentrace` search.
//!
//! A penalty is a number (the same λ for every column) or `{ simple,
//! nonlinear?, interaction?, nonlinearInteraction? }`; unset components
//! fall back as in rms (interaction ← nonlinear ← simple). Penalty
//! matrices are row-major `p × p` over the slope columns of the term map.

use anofox_regression::solvers::{
    base_penalty_matrix, effective_df_by_term, penalized_ols as rust_penalized_ols, penalty_grid,
    penalty_matrix, penalty_multipliers, pentrace as rust_pentrace, pentrace_ols, PenalizedOls,
    PenaltySpec, Pentrace as RustPentrace, PentraceCriterion, PentraceRow,
};
use faer::{Col, Mat};
use serde::Deserialize;
use wasm_bindgen::prelude::*;

use crate::convert::{impl_to_js, object, ToJs};
use crate::design::TermMap;
use crate::inference::square;
use crate::regressors::{fit_error, mat_flat, parse_link, to_col, to_mat, FittedOrdinal};
use crate::survival::usizes;

impl_to_js!(PentraceRow {
    penalty,
    df,
    lr_chisq,
    aic,
    bic,
    aic_c
});
impl_to_js!(PenalizedOls {
    coefficients,
    vcov,
    vcov_sandwich,
    residuals,
    sigma,
    r_squared,
    lr_chisq,
    effective_df_diagonal,
    effective_df
});

impl ToJs for PenaltySpec {
    fn to_js(&self) -> JsValue {
        object(&[
            ("simple", self.simple.to_js()),
            ("nonlinear", self.nonlinear.to_js()),
            ("interaction", self.interaction.to_js()),
            ("nonlinear_interaction", self.nonlinear_interaction.to_js()),
        ])
    }
}

#[derive(Deserialize)]
#[serde(untagged)]
enum SpecJs {
    Uniform(f64),
    #[serde(rename_all = "camelCase")]
    Parts {
        simple: Option<f64>,
        nonlinear: Option<f64>,
        interaction: Option<f64>,
        nonlinear_interaction: Option<f64>,
    },
}

impl From<SpecJs> for PenaltySpec {
    fn from(s: SpecJs) -> Self {
        match s {
            SpecJs::Uniform(l) => PenaltySpec::uniform(l),
            SpecJs::Parts {
                simple,
                nonlinear,
                interaction,
                nonlinear_interaction,
            } => PenaltySpec::default()
                .simple(simple.unwrap_or(0.0))
                .nonlinear(nonlinear)
                .interaction(interaction)
                .nonlinear_interaction(nonlinear_interaction),
        }
    }
}

fn spec(v: JsValue) -> Result<PenaltySpec, JsError> {
    let s: SpecJs = serde_wasm_bindgen::from_value(v).map_err(|e| {
        JsError::new(&format!(
            "penalty: expected a number or {{ simple, ... }} ({})",
            e
        ))
    })?;
    Ok(s.into())
}

fn specs(v: JsValue) -> Result<Vec<PenaltySpec>, JsError> {
    let s: Vec<SpecJs> = serde_wasm_bindgen::from_value(v).map_err(|e| {
        JsError::new(&format!(
            "penalties: expected an array of penalties ({})",
            e
        ))
    })?;
    Ok(s.into_iter().map(Into::into).collect())
}

fn parse_criterion(v: Option<&str>) -> Result<PentraceCriterion, JsError> {
    match v {
        None | Some("aic.c") | Some("aicc") => Ok(PentraceCriterion::CorrectedAic),
        Some("aic") => Ok(PentraceCriterion::Aic),
        Some("bic") => Ok(PentraceCriterion::Bic),
        Some(other) => Err(JsError::new(&format!(
            "Unknown criterion '{}': expected aic.c, aic or bic",
            other
        ))),
    }
}

fn check_terms(terms: &TermMap, n_cols: usize) -> Result<(), JsError> {
    if terms.inner.n_columns() != n_cols {
        return Err(JsError::new(&format!(
            "The term map has {} columns, x has {}",
            terms.inner.n_columns(),
            n_cols
        )));
    }
    Ok(())
}

/// The four penalties after the rms fallbacks: `[simple, nonlinear,
/// interaction, nonlinearInteraction]`.
#[wasm_bindgen(js_name = resolvePenalty)]
pub fn resolve_penalty(penalty: JsValue) -> Result<Vec<f64>, JsError> {
    Ok(spec(penalty)?.resolved().to_vec())
}

/// λ of every slope column of the term map under `penalty`.
#[wasm_bindgen(js_name = penaltyMultipliers)]
pub fn penalty_multipliers_js(terms: &TermMap, penalty: JsValue) -> Result<Vec<f64>, JsError> {
    Ok(penalty_multipliers(&terms.inner, &spec(penalty)?))
}

/// rms's unscaled penalty matrix `P₀` (column variances on the diagonal;
/// `I − 1/(L + 1)` blocks for the categorical terms, by term index).
#[wasm_bindgen(js_name = basePenaltyMatrix)]
pub fn base_penalty_matrix_js(
    x: &[f64],
    n_rows: usize,
    n_cols: usize,
    terms: &TermMap,
    categorical: &[u32],
) -> Result<Vec<f64>, JsError> {
    check_terms(terms, n_cols)?;
    Ok(mat_flat(
        &base_penalty_matrix(
            &to_mat(x, n_rows, n_cols)?,
            &terms.inner,
            &usizes(categorical),
        )
        .map_err(fit_error)?,
    ))
}

/// The penalty matrix `A P₀ A` of `penalty` (pass it to
/// `OrdinalRegressor.setPenaltyMatrix` or `penalizedOls`).
#[wasm_bindgen(js_name = penaltyMatrix)]
pub fn penalty_matrix_js(
    x: &[f64],
    n_rows: usize,
    n_cols: usize,
    terms: &TermMap,
    categorical: &[u32],
    penalty: JsValue,
) -> Result<Vec<f64>, JsError> {
    check_terms(terms, n_cols)?;
    Ok(mat_flat(
        &penalty_matrix(
            &to_mat(x, n_rows, n_cols)?,
            &terms.inner,
            &usizes(categorical),
            &spec(penalty)?,
        )
        .map_err(fit_error)?,
    ))
}

/// Effective d.f. per term from the per-column effective d.f.
#[wasm_bindgen(js_name = effectiveDfByTerm)]
pub fn effective_df_by_term_js(terms: &TermMap, diagonal: &[f64]) -> Result<Vec<f64>, JsError> {
    if diagonal.len() != terms.inner.n_columns() {
        return Err(JsError::new(&format!(
            "Expected {} values, got {}",
            terms.inner.n_columns(),
            diagonal.len()
        )));
    }
    Ok(effective_df_by_term(
        &terms.inner,
        &Col::from_fn(diagonal.len(), |i| diagonal[i]),
    ))
}

/// Grid of penalties (`expand.grid` with rms's `complex.more = TRUE`
/// rule); an empty array leaves that component unset.
#[wasm_bindgen(js_name = penaltyGrid)]
pub fn penalty_grid_js(
    simple: &[f64],
    nonlinear: &[f64],
    interaction: &[f64],
    nonlinear_interaction: &[f64],
) -> JsValue {
    penalty_grid(simple, nonlinear, interaction, nonlinear_interaction).to_js()
}

/// Penalized least squares (`rms::ols(penalty.matrix = )`) with the
/// row-major `p × p` penalty on the slopes: `{ coefficients (intercept
/// first), vcov, vcovSandwich, residuals, sigma, rSquared, lrChisq,
/// effectiveDfDiagonal, effectiveDf }`.
#[wasm_bindgen(js_name = penalizedOls)]
pub fn penalized_ols_js(
    x: &[f64],
    n_rows: usize,
    n_cols: usize,
    y: &[f64],
    penalty: &[f64],
) -> Result<JsValue, JsError> {
    let x = to_mat(x, n_rows, n_cols)?;
    let y = to_col(y, n_rows, "y")?;
    let pm: Mat<f64> = square(penalty, n_cols, "penalty")?;
    Ok(rust_penalized_ols(&x, &y, &pm).map_err(fit_error)?.to_js())
}

/// The trace of a penalized ordinal / binary logistic model
/// (`rms::pentrace`).
#[wasm_bindgen]
pub struct Pentrace {
    inner: RustPentrace,
    n_features: usize,
}

#[wasm_bindgen]
impl Pentrace {
    /// `{ rows: [{ penalty, df, lrChisq, aic, bic, aicC }], best }`.
    #[wasm_bindgen(js_name = getResult)]
    pub fn get_result(&self) -> JsValue {
        object(&[
            ("rows", self.inner.rows.to_js()),
            ("best", self.inner.best.to_js()),
        ])
    }

    /// The fit at the best penalty.
    #[wasm_bindgen(js_name = bestFit)]
    pub fn best_fit(&self) -> FittedOrdinal {
        FittedOrdinal::from_rust(self.inner.best_fit.clone(), self.n_features)
    }
}

/// `rms::pentrace` of a cumulative link model of `y` on `x` (slope columns
/// described by `terms`, categorical terms `categorical`) over `penalties`;
/// `link` as in `OrdinalRegressor` (default "logistic"), `criterion`
/// "aic.c" (default), "aic" or "bic".
#[allow(clippy::too_many_arguments)]
#[wasm_bindgen]
pub fn pentrace(
    x: &[f64],
    n_rows: usize,
    n_cols: usize,
    y: &[f64],
    terms: &TermMap,
    categorical: &[u32],
    penalties: JsValue,
    link: Option<String>,
    criterion: Option<String>,
) -> Result<Pentrace, JsError> {
    check_terms(terms, n_cols)?;
    let inner = rust_pentrace(
        &to_mat(x, n_rows, n_cols)?,
        &to_col(y, n_rows, "y")?,
        &terms.inner,
        &usizes(categorical),
        &specs(penalties)?,
        parse_link(link.as_deref().unwrap_or("logistic"))?,
        parse_criterion(criterion.as_deref())?,
    )
    .map_err(fit_error)?;
    Ok(Pentrace {
        inner,
        n_features: n_cols,
    })
}

/// `pentrace` for least squares: `{ rows, best, bestFit }` with `bestFit`
/// as `penalizedOls`.
#[allow(clippy::too_many_arguments)]
#[wasm_bindgen(js_name = pentraceOls)]
pub fn pentrace_ols_js(
    x: &[f64],
    n_rows: usize,
    n_cols: usize,
    y: &[f64],
    terms: &TermMap,
    categorical: &[u32],
    penalties: JsValue,
    criterion: Option<String>,
) -> Result<JsValue, JsError> {
    check_terms(terms, n_cols)?;
    let t = pentrace_ols(
        &to_mat(x, n_rows, n_cols)?,
        &to_col(y, n_rows, "y")?,
        &terms.inner,
        &usizes(categorical),
        &specs(penalties)?,
        parse_criterion(criterion.as_deref())?,
    )
    .map_err(fit_error)?;
    Ok(object(&[
        ("rows", t.rows.to_js()),
        ("best", t.best.to_js()),
        ("best_fit", t.best_fit.to_js()),
    ]))
}
