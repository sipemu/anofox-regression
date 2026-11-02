//! Regression model wrappers for WebAssembly.

/// Expose the Rust `IterativeFit` trait (`converged`, `iterations`) on a
/// fitted wrapper whose inner model is in the `fitted` field. Invoked in each
/// wrapper's module, since that field is private.
macro_rules! iterative_fit {
    ($($wrapper:ident),* $(,)?) => {$(
        #[wasm_bindgen]
        impl $wrapper {
            /// Whether the fitting algorithm met its stopping rule.
            #[wasm_bindgen(js_name = getConverged)]
            pub fn get_converged(&self) -> bool {
                anofox_regression::solvers::IterativeFit::converged(&self.fitted)
            }

            /// Iterations (or passes, epochs) the fitting algorithm used.
            #[wasm_bindgen(js_name = getIterations)]
            pub fn get_iterations(&self) -> usize {
                anofox_regression::solvers::IterativeFit::iterations(&self.fitted)
            }
        }
    )*};
}

mod aft;
mod aid;
mod alm;
mod bayesian;
mod binomial;
mod bls;
mod continuation_ratio;
mod elastic_net;
mod gamma;
mod gls;
mod huber;
mod isotonic;
mod lars;
mod lm_dynamic;
mod logistic;
mod lowess;
mod negative_binomial;
mod ols;
mod ordinal;
mod passive_aggressive;
mod penalized_glm;
mod pls;
mod poisson;
mod pspline;
mod quantile;
mod ransac;
mod ridge;
mod rls;
mod theil_sen;
mod tweedie;
mod wls;

pub use aft::*;
pub use aid::*;
pub use alm::*;
pub use bayesian::*;
pub use binomial::*;
pub use bls::*;
pub use continuation_ratio::*;
pub use elastic_net::*;
pub use gamma::*;
pub use gls::*;
pub use huber::*;
pub use isotonic::*;
pub use lars::*;
pub use lm_dynamic::*;
pub use logistic::*;
pub use lowess::*;
pub use negative_binomial::*;
pub use ols::*;
pub use ordinal::*;
pub use passive_aggressive::*;
pub use penalized_glm::*;
pub use pls::*;
pub use poisson::*;
pub use pspline::*;
pub use quantile::*;
pub use ransac::*;
pub use ridge::*;
pub use rls::*;
pub use theil_sen::*;
pub use tweedie::*;
pub use wls::*;

use anofox_regression::core::NaAction;
use anofox_regression::solvers::penalized_glm::PriorSpec;
use faer::{Col, ColRef, Mat};
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

/// Parse an R `na.action` name: "omit", "exclude", "fail" or "pass".
pub(crate) fn parse_na_action(action: &str) -> Result<NaAction, JsError> {
    match action {
        "omit" => Ok(NaAction::Omit),
        "exclude" => Ok(NaAction::Exclude),
        "fail" => Ok(NaAction::Fail),
        "pass" => Ok(NaAction::Pass),
        other => Err(JsError::new(&format!(
            "Unknown na action '{}': expected omit, exclude, fail or pass",
            other
        ))),
    }
}

/// A row-aligned vector (weights, offset) as a column, checked against the
/// number of rows.
pub(crate) fn row_column(
    v: &Option<Vec<f64>>,
    n_rows: usize,
    what: &str,
) -> Result<Option<Col<f64>>, JsError> {
    match v {
        Some(v) if v.len() != n_rows => Err(JsError::new(&format!(
            "Expected {} {} values, got {}",
            n_rows,
            what,
            v.len()
        ))),
        Some(v) => Ok(Some(Col::from_fn(n_rows, |i| v[i]))),
        None => Ok(None),
    }
}

/// A row-major `n_rows x n_cols` matrix from a flat JS array.
pub(crate) fn to_mat(x: &[f64], n_rows: usize, n_cols: usize) -> Result<Mat<f64>, JsError> {
    if x.len() != n_rows * n_cols {
        return Err(JsError::new(&format!(
            "Expected {} elements for {}x{} matrix, got {}",
            n_rows * n_cols,
            n_rows,
            n_cols,
            x.len()
        )));
    }
    Ok(Mat::from_fn(n_rows, n_cols, |i, j| x[i * n_cols + j]))
}

/// A column from a JS array that must have one value per row.
pub(crate) fn to_col(v: &[f64], n_rows: usize, what: &str) -> Result<Col<f64>, JsError> {
    if v.len() != n_rows {
        return Err(JsError::new(&format!(
            "Expected {} {} values, got {}",
            n_rows,
            what,
            v.len()
        )));
    }
    Ok(Col::from_fn(n_rows, |i| v[i]))
}

/// A faer column as a JS array.
pub(crate) fn col_vec(c: ColRef<'_, f64>) -> Vec<f64> {
    c.iter().copied().collect()
}

/// A faer matrix as a row-major flat JS array.
pub(crate) fn mat_flat(m: &Mat<f64>) -> Vec<f64> {
    (0..m.nrows())
        .flat_map(|i| (0..m.ncols()).map(move |j| m[(i, j)]))
        .collect()
}

/// A faer matrix as nested rows (for serde results).
pub(crate) fn mat_rows(m: &Mat<f64>) -> Vec<Vec<f64>> {
    (0..m.nrows())
        .map(|i| (0..m.ncols()).map(|j| m[(i, j)]).collect())
        .collect()
}

/// Serialize a result struct to a JS object.
pub(crate) fn to_js<T: Serialize>(value: &T) -> Result<JsValue, JsError> {
    serde_wasm_bindgen::to_value(value).map_err(|e| JsError::new(&e.to_string()))
}

/// Map a solver error to a JS error.
pub(crate) fn fit_error(e: impl std::fmt::Display) -> JsError {
    JsError::new(&format!("Regression failed: {}", e))
}

/// A coefficient prior as passed from JS:
/// `{ kind: "flat" | "normal" | "laplace", loc?: number, scale?: number }`.
#[derive(Serialize, Deserialize)]
pub(crate) struct JsPrior {
    kind: String,
    #[serde(default)]
    loc: f64,
    #[serde(default)]
    scale: Option<f64>,
}

/// Parse a JS array of priors (see [`JsPrior`]).
pub(crate) fn parse_priors(priors: JsValue) -> Result<Vec<PriorSpec>, JsError> {
    let priors: Vec<JsPrior> =
        serde_wasm_bindgen::from_value(priors).map_err(|e| JsError::new(&e.to_string()))?;
    priors
        .into_iter()
        .map(|p| match (p.kind.as_str(), p.scale) {
            ("flat", _) => Ok(PriorSpec::flat()),
            ("normal", Some(scale)) => Ok(PriorSpec::normal(p.loc, scale)),
            ("laplace", Some(scale)) => Ok(PriorSpec::laplace(p.loc, scale)),
            ("normal" | "laplace", None) => {
                Err(JsError::new(&format!("A {} prior needs a scale", p.kind)))
            }
            (other, _) => Err(JsError::new(&format!(
                "Unknown prior kind '{}': expected flat, normal or laplace",
                other
            ))),
        })
        .collect()
}

/// Parse a covariance type for penalized/MAP fits: "laplace", "sandwich"
/// or "naive".
pub(crate) fn parse_vcov_type(
    vcov: &str,
) -> Result<anofox_regression::solvers::penalized_glm::VcovType, JsError> {
    use anofox_regression::solvers::penalized_glm::VcovType;
    match vcov {
        "laplace" => Ok(VcovType::Laplace),
        "sandwich" => Ok(VcovType::Sandwich),
        "naive" => Ok(VcovType::Naive),
        other => Err(JsError::new(&format!(
            "Unknown vcov type '{}': expected laplace, sandwich or naive",
            other
        ))),
    }
}
