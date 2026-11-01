//! Regression model wrappers for WebAssembly.

mod aid;
mod alm;
mod bayesian;
mod binomial;
mod bls;
mod elastic_net;
mod gamma;
mod huber;
mod isotonic;
mod lars;
mod lm_dynamic;
mod logistic;
mod lowess;
mod negative_binomial;
mod ols;
mod passive_aggressive;
mod pls;
mod poisson;
mod quantile;
mod ransac;
mod ridge;
mod rls;
mod theil_sen;
mod tweedie;
mod wls;

pub use aid::*;
pub use alm::*;
pub use bayesian::*;
pub use binomial::*;
pub use bls::*;
pub use elastic_net::*;
pub use gamma::*;
pub use huber::*;
pub use isotonic::*;
pub use lars::*;
pub use lm_dynamic::*;
pub use logistic::*;
pub use lowess::*;
pub use negative_binomial::*;
pub use ols::*;
pub use passive_aggressive::*;
pub use pls::*;
pub use poisson::*;
pub use quantile::*;
pub use ransac::*;
pub use ridge::*;
pub use rls::*;
pub use theil_sen::*;
pub use tweedie::*;
pub use wls::*;

use anofox_regression::core::NaAction;
use faer::Col;
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
