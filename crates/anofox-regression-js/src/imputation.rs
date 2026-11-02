//! Multiple imputation by predictive mean matching
//! (`anofox_regression::imputation`, `Hmisc::aregImpute`, RMS §3.7).
//! Missing values are `NaN` in the row-major data matrix.

use anofox_regression::imputation::{
    areg_impute, AregImpute as RustAregImpute, AregImputeOptions, DonorMatch, ImputeVariable,
};
use serde::Deserialize;
use wasm_bindgen::prelude::*;

use crate::convert::{object, ToJs};
use crate::reduction::{parse, variable_kinds};
use crate::regressors::{fit_error, mat_flat, to_mat};

/// Multiple imputations of the missing values of a data matrix.
#[wasm_bindgen]
pub struct AregImpute {
    inner: RustAregImpute,
    n_cols: usize,
}

#[wasm_bindgen]
impl AregImpute {
    /// `{ missing, imputed, rSquared }`: per column, the 0-based rows that
    /// were missing, their imputations (one array per missing row with one
    /// value per imputation) and the R² of the final prediction model.
    #[wasm_bindgen(js_name = getResult)]
    pub fn get_result(&self) -> JsValue {
        let a = &self.inner;
        object(&[
            ("missing", a.missing.to_js()),
            ("imputed", a.imputed.to_js()),
            ("r_squared", a.r_squared.to_js()),
        ])
    }

    /// Completed data set `m` (0-based imputation) of `x`, row-major.
    #[wasm_bindgen]
    pub fn complete(&self, x: &[f64], n_rows: usize, m: usize) -> Result<Vec<f64>, JsError> {
        let n_impute = self
            .inner
            .imputed
            .iter()
            .flatten()
            .next()
            .map_or(usize::MAX, Vec::len);
        if m >= n_impute {
            return Err(JsError::new(&format!(
                "Imputation {} out of range (have {})",
                m, n_impute
            )));
        }
        Ok(mat_flat(
            &self.inner.complete(&to_mat(x, n_rows, self.n_cols)?, m),
        ))
    }
}

#[derive(Deserialize, Default)]
#[serde(rename_all = "camelCase")]
struct AregImputeOpts {
    n_impute: Option<usize>,
    nk: Option<usize>,
    tlinear: Option<bool>,
    burnin: Option<usize>,
    matching: Option<String>,
    fweighted: Option<f64>,
    seed: Option<i32>,
}

/// `Hmisc::aregImpute`: `kinds` gives "numeric" (spline), "linear" or
/// "categorical" per column; options `{ nImpute? (5), nk? (3), tlinear?,
/// burnin?, matching?: "weighted" | "closest", fweighted? (0.2), seed? }`.
/// With the same seed the imputations replay R's.
#[wasm_bindgen(js_name = aregImpute)]
pub fn areg_impute_js(
    x: &[f64],
    n_rows: usize,
    n_cols: usize,
    kinds: Vec<String>,
    options: JsValue,
) -> Result<AregImpute, JsError> {
    let kinds = variable_kinds(
        kinds,
        ImputeVariable::Numeric,
        ImputeVariable::Linear,
        ImputeVariable::Categorical,
    )?;
    let o: AregImputeOpts = parse(options)?;
    let mut opts = AregImputeOptions::default();
    if let Some(v) = o.n_impute {
        opts = opts.n_impute(v);
    }
    if let Some(v) = o.nk {
        opts = opts.nk(v);
    }
    if let Some(v) = o.tlinear {
        opts = opts.tlinear(v);
    }
    if o.burnin.is_some() {
        opts = opts.burnin(o.burnin);
    }
    match o.matching.as_deref() {
        None | Some("weighted") => {
            opts = opts.matching(DonorMatch::Weighted(o.fweighted.unwrap_or(0.2)))
        }
        Some("closest") => opts = opts.matching(DonorMatch::Closest),
        Some(other) => {
            return Err(JsError::new(&format!(
                "Unknown matching '{}': expected weighted or closest",
                other
            )))
        }
    }
    if let Some(v) = o.seed {
        opts = opts.seed(v);
    }
    Ok(AregImpute {
        inner: areg_impute(&to_mat(x, n_rows, n_cols)?, &kinds, &opts).map_err(fit_error)?,
        n_cols,
    })
}
