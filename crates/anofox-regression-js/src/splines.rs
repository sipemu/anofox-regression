//! Spline bases (`anofox_regression::splines`): restricted cubic splines
//! (`Hmisc::rcspline.eval`, `rms::rcs`), linear splines (`rms::lsp`) and
//! restricted interactions (`rms::%ia%`) and B-splines (sklearn's
//! `SplineTransformer`).
//!
//! Basis matrices are flat row-major arrays (`nRows × nColumns`), ready to
//! pass to a regressor's `fit`.

use anofox_regression::splines::{
    default_knot_quantiles, rcs_default_knots, rcs_default_knots_with, restricted_interaction,
    BSpline as RustBSpline, Basis as RustBasis, KnotPlacement, Lsp as RustLsp, Rcs as RustRcs,
    RcsNorm, RcsRestated as RustRcsRestated, SplineExtrapolation,
};
use wasm_bindgen::prelude::*;

use crate::convert::{object, ToJs};
use crate::regressors::{fit_error, mat_flat, to_mat};

/// Columns of a basis with their names and nonlinearity flags.
#[wasm_bindgen]
#[derive(Clone)]
pub struct Basis {
    pub(crate) inner: RustBasis,
}

#[wasm_bindgen]
impl Basis {
    /// Basis from a row-major `nRows × names.length` matrix, column names
    /// and nonlinearity flags (1 = nonlinear), e.g. dummy columns of a
    /// factor.
    #[wasm_bindgen(constructor)]
    pub fn new(
        columns: &[f64],
        n_rows: usize,
        names: Vec<String>,
        nonlinear: &[u8],
    ) -> Result<Basis, JsError> {
        let m = to_mat(columns, n_rows, names.len())?;
        let nonlinear = nonlinear.iter().map(|&v| v != 0).collect();
        Ok(Self {
            inner: RustBasis::new(m, names, nonlinear).map_err(fit_error)?,
        })
    }

    /// A single linear column.
    #[wasm_bindgen]
    pub fn linear(x: &[f64], name: &str) -> Basis {
        Self {
            inner: RustBasis::linear(x, name),
        }
    }

    /// The basis matrix, row-major.
    #[wasm_bindgen]
    pub fn columns(&self) -> Vec<f64> {
        mat_flat(&self.inner.columns)
    }

    /// Number of rows.
    #[wasm_bindgen(js_name = nRows)]
    pub fn n_rows(&self) -> usize {
        self.inner.columns.nrows()
    }

    /// Number of columns.
    #[wasm_bindgen(js_name = nColumns)]
    pub fn n_columns(&self) -> usize {
        self.inner.n_columns()
    }

    /// Column names, rms style (`x`, `x'`, `x''`, …; `a * b` for products).
    #[wasm_bindgen]
    pub fn names(&self) -> Vec<String> {
        self.inner.names.clone()
    }

    /// Nonlinearity flag of every column.
    #[wasm_bindgen]
    pub fn nonlinear(&self) -> Vec<u8> {
        self.inner.nonlinear.iter().map(|&b| u8::from(b)).collect()
    }

    /// Indices of the nonlinear columns (the chunk tested for linearity).
    #[wasm_bindgen(js_name = nonlinearColumns)]
    pub fn nonlinear_columns(&self) -> Vec<usize> {
        self.inner.nonlinear_columns()
    }
}

fn parse_norm(norm: &str) -> Result<RcsNorm, JsError> {
    match norm.to_ascii_lowercase().as_str() {
        "none" | "0" => Ok(RcsNorm::None),
        "last-interval" | "1" => Ok(RcsNorm::LastInterval),
        "range" | "2" => Ok(RcsNorm::Range),
        other => Err(JsError::new(&format!(
            "Unknown norm '{}': expected none, last-interval or range",
            other
        ))),
    }
}

/// Restricted cubic spline basis with fixed knots.
#[wasm_bindgen]
#[derive(Clone)]
pub struct Rcs {
    pub(crate) inner: RustRcs,
}

#[wasm_bindgen]
impl Rcs {
    /// Spline with the given knots (sorted and de-duplicated; at least 3).
    #[wasm_bindgen(constructor)]
    pub fn new(knots: &[f64]) -> Result<Rcs, JsError> {
        Ok(Self {
            inner: RustRcs::new(knots).map_err(fit_error)?,
        })
    }

    /// Spline with `nk` default knots placed on `x` (`rcsDefaultKnots`).
    #[wasm_bindgen(js_name = fromData)]
    pub fn from_data(x: &[f64], nk: usize) -> Result<Rcs, JsError> {
        Ok(Self {
            inner: RustRcs::from_data(x, nk).map_err(fit_error)?,
        })
    }

    /// Normalisation of the nonlinear terms: "none" (`norm = 0`),
    /// "last-interval" (1) or "range" (2, the default of rms).
    #[wasm_bindgen(js_name = setNorm)]
    pub fn set_norm(&mut self, norm: &str) -> Result<(), JsError> {
        self.inner = self.inner.clone().with_norm(parse_norm(norm)?);
        Ok(())
    }

    /// The normalisation: "none", "last-interval" or "range".
    #[wasm_bindgen]
    pub fn norm(&self) -> String {
        match self.inner.norm() {
            RcsNorm::None => "none",
            RcsNorm::LastInterval => "last-interval",
            _ => "range",
        }
        .to_string()
    }

    /// The knots.
    #[wasm_bindgen]
    pub fn knots(&self) -> Vec<f64> {
        self.inner.knots().to_vec()
    }

    /// Number of basis columns (knots − 1).
    #[wasm_bindgen(js_name = nColumns)]
    pub fn n_columns(&self) -> usize {
        self.inner.n_columns()
    }

    /// The basis at `x`, columns named `name`, `name'`, `name''`, ….
    #[wasm_bindgen]
    pub fn basis(&self, x: &[f64], name: &str) -> Basis {
        Basis {
            inner: self.inner.basis(x, name),
        }
    }

    /// Algebraic form of a fitted spline (`Hmisc::rcspline.restate`) from
    /// the intercept and the `knots − 1` basis coefficients.
    #[wasm_bindgen]
    pub fn restate(&self, intercept: f64, coef: &[f64]) -> Result<RcsRestated, JsError> {
        Ok(RcsRestated {
            inner: self.inner.restate(intercept, coef).map_err(fit_error)?,
        })
    }
}

/// A fitted restricted cubic spline in truncated-power form:
/// `intercept + linear·x + Σⱼ cubic[j]·(x − knots[j])₊³`.
#[wasm_bindgen]
pub struct RcsRestated {
    inner: RustRcsRestated,
}

#[wasm_bindgen]
impl RcsRestated {
    /// `{ intercept, linear, knots, cubic }`.
    #[wasm_bindgen(js_name = getResult)]
    pub fn get_result(&self) -> JsValue {
        let r = &self.inner;
        object(&[
            ("intercept", r.intercept.to_js()),
            ("linear", r.linear.to_js()),
            ("knots", r.knots.to_js()),
            ("cubic", r.cubic.to_js()),
        ])
    }

    /// Evaluate the fitted function at `x`.
    #[wasm_bindgen]
    pub fn eval(&self, x: f64) -> f64 {
        self.inner.eval(x)
    }
}

/// Linear spline basis (`rms::lsp`).
#[wasm_bindgen]
#[derive(Clone)]
pub struct Lsp {
    pub(crate) inner: RustLsp,
}

#[wasm_bindgen]
impl Lsp {
    /// Linear spline with the given knots (sorted and de-duplicated).
    #[wasm_bindgen(constructor)]
    pub fn new(knots: &[f64]) -> Result<Lsp, JsError> {
        Ok(Self {
            inner: RustLsp::new(knots).map_err(fit_error)?,
        })
    }

    /// The knots.
    #[wasm_bindgen]
    pub fn knots(&self) -> Vec<f64> {
        self.inner.knots().to_vec()
    }

    /// Basis `[x, (x − t₁)₊, (x − t₂)₊, …]` (hinge terms nonlinear).
    #[wasm_bindgen]
    pub fn basis(&self, x: &[f64], name: &str) -> Basis {
        Basis {
            inner: self.inner.basis(x, name),
        }
    }
}

/// Knot quantiles of RMS Table 2.3 for `nk` knots.
#[wasm_bindgen(js_name = defaultKnotQuantiles)]
pub fn default_knot_quantiles_js(nk: usize) -> Vec<f64> {
    default_knot_quantiles(nk)
}

/// Default knots of a restricted cubic spline with `nk` knots, as
/// `Hmisc::rcspline.eval(x, nk, knots.only = TRUE)`; `fractied` defaults to
/// 0.05.
#[wasm_bindgen(js_name = rcsDefaultKnots)]
pub fn rcs_default_knots_js(
    x: &[f64],
    nk: usize,
    fractied: Option<f64>,
) -> Result<Vec<f64>, JsError> {
    match fractied {
        Some(f) => rcs_default_knots_with(x, nk, f),
        None => rcs_default_knots(x, nk),
    }
    .map_err(fit_error)
}

/// Restricted interaction `a %ia% b`: products without doubly nonlinear
/// terms.
#[wasm_bindgen(js_name = restrictedInteraction)]
pub fn restricted_interaction_js(a: &Basis, b: &Basis) -> Result<Basis, JsError> {
    Ok(Basis {
        inner: restricted_interaction(&a.inner, &b.inner).map_err(fit_error)?,
    })
}

fn parse_extrapolation(name: &str) -> Result<SplineExtrapolation, JsError> {
    match name.to_ascii_lowercase().as_str() {
        "constant" => Ok(SplineExtrapolation::Constant),
        "linear" => Ok(SplineExtrapolation::Linear),
        "continue" => Ok(SplineExtrapolation::Continue),
        "periodic" => Ok(SplineExtrapolation::Periodic),
        "error" => Ok(SplineExtrapolation::Error),
        other => Err(JsError::new(&format!(
            "Unknown extrapolation '{other}': expected constant, linear, continue, periodic or error"
        ))),
    }
}

/// B-spline basis of one variable, as scikit-learn's `SplineTransformer`.
#[wasm_bindgen]
#[derive(Clone)]
pub struct BSpline {
    inner: RustBSpline,
}

#[wasm_bindgen]
impl BSpline {
    /// Basis of degree `degree` on explicit base knots (at least 2,
    /// strictly increasing).
    #[wasm_bindgen(constructor)]
    pub fn new(base_knots: &[f64], degree: usize) -> Result<BSpline, JsError> {
        Ok(Self {
            inner: RustBSpline::new(base_knots, degree).map_err(fit_error)?,
        })
    }

    /// Basis with `nKnots` base knots on `x`, "uniform" (default) or
    /// "quantile".
    #[wasm_bindgen(js_name = fromData)]
    pub fn from_data(
        x: &[f64],
        n_knots: usize,
        degree: usize,
        knots: Option<String>,
    ) -> Result<BSpline, JsError> {
        let placement = match knots.as_deref().unwrap_or("uniform") {
            "uniform" => KnotPlacement::Uniform,
            "quantile" => KnotPlacement::Quantile,
            other => {
                return Err(JsError::new(&format!(
                    "Unknown knots '{other}': expected uniform or quantile"
                )))
            }
        };
        Ok(Self {
            inner: RustBSpline::from_data(x, n_knots, degree, placement).map_err(fit_error)?,
        })
    }

    /// "constant" (default), "linear", "continue", "periodic" or "error".
    #[wasm_bindgen(js_name = setExtrapolation)]
    pub fn set_extrapolation(&mut self, extrapolation: &str) -> Result<(), JsError> {
        self.inner = self
            .inner
            .clone()
            .with_extrapolation(parse_extrapolation(extrapolation)?);
        Ok(())
    }

    /// Keep (default) or drop the last spline (sklearn `include_bias`).
    #[wasm_bindgen(js_name = setBias)]
    pub fn set_bias(&mut self, include_bias: bool) {
        self.inner = self.inner.clone().with_bias(include_bias);
    }

    /// The base knots.
    #[wasm_bindgen(js_name = baseKnots)]
    pub fn base_knots(&self) -> Vec<f64> {
        self.inner.base_knots().to_vec()
    }

    /// The full knot vector (base knots plus `degree` on each side).
    #[wasm_bindgen]
    pub fn knots(&self) -> Vec<f64> {
        self.inner.knots()
    }

    /// The degree.
    #[wasm_bindgen]
    pub fn degree(&self) -> usize {
        self.inner.degree()
    }

    /// The extrapolation mode.
    #[wasm_bindgen]
    pub fn extrapolation(&self) -> String {
        match self.inner.extrapolation() {
            SplineExtrapolation::Linear => "linear",
            SplineExtrapolation::Continue => "continue",
            SplineExtrapolation::Periodic => "periodic",
            SplineExtrapolation::Error => "error",
            _ => "constant",
        }
        .to_string()
    }

    /// Whether the last spline is kept.
    #[wasm_bindgen(js_name = includeBias)]
    pub fn include_bias(&self) -> bool {
        self.inner.include_bias()
    }

    /// Number of basis columns.
    #[wasm_bindgen(js_name = nColumns)]
    pub fn n_columns(&self) -> usize {
        self.inner.n_columns()
    }

    /// The basis at `x` (columns `name_sp_0, …`).
    #[wasm_bindgen]
    pub fn basis(&self, x: &[f64], name: &str) -> Result<Basis, JsError> {
        Ok(Basis {
            inner: self.inner.basis(x, name).map_err(fit_error)?,
        })
    }
}
