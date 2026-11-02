//! Term maps and design matrices (`anofox_regression::design`): which slope
//! columns belong to which model term, for ANOVA, step-down and model
//! approximation, and design matrices assembled from spline bases.

use anofox_regression::design::{Design as RustDesign, Interaction, Term, TermMap as RustTermMap};
use wasm_bindgen::prelude::*;

use crate::convert::{impl_to_js, ToJs};
use crate::regressors::{fit_error, mat_flat};
use crate::splines::Basis;

impl_to_js!(Interaction {
    factors,
    components,
    restricted
});
impl_to_js!(Term {
    name,
    columns,
    nonlinear,
    interaction
});

/// Which slope columns (no intercept) belong to which term: main effects
/// with linear / nonlinear columns and two-way interactions.
#[wasm_bindgen]
#[derive(Clone, Default)]
pub struct TermMap {
    pub(crate) inner: RustTermMap,
}

#[wasm_bindgen]
impl TermMap {
    /// An empty term map.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a main-effect term with one column per entry of `nonlinear`
    /// (1 = a nonlinear column, e.g. a spline basis term); returns its index.
    #[wasm_bindgen(js_name = addMain)]
    pub fn add_main(&mut self, name: &str, nonlinear: &[u8]) -> usize {
        let nonlinear: Vec<bool> = nonlinear.iter().map(|&v| v != 0).collect();
        self.inner.add_main(name, &nonlinear)
    }

    /// Append the interaction of main-effect terms `a` and `b` (the full
    /// tensor product, or the rms `%ia%` product with `restricted`);
    /// returns its index.
    #[wasm_bindgen(js_name = addInteraction)]
    pub fn add_interaction(
        &mut self,
        a: usize,
        b: usize,
        restricted: bool,
    ) -> Result<usize, JsError> {
        self.inner
            .add_interaction(a, b, restricted)
            .map_err(fit_error)
    }

    /// Number of slope columns.
    #[wasm_bindgen(js_name = nColumns)]
    pub fn n_columns(&self) -> usize {
        self.inner.n_columns()
    }

    /// Index of the term called `name`.
    #[wasm_bindgen]
    pub fn find(&self, name: &str) -> Option<usize> {
        self.inner.find(name)
    }

    /// The terms: `[{ name, columns, nonlinear, interaction? }]`.
    #[wasm_bindgen]
    pub fn terms(&self) -> JsValue {
        self.inner.terms().to_js()
    }
}

/// A design matrix (no intercept column) built from basis blocks, with its
/// term map: main effects from `Basis` objects, then interactions.
#[wasm_bindgen]
#[derive(Clone, Default)]
pub struct Design {
    pub(crate) inner: RustDesign,
}

#[wasm_bindgen]
impl Design {
    /// An empty design.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a main-effect term made of `basis`; returns its term index.
    #[wasm_bindgen]
    pub fn add(&mut self, name: &str, basis: &Basis) -> Result<usize, JsError> {
        self.inner.add(name, basis.inner.clone()).map_err(fit_error)
    }

    /// Append the full tensor-product interaction of terms `a` and `b`
    /// (rms `a * b`); returns its term index.
    #[wasm_bindgen(js_name = addInteraction)]
    pub fn add_interaction(&mut self, a: usize, b: usize) -> Result<usize, JsError> {
        self.inner.add_interaction(a, b).map_err(fit_error)
    }

    /// Append the restricted interaction `a %ia% b`; returns its term index.
    #[wasm_bindgen(js_name = addRestrictedInteraction)]
    pub fn add_restricted_interaction(&mut self, a: usize, b: usize) -> Result<usize, JsError> {
        self.inner
            .add_restricted_interaction(a, b)
            .map_err(fit_error)
    }

    /// The design matrix, row-major `nRows × nColumns`.
    #[wasm_bindgen]
    pub fn matrix(&self) -> Vec<f64> {
        mat_flat(&self.inner.matrix())
    }

    /// Number of rows.
    #[wasm_bindgen(js_name = nRows)]
    pub fn n_rows(&self) -> usize {
        self.inner.matrix().nrows()
    }

    /// Number of columns.
    #[wasm_bindgen(js_name = nColumns)]
    pub fn n_columns(&self) -> usize {
        self.inner.column_names().len()
    }

    /// Column names.
    #[wasm_bindgen(js_name = columnNames)]
    pub fn column_names(&self) -> Vec<String> {
        self.inner.column_names().to_vec()
    }

    /// The term map (for `anova`, `fastbw`, …).
    #[wasm_bindgen]
    pub fn terms(&self) -> TermMap {
        TermMap {
            inner: self.inner.terms().clone(),
        }
    }

    /// Basis block of term `t` (for interactions, of the product).
    #[wasm_bindgen]
    pub fn block(&self, t: usize) -> Option<Basis> {
        self.inner.block(t).map(|b| Basis { inner: b.clone() })
    }
}
