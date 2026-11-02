//! Term maps (`anofox_regression::design`): which slope columns belong to
//! which model term, for ANOVA, step-down and model approximation.

use anofox_regression::design::{Interaction, Term, TermMap as RustTermMap};
use wasm_bindgen::prelude::*;

use crate::convert::{impl_to_js, ToJs};
use crate::regressors::fit_error;

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
