//! Streaming and mergeable fit states (`solvers::MomentAccumulator`,
//! `solvers::SufficientStats`): accumulate rows in chunks or on several
//! workers, merge, and fit OLS / ridge from the state with
//! `OlsRegressor.fitFromState` / `RidgeRegressor.fitFromState`.

use anofox_regression::solvers::{
    MomentAccumulator as RustMomentAccumulator, SufficientStats as RustSufficientStats,
};
use faer::{Col, Mat};
use wasm_bindgen::prelude::*;

use crate::convert::ToJs;
use crate::regressors::{fit_error, mat_flat, to_mat};

/// Raw moments `XᵀX`, `Xᵀy`, `Σx`, `Σy` and `n` of streamed rows.
#[wasm_bindgen]
#[derive(Clone)]
pub struct MomentAccumulator {
    pub(crate) inner: RustMomentAccumulator,
}

#[wasm_bindgen]
impl MomentAccumulator {
    /// An empty accumulator for `nFeatures` columns.
    #[wasm_bindgen(constructor)]
    pub fn new(n_features: usize) -> Self {
        Self {
            inner: RustMomentAccumulator::new(n_features),
        }
    }

    /// Number of columns.
    #[wasm_bindgen(js_name = nFeatures)]
    pub fn n_features(&self) -> usize {
        self.inner.n_features()
    }

    /// Rows seen.
    #[wasm_bindgen]
    pub fn n(&self) -> usize {
        self.inner.n()
    }

    /// `Σx` per column.
    #[wasm_bindgen(js_name = sumX)]
    pub fn sum_x(&self) -> Vec<f64> {
        self.inner.sum_x().iter().copied().collect()
    }

    /// `Σy`.
    #[wasm_bindgen(js_name = sumY)]
    pub fn sum_y(&self) -> f64 {
        self.inner.sum_y()
    }

    /// `XᵀX`, row-major.
    #[wasm_bindgen]
    pub fn xtx(&self) -> Vec<f64> {
        mat_flat(self.inner.xtx())
    }

    /// `Xᵀy`.
    #[wasm_bindgen]
    pub fn xty(&self) -> Vec<f64> {
        self.inner.xty().iter().copied().collect()
    }

    /// Add one row.
    #[wasm_bindgen(js_name = pushRow)]
    pub fn push_row(&mut self, x_row: &[f64], y: f64) -> Result<(), JsError> {
        self.inner.push_row(x_row, y).map_err(fit_error)
    }

    /// Add the rows of a row-major `nRows × nFeatures` chunk.
    #[wasm_bindgen(js_name = pushRows)]
    pub fn push_rows(&mut self, x: &[f64], n_rows: usize, y: &[f64]) -> Result<(), JsError> {
        let p = self.inner.n_features();
        to_mat(x, n_rows, p)?;
        if y.len() != n_rows {
            return Err(JsError::new(&format!(
                "Expected {} y values, got {}",
                n_rows,
                y.len()
            )));
        }
        for i in 0..n_rows {
            self.inner
                .push_row(&x[i * p..(i + 1) * p], y[i])
                .map_err(fit_error)?;
        }
        Ok(())
    }

    /// Add the moments of another accumulator.
    #[wasm_bindgen]
    pub fn merge(&mut self, other: &MomentAccumulator) -> Result<(), JsError> {
        self.inner.merge(&other.inner).map_err(fit_error)
    }

    /// Reset to empty.
    #[wasm_bindgen]
    pub fn clear(&mut self) {
        self.inner.clear()
    }
}

/// Mergeable, serialisable sufficient statistics (centred moments with
/// optional weights) from which OLS and ridge fits are exact.
#[wasm_bindgen]
#[derive(Clone)]
pub struct SufficientStats {
    pub(crate) inner: RustSufficientStats,
}

#[wasm_bindgen]
impl SufficientStats {
    /// An empty state for `nFeatures` columns.
    #[wasm_bindgen(constructor)]
    pub fn new(n_features: usize) -> Self {
        Self {
            inner: RustSufficientStats::new(n_features),
        }
    }

    /// State of a row-major data block, optionally weighted.
    #[wasm_bindgen(js_name = fromData)]
    pub fn from_data(
        x: &[f64],
        n_rows: usize,
        n_cols: usize,
        y: &[f64],
        weights: Option<Vec<f64>>,
    ) -> Result<SufficientStats, JsError> {
        let xm = to_mat(x, n_rows, n_cols)?;
        let yc = Col::from_fn(y.len(), |i| y[i]);
        let w = weights.map(|w| Col::from_fn(w.len(), |i| w[i]));
        Ok(Self {
            inner: RustSufficientStats::from_data(&xm, &yc, w.as_ref()).map_err(fit_error)?,
        })
    }

    /// Restore a state written by `toBytes`.
    #[wasm_bindgen(js_name = fromBytes)]
    pub fn from_bytes(bytes: &[u8]) -> Result<SufficientStats, JsError> {
        Ok(Self {
            inner: RustSufficientStats::from_bytes(bytes).map_err(fit_error)?,
        })
    }

    /// Serialise the state (versioned binary format).
    #[wasm_bindgen(js_name = toBytes)]
    pub fn to_bytes(&self) -> Vec<u8> {
        self.inner.to_bytes()
    }

    /// Add one row.
    #[wasm_bindgen]
    pub fn push(&mut self, x_row: &[f64], y: f64) -> Result<(), JsError> {
        self.inner.push(x_row, y).map_err(fit_error)
    }

    /// Add one weighted row.
    #[wasm_bindgen(js_name = pushWeighted)]
    pub fn push_weighted(&mut self, x_row: &[f64], y: f64, w: f64) -> Result<(), JsError> {
        self.inner.push_weighted(x_row, y, w).map_err(fit_error)
    }

    /// Merge another state into this one.
    #[wasm_bindgen]
    pub fn merge(&mut self, other: &SufficientStats) -> Result<(), JsError> {
        self.inner.merge(&other.inner).map_err(fit_error)
    }

    /// A new state holding both.
    #[wasm_bindgen]
    pub fn merged(&self, other: &SufficientStats) -> Result<SufficientStats, JsError> {
        Ok(Self {
            inner: self.inner.merged(&other.inner).map_err(fit_error)?,
        })
    }

    /// Reset to empty.
    #[wasm_bindgen]
    pub fn clear(&mut self) {
        self.inner.clear()
    }

    /// Number of columns.
    #[wasm_bindgen(js_name = nFeatures)]
    pub fn n_features(&self) -> usize {
        self.inner.n_features()
    }

    /// Rows seen.
    #[wasm_bindgen(js_name = nObservations)]
    pub fn n_observations(&self) -> usize {
        self.inner.n_observations()
    }

    /// Rows with positive weight.
    #[wasm_bindgen(js_name = nPositive)]
    pub fn n_positive(&self) -> usize {
        self.inner.n_positive()
    }

    /// Whether no row was added.
    #[wasm_bindgen(js_name = isEmpty)]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Whether weights were used.
    #[wasm_bindgen(js_name = isWeighted)]
    pub fn is_weighted(&self) -> bool {
        self.inner.is_weighted()
    }

    /// `Σw`.
    #[wasm_bindgen(js_name = sumWeights)]
    pub fn sum_weights(&self) -> f64 {
        self.inner.sum_weights()
    }

    /// `Σ log w` over the positive weights.
    #[wasm_bindgen(js_name = sumLogWeights)]
    pub fn sum_log_weights(&self) -> f64 {
        self.inner.sum_log_weights()
    }

    /// Weighted column means.
    #[wasm_bindgen(js_name = meanX)]
    pub fn mean_x(&self) -> Vec<f64> {
        self.inner.mean_x().iter().copied().collect()
    }

    /// Weighted mean of `y`.
    #[wasm_bindgen(js_name = meanY)]
    pub fn mean_y(&self) -> f64 {
        self.inner.mean_y()
    }

    /// Centred `XᵀWX`, row-major.
    #[wasm_bindgen(js_name = centeredXx)]
    pub fn centered_xx(&self) -> Vec<f64> {
        let m: Mat<f64> = self.inner.centered_xx();
        mat_flat(&m)
    }

    /// Centred `XᵀWy`.
    #[wasm_bindgen(js_name = centeredXy)]
    pub fn centered_xy(&self) -> Vec<f64> {
        self.inner.centered_xy().iter().copied().collect()
    }

    /// Centred `yᵀWy`.
    #[wasm_bindgen(js_name = centeredYy)]
    pub fn centered_yy(&self) -> f64 {
        self.inner.centered_yy()
    }

    /// Summary as a plain object: `{ nFeatures, nObservations, sumWeights,
    /// meanX, meanY }`.
    #[wasm_bindgen(js_name = getResult)]
    pub fn get_result(&self) -> JsValue {
        crate::convert::object(&[
            ("n_features", self.inner.n_features().to_js()),
            ("n_observations", self.inner.n_observations().to_js()),
            ("sum_weights", self.inner.sum_weights().to_js()),
            ("mean_x", self.inner.mean_x().to_js()),
            ("mean_y", self.inner.mean_y().to_js()),
        ])
    }
}
