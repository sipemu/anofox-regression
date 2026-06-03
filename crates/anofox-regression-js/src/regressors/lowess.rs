//! LOWESS (Locally Weighted Scatterplot Smoothing) wrapper for WebAssembly.
//!
//! Unlike the other modules in this crate, LOWESS is exposed as plain functions
//! rather than a regressor — there is no model object to fit or persist.

use anofox_regression::solvers::lowess::{lowess_smooth, lowess_smooth_weights};
use faer::{Col, Mat};
use wasm_bindgen::prelude::*;

/// LOWESS-smoothed copy of a single time series.
///
/// # Arguments
/// * `y` - Values to smooth.
/// * `span` - Fraction of data used for each local regression (clamped to [0, 1]).
#[wasm_bindgen(js_name = lowessSmooth)]
pub fn lowess_smooth_js(y: &[f64], span: f64) -> Vec<f64> {
    let y_col = Col::from_fn(y.len(), |i| y[i]);
    let smoothed = lowess_smooth(&y_col, span);
    (0..smoothed.nrows()).map(|i| smoothed[i]).collect()
}

/// Apply LOWESS smoothing to each column of a weight matrix.
///
/// Rows are renormalized to sum to 1 after smoothing, matching the behavior
/// expected by `lmDynamic`.
///
/// # Arguments
/// * `weights` - Flat row-major weight matrix.
/// * `n_rows` - Number of rows (observations / time steps).
/// * `n_cols` - Number of columns (models).
/// * `span` - Fraction of data used for each local regression (clamped to [0, 1]).
#[wasm_bindgen(js_name = lowessSmoothWeights)]
pub fn lowess_smooth_weights_js(
    weights: &[f64],
    n_rows: usize,
    n_cols: usize,
    span: f64,
) -> Result<Vec<f64>, JsError> {
    if weights.len() != n_rows * n_cols {
        return Err(JsError::new(&format!(
            "Expected {} elements for {}x{} matrix, got {}",
            n_rows * n_cols,
            n_rows,
            n_cols,
            weights.len()
        )));
    }

    let mat = Mat::from_fn(n_rows, n_cols, |i, j| weights[i * n_cols + j]);
    let smoothed = lowess_smooth_weights(&mat, span);

    let mut out = Vec::with_capacity(n_rows * n_cols);
    for i in 0..n_rows {
        for j in 0..n_cols {
            out.push(smoothed[(i, j)]);
        }
    }
    Ok(out)
}
