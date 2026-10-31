//! Aliased-column handling shared by the IRLS GLM solvers.
//!
//! Like [`OlsRegressor`](crate::solvers::OlsRegressor) and R's `glm`, the GLMs
//! drop columns that are linearly dependent on the intercept and the earlier
//! columns (constant columns with an intercept, duplicates, exact linear
//! combinations), report them as aliased with a `NaN` coefficient (and `NaN`
//! inference), and ignore them when predicting. Of a collinear set the
//! *later* column is dropped, as R's limited-pivoting QR (`dqrdc2`) does.

use crate::core::RegressionResult;
use crate::utils::{detect_constant_columns, lm_alias_columns};
use faer::{Col, Mat};

/// Relative residual-norm threshold of the collinearity test: R's `glm.fit`
/// QR tolerance `min(1e-7, epsilon / 1000)` at the default `epsilon = 1e-8`.
const GLM_ALIAS_TOLERANCE: f64 = 1e-11;

/// Columns to keep (those not aliased with the intercept and earlier
/// columns), or `None` when nothing has to be dropped.
pub(crate) fn columns_to_keep(x: &Mat<f64>, with_intercept: bool, tol: f64) -> Option<Vec<usize>> {
    if x.nrows() == 0 {
        return None;
    }
    let constant = if with_intercept {
        detect_constant_columns(x, tol)
    } else {
        vec![false; x.ncols()]
    };
    let ones = Col::from_fn(x.nrows(), |_| 1.0);
    let aliased = lm_alias_columns(
        x,
        with_intercept.then_some(&ones),
        &constant,
        GLM_ALIAS_TOLERANCE,
    );
    if !aliased.iter().any(|&a| a) {
        return None;
    }
    Some((0..x.ncols()).filter(|&j| !aliased[j]).collect())
}

/// `x[:, keep]`.
pub(crate) fn select_columns(x: &Mat<f64>, keep: &[usize]) -> Mat<f64> {
    Mat::from_fn(x.nrows(), keep.len(), |i, j| x[(i, keep[j])])
}

fn expand_col(c: &Col<f64>, keep: &[usize], p: usize) -> Col<f64> {
    let mut out = Col::from_fn(p, |_| f64::NAN);
    for (k, &j) in keep.iter().enumerate() {
        out[j] = c[k];
    }
    out
}

/// Re-expand a coefficient covariance of a fit on `x[:, keep]` (intercept
/// first when `with_intercept`) to the original `p` columns, with zero rows
/// and columns for the dropped ones.
pub(crate) fn expand_vcov(
    v: &Mat<f64>,
    keep: &[usize],
    p: usize,
    with_intercept: bool,
) -> Mat<f64> {
    let off = usize::from(with_intercept);
    let index: Vec<usize> = (0..off).chain(keep.iter().map(|&j| j + off)).collect();
    let mut out = Mat::zeros(p + off, p + off);
    for (a, &i) in index.iter().enumerate() {
        for (b, &j) in index.iter().enumerate() {
            out[(i, j)] = v[(a, b)];
        }
    }
    out
}

/// Re-expand a result fitted on `x[:, keep]` to the original `p` columns:
/// dropped columns get `NaN` coefficients / inference and `aliased = true`.
pub(crate) fn expand_result(r: &RegressionResult, keep: &[usize], p: usize) -> RegressionResult {
    let mut out = r.clone();
    out.coefficients = expand_col(&r.coefficients, keep, p);
    out.std_errors = r.std_errors.as_ref().map(|c| expand_col(c, keep, p));
    out.t_statistics = r.t_statistics.as_ref().map(|c| expand_col(c, keep, p));
    out.p_values = r.p_values.as_ref().map(|c| expand_col(c, keep, p));
    out.conf_interval_lower = r
        .conf_interval_lower
        .as_ref()
        .map(|c| expand_col(c, keep, p));
    out.conf_interval_upper = r
        .conf_interval_upper
        .as_ref()
        .map(|c| expand_col(c, keep, p));
    let mut aliased = vec![true; p];
    for (k, &j) in keep.iter().enumerate() {
        aliased[j] = r.aliased.get(k).copied().unwrap_or(false);
    }
    out.aliased = aliased;
    out.column_permutation = None;
    out
}
