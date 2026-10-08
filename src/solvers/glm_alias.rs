//! Constant-column handling shared by the IRLS GLM solvers.
//!
//! With an intercept, a constant feature column is perfectly collinear with
//! the intercept. Like [`OlsRegressor`](crate::solvers::OlsRegressor) (and R's
//! `glm`), the GLMs drop such columns from the fit, report them as aliased
//! with a `NaN` coefficient (and `NaN` inference), and ignore them when
//! predicting.

use crate::core::RegressionResult;
use crate::utils::detect_constant_columns;
use faer::{Col, Mat};

/// Columns to keep when an intercept model contains constant columns, or
/// `None` when nothing has to be dropped.
pub(crate) fn columns_to_keep(x: &Mat<f64>, with_intercept: bool, tol: f64) -> Option<Vec<usize>> {
    if !with_intercept || x.nrows() == 0 {
        return None;
    }
    let constant = detect_constant_columns(x, tol);
    if !constant.iter().any(|&c| c) {
        return None;
    }
    Some((0..x.ncols()).filter(|&j| !constant[j]).collect())
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
