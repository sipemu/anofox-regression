//! Matrix utility functions.

use faer::{Col, Mat};

/// Detect columns that are constant (zero variance).
pub fn detect_constant_columns(x: &Mat<f64>, tolerance: f64) -> Vec<bool> {
    let n_cols = x.ncols();
    let n_rows = x.nrows();

    if n_rows == 0 {
        return vec![true; n_cols];
    }

    let mut constant = vec![false; n_cols];

    for j in 0..n_cols {
        let first = x[(0, j)];
        let all_same = (1..n_rows).all(|i| (x[(i, j)] - first).abs() < tolerance);
        constant[j] = all_same;
    }

    constant
}

/// Detect constant columns with a **scale-relative** tolerance.
///
/// Column `j` is constant when `max_i x_ij − min_i x_ij ≤ tolerance · max_i |x_ij|`
/// (an all-zero column is constant). Unlike [`detect_constant_columns`], whose
/// tolerance is absolute, this does not flag non-constant columns measured in
/// tiny units (e.g. `x · 1e-12`) and does not miss constant offsets of large
/// columns. R's `lm` behaves the same way, since its pivoting tolerance is
/// relative to the column norm.
pub fn detect_constant_columns_relative(x: &Mat<f64>, tolerance: f64) -> Vec<bool> {
    let n_rows = x.nrows();
    (0..x.ncols())
        .map(|j| {
            if n_rows == 0 {
                return true;
            }
            let (mut lo, mut hi, mut amax) = (f64::INFINITY, f64::NEG_INFINITY, 0.0_f64);
            for i in 0..n_rows {
                let v = x[(i, j)];
                lo = lo.min(v);
                hi = hi.max(v);
                amax = amax.max(v.abs());
            }
            hi - lo <= tolerance * amax
        })
        .collect()
}

/// Detect columns that are identically zero.
pub fn detect_zero_columns(x: &Mat<f64>) -> Vec<bool> {
    (0..x.ncols())
        .map(|j| (0..x.nrows()).all(|i| x[(i, j)] == 0.0))
        .collect()
}

/// Center a matrix by subtracting column means.
pub fn center_columns(x: &Mat<f64>) -> (Mat<f64>, Col<f64>) {
    let n_rows = x.nrows();
    let n_cols = x.ncols();

    let mut means = Col::zeros(n_cols);
    let mut centered = Mat::zeros(n_rows, n_cols);

    for j in 0..n_cols {
        let sum: f64 = (0..n_rows).map(|i| x[(i, j)]).sum();
        means[j] = sum / n_rows as f64;

        for i in 0..n_rows {
            centered[(i, j)] = x[(i, j)] - means[j];
        }
    }

    (centered, means)
}

/// Center a vector by subtracting the mean.
pub fn center_vector(y: &Col<f64>) -> (Col<f64>, f64) {
    let n = y.nrows();
    let mean: f64 = y.iter().sum::<f64>() / n as f64;

    let centered = Col::from_fn(n, |i| y[i] - mean);

    (centered, mean)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_constant_columns() {
        let mut x = Mat::zeros(5, 3);
        // Column 0: constant
        for i in 0..5 {
            x[(i, 0)] = 1.0;
            x[(i, 1)] = i as f64;
            x[(i, 2)] = 2.0;
        }

        let constant = detect_constant_columns(&x, 1e-10);
        assert!(constant[0]); // constant
        assert!(!constant[1]); // not constant
        assert!(constant[2]); // constant
    }

    #[test]
    fn test_center_columns() {
        let mut x = Mat::zeros(4, 2);
        x[(0, 0)] = 1.0;
        x[(1, 0)] = 2.0;
        x[(2, 0)] = 3.0;
        x[(3, 0)] = 4.0;
        x[(0, 1)] = 10.0;
        x[(1, 1)] = 20.0;
        x[(2, 1)] = 30.0;
        x[(3, 1)] = 40.0;

        let (centered, means) = center_columns(&x);

        assert!((means[0] - 2.5).abs() < 1e-10);
        assert!((means[1] - 25.0).abs() < 1e-10);

        // Check centered values sum to zero
        let col0_sum: f64 = (0..4).map(|i| centered[(i, 0)]).sum();
        let col1_sum: f64 = (0..4).map(|i| centered[(i, 1)]).sum();
        assert!(col0_sum.abs() < 1e-10);
        assert!(col1_sum.abs() < 1e-10);
    }

    #[test]
    fn test_center_vector() {
        let y = Col::from_fn(4, |i| (i + 1) as f64); // [1, 2, 3, 4]
        let (centered, mean) = center_vector(&y);

        assert!((mean - 2.5).abs() < 1e-10);
        assert!(centered.iter().sum::<f64>().abs() < 1e-10);
    }

    #[test]
    fn test_detect_constant_columns_empty() {
        let x = Mat::<f64>::zeros(0, 3);
        let constant = detect_constant_columns(&x, 1e-10);
        // With 0 rows, all columns should be considered constant
        assert_eq!(constant.len(), 3);
        assert!(constant.iter().all(|&c| c));
    }

    #[test]
    fn test_detect_constant_columns_with_tolerance() {
        let mut x = Mat::zeros(3, 2);
        // Column 0: nearly constant [1.0, 1.000001, 1.0]
        x[(0, 0)] = 1.0;
        x[(1, 0)] = 1.000001;
        x[(2, 0)] = 1.0;
        // Column 1: varying [1.0, 2.0, 3.0]
        x[(0, 1)] = 1.0;
        x[(1, 1)] = 2.0;
        x[(2, 1)] = 3.0;

        // With tight tolerance, column 0 is not constant
        let constant_tight = detect_constant_columns(&x, 1e-10);
        assert!(!constant_tight[0]);
        assert!(!constant_tight[1]);

        // With loose tolerance, column 0 is constant
        let constant_loose = detect_constant_columns(&x, 1e-5);
        assert!(constant_loose[0]);
        assert!(!constant_loose[1]);
    }
}

/// Default tolerance of the R-`lm`-style aliasing test ([`lm_alias_columns`]).
///
/// Matches `tol = 1e-7` used by R's `lm.fit` / `lm.wfit` (LINPACK `dqrdc2`).
pub const LM_ALIAS_TOLERANCE: f64 = 1e-7;

/// Choose aliased columns the way R's `lm` does (LINPACK `dqrdc2` with
/// *limited* pivoting).
///
/// Columns are processed in their given order. Column `j` is aliased when the
/// norm of its residual after projecting out the intercept column (if any)
/// and all earlier **non-aliased** columns falls below `tol` times its
/// original norm. Hence the *later* column of a collinear set is aliased, the
/// decision is invariant to rescaling any column, and exact duplicates or
/// linear combinations are caught with a margin of ~9 orders of magnitude
/// over floating-point rounding.
///
/// * `x` – design columns as R sees them (uncentered; multiplied by
///   `sqrt(w)` for weighted fits).
/// * `intercept` – the intercept column as R sees it (ones, or `sqrt(w)`),
///   placed first and never aliased; `None` for models without intercept.
/// * `pre_aliased` – columns already known to be aliased (e.g. constants).
///
/// The projection uses modified Gram–Schmidt with one re-orthogonalisation
/// pass, which is accurate to working precision.
pub fn lm_alias_columns(
    x: &Mat<f64>,
    intercept: Option<&Col<f64>>,
    pre_aliased: &[bool],
    tol: f64,
) -> Vec<bool> {
    let n = x.nrows();
    let p = x.ncols();
    let mut basis: Vec<Vec<f64>> = Vec::with_capacity(p + 1);
    let norm = |v: &[f64]| v.iter().map(|a| a * a).sum::<f64>().sqrt();

    if let Some(ic) = intercept {
        let v: Vec<f64> = (0..n).map(|i| ic[i]).collect();
        let nv = norm(&v);
        if nv > 0.0 {
            basis.push(v.iter().map(|a| a / nv).collect());
        }
    }

    let mut aliased = vec![false; p];
    for j in 0..p {
        if pre_aliased.get(j).copied().unwrap_or(false) {
            aliased[j] = true;
            continue;
        }
        let mut v: Vec<f64> = (0..n).map(|i| x[(i, j)]).collect();
        let original = norm(&v);
        if original == 0.0 || !original.is_finite() {
            aliased[j] = true;
            continue;
        }
        for _ in 0..2 {
            for q in &basis {
                let d: f64 = q.iter().zip(&v).map(|(a, b)| a * b).sum();
                for (vi, qi) in v.iter_mut().zip(q) {
                    *vi -= d * qi;
                }
            }
        }
        let residual = norm(&v);
        if residual < tol * original {
            aliased[j] = true;
        } else {
            basis.push(v.iter().map(|a| a / residual).collect());
        }
    }
    aliased
}

/// Solve a least-squares problem on the non-aliased columns only.
///
/// Builds the sub-matrix of `x` holding the columns with `aliased[j] ==
/// false`, rescales each to unit norm (so the solver's own rank threshold
/// is independent of column scaling), calls `solve` on it with no columns
/// pre-aliased and maps the result back: aliased columns get NaN, the rest
/// are un-scaled. Columns the solver itself drops are also reported aliased.
pub fn solve_on_kept_columns<E, F>(
    x: &Mat<f64>,
    aliased: &[bool],
    solve: F,
) -> Result<(Col<f64>, Vec<bool>, usize), E>
where
    F: FnOnce(&Mat<f64>, &[bool]) -> Result<(Col<f64>, Vec<bool>, usize), E>,
{
    let p = x.ncols();
    let n = x.nrows();
    let kept: Vec<usize> = (0..p).filter(|&j| !aliased[j]).collect();
    let mut out_aliased = aliased.to_vec();
    let mut coefficients = Col::from_fn(p, |_| f64::NAN);
    if kept.is_empty() {
        return Ok((coefficients, vec![true; p], 0));
    }
    let scales: Vec<f64> = kept
        .iter()
        .map(|&j| {
            let s = (0..n).map(|i| x[(i, j)] * x[(i, j)]).sum::<f64>().sqrt();
            if s > 0.0 && s.is_finite() {
                s
            } else {
                1.0
            }
        })
        .collect();
    let xr = Mat::from_fn(n, kept.len(), |i, k| x[(i, kept[k])] / scales[k]);
    let (beta, sub_aliased, rank) = solve(&xr, &vec![false; kept.len()])?;
    for (k, &j) in kept.iter().enumerate() {
        if sub_aliased[k] || beta[k].is_nan() {
            out_aliased[j] = true;
        } else {
            coefficients[j] = beta[k] / scales[k];
        }
    }
    Ok((coefficients, out_aliased, rank))
}
