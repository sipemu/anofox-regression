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
