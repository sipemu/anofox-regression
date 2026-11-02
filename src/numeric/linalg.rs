//! Small dense Cholesky kernels on row-major `Vec<Vec<f64>>` and faer
//! matrices, for the solvers that factor small SPD systems by hand.

// Index loops mirror the textbook recurrences; keep them readable.
#![allow(clippy::needless_range_loop)]

use faer::{Col, Mat};

/// Lower-triangular Cholesky factor `L` (`A = L Lᵀ`) of the `n × n` SPD
/// matrix whose lower triangle `a(i, j)` (`j ≤ i`) returns.
///
/// Rows are filled in order, each entry subtracting `L[i][k]·L[j][k]` for
/// ascending `k`. `None` when a pivot is not positive and finite.
fn cholesky_with(n: usize, a: impl Fn(usize, usize) -> f64) -> Option<Vec<Vec<f64>>> {
    let mut l = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = a(i, j);
            for k in 0..j {
                s -= l[i][k] * l[j][k];
            }
            if i == j {
                if !(s > 0.0 && s.is_finite()) {
                    return None;
                }
                l[i][j] = s.sqrt();
            } else {
                l[i][j] = s / l[j][j];
            }
        }
    }
    Some(l)
}

/// Lower-triangular Cholesky factor `L` (`A = L Lᵀ`) of a small SPD matrix;
/// `None` when a pivot is not positive and finite.
pub(crate) fn cholesky(a: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    cholesky_with(a.len(), |i, j| a[i][j])
}

/// [`cholesky`] for a faer matrix (only the lower triangle is read).
pub(crate) fn cholesky_mat(a: &Mat<f64>) -> Option<Mat<f64>> {
    let n = a.nrows();
    let l = cholesky_with(n, |i, j| a[(i, j)])?;
    Some(Mat::from_fn(n, n, |i, j| l[i][j]))
}

/// Solve `A x = b` given the Cholesky factor `L` of `A` (forward then back
/// substitution).
pub(crate) fn chol_solve(l: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let p = l.len();
    let mut y = vec![0.0; p];
    for i in 0..p {
        let mut sum = b[i];
        for k in 0..i {
            sum -= l[i][k] * y[k];
        }
        y[i] = sum / l[i][i];
    }
    let mut x = vec![0.0; p];
    for i in (0..p).rev() {
        let mut sum = y[i];
        for k in (i + 1)..p {
            sum -= l[k][i] * x[k];
        }
        x[i] = sum / l[i][i];
    }
    x
}

/// Solve `A X = B` columnwise given the Cholesky factor `L` of `A`.
/// `b` is `q × c`; the result is `q × c`.
pub(crate) fn chol_solve_mat(l: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let q = b.len();
    let c = if q > 0 { b[0].len() } else { 0 };
    let mut out = vec![vec![0.0; c]; q];
    for col in 0..c {
        let rhs: Vec<f64> = (0..q).map(|r| b[r][col]).collect();
        let sol = chol_solve(l, &rhs);
        for r in 0..q {
            out[r][col] = sol[r];
        }
    }
    out
}

/// Inverse of an SPD matrix from its Cholesky factor `L`.
pub(crate) fn chol_inverse(l: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let p = l.len();
    let mut inv = vec![vec![0.0; p]; p];
    for col in 0..p {
        let mut e = vec![0.0; p];
        e[col] = 1.0;
        let x = chol_solve(l, &e);
        for (row, xr) in x.iter().enumerate() {
            inv[row][col] = *xr;
        }
    }
    inv
}

/// Solve the symmetric system `A x = b` by a Cholesky factorisation that
/// never fails: a non-positive pivot is replaced by `√(A[j][j] + 1e-10)`.
///
/// Used by the IRLS solvers (Huber, quantile) whose normal equations can be
/// numerically semi-definite.
pub(crate) fn solve_spd_jittered(a: &Mat<f64>, b: &Col<f64>) -> Col<f64> {
    let n = a.nrows();

    // Cholesky decomposition: A = LL'
    let mut l: Mat<f64> = Mat::zeros(n, n);
    for j in 0..n {
        let mut sum = 0.0;
        for k in 0..j {
            sum += l[(j, k)].powi(2);
        }
        let diag = a[(j, j)] - sum;
        if diag <= 0.0 {
            // Add small regularization if needed
            l[(j, j)] = (a[(j, j)] + 1e-10).sqrt();
        } else {
            l[(j, j)] = diag.sqrt();
        }

        for i in (j + 1)..n {
            let mut sum = 0.0;
            for k in 0..j {
                sum += l[(i, k)] * l[(j, k)];
            }
            l[(i, j)] = (a[(i, j)] - sum) / l[(j, j)];
        }
    }

    // Forward substitution: Ly = b
    let mut y = Col::zeros(n);
    for i in 0..n {
        let mut sum = b[i];
        for j in 0..i {
            sum -= l[(i, j)] * y[j];
        }
        y[i] = sum / l[(i, i)];
    }

    // Backward substitution: L'x = y
    let mut x = Col::zeros(n);
    for i in (0..n).rev() {
        let mut sum = y[i];
        for j in (i + 1)..n {
            sum -= l[(j, i)] * x[j];
        }
        x[i] = sum / l[(i, i)];
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spd() -> Vec<Vec<f64>> {
        vec![
            vec![4.0, 2.0, 0.6],
            vec![2.0, 5.0, 1.5],
            vec![0.6, 1.5, 3.0],
        ]
    }

    #[test]
    fn cholesky_reproduces_matrix() {
        let a = spd();
        let l = cholesky(&a).expect("SPD");
        for i in 0..3 {
            for j in 0..3 {
                if j > i {
                    assert_eq!(l[i][j], 0.0);
                }
                let llt: f64 = (0..3).map(|k| l[i][k] * l[j][k]).sum();
                assert!((llt - a[i][j]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn solve_and_inverse() {
        let a = spd();
        let l = cholesky(&a).unwrap();
        let b = [1.0, -2.0, 0.5];
        let x = chol_solve(&l, &b);
        for i in 0..3 {
            let ax: f64 = (0..3).map(|k| a[i][k] * x[k]).sum();
            assert!((ax - b[i]).abs() < 1e-12);
        }
        let inv = chol_inverse(&l);
        for i in 0..3 {
            for j in 0..3 {
                let e: f64 = (0..3).map(|k| a[i][k] * inv[k][j]).sum();
                assert!((e - if i == j { 1.0 } else { 0.0 }).abs() < 1e-12);
            }
        }
        let bm = vec![vec![1.0, 0.0], vec![-2.0, 1.0], vec![0.5, 2.0]];
        let xm = chol_solve_mat(&l, &bm);
        for i in 0..3 {
            assert_eq!(xm[i][0], x[i]);
        }
    }

    #[test]
    fn bad_pivot_is_rejected() {
        assert!(cholesky(&[vec![1.0, 2.0], vec![2.0, 1.0]]).is_none());
        assert!(cholesky(&[vec![0.0]]).is_none());
        assert!(cholesky(&[vec![f64::NAN]]).is_none());
        assert!(cholesky(&[vec![f64::INFINITY]]).is_none());
        assert!(cholesky_mat(&Mat::from_fn(1, 1, |_, _| -1.0)).is_none());
    }

    #[test]
    fn mat_entry_point_matches() {
        let a = spd();
        let l = cholesky(&a).unwrap();
        let lm = cholesky_mat(&Mat::from_fn(3, 3, |i, j| a[i][j])).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(lm[(i, j)], l[i][j]);
            }
        }
    }

    #[test]
    fn jittered_solve_matches_exact_solve() {
        let a = spd();
        let l = cholesky(&a).unwrap();
        let b = [1.0, -2.0, 0.5];
        let x = chol_solve(&l, &b);
        let xj = solve_spd_jittered(
            &Mat::from_fn(3, 3, |i, j| a[i][j]),
            &Col::from_fn(3, |i| b[i]),
        );
        for i in 0..3 {
            assert!((xj[i] - x[i]).abs() < 1e-12);
        }
    }
}
