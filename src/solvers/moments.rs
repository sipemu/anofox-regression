//! Streaming moment accumulator for OLS / Ridge.
//!
//! For large panels (millions of rows) the explicit `N × p` design matrix is
//! often infeasible to materialise — but the closed-form OLS / Ridge solve
//! only depends on the rank-`p` *sufficient statistics*:
//!
//! ```text
//! n   = Σ rows                               (scalar)
//! sum_x = Σ xᵢ                               (length p)
//! sum_y = Σ yᵢ                               (scalar)
//! xtx   = Σ xᵢ xᵢᵀ                           (p × p)
//! xty   = Σ xᵢ yᵢ                            (length p)
//! ```
//!
//! These are additive over disjoint row chunks, so callers can accumulate
//! them across streamed batches (in memory, across files, across threads)
//! and then solve once at the end. See [`MomentAccumulator`] for the helper
//! and [`crate::solvers::OlsRegressor::fit_from_moments`] /
//! [`crate::solvers::RidgeRegressor::fit_from_moments`] for the matching
//! solvers.
//!
//! # Worked example
//!
//! ```rust,ignore
//! use anofox_regression::solvers::{MomentAccumulator, RidgeRegressor};
//!
//! let mut acc = MomentAccumulator::new(3); // 3 features
//! for (x_row, y) in stream_of_rows() {
//!     acc.push_row(&x_row, y)?;
//! }
//!
//! let fitted = RidgeRegressor::builder()
//!     .lambda(0.1)
//!     .with_intercept(true)
//!     .build()
//!     .fit_from_accumulator(&acc)?;
//! ```

#![allow(clippy::needless_range_loop)]

use crate::solvers::traits::RegressionError;
use faer::{Col, Mat};

/// Streaming accumulator for the moments `(n, Σx, Σy, XᵀX, Xᵀy)` required
/// by [`crate::solvers::OlsRegressor::fit_from_moments`] and
/// [`crate::solvers::RidgeRegressor::fit_from_moments`].
///
/// Cheap to construct, cheap to push into (`O(p²)` per row, `O(p)`
/// storage), trivially parallelisable via [`MomentAccumulator::merge`].
#[derive(Debug, Clone)]
pub struct MomentAccumulator {
    n_features: usize,
    n: usize,
    sum_x: Col<f64>,
    sum_y: f64,
    xtx: Mat<f64>,
    xty: Col<f64>,
}

impl MomentAccumulator {
    /// Allocate an accumulator for `n_features` predictors.
    pub fn new(n_features: usize) -> Self {
        Self {
            n_features,
            n: 0,
            sum_x: Col::zeros(n_features),
            sum_y: 0.0,
            xtx: Mat::zeros(n_features, n_features),
            xty: Col::zeros(n_features),
        }
    }

    /// Number of features the accumulator was sized for.
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// Number of rows accumulated so far.
    pub fn n(&self) -> usize {
        self.n
    }

    /// `Σ xᵢ` — column-wise sum of predictors.
    pub fn sum_x(&self) -> &Col<f64> {
        &self.sum_x
    }

    /// `Σ yᵢ` — sum of targets.
    pub fn sum_y(&self) -> f64 {
        self.sum_y
    }

    /// `Σ xᵢ xᵢᵀ` — uncentered Gram (`p × p`).
    pub fn xtx(&self) -> &Mat<f64> {
        &self.xtx
    }

    /// `Σ xᵢ yᵢ` — uncentered X-y cross moment.
    pub fn xty(&self) -> &Col<f64> {
        &self.xty
    }

    /// Add a single observation `(x_row, y)`.
    ///
    /// `x_row` must have length `n_features`.
    pub fn push_row(&mut self, x_row: &[f64], y: f64) -> Result<(), RegressionError> {
        if x_row.len() != self.n_features {
            return Err(RegressionError::DimensionMismatch {
                x_rows: 1,
                y_len: x_row.len(),
            });
        }
        // Rank-1 update of xtx, plus xty and the running sums.
        for i in 0..self.n_features {
            let xi = x_row[i];
            self.sum_x[i] += xi;
            self.xty[i] += xi * y;
            for j in 0..self.n_features {
                self.xtx[(i, j)] += xi * x_row[j];
            }
        }
        self.sum_y += y;
        self.n += 1;
        Ok(())
    }

    /// Add the moments from another accumulator (e.g. computed on a
    /// different worker thread) into this one.
    ///
    /// Errors if the two accumulators were sized for different feature
    /// counts.
    pub fn merge(&mut self, other: &Self) -> Result<(), RegressionError> {
        if other.n_features != self.n_features {
            return Err(RegressionError::DimensionMismatch {
                x_rows: self.n_features,
                y_len: other.n_features,
            });
        }
        for i in 0..self.n_features {
            self.sum_x[i] += other.sum_x[i];
            self.xty[i] += other.xty[i];
            for j in 0..self.n_features {
                self.xtx[(i, j)] += other.xtx[(i, j)];
            }
        }
        self.sum_y += other.sum_y;
        self.n += other.n;
        Ok(())
    }

    /// Reset the accumulator to its empty state without re-allocating.
    pub fn clear(&mut self) {
        self.n = 0;
        self.sum_y = 0.0;
        for i in 0..self.n_features {
            self.sum_x[i] = 0.0;
            self.xty[i] = 0.0;
            for j in 0..self.n_features {
                self.xtx[(i, j)] = 0.0;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pushing rows one-by-one must produce the same moments as computing
    /// them in one shot from the same data.
    #[test]
    fn push_row_matches_batched_moments() {
        let rows = [
            (vec![1.0, 2.0], 0.5),
            (vec![-1.0, 0.0], 1.5),
            (vec![0.3, 0.7], 2.0),
            (vec![2.0, -1.0], -0.4),
            (vec![0.1, 0.4], 0.9),
        ];
        let p = 2;
        let n = rows.len();

        // Streamed.
        let mut acc = MomentAccumulator::new(p);
        for (x, y) in &rows {
            acc.push_row(x, *y).unwrap();
        }

        // Batch reference.
        let mut sum_x = vec![0.0_f64; p];
        let mut sum_y = 0.0_f64;
        let mut xtx = vec![vec![0.0_f64; p]; p];
        let mut xty = vec![0.0_f64; p];
        for (x, y) in &rows {
            for i in 0..p {
                sum_x[i] += x[i];
                xty[i] += x[i] * y;
                for j in 0..p {
                    xtx[i][j] += x[i] * x[j];
                }
            }
            sum_y += y;
        }

        assert_eq!(acc.n(), n);
        for i in 0..p {
            assert!((acc.sum_x()[i] - sum_x[i]).abs() < 1e-12);
            assert!((acc.xty()[i] - xty[i]).abs() < 1e-12);
            for j in 0..p {
                assert!((acc.xtx()[(i, j)] - xtx[i][j]).abs() < 1e-12);
            }
        }
        assert!((acc.sum_y() - sum_y).abs() < 1e-12);
    }

    /// `merge` is exactly equivalent to pushing the merged rows into a
    /// single accumulator.
    #[test]
    fn merge_matches_single_accumulator() {
        let p = 3;
        let mut all = MomentAccumulator::new(p);
        let mut chunk_a = MomentAccumulator::new(p);
        let mut chunk_b = MomentAccumulator::new(p);

        let rows: Vec<(Vec<f64>, f64)> = (0..20)
            .map(|i| {
                let t = i as f64 / 5.0;
                (vec![t.sin(), (t * 0.3).cos(), t * 0.1], 0.5 + 0.2 * t)
            })
            .collect();

        for (idx, (x, y)) in rows.iter().enumerate() {
            all.push_row(x, *y).unwrap();
            if idx < 10 {
                chunk_a.push_row(x, *y).unwrap();
            } else {
                chunk_b.push_row(x, *y).unwrap();
            }
        }
        chunk_a.merge(&chunk_b).unwrap();

        assert_eq!(chunk_a.n(), all.n());
        for i in 0..p {
            assert!((chunk_a.sum_x()[i] - all.sum_x()[i]).abs() < 1e-12);
            assert!((chunk_a.xty()[i] - all.xty()[i]).abs() < 1e-12);
            for j in 0..p {
                assert!((chunk_a.xtx()[(i, j)] - all.xtx()[(i, j)]).abs() < 1e-12);
            }
        }
        assert!((chunk_a.sum_y() - all.sum_y()).abs() < 1e-12);
    }

    #[test]
    fn push_row_rejects_wrong_arity() {
        let mut acc = MomentAccumulator::new(2);
        let err = acc.push_row(&[1.0, 2.0, 3.0], 0.0).unwrap_err();
        assert!(matches!(err, RegressionError::DimensionMismatch { .. }));
    }

    #[test]
    fn merge_rejects_arity_mismatch() {
        let mut a = MomentAccumulator::new(2);
        let b = MomentAccumulator::new(3);
        let err = a.merge(&b).unwrap_err();
        assert!(matches!(err, RegressionError::DimensionMismatch { .. }));
    }
}
