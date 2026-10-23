//! Nested-model F test: does a full linear model fit significantly better
//! than a reduced model whose design it contains?
//!
//! This is R's `anova(lm(reduced), lm(full))` for two nested OLS fits on the
//! same rows:
//!
//! ```text
//!   F = ((RSS_r − RSS_f) / df1) / (RSS_f / df2),   F ~ F(df1, df2) under H0
//!   df1 = p_f − p_r      (parameters the full model adds)
//!   df2 = n − p_f        (residual df of the full model)
//! ```
//!
//! `p_r` and `p_f` are the **numerical ranks** of the two designs (plus the
//! intercept), as R uses `df.residual`, so rank-deficient designs give the same
//! answer as R: aliased columns add no parameters. When the full model adds no
//! rank (`df1 = 0`) or leaves no residual df (`df2 = 0`), the statistic and
//! p-value are NaN, as R reports `NA`.
//!
//! An empty reduced design is the intercept-only model (or, without an
//! intercept, the zero model), so the test then reproduces the overall F test
//! of the full model.
//!
//! # Nesting
//!
//! The reduced design must lie in the column space of the full design. This is
//! checked numerically: regressing each reduced column on the full design must
//! leave a residual sum of squares of at most `rank_tolerance · ‖x_k‖²`, so a
//! rescaled or shifted copy of a full column also counts. Otherwise
//! [`NestedFTestError::NotNested`] is returned. [`nested_f_test_indices`] picks
//! the reduced columns out of the full design, so nesting holds by construction.
//!
//! The adjusted R² values follow `summary.lm` (centred TSS with an intercept,
//! uncentred without).
//!
//! # Example
//!
//! ```
//! use anofox_regression::inference::{nested_f_test_indices, NestedFTestOptions};
//! use faer::{Col, Mat};
//!
//! let x1 = [1.2, 2.5, 3.1, 4.8, 5.0, 6.7, 7.3, 8.1, 9.6, 10.2, 11.5, 12.9];
//! let x2 = [0.5, -1.2, 2.3, 0.8, -0.4, 1.9, -2.1, 0.3, 1.1, -0.7, 2.6, -1.5];
//! let y = [2.9, 4.1, 7.8, 8.2, 9.1, 14.0, 10.9, 14.6, 17.9, 17.1, 23.0, 20.4];
//! let x = Mat::from_fn(12, 2, |i, j| if j == 0 { x1[i] } else { x2[i] });
//! let y = Col::from_fn(12, |i| y[i]);
//!
//! // y ~ x1 against y ~ x1 + x2
//! let t = nested_f_test_indices(&x, &y, &[0], &NestedFTestOptions::default()).unwrap();
//! assert_eq!((t.df1, t.df2), (1, 9));
//! assert!(t.p_value < 0.01);
//! ```

use crate::solvers::{FittedRegressor, OlsRegressor, RegressionError, Regressor};
use faer::{Col, Mat};
use statrs::distribution::{ContinuousCDF, FisherSnedecor};
use thiserror::Error;

/// Errors from [`nested_f_test`] and [`nested_f_test_indices`].
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum NestedFTestError {
    /// `x_reduced`, `x_full` and `y` do not have the same number of rows.
    #[error(
        "dimension mismatch: x_reduced has {reduced_rows} rows, x_full {full_rows}, y {y_len}"
    )]
    DimensionMismatch {
        /// Rows of the reduced design.
        reduced_rows: usize,
        /// Rows of the full design.
        full_rows: usize,
        /// Length of `y`.
        y_len: usize,
    },
    /// A reduced column index is out of range or repeated.
    #[error("invalid reduced column index {index} for a full design with {n_columns} columns")]
    InvalidIndex {
        /// The offending (0-based) index.
        index: usize,
        /// Number of columns of the full design.
        n_columns: usize,
    },
    /// The reduced design is not contained in the column space of the full
    /// design, so the models are not nested.
    #[error(
        "models are not nested: reduced column {column} is not in the span of the full design"
    )]
    NotNested {
        /// (0-based) reduced column that raises the rank of the full design.
        column: usize,
    },
    /// One of the OLS fits failed.
    #[error("{0}")]
    Fit(#[from] RegressionError),
}

/// Options for [`nested_f_test`].
#[derive(Debug, Clone)]
pub struct NestedFTestOptions {
    /// Fit an intercept in both models (default `true`).
    pub with_intercept: bool,
    /// Relative QR rank tolerance of both fits and of the nesting check
    /// (default `1e-10`, as [`OlsRegressor`]).
    pub rank_tolerance: f64,
}

impl Default for NestedFTestOptions {
    fn default() -> Self {
        Self {
            with_intercept: true,
            rank_tolerance: 1e-10,
        }
    }
}

/// Result of a nested-model F test.
#[derive(Debug, Clone, PartialEq)]
pub struct NestedFTest {
    /// `((RSS_r − RSS_f) / df1) / (RSS_f / df2)`; NaN when `df1` or `df2` is 0.
    pub f_statistic: f64,
    /// Parameters the full model adds: `p_full − p_reduced`.
    pub df1: usize,
    /// Residual degrees of freedom of the full model: `n − p_full`.
    pub df2: usize,
    /// Upper tail of `F(df1, df2)` at `f_statistic`; NaN when the statistic is.
    pub p_value: f64,
    /// Residual sum of squares of the reduced model.
    pub rss_reduced: f64,
    /// Residual sum of squares of the full model.
    pub rss_full: f64,
    /// Adjusted R² of the reduced model.
    pub adj_r_squared_reduced: f64,
    /// Adjusted R² of the full model.
    pub adj_r_squared_full: f64,
    /// Estimated parameters of the reduced model (rank, plus the intercept).
    pub n_parameters_reduced: usize,
    /// Estimated parameters of the full model (rank, plus the intercept).
    pub n_parameters_full: usize,
    /// Rows used by both fits.
    pub n_observations: usize,
}

impl NestedFTest {
    /// `adj_r_squared_full − adj_r_squared_reduced`.
    pub fn delta_adj_r_squared(&self) -> f64 {
        self.adj_r_squared_full - self.adj_r_squared_reduced
    }
}

/// Summary of one OLS fit: RSS, adjusted R² and the number of parameters.
pub(super) struct FitSummary {
    pub(super) rss: f64,
    pub(super) adj_r_squared: f64,
    pub(super) n_parameters: usize,
}

pub(super) fn fit_summary(
    x: &Mat<f64>,
    y: &Col<f64>,
    options: &NestedFTestOptions,
) -> Result<FitSummary, RegressionError> {
    if x.ncols() == 0 {
        return Ok(empty_model_summary(y, options.with_intercept));
    }
    let fitted = OlsRegressor::builder()
        .with_intercept(options.with_intercept)
        .rank_tolerance(options.rank_tolerance)
        .compute_inference(false)
        .build()
        .fit(x, y);
    let fitted = match fitted {
        Ok(f) => f,
        // Without an intercept, an all-zero design is the zero model.
        Err(RegressionError::AllFeaturesConstant) if !options.with_intercept => {
            return Ok(empty_model_summary(y, false));
        }
        Err(e) => return Err(e),
    };
    let r = fitted.result();
    Ok(FitSummary {
        rss: r.rss(),
        adj_r_squared: r.adj_r_squared,
        n_parameters: r.n_parameters,
    })
}

/// The model with no regressors: the mean (intercept) or zero.
fn empty_model_summary(y: &Col<f64>, with_intercept: bool) -> FitSummary {
    let n = y.nrows();
    let center = if with_intercept && n > 0 {
        y.iter().sum::<f64>() / n as f64
    } else {
        0.0
    };
    let residuals = Col::from_fn(n, |i| y[i] - center);
    let n_parameters = usize::from(with_intercept);
    let stats = crate::solvers::fit_stats::linear_fit_stats(
        y,
        &residuals,
        None,
        with_intercept,
        n_parameters,
        false,
    );
    FitSummary {
        rss: residuals.iter().map(|r| r * r).sum(),
        adj_r_squared: stats.adj_r_squared,
        n_parameters,
    }
}

/// Check that every reduced column lies in the span of the full design (and
/// the intercept): regressing it on the full design must leave a residual sum
/// of squares of at most `rank_tolerance · ‖x_k‖²`.
fn check_nested(
    x_reduced: &Mat<f64>,
    x_full: &Mat<f64>,
    options: &NestedFTestOptions,
) -> Result<(), NestedFTestError> {
    for k in 0..x_reduced.ncols() {
        let column = Col::from_fn(x_reduced.nrows(), |i| x_reduced[(i, k)]);
        let norm_sq: f64 = column.iter().map(|v| v * v).sum();
        let rss = fit_summary(x_full, &column, options)?.rss;
        if rss > options.rank_tolerance * norm_sq {
            return Err(NestedFTestError::NotNested { column: k });
        }
    }
    Ok(())
}

/// Nested-model F test of the OLS fit of `y` on `x_reduced` against the fit on
/// `x_full` (R's `anova(lm(y ~ X_r), lm(y ~ X_f))`).
///
/// `F = ((RSS_r − RSS_f) / df1) / (RSS_f / df2)` with `df1 = p_full −
/// p_reduced` and `df2 = n − p_full`, where the parameter counts are the
/// numerical ranks plus the intercept (R's `df.residual`); `df1 = 0` or
/// `df2 = 0` gives a NaN statistic and p-value, as R reports `NA`.
///
/// `x_reduced` may have zero columns (the intercept-only model, so the test is
/// the overall F test). Its columns must lie in the span of `x_full`: each is
/// regressed on `x_full`, and a residual sum of squares above
/// `rank_tolerance · ‖x_k‖²` returns [`NestedFTestError::NotNested`]. Fits that
/// cannot be computed (for example fewer rows than full-design columns)
/// return [`NestedFTestError::Fit`].
pub fn nested_f_test(
    x_reduced: &Mat<f64>,
    x_full: &Mat<f64>,
    y: &Col<f64>,
    options: &NestedFTestOptions,
) -> Result<NestedFTest, NestedFTestError> {
    if x_reduced.nrows() != y.nrows() || x_full.nrows() != y.nrows() {
        return Err(NestedFTestError::DimensionMismatch {
            reduced_rows: x_reduced.nrows(),
            full_rows: x_full.nrows(),
            y_len: y.nrows(),
        });
    }
    let n = y.nrows();
    let full = fit_summary(x_full, y, options)?;
    let reduced = fit_summary(x_reduced, y, options)?;
    check_nested(x_reduced, x_full, options)?;

    let df1 = full.n_parameters.saturating_sub(reduced.n_parameters);
    let df2 = n.saturating_sub(full.n_parameters);
    let f_statistic = if df1 > 0 && df2 > 0 {
        // The full fit cannot have a larger RSS; clamp rounding noise at 0.
        let gain = (reduced.rss - full.rss).max(0.0);
        (gain / df1 as f64) / (full.rss / df2 as f64)
    } else {
        f64::NAN
    };
    let p_value = if f_statistic.is_nan() {
        f64::NAN
    } else {
        FisherSnedecor::new(df1 as f64, df2 as f64)
            .map(|d| d.sf(f_statistic))
            .unwrap_or(f64::NAN)
    };

    Ok(NestedFTest {
        f_statistic,
        df1,
        df2,
        p_value,
        rss_reduced: reduced.rss,
        rss_full: full.rss,
        adj_r_squared_reduced: reduced.adj_r_squared,
        adj_r_squared_full: full.adj_r_squared,
        n_parameters_reduced: reduced.n_parameters,
        n_parameters_full: full.n_parameters,
        n_observations: n,
    })
}

/// [`nested_f_test`] with the reduced design given as (0-based) column indices
/// of `x_full`, so the models are nested by construction. An empty `reduced`
/// is the intercept-only model.
pub fn nested_f_test_indices(
    x_full: &Mat<f64>,
    y: &Col<f64>,
    reduced: &[usize],
    options: &NestedFTestOptions,
) -> Result<NestedFTest, NestedFTestError> {
    let p = x_full.ncols();
    for (k, &index) in reduced.iter().enumerate() {
        if index >= p || reduced[..k].contains(&index) {
            return Err(NestedFTestError::InvalidIndex {
                index,
                n_columns: p,
            });
        }
    }
    let x_reduced = Mat::from_fn(x_full.nrows(), reduced.len(), |i, k| {
        x_full[(i, reduced[k])]
    });
    nested_f_test(&x_reduced, x_full, y, options)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data() -> (Mat<f64>, Col<f64>) {
        let x1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let x2 = [0.3, -0.2, 0.9, 0.1, -0.5, 0.4, 0.8, -0.7];
        let y = [1.1, 2.3, 2.8, 4.4, 4.9, 6.2, 7.1, 7.6];
        (
            Mat::from_fn(8, 2, |i, j| if j == 0 { x1[i] } else { x2[i] }),
            Col::from_fn(8, |i| y[i]),
        )
    }

    #[test]
    fn invalid_indices_are_rejected() {
        let (x, y) = data();
        let opts = NestedFTestOptions::default();
        assert!(matches!(
            nested_f_test_indices(&x, &y, &[2], &opts),
            Err(NestedFTestError::InvalidIndex { index: 2, .. })
        ));
        assert!(matches!(
            nested_f_test_indices(&x, &y, &[0, 0], &opts),
            Err(NestedFTestError::InvalidIndex { index: 0, .. })
        ));
    }

    #[test]
    fn non_nested_designs_are_rejected() {
        let (x, y) = data();
        let x_full = Mat::from_fn(8, 1, |i, _| x[(i, 0)]);
        let x_reduced = Mat::from_fn(8, 1, |i, _| x[(i, 1)]);
        let r = nested_f_test(&x_reduced, &x_full, &y, &NestedFTestOptions::default());
        assert!(matches!(r, Err(NestedFTestError::NotNested { column: 0 })));
    }

    #[test]
    fn rescaled_reduced_column_is_nested() {
        let (x, y) = data();
        let x_reduced = Mat::from_fn(8, 1, |i, _| 2.0 * x[(i, 0)] + 1.0);
        let r = nested_f_test(&x_reduced, &x, &y, &NestedFTestOptions::default()).unwrap();
        assert_eq!(r.df1, 1);
    }

    #[test]
    fn dimension_mismatch_is_rejected() {
        let (x, _) = data();
        let y = Col::from_fn(7, |i| i as f64);
        assert!(matches!(
            nested_f_test_indices(&x, &y, &[0], &NestedFTestOptions::default()),
            Err(NestedFTestError::DimensionMismatch { .. })
        ));
    }
}
