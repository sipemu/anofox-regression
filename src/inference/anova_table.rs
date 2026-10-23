//! Per-term ANOVA tables for OLS: sequential (type I) and partial (type II)
//! sums of squares.
//!
//! This is R's `anova(lm(...))` (type I) and `car::Anova(fit, type = 2)`
//! (type II) for a linear model without interactions. A *term* is one column
//! of the design or a group of columns (for example the dummies of a factor);
//! by default every column is its own term.
//!
//! ```text
//!   type I   SS_k = RSS(terms 1..k−1) − RSS(terms 1..k)     (in the given order)
//!   type II  SS_k = RSS(all terms but k) − RSS(all terms)
//!   df_k     = increase of the numerical rank that term k brings
//!   F_k      = (SS_k / df_k) / (RSS_full / df_res),   df_res = n − p_full
//! ```
//!
//! Every F test uses the residual mean square of the full model, so the type I
//! sums of squares add up to `TSS − RSS_full` and a type I table depends on the
//! order of the terms. Without interactions no term contains another, so the
//! type II sum of squares of a term is the drop-one comparison of the model
//! without it against the full model: the [`nested_f_test`] of the two, and
//! invariant to the term order. For the last term, types I and II agree.
//!
//! Degrees of freedom count the numerical rank, as R does. A term whose columns
//! are all aliased with the terms before it (type I) or with all other terms
//! (type II) adds no rank: its row has `df = 0`, `sum_sq = 0` and NaN `F` and
//! p-value (R drops such a term from `anova()`; `car::Anova` refuses aliased
//! models). With no residual degrees of freedom every F and p-value is NaN.
//!
//! Without an intercept the sums of squares are uncentred (the first type I
//! term is compared against the zero model), as in R's `y ~ 0 + ...`.
//!
//! [`nested_f_test`]: super::nested_f_test
//!
//! # Example
//!
//! ```
//! use anofox_regression::inference::{anova_table, AnovaTableOptions, AnovaType};
//! use faer::{Col, Mat};
//!
//! let x1 = [1.2, 2.5, 3.1, 4.8, 5.0, 6.7, 7.3, 8.1, 9.6, 10.2, 11.5, 12.9];
//! let x2 = [0.5, -1.2, 2.3, 0.8, -0.4, 1.9, -2.1, 0.3, 1.1, -0.7, 2.6, -1.5];
//! let y = [2.9, 4.1, 7.8, 8.2, 9.1, 14.0, 10.9, 14.6, 17.9, 17.1, 23.0, 20.4];
//! let x = Mat::from_fn(12, 2, |i, j| if j == 0 { x1[i] } else { x2[i] });
//! let y = Col::from_fn(12, |i| y[i]);
//!
//! let options = AnovaTableOptions {
//!     anova_type: AnovaType::II,
//!     ..Default::default()
//! };
//! // One term per column: anova(lm(y ~ x1 + x2)) in type II.
//! let table = anova_table(&x, &y, None, &options).unwrap();
//! assert_eq!(table.terms.len(), 2);
//! assert_eq!(table.residual_df, 9);
//! assert!(table.terms[1].p_value < 0.01);
//! ```

use super::nested_f::{fit_summary, FitSummary, NestedFTestOptions};
use crate::solvers::RegressionError;
use faer::{Col, Mat};
use statrs::distribution::{ContinuousCDF, FisherSnedecor};
use thiserror::Error;

/// Which sums of squares an ANOVA table reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AnovaType {
    /// Sequential sums of squares, each term adjusted for the terms before
    /// it (R's `anova(lm(...))`).
    #[default]
    I,
    /// Partial sums of squares, each term adjusted for all other terms
    /// (`car::Anova(fit, type = 2)` without interactions).
    II,
}

/// Errors from [`anova_table`].
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum AnovaTableError {
    /// `x` and `y` do not have the same number of rows.
    #[error("dimension mismatch: x has {x_rows} rows, y {y_len}")]
    DimensionMismatch {
        /// Rows of the design.
        x_rows: usize,
        /// Length of `y`.
        y_len: usize,
    },
    /// The terms do not partition the design columns: a term is empty, a
    /// column index is out of range, or a column is in no term or in two.
    #[error("invalid terms: {0}")]
    InvalidTerms(String),
    /// One of the OLS fits failed.
    #[error("{0}")]
    Fit(#[from] RegressionError),
}

/// Options for [`anova_table`].
#[derive(Debug, Clone)]
pub struct AnovaTableOptions {
    /// Sequential (type I, default) or partial (type II) sums of squares.
    pub anova_type: AnovaType,
    /// Fit an intercept in every model (default `true`).
    pub with_intercept: bool,
    /// Relative QR rank tolerance of the fits (default `1e-10`, as
    /// [`OlsRegressor`](crate::solvers::OlsRegressor)).
    pub rank_tolerance: f64,
}

impl Default for AnovaTableOptions {
    fn default() -> Self {
        Self {
            anova_type: AnovaType::I,
            with_intercept: true,
            rank_tolerance: 1e-10,
        }
    }
}

/// One term (row) of an ANOVA table.
#[derive(Debug, Clone, PartialEq)]
pub struct AnovaTerm {
    /// Position of the term in the term list (0-based).
    pub term: usize,
    /// Design columns of the term (0-based).
    pub columns: Vec<usize>,
    /// Degrees of freedom: the rank the term adds.
    pub df: usize,
    /// Sum of squares of the term (type I or II).
    pub sum_sq: f64,
    /// `sum_sq / df`; NaN when `df = 0`.
    pub mean_sq: f64,
    /// `mean_sq / residual_mean_sq`; NaN when `df = 0` or `residual_df = 0`.
    pub f_statistic: f64,
    /// Upper tail of `F(df, residual_df)` at `f_statistic`; NaN when it is.
    pub p_value: f64,
}

/// A per-term ANOVA table of an OLS fit.
#[derive(Debug, Clone, PartialEq)]
pub struct AnovaTable {
    /// The type of the sums of squares.
    pub anova_type: AnovaType,
    /// One row per term, in the order given.
    pub terms: Vec<AnovaTerm>,
    /// Residual degrees of freedom of the full model: `n − p_full`.
    pub residual_df: usize,
    /// Residual sum of squares of the full model.
    pub residual_sum_sq: f64,
    /// `residual_sum_sq / residual_df`; NaN when `residual_df = 0`.
    pub residual_mean_sq: f64,
    /// Rows used by the fits.
    pub n_observations: usize,
}

/// Check that `terms` partitions the `p` design columns.
fn validate_terms(terms: &[Vec<usize>], p: usize) -> Result<(), AnovaTableError> {
    let mut seen = vec![false; p];
    for (k, term) in terms.iter().enumerate() {
        if term.is_empty() {
            return Err(AnovaTableError::InvalidTerms(format!("term {k} is empty")));
        }
        for &c in term {
            if c >= p {
                return Err(AnovaTableError::InvalidTerms(format!(
                    "column {c} of term {k} is out of range for {p} columns"
                )));
            }
            if seen[c] {
                return Err(AnovaTableError::InvalidTerms(format!(
                    "column {c} is in more than one term"
                )));
            }
            seen[c] = true;
        }
    }
    match seen.iter().position(|s| !s) {
        Some(c) => Err(AnovaTableError::InvalidTerms(format!(
            "column {c} is in no term"
        ))),
        None => Ok(()),
    }
}

fn columns_of(x: &Mat<f64>, columns: &[usize]) -> Mat<f64> {
    Mat::from_fn(x.nrows(), columns.len(), |i, j| x[(i, columns[j])])
}

/// Build a term row from the RSS and rank of the models without and with it.
fn term_row(
    term: usize,
    columns: &[usize],
    without: &FitSummary,
    with: &FitSummary,
    residual_df: usize,
    residual_mean_sq: f64,
) -> AnovaTerm {
    let df = with.n_parameters.saturating_sub(without.n_parameters);
    // The larger model cannot have a larger RSS; clamp rounding noise at 0.
    let sum_sq = if df > 0 {
        (without.rss - with.rss).max(0.0)
    } else {
        0.0
    };
    let mean_sq = if df > 0 { sum_sq / df as f64 } else { f64::NAN };
    let f_statistic = if df > 0 && residual_df > 0 {
        mean_sq / residual_mean_sq
    } else {
        f64::NAN
    };
    let p_value = if f_statistic.is_nan() {
        f64::NAN
    } else {
        FisherSnedecor::new(df as f64, residual_df as f64)
            .map(|d| d.sf(f_statistic))
            .unwrap_or(f64::NAN)
    };
    AnovaTerm {
        term,
        columns: columns.to_vec(),
        df,
        sum_sq,
        mean_sq,
        f_statistic,
        p_value,
    }
}

/// Per-term ANOVA table of the OLS fit of `y` on `x`: R's `anova(lm(...))`
/// ([`AnovaType::I`]) or `car::Anova(fit, type = 2)` ([`AnovaType::II`]).
///
/// `terms` lists the design columns (0-based) of each term, in table order;
/// they must partition the columns of `x`. `None` makes every column its own
/// term, in column order. Type I fits the models with terms `1..k` for every
/// `k`; type II fits the full model and each model without one term.
///
/// Every F test divides by the full model's residual mean square. Degrees of
/// freedom count the numerical rank: a term that adds no rank (aliased with
/// the terms before it, or for type II with all other terms) gets `df = 0`,
/// `sum_sq = 0` and NaN `F` / p-value, and `residual_df = 0` makes every F
/// NaN. Without an intercept the sums of squares are uncentred. Fits that
/// cannot be computed (for example fewer rows than columns) return
/// [`AnovaTableError::Fit`].
pub fn anova_table(
    x: &Mat<f64>,
    y: &Col<f64>,
    terms: Option<&[Vec<usize>]>,
    options: &AnovaTableOptions,
) -> Result<AnovaTable, AnovaTableError> {
    if x.nrows() != y.nrows() {
        return Err(AnovaTableError::DimensionMismatch {
            x_rows: x.nrows(),
            y_len: y.nrows(),
        });
    }
    let p = x.ncols();
    let default_terms: Vec<Vec<usize>>;
    let terms = match terms {
        Some(t) => t,
        None => {
            default_terms = (0..p).map(|c| vec![c]).collect();
            &default_terms
        }
    };
    validate_terms(terms, p)?;

    let fit_options = NestedFTestOptions {
        with_intercept: options.with_intercept,
        rank_tolerance: options.rank_tolerance,
    };
    let fit = |columns: &[usize]| fit_summary(&columns_of(x, columns), y, &fit_options);

    let n = y.nrows();
    let all: Vec<usize> = terms.iter().flatten().copied().collect();
    let full = fit(&all)?;
    let residual_df = n.saturating_sub(full.n_parameters);
    let residual_mean_sq = if residual_df > 0 {
        full.rss / residual_df as f64
    } else {
        f64::NAN
    };

    let mut rows = Vec::with_capacity(terms.len());
    match options.anova_type {
        AnovaType::I => {
            let mut previous = fit(&[])?;
            let mut columns = Vec::with_capacity(p);
            for (k, term) in terms.iter().enumerate() {
                columns.extend_from_slice(term);
                let current = fit(&columns)?;
                rows.push(term_row(
                    k,
                    term,
                    &previous,
                    &current,
                    residual_df,
                    residual_mean_sq,
                ));
                previous = current;
            }
        }
        AnovaType::II => {
            for (k, term) in terms.iter().enumerate() {
                let others: Vec<usize> = terms
                    .iter()
                    .enumerate()
                    .filter(|&(j, _)| j != k)
                    .flat_map(|(_, t)| t.iter().copied())
                    .collect();
                let without = fit(&others)?;
                rows.push(term_row(
                    k,
                    term,
                    &without,
                    &full,
                    residual_df,
                    residual_mean_sq,
                ));
            }
        }
    }

    Ok(AnovaTable {
        anova_type: options.anova_type,
        terms: rows,
        residual_df,
        residual_sum_sq: full.rss,
        residual_mean_sq,
        n_observations: n,
    })
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
    fn invalid_terms_are_rejected() {
        let (x, y) = data();
        let opts = AnovaTableOptions::default();
        for terms in [
            vec![vec![0], vec![]],
            vec![vec![0], vec![2]],
            vec![vec![0], vec![0, 1]],
            vec![vec![1]],
        ] {
            assert!(matches!(
                anova_table(&x, &y, Some(&terms), &opts),
                Err(AnovaTableError::InvalidTerms(_))
            ));
        }
    }

    #[test]
    fn dimension_mismatch_is_rejected() {
        let (x, _) = data();
        let y = Col::from_fn(7, |i| i as f64);
        assert!(matches!(
            anova_table(&x, &y, None, &AnovaTableOptions::default()),
            Err(AnovaTableError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn type_i_sums_to_explained_ss_and_last_term_matches_type_ii() {
        let (x, y) = data();
        let t1 = anova_table(&x, &y, None, &AnovaTableOptions::default()).unwrap();
        let t2 = anova_table(
            &x,
            &y,
            None,
            &AnovaTableOptions {
                anova_type: AnovaType::II,
                ..Default::default()
            },
        )
        .unwrap();
        let mean = y.iter().sum::<f64>() / 8.0;
        let tss: f64 = y.iter().map(|v| (v - mean).powi(2)).sum();
        let explained: f64 = t1.terms.iter().map(|t| t.sum_sq).sum();
        assert!((explained + t1.residual_sum_sq - tss).abs() < 1e-10 * tss);
        assert!((t1.terms[1].sum_sq - t2.terms[1].sum_sq).abs() < 1e-10);
        assert_eq!(t1.residual_df, 5);
    }

    #[test]
    fn no_residual_df_gives_nan() {
        let x = Mat::from_fn(3, 2, |i, j| ((i + 1) * (j + 2)) as f64 + (i * i) as f64);
        let y = Col::from_fn(3, |i| [1.0, 3.0, 2.0][i]);
        let t = anova_table(&x, &y, None, &AnovaTableOptions::default()).unwrap();
        assert_eq!(t.residual_df, 0);
        assert!(t.residual_mean_sq.is_nan());
        assert!(t
            .terms
            .iter()
            .all(|r| r.f_statistic.is_nan() && r.p_value.is_nan()));
    }
}
