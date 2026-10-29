//! Shrinkage and model approximation (RMS §4.5, §5.5.2, Ch 19).
//!
//! - [`heuristic_shrinkage`]: the van Houwelingen–le Cessie estimate of the
//!   calibration slope a new sample would show, `γ̂ = (LR χ² − p) / LR χ²`,
//!   and [`shrink_slopes`] to apply it.
//! - [`approximate_model`]: approximate a full model by a simpler one by
//!   regressing its linear predictor `Xβ̂` on a subset of the terms (OLS), with
//!   the R² of the approximation and the covariance of the approximate
//!   coefficients, `W V̂ Wᵀ` with `W = (ZᵀZ)⁻¹ZᵀX` (RMS §5.5.2).
//! - [`approximation_path`]: the Ch 19 table: delete terms in the order of
//!   `rms::fastbw` on the approximation `ols(lp ~ ., sigma = 1)` and report
//!   the R² of approximating the full linear predictor after each deletion.

use super::stepdown::{fastbw, StepRule, StepType};
use crate::design::TermMap;
use faer::linalg::solvers::Solve;
use faer::{Col, Mat, Side};

/// Errors of the approximation functions.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum ApproximationError {
    /// Inputs have inconsistent sizes.
    #[error("dimension mismatch: {0}")]
    Dimension(String),
    /// A term index is out of range.
    #[error("term {0} is out of range")]
    Term(usize),
    /// The design of the approximating model is singular.
    #[error("the approximating design is singular")]
    Singular,
}

/// Heuristic shrinkage factor `γ̂ = (LR χ² − p) / LR χ²` (van Houwelingen &
/// le Cessie 1990; RMS §4.5) from the model likelihood-ratio χ² and its
/// d.f. `p`. It estimates the calibration slope of the linear predictor in
/// new data; values well below 1 indicate overfitting. It is negative when
/// `LR χ² < p` and NaN when `LR χ² = 0`.
pub fn heuristic_shrinkage(lr_chisq: f64, df: f64) -> f64 {
    (lr_chisq - df) / lr_chisq
}

/// The coefficients with the slopes (all but the first `n_intercepts`)
/// multiplied by `gamma`. The intercepts are left as they are; re-estimate
/// them for the shrunken linear predictor (for OLS, `ȳ − x̄ᵀβ̂_shrunk`; for a
/// GLM, refit the intercept with the shrunken linear predictor as offset).
pub fn shrink_slopes(beta: &Col<f64>, n_intercepts: usize, gamma: f64) -> Col<f64> {
    Col::from_fn(beta.nrows(), |j| {
        if j < n_intercepts {
            beta[j]
        } else {
            gamma * beta[j]
        }
    })
}

/// An approximation of a full model by a subset of its terms.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelApproximation {
    /// Terms of the approximating model, in [`TermMap`] order.
    pub kept_terms: Vec<usize>,
    /// Slope columns of the approximating model (indices into `x`).
    pub columns: Vec<usize>,
    /// Intercept followed by the coefficients of `columns`.
    pub coefficients: Col<f64>,
    /// Covariance of `coefficients`, `W V̂ Wᵀ`.
    pub vcov: Mat<f64>,
    /// R² of the approximation of the full linear predictor.
    pub r_squared: f64,
}

/// One deletion of [`approximation_path`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ApproximationStep {
    /// Deleted term.
    pub term: usize,
    /// R² of the approximation by the terms still in the model.
    pub r_squared: f64,
}

/// OLS of `y` on `z`: coefficients and `(ZᵀZ)⁻¹`.
fn ols(z: &Mat<f64>, y: &Col<f64>) -> Result<(Col<f64>, Mat<f64>), ApproximationError> {
    let ztz = z.transpose() * z;
    let llt = ztz
        .llt(Side::Lower)
        .map_err(|_| ApproximationError::Singular)?;
    let inv = llt.solve(Mat::<f64>::identity(z.ncols(), z.ncols()));
    let coef = llt.solve(z.transpose() * y);
    Ok((coef, inv))
}

/// `[1, x[:, columns]]`.
fn design(x: &Mat<f64>, columns: &[usize]) -> Mat<f64> {
    Mat::from_fn(x.nrows(), columns.len() + 1, |i, j| {
        if j == 0 {
            1.0
        } else {
            x[(i, columns[j - 1])]
        }
    })
}

fn r_squared(z: &Mat<f64>, coef: &Col<f64>, y: &Col<f64>) -> f64 {
    let n = y.nrows();
    let mean = y.iter().sum::<f64>() / n as f64;
    let fitted = z * coef;
    let rss: f64 = (0..n).map(|i| (y[i] - fitted[i]).powi(2)).sum();
    let tss: f64 = (0..n).map(|i| (y[i] - mean).powi(2)).sum();
    1.0 - rss / tss
}

fn check(x: &Mat<f64>, lp: &Col<f64>, terms: &TermMap) -> Result<(), ApproximationError> {
    if lp.nrows() != x.nrows() {
        return Err(ApproximationError::Dimension(format!(
            "lp has {} rows, x has {}",
            lp.nrows(),
            x.nrows()
        )));
    }
    if terms.n_columns() != x.ncols() {
        return Err(ApproximationError::Dimension(format!(
            "the term map has {} columns, x has {}",
            terms.n_columns(),
            x.ncols()
        )));
    }
    Ok(())
}

/// Approximate a full model by the terms `keep` (RMS §5.5.2).
///
/// `x` holds the slope columns of the full model described by `terms`, `lp`
/// its linear predictor `β̂₀ + Xβ̂` (any fit: OLS, logistic, Cox, ...), and
/// `vcov` the covariance of `(β̂₀, β̂)`, intercept first. The approximation
/// is the OLS of `lp` on an intercept and the columns of `keep`; its
/// coefficients `γ̂ = W (β̂₀, β̂)` with `W = (ZᵀZ)⁻¹ZᵀX̃`, `X̃ = [1, X]`, have
/// covariance `W V̂ Wᵀ` (the `ols(lp ~ ., sigma = 1)` / `vcov` construction
/// of RMS §5.5.2).
pub fn approximate_model(
    x: &Mat<f64>,
    lp: &Col<f64>,
    vcov: &Mat<f64>,
    terms: &TermMap,
    keep: &[usize],
) -> Result<ModelApproximation, ApproximationError> {
    check(x, lp, terms)?;
    let p = x.ncols();
    if vcov.nrows() != p + 1 || vcov.ncols() != p + 1 {
        return Err(ApproximationError::Dimension(format!(
            "vcov is {}×{}, expected {}×{} (intercept first)",
            vcov.nrows(),
            vcov.ncols(),
            p + 1,
            p + 1
        )));
    }
    let mut kept_terms = keep.to_vec();
    kept_terms.sort_unstable();
    kept_terms.dedup();
    let mut columns = Vec::new();
    for &t in &kept_terms {
        let term = terms.terms().get(t).ok_or(ApproximationError::Term(t))?;
        columns.extend(&term.columns);
    }
    let z = design(x, &columns);
    let (coefficients, inv) = ols(&z, lp)?;
    let x_full = design(x, &(0..p).collect::<Vec<_>>());
    let w = &inv * (z.transpose() * &x_full);
    let vcov_approx = &w * vcov * w.transpose();
    Ok(ModelApproximation {
        kept_terms,
        columns,
        r_squared: r_squared(&z, &coefficients, lp),
        coefficients,
        vcov: vcov_approx,
    })
}

/// The approximation table of RMS Ch 19: the terms in the order in which
/// `rms::fastbw` deletes them from `ols(lp ~ <all terms>, sigma = 1)` (AIC
/// rule, every term deleted), each with the R² of approximating `lp` by the
/// terms that remain. The last step has R² 0 (intercept only).
pub fn approximation_path(
    x: &Mat<f64>,
    lp: &Col<f64>,
    terms: &TermMap,
) -> Result<Vec<ApproximationStep>, ApproximationError> {
    check(x, lp, terms)?;
    let all: Vec<usize> = (0..x.ncols()).collect();
    let z = design(x, &all);
    let (coef, inv) = ols(&z, lp)?;
    let order = fastbw(
        &coef,
        &inv,
        1,
        terms,
        StepRule::Aic {
            k: 2.0,
            threshold: f64::INFINITY,
        },
        StepType::Residual,
    );
    let mut remaining: Vec<usize> = (0..terms.terms().len()).collect();
    let mut steps = Vec::with_capacity(order.deleted.len());
    for deletion in &order.deleted {
        remaining.retain(|&t| t != deletion.term);
        let columns: Vec<usize> = remaining
            .iter()
            .flat_map(|&t| terms.terms()[t].columns.iter().copied())
            .collect();
        let r2 = if columns.is_empty() {
            0.0
        } else {
            let zk = design(x, &columns);
            let (ck, _) = ols(&zk, lp)?;
            r_squared(&zk, &ck, lp)
        };
        steps.push(ApproximationStep {
            term: deletion.term,
            r_squared: r2,
        });
    }
    Ok(steps)
}
