//! Multi-degree-of-freedom Wald tests from a single fit and the
//! `rms::anova.rms` table (RMS §2.6, §4).
//!
//! Everything works from an estimate `β̂` and its covariance `V̂`, so any fit
//! qualifies, and a robust, cluster or bootstrap covariance can be passed in
//! place of the model-based one.
//!
//! * [`wald_test`] – `H₀: Cβ = r`, `W = (Cβ̂ − r)ᵀ (C V̂ Cᵀ)⁻¹ (Cβ̂ − r)` on
//!   `q = rows(C)` degrees of freedom, referred to `χ²(q)`, or as
//!   `F = W / q` to `F(q, df)` when the dispersion is estimated (as
//!   `car::linearHypothesis`).
//! * [`wald_chunk`] – the chunk test that a set of coefficients is zero.
//! * [`anova`] – the `anova.rms` table for a design's [`TermMap`]: for every
//!   predictor the test of all its columns including interactions, the test
//!   of its interactions and of its nonlinear terms; for every interaction
//!   the nonlinear-interaction tests; then the pooled `TOTAL NONLINEAR`,
//!   `TOTAL INTERACTION`, `TOTAL NONLINEAR + INTERACTION` and `TOTAL` tests.
//!   Rows, labels and the columns tested follow `anova.rms` (with
//!   `main.effect = FALSE`, `india = TRUE`, `indnl = TRUE`).
//!
//! `β̂` and `V̂` hold the slopes only (no intercept), in the column order of
//! the term map; see [`drop_intercept`].

use crate::design::TermMap;
use faer::linalg::solvers::Solve;
use faer::{Col, Mat};
use statrs::distribution::{ChiSquared, ContinuousCDF, FisherSnedecor};
use thiserror::Error;

/// Reference distribution of a Wald test.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WaldDistribution {
    /// `W ~ χ²(q)` (fixed dispersion, or large samples).
    ChiSquare,
    /// `W / q ~ F(q, df_residual)` (estimated dispersion, e.g. OLS).
    F {
        /// Residual degrees of freedom.
        df_residual: f64,
    },
}

/// Errors of a Wald test.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum WaldError {
    /// Inconsistent dimensions of `β̂`, `V̂`, `C` or `r`.
    #[error("dimension mismatch: {0}")]
    Dimension(String),
    /// A column index is out of range.
    #[error("column {0} is out of range")]
    Column(usize),
    /// `C V̂ Cᵀ` is singular.
    #[error("the covariance of the tested contrasts is singular")]
    Singular,
    /// No contrast to test.
    #[error("nothing to test")]
    Empty,
}

/// Result of a Wald test.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WaldTest {
    /// Wald χ² statistic `W`.
    pub chi_square: f64,
    /// Numerator degrees of freedom `q`.
    pub df: usize,
    /// `W / q` for an F test.
    pub f: Option<f64>,
    /// Residual degrees of freedom of an F test.
    pub df_residual: Option<f64>,
    /// p-value from the reference distribution.
    pub p_value: f64,
}

impl WaldTest {
    fn new(chi_square: f64, df: usize, dist: WaldDistribution) -> Self {
        let q = df as f64;
        let (f, df_residual, p_value) = match dist {
            WaldDistribution::ChiSquare => (
                None,
                None,
                ChiSquared::new(q).map_or(f64::NAN, |d| d.sf(chi_square)),
            ),
            WaldDistribution::F { df_residual } => {
                let f = chi_square / q;
                let p = FisherSnedecor::new(q, df_residual).map_or(f64::NAN, |d| d.sf(f));
                (Some(f), Some(df_residual), p)
            }
        };
        Self {
            chi_square,
            df,
            f,
            df_residual,
            p_value,
        }
    }

    /// `χ² − d.f.`, the chance-corrected measure of predictor importance of
    /// RMS §4.
    pub fn chi_square_minus_df(&self) -> f64 {
        self.chi_square - self.df as f64
    }
}

/// Wald test of `H₀: Cβ = r` (`r = 0` when `None`).
pub fn wald_test(
    beta: &Col<f64>,
    vcov: &Mat<f64>,
    c: &Mat<f64>,
    r: Option<&Col<f64>>,
    dist: WaldDistribution,
) -> Result<WaldTest, WaldError> {
    let p = beta.nrows();
    if vcov.nrows() != p || vcov.ncols() != p {
        return Err(WaldError::Dimension(format!(
            "beta has {p} entries, vcov is {}x{}",
            vcov.nrows(),
            vcov.ncols()
        )));
    }
    if c.ncols() != p {
        return Err(WaldError::Dimension(format!(
            "C has {} columns, beta has {p} entries",
            c.ncols()
        )));
    }
    let q = c.nrows();
    if q == 0 {
        return Err(WaldError::Empty);
    }
    if let Some(r) = r {
        if r.nrows() != q {
            return Err(WaldError::Dimension(format!(
                "r has {} entries, C has {q} rows",
                r.nrows()
            )));
        }
    }
    let mut d = c * beta;
    if let Some(r) = r {
        d -= r;
    }
    let cvc = c * vcov * c.transpose();
    let w = quadratic_form_inverse(&cvc, &d)?;
    Ok(WaldTest::new(w, q, dist))
}

/// Chunk test that the coefficients at `columns` are all zero.
pub fn wald_chunk(
    beta: &Col<f64>,
    vcov: &Mat<f64>,
    columns: &[usize],
    dist: WaldDistribution,
) -> Result<WaldTest, WaldError> {
    let p = beta.nrows();
    if vcov.nrows() != p || vcov.ncols() != p {
        return Err(WaldError::Dimension(format!(
            "beta has {p} entries, vcov is {}x{}",
            vcov.nrows(),
            vcov.ncols()
        )));
    }
    if columns.is_empty() {
        return Err(WaldError::Empty);
    }
    if let Some(&j) = columns.iter().find(|&&j| j >= p) {
        return Err(WaldError::Column(j));
    }
    let q = columns.len();
    let b = Col::from_fn(q, |a| beta[columns[a]]);
    let v = Mat::from_fn(q, q, |a, c| vcov[(columns[a], columns[c])]);
    Ok(WaldTest::new(quadratic_form_inverse(&v, &b)?, q, dist))
}

/// `dᵀ A⁻¹ d` for a symmetric positive-definite `A`.
fn quadratic_form_inverse(a: &Mat<f64>, d: &Col<f64>) -> Result<f64, WaldError> {
    let x = match a.llt(faer::Side::Lower) {
        Ok(llt) => llt.solve(d),
        Err(_) => return Err(WaldError::Singular),
    };
    let w: f64 = (0..d.nrows()).map(|i| d[i] * x[i]).sum();
    if w.is_finite() {
        Ok(w)
    } else {
        Err(WaldError::Singular)
    }
}

/// Split `(intercept, slopes)` off a coefficient vector and covariance whose
/// first entry is the intercept; returns the slopes and their covariance.
pub fn drop_intercept(beta: &Col<f64>, vcov: &Mat<f64>) -> (Col<f64>, Mat<f64>) {
    let p = beta.nrows().saturating_sub(1);
    (
        Col::from_fn(p, |j| beta[j + 1]),
        Mat::from_fn(p, p, |i, j| vcov[(i + 1, j + 1)]),
    )
}

/// What an ANOVA row tests (`vinfo` type of `anova.rms`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnovaRowKind {
    /// A predictor without interactions.
    MainEffect,
    /// A predictor and all interactions it is part of.
    CombinedEffect,
    /// An interaction term and the higher-order terms containing it.
    Interaction,
    /// All interactions of a predictor.
    CombinedInteractions,
    /// The nonlinear columns of a predictor (and of its interactions).
    Nonlinear,
    /// The nonlinear columns of an interaction.
    NonlinearInteraction,
    /// The doubly nonlinear columns of an interaction.
    DoublyNonlinearInteraction,
    /// Columns nonlinear in the first factor of an interaction.
    NonlinearInteractionFirst,
    /// Columns nonlinear in the second factor of an interaction.
    NonlinearInteractionSecond,
    /// All nonlinear columns.
    TotalNonlinear,
    /// All interaction columns.
    TotalInteraction,
    /// All nonlinear and interaction columns.
    TotalNonlinearInteraction,
    /// All columns.
    Total,
}

/// One row of an [`WaldAnovaTable`].
#[derive(Debug, Clone, PartialEq)]
pub struct AnovaRow {
    /// Label as printed by `anova.rms` (sub-rows start with a space).
    pub label: String,
    /// What the row tests.
    pub kind: AnovaRowKind,
    /// Term the row belongs to (`None` for the totals).
    pub term: Option<usize>,
    /// Slope columns tested.
    pub columns: Vec<usize>,
    /// The test (NaN statistics if the covariance block is singular).
    pub test: WaldTest,
    /// Partial sum of squares `χ²·σ²` (OLS with σ² given).
    pub partial_ss: Option<f64>,
}

/// `anova.rms`-style table of Wald tests.
#[derive(Debug, Clone, PartialEq)]
pub struct WaldAnovaTable {
    /// Rows in `anova.rms` order, ending with `TOTAL`.
    pub rows: Vec<AnovaRow>,
    /// `(d.f., SS, MS)` of the residual (the `ERROR` row), for an F test
    /// with σ² given.
    pub error: Option<(f64, f64, f64)>,
}

impl WaldAnovaTable {
    /// Row with the given (trimmed) label, first match.
    pub fn row(&self, label: &str) -> Option<&AnovaRow> {
        self.rows.iter().find(|r| r.label.trim() == label.trim())
    }
}

/// Options of [`anova`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AnovaOptions {
    /// Reference distribution of every row.
    pub distribution: WaldDistribution,
    /// Residual variance σ² (OLS): adds partial sums of squares and the
    /// `ERROR` row, as `anova.rms(ss = TRUE)`.
    pub sigma2: Option<f64>,
}

impl Default for AnovaOptions {
    fn default() -> Self {
        Self {
            distribution: WaldDistribution::ChiSquare,
            sigma2: None,
        }
    }
}

const HIGHER: &str = "(Factor+Higher Order Factors)";

/// The `rms::anova.rms` table of `β̂` (slopes, in term-map column order)
/// with covariance `V̂`.
pub fn anova(
    beta: &Col<f64>,
    vcov: &Mat<f64>,
    terms: &TermMap,
    options: &AnovaOptions,
) -> Result<WaldAnovaTable, WaldError> {
    let nc = terms.n_columns();
    if beta.nrows() != nc || vcov.nrows() != nc || vcov.ncols() != nc {
        return Err(WaldError::Dimension(format!(
            "the term map has {nc} columns, beta has {} and vcov is {}x{}",
            beta.nrows(),
            vcov.nrows(),
            vcov.ncols()
        )));
    }
    let dist = options.distribution;
    let test = |idx: &[usize]| {
        wald_chunk(beta, vcov, idx, dist).unwrap_or(WaldTest {
            chi_square: f64::NAN,
            df: idx.len(),
            f: None,
            df_residual: None,
            p_value: f64::NAN,
        })
    };
    let all = terms.terms();
    let mut rows: Vec<AnovaRow> = Vec::new();
    let push = |rows: &mut Vec<AnovaRow>,
                label: String,
                kind,
                term: Option<usize>,
                columns: Vec<usize>| {
        let t = test(&columns);
        rows.push(AnovaRow {
            label,
            kind,
            term,
            columns,
            test: t,
            partial_ss: None,
        });
    };
    let mut all_slopes = vec![false; nc];
    let mut all_ia = vec![false; nc];
    let mut all_nonlin = vec![false; nc];
    let (mut num_ia, mut num_nonlin) = (0usize, 0usize);

    // Factors of a term: itself for a main effect, its two factors otherwise.
    let factors = |t: usize| -> Vec<usize> {
        match &all[t].interaction {
            Some(ia) => vec![ia.factors.0, ia.factors.1],
            None => vec![t],
        }
    };

    for (i, term) in all.iter().enumerate() {
        let main = term.columns.clone();
        let nl = &term.nonlinear;
        let low = factors(i);
        for &c in &main {
            all_slopes[c] = true;
        }
        // Interaction terms (other than this one) containing all of `low`.
        let higher: Vec<usize> = (0..all.len())
            .filter(|&k| k != i && all[k].interaction.is_some())
            .filter(|&k| {
                let f = factors(k);
                low.iter().all(|x| f.contains(x))
            })
            .collect();
        let is_ia = term.interaction.is_some();
        let in_any = is_ia || !higher.is_empty();
        let mut nonlin_ia: Vec<usize> = Vec::new();

        if !in_any {
            push(
                &mut rows,
                term.name.clone(),
                AnovaRowKind::MainEffect,
                Some(i),
                main.clone(),
            );
        } else {
            let mut ia_index: Vec<usize> = Vec::new();
            for &k in &higher {
                let hk = &all[k];
                let comps = &hk.interaction.as_ref().expect("interaction").components;
                let fk = factors(k);
                ia_index.extend(&hk.columns);
                for x in &low {
                    let pos = fk.iter().position(|f| f == x).expect("factor");
                    for (c, comp) in hk.columns.iter().zip(comps) {
                        let nonlinear = if pos == 0 { comp.0 } else { comp.1 };
                        if nonlinear && !nonlin_ia.contains(c) {
                            nonlin_ia.push(*c);
                        }
                    }
                }
            }
            let idx: Vec<usize> = main.iter().chain(&ia_index).copied().collect();
            for &c in &idx {
                all_slopes[c] = true;
            }
            let kind = if is_ia {
                AnovaRowKind::Interaction
            } else {
                AnovaRowKind::CombinedEffect
            };
            push(
                &mut rows,
                format!("{}  {HIGHER}", term.name),
                kind,
                Some(i),
                idx,
            );
            if !is_ia && !higher.is_empty() {
                push(
                    &mut rows,
                    " All Interactions".into(),
                    AnovaRowKind::CombinedInteractions,
                    Some(i),
                    ia_index,
                );
            }
        }

        if nl.iter().any(|&v| v) && nl.iter().any(|&v| !v) {
            let idx: Vec<usize> = main
                .iter()
                .zip(nl)
                .filter(|(_, &v)| v)
                .map(|(&c, _)| c)
                .chain(nonlin_ia.iter().copied())
                .collect();
            num_nonlin += 1;
            for &c in &idx {
                all_nonlin[c] = true;
            }
            let label = if nonlin_ia.is_empty() {
                " Nonlinear".to_string()
            } else {
                format!(" Nonlinear {HIGHER}")
            };
            let kind = if is_ia {
                AnovaRowKind::NonlinearInteraction
            } else {
                AnovaRowKind::Nonlinear
            };
            push(&mut rows, label, kind, Some(i), idx);
        }

        if let Some(ia) = &term.interaction {
            num_ia += 1;
            for &c in &main {
                all_ia[c] = true;
            }
            let select = |f: &dyn Fn(&(bool, bool)) -> bool| -> Vec<usize> {
                main.iter()
                    .zip(&ia.components)
                    .filter(|(_, comp)| f(comp))
                    .map(|(&c, _)| c)
                    .collect()
            };
            let xy = select(&|c| c.0 || c.1);
            if !xy.is_empty() {
                num_nonlin += 1;
                for &c in &xy {
                    all_nonlin[c] = true;
                }
                push(
                    &mut rows,
                    " Nonlinear Interaction : f(A,B) vs. AB".into(),
                    AnovaRowKind::NonlinearInteraction,
                    Some(i),
                    xy.clone(),
                );
                let both = select(&|c| c.0 && c.1);
                if !both.is_empty() {
                    push(
                        &mut rows,
                        " f(A,B) vs. Af(B) + Bg(A)".into(),
                        AnovaRowKind::DoublyNonlinearInteraction,
                        Some(i),
                        both,
                    );
                }
                let x = select(&|c| c.0);
                if !x.is_empty() && x.len() != xy.len() {
                    push(
                        &mut rows,
                        format!(
                            " Nonlinear Interaction in {} vs. Af(B)",
                            all[ia.factors.0].name
                        ),
                        AnovaRowKind::NonlinearInteractionFirst,
                        Some(i),
                        x,
                    );
                }
                let y = select(&|c| c.1);
                if !y.is_empty() && y.len() != xy.len() {
                    push(
                        &mut rows,
                        format!(
                            " Nonlinear Interaction in {} vs. Bg(A)",
                            all[ia.factors.1].name
                        ),
                        AnovaRowKind::NonlinearInteractionSecond,
                        Some(i),
                        y,
                    );
                }
            }
        }
    }

    if rows.iter().all(|r| r.label.contains(HIGHER)) {
        for r in &mut rows {
            r.label = r.label.replace(HIGHER, "").trim_end().to_string();
        }
    }
    let flagged = |v: &[bool]| (0..nc).filter(|&j| v[j]).collect::<Vec<_>>();
    if num_nonlin > 1 {
        push(
            &mut rows,
            "TOTAL NONLINEAR".into(),
            AnovaRowKind::TotalNonlinear,
            None,
            flagged(&all_nonlin),
        );
    }
    if num_ia > 1 {
        push(
            &mut rows,
            "TOTAL INTERACTION".into(),
            AnovaRowKind::TotalInteraction,
            None,
            flagged(&all_ia),
        );
    }
    if num_nonlin > 0 && num_ia > 0 {
        let v: Vec<bool> = (0..nc).map(|j| all_nonlin[j] || all_ia[j]).collect();
        push(
            &mut rows,
            "TOTAL NONLINEAR + INTERACTION".into(),
            AnovaRowKind::TotalNonlinearInteraction,
            None,
            flagged(&v),
        );
    }
    let v: Vec<bool> = (0..nc).map(|j| all_slopes[j] || all_ia[j]).collect();
    push(
        &mut rows,
        "TOTAL".into(),
        AnovaRowKind::Total,
        None,
        flagged(&v),
    );

    let error = match (options.sigma2, dist) {
        (Some(s2), WaldDistribution::F { df_residual }) => {
            for r in &mut rows {
                r.partial_ss = Some(r.test.chi_square * s2);
            }
            Some((df_residual, s2 * df_residual, s2))
        }
        (Some(s2), WaldDistribution::ChiSquare) => {
            for r in &mut rows {
                r.partial_ss = Some(r.test.chi_square * s2);
            }
            None
        }
        _ => None,
    };
    Ok(WaldAnovaTable { rows, error })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_coefficient_wald_is_z_squared() {
        let beta = Col::from_fn(2, |j| [1.5, -0.4][j]);
        let vcov = Mat::from_fn(2, 2, |i, j| [[0.25, 0.01], [0.01, 0.04]][i][j]);
        let t = wald_chunk(&beta, &vcov, &[0], WaldDistribution::ChiSquare).unwrap();
        assert!((t.chi_square - 9.0).abs() < 1e-12);
        assert!((t.p_value - 2.699796063260207e-3).abs() < 1e-12);
        let c = Mat::from_fn(1, 2, |_, j| [1.0, 0.0][j]);
        let r = Col::from_fn(1, |_| 1.5);
        let t = wald_test(&beta, &vcov, &c, Some(&r), WaldDistribution::ChiSquare).unwrap();
        assert_eq!(t.chi_square, 0.0);
        assert!((t.chi_square_minus_df() + 1.0).abs() < 1e-12);
    }

    #[test]
    fn errors() {
        let beta = Col::from_fn(2, |_| 1.0);
        let vcov = Mat::<f64>::zeros(2, 2);
        let d = WaldDistribution::ChiSquare;
        assert_eq!(wald_chunk(&beta, &vcov, &[0], d), Err(WaldError::Singular));
        assert_eq!(wald_chunk(&beta, &vcov, &[], d), Err(WaldError::Empty));
        assert_eq!(wald_chunk(&beta, &vcov, &[2], d), Err(WaldError::Column(2)));
        assert!(wald_test(&beta, &vcov, &Mat::zeros(1, 3), None, d).is_err());
    }
}
