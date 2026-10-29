//! Penalized maximum likelihood à la rms (RMS §9.10): differential penalties
//! by term type, effective degrees of freedom per term, and `pentrace`'s
//! search for the penalty.
//!
//! The penalty is `½ βᵀPβ` with `P = A P₀ A`: `P₀` is [`base_penalty_matrix`]
//! (`rms:::Penalty.matrix`: the column variances on the diagonal, so the
//! penalty acts on standardised coefficients, and `I − 1/(L+1)` on the `L`
//! dummies of a categorical term), and `A = diag(√λ_j)` with `λ_j` the
//! penalty of the column's type ([`penalty_multipliers`],
//! `rms:::Penalty.setup`): simple (linear main effect), nonlinear (spline
//! terms of a main effect), interaction, or nonlinear interaction.
//! [`pentrace`] fits the model over a grid of penalties
//! ([`penalty_grid`]) with [`OrdinalRegressor`] (binary or ordinal logistic,
//! `lrm` / `orm`) and scores each by AIC, BIC and corrected AIC with the
//! effective d.f.; [`penalized_ols`] / [`pentrace_ols`] do the same for
//! least squares (`ols`).

use crate::design::TermMap;
use crate::solvers::ordinal::{FittedOrdinal, OrdinalLink, OrdinalRegressor};
use crate::solvers::traits::RegressionError;
use faer::{Col, Mat};

/// Penalties by column type (`lrm(penalty = list(simple, nonlinear,
/// interaction, nonlinear.interaction))`). Unset components fall back as in
/// rms: nonlinear to simple, interaction to nonlinear, nonlinear interaction
/// to interaction.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct PenaltySpec {
    /// Linear main-effect columns (and categorical dummies).
    pub simple: f64,
    /// Nonlinear columns of a main effect.
    pub nonlinear: Option<f64>,
    /// Linear interaction columns.
    pub interaction: Option<f64>,
    /// Nonlinear interaction columns.
    pub nonlinear_interaction: Option<f64>,
}

impl PenaltySpec {
    /// The same penalty for every column (`penalty = λ`).
    pub fn uniform(lambda: f64) -> Self {
        Self {
            simple: lambda,
            ..Self::default()
        }
    }

    /// `[simple, nonlinear, interaction, nonlinear interaction]` after the
    /// fallbacks.
    pub fn resolved(&self) -> [f64; 4] {
        let nonlinear = self.nonlinear.unwrap_or(self.simple);
        let interaction = self.interaction.unwrap_or(nonlinear);
        let nonlinear_interaction = self.nonlinear_interaction.unwrap_or(interaction);
        [self.simple, nonlinear, interaction, nonlinear_interaction]
    }
}

/// The penalty `λ_j` of every slope column (`rms:::Penalty.setup`'s
/// `multiplier`).
pub fn penalty_multipliers(terms: &TermMap, spec: &PenaltySpec) -> Vec<f64> {
    let [simple, nonlinear, interaction, nonlinear_interaction] = spec.resolved();
    let mut out = vec![0.0; terms.n_columns()];
    for term in terms.terms() {
        let is_interaction = term.interaction.is_some();
        for (&c, &nl) in term.columns.iter().zip(&term.nonlinear) {
            out[c] = match (nl, is_interaction) {
                (false, false) => simple,
                (true, false) => nonlinear,
                (false, true) => interaction,
                (true, true) => nonlinear_interaction,
            };
        }
    }
    out
}

/// `rms:::Penalty.matrix`: the sample variances of the columns of `x` on the
/// diagonal, except that the `L` dummy columns of each term in `categorical`
/// get the block `I − 1/(L + 1)` (penalising the differences between all
/// `L + 1` levels, including the reference).
pub fn base_penalty_matrix(
    x: &Mat<f64>,
    terms: &TermMap,
    categorical: &[usize],
) -> Result<Mat<f64>, RegressionError> {
    let (n, p) = (x.nrows(), x.ncols());
    if terms.n_columns() != p {
        return Err(RegressionError::NumericalError(format!(
            "the term map has {} columns, x has {p}",
            terms.n_columns()
        )));
    }
    if n < 2 {
        return Err(RegressionError::InsufficientObservations { needed: 2, got: n });
    }
    let mut pm = Mat::zeros(p, p);
    for j in 0..p {
        let mean = (0..n).map(|i| x[(i, j)]).sum::<f64>() / n as f64;
        pm[(j, j)] = (0..n).map(|i| (x[(i, j)] - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
    }
    for &t in categorical {
        let term = terms.terms().get(t).ok_or_else(|| {
            RegressionError::NumericalError(format!("categorical term {t} is out of range"))
        })?;
        let l = term.columns.len();
        for &a in &term.columns {
            for &b in &term.columns {
                pm[(a, b)] = f64::from(u8::from(a == b)) - 1.0 / (l + 1) as f64;
            }
        }
    }
    Ok(pm)
}

/// `A P₀ A` with `A = diag(√λ_j)` (the penalty matrix `lrm` builds from
/// `penalty = list(...)`).
pub fn penalty_matrix(
    x: &Mat<f64>,
    terms: &TermMap,
    categorical: &[usize],
    spec: &PenaltySpec,
) -> Result<Mat<f64>, RegressionError> {
    let base = base_penalty_matrix(x, terms, categorical)?;
    let a: Vec<f64> = penalty_multipliers(terms, spec)
        .iter()
        .map(|l| l.sqrt())
        .collect();
    Ok(Mat::from_fn(base.nrows(), base.ncols(), |r, c| {
        a[r] * base[(r, c)] * a[c]
    }))
}

/// Effective d.f. of each term: the sum of the per-column effective d.f.
/// (e.g. [`FittedOrdinal::effective_df_diagonal`]) over its columns.
pub fn effective_df_by_term(terms: &TermMap, diagonal: &Col<f64>) -> Vec<f64> {
    terms
        .terms()
        .iter()
        .map(|t| t.columns.iter().map(|&c| diagonal[c]).sum())
        .collect()
}

/// The penalties of `pentrace(fit, list(simple = .., nonlinear = .., ...))`:
/// every combination of the given values (`simple` varying fastest, as R's
/// `expand.grid`), keeping only those in which no component is smaller than
/// an earlier one (`complex.more = TRUE`). An empty slice leaves that
/// component unset (it falls back as in [`PenaltySpec::resolved`]).
pub fn penalty_grid(
    simple: &[f64],
    nonlinear: &[f64],
    interaction: &[f64],
    nonlinear_interaction: &[f64],
) -> Vec<PenaltySpec> {
    let opt = |v: &[f64]| -> Vec<Option<f64>> {
        if v.is_empty() {
            vec![None]
        } else {
            v.iter().map(|&x| Some(x)).collect()
        }
    };
    let simple = if simple.is_empty() {
        &[0.0][..]
    } else {
        simple
    };
    let mut out = Vec::new();
    for &nli in &opt(nonlinear_interaction) {
        for &ia in &opt(interaction) {
            for &nl in &opt(nonlinear) {
                for &s in simple {
                    let given: Vec<f64> = [Some(s), nl, ia, nli].into_iter().flatten().collect();
                    if given.windows(2).all(|w| w[1] >= w[0]) {
                        out.push(PenaltySpec {
                            simple: s,
                            nonlinear: nl,
                            interaction: ia,
                            nonlinear_interaction: nli,
                        });
                    }
                }
            }
        }
    }
    out
}

/// The criterion [`pentrace`] maximises.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PentraceCriterion {
    /// `LR χ² − 2 df (1 + (df + 1)/(n − df − 1))` (rms default).
    #[default]
    CorrectedAic,
    /// `LR χ² − 2 df`.
    Aic,
    /// `LR χ² − df log n`.
    Bic,
}

/// One penalty of [`pentrace`].
#[derive(Debug, Clone, PartialEq)]
pub struct PentraceRow {
    /// The penalty.
    pub penalty: PenaltySpec,
    /// Effective d.f. of the slopes.
    pub df: f64,
    /// Model LR χ² (unpenalized log-likelihood at the penalized estimate).
    pub lr_chisq: f64,
    /// `LR − 2 df`.
    pub aic: f64,
    /// `LR − df log n`.
    pub bic: f64,
    /// `LR − 2 df (1 + (df + 1)/(n − df − 1))`.
    pub aic_c: f64,
}

/// Result of [`pentrace`] / [`pentrace_ols`].
#[derive(Debug, Clone)]
pub struct Pentrace<F = FittedOrdinal> {
    /// One row per penalty, in the order given.
    pub rows: Vec<PentraceRow>,
    /// Index of the row with the largest criterion.
    pub best: usize,
    /// The fit at the best penalty.
    pub best_fit: F,
}

/// Fit every penalty with `fit` (returning the fit, its effective d.f. and
/// its LR χ²) and keep the best by `criterion`.
fn trace<F>(
    x: &Mat<f64>,
    terms: &TermMap,
    categorical: &[usize],
    penalties: &[PenaltySpec],
    criterion: PentraceCriterion,
    fit: impl Fn(Mat<f64>) -> Result<(F, f64, f64), RegressionError>,
) -> Result<Pentrace<F>, RegressionError> {
    if penalties.is_empty() {
        return Err(RegressionError::NumericalError(
            "no penalties to trace".to_string(),
        ));
    }
    let base = base_penalty_matrix(x, terms, categorical)?;
    let n = x.nrows() as f64;
    let mut rows = Vec::with_capacity(penalties.len());
    let mut best: Option<(usize, f64, F)> = None;
    for (i, spec) in penalties.iter().enumerate() {
        let a: Vec<f64> = penalty_multipliers(terms, spec)
            .iter()
            .map(|l| l.sqrt())
            .collect();
        let pm = Mat::from_fn(base.nrows(), base.ncols(), |r, c| {
            a[r] * base[(r, c)] * a[c]
        });
        let (fitted, df, lr) = fit(pm)?;
        let row = PentraceRow {
            penalty: *spec,
            df,
            lr_chisq: lr,
            aic: lr - 2.0 * df,
            bic: lr - df * n.ln(),
            aic_c: lr - 2.0 * df * (1.0 + (df + 1.0) / (n - df - 1.0)),
        };
        let value = match criterion {
            PentraceCriterion::CorrectedAic => row.aic_c,
            PentraceCriterion::Aic => row.aic,
            PentraceCriterion::Bic => row.bic,
        };
        if best.as_ref().is_none_or(|b| value > b.1) {
            best = Some((i, value, fitted));
        }
        rows.push(row);
    }
    let (best, _, best_fit) = best.expect("at least one penalty");
    Ok(Pentrace {
        rows,
        best,
        best_fit,
    })
}

/// `rms::pentrace` for binary and ordinal regression: fit the model of `x`
/// (slope columns described by `terms`, with the categorical terms
/// `categorical`) and `y` with every penalty of `penalties` and score it by
/// the effective d.f. The best penalty maximises `criterion`.
pub fn pentrace(
    x: &Mat<f64>,
    y: &Col<f64>,
    terms: &TermMap,
    categorical: &[usize],
    penalties: &[PenaltySpec],
    link: OrdinalLink,
    criterion: PentraceCriterion,
) -> Result<Pentrace, RegressionError> {
    trace(x, terms, categorical, penalties, criterion, |pm| {
        let fit = OrdinalRegressor::builder()
            .link(link)
            .penalty_matrix(pm)
            .build()
            .fit(x, y)?;
        let (df, lr) = (fit.effective_df(), fit.lr_chisq());
        Ok((fit, df, lr))
    })
}

/// A penalized least-squares fit (`rms::ols(penalty = )`, `lm.pfit`).
#[derive(Debug, Clone)]
pub struct PenalizedOls {
    /// Intercept followed by the slopes.
    pub coefficients: Col<f64>,
    /// `σ̂² (X̃ᵀX̃ + P̃)⁻¹` (rms `var.penalty = "simple"`), intercept first.
    pub vcov: Mat<f64>,
    /// `σ̂² Z X̃ᵀX̃ Z` with `Z = (X̃ᵀX̃ + P̃)⁻¹` (`var.penalty = "sandwich"`).
    pub vcov_sandwich: Mat<f64>,
    /// Residuals `y − X̃β̂`.
    pub residuals: Col<f64>,
    /// `σ̂ = √((SSE + β̂ᵀP̃β̂)/n)`.
    pub sigma: f64,
    /// `1 − SSE/SST`.
    pub r_squared: f64,
    /// `n(1 + log(SST/n)) − n log σ̂² − SSE/σ̂²` (`Model L.R.`).
    pub lr_chisq: f64,
    /// Effective d.f. per coefficient, intercept first:
    /// `diag((X̃ᵀX̃ / (SSE/n)) V)` (`effective.df.diagonal`).
    pub effective_df_diagonal: Col<f64>,
    /// Effective d.f. of the slopes, `Σ diag − 1` (`d.f.`).
    pub effective_df: f64,
}

/// Penalized least squares with the penalty `βᵀPβ` on the slopes (the
/// intercept is unpenalized), as `rms::ols(penalty = , penalty.matrix = )`.
pub fn penalized_ols(
    x: &Mat<f64>,
    y: &Col<f64>,
    penalty: &Mat<f64>,
) -> Result<PenalizedOls, RegressionError> {
    use faer::linalg::solvers::Solve;
    let (n, p) = (x.nrows(), x.ncols());
    if y.nrows() != n {
        return Err(RegressionError::DimensionMismatch {
            x_rows: n,
            y_len: y.nrows(),
        });
    }
    if penalty.nrows() != p || penalty.ncols() != p {
        return Err(RegressionError::NumericalError(format!(
            "the penalty matrix is {}×{}, expected {p}×{p}",
            penalty.nrows(),
            penalty.ncols()
        )));
    }
    if n < 2 {
        return Err(RegressionError::InsufficientObservations { needed: 2, got: n });
    }
    let xt = Mat::from_fn(n, p + 1, |i, j| if j == 0 { 1.0 } else { x[(i, j - 1)] });
    let pm = Mat::from_fn(p + 1, p + 1, |r, c| {
        if r == 0 || c == 0 {
            0.0
        } else {
            penalty[(r - 1, c - 1)]
        }
    });
    let xpx = xt.transpose() * &xt;
    let a = &xpx + &pm;
    let z = a.partial_piv_lu().solve(Mat::<f64>::identity(p + 1, p + 1));
    let coefficients = &z * (xt.transpose() * y);
    let residuals = y - &xt * &coefficients;
    let sse: f64 = residuals.iter().map(|r| r * r).sum();
    let quad: f64 = (0..=p)
        .map(|r| {
            (0..=p)
                .map(|c| coefficients[r] * pm[(r, c)] * coefficients[c])
                .sum::<f64>()
        })
        .sum();
    let nf = n as f64;
    let s2 = (sse + quad) / nf;
    let vcov = Mat::from_fn(p + 1, p + 1, |r, c| s2 * z[(r, c)]);
    let zxz = &z * &xpx * &z;
    let vcov_sandwich = Mat::from_fn(p + 1, p + 1, |r, c| s2 * zxz[(r, c)]);
    let mean = y.iter().sum::<f64>() / nf;
    let sst: f64 = y.iter().map(|v| (v - mean).powi(2)).sum();
    let lr_chisq = nf * (1.0 + (sst / nf).ln()) - nf * s2.ln() - sse / s2;
    let s2_unpen = sse / nf;
    let xv = &xpx * &vcov;
    let effective_df_diagonal = Col::from_fn(p + 1, |j| xv[(j, j)] / s2_unpen);
    let effective_df = effective_df_diagonal.iter().sum::<f64>() - 1.0;
    Ok(PenalizedOls {
        coefficients,
        vcov,
        vcov_sandwich,
        residuals,
        sigma: s2.sqrt(),
        r_squared: 1.0 - sse / sst,
        lr_chisq,
        effective_df_diagonal,
        effective_df,
    })
}

/// [`pentrace`] for least squares (`pentrace` on an `ols` fit), with the
/// d.f. and LR χ² of [`penalized_ols`].
pub fn pentrace_ols(
    x: &Mat<f64>,
    y: &Col<f64>,
    terms: &TermMap,
    categorical: &[usize],
    penalties: &[PenaltySpec],
    criterion: PentraceCriterion,
) -> Result<Pentrace<PenalizedOls>, RegressionError> {
    trace(x, terms, categorical, penalties, criterion, |pm| {
        let fit = penalized_ols(x, y, &pm)?;
        let (df, lr) = (fit.effective_df, fit.lr_chisq);
        Ok((fit, df, lr))
    })
}
