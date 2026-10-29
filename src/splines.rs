//! Regression-spline bases for ordinary design matrices (RMS §2.4).
//!
//! Unpenalised bases whose columns go straight into an OLS / GLM / AFT
//! design, as `rms::rcs` and `rms::lsp` do in R:
//!
//! * [`Rcs`] – restricted (natural) cubic spline in Harrell's truncated-power
//!   parametrisation (`Hmisc::rcspline.eval`), linear beyond the outer knots.
//!   With `k` knots it has `k − 1` columns: `x` and `k − 2` nonlinear terms.
//!   Default knots ([`rcs_default_knots`]) follow `rcspline.eval`: the
//!   quantiles of RMS Table 2.3 ([`default_knot_quantiles`]), the outer knots
//!   moved to the 5th smallest / largest value when `n < 100`, and special
//!   handling of heavily tied extremes.
//! * [`Lsp`] – linear spline `x, (x − t₁)₊, …` (`rms::lsp`).
//! * [`restricted_interaction`] – `rms::%ia%`: the products of two bases
//!   without the doubly nonlinear terms (RMS §2.7.2).
//!
//! Every basis is returned as a [`Basis`] carrying column names (`x`, `x'`,
//! `x''`, … as rms) and a `nonlinear` flag per column, which is what a chunk
//! test of nonlinearity needs. Keep the [`Rcs`] / [`Lsp`] value to rebuild the
//! same basis on new data for prediction. [`Rcs::restate`] gives the
//! algebraic form of a fitted spline (`Hmisc::rcspline.restate`).
//!
//! # Example
//!
//! ```rust
//! use anofox_regression::splines::Rcs;
//!
//! let x: Vec<f64> = (0..200).map(|i| (i as f64 * 0.37).sin() * 3.0 + i as f64 * 0.01).collect();
//! let rcs = Rcs::from_data(&x, 4).unwrap();          // knots at the default quantiles
//! let basis = rcs.basis(&x, "x");                    // 200 × 3: x, x', x''
//! assert_eq!(basis.names, ["x", "x'", "x''"]);
//! assert_eq!(basis.nonlinear, [false, true, true]);
//! let new = rcs.basis(&[0.0, 1.0], "x");             // same knots for prediction
//! assert_eq!(new.columns.nrows(), 2);
//! ```

use faer::Mat;
use thiserror::Error;

/// Errors from building a spline basis.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum SplineError {
    /// Default knots need at least 6 non-missing observations.
    #[error("knots not specified, and < 6 non-missing observations")]
    TooFewObservations,
    /// A restricted cubic spline needs at least 3 knots.
    #[error("need at least 3 knots, got {0}")]
    TooFewKnots(usize),
    /// Knots must be finite.
    #[error("knots must be finite")]
    NonFiniteKnots,
    /// Fewer than 3 distinct knots could be placed (too few distinct values).
    #[error("fewer than 3 unique knots could be placed")]
    TooFewUniqueKnots,
    /// A coefficient vector had the wrong length.
    #[error("expected {expected} coefficients, got {got}")]
    CoefficientLength { expected: usize, got: usize },
    /// Names or flags do not match the number of columns.
    #[error("expected {expected} names / flags, got {got}")]
    Length { expected: usize, got: usize },
    /// Bases of a product have different row counts.
    #[error("bases have {0} and {1} rows")]
    RowMismatch(usize, usize),
}

/// Columns of a basis with their names and nonlinearity flags.
#[derive(Debug, Clone, PartialEq)]
pub struct Basis {
    /// `n × m` basis matrix.
    pub columns: Mat<f64>,
    /// Column names, rms style (`x`, `x'`, `x''`, …; `a * b` for products).
    pub names: Vec<String>,
    /// `true` for columns that carry nonlinear (or doubly nonlinear) terms.
    pub nonlinear: Vec<bool>,
}

impl Basis {
    /// Basis from columns, names and nonlinearity flags (e.g. dummy columns
    /// of a factor, all linear).
    pub fn new(
        columns: Mat<f64>,
        names: Vec<String>,
        nonlinear: Vec<bool>,
    ) -> Result<Self, SplineError> {
        let m = columns.ncols();
        for got in [names.len(), nonlinear.len()] {
            if got != m {
                return Err(SplineError::Length { expected: m, got });
            }
        }
        Ok(Self {
            columns,
            names,
            nonlinear,
        })
    }

    /// A single linear column.
    pub fn linear(x: &[f64], name: &str) -> Self {
        Self {
            columns: Mat::from_fn(x.len(), 1, |i, _| x[i]),
            names: vec![name.to_string()],
            nonlinear: vec![false],
        }
    }

    /// Number of columns.
    pub fn n_columns(&self) -> usize {
        self.columns.ncols()
    }

    /// Indices of the nonlinear columns (the chunk tested for linearity).
    pub fn nonlinear_columns(&self) -> Vec<usize> {
        (0..self.nonlinear.len())
            .filter(|&j| self.nonlinear[j])
            .collect()
    }
}

/// Normalisation of the nonlinear terms (`norm` of `Hmisc::rcspline.eval`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RcsNorm {
    /// No normalisation (`norm = 0`).
    None,
    /// Divide by `(t_k − t_{k−1})³` (`norm = 1`).
    LastInterval,
    /// Divide by `(t_k − t₁)²` (`norm = 2`, the default of `rcspline.eval`
    /// and `rms::rcs`), which puts the terms on the scale of `x`.
    #[default]
    Range,
}

/// Default fraction of tied values at an extreme above which that extreme is
/// handled separately (`fractied` of `rcspline.eval`).
pub const DEFAULT_FRACTIED: f64 = 0.05;

/// Knot quantiles of RMS Table 2.3: `nk` equally spaced quantiles from
/// `outer` to `1 − outer`, with `outer = 0.1` for 3 knots, `0.05` for 4–6 and
/// `0.025` for 7 or more. E.g. 5 knots: 0.05, 0.275, 0.5, 0.725, 0.95.
pub fn default_knot_quantiles(nk: usize) -> Vec<f64> {
    seq(outer_quantile(nk), 1.0 - outer_quantile(nk), nk)
}

fn outer_quantile(nk: usize) -> f64 {
    if nk > 6 {
        0.025
    } else if nk > 3 {
        0.05
    } else {
        0.1
    }
}

/// Default knots for a restricted cubic spline with `nk` knots, as
/// `Hmisc::rcspline.eval(x, nk = nk, knots.only = TRUE)` (`fractied = 0.05`).
/// Non-finite values of `x` are ignored (as missing).
pub fn rcs_default_knots(x: &[f64], nk: usize) -> Result<Vec<f64>, SplineError> {
    rcs_default_knots_with(x, nk, DEFAULT_FRACTIED)
}

/// [`rcs_default_knots`] with an explicit `fractied`:
///
/// * Fewer than `nk + 3` distinct values: the knots are the distinct values
///   without the minimum and maximum.
/// * Otherwise, if no interior value has a relative frequency of `fractied`
///   or more but an extreme does, the knot at that end is the next distinct
///   value, and the remaining knots are placed on the values beyond it.
/// * The remaining knots are the [`default_knot_quantiles`] (type-7
///   quantiles, as R's `quantile`); when these are not distinct enough the
///   alternate algorithm of `rcspline.eval` is used.
/// * With fewer than 100 values the outer knots are moved to the 5th smallest
///   and 5th largest value.
pub fn rcs_default_knots_with(
    x: &[f64],
    nk: usize,
    fractied: f64,
) -> Result<Vec<f64>, SplineError> {
    let mut xx: Vec<f64> = x.iter().copied().filter(|v| v.is_finite()).collect();
    let n = xx.len();
    if n < 6 {
        return Err(SplineError::TooFewObservations);
    }
    if nk < 3 {
        return Err(SplineError::TooFewKnots(nk));
    }
    xx.sort_by(|a, b| a.total_cmp(b));
    let (xu, counts) = unique_counts(&xx);
    let nxu = xu.len();

    let knots = if nxu <= nk + 2 {
        xu[1..nxu - 1].to_vec()
    } else {
        let outer = outer_quantile(nk);
        let mut nke = nk;
        let (mut first, mut last) = (None, None);
        if fractied > 0.0 && fractied < 1.0 {
            let f = |c: usize| c as f64 / n as f64;
            let interior_max = counts[1..nxu - 1]
                .iter()
                .map(|&c| f(c))
                .fold(f64::NEG_INFINITY, f64::max);
            if interior_max < fractied {
                if f(counts[0]) >= fractied {
                    let k = xu[1];
                    xx.retain(|&v| v > k);
                    first = Some(k);
                    nke -= 1;
                }
                if f(counts[nxu - 1]) >= fractied {
                    // Largest value below the maximum of the remaining values.
                    let max = xx.last().copied().unwrap_or(f64::NAN);
                    let k = xx.iter().rev().copied().find(|&v| v < max);
                    if let Some(k) = k {
                        xx.retain(|&v| v < k);
                        last = Some(k);
                    } else {
                        // R: max(numeric(0)) = -Inf, which removes every value.
                        xx.clear();
                        last = Some(f64::NEG_INFINITY);
                    }
                    nke -= 1;
                }
            }
        }
        let mut knots = place_knots(&xx, &xu, nke, outer, first, last)?;
        if nke > 1 && xx.len() < 100 {
            if first.is_none() {
                knots[0] = xx.get(4).copied().unwrap_or(f64::NAN);
            }
            if last.is_none() {
                knots[nke - 1] = xx
                    .len()
                    .checked_sub(5)
                    .and_then(|i| xx.get(i))
                    .copied()
                    .unwrap_or(f64::NAN);
            }
        }
        first.into_iter().chain(knots).chain(last).collect()
    };
    let knots = sorted_unique(knots);
    if knots.len() < 3 {
        return Err(SplineError::TooFewUniqueKnots);
    }
    Ok(knots)
}

/// The `nke` knots of `rcspline.eval` placed on the (sorted) values `xx`
/// left after removing tied extremes. `xu` are the distinct values of the
/// full data.
fn place_knots(
    xx: &[f64],
    xu: &[f64],
    nke: usize,
    outer: f64,
    first: Option<f64>,
    last: Option<f64>,
) -> Result<Vec<f64>, SplineError> {
    if nke == 1 {
        return Ok(vec![quantile7(xx, 0.5)]);
    }
    if xu.len() <= nke {
        return Ok(xu.to_vec());
    }
    let probs = if nke == 2 {
        seq(0.5, 1.0 - outer, nke)
    } else {
        seq(outer, 1.0 - outer, nke)
    };
    let knots: Vec<f64> = probs.iter().map(|&p| quantile7(xx, p)).collect();
    if n_unique(&knots) >= nke.min(3) {
        return Ok(knots);
    }
    // Alternate algorithm: twice as many quantiles.
    let mut knots: Vec<f64> = seq(outer, 1.0 - outer, 2 * nke)
        .iter()
        .map(|&p| quantile7(xx, p))
        .collect();
    if let Some(f) = first {
        if n_unique(&knots) < 3 {
            let mid = match last {
                Some(l) => (f + l) / 2.0,
                None => quantile7(xx, 0.5),
            };
            let end = last.unwrap_or_else(|| quantile7(xx, 1.0 - outer));
            knots = vec![f, mid, end];
            knots.sort_by(|a, b| a.total_cmp(b));
        }
    }
    if n_unique(&knots) < 3 {
        return Err(SplineError::TooFewUniqueKnots);
    }
    Ok(knots)
}

/// Restricted cubic spline basis with fixed knots.
#[derive(Debug, Clone, PartialEq)]
pub struct Rcs {
    knots: Vec<f64>,
    norm: RcsNorm,
}

impl Rcs {
    /// Spline with the given knots (sorted and de-duplicated; at least 3
    /// distinct finite knots).
    pub fn new(knots: &[f64]) -> Result<Self, SplineError> {
        if knots.iter().any(|k| !k.is_finite()) {
            return Err(SplineError::NonFiniteKnots);
        }
        let knots = sorted_unique(knots.to_vec());
        if knots.len() < 3 {
            return Err(SplineError::TooFewKnots(knots.len()));
        }
        Ok(Self {
            knots,
            norm: RcsNorm::Range,
        })
    }

    /// Spline with `nk` default knots placed on `x` ([`rcs_default_knots`]).
    pub fn from_data(x: &[f64], nk: usize) -> Result<Self, SplineError> {
        Self::new(&rcs_default_knots(x, nk)?)
    }

    /// Use another normalisation of the nonlinear terms (default
    /// [`RcsNorm::Range`]).
    pub fn with_norm(mut self, norm: RcsNorm) -> Self {
        self.norm = norm;
        self
    }

    /// The knots.
    pub fn knots(&self) -> &[f64] {
        &self.knots
    }

    /// The normalisation.
    pub fn norm(&self) -> RcsNorm {
        self.norm
    }

    /// Number of basis columns, `k − 1` (including the linear term).
    pub fn n_columns(&self) -> usize {
        self.knots.len() - 1
    }

    /// Basis `[x, x', x'', …]` at `x`, as `rms::rcs` /
    /// `rcspline.eval(x, knots, inclx = TRUE)`. Column `j ≥ 1` is
    ///
    /// ```text
    /// [(x − t_j)₊³ − (x − t_{k−1})₊³ (t_k − t_j)/(t_k − t_{k−1})
    ///             + (x − t_k)₊³ (t_{k−1} − t_j)/(t_k − t_{k−1})] / kd³
    /// ```
    ///
    /// with `kd³ = (t_k − t₁)²` by default. A non-finite `x` gives a row of
    /// NaN (missing).
    pub fn basis(&self, x: &[f64], name: &str) -> Basis {
        let m = self.n_columns();
        let columns = Mat::from_fn(x.len(), m, |i, j| {
            let v = x[i];
            if !v.is_finite() {
                f64::NAN
            } else if j == 0 {
                v
            } else {
                self.term(v, j - 1)
            }
        });
        Basis {
            columns,
            names: (0..m).map(|j| primed(name, j)).collect(),
            nonlinear: (0..m).map(|j| j > 0).collect(),
        }
    }

    /// Nonlinear term `j` (0-based, `j < k − 2`) at `x`.
    fn term(&self, x: f64, j: usize) -> f64 {
        let t = &self.knots;
        let k = t.len();
        let (tk, tk1) = (t[k - 1], t[k - 2]);
        let kd = self.scale();
        let c = |a: f64| ((x - a) / kd).max(0.0).powi(3);
        c(t[j]) + ((tk1 - t[j]) * c(tk) - (tk - t[j]) * c(tk1)) / (tk - tk1)
    }

    /// `kd` of `rcspline.eval` (the terms are cubes of `(x − t)/kd`).
    fn scale(&self) -> f64 {
        let t = &self.knots;
        let k = t.len();
        match self.norm {
            RcsNorm::None => 1.0,
            RcsNorm::LastInterval => t[k - 1] - t[k - 2],
            RcsNorm::Range => (t[k - 1] - t[0]).powf(2.0 / 3.0),
        }
    }

    /// Algebraic form of a fitted spline, as `Hmisc::rcspline.restate`.
    ///
    /// `coef` are the `k − 1` coefficients of [`Self::basis`] (linear term
    /// first). The result writes the fitted function as
    /// `intercept + linear·x + Σⱼ cubic[j]·(x − t_j)₊³` over all `k` knots;
    /// the last two cubic coefficients are the ones implied by the
    /// linearity restriction.
    pub fn restate(&self, intercept: f64, coef: &[f64]) -> Result<RcsRestated, SplineError> {
        let k = self.knots.len();
        if coef.len() != k - 1 {
            return Err(SplineError::CoefficientLength {
                expected: k - 1,
                got: coef.len(),
            });
        }
        let t = &self.knots;
        let (tk, tk1) = (t[k - 1], t[k - 2]);
        let kd3 = self.scale().powi(3);
        let mut cubic: Vec<f64> = coef[1..].iter().map(|c| c / kd3).collect();
        // Coefficients on (x − t_{k−1})₊³ and (x − t_k)₊³ that cancel the
        // quadratic and cubic terms beyond t_k.
        let weighted =
            |at: f64| -> f64 { cubic.iter().enumerate().map(|(j, c)| c * (t[j] - at)).sum() };
        let c_last_but_one = weighted(tk) / (tk - tk1);
        let c_last = weighted(tk1) / (tk1 - tk);
        cubic.push(c_last_but_one);
        cubic.push(c_last);
        Ok(RcsRestated {
            intercept,
            linear: coef[0],
            knots: t.clone(),
            cubic,
        })
    }
}

/// A fitted restricted cubic spline in unrestricted truncated-power form
/// (see [`Rcs::restate`]).
#[derive(Debug, Clone, PartialEq)]
pub struct RcsRestated {
    /// Constant term.
    pub intercept: f64,
    /// Coefficient of `x`.
    pub linear: f64,
    /// The knots `t₁ … t_k`.
    pub knots: Vec<f64>,
    /// Coefficient of `(x − t_j)₊³` for every knot.
    pub cubic: Vec<f64>,
}

impl RcsRestated {
    /// Evaluate the fitted function at `x`.
    pub fn eval(&self, x: f64) -> f64 {
        self.intercept
            + self.linear * x
            + self
                .knots
                .iter()
                .zip(&self.cubic)
                .map(|(t, c)| c * (x - t).max(0.0).powi(3))
                .sum::<f64>()
    }
}

/// Linear spline basis (`rms::lsp`).
#[derive(Debug, Clone, PartialEq)]
pub struct Lsp {
    knots: Vec<f64>,
}

impl Lsp {
    /// Linear spline with the given knots (sorted and de-duplicated).
    pub fn new(knots: &[f64]) -> Result<Self, SplineError> {
        if knots.iter().any(|k| !k.is_finite()) {
            return Err(SplineError::NonFiniteKnots);
        }
        Ok(Self {
            knots: sorted_unique(knots.to_vec()),
        })
    }

    /// The knots.
    pub fn knots(&self) -> &[f64] {
        &self.knots
    }

    /// Basis `[x, (x − t₁)₊, (x − t₂)₊, …]`; the hinge terms are flagged
    /// nonlinear, as in rms.
    pub fn basis(&self, x: &[f64], name: &str) -> Basis {
        let m = self.knots.len() + 1;
        let columns = Mat::from_fn(x.len(), m, |i, j| {
            let v = x[i];
            if !v.is_finite() {
                f64::NAN
            } else if j == 0 {
                v
            } else {
                (v - self.knots[j - 1]).max(0.0)
            }
        });
        Basis {
            columns,
            names: (0..m).map(|j| primed(name, j)).collect(),
            nonlinear: (0..m).map(|j| j > 0).collect(),
        }
    }
}

/// Restricted interaction of two bases (`rms::%ia%`, RMS §2.7.2).
///
/// Forms the products `a_i · b_j` in the order `i` then `j`, but stops the
/// inner loop at the first pair where both columns are nonlinear, so no
/// doubly nonlinear product enters. For `rcs(x1, 4) %ia% rcs(x2, 3)` this
/// gives `x1*x2, x1*x2', x1'*x2, x1''*x2`. A product is flagged nonlinear
/// if either factor is.
pub fn restricted_interaction(a: &Basis, b: &Basis) -> Result<Basis, SplineError> {
    let n = a.columns.nrows();
    if b.columns.nrows() != n {
        return Err(SplineError::RowMismatch(n, b.columns.nrows()));
    }
    let mut pairs = Vec::new();
    for i in 0..a.n_columns() {
        for j in 0..b.n_columns() {
            if a.nonlinear[i] && b.nonlinear[j] {
                break;
            }
            pairs.push((i, j));
        }
    }
    Ok(Basis {
        columns: Mat::from_fn(n, pairs.len(), |r, c| {
            let (i, j) = pairs[c];
            a.columns[(r, i)] * b.columns[(r, j)]
        }),
        names: pairs
            .iter()
            .map(|&(i, j)| format!("{} * {}", a.names[i], b.names[j]))
            .collect(),
        nonlinear: pairs
            .iter()
            .map(|&(i, j)| a.nonlinear[i] || b.nonlinear[j])
            .collect(),
    })
}

/// `name` followed by `j` primes.
fn primed(name: &str, j: usize) -> String {
    format!("{name}{}", "'".repeat(j))
}

/// `seq(from, to, length.out = n)`.
fn seq(from: f64, to: f64, n: usize) -> Vec<f64> {
    if n == 1 {
        return vec![from];
    }
    let by = (to - from) / (n - 1) as f64;
    (0..n).map(|i| from + by * i as f64).collect()
}

/// R's default (type 7) quantile of sorted values; NaN if empty.
fn quantile7(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    let h = (n - 1) as f64 * p;
    // R rounds h within 4·eps to avoid spurious interpolation.
    let fuzz = 4.0 * f64::EPSILON;
    let lo = (h + fuzz).floor();
    let frac = h - lo;
    let lo = lo as usize;
    if frac.abs() < fuzz || lo + 1 >= n {
        sorted[lo.min(n - 1)]
    } else {
        (1.0 - frac) * sorted[lo] + frac * sorted[lo + 1]
    }
}

/// Distinct values of sorted data and their counts.
fn unique_counts(sorted: &[f64]) -> (Vec<f64>, Vec<usize>) {
    let mut u: Vec<f64> = Vec::new();
    let mut c: Vec<usize> = Vec::new();
    for &v in sorted {
        if u.last() == Some(&v) {
            *c.last_mut().unwrap() += 1;
        } else {
            u.push(v);
            c.push(1);
        }
    }
    (u, c)
}

fn n_unique(v: &[f64]) -> usize {
    sorted_unique(v.to_vec()).len()
}

/// Sort, drop NaN and duplicates.
fn sorted_unique(mut v: Vec<f64>) -> Vec<f64> {
    v.retain(|x| !x.is_nan());
    v.sort_by(|a, b| a.total_cmp(b));
    v.dedup();
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_2_3_quantiles() {
        let close = |a: &[f64], b: &[f64]| a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-4);
        assert!(close(&default_knot_quantiles(3), &[0.1, 0.5, 0.9]));
        assert!(close(&default_knot_quantiles(4), &[0.05, 0.35, 0.65, 0.95]));
        assert!(close(
            &default_knot_quantiles(5),
            &[0.05, 0.275, 0.5, 0.725, 0.95]
        ));
        assert!(close(
            &default_knot_quantiles(6),
            &[0.05, 0.23, 0.41, 0.59, 0.77, 0.95]
        ));
        assert!(close(
            &default_knot_quantiles(7),
            &[0.025, 0.1833, 0.3417, 0.5, 0.6583, 0.8167, 0.975]
        ));
    }

    #[test]
    fn linear_beyond_the_outer_knots() {
        let rcs = Rcs::new(&[0.0, 1.0, 2.5, 4.0]).unwrap();
        let xs = [5.0, 6.0, 7.0, -3.0, -2.0, -1.0];
        let b = rcs.basis(&xs, "x");
        for j in 1..b.n_columns() {
            // Second differences vanish to the right; terms vanish to the left.
            let d2 = b.columns[(2, j)] - 2.0 * b.columns[(1, j)] + b.columns[(0, j)];
            assert!(d2.abs() < 1e-10, "column {j}: {d2}");
            for r in 3..6 {
                assert_eq!(b.columns[(r, j)], 0.0);
            }
        }
    }

    #[test]
    fn restate_reproduces_the_basis_fit() {
        let rcs = Rcs::new(&[-1.0, 0.0, 0.5, 2.0, 3.0]).unwrap();
        let coef = [0.7, -1.3, 2.1, 0.4];
        let r = rcs.restate(0.25, &coef).unwrap();
        for i in 0..50 {
            let x = -3.0 + i as f64 * 0.15;
            let b = rcs.basis(&[x], "x");
            let direct: f64 = 0.25 + (0..4).map(|j| coef[j] * b.columns[(0, j)]).sum::<f64>();
            assert!((r.eval(x) - direct).abs() < 1e-10, "{x}");
        }
        assert!(rcs.restate(0.0, &[1.0]).is_err());
    }

    #[test]
    fn errors() {
        assert_eq!(Rcs::new(&[1.0, 1.0, 2.0]), Err(SplineError::TooFewKnots(2)));
        assert_eq!(
            Rcs::new(&[1.0, f64::NAN, 2.0]),
            Err(SplineError::NonFiniteKnots)
        );
        assert_eq!(
            rcs_default_knots(&[1.0, 2.0, 3.0], 3),
            Err(SplineError::TooFewObservations)
        );
        assert_eq!(
            rcs_default_knots(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2),
            Err(SplineError::TooFewKnots(2))
        );
        assert!(rcs_default_knots(&[1.0, 1.0, 1.0, 2.0, 2.0, 3.0], 3).is_err());
    }

    #[test]
    fn missing_x_gives_nan_rows() {
        let rcs = Rcs::new(&[0.0, 1.0, 2.0]).unwrap();
        let b = rcs.basis(&[f64::NAN, 1.5], "x");
        assert!(b.columns[(0, 0)].is_nan() && b.columns[(0, 1)].is_nan());
        assert!(b.columns[(1, 1)] > 0.0);
    }
}
