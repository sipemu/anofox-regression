//! Partial effects and contrasts on the covariate scale (RMS §5.1), as
//! `rms::datadist`, `summary.rms`, `contrast.rms` and `Predict`.
//!
//! A [`ModelSpec`] records how each predictor enters the design (linear,
//! restricted cubic spline, linear spline or categorical with treatment
//! contrasts) and the interactions, together with `datadist`-style
//! [`Limits`]. It rebuilds design rows for any covariate setting (others at
//! their adjust-to values), so effects of predictors represented by several
//! spline / interaction columns can be summarised:
//!
//! * [`ModelSpec::summary`] – inter-quartile-range (or level vs. adjust-to
//!   level) effects with CIs, and their ratios (`exp`: odds / hazard / time
//!   ratios), as `summary.rms`;
//! * [`ModelSpec::contrast`] / [`ModelSpec::double_contrast`] – single and
//!   double-difference contrasts between settings, with any covariance
//!   (model, robust or bootstrap), as `contrast.rms`;
//! * [`ModelSpec::predict_grid`] – the partial effect of one predictor over
//!   a grid with pointwise CIs, as `Predict`.
//!
//! `beta` and `vcov` are the full coefficient vector and covariance in the
//! design's column order, intercept first when `intercept` is set.

use crate::design::{Design, DesignError};
use crate::diagnostics::accuracy::quantile7;
use crate::splines::{Basis, Lsp, Rcs};
use faer::{Col, Mat};
use statrs::distribution::{ContinuousCDF, Normal, StudentsT};

/// `datadist` limits of a predictor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Limits {
    /// Low value of the effect (`Low:effect`).
    pub low_effect: f64,
    /// Adjust-to value (median; mode for categorical predictors).
    pub adjust_to: f64,
    /// High value of the effect (`High:effect`).
    pub high_effect: f64,
    /// Lower end of the prediction grid (`Low:prediction`).
    pub low_prediction: f64,
    /// Upper end of the prediction grid (`High:prediction`).
    pub high_prediction: f64,
}

impl Limits {
    /// `datadist` limits of a numeric predictor: quartiles for the effect,
    /// the median to adjust to, the `10/max(n, 200)` quantiles for
    /// predictions (with rms's rules for fewer than 4 distinct values).
    pub fn numeric(x: &[f64]) -> Self {
        let mut y: Vec<f64> = x.iter().copied().filter(|v| v.is_finite()).collect();
        y.sort_by(|a, b| a.total_cmp(b));
        let mut values = y.clone();
        values.dedup();
        let nu = values.len();
        let n = y.len();
        let (lo, hi) = (values[0], values[nu - 1]);
        let q = |p: f64| quantile7(&y, p);
        let qd = 10.0 / (n.max(200) as f64);
        let (mut lp, mut hp) = if nu < 4 {
            (lo, hi)
        } else {
            (q(qd), q(1.0 - qd))
        };
        if lp == hp {
            (lp, hp) = (lo, hi);
        }
        let adjust_to = match nu {
            0..=2 => values[0],
            3 => values[1],
            _ => q(0.5),
        };
        let (mut le, mut he) = if nu < 4 { (lo, hi) } else { (q(0.25), q(0.75)) };
        if le == he {
            (le, he) = (lo, hi);
        }
        Self {
            low_effect: le,
            adjust_to,
            high_effect: he,
            low_prediction: lp,
            high_prediction: hp,
        }
    }
}

/// How a predictor enters the design.
#[derive(Debug, Clone, PartialEq)]
pub enum PredictorKind {
    /// One linear column.
    Linear,
    /// Restricted cubic spline.
    Rcs(Rcs),
    /// Linear spline.
    Lsp(Lsp),
    /// Categorical with these (sorted) levels; dummy columns for every
    /// level but the first (treatment contrasts, as R).
    Categorical(Vec<f64>),
}

/// A predictor of a [`ModelSpec`].
#[derive(Debug, Clone, PartialEq)]
pub struct Predictor {
    /// Name.
    pub name: String,
    /// Transformation.
    pub kind: PredictorKind,
    /// `datadist` limits.
    pub limits: Limits,
}

impl Predictor {
    fn basis(&self, x: &[f64]) -> Basis {
        match &self.kind {
            PredictorKind::Linear => Basis::linear(x, &self.name),
            PredictorKind::Rcs(r) => r.basis(x, &self.name),
            PredictorKind::Lsp(l) => l.basis(x, &self.name),
            PredictorKind::Categorical(levels) => {
                let m = levels.len() - 1;
                Basis {
                    columns: Mat::from_fn(x.len(), m, |i, j| {
                        f64::from(u8::from(x[i] == levels[j + 1]))
                    }),
                    names: levels[1..]
                        .iter()
                        .map(|l| format!("{}={l}", self.name))
                        .collect(),
                    nonlinear: vec![false; m],
                }
            }
        }
    }
}

/// Critical-value distribution of effects and contrasts.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Critical {
    /// Standard normal (GLMs, Cox, AFT).
    Normal,
    /// Student t with these residual degrees of freedom (OLS).
    StudentT(f64),
}

impl Critical {
    fn quantile(self, p: f64) -> f64 {
        match self {
            Critical::Normal => Normal::new(0.0, 1.0).expect("normal").inverse_cdf(p),
            Critical::StudentT(df) => {
                StudentsT::new(0.0, 1.0, df).map_or(f64::NAN, |d| d.inverse_cdf(p))
            }
        }
    }
    fn two_sided(self, stat: f64) -> f64 {
        2.0 * match self {
            Critical::Normal => Normal::new(0.0, 1.0).expect("normal").sf(stat.abs()),
            Critical::StudentT(df) => {
                StudentsT::new(0.0, 1.0, df).map_or(f64::NAN, |d| d.sf(stat.abs()))
            }
        }
    }
}

/// A contrast estimate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ContrastEstimate {
    /// Estimate `cᵀβ`.
    pub estimate: f64,
    /// Standard error `√(cᵀVc)`.
    pub se: f64,
    /// Lower confidence limit.
    pub lower: f64,
    /// Upper confidence limit.
    pub upper: f64,
    /// `estimate / se`.
    pub statistic: f64,
    /// Two-sided p-value.
    pub p_value: f64,
}

/// One row of [`ModelSpec::summary`].
#[derive(Debug, Clone, PartialEq)]
pub struct EffectRow {
    /// Label (`x1`, or `g - 3:2` for a level against the adjust-to level).
    pub label: String,
    /// Predictor index.
    pub predictor: usize,
    /// Low value.
    pub low: f64,
    /// High value.
    pub high: f64,
    /// The effect of going from `low` to `high`.
    pub effect: ContrastEstimate,
}

/// Model specification with `datadist`-style limits.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ModelSpec {
    predictors: Vec<Predictor>,
    interactions: Vec<(usize, usize, bool)>,
}

impl ModelSpec {
    /// Empty specification.
    pub fn new() -> Self {
        Self::default()
    }

    fn push(&mut self, name: &str, kind: PredictorKind, limits: Limits) -> usize {
        self.predictors.push(Predictor {
            name: name.to_string(),
            kind,
            limits,
        });
        self.predictors.len() - 1
    }

    /// Linear predictor; limits from its data `x`.
    pub fn linear(&mut self, name: &str, x: &[f64]) -> usize {
        self.push(name, PredictorKind::Linear, Limits::numeric(x))
    }

    /// Spline predictor with the given basis; limits from `x`.
    pub fn rcs(&mut self, name: &str, rcs: Rcs, x: &[f64]) -> usize {
        self.push(name, PredictorKind::Rcs(rcs), Limits::numeric(x))
    }

    /// Linear-spline predictor; limits from `x`.
    pub fn lsp(&mut self, name: &str, lsp: Lsp, x: &[f64]) -> usize {
        self.push(name, PredictorKind::Lsp(lsp), Limits::numeric(x))
    }

    /// Categorical predictor with levels taken from `x`; adjusted to the
    /// most frequent level (the first such level), as `datadist`.
    pub fn categorical(&mut self, name: &str, x: &[f64]) -> usize {
        let mut levels: Vec<f64> = x.to_vec();
        levels.sort_by(|a, b| a.total_cmp(b));
        levels.dedup();
        let count = |l: f64| x.iter().filter(|&&v| v == l).count();
        let mode = levels
            .iter()
            .copied()
            .fold((f64::NAN, 0), |(m, c), l| {
                if count(l) > c {
                    (l, count(l))
                } else {
                    (m, c)
                }
            })
            .0;
        let (lo, hi) = (levels[0], levels[levels.len() - 1]);
        let limits = Limits {
            low_effect: f64::NAN,
            adjust_to: mode,
            high_effect: f64::NAN,
            low_prediction: lo,
            high_prediction: hi,
        };
        self.push(name, PredictorKind::Categorical(levels), limits)
    }

    /// Full tensor-product interaction (`a * b`).
    pub fn interaction(&mut self, a: usize, b: usize) {
        self.interactions.push((a, b, false));
    }

    /// Restricted interaction (`a %ia% b`).
    pub fn restricted_interaction(&mut self, a: usize, b: usize) {
        self.interactions.push((a, b, true));
    }

    /// The predictors.
    pub fn predictors(&self) -> &[Predictor] {
        &self.predictors
    }

    /// Design (no intercept column) of the data, one column slice per
    /// predictor; main effects in predictor order, then the interactions.
    pub fn design(&self, columns: &[&[f64]]) -> Result<Design, DesignError> {
        let mut d = Design::new();
        for (p, x) in self.predictors.iter().zip(columns) {
            d.add(&p.name, p.basis(x))?;
        }
        for &(a, b, r) in &self.interactions {
            if r {
                d.add_restricted_interaction(a, b)?;
            } else {
                d.add_interaction(a, b)?;
            }
        }
        Ok(d)
    }

    /// Design row for a setting: given predictor values, others at their
    /// adjust-to values; a leading 1 when `intercept`.
    pub fn row(&self, setting: &[(usize, f64)], intercept: bool) -> Vec<f64> {
        let values: Vec<f64> = (0..self.predictors.len())
            .map(|k| {
                setting
                    .iter()
                    .rev()
                    .find(|s| s.0 == k)
                    .map_or(self.predictors[k].limits.adjust_to, |s| s.1)
            })
            .collect();
        let cols: Vec<Vec<f64>> = values.iter().map(|&v| vec![v]).collect();
        let refs: Vec<&[f64]> = cols.iter().map(Vec::as_slice).collect();
        let m = self.design(&refs).expect("valid specification").matrix();
        let mut row = Vec::with_capacity(m.ncols() + 1);
        if intercept {
            row.push(1.0);
        }
        row.extend((0..m.ncols()).map(|j| m[(0, j)]));
        row
    }

    fn estimate(
        c: &[f64],
        beta: &Col<f64>,
        vcov: &Mat<f64>,
        crit: Critical,
        level: f64,
    ) -> ContrastEstimate {
        let k = c.len();
        let est: f64 = (0..k).map(|j| c[j] * beta[j]).sum();
        let var: f64 = (0..k)
            .map(|a| (0..k).map(|b| c[a] * vcov[(a, b)] * c[b]).sum::<f64>())
            .sum();
        let se = var.max(0.0).sqrt();
        let q = crit.quantile(1.0 - (1.0 - level) / 2.0);
        let statistic = est / se;
        ContrastEstimate {
            estimate: est,
            se,
            lower: est - q * se,
            upper: est + q * se,
            statistic,
            p_value: crit.two_sided(statistic),
        }
    }

    /// Contrast `row(a) − row(b)` (`contrast.rms(fit, a, b)`).
    #[allow(clippy::too_many_arguments)]
    pub fn contrast(
        &self,
        beta: &Col<f64>,
        vcov: &Mat<f64>,
        intercept: bool,
        a: &[(usize, f64)],
        b: &[(usize, f64)],
        crit: Critical,
        level: f64,
    ) -> ContrastEstimate {
        let (ra, rb) = (self.row(a, intercept), self.row(b, intercept));
        let c: Vec<f64> = ra.iter().zip(&rb).map(|(x, y)| x - y).collect();
        Self::estimate(&c, beta, vcov, crit, level)
    }

    /// Double-difference contrast `(row(a) − row(b)) − (row(a2) − row(b2))`
    /// (`contrast.rms(fit, a, b, a2, b2)`), e.g. an interaction effect.
    #[allow(clippy::too_many_arguments)]
    pub fn double_contrast(
        &self,
        beta: &Col<f64>,
        vcov: &Mat<f64>,
        intercept: bool,
        a: &[(usize, f64)],
        b: &[(usize, f64)],
        a2: &[(usize, f64)],
        b2: &[(usize, f64)],
        crit: Critical,
        level: f64,
    ) -> ContrastEstimate {
        let rows = [a, b, a2, b2].map(|s| self.row(s, intercept));
        let c: Vec<f64> = (0..rows[0].len())
            .map(|j| rows[0][j] - rows[1][j] - rows[2][j] + rows[3][j])
            .collect();
        Self::estimate(&c, beta, vcov, crit, level)
    }

    /// `summary.rms`: for each numeric predictor the effect of going from
    /// its low to its high effect value, for each categorical predictor the
    /// effect of every other level against the adjust-to level, with all
    /// other predictors at their adjust-to values. `exp` of an effect and
    /// its limits gives the odds / hazard / time ratio.
    pub fn summary(
        &self,
        beta: &Col<f64>,
        vcov: &Mat<f64>,
        intercept: bool,
        crit: Critical,
        level: f64,
    ) -> Vec<EffectRow> {
        let mut out = Vec::new();
        for (k, p) in self.predictors.iter().enumerate() {
            let pairs: Vec<(String, f64, f64)> = match &p.kind {
                PredictorKind::Categorical(levels) => levels
                    .iter()
                    .filter(|&&l| l != p.limits.adjust_to)
                    .map(|&l| {
                        (
                            format!("{} - {l}:{}", p.name, p.limits.adjust_to),
                            p.limits.adjust_to,
                            l,
                        )
                    })
                    .collect(),
                _ => vec![(p.name.clone(), p.limits.low_effect, p.limits.high_effect)],
            };
            for (label, low, high) in pairs {
                let effect = self.contrast(
                    beta,
                    vcov,
                    intercept,
                    &[(k, high)],
                    &[(k, low)],
                    crit,
                    level,
                );
                out.push(EffectRow {
                    label,
                    predictor: k,
                    low,
                    high,
                    effect,
                });
            }
        }
        out
    }

    /// `Predict(fit, x)`: the linear predictor (with the intercept, when
    /// fitted) over `values` of predictor `k` (default: 200 points between
    /// its prediction limits), others at their adjust-to values, with
    /// pointwise confidence limits. Returns `(values, estimates)`.
    #[allow(clippy::too_many_arguments)]
    pub fn predict_grid(
        &self,
        beta: &Col<f64>,
        vcov: &Mat<f64>,
        intercept: bool,
        k: usize,
        values: Option<Vec<f64>>,
        crit: Critical,
        level: f64,
    ) -> (Vec<f64>, Vec<ContrastEstimate>) {
        let values = values.unwrap_or_else(|| {
            let l = self.predictors[k].limits;
            let by = (l.high_prediction - l.low_prediction) / 199.0;
            (0..200)
                .map(|i| {
                    if i == 199 {
                        l.high_prediction
                    } else {
                        l.low_prediction + i as f64 * by
                    }
                })
                .collect()
        });
        let est = values
            .iter()
            .map(|&v| Self::estimate(&self.row(&[(k, v)], intercept), beta, vcov, crit, level))
            .collect();
        (values, est)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn categorical_rows_and_mode() {
        let mut s = ModelSpec::new();
        let g = s.categorical("g", &[1.0, 2.0, 2.0, 3.0, 2.0]);
        assert_eq!(s.predictors()[g].limits.adjust_to, 2.0);
        assert_eq!(s.row(&[], true), vec![1.0, 1.0, 0.0]);
        assert_eq!(s.row(&[(g, 3.0)], false), vec![0.0, 1.0]);
    }

    #[test]
    fn limits_follow_datadist_for_few_values() {
        let l = Limits::numeric(&[1.0, 2.0, 2.0, 3.0]);
        assert_eq!((l.low_effect, l.adjust_to, l.high_effect), (1.0, 2.0, 3.0));
    }
}
