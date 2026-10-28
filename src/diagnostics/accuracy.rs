//! Predictive accuracy indexes: discrimination, calibration and explained
//! variation (RMS §4.5, §10.8).
//!
//! Free functions over `(y, prediction, [weights])`, so they apply to any
//! model:
//!
//! * [`concordance`] – concordant / discordant / tied pair counts of a
//!   prediction and an ordered response, with the concordance probability
//!   `C`, Somers' `Dxy`, Goodman–Kruskal `γ` and Kendall `τ-a`
//!   (`survival::concordancefit`, the rank statistics of `rms::lrm`).
//!   O(n log n).
//! * [`somers2`] – `C` (ROC area) and `Dxy` for a binary response
//!   (`Hmisc::somers2`).
//! * [`rcorr_cens`] – Harrell's `C` for right-censored data with the standard
//!   deviation of `Dxy` (`Hmisc::rcorr.cens`). O(n log n).
//! * [`brier_score`], [`gini_md`] (`Hmisc::GiniMd`, the g-index of a linear
//!   predictor), [`nagelkerke_r2`] and [`r2_measures`] (Maddala / Cox–Snell
//!   `R²` with the adjusted and effective-sample-size variants of RMS 2nd
//!   ed. §4.5, `Hmisc::R2Measures`).
//! * [`lrm_stats`] – the `stats` vector of a binary `rms::lrm` fit from its
//!   linear predictor.
//! * [`val_prob`] – `rms::val.prob`: external validation of predicted
//!   probabilities (Dxy, C, R², D, U, Q, Brier, calibration intercept and
//!   slope, Emax / E90 / Eavg from a lowess calibration curve, Spiegelhalter
//!   z).

use crate::solvers::lowess::{lowess, LowessOptions};
use statrs::distribution::{ChiSquared, ContinuousCDF, Normal};
use thiserror::Error;

/// Errors of the accuracy indexes.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum AccuracyError {
    /// Inputs of different lengths.
    #[error("inputs have lengths {0} and {1}")]
    Length(usize, usize),
    /// Too few usable observations.
    #[error("need at least {0} usable observations")]
    TooFew(usize),
    /// The response must be 0/1.
    #[error("the response must be binary (0 or 1)")]
    NotBinary,
    /// Weights must be finite and non-negative.
    #[error("weights must be finite and non-negative")]
    InvalidWeights,
    /// No pair could be compared.
    #[error("no usable pairs")]
    NoPairs,
}

/// Weighted Fenwick tree over ranks.
struct Fenwick {
    t: Vec<f64>,
    total: f64,
}

impl Fenwick {
    fn new(n: usize) -> Self {
        Self {
            t: vec![0.0; n + 1],
            total: 0.0,
        }
    }
    fn add(&mut self, rank: usize, w: f64) {
        self.total += w;
        let mut i = rank + 1;
        while i < self.t.len() {
            self.t[i] += w;
            i += 1 << i.trailing_zeros();
        }
    }
    /// Sum over ranks `< rank`.
    fn below(&self, rank: usize) -> f64 {
        let mut i = rank;
        let mut s = 0.0;
        while i > 0 {
            s += self.t[i];
            i -= 1 << i.trailing_zeros();
        }
        s
    }
    /// `(below, equal, above)` of `rank`.
    fn split(&self, rank: usize) -> (f64, f64, f64) {
        let lo = self.below(rank);
        let eq = self.below(rank + 1) - lo;
        (lo, eq, self.total - lo - eq)
    }
}

/// Dense ranks (0-based) of `x` among its distinct values.
fn dense_ranks(x: &[f64]) -> (Vec<usize>, usize) {
    let mut v: Vec<f64> = x.to_vec();
    v.sort_by(|a, b| a.total_cmp(b));
    v.dedup();
    let r = x.iter().map(|a| v.partition_point(|b| b < a)).collect();
    (r, v.len())
}

fn check_weights(n: usize, w: Option<&[f64]>) -> Result<(), AccuracyError> {
    if let Some(w) = w {
        if w.len() != n {
            return Err(AccuracyError::Length(n, w.len()));
        }
        if w.iter().any(|&v| !v.is_finite() || v < 0.0) {
            return Err(AccuracyError::InvalidWeights);
        }
    }
    Ok(())
}

/// Pair counts of a prediction and an ordered response, weighted by
/// `wᵢ·wⱼ` (`survival::concordancefit` counts).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Concordance {
    /// Pairs ordered the same way by `x` and `y`.
    pub concordant: f64,
    /// Pairs ordered oppositely.
    pub discordant: f64,
    /// Pairs tied on `x` only.
    pub tied_x: f64,
    /// Pairs tied on `y` only.
    pub tied_y: f64,
    /// Pairs tied on both.
    pub tied_xy: f64,
}

impl Concordance {
    /// Concordance probability `(conc + tied_x/2) / (conc + disc + tied_x)`
    /// (the ROC area for a binary response).
    pub fn c(&self) -> f64 {
        (self.concordant + 0.5 * self.tied_x) / self.comparable()
    }
    /// Somers' `Dxy = (conc − disc) / (conc + disc + tied_x) = 2(C − ½)`.
    pub fn dxy(&self) -> f64 {
        (self.concordant - self.discordant) / self.comparable()
    }
    /// Goodman–Kruskal `γ = (conc − disc) / (conc + disc)`.
    pub fn gamma(&self) -> f64 {
        (self.concordant - self.discordant) / (self.concordant + self.discordant)
    }
    /// Kendall `τ-a = (conc − disc) / all pairs`.
    pub fn tau_a(&self) -> f64 {
        (self.concordant - self.discordant) / self.pairs()
    }
    /// Pairs with different `y`.
    pub fn comparable(&self) -> f64 {
        self.concordant + self.discordant + self.tied_x
    }
    /// All pairs.
    pub fn pairs(&self) -> f64 {
        self.comparable() + self.tied_y + self.tied_xy
    }
}

/// Concordance of the prediction `x` with the response `y` (pairs with
/// different `y`; larger `x` is expected with larger `y`). Rows with a
/// non-finite value are skipped. O(n log n).
pub fn concordance(
    x: &[f64],
    y: &[f64],
    weights: Option<&[f64]>,
) -> Result<Concordance, AccuracyError> {
    if x.len() != y.len() {
        return Err(AccuracyError::Length(x.len(), y.len()));
    }
    check_weights(x.len(), weights)?;
    let keep: Vec<usize> = (0..x.len())
        .filter(|&i| x[i].is_finite() && y[i].is_finite())
        .collect();
    let xs: Vec<f64> = keep.iter().map(|&i| x[i]).collect();
    let ys: Vec<f64> = keep.iter().map(|&i| y[i]).collect();
    let ws: Vec<f64> = keep
        .iter()
        .map(|&i| weights.map_or(1.0, |w| w[i]))
        .collect();
    let (rank, m) = dense_ranks(&xs);
    let mut order: Vec<usize> = (0..xs.len()).collect();
    order.sort_by(|&a, &b| ys[a].total_cmp(&ys[b]).then(rank[a].cmp(&rank[b])));
    let mut fw = Fenwick::new(m);
    let mut out = Concordance {
        concordant: 0.0,
        discordant: 0.0,
        tied_x: 0.0,
        tied_y: 0.0,
        tied_xy: 0.0,
    };
    let pairs = |s: f64, s2: f64| (s * s - s2) / 2.0;
    let mut g = 0;
    while g < order.len() {
        let mut h = g;
        while h < order.len() && ys[order[h]] == ys[order[g]] {
            h += 1;
        }
        let group = &order[g..h];
        let (mut s, mut s2) = (0.0, 0.0);
        for &j in group {
            let (lo, eq, hi) = fw.split(rank[j]);
            out.concordant += ws[j] * lo;
            out.tied_x += ws[j] * eq;
            out.discordant += ws[j] * hi;
            s += ws[j];
            s2 += ws[j] * ws[j];
        }
        // Within the y-tie group, split by x ties (group sorted by rank).
        let mut tied_xy = 0.0;
        let mut a = 0;
        while a < group.len() {
            let mut b = a;
            let (mut t, mut t2) = (0.0, 0.0);
            while b < group.len() && rank[group[b]] == rank[group[a]] {
                t += ws[group[b]];
                t2 += ws[group[b]] * ws[group[b]];
                b += 1;
            }
            tied_xy += pairs(t, t2);
            a = b;
        }
        out.tied_xy += tied_xy;
        out.tied_y += pairs(s, s2) - tied_xy;
        for &j in group {
            fw.add(rank[j], ws[j]);
        }
        g = h;
    }
    Ok(out)
}

/// `C` and `Dxy` of `Hmisc::somers2`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Somers2 {
    /// ROC area.
    pub c: f64,
    /// `2(C − ½)`.
    pub dxy: f64,
    /// Number of observations (sum of weights when weighted).
    pub n: f64,
}

/// ROC area and Somers' `Dxy` of predictions `p` for a 0/1 response
/// (`Hmisc::somers2`, weights as case weights). NaN when `y` has only one
/// class.
pub fn somers2(p: &[f64], y: &[f64], weights: Option<&[f64]>) -> Result<Somers2, AccuracyError> {
    if y.iter().any(|&v| v.is_finite() && v != 0.0 && v != 1.0) {
        return Err(AccuracyError::NotBinary);
    }
    let c = concordance(p, y, weights)?;
    let n = (0..p.len())
        .filter(|&i| p[i].is_finite() && y[i].is_finite())
        .map(|i| weights.map_or(1.0, |w| w[i]))
        .sum();
    if n < 2.0 {
        return Err(AccuracyError::TooFew(2));
    }
    Ok(Somers2 {
        c: c.c(),
        dxy: c.dxy(),
        n,
    })
}

/// Harrell's `C` for right-censored data (`Hmisc::rcorr.cens`).
///
/// The pair counts are over ordered pairs, as `rcorr.cens` reports them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CensoredConcordance {
    /// Concordance probability.
    pub c: f64,
    /// Somers' `Dxy = 2(C − ½)`.
    pub dxy: f64,
    /// Standard deviation of `Dxy`.
    pub sd: f64,
    /// Observations used.
    pub n: usize,
    /// Observations skipped for missing values.
    pub missing: usize,
    /// Uncensored observations.
    pub uncensored: usize,
    /// Relevant (usable) ordered pairs.
    pub relevant_pairs: f64,
    /// Concordant ordered pairs (ties in `x` count ½).
    pub concordant: f64,
    /// Ordered pairs whose order is unknown because of censoring.
    pub uncertain: f64,
}

/// Harrell's `C` of the prediction `x` (larger `x` = longer survival) for
/// right-censored times (`Hmisc::rcorr.cens`).
///
/// A pair is usable when the shorter time is an event, or the times are
/// equal and only one is an event (the event comes first); the pair is
/// concordant when the earlier failure has the smaller `x`, and ties in `x`
/// count ½. `S.D.` is `sqrt(Σᵢ (cᵢ − dᵢ − Dxy·rᵢ)²) / R` with per-observation
/// concordant, discordant and usable counts `cᵢ, dᵢ, rᵢ` and `R` usable
/// pairs. Rows with a non-finite value are skipped. O(n log n).
pub fn rcorr_cens(
    x: &[f64],
    time: &[f64],
    event: &[bool],
) -> Result<CensoredConcordance, AccuracyError> {
    if x.len() != time.len() {
        return Err(AccuracyError::Length(x.len(), time.len()));
    }
    if x.len() != event.len() {
        return Err(AccuracyError::Length(x.len(), event.len()));
    }
    let keep: Vec<usize> = (0..x.len())
        .filter(|&i| x[i].is_finite() && time[i].is_finite())
        .collect();
    let missing = x.len() - keep.len();
    let xs: Vec<f64> = keep.iter().map(|&i| x[i]).collect();
    let ts: Vec<f64> = keep.iter().map(|&i| time[i]).collect();
    let es: Vec<bool> = keep.iter().map(|&i| event[i]).collect();
    let n = xs.len();
    let (rank, m) = dense_ranks(&xs);
    let (mut c, mut d, mut r) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| ts[a].total_cmp(&ts[b]));
    let groups: Vec<&[usize]> = order.chunk_by(|&a, &b| ts[a] == ts[b]).collect();

    // The earlier member (an event) of every usable pair: compare with all
    // later times and with censored rows at the same time.
    let (mut nrel, mut nconc) = (0.0, 0.0);
    let mut fw = Fenwick::new(m);
    for g in groups.iter().rev() {
        for &j in g.iter().filter(|&&j| !es[j]) {
            fw.add(rank[j], 1.0);
        }
        for &i in g.iter().filter(|&&i| es[i]) {
            let (lo, eq, hi) = fw.split(rank[i]);
            let conc = hi + 0.5 * eq;
            c[i] += conc;
            d[i] += lo + 0.5 * eq;
            r[i] += fw.total;
            nrel += fw.total;
            nconc += conc;
        }
        for &i in g.iter().filter(|&&i| es[i]) {
            fw.add(rank[i], 1.0);
        }
    }
    // The later member: events at earlier times, and for a censored row also
    // events at the same time.
    let mut fe = Fenwick::new(m);
    let mut tied_events = 0.0;
    for g in &groups {
        let later = |j: usize, c: &mut [f64], d: &mut [f64], r: &mut [f64], fe: &Fenwick| {
            let (lo, eq, hi) = fe.split(rank[j]);
            c[j] += lo + 0.5 * eq;
            d[j] += hi + 0.5 * eq;
            r[j] += fe.total;
        };
        for &j in g.iter().filter(|&&j| es[j]) {
            later(j, &mut c, &mut d, &mut r, &fe);
        }
        let k = g.iter().filter(|&&j| es[j]).count() as f64;
        tied_events += k * (k - 1.0) / 2.0;
        for &i in g.iter().filter(|&&i| es[i]) {
            fe.add(rank[i], 1.0);
        }
        for &j in g.iter().filter(|&&j| !es[j]) {
            later(j, &mut c, &mut d, &mut r, &fe);
        }
    }
    if nrel == 0.0 {
        return Err(AccuracyError::NoPairs);
    }
    let cidx = nconc / nrel;
    let dxy = 2.0 * cidx - 1.0;
    let ss: f64 = (0..n).map(|i| (c[i] - d[i] - dxy * r[i]).powi(2)).sum();
    let nf = n as f64;
    Ok(CensoredConcordance {
        c: cidx,
        dxy,
        sd: ss.sqrt() / nrel,
        n,
        missing,
        uncensored: es.iter().filter(|&&e| e).count(),
        relevant_pairs: 2.0 * nrel,
        concordant: 2.0 * nconc,
        uncertain: nf * (nf - 1.0) - 2.0 * nrel - 2.0 * tied_events,
    })
}

/// Brier score `Σ wᵢ (pᵢ − yᵢ)² / Σ wᵢ`.
pub fn brier_score(y: &[f64], p: &[f64], weights: Option<&[f64]>) -> Result<f64, AccuracyError> {
    if y.len() != p.len() {
        return Err(AccuracyError::Length(y.len(), p.len()));
    }
    check_weights(y.len(), weights)?;
    let w = |i: usize| weights.map_or(1.0, |w| w[i]);
    let sw: f64 = (0..y.len()).map(w).sum();
    if sw <= 0.0 {
        return Err(AccuracyError::TooFew(1));
    }
    Ok((0..y.len())
        .map(|i| w(i) * (p[i] - y[i]).powi(2))
        .sum::<f64>()
        / sw)
}

/// Gini's mean difference `Σ_{i≠j} |xᵢ − xⱼ| / (n(n − 1))`
/// (`Hmisc::GiniMd`); the g-index when `x` is a linear predictor. NaN for
/// fewer than 2 values. O(n log n).
pub fn gini_md(x: &[f64]) -> f64 {
    let n = x.len();
    if n < 2 || x.iter().any(|v| !v.is_finite()) {
        return f64::NAN;
    }
    let nf = n as f64;
    let mean = x.iter().sum::<f64>() / nf;
    let mut s: Vec<f64> = x.iter().map(|v| v - mean).collect();
    s.sort_by(|a, b| a.total_cmp(b));
    s.iter()
        .enumerate()
        .map(|(i, v)| 4.0 * ((i + 1) as f64 - (nf - 1.0) / 2.0) / nf / (nf - 1.0) * v)
        .sum()
}

/// Nagelkerke `R² = (1 − exp(−LR/n)) / (1 − exp(−D₀/n))` from the model
/// likelihood-ratio χ² `LR` and the null deviance `D₀` (as `rms::lrm`).
pub fn nagelkerke_r2(lr: f64, null_deviance: f64, n: f64) -> f64 {
    (1.0 - (-lr / n).exp()) / (1.0 - (-null_deviance / n).exp())
}

/// Explained-variation measures of `Hmisc::R2Measures` (RMS 2nd ed. §4.5).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct R2Measures {
    /// Maddala / Cox–Snell `R²(n) = 1 − exp(−LR/n)`.
    pub r2: f64,
    /// `R²(p, n) = 1 − exp(−max(LR − p, 0)/n)`.
    pub r2_adj: f64,
    /// `R²(m)` with the effective sample size `m`, if given (NaN if
    /// `m = n`, as R).
    pub r2_ess: Option<f64>,
    /// `R²(p, m)`.
    pub r2_adj_ess: Option<f64>,
}

/// `R2Measures(lr, p, n, ess)`: `lr` is the model LR χ² on `p` d.f. with
/// `n` observations; `ess` the effective sample size (see
/// [`effective_sample_size`]).
pub fn r2_measures(lr: f64, p: f64, n: f64, ess: Option<f64>) -> R2Measures {
    let r2 = |n: f64| 1.0 - (-lr / n).exp();
    let adj = |n: f64| 1.0 - (-(lr - p).max(0.0) / n).exp();
    let (r2_ess, r2_adj_ess) = match ess {
        Some(m) if m == n => (Some(f64::NAN), Some(f64::NAN)),
        Some(m) => (Some(r2(m)), Some(adj(m))),
        None => (None, None),
    };
    R2Measures {
        r2: r2(n),
        r2_adj: adj(n),
        r2_ess,
        r2_adj_ess,
    }
}

/// Effective sample size `n (1 − Σ Pₖ³)` of an ordinal / binary response
/// with category frequencies `freqs` (`Pₖ = freqₖ / Σ freq`); `3 n₁ n₀ / n`
/// for a binary response.
pub fn effective_sample_size(n: f64, freqs: &[f64]) -> f64 {
    let s: f64 = freqs.iter().sum();
    n * (1.0 - freqs.iter().map(|f| (f / s).powi(3)).sum::<f64>())
}

/// The `stats` vector of a binary `rms::lrm` fit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LrmStats {
    /// Observations.
    pub obs: usize,
    /// Sum of weights.
    pub sum_weights: f64,
    /// Model likelihood-ratio χ².
    pub model_lr: f64,
    /// Its degrees of freedom.
    pub df: f64,
    /// Its p-value.
    pub p_value: f64,
    /// Concordance (ROC area).
    pub c: f64,
    /// Somers' `Dxy`.
    pub dxy: f64,
    /// Goodman–Kruskal `γ`.
    pub gamma: f64,
    /// Kendall `τ-a`.
    pub tau_a: f64,
    /// Nagelkerke `R²`.
    pub r2: f64,
    /// `R2Measures` with the effective sample size `3 n₁ n₀ / n`.
    pub r2_measures: R2Measures,
    /// Brier score.
    pub brier: f64,
    /// g-index (Gini mean difference of the linear predictor).
    pub g: f64,
    /// `exp(g)`, on the odds-ratio scale.
    pub gr: f64,
    /// Gini mean difference of the predicted probabilities.
    pub gp: f64,
}

pub(crate) fn plogis(v: f64) -> f64 {
    if v >= 0.0 {
        1.0 / (1.0 + (-v).exp())
    } else {
        let e = v.exp();
        e / (1.0 + e)
    }
}

/// `log(1 + exp(v))` without overflow.
fn log1pexp(v: f64) -> f64 {
    if v > 0.0 {
        v + (-v).exp().ln_1p()
    } else {
        v.exp().ln_1p()
    }
}

/// Binomial deviance `−2 Σ wᵢ [yᵢ ηᵢ − log(1 + e^ηᵢ)]` at linear predictor `η`.
pub(crate) fn binomial_deviance(y: &[f64], eta: &[f64], w: &dyn Fn(usize) -> f64) -> f64 {
    -2.0 * (0..y.len())
        .map(|i| w(i) * (y[i] * eta[i] - log1pexp(eta[i])))
        .sum::<f64>()
}

/// `rms::lrm` stats of a binary model from its 0/1 response `y`, its linear
/// predictor `lp` (with intercept, at the maximum likelihood estimate) and
/// the model d.f. The likelihood ratio is the null minus the model
/// deviance, both computed from `y` and `lp`.
pub fn lrm_stats(
    y: &[f64],
    lp: &[f64],
    weights: Option<&[f64]>,
    df: f64,
) -> Result<LrmStats, AccuracyError> {
    let n = y.len();
    if lp.len() != n {
        return Err(AccuracyError::Length(n, lp.len()));
    }
    check_weights(n, weights)?;
    if y.iter().any(|&v| v != 0.0 && v != 1.0) {
        return Err(AccuracyError::NotBinary);
    }
    if n < 2 {
        return Err(AccuracyError::TooFew(2));
    }
    let w = |i: usize| weights.map_or(1.0, |w| w[i]);
    let sw: f64 = (0..n).map(w).sum();
    let sw1: f64 = (0..n).map(|i| w(i) * y[i]).sum();
    let p0 = sw1 / sw;
    let null_eta = vec![(p0 / (1.0 - p0)).ln(); n];
    let null_dev = binomial_deviance(y, &null_eta, &w);
    let model_lr = null_dev - binomial_deviance(y, lp, &w);
    let prob: Vec<f64> = lp.iter().map(|&v| plogis(v)).collect();
    let conc = concordance(lp, y, weights)?;
    let g = gini_md(lp);
    Ok(LrmStats {
        obs: n,
        sum_weights: sw,
        model_lr,
        df,
        p_value: chisq_sf(model_lr, df),
        c: conc.c(),
        dxy: conc.dxy(),
        gamma: conc.gamma(),
        tau_a: conc.tau_a(),
        r2: nagelkerke_r2(model_lr, null_dev, sw),
        r2_measures: r2_measures(
            model_lr,
            df,
            sw,
            Some(effective_sample_size(sw, &[sw - sw1, sw1])),
        ),
        brier: brier_score(y, &prob, weights)?,
        g,
        gr: g.exp(),
        gp: gini_md(&prob),
    })
}

fn chisq_sf(x: f64, df: f64) -> f64 {
    if df <= 0.0 {
        return 1.0;
    }
    ChiSquared::new(df).map_or(f64::NAN, |d| d.sf(x))
}

/// Output of [`val_prob`] (`rms::val.prob`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ValProb {
    /// Somers' `Dxy`.
    pub dxy: f64,
    /// ROC area.
    pub c: f64,
    /// Nagelkerke `R²` of the predictions taken as fixed.
    pub r2: f64,
    /// Discrimination index `D = (LR − 1) / n`.
    pub d: f64,
    /// `LR` of `D` (1 d.f.).
    pub d_chisq: f64,
    /// p-value of `D`.
    pub d_p: f64,
    /// Unreliability index `U = (χ²_U − 2) / n`.
    pub u: f64,
    /// `χ²_U`: deviance of the predictions minus that of the recalibrated
    /// predictions (2 d.f.).
    pub u_chisq: f64,
    /// p-value of `U`.
    pub u_p: f64,
    /// Quality index `Q = D − U`.
    pub q: f64,
    /// Brier score.
    pub brier: f64,
    /// Calibration intercept of the logistic recalibration
    /// `logit P(y = 1) = a + b·logit(p)`.
    pub intercept: f64,
    /// Calibration slope `b`.
    pub slope: f64,
    /// Maximum absolute difference between `p` and the lowess calibration
    /// curve.
    pub emax: f64,
    /// 0.9 quantile of that difference.
    pub e90: f64,
    /// Mean of that difference.
    pub eavg: f64,
    /// Spiegelhalter's z statistic.
    pub s_z: f64,
    /// Its two-sided p-value.
    pub s_p: f64,
}

/// External validation of predicted probabilities `p` against 0/1
/// outcomes `y`, as `rms::val.prob(p, y, pl = FALSE)`.
///
/// Probabilities of exactly 0 or 1 are left out of the logistic
/// calibration (as R does); the calibration curve is `lowess(p, y, iter = 0)`.
pub fn val_prob(p: &[f64], y: &[f64]) -> Result<ValProb, AccuracyError> {
    if p.len() != y.len() {
        return Err(AccuracyError::Length(p.len(), y.len()));
    }
    let keep: Vec<usize> = (0..p.len())
        .filter(|&i| p[i].is_finite() && y[i].is_finite())
        .collect();
    let p: Vec<f64> = keep.iter().map(|&i| p[i]).collect();
    let y: Vec<f64> = keep.iter().map(|&i| y[i]).collect();
    if y.iter().any(|&v| v != 0.0 && v != 1.0) {
        return Err(AccuracyError::NotBinary);
    }
    if p.len() < 2 {
        return Err(AccuracyError::TooFew(2));
    }
    let logit: Vec<f64> = p.iter().map(|&v| (v / (1.0 - v)).ln()).collect();
    let spi = spiegelhalter(&p, &y);
    let brier = p.iter().zip(&y).map(|(a, b)| (a - b).powi(2)).sum::<f64>() / p.len() as f64;
    let one = |_: usize| 1.0;

    let first = p[0];
    if p.iter().all(|&v| v == first) {
        let n = y.len() as f64;
        let ybar = y.iter().sum::<f64>() / n;
        let intc = (ybar / (1.0 - ybar)).ln();
        let d = -1.0 / n;
        let l01 = binomial_deviance(&y, &logit, &one);
        let lcal = binomial_deviance(&y, &vec![intc; y.len()], &one);
        let u_chisq = l01 - lcal;
        let u = (u_chisq - 1.0) / n;
        let e = (first - ybar).abs();
        return Ok(ValProb {
            dxy: 0.0,
            c: 0.5,
            r2: 0.0,
            d,
            d_chisq: 0.0,
            d_p: 1.0,
            u,
            u_chisq,
            u_p: chisq_sf(u_chisq, 1.0),
            q: d - u,
            brier,
            intercept: intc,
            slope: 0.0,
            emax: 0.0,
            e90: e,
            eavg: e,
            s_z: spi.0,
            s_p: spi.1,
        });
    }

    let fin: Vec<usize> = (0..p.len()).filter(|&i| logit[i].is_finite()).collect();
    let lf: Vec<f64> = fin.iter().map(|&i| logit[i]).collect();
    let yf: Vec<f64> = fin.iter().map(|&i| y[i]).collect();
    let n = yf.len() as f64;
    let ybar = yf.iter().sum::<f64>() / n;
    let null_dev = binomial_deviance(&yf, &vec![(ybar / (1.0 - ybar)).ln(); yf.len()], &one);
    let l01 = binomial_deviance(&yf, &lf, &one);
    let lr = null_dev - l01;
    let conc = concordance(&lf, &yf, None)?;
    let (a, b, recal_dev) = logistic_recalibration(&lf, &yf);
    let u_chisq = l01 - recal_dev;
    let u = (u_chisq - 2.0) / n;
    let d = (lr - 1.0) / n;

    // Calibration curve: lowess(p, y, iter = 0), evaluated at p with
    // approx(ties = mean).
    let sm = lowess(
        &p,
        &y,
        &LowessOptions {
            frac: 2.0 / 3.0,
            iterations: 0,
            delta: None,
        },
    )
    .map_err(|_| AccuracyError::TooFew(2))?;
    let cal = approx_ties_mean(&sm.x, &sm.fitted, &p);
    let mut er: Vec<f64> = p.iter().zip(&cal).map(|(a, b)| (a - b).abs()).collect();
    let eavg = er.iter().sum::<f64>() / er.len() as f64;
    let emax = er.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    er.sort_by(|a, b| a.total_cmp(b));
    let e90 = quantile7(&er, 0.9);

    Ok(ValProb {
        dxy: conc.dxy(),
        c: conc.c(),
        r2: nagelkerke_r2(lr, null_dev, n),
        d,
        d_chisq: lr,
        d_p: chisq_sf(lr, 1.0),
        u,
        u_chisq,
        u_p: chisq_sf(u_chisq, 2.0),
        q: d - u,
        brier,
        intercept: a,
        slope: b,
        emax,
        e90,
        eavg,
        s_z: spi.0,
        s_p: spi.1,
    })
}

/// Spiegelhalter's z and its two-sided p-value.
fn spiegelhalter(p: &[f64], y: &[f64]) -> (f64, f64) {
    let num: f64 = p
        .iter()
        .zip(y)
        .map(|(p, y)| (y - p) * (1.0 - 2.0 * p))
        .sum();
    let den: f64 = p
        .iter()
        .map(|p| (1.0 - 2.0 * p).powi(2) * p * (1.0 - p))
        .sum::<f64>()
        .sqrt();
    let z = num / den;
    let pz = Normal::new(0.0, 1.0).map_or(f64::NAN, |d| 2.0 * d.cdf(-z.abs()));
    (z, pz)
}

/// Maximum-likelihood fit of `logit P(y = 1) = a + b·x`; returns `(a, b,
/// deviance)`.
pub(crate) fn logistic_recalibration(x: &[f64], y: &[f64]) -> (f64, f64, f64) {
    let n = y.len() as f64;
    let ybar = y.iter().sum::<f64>() / n;
    let (mut a, mut b) = ((ybar / (1.0 - ybar)).ln(), 0.0);
    let one = |_: usize| 1.0;
    let dev = |a: f64, b: f64| {
        let eta: Vec<f64> = x.iter().map(|&v| a + b * v).collect();
        binomial_deviance(y, &eta, &one)
    };
    let mut current = dev(a, b);
    for _ in 0..100 {
        // Newton step on (a, b).
        let (mut g0, mut g1, mut h00, mut h01, mut h11) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for (xi, yi) in x.iter().zip(y) {
            let pi = plogis(a + b * xi);
            let w = pi * (1.0 - pi);
            g0 += yi - pi;
            g1 += (yi - pi) * xi;
            h00 += w;
            h01 += w * xi;
            h11 += w * xi * xi;
        }
        let det = h00 * h11 - h01 * h01;
        if det.abs() < 1e-300 {
            break;
        }
        let (da, db) = ((h11 * g0 - h01 * g1) / det, (h00 * g1 - h01 * g0) / det);
        let mut step = 1.0;
        let mut next = dev(a + da, b + db);
        while next > current + 1e-12 && step > 1e-8 {
            step /= 2.0;
            next = dev(a + step * da, b + step * db);
        }
        a += step * da;
        b += step * db;
        let done = (current - next).abs() < 1e-12 * (1.0 + next.abs());
        current = next;
        if done {
            break;
        }
    }
    (a, b, current)
}

/// `approx(x, y, xout, ties = mean)` for sorted `x` and `xout` within its
/// range.
pub(crate) fn approx_ties_mean(x: &[f64], y: &[f64], xout: &[f64]) -> Vec<f64> {
    let (mut ux, mut uy): (Vec<f64>, Vec<f64>) = (Vec::new(), Vec::new());
    let mut i = 0;
    while i < x.len() {
        let mut j = i;
        let mut s = 0.0;
        while j < x.len() && x[j] == x[i] {
            s += y[j];
            j += 1;
        }
        ux.push(x[i]);
        uy.push(s / (j - i) as f64);
        i = j;
    }
    xout.iter()
        .map(|&v| {
            let k = ux.partition_point(|&u| u < v);
            if k < ux.len() && ux[k] == v {
                uy[k]
            } else if k == 0 || k == ux.len() {
                f64::NAN
            } else {
                let (x0, x1, y0, y1) = (ux[k - 1], ux[k], uy[k - 1], uy[k]);
                y0 + (y1 - y0) * (v - x0) / (x1 - x0)
            }
        })
        .collect()
}

/// R's default (type 7) quantile of sorted values.
pub(crate) fn quantile7(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    let h = (n - 1) as f64 * p;
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Brute-force pair counts.
    fn brute(x: &[f64], y: &[f64], w: &[f64]) -> Concordance {
        let mut c = Concordance {
            concordant: 0.0,
            discordant: 0.0,
            tied_x: 0.0,
            tied_y: 0.0,
            tied_xy: 0.0,
        };
        for i in 0..x.len() {
            for j in (i + 1)..x.len() {
                let ww = w[i] * w[j];
                let s = (x[i] - x[j]) * (y[i] - y[j]);
                if x[i] == x[j] && y[i] == y[j] {
                    c.tied_xy += ww;
                } else if x[i] == x[j] {
                    c.tied_x += ww;
                } else if y[i] == y[j] {
                    c.tied_y += ww;
                } else if s > 0.0 {
                    c.concordant += ww;
                } else {
                    c.discordant += ww;
                }
            }
        }
        c
    }

    #[test]
    fn fenwick_concordance_matches_brute_force() {
        let x: Vec<f64> = (0..60).map(|i| ((i * 7) % 11) as f64).collect();
        let y: Vec<f64> = (0..60).map(|i| ((i * 5) % 7) as f64).collect();
        let w: Vec<f64> = (0..60).map(|i| 0.5 + (i % 3) as f64).collect();
        let a = concordance(&x, &y, Some(&w)).unwrap();
        let b = brute(&x, &y, &w);
        for (u, v) in [
            (a.concordant, b.concordant),
            (a.discordant, b.discordant),
            (a.tied_x, b.tied_x),
            (a.tied_y, b.tied_y),
            (a.tied_xy, b.tied_xy),
        ] {
            assert!((u - v).abs() < 1e-9, "{u} vs {v}");
        }
    }

    #[test]
    fn gini_md_matches_definition() {
        let x: [f64; 5] = [3.0, -1.0, 4.0, 1.5, 9.0];
        let mut s = 0.0;
        for a in x {
            for b in x {
                s += (a - b).abs();
            }
        }
        assert!((gini_md(&x) - s / 20.0).abs() < 1e-12);
        assert!(gini_md(&[1.0]).is_nan());
    }

    #[test]
    fn approx_collapses_ties() {
        let v = approx_ties_mean(
            &[0.0, 1.0, 1.0, 2.0],
            &[0.0, 1.0, 3.0, 2.0],
            &[0.5, 1.0, 1.5],
        );
        assert_eq!(v, vec![1.0, 2.0, 2.0]);
    }

    #[test]
    fn errors() {
        assert_eq!(
            concordance(&[1.0], &[1.0, 2.0], None),
            Err(AccuracyError::Length(1, 2))
        );
        assert_eq!(
            somers2(&[0.1, 0.2], &[0.0, 2.0], None),
            Err(AccuracyError::NotBinary)
        );
        assert_eq!(
            brier_score(&[1.0], &[0.5], Some(&[-1.0])),
            Err(AccuracyError::InvalidWeights)
        );
        assert_eq!(
            rcorr_cens(&[1.0, 2.0], &[1.0, 2.0], &[false, false]),
            Err(AccuracyError::NoPairs)
        );
    }
}
