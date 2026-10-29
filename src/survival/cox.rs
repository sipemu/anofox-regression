//! Cox proportional hazards regression (RMS Ch 20), as `survival::coxph`.
//!
//! The partial likelihood is maximised by Newton–Raphson with step halving
//! (coxph's algorithm), with
//!
//! * Efron (default) or Breslow handling of tied event times,
//! * case weights and offsets,
//! * strata (separate baseline hazards),
//! * counting-process data `(entry, stop]` for left truncation and
//!   time-dependent covariates,
//! * an optional ridge penalty `λ/2 Σβ²` (`coxph(... ridge(..., theta = λ,
//!   scale = FALSE))`).
//!
//! A [`CoxFit`] reports the coefficients, their covariance (inverse
//! information), the likelihood-ratio, Wald and score tests, Harrell's C,
//! `rms::cph`'s R², score residuals, the cluster-robust covariance and
//! predicted survival curves with confidence limits for new covariates
//! (`survfit.coxph`).
//!
//! # Example
//!
//! ```rust
//! use anofox_regression::survival::cox::{coxph, CoxInput, CoxOptions};
//! use faer::Mat;
//!
//! let time = [6.0, 7.0, 10.0, 15.0, 19.0, 25.0, 3.0, 9.0, 11.0, 20.0];
//! let event = [true, false, true, true, false, true, true, true, false, true];
//! let x = Mat::from_fn(10, 1, |i, _| (i % 2) as f64);
//! let fit = coxph(&x, &CoxInput::new(&time, &event), &CoxOptions::default()).unwrap();
//! assert!(fit.converged);
//! assert_eq!(fit.coefficients.len(), 1);
//! ```

#![allow(clippy::needless_range_loop)]

use super::{conf_limits, timefix, ConfType, SurvivalError};
use crate::inference::{cluster_sandwich, ClusterError, ClusterOptions};
use faer::linalg::solvers::Solve;
use faer::Mat;
use statrs::distribution::{ChiSquared, ContinuousCDF, Normal};

/// Handling of tied event times.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Ties {
    /// Efron's approximation (coxph's default).
    #[default]
    Efron,
    /// Breslow's approximation.
    Breslow,
}

/// Options of [`coxph`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CoxOptions {
    /// Tie handling.
    pub ties: Ties,
    /// Maximum Newton iterations (coxph: 20).
    pub max_iter: usize,
    /// Convergence: `|1 − ℓ_old/ℓ_new| ≤ eps` (coxph: 1e-9).
    pub eps: f64,
    /// Ridge penalty `λ/2 Σβ²` (0 = none).
    pub ridge: f64,
    /// Confidence level of predicted survival curves.
    pub conf_level: f64,
    /// Merge near-equal times as R's `Surv(..., timefix = TRUE)`.
    pub timefix: bool,
}

impl Default for CoxOptions {
    fn default() -> Self {
        Self {
            ties: Ties::Efron,
            max_iter: 20,
            eps: 1e-9,
            ridge: 0.0,
            conf_level: 0.95,
            timefix: true,
        }
    }
}

/// Survival data of a Cox model.
#[derive(Debug, Clone, Copy)]
pub struct CoxInput<'a> {
    /// Stop (event / censoring) times.
    pub time: &'a [f64],
    /// Event indicators.
    pub event: &'a [bool],
    /// Entry times (counting-process data; at risk on `(entry, time]`).
    pub entry: Option<&'a [f64]>,
    /// Case weights.
    pub weights: Option<&'a [f64]>,
    /// Offsets added to the linear predictor.
    pub offset: Option<&'a [f64]>,
    /// Stratum codes.
    pub strata: Option<&'a [usize]>,
}

impl<'a> CoxInput<'a> {
    /// Right-censored data.
    pub fn new(time: &'a [f64], event: &'a [bool]) -> Self {
        Self {
            time,
            event,
            entry: None,
            weights: None,
            offset: None,
            strata: None,
        }
    }
    /// Entry times (counting-process data).
    pub fn entry(mut self, v: &'a [f64]) -> Self {
        self.entry = Some(v);
        self
    }
    /// Case weights.
    pub fn weights(mut self, v: &'a [f64]) -> Self {
        self.weights = Some(v);
        self
    }
    /// Offsets.
    pub fn offset(mut self, v: &'a [f64]) -> Self {
        self.offset = Some(v);
        self
    }
    /// Stratum codes.
    pub fn strata(mut self, v: &'a [usize]) -> Self {
        self.strata = Some(v);
        self
    }
}

/// Data of a fit, owned, with centred covariates.
#[derive(Debug, Clone)]
struct Data {
    x: Mat<f64>,
    means: Vec<f64>,
    stop: Vec<f64>,
    entry: Option<Vec<f64>>,
    event: Vec<bool>,
    w: Vec<f64>,
    off: Vec<f64>,
    strata: Vec<usize>,
    /// Per stratum: rows sorted by stop (descending) and by entry (descending).
    groups: Vec<(usize, Vec<usize>, Vec<usize>)>,
}

/// Risk-set sums at one event time (centred covariates, current β).
#[derive(Debug, Clone)]
struct EventTime {
    stratum: usize,
    time: f64,
    /// Σ w r over the risk set.
    s0: f64,
    /// Σ w r x.
    s1: Vec<f64>,
    /// Σ w r over the deaths.
    d0: f64,
    /// Σ w r x over the deaths.
    d1: Vec<f64>,
    /// Σ w over the deaths.
    wd: f64,
    /// Number of death rows.
    m: usize,
}

impl Data {
    fn new(x: &Mat<f64>, input: &CoxInput, timefix_on: bool) -> Result<Self, SurvivalError> {
        let n = input.time.len();
        if x.nrows() != n {
            return Err(SurvivalError::Length(x.nrows(), n));
        }
        super::SurvInput {
            time: input.time,
            event: input.event,
            entry: input.entry,
            weights: input.weights,
        }
        .validate()?;
        for (len, v) in [
            (input.offset.map(<[f64]>::len), "offset"),
            (input.strata.map(<[usize]>::len), "strata"),
        ] {
            if len.is_some_and(|l| l != n) {
                return Err(SurvivalError::Invalid(if v == "offset" {
                    "offset has the wrong length"
                } else {
                    "strata has the wrong length"
                }));
            }
        }
        if input
            .offset
            .is_some_and(|o| o.iter().any(|v| !v.is_finite()))
            || (0..x.nrows()).any(|i| (0..x.ncols()).any(|j| !x[(i, j)].is_finite()))
        {
            return Err(SurvivalError::Invalid(
                "covariates and offsets must be finite",
            ));
        }
        let (stop, entry) = if timefix_on {
            timefix(input.time, input.entry)?
        } else {
            (input.time.to_vec(), input.entry.map(<[f64]>::to_vec))
        };
        let p = x.ncols();
        let means: Vec<f64> = (0..p)
            .map(|j| (0..n).map(|i| x[(i, j)]).sum::<f64>() / n as f64)
            .collect();
        let strata = input.strata.map_or(vec![0; n], <[usize]>::to_vec);
        let mut codes = strata.clone();
        codes.sort_unstable();
        codes.dedup();
        let groups = codes
            .iter()
            .map(|&s| {
                let rows: Vec<usize> = (0..n).filter(|&i| strata[i] == s).collect();
                let mut by_stop = rows.clone();
                by_stop.sort_by(|&a, &b| stop[b].total_cmp(&stop[a]));
                let mut by_entry = rows;
                if let Some(e) = &entry {
                    by_entry.sort_by(|&a, &b| e[b].total_cmp(&e[a]));
                } else {
                    by_entry.clear();
                }
                (s, by_stop, by_entry)
            })
            .collect();
        Ok(Self {
            x: Mat::from_fn(n, p, |i, j| x[(i, j)] - means[j]),
            means,
            stop,
            entry,
            event: input.event.to_vec(),
            w: input.weights.map_or(vec![1.0; n], <[f64]>::to_vec),
            off: input.offset.map_or(vec![0.0; n], <[f64]>::to_vec),
            strata,
            groups,
        })
    }

    fn eta(&self, beta: &[f64]) -> Vec<f64> {
        (0..self.x.nrows())
            .map(|i| {
                self.off[i]
                    + (0..beta.len())
                        .map(|j| self.x[(i, j)] * beta[j])
                        .sum::<f64>()
            })
            .collect()
    }

    /// Walk the event times of every stratum (descending), maintaining the
    /// risk-set sums; `visit` gets the sums and `Σ w r x xᵀ` (row-major).
    fn sweep(
        &self,
        beta: &[f64],
        with_s2: bool,
        mut visit: impl FnMut(&EventTime, &[f64], f64, &[f64]),
    ) {
        let p = self.x.ncols();
        let eta = self.eta(beta);
        let r: Vec<f64> = eta.iter().map(|e| e.exp()).collect();
        for (s, by_stop, by_entry) in &self.groups {
            let (mut s0, mut s1, mut s2) = (
                0.0,
                vec![0.0; p],
                vec![0.0; if with_s2 { p * p } else { 0 }],
            );
            let add = |i: usize, sign: f64, s0: &mut f64, s1: &mut [f64], s2: &mut [f64]| {
                let wr = sign * self.w[i] * r[i];
                *s0 += wr;
                for a in 0..p {
                    let xa = self.x[(i, a)];
                    s1[a] += wr * xa;
                    if with_s2 {
                        for b in 0..p {
                            s2[a * p + b] += wr * xa * self.x[(i, b)];
                        }
                    }
                }
            };
            let (mut ia, mut ie) = (0, 0);
            let mut k = 0;
            while k < by_stop.len() {
                let t = self.stop[by_stop[k]];
                let mut kk = k;
                while kk < by_stop.len() && self.stop[by_stop[kk]] == t {
                    kk += 1;
                }
                let deaths: Vec<usize> = by_stop[k..kk]
                    .iter()
                    .copied()
                    .filter(|&i| self.event[i])
                    .collect();
                if !deaths.is_empty() {
                    while ia < by_stop.len() && self.stop[by_stop[ia]] >= t {
                        add(by_stop[ia], 1.0, &mut s0, &mut s1, &mut s2);
                        ia += 1;
                    }
                    if let Some(e) = &self.entry {
                        while ie < by_entry.len() && e[by_entry[ie]] >= t {
                            add(by_entry[ie], -1.0, &mut s0, &mut s1, &mut s2);
                            ie += 1;
                        }
                    }
                    let mut et = EventTime {
                        stratum: *s,
                        time: t,
                        s0,
                        s1: s1.clone(),
                        d0: 0.0,
                        d1: vec![0.0; p],
                        wd: 0.0,
                        m: deaths.len(),
                    };
                    let mut d2 = vec![0.0; if with_s2 { p * p } else { 0 }];
                    let mut eta_sum = 0.0;
                    let mut xd = vec![0.0; p];
                    for &i in &deaths {
                        let wr = self.w[i] * r[i];
                        et.d0 += wr;
                        et.wd += self.w[i];
                        eta_sum += self.w[i] * eta[i];
                        for a in 0..p {
                            let xa = self.x[(i, a)];
                            et.d1[a] += wr * xa;
                            xd[a] += self.w[i] * xa;
                            if with_s2 {
                                for b in 0..p {
                                    d2[a * p + b] += wr * xa * self.x[(i, b)];
                                }
                            }
                        }
                    }
                    let mut packed = s2.clone();
                    packed.extend_from_slice(&d2);
                    visit(&et, &packed, eta_sum, &xd);
                }
                k = kk;
            }
        }
    }

    /// Penalised log partial likelihood, score and information.
    fn derivatives(&self, beta: &[f64], ties: Ties, ridge: f64) -> (f64, Vec<f64>, Mat<f64>) {
        let p = self.x.ncols();
        let mut ll = 0.0;
        let mut u = vec![0.0; p];
        let mut info = Mat::<f64>::zeros(p, p);
        self.sweep(beta, true, |et, s2d2, eta_sum, xd| {
            let (s2, d2) = s2d2.split_at(p * p);
            ll += eta_sum;
            for a in 0..p {
                u[a] += xd[a];
            }
            let steps: Vec<(f64, f64)> = match ties {
                Ties::Breslow => vec![(0.0, et.wd)],
                Ties::Efron => (0..et.m)
                    .map(|k| (k as f64 / et.m as f64, et.wd / et.m as f64))
                    .collect(),
            };
            for (f, wt) in steps {
                let den = et.s0 - f * et.d0;
                ll -= wt * den.ln();
                let mean: Vec<f64> = (0..p).map(|a| (et.s1[a] - f * et.d1[a]) / den).collect();
                for a in 0..p {
                    u[a] -= wt * mean[a];
                    for b in 0..p {
                        let second = (s2[a * p + b] - f * d2[a * p + b]) / den;
                        info[(a, b)] += wt * (second - mean[a] * mean[b]);
                    }
                }
            }
        });
        for a in 0..p {
            ll -= 0.5 * ridge * beta[a] * beta[a];
            u[a] -= ridge * beta[a];
            info[(a, a)] += ridge;
        }
        (ll, u, info)
    }

    fn event_times(&self, beta: &[f64]) -> Vec<EventTime> {
        let mut out = Vec::new();
        self.sweep(beta, false, |et, _, _, _| out.push(et.clone()));
        out
    }
}

fn inverse(m: &Mat<f64>) -> Option<Mat<f64>> {
    let k = m.nrows();
    let llt = m.llt(faer::Side::Lower).ok()?;
    Some(llt.solve(Mat::<f64>::identity(k, k)))
}

fn quad(v: &[f64], m: &Mat<f64>) -> f64 {
    let k = v.len();
    (0..k)
        .map(|a| (0..k).map(|b| v[a] * m[(a, b)] * v[b]).sum::<f64>())
        .sum()
}

/// Time transformation of the proportional-hazards test ([`CoxFit::cox_zph`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ZphTransform {
    /// `1 − S(t−)` of the pooled Kaplan–Meier curve (`cox.zph`'s default).
    #[default]
    Km,
    /// Rank of the time.
    Rank,
    /// The time itself.
    Identity,
    /// Log of the time.
    Log,
}

/// One row of the proportional-hazards test.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ZphTest {
    /// Score χ².
    pub chisq: f64,
    /// Degrees of freedom.
    pub df: usize,
    /// p-value.
    pub p_value: f64,
}

/// Result of [`CoxFit::cox_zph`] (`survival::cox.zph`).
#[derive(Debug, Clone, PartialEq)]
pub struct CoxZph {
    /// One test per term (column group).
    pub terms: Vec<ZphTest>,
    /// The global test.
    pub global: ZphTest,
    /// Transformed time of each event row (order of
    /// [`CoxFit::schoenfeld_residuals`]).
    pub x: Vec<f64>,
    /// Untransformed time of each event row.
    pub time: Vec<f64>,
    /// Scaled Schoenfeld residuals (`events × p`), for plotting β(t).
    pub y: Mat<f64>,
}

/// A fitted Cox model.
#[derive(Debug, Clone)]
pub struct CoxFit {
    /// Coefficients.
    pub coefficients: Vec<f64>,
    /// Covariance (inverse of the (penalised) information).
    pub var: Mat<f64>,
    /// Standard errors.
    pub std_errors: Vec<f64>,
    /// Wald z statistics.
    pub z: Vec<f64>,
    /// Two-sided p-values.
    pub p_values: Vec<f64>,
    /// Log partial likelihood at β = 0 and at the estimate (`loglik` of coxph).
    pub loglik: (f64, f64),
    /// Likelihood-ratio χ² `2(ℓ̂ − ℓ₀)`.
    pub lr_test: f64,
    /// Wald χ² `β̂ᵀ V⁻¹ β̂`.
    pub wald_test: f64,
    /// Score χ² at β = 0.
    pub score_test: f64,
    /// Degrees of freedom of the tests.
    pub df: usize,
    /// Observations.
    pub n: usize,
    /// Number of events.
    pub n_events: usize,
    /// Newton iterations used.
    pub iterations: usize,
    /// Whether the iteration converged.
    pub converged: bool,
    ties: Ties,
    ridge: f64,
    conf_level: f64,
    data: Data,
}

/// Fit a Cox model of the survival data on the covariates `x` (no
/// intercept column).
pub fn coxph(
    x: &Mat<f64>,
    input: &CoxInput,
    options: &CoxOptions,
) -> Result<CoxFit, SurvivalError> {
    let data = Data::new(x, input, options.timefix)?;
    let p = x.ncols();
    if p == 0 {
        return Err(SurvivalError::Invalid("no covariates"));
    }
    if !input.event.iter().any(|&e| e) {
        return Err(SurvivalError::Invalid("no events"));
    }
    let mut beta = vec![0.0; p];
    let (ll0, u0, i0) = data.derivatives(&beta, options.ties, options.ridge);
    let i0inv = inverse(&i0).ok_or(SurvivalError::Invalid("singular information at β = 0"))?;
    let score_test = quad(&u0, &i0inv);
    let (mut ll, mut u, mut info) = (ll0, u0, i0);
    let mut converged = false;
    let mut iterations = 0;
    for it in 1..=options.max_iter {
        iterations = it;
        let inv = inverse(&info).ok_or(SurvivalError::Invalid("singular information matrix"))?;
        let step: Vec<f64> = (0..p)
            .map(|a| (0..p).map(|b| inv[(a, b)] * u[b]).sum())
            .collect();
        let mut cand: Vec<f64> = (0..p).map(|a| beta[a] + step[a]).collect();
        let (mut ll_new, mut u_new, mut info_new) =
            data.derivatives(&cand, options.ties, options.ridge);
        let mut halvings = 0;
        while (!ll_new.is_finite() || ll_new < ll) && halvings < 30 {
            cand = (0..p).map(|a| (cand[a] + beta[a]) / 2.0).collect();
            (ll_new, u_new, info_new) = data.derivatives(&cand, options.ties, options.ridge);
            halvings += 1;
        }
        let done = (1.0 - ll / ll_new).abs() <= options.eps;
        beta = cand;
        ll = ll_new;
        u = u_new;
        info = info_new;
        if done {
            converged = true;
            break;
        }
    }
    let _ = u;
    let var = inverse(&info).ok_or(SurvivalError::Invalid("singular information matrix"))?;
    let std_errors: Vec<f64> = (0..p).map(|a| var[(a, a)].sqrt()).collect();
    let z: Vec<f64> = (0..p).map(|a| beta[a] / std_errors[a]).collect();
    let norm = Normal::new(0.0, 1.0).expect("standard normal");
    let p_values = z.iter().map(|v| 2.0 * norm.cdf(-v.abs())).collect();
    let wald_test = quad(&beta, &info);
    // The likelihood ratio uses the unpenalised log-likelihoods.
    let pen = 0.5 * options.ridge * beta.iter().map(|b| b * b).sum::<f64>();
    let ll_unpen = ll + pen;
    Ok(CoxFit {
        coefficients: beta,
        var,
        std_errors,
        z,
        p_values,
        loglik: (ll0, ll_unpen),
        lr_test: 2.0 * (ll_unpen - ll0),
        wald_test,
        score_test,
        df: p,
        n: input.time.len(),
        n_events: input.event.iter().filter(|&&e| e).count(),
        iterations,
        converged,
        ties: options.ties,
        ridge: options.ridge,
        conf_level: options.conf_level,
        data,
    })
}

/// A predicted survival curve of a Cox model.
#[derive(Debug, Clone, PartialEq)]
pub struct CoxSurvCurve {
    /// Distinct times of the stratum (events and censorings).
    pub time: Vec<f64>,
    /// Number at risk.
    pub n_risk: Vec<f64>,
    /// Cumulative hazard for the covariates.
    pub cumhaz: Vec<f64>,
    /// Survival `exp(−H)`.
    pub surv: Vec<f64>,
    /// Standard error of the cumulative hazard.
    pub std_err: Vec<f64>,
    /// Lower confidence limit (log scale).
    pub lower: Vec<f64>,
    /// Upper confidence limit.
    pub upper: Vec<f64>,
}

impl CoxFit {
    /// p-value of a test statistic on [`Self::df`] degrees of freedom.
    pub fn test_p_value(&self, chisq: f64) -> f64 {
        ChiSquared::new(self.df as f64).map_or(f64::NAN, |d| d.sf(chisq))
    }

    /// Linear predictors `xβ + offset` of the fitted rows (uncentred).
    pub fn linear_predictors(&self) -> Vec<f64> {
        let shift: f64 = (0..self.df)
            .map(|j| self.data.means[j] * self.coefficients[j])
            .sum();
        self.data
            .eta(&self.coefficients)
            .iter()
            .map(|e| e + shift)
            .collect()
    }

    /// Harrell's C of the fit (larger linear predictor = shorter survival),
    /// within strata, as coxph's `concordance`.
    pub fn concordance(&self) -> f64 {
        let lp = self.linear_predictors();
        let (mut conc, mut rel) = (0.0, 0.0);
        for (_, rows, _) in &self.data.groups {
            let x: Vec<f64> = rows.iter().map(|&i| -lp[i]).collect();
            let t: Vec<f64> = rows.iter().map(|&i| self.data.stop[i]).collect();
            let e: Vec<bool> = rows.iter().map(|&i| self.data.event[i]).collect();
            if let Ok(c) = crate::diagnostics::accuracy::rcorr_cens(&x, &t, &e) {
                conc += c.concordant;
                rel += c.relevant_pairs;
            }
        }
        conc / rel
    }

    /// Somers' Dxy `2(C − ½)`, as `rms::cph`.
    pub fn dxy(&self) -> f64 {
        2.0 * self.concordance() - 1.0
    }

    /// `rms::cph`'s R²: `(1 − exp(−LR/n)) / (1 − exp(2ℓ₀/n))`.
    pub fn r2(&self) -> f64 {
        let n = self.n as f64;
        (1.0 - (-self.lr_test / n).exp()) / (1.0 - (2.0 * self.loglik.0 / n).exp())
    }

    /// Score residuals (`residuals(fit, type = "score")`), `n × p`.
    pub fn score_residuals(&self) -> Mat<f64> {
        self.residual_parts().1
    }

    /// Martingale residuals `δᵢ − Ĥᵢ` (`residuals(fit, type = "martingale")`),
    /// with the Efron-adjusted cumulative hazard of each row.
    pub fn martingale_residuals(&self) -> Vec<f64> {
        self.residual_parts().0
    }

    /// Deviance residuals `sign(M)·√(−2(M + δ log(δ − M)))`.
    pub fn deviance_residuals(&self) -> Vec<f64> {
        self.martingale_residuals()
            .iter()
            .zip(&self.data.event)
            .map(|(&m, &e)| {
                let d = f64::from(u8::from(e));
                let inner = if e { m + d * (d - m).ln() } else { m };
                m.signum() * (-2.0 * inner).max(0.0).sqrt()
            })
            .collect()
    }

    /// `dfbeta` residuals: the approximate change in β̂ when row `i` is left
    /// out, `wᵢ · score residualᵢ · V` (weighted, as `residuals(fit, type =
    /// "dfbeta")` in survival ≥ 3).
    pub fn dfbeta(&self) -> Mat<f64> {
        let r = self.score_residuals();
        let w = &self.data.w;
        Mat::from_fn(r.nrows(), r.ncols(), |i, j| w[i] * r[(i, j)]) * &self.var
    }

    /// `dfbetas`: [`Self::dfbeta`] divided by the standard errors.
    pub fn dfbetas(&self) -> Mat<f64> {
        let d = self.dfbeta();
        Mat::from_fn(d.nrows(), d.ncols(), |i, j| d[(i, j)] / self.std_errors[j])
    }

    /// Schoenfeld residuals `xᵢ − x̄(tᵢ)` of the event rows (Efron-averaged
    /// over tied deaths), ordered by stratum and time; returns the row
    /// indices, their times and the `events × p` residuals. With `scaled`
    /// they are `β̂ + n_d V rᵢ` (`type = "scaledsch"`), `n_d` the number of
    /// events.
    pub fn schoenfeld_residuals(&self, scaled: bool) -> (Vec<usize>, Vec<f64>, Mat<f64>) {
        let d = &self.data;
        let p = self.df;
        let g = self.residual_parts().2;
        let mut rows: Vec<usize> = (0..d.x.nrows()).filter(|&i| d.event[i]).collect();
        rows.sort_by(|&a, &b| {
            d.strata[a]
                .cmp(&d.strata[b])
                .then(d.stop[a].total_cmp(&d.stop[b]))
        });
        let raw = Mat::from_fn(rows.len(), p, |k, j| d.x[(rows[k], j)] - g[(rows[k], j)]);
        let out = if scaled {
            let nd = rows.len() as f64;
            let rv = &raw * &self.var;
            Mat::from_fn(rows.len(), p, |k, j| self.coefficients[j] + nd * rv[(k, j)])
        } else {
            raw
        };
        let times = rows.iter().map(|&i| d.stop[i]).collect();
        (rows, times, out)
    }

    /// Martingale residuals, score residuals and, for every event row, the
    /// Efron-averaged risk-set mean `(1/m) Σ_k x̄_k` at its time.
    fn residual_parts(&self) -> (Vec<f64>, Mat<f64>, Mat<f64>) {
        let d = &self.data;
        let (n, p) = (d.x.nrows(), d.x.ncols());
        let eta = d.eta(&self.coefficients);
        let r: Vec<f64> = eta.iter().map(|e| e.exp()).collect();
        let times = d.event_times(&self.coefficients);
        // Per event time: a = Σ_k dΛ_k, b = Σ_k dΛ_k x̄_k, and for its deaths
        // c = Σ_k (k/m) dΛ_k, e = Σ_k (k/m) dΛ_k x̄_k, g = (1/m) Σ_k x̄_k.
        struct Step {
            a: f64,
            b: Vec<f64>,
            c: f64,
            e: Vec<f64>,
            g: Vec<f64>,
        }
        let steps: Vec<Step> = times
            .iter()
            .map(|et| {
                let ks: Vec<(f64, f64)> = match self.ties {
                    Ties::Breslow => vec![(0.0, et.wd)],
                    Ties::Efron => (0..et.m)
                        .map(|k| (k as f64 / et.m as f64, et.wd / et.m as f64))
                        .collect(),
                };
                let nk = ks.len() as f64;
                let mut s = Step {
                    a: 0.0,
                    b: vec![0.0; p],
                    c: 0.0,
                    e: vec![0.0; p],
                    g: vec![0.0; p],
                };
                for (f, wt) in ks {
                    let den = et.s0 - f * et.d0;
                    let dl = wt / den;
                    s.a += dl;
                    s.c += f * dl;
                    for j in 0..p {
                        let xb = (et.s1[j] - f * et.d1[j]) / den;
                        s.b[j] += dl * xb;
                        s.e[j] += f * dl * xb;
                        s.g[j] += xb / nk;
                    }
                }
                s
            })
            .collect();
        let mut out = Mat::<f64>::zeros(n, p);
        let mut mart = vec![0.0; n];
        let mut xbar = Mat::<f64>::zeros(n, p);
        for (sidx, (_, rows, _)) in d.groups.iter().enumerate() {
            let _ = sidx;
            let stratum = d.strata[rows[0]];
            // Event times of this stratum in increasing order, with prefix
            // sums of a and b.
            let mut idx: Vec<usize> = (0..times.len())
                .filter(|&k| times[k].stratum == stratum)
                .collect();
            idx.sort_by(|&u, &v| times[u].time.total_cmp(&times[v].time));
            let ts: Vec<f64> = idx.iter().map(|&k| times[k].time).collect();
            let mut pa = vec![0.0; idx.len() + 1];
            let mut pb = vec![vec![0.0; p]; idx.len() + 1];
            for (q, &k) in idx.iter().enumerate() {
                pa[q + 1] = pa[q] + steps[k].a;
                for j in 0..p {
                    pb[q + 1][j] = pb[q][j] + steps[k].b[j];
                }
            }
            for &i in rows {
                // Event times in (entry, stop].
                let hi = ts.partition_point(|&t| t <= d.stop[i]);
                let lo = d
                    .entry
                    .as_ref()
                    .map_or(0, |e| ts.partition_point(|&t| t <= e[i]));
                let a = pa[hi] - pa[lo];
                mart[i] = -r[i] * a;
                for j in 0..p {
                    let b = pb[hi][j] - pb[lo][j];
                    out[(i, j)] = -r[i] * (d.x[(i, j)] * a - b);
                }
                if d.event[i] {
                    let q = ts.partition_point(|&t| t < d.stop[i]);
                    let st = &steps[idx[q]];
                    mart[i] += 1.0 + r[i] * st.c;
                    for j in 0..p {
                        xbar[(i, j)] = st.g[j];
                    }
                    for j in 0..p {
                        out[(i, j)] +=
                            d.x[(i, j)] - st.g[j] + r[i] * (d.x[(i, j)] * st.c - st.e[j]);
                    }
                }
            }
        }
        (mart, out, xbar)
    }

    /// Cluster-robust covariance (`coxph(..., cluster = id)`): the sandwich
    /// of the weighted score residuals summed within clusters. Each row is
    /// its own cluster for `coxph(..., robust = TRUE)`.
    pub fn robust_vcov(&self, clusters: &[usize]) -> Result<Mat<f64>, ClusterError> {
        let res = self.score_residuals();
        let w = &self.data.w;
        let scores = Mat::from_fn(res.nrows(), res.ncols(), |i, j| w[i] * res[(i, j)]);
        cluster_sandwich(&self.var, &scores, &[clusters], &ClusterOptions::robcov())
    }

    /// Grambsch–Therneau test of proportional hazards, as `cox.zph` of
    /// survival ≥ 3: the score test, at `(β̂, 0)`, for adding the
    /// time-varying covariates `g(t)·x` with `g` the transformed event time
    /// centred at its mean over the event rows. One test per term of
    /// `terms` (column groups; one per column when `None`) plus the global
    /// test.
    pub fn cox_zph(
        &self,
        transform: ZphTransform,
        terms: Option<&[Vec<usize>]>,
    ) -> Result<CoxZph, SurvivalError> {
        let d = &self.data;
        let (n, p) = (d.x.nrows(), self.df);
        let ttimes: Vec<f64> = match transform {
            ZphTransform::Identity => d.stop.clone(),
            ZphTransform::Log => d.stop.iter().map(|t| t.ln()).collect(),
            ZphTransform::Rank => {
                let mut idx: Vec<usize> = (0..n).collect();
                idx.sort_by(|&a, &b| d.stop[a].total_cmp(&d.stop[b]));
                let mut rank = vec![0.0; n];
                let mut a = 0;
                while a < n {
                    let mut b = a;
                    while b < n && d.stop[idx[b]] == d.stop[idx[a]] {
                        b += 1;
                    }
                    let avg = (a + b + 1) as f64 / 2.0;
                    for &i in &idx[a..b] {
                        rank[i] = avg;
                    }
                    a = b;
                }
                rank
            }
            ZphTransform::Km => {
                let input = super::SurvInput {
                    time: &d.stop,
                    event: &d.event,
                    entry: d.entry.as_deref(),
                    weights: None,
                };
                let opts = super::SurvfitOptions {
                    timefix: false,
                    ..Default::default()
                };
                let km = super::survfit(&input, None, &opts)?.remove(0);
                d.stop
                    .iter()
                    .map(|&t| {
                        let k = km.time.partition_point(|&s| s < t);
                        1.0 - if k == 0 { 1.0 } else { km.surv[k - 1] }
                    })
                    .collect()
            }
        };
        let n_ev = d.event.iter().filter(|&&e| e).count() as f64;
        let mean = (0..n)
            .filter(|&i| d.event[i])
            .map(|i| ttimes[i])
            .sum::<f64>()
            / n_ev;
        let mut by_time: Vec<(f64, f64)> = (0..n).map(|i| (d.stop[i], ttimes[i] - mean)).collect();
        by_time.sort_by(|a, b| a.0.total_cmp(&b.0));
        let gfun = |t: f64| by_time[by_time.partition_point(|v| v.0 < t)].1;

        let mut imat = Mat::<f64>::zeros(2 * p, 2 * p);
        let mut u2 = vec![0.0; p];
        d.sweep(&self.coefficients, true, |et, s2d2, _, xd| {
            let (s2, d2) = s2d2.split_at(p * p);
            let g = gfun(et.time);
            let steps: Vec<(f64, f64)> = match self.ties {
                Ties::Breslow => vec![(0.0, et.wd)],
                Ties::Efron => (0..et.m)
                    .map(|k| (k as f64 / et.m as f64, et.wd / et.m as f64))
                    .collect(),
            };
            for a in 0..p {
                u2[a] += g * xd[a];
            }
            for (f, wt) in steps {
                let den = et.s0 - f * et.d0;
                let m: Vec<f64> = (0..p).map(|a| (et.s1[a] - f * et.d1[a]) / den).collect();
                for a in 0..p {
                    u2[a] -= g * wt * m[a];
                    for b in 0..p {
                        let v = (s2[a * p + b] - f * d2[a * p + b]) / den - m[a] * m[b];
                        imat[(a, b)] += wt * v;
                        imat[(a, p + b)] += g * wt * v;
                        imat[(p + a, b)] += g * wt * v;
                        imat[(p + a, p + b)] += g * g * wt * v;
                    }
                }
            }
        });
        if self.ridge > 0.0 {
            for a in 0..2 * p {
                imat[(a, a)] += self.ridge;
            }
        }
        let test = |jj: &[usize]| -> ZphTest {
            let kk: Vec<usize> = (0..p).chain(jj.iter().map(|j| p + j)).collect();
            let q = kk.len();
            let sub = Mat::from_fn(q, q, |a, b| imat[(kk[a], kk[b])]);
            let u = faer::Col::from_fn(q, |a| if a < p { 0.0 } else { u2[jj[a - p]] });
            let x = sub.partial_piv_lu().solve(&u);
            let chisq: f64 = (0..q).map(|a| x[a] * u[a]).sum();
            ZphTest {
                chisq,
                df: jj.len(),
                p_value: ChiSquared::new(jj.len() as f64).map_or(f64::NAN, |c| c.sf(chisq)),
            }
        };
        let default_terms: Vec<Vec<usize>> = (0..p).map(|j| vec![j]).collect();
        let groups = terms.unwrap_or(&default_terms);
        if groups.iter().flatten().any(|&j| j >= p) {
            return Err(SurvivalError::Invalid("term column out of range"));
        }
        let all: Vec<usize> = (0..p).collect();
        let (rows, time, y) = self.schoenfeld_residuals(true);
        Ok(CoxZph {
            terms: groups.iter().map(|g| test(g)).collect(),
            global: test(&all),
            x: rows.iter().map(|&i| ttimes[i]).collect(),
            time,
            y,
        })
    }

    /// Predicted survival curve for the covariate row `x0` (uncentred, no
    /// intercept) in stratum `stratum` (`None` for an unstratified fit), as
    /// `survfit(fit, newdata)`: the Breslow / Efron cumulative hazard
    /// matching the tie method, its standard error (including the
    /// uncertainty of β̂) and log-scale confidence limits.
    pub fn predict_survival(
        &self,
        x0: &[f64],
        stratum: Option<usize>,
    ) -> Result<CoxSurvCurve, SurvivalError> {
        let d = &self.data;
        let p = self.df;
        if x0.len() != p {
            return Err(SurvivalError::Length(p, x0.len()));
        }
        let stratum = match stratum {
            Some(s) => s,
            None if d.groups.len() == 1 => d.groups[0].0,
            None => {
                return Err(SurvivalError::Invalid(
                    "the fit is stratified; give a stratum",
                ))
            }
        };
        let Some((_, rows, _)) = d.groups.iter().find(|g| g.0 == stratum) else {
            return Err(SurvivalError::Invalid("unknown stratum"));
        };
        let xc: Vec<f64> = (0..p).map(|j| x0[j] - d.means[j]).collect();
        let lp0: f64 = (0..p).map(|j| xc[j] * self.coefficients[j]).sum();
        let scale = (-lp0).exp();
        let times = d.event_times(&self.coefficients);
        let mut ev: Vec<&EventTime> = times.iter().filter(|e| e.stratum == stratum).collect();
        ev.sort_by(|a, b| a.time.total_cmp(&b.time));
        // Per event time: hazard, variance term and x̄ term for x0.
        let mut haz = Vec::with_capacity(ev.len());
        for et in &ev {
            let ks: Vec<(f64, f64)> = match self.ties {
                Ties::Breslow => vec![(0.0, et.wd)],
                Ties::Efron => (0..et.m)
                    .map(|k| (k as f64 / et.m as f64, et.wd / et.m as f64))
                    .collect(),
            };
            let (mut h, mut v, mut xs) = (0.0, 0.0, vec![0.0; p]);
            for (f, wt) in ks {
                let den = (et.s0 - f * et.d0) * scale;
                h += wt / den;
                v += wt / (den * den);
                for j in 0..p {
                    let s1 = (et.s1[j] - f * et.d1[j] - xc[j] * (et.s0 - f * et.d0)) * scale;
                    xs[j] += wt * s1 / (den * den);
                }
            }
            haz.push((et.time, h, v, xs));
        }
        let mut times_all: Vec<f64> = rows.iter().map(|&i| d.stop[i]).collect();
        times_all.sort_by(|a, b| a.total_cmp(b));
        times_all.dedup();
        let z = Normal::new(0.0, 1.0)
            .map(|nd| nd.inverse_cdf(1.0 - (1.0 - self.conf_level) / 2.0))
            .unwrap_or(f64::NAN);
        let (mut hcum, mut vcum, mut dcum) = (0.0, 0.0, vec![0.0; p]);
        let mut q = 0;
        let mut out = CoxSurvCurve {
            time: times_all.clone(),
            n_risk: Vec::new(),
            cumhaz: Vec::new(),
            surv: Vec::new(),
            std_err: Vec::new(),
            lower: Vec::new(),
            upper: Vec::new(),
        };
        for &t in &times_all {
            while q < haz.len() && haz[q].0 <= t {
                hcum += haz[q].1;
                vcum += haz[q].2;
                for j in 0..p {
                    dcum[j] += haz[q].3[j];
                }
                q += 1;
            }
            let nr: f64 = rows
                .iter()
                .filter(|&&i| d.stop[i] >= t && d.entry.as_ref().is_none_or(|e| e[i] < t))
                .map(|&i| d.w[i])
                .sum();
            let se = (vcum + quad(&dcum, &self.var)).sqrt();
            let s = (-hcum).exp();
            let (lo, hi) = conf_limits(s, se, z, ConfType::Log);
            out.n_risk.push(nr);
            out.cumhaz.push(hcum);
            out.surv.push(s);
            out.std_err.push(se);
            out.lower.push(lo);
            out.upper.push(hi);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_bad_input() {
        let t = [1.0, 2.0, 3.0];
        let o = CoxOptions::default();
        let x = Mat::from_fn(3, 1, |i, _| i as f64);
        assert!(coxph(&x, &CoxInput::new(&t, &[false; 3]), &o).is_err());
        assert!(coxph(&Mat::zeros(3, 0), &CoxInput::new(&t, &[true; 3]), &o).is_err());
        assert!(coxph(&Mat::zeros(2, 1), &CoxInput::new(&t, &[true; 3]), &o).is_err());
        let bad = [f64::NAN, 0.0, 0.0];
        assert!(coxph(&x, &CoxInput::new(&t, &[true; 3]).offset(&bad), &o).is_err());
    }

    #[test]
    fn prediction_needs_a_stratum_for_stratified_fits() {
        let t = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let e = [true, true, false, true, true, true];
        let s = [0, 0, 0, 1, 1, 1];
        let x = Mat::from_fn(6, 1, |i, _| [0.5, -1.0, 2.0, 0.1, 1.2, -0.3][i]);
        let fit = coxph(
            &x,
            &CoxInput::new(&t, &e).strata(&s),
            &CoxOptions::default(),
        )
        .unwrap();
        assert!(fit.predict_survival(&[0.0], None).is_err());
        assert!(fit.predict_survival(&[0.0], Some(7)).is_err());
        let c = fit.predict_survival(&[0.0], Some(1)).unwrap();
        assert_eq!(c.time, vec![4.0, 5.0, 6.0]);
        assert!(c.surv.windows(2).all(|w| w[1] <= w[0]));
    }
}
