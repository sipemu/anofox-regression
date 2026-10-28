//! Nonparametric survival analysis (RMS Ch 17): Kaplan–Meier and
//! Nelson–Aalen / Fleming–Harrington estimates, and the log-rank / G-ρ tests.
//!
//! * [`survfit`] – `survival::survfit(Surv(...) ~ strata)`: the Kaplan–Meier
//!   survival curve with Greenwood standard errors and `log` / `log-log` /
//!   `plain` confidence intervals, and the Nelson–Aalen (or tie-corrected
//!   Fleming–Harrington) cumulative hazard with its standard error, per
//!   stratum. Supports case weights and left truncation (counting-process
//!   data `(entry, stop]`). [`SurvCurve::median`] and
//!   [`SurvCurve::restricted_mean`] reproduce `summary(survfit)$table`.
//! * [`survdiff`] – `survival::survdiff`: the k-sample log-rank test
//!   (`ρ = 0`) and the Fleming–Harrington G-ρ family (`ρ = 1` is the
//!   Peto–Peto modification of the Gehan–Wilcoxon test), optionally
//!   stratified.
//!
//! # Example
//!
//! ```rust
//! use anofox_regression::survival::{kaplan_meier, SurvfitOptions};
//!
//! let time = [6.0, 7.0, 10.0, 15.0, 19.0, 25.0];
//! let event = [true, false, true, true, false, true];
//! let km = kaplan_meier(&time, &event, &SurvfitOptions::default()).unwrap();
//! assert_eq!(km.time, vec![6.0, 7.0, 10.0, 15.0, 19.0, 25.0]);
//! assert!((km.surv[0] - 5.0 / 6.0).abs() < 1e-12);
//! assert_eq!(km.median().estimate, 15.0);
//! ```

use faer::linalg::solvers::Solve;
use faer::{Col, Mat};
use statrs::distribution::{ChiSquared, ContinuousCDF, Normal};
use thiserror::Error;

/// Errors of the survival estimators.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum SurvivalError {
    /// Inputs of different lengths.
    #[error("inputs have lengths {0} and {1}")]
    Length(usize, usize),
    /// A time or weight is invalid.
    #[error("invalid data: {0}")]
    Invalid(&'static str),
    /// No observation.
    #[error("no observations")]
    Empty,
}

/// Confidence interval of the survival curve (`conf.type` of `survfit`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ConfType {
    /// `S·exp(±z·se)`, upper limit capped at 1 (the default).
    #[default]
    Log,
    /// `S^exp(±z·se / log S)`.
    LogLog,
    /// `S ± z·se·S`, clipped to `[0, 1]`.
    Plain,
}

/// Cumulative hazard estimator (`ctype` of `survfit`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HazardType {
    /// Nelson–Aalen, `Σ d/n` (`ctype = 1`).
    #[default]
    NelsonAalen,
    /// Fleming–Harrington tie correction, `Σ_{j<d} 1/(n − j)` (`ctype = 2`).
    FlemingHarrington,
}

/// Options of [`survfit`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurvfitOptions {
    /// Confidence level of the intervals (default 0.95).
    pub conf_level: f64,
    /// Interval type (default [`ConfType::Log`]).
    pub conf_type: ConfType,
    /// Cumulative hazard estimator (default Nelson–Aalen).
    pub hazard: HazardType,
    /// Merge times that differ only by floating-point noise, as R's
    /// `Surv(..., timefix = TRUE)` (`aeqSurv`) does by default.
    pub timefix: bool,
}

impl Default for SurvfitOptions {
    fn default() -> Self {
        Self {
            conf_level: 0.95,
            conf_type: ConfType::Log,
            hazard: HazardType::NelsonAalen,
            timefix: true,
        }
    }
}

/// Survival data: right-censored `(time, event)`, optionally left-truncated
/// (`entry < time`; the row is at risk on `(entry, time]`) and weighted.
#[derive(Debug, Clone, Copy)]
pub struct SurvInput<'a> {
    /// Event or censoring time (the `stop` time).
    pub time: &'a [f64],
    /// `true` for an event, `false` for a censored row.
    pub event: &'a [bool],
    /// Entry (left-truncation) times.
    pub entry: Option<&'a [f64]>,
    /// Case weights.
    pub weights: Option<&'a [f64]>,
}

impl<'a> SurvInput<'a> {
    /// Right-censored data.
    pub fn new(time: &'a [f64], event: &'a [bool]) -> Self {
        Self {
            time,
            event,
            entry: None,
            weights: None,
        }
    }
    /// Add entry times (counting-process data).
    pub fn entry(mut self, entry: &'a [f64]) -> Self {
        self.entry = Some(entry);
        self
    }
    /// Add case weights.
    pub fn weights(mut self, weights: &'a [f64]) -> Self {
        self.weights = Some(weights);
        self
    }

    fn validate(&self) -> Result<(), SurvivalError> {
        let n = self.time.len();
        if self.event.len() != n {
            return Err(SurvivalError::Length(n, self.event.len()));
        }
        if n == 0 {
            return Err(SurvivalError::Empty);
        }
        if self.time.iter().any(|t| !t.is_finite()) {
            return Err(SurvivalError::Invalid("times must be finite"));
        }
        if let Some(e) = self.entry {
            if e.len() != n {
                return Err(SurvivalError::Length(n, e.len()));
            }
            if (0..n).any(|i| !e[i].is_finite() || e[i] >= self.time[i]) {
                return Err(SurvivalError::Invalid(
                    "entry must be finite and before the stop time",
                ));
            }
        }
        if let Some(w) = self.weights {
            if w.len() != n {
                return Err(SurvivalError::Length(n, w.len()));
            }
            if w.iter().any(|v| !v.is_finite() || *v < 0.0) {
                return Err(SurvivalError::Invalid(
                    "weights must be finite and non-negative",
                ));
            }
        }
        Ok(())
    }
}

/// `survival:::aeqSurv`: map every time (entry and stop) to the first of a
/// run of distinct values whose gaps are within `√ε` (absolutely, or
/// relative to the mean absolute time).
fn timefix(
    time: &[f64],
    entry: Option<&[f64]>,
) -> Result<(Vec<f64>, Option<Vec<f64>>), SurvivalError> {
    let tol = f64::EPSILON.sqrt();
    let mut y: Vec<f64> = time.iter().chain(entry.unwrap_or(&[])).copied().collect();
    y.sort_by(|a, b| a.total_cmp(b));
    y.dedup();
    let mean_abs = y.iter().map(|v| v.abs()).sum::<f64>() / y.len() as f64;
    let mut cuts = vec![y[0]];
    for k in 1..y.len() {
        let dy = y[k] - y[k - 1];
        if !(dy <= tol || dy / mean_abs <= tol) {
            cuts.push(y[k]);
        }
    }
    let snap = |v: f64| cuts[cuts.partition_point(|&c| c <= v) - 1];
    let t: Vec<f64> = time.iter().map(|&v| snap(v)).collect();
    let e = entry.map(|e| e.iter().map(|&v| snap(v)).collect::<Vec<f64>>());
    if let Some(e) = &e {
        if e.iter().zip(&t).any(|(a, b)| a >= b) {
            return Err(SurvivalError::Invalid(
                "an interval has effective length 0 after timefix",
            ));
        }
    }
    Ok((t, e))
}

/// One survival curve (one stratum), at every distinct stop time.
#[derive(Debug, Clone, PartialEq)]
pub struct SurvCurve {
    /// Stratum code (`None` without strata).
    pub stratum: Option<usize>,
    /// Number of rows.
    pub records: usize,
    /// Distinct event / censoring times.
    pub time: Vec<f64>,
    /// Number (sum of weights) at risk just before each time.
    pub n_risk: Vec<f64>,
    /// Events at each time.
    pub n_event: Vec<f64>,
    /// Censorings at each time.
    pub n_censor: Vec<f64>,
    /// Kaplan–Meier survival.
    pub surv: Vec<f64>,
    /// Greenwood standard error of `−log S` (`std.err` of `survfit` with
    /// `robust = FALSE`; for non-integer weights R defaults to a robust
    /// variance).
    pub std_err: Vec<f64>,
    /// Lower confidence limit of `S`.
    pub lower: Vec<f64>,
    /// Upper confidence limit of `S`.
    pub upper: Vec<f64>,
    /// Cumulative hazard.
    pub cumhaz: Vec<f64>,
    /// Standard error of the cumulative hazard.
    pub std_chaz: Vec<f64>,
    /// Smallest entry / stop time (start of the curve for the restricted mean).
    min_time: f64,
}

/// A survival-time quantile with its confidence limits.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurvQuantile {
    /// Estimate (NaN if the curve does not reach the level).
    pub estimate: f64,
    /// Lower confidence limit: the first time the lower confidence curve
    /// reaches the level (as `survfit`).
    pub lower: f64,
    /// Upper confidence limit.
    pub upper: f64,
}

/// The `summary(survfit)$table` row of a curve.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurvSummary {
    /// Rows used.
    pub records: usize,
    /// Largest number at risk.
    pub n_max: f64,
    /// Number at risk at the first time.
    pub n_start: f64,
    /// Total events.
    pub events: f64,
    /// Restricted mean survival time.
    pub rmean: f64,
    /// Its standard error.
    pub se_rmean: f64,
    /// Median survival.
    pub median: f64,
    /// Lower confidence limit of the median.
    pub median_lower: f64,
    /// Upper confidence limit of the median.
    pub median_upper: f64,
}

/// First time a curve reaches 0.5 (`minmin` of `survival:::survmean`): if
/// the curve equals 0.5 over an interval, the midpoint of that interval.
fn first_below_half(y: &[f64], x: &[f64]) -> f64 {
    let tol = f64::EPSILON.sqrt();
    let keep: Vec<usize> = (0..y.len())
        .filter(|&i| !y[i].is_nan() && y[i] < 0.5 + tol)
        .collect();
    let Some(&first) = keep.first() else {
        return f64::NAN;
    };
    if (y[first] - 0.5).abs() < tol {
        if let Some(&next) = keep.iter().find(|&&i| y[i] < y[first]) {
            return (x[first] + x[next]) / 2.0;
        }
    }
    x[first]
}

impl SurvCurve {
    /// Median survival with the limits from the confidence band, as
    /// `summary(survfit)$table`.
    pub fn median(&self) -> SurvQuantile {
        SurvQuantile {
            estimate: first_below_half(&self.surv, &self.time),
            lower: first_below_half(&self.lower, &self.time),
            upper: first_below_half(&self.upper, &self.time),
        }
    }

    /// Restricted mean survival time up to `end_time` and its standard error
    /// (`survival:::survmean`); the curve starts at `min(0, first time)`.
    pub fn restricted_mean(&self, end_time: f64) -> (f64, f64) {
        let start = self.min_time.min(0.0);
        let keep: Vec<usize> = (0..self.time.len())
            .filter(|&i| self.time[i] <= end_time)
            .collect();
        let hh = |i: usize| {
            let (n, d) = (self.n_risk[i], self.n_event[i]);
            if n - d == 0.0 {
                0.0
            } else {
                d / (n * (n - d))
            }
        };
        let (times, surv, h): (Vec<f64>, Vec<f64>, Vec<f64>) = if keep.is_empty() {
            (vec![end_time], vec![1.0], vec![0.0])
        } else {
            let last = *keep.last().expect("non-empty");
            let mut t: Vec<f64> = keep.iter().map(|&i| self.time[i]).collect();
            let mut s: Vec<f64> = keep.iter().map(|&i| self.surv[i]).collect();
            let mut h: Vec<f64> = keep.iter().map(|&i| hh(i)).collect();
            t.push(end_time);
            s.push(self.surv[last]);
            h.push(0.0);
            (t, s, h)
        };
        let n = times.len();
        let mut rect = Vec::with_capacity(n);
        let mut prev = start;
        for k in 0..n {
            let height = if k == 0 { 1.0 } else { surv[k - 1] };
            rect.push((times[k] - prev) * height);
            prev = times[k];
        }
        let mean: f64 = rect.iter().sum();
        // varmean = Σ cumsum(rev(rect[-1]))² · rev(hh)[-1]
        let rr: Vec<f64> = rect[1..].iter().rev().copied().collect();
        let hr: Vec<f64> = h.iter().rev().skip(1).copied().collect();
        let mut acc = 0.0;
        let mut var = 0.0;
        for (r, h) in rr.iter().zip(&hr) {
            acc += r;
            var += acc * acc * h;
        }
        (mean, var.sqrt())
    }

    /// `summary(survfit)$table` with the restricted mean up to `end_time`
    /// (`survfit` uses the last time of the curve, or with strata the
    /// largest last time over the strata).
    pub fn summary(&self, end_time: f64) -> SurvSummary {
        let (rmean, se_rmean) = self.restricted_mean(end_time);
        let m = self.median();
        SurvSummary {
            records: self.records,
            n_max: self.n_risk.iter().copied().fold(0.0, f64::max),
            n_start: self.n_risk.first().copied().unwrap_or(0.0),
            events: self.n_event.iter().sum(),
            rmean,
            se_rmean,
            median: m.estimate,
            median_lower: m.lower,
            median_upper: m.upper,
        }
    }

    /// Survival at time `t` (right-continuous step function, 1 before the
    /// first time).
    pub fn survival_at(&self, t: f64) -> f64 {
        let k = self.time.partition_point(|&s| s <= t);
        if k == 0 {
            1.0
        } else {
            self.surv[k - 1]
        }
    }
}

/// Kaplan–Meier curve of right-censored data without strata or weights.
pub fn kaplan_meier(
    time: &[f64],
    event: &[bool],
    options: &SurvfitOptions,
) -> Result<SurvCurve, SurvivalError> {
    let mut v = survfit(&SurvInput::new(time, event), None, options)?;
    Ok(v.remove(0))
}

/// Kaplan–Meier / Nelson–Aalen curves per stratum (codes in increasing
/// order), as `survfit(Surv(...) ~ strata)`.
pub fn survfit(
    input: &SurvInput,
    strata: Option<&[usize]>,
    options: &SurvfitOptions,
) -> Result<Vec<SurvCurve>, SurvivalError> {
    input.validate()?;
    if !(options.conf_level > 0.0 && options.conf_level < 1.0) {
        return Err(SurvivalError::Invalid("conf_level must be in (0, 1)"));
    }
    let n = input.time.len();
    if let Some(s) = strata {
        if s.len() != n {
            return Err(SurvivalError::Length(n, s.len()));
        }
    }
    let mut codes: Vec<usize> = strata.map_or(vec![0], |s| s.to_vec());
    codes.sort_unstable();
    codes.dedup();
    let z = Normal::new(0.0, 1.0)
        .map(|d| d.inverse_cdf(1.0 - (1.0 - options.conf_level) / 2.0))
        .unwrap_or(f64::NAN);
    let fixed = if options.timefix {
        Some(timefix(input.time, input.entry)?)
    } else {
        None
    };
    let input = match &fixed {
        Some((t, e)) => SurvInput {
            time: t,
            event: input.event,
            entry: e.as_deref(),
            weights: input.weights,
        },
        None => *input,
    };
    let input = &input;
    Ok(codes
        .iter()
        .map(|&c| {
            let rows: Vec<usize> = (0..n)
                .filter(|&i| strata.is_none_or(|s| s[i] == c))
                .collect();
            let mut curve = one_curve(input, &rows, options, z);
            curve.stratum = strata.map(|_| c);
            curve
        })
        .collect())
}

fn one_curve(input: &SurvInput, rows: &[usize], options: &SurvfitOptions, z: f64) -> SurvCurve {
    let w = |i: usize| input.weights.map_or(1.0, |w| w[i]);
    let mut times: Vec<f64> = rows.iter().map(|&i| input.time[i]).collect();
    times.sort_by(|a, b| a.total_cmp(b));
    times.dedup();
    // Sorted (stop, weight) and (entry, weight) with suffix sums for the
    // risk sets: at risk at t iff entry < t <= stop.
    let suffix = |mut v: Vec<(f64, f64)>| {
        v.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut s = vec![0.0; v.len() + 1];
        for k in (0..v.len()).rev() {
            s[k] = s[k + 1] + v[k].1;
        }
        (v, s)
    };
    let (stops, stop_sum) = suffix(rows.iter().map(|&i| (input.time[i], w(i))).collect());
    let entries = input
        .entry
        .map(|e| suffix(rows.iter().map(|&i| (e[i], w(i))).collect()));
    let at_least = |v: &[(f64, f64)], s: &[f64], t: f64| s[v.partition_point(|p| p.0 < t)];
    let min_time = rows
        .iter()
        .map(|&i| input.entry.map_or(input.time[i], |e| e[i]))
        .fold(f64::INFINITY, f64::min);

    let mut ev = vec![0.0; times.len()];
    let mut cens = vec![0.0; times.len()];
    let mut n_ev_rows = vec![0usize; times.len()];
    for &i in rows {
        let k = times.partition_point(|&t| t < input.time[i]);
        if input.event[i] {
            ev[k] += w(i);
            n_ev_rows[k] += usize::from(w(i) > 0.0);
        } else {
            cens[k] += w(i);
        }
    }
    let m = times.len();
    let mut out = SurvCurve {
        stratum: None,
        records: rows.len(),
        time: times.clone(),
        n_risk: vec![0.0; m],
        n_event: ev.clone(),
        n_censor: cens,
        surv: vec![0.0; m],
        std_err: vec![0.0; m],
        lower: vec![0.0; m],
        upper: vec![0.0; m],
        cumhaz: vec![0.0; m],
        std_chaz: vec![0.0; m],
        min_time,
    };
    let (mut s, mut varhaz, mut h, mut varh) = (1.0, 0.0, 0.0, 0.0);
    for k in 0..m {
        let t = times[k];
        let nr = at_least(&stops, &stop_sum, t)
            - entries.as_ref().map_or(0.0, |(v, sm)| at_least(v, sm, t));
        let d = ev[k];
        out.n_risk[k] = nr;
        if d > 0.0 {
            s *= 1.0 - d / nr;
            varhaz += if nr - d > 0.0 {
                d / (nr * (nr - d))
            } else {
                f64::INFINITY
            };
            match options.hazard {
                HazardType::NelsonAalen => {
                    h += d / nr;
                    varh += d / (nr * nr);
                }
                HazardType::FlemingHarrington => {
                    let rows_k = n_ev_rows[k].max(1) as f64;
                    let wbar = d / rows_k;
                    for j in 0..n_ev_rows[k].max(1) {
                        let r = nr - j as f64 * wbar;
                        h += wbar / r;
                        varh += wbar * wbar / (r * r);
                    }
                }
            }
        }
        out.surv[k] = s;
        out.std_err[k] = varhaz.sqrt();
        out.cumhaz[k] = h;
        out.std_chaz[k] = varh.sqrt();
        let (lo, hi) = conf_limits(s, out.std_err[k], z, options.conf_type);
        out.lower[k] = lo;
        out.upper[k] = hi;
    }
    out
}

/// `survfit_confint` of the survival package.
fn conf_limits(p: f64, se: f64, z: f64, kind: ConfType) -> (f64, f64) {
    match kind {
        ConfType::Plain => {
            let e = z * se * p;
            ((p - e).max(0.0), (p + e).min(1.0))
        }
        ConfType::Log => {
            if p == 0.0 {
                return (f64::NAN, f64::NAN);
            }
            let e = z * se;
            ((p.ln() - e).exp(), (p.ln() + e).exp().min(1.0))
        }
        ConfType::LogLog => {
            if p == 0.0 || p == 1.0 {
                return (f64::NAN, f64::NAN);
            }
            let e = z * se / p.ln();
            let ll = (-p.ln()).ln();
            ((-(ll - e).exp()).exp(), (-(ll + e).exp()).exp())
        }
    }
}

/// Result of [`survdiff`].
#[derive(Debug, Clone, PartialEq)]
pub struct SurvDiff {
    /// Group codes, in increasing order.
    pub groups: Vec<usize>,
    /// Rows per group.
    pub n: Vec<usize>,
    /// (Weighted) observed events per group.
    pub observed: Vec<f64>,
    /// (Weighted) expected events per group.
    pub expected: Vec<f64>,
    /// Covariance of `observed − expected`.
    pub variance: Mat<f64>,
    /// χ² statistic.
    pub chisq: f64,
    /// Degrees of freedom (groups with expected events, minus one).
    pub df: usize,
    /// p-value.
    pub p_value: f64,
}

/// k-sample G-ρ test of right-censored data (`survival::survdiff`, with
/// R's default `timefix`):
/// `ρ = 0` is the log-rank test, `ρ = 1` the Peto–Peto test. Each event
/// time is weighted by `S(t−)^ρ`, the pooled Kaplan–Meier estimate just
/// before `t` within its stratum; with `strata` the observed, expected and
/// variance are summed over strata.
pub fn survdiff(
    time: &[f64],
    event: &[bool],
    group: &[usize],
    strata: Option<&[usize]>,
    rho: f64,
) -> Result<SurvDiff, SurvivalError> {
    SurvInput::new(time, event).validate()?;
    let (fixed, _) = timefix(time, None)?;
    let time = &fixed[..];
    let n = time.len();
    if group.len() != n {
        return Err(SurvivalError::Length(n, group.len()));
    }
    if let Some(s) = strata {
        if s.len() != n {
            return Err(SurvivalError::Length(n, s.len()));
        }
    }
    let mut groups: Vec<usize> = group.to_vec();
    groups.sort_unstable();
    groups.dedup();
    let k = groups.len();
    let gi: Vec<usize> = group
        .iter()
        .map(|g| groups.binary_search(g).expect("present"))
        .collect();
    let mut st: Vec<usize> = strata.map_or(vec![0], |s| s.to_vec());
    st.sort_unstable();
    st.dedup();

    let mut obs = vec![0.0; k];
    let mut exp = vec![0.0; k];
    let mut var = Mat::<f64>::zeros(k, k);
    for &s in &st {
        let mut rows: Vec<usize> = (0..n)
            .filter(|&i| strata.is_none_or(|v| v[i] == s))
            .collect();
        rows.sort_by(|&a, &b| time[a].total_cmp(&time[b]));
        // At risk per group: rows with time >= t; walk the sorted rows.
        let mut at_risk = vec![0.0; k];
        for &i in &rows {
            at_risk[gi[i]] += 1.0;
        }
        let mut km: f64 = 1.0;
        let mut a = 0;
        while a < rows.len() {
            let t = time[rows[a]];
            let mut b = a;
            let mut d_g = vec![0.0; k];
            let mut leaving = vec![0.0; k];
            while b < rows.len() && time[rows[b]] == t {
                let i = rows[b];
                if event[i] {
                    d_g[gi[i]] += 1.0;
                }
                leaving[gi[i]] += 1.0;
                b += 1;
            }
            let d: f64 = d_g.iter().sum();
            let nr: f64 = at_risk.iter().sum();
            if d > 0.0 {
                let wt = km.powf(rho);
                for g in 0..k {
                    obs[g] += wt * d_g[g];
                    exp[g] += wt * d * at_risk[g] / nr;
                }
                if nr > 1.0 {
                    let f = wt * wt * d * (nr - d) / (nr - 1.0);
                    for g in 0..k {
                        for h in 0..k {
                            let delta = if g == h { 1.0 } else { 0.0 };
                            var[(g, h)] += f * at_risk[g] / nr * (delta - at_risk[h] / nr);
                        }
                    }
                }
                km *= 1.0 - d / nr;
            }
            for g in 0..k {
                at_risk[g] -= leaving[g];
            }
            a = b;
        }
    }
    let used: Vec<usize> = (0..k).filter(|&g| exp[g] > 0.0).collect();
    let (chisq, df) = if used.len() < 2 {
        (0.0, 0)
    } else {
        let idx = &used[1..];
        let q = idx.len();
        let diff = Col::from_fn(q, |a| obs[idx[a]] - exp[idx[a]]);
        let v = Mat::from_fn(q, q, |a, b| var[(idx[a], idx[b])]);
        let x = v.partial_piv_lu().solve(&diff);
        ((0..q).map(|a| x[a] * diff[a]).sum(), q)
    };
    let p_value = if df == 0 {
        f64::NAN
    } else {
        ChiSquared::new(df as f64).map_or(f64::NAN, |d| d.sf(chisq))
    };
    Ok(SurvDiff {
        n: groups
            .iter()
            .enumerate()
            .map(|(g, _)| gi.iter().filter(|&&v| v == g).count())
            .collect(),
        groups,
        observed: obs,
        expected: exp,
        variance: var,
        chisq,
        df,
        p_value,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hand_computed_km() {
        let time = [1.0, 2.0, 2.0, 3.0, 4.0];
        let event = [true, true, false, true, false];
        let km = kaplan_meier(&time, &event, &SurvfitOptions::default()).unwrap();
        assert_eq!(km.time, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(km.n_risk, vec![5.0, 4.0, 2.0, 1.0]);
        let s = [0.8, 0.8 * 0.75, 0.8 * 0.75 * 0.5, 0.8 * 0.75 * 0.5];
        for (a, b) in km.surv.iter().zip(s) {
            assert!((a - b).abs() < 1e-12);
        }
        assert_eq!(km.survival_at(0.5), 1.0);
        assert!((km.survival_at(2.5) - 0.6).abs() < 1e-12);
    }

    #[test]
    fn errors() {
        let o = SurvfitOptions::default();
        assert!(kaplan_meier(&[], &[], &o).is_err());
        assert!(kaplan_meier(&[1.0], &[true, false], &o).is_err());
        let i = SurvInput::new(&[1.0, 2.0], &[true, true]).entry(&[1.0, 0.0]);
        assert!(survfit(&i, None, &o).is_err());
        assert!(survdiff(&[1.0, 2.0], &[true, true], &[0], None, 0.0).is_err());
    }
}
