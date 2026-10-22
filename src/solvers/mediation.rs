//! Causal mediation analysis for linear models (issue #91).
//!
//! How much of the effect of a treatment `X` on an outcome `Y` runs through a
//! mediator `M`? Two OLS models are fitted on the same rows:
//!
//! ```text
//!   mediator model   M ~ X + C          a  = coefficient of X
//!   outcome model    Y ~ X + M + C      b  = coefficient of M,  c' = coefficient of X
//! ```
//!
//! With `delta = treat_value - control_value` the effects are
//!
//! | effect | value |
//! |---|---|
//! | ACME (average causal mediation effect, indirect) | `a * b * delta` |
//! | ADE (average direct effect) | `c' * delta` |
//! | total effect | `(a * b + c') * delta` (equals `c * delta` of `Y ~ X + C`) |
//! | proportion mediated | `ACME / total` |
//!
//! The scope is the linear, no-interaction case of R's `mediation::mediate`:
//! continuous mediator and outcome, one treatment (binary or continuous) and
//! optional covariates `C`. Rows with a non-finite value in any of `x`, `m`,
//! `y` or a covariate are dropped from both models, so they use the same rows.
//!
//! # Inference
//!
//! * [`MediationMethod::Bootstrap`] (default): nonparametric case-resampling
//!   bootstrap. Both models are refitted on each of `n_sims` resamples drawn with
//!   the crate's seeded generator. The point estimates are those of the original
//!   data, `std_error` is the standard deviation of the draws, the interval is
//!   the percentile interval (R's `quantile` type 7) or `mediate`'s BCa interval
//!   ([`BootCi`]), and the two-sided p-value is `mediate`'s
//!   `2 * min(#{draws > 0}, #{draws < 0}) / #draws`. `statistic` is NaN.
//! * [`MediationMethod::Sobel`]: delta-method (first-order Sobel) standard
//!   error `sqrt(a^2 se_b^2 + b^2 se_a^2)` for the ACME with a normal z test;
//!   the ADE and the total effect use the OLS t inference of `c'` (outcome model)
//!   and of `c` (`Y ~ X + C`); the proportion mediated uses the delta method with
//!   the outcome model's covariance of `b` and `c'`.
//! * [`MediationMethod::QuasiBayesian`]: `mediate`'s default Monte Carlo. The
//!   coefficients are drawn from their asymptotic normal distributions
//!   (`N(beta_hat, vcov)` of each model); as in `mediate`, the point estimates are
//!   the means of the draws (the median for the proportion mediated).
//!
//! # Agreement with R
//!
//! Point estimates match `mediation::mediate(model.m, model.y, treat, mediator,
//! boot = TRUE)` exactly; given the same draws the intervals and p-values match
//! too (see `tests/r_validation_mediation.rs`). The draws themselves come from
//! this crate's generator, so bootstrap and Monte Carlo intervals agree with R
//! only within simulation error.
//!
//! # Example
//!
//! ```
//! use anofox_regression::solvers::mediation::{mediate, MediationMethod, MediationOptions};
//!
//! let x = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
//! let m = [0.1, 0.4, 0.2, 0.0, 1.1, 0.8, 1.3, 0.9];
//! let y = [0.3, 0.7, 0.2, 0.1, 1.9, 1.4, 2.2, 1.5];
//! let opts = MediationOptions { method: MediationMethod::Sobel, ..Default::default() };
//! let r = mediate(&y, &m, &x, &[] as &[Vec<f64>], &opts).unwrap();
//! assert!((r.acme.estimate + r.ade.estimate - r.total.estimate).abs() < 1e-12);
//! ```

use crate::solvers::{FittedRegressor, OlsRegressor, Regressor};
use faer::{Col, Mat};
use statrs::distribution::{ContinuousCDF, Normal, StudentsT};
use thiserror::Error;

/// Errors from [`mediate`].
#[derive(Error, Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum MediationError {
    /// `x`, `m`, `y` and the covariates have different lengths.
    #[error("Dimension mismatch: {field} has {got} values, expected {expected}")]
    DimensionMismatch {
        /// The input with the wrong length.
        field: &'static str,
        /// Its length.
        got: usize,
        /// The length of `y`.
        expected: usize,
    },
    /// Too few complete rows to fit the outcome model with a residual degree of freedom.
    #[error("Insufficient data: {usable} complete rows, need at least {needed}")]
    InsufficientData {
        /// Number of complete rows.
        usable: usize,
        /// Minimum number of rows.
        needed: usize,
    },
    /// The treatment or the mediator is constant or collinear with the covariates,
    /// so `a`, `b` or `c'` is not identified.
    #[error("Not identified: {0}")]
    NotIdentified(String),
    /// Fewer than two bootstrap resamples could be fitted.
    #[error("Bootstrap failed: only {fitted} of {requested} resamples could be fitted")]
    BootstrapFailed {
        /// Resamples whose fits gave finite effects.
        fitted: usize,
        /// `n_sims`.
        requested: usize,
    },
    /// An option outside its domain.
    #[error("Invalid value for {field}: {message}")]
    InvalidValue {
        /// Name of the offending option.
        field: &'static str,
        /// What is wrong with it.
        message: String,
    },
}

/// How the uncertainty of the effects is estimated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MediationMethod {
    /// Nonparametric case-resampling bootstrap (`mediate(boot = TRUE)`).
    #[default]
    Bootstrap,
    /// Delta-method (Sobel) standard errors and normal/t intervals.
    Sobel,
    /// Quasi-Bayesian Monte Carlo over the coefficients' sampling distributions
    /// (`mediate(boot = FALSE)`).
    QuasiBayesian,
}

/// Interval type of the resampling methods.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BootCi {
    /// Percentile interval, R's `quantile(draws, c(low, high))` (type 7).
    #[default]
    Percentile,
    /// `mediate`'s bias-corrected and accelerated interval (`boot.ci.type = "bca"`):
    /// bias correction from the share of draws below their mean, acceleration
    /// from the skewness of the draws. Only used by [`MediationMethod::Bootstrap`].
    Bca,
}

/// Options for [`mediate`].
#[derive(Debug, Clone, PartialEq)]
pub struct MediationOptions {
    /// Inference method.
    pub method: MediationMethod,
    /// Interval type of the bootstrap.
    pub boot_ci: BootCi,
    /// Number of bootstrap resamples or Monte Carlo draws (at least 2).
    pub n_sims: usize,
    /// Seed of the resampling / Monte Carlo generator.
    pub seed: u64,
    /// Confidence level of the intervals, in `(0, 1)`.
    pub conf_level: f64,
    /// Treatment value of the contrast (`treat.value` in `mediate`).
    pub treat_value: f64,
    /// Control value of the contrast (`control.value` in `mediate`).
    pub control_value: f64,
}

impl Default for MediationOptions {
    fn default() -> Self {
        Self {
            method: MediationMethod::Bootstrap,
            boot_ci: BootCi::Percentile,
            n_sims: 1000,
            seed: 42,
            conf_level: 0.95,
            treat_value: 1.0,
            control_value: 0.0,
        }
    }
}

/// One effect with its inference.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MediationEffect {
    /// Point estimate.
    pub estimate: f64,
    /// Standard error (the standard deviation of the draws for the resampling methods).
    pub std_error: f64,
    /// z or t statistic for [`MediationMethod::Sobel`]; NaN otherwise.
    pub statistic: f64,
    /// Two-sided p-value.
    pub p_value: f64,
    /// Lower confidence bound.
    pub conf_low: f64,
    /// Upper confidence bound.
    pub conf_high: f64,
}

impl MediationEffect {
    fn nan() -> Self {
        Self {
            estimate: f64::NAN,
            std_error: f64::NAN,
            statistic: f64::NAN,
            p_value: f64::NAN,
            conf_low: f64::NAN,
            conf_high: f64::NAN,
        }
    }

    /// Summarise simulation draws of an effect the way `mediation::mediate` does:
    /// `std_error` is the standard deviation of the finite draws, the interval is
    /// the percentile or BCa interval and the p-value is
    /// `2 * min(#{draws > 0}, #{draws < 0}) / #draws` (1 when `estimate` is 0).
    /// `statistic` is NaN.
    pub fn from_draws(estimate: f64, draws: &[f64], conf_level: f64, ci: BootCi) -> Self {
        let mut d: Vec<f64> = draws.iter().copied().filter(|v| v.is_finite()).collect();
        if d.len() < 2 || !estimate.is_finite() {
            return Self {
                estimate,
                ..Self::nan()
            };
        }
        let nd = d.len() as f64;
        let mean = d.iter().sum::<f64>() / nd;
        let sd = (d.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (nd - 1.0)).sqrt();
        let p_value = if estimate == 0.0 {
            1.0
        } else {
            let pos = d.iter().filter(|&&v| v > 0.0).count();
            let neg = d.iter().filter(|&&v| v < 0.0).count();
            (2.0 * pos.min(neg) as f64 / nd).min(1.0)
        };
        let low = (1.0 - conf_level) / 2.0;
        let high = 1.0 - low;
        let (p_lo, p_hi) = match ci {
            BootCi::Percentile => (low, high),
            BootCi::Bca => bca_levels(&d, mean, low, high),
        };
        d.sort_by(|a, b| a.total_cmp(b));
        Self {
            estimate,
            std_error: sd,
            statistic: f64::NAN,
            p_value,
            conf_low: quantile_type7(&d, p_lo),
            conf_high: quantile_type7(&d, p_hi),
        }
    }
}

/// Result of [`mediate`].
#[derive(Debug, Clone, PartialEq)]
pub struct MediationResult {
    /// Average causal mediation effect (indirect effect).
    pub acme: MediationEffect,
    /// Average direct effect.
    pub ade: MediationEffect,
    /// Total effect.
    pub total: MediationEffect,
    /// Proportion mediated, `acme / total` (NaN when the total effect is 0).
    pub prop_mediated: MediationEffect,
    /// Coefficient of the treatment in the mediator model.
    pub a: f64,
    /// Coefficient of the mediator in the outcome model.
    pub b: f64,
    /// Coefficient of the treatment in the outcome model (the direct path).
    pub c_prime: f64,
    /// R² of the mediator model.
    pub r_squared_mediator: f64,
    /// R² of the outcome model.
    pub r_squared_outcome: f64,
    /// Complete rows used by both models.
    pub n_obs: usize,
    /// Resamples or draws that entered the summaries (0 for Sobel).
    pub n_sims: usize,
    /// The method used.
    pub method: MediationMethod,
    /// The interval type used (meaningful for the bootstrap).
    pub boot_ci: BootCi,
    /// Confidence level of the intervals.
    pub conf_level: f64,
}

/// Causal mediation analysis of the effect of `x` on `y` through `m`, adjusting
/// for `covariates` (one slice per covariate, each as long as `y`; may be empty).
pub fn mediate<C: AsRef<[f64]>>(
    y: &[f64],
    m: &[f64],
    x: &[f64],
    covariates: &[C],
    options: &MediationOptions,
) -> Result<MediationResult, MediationError> {
    validate_options(options)?;
    let n_in = y.len();
    for (field, len) in [("m", m.len()), ("x", x.len())] {
        if len != n_in {
            return Err(MediationError::DimensionMismatch {
                field,
                got: len,
                expected: n_in,
            });
        }
    }
    for c in covariates {
        if c.as_ref().len() != n_in {
            return Err(MediationError::DimensionMismatch {
                field: "covariates",
                got: c.as_ref().len(),
                expected: n_in,
            });
        }
    }

    // Listwise deletion: both models see the same rows.
    let rows: Vec<usize> = (0..n_in)
        .filter(|&i| {
            y[i].is_finite()
                && m[i].is_finite()
                && x[i].is_finite()
                && covariates.iter().all(|c| c.as_ref()[i].is_finite())
        })
        .collect();
    let data = Data {
        y: rows.iter().map(|&i| y[i]).collect(),
        m: rows.iter().map(|&i| m[i]).collect(),
        x: rows.iter().map(|&i| x[i]).collect(),
        c: covariates
            .iter()
            .map(|c| rows.iter().map(|&i| c.as_ref()[i]).collect())
            .collect(),
    };
    let n = rows.len();
    // Outcome model: intercept, x, m and the covariates, plus one residual df.
    let needed = data.c.len() + 4;
    if n < needed {
        return Err(MediationError::InsufficientData { usable: n, needed });
    }

    let all: Vec<usize> = (0..n).collect();
    let fit = fit_paths(&data, &all, true)
        .ok_or_else(|| MediationError::NotIdentified("the models could not be fitted".into()))?;
    let delta = options.treat_value - options.control_value;
    let acme = fit.a * fit.b * delta;
    let ade = fit.c_prime * delta;
    let total = acme + ade;
    let prop = acme / total;

    let (acme_e, ade_e, total_e, prop_e, n_sims) = match options.method {
        MediationMethod::Sobel => {
            let (a_e, d_e, t_e, p_e) = sobel(&data, &fit, delta, options.conf_level)?;
            (a_e, d_e, t_e, p_e, 0)
        }
        MediationMethod::Bootstrap => {
            let draws = bootstrap_draws(&data, options, delta)?;
            let s = |est: f64, d: &[f64]| {
                MediationEffect::from_draws(est, d, options.conf_level, options.boot_ci)
            };
            (
                s(acme, &draws.acme),
                s(ade, &draws.ade),
                s(total, &draws.total),
                s(prop, &draws.prop),
                draws.acme.len(),
            )
        }
        MediationMethod::QuasiBayesian => {
            let draws = quasi_bayesian_draws(&fit, options, delta)?;
            let mean = |d: &[f64]| d.iter().sum::<f64>() / d.len() as f64;
            let s = |est: f64, d: &[f64]| {
                MediationEffect::from_draws(est, d, options.conf_level, BootCi::Percentile)
            };
            (
                s(mean(&draws.acme), &draws.acme),
                s(mean(&draws.ade), &draws.ade),
                s(mean(&draws.total), &draws.total),
                s(median(&draws.prop), &draws.prop),
                draws.acme.len(),
            )
        }
    };

    Ok(MediationResult {
        acme: acme_e,
        ade: ade_e,
        total: total_e,
        prop_mediated: prop_e,
        a: fit.a,
        b: fit.b,
        c_prime: fit.c_prime,
        r_squared_mediator: fit.r2_m,
        r_squared_outcome: fit.r2_y,
        n_obs: n,
        n_sims,
        method: options.method,
        boot_ci: options.boot_ci,
        conf_level: options.conf_level,
    })
}

fn validate_options(o: &MediationOptions) -> Result<(), MediationError> {
    if !(o.conf_level > 0.0 && o.conf_level < 1.0) {
        return Err(MediationError::InvalidValue {
            field: "conf_level",
            message: format!("must lie strictly between 0 and 1, got {}", o.conf_level),
        });
    }
    if o.method != MediationMethod::Sobel && o.n_sims < 2 {
        return Err(MediationError::InvalidValue {
            field: "n_sims",
            message: format!("must be at least 2, got {}", o.n_sims),
        });
    }
    if !o.treat_value.is_finite() || !o.control_value.is_finite() {
        return Err(MediationError::InvalidValue {
            field: "treat_value",
            message: "treat_value and control_value must be finite".into(),
        });
    }
    if o.treat_value == o.control_value {
        return Err(MediationError::InvalidValue {
            field: "treat_value",
            message: "treat_value and control_value must differ".into(),
        });
    }
    Ok(())
}

/// Complete rows, column by column.
struct Data {
    y: Vec<f64>,
    m: Vec<f64>,
    x: Vec<f64>,
    c: Vec<Vec<f64>>,
}

/// The fitted paths and, for the original data, their inference.
struct Paths {
    a: f64,
    b: f64,
    c_prime: f64,
    r2_m: f64,
    r2_y: f64,
    /// Var(a).
    var_a: f64,
    /// Covariance of (b, c') in the outcome model: [var_b, cov, var_c'].
    vcov_bc: [f64; 3],
    df_y: f64,
}

/// Fit `m ~ x + c` and `y ~ x + m + c` on the rows `idx` (with repetition).
/// `None` when a path is aliased or not finite.
fn fit_paths(d: &Data, idx: &[usize], inference: bool) -> Option<Paths> {
    let n = idx.len();
    let k = d.c.len();
    let xm = Mat::from_fn(n, 1 + k, |i, j| {
        if j == 0 {
            d.x[idx[i]]
        } else {
            d.c[j - 1][idx[i]]
        }
    });
    let xy = Mat::from_fn(n, 2 + k, |i, j| match j {
        0 => d.x[idx[i]],
        1 => d.m[idx[i]],
        _ => d.c[j - 2][idx[i]],
    });
    let ym = Col::from_fn(n, |i| d.m[idx[i]]);
    let yy = Col::from_fn(n, |i| d.y[idx[i]]);
    let ols = OlsRegressor::builder()
        .with_intercept(true)
        .compute_inference(inference)
        .build();
    let fm = ols.fit(&xm, &ym).ok()?;
    let fy = ols.fit(&xy, &yy).ok()?;
    let rm = fm.result();
    let ry = fy.result();
    if rm.aliased[0] || ry.aliased[0] || ry.aliased[1] {
        return None;
    }
    let (a, c_prime, b) = (rm.coefficients[0], ry.coefficients[0], ry.coefficients[1]);
    if !(a.is_finite() && b.is_finite() && c_prime.is_finite()) {
        return None;
    }
    let mut p = Paths {
        a,
        b,
        c_prime,
        r2_m: rm.r_squared,
        r2_y: ry.r_squared,
        var_a: f64::NAN,
        vcov_bc: [f64::NAN; 3],
        df_y: (n - ry.rank - 1) as f64,
    };
    if inference {
        // variance_factor() puts the intercept at index 0, then the features.
        let vm = fm.variance_factor()?;
        let vy = fy.variance_factor()?;
        p.var_a = rm.mse * vm[(1, 1)];
        p.vcov_bc = [
            ry.mse * vy[(2, 2)],
            ry.mse * vy[(2, 1)],
            ry.mse * vy[(1, 1)],
        ];
    }
    Some(p)
}

fn sobel(
    d: &Data,
    f: &Paths,
    delta: f64,
    conf_level: f64,
) -> Result<
    (
        MediationEffect,
        MediationEffect,
        MediationEffect,
        MediationEffect,
    ),
    MediationError,
> {
    let normal = Normal::new(0.0, 1.0).expect("standard normal");
    let zq = normal.inverse_cdf(1.0 - (1.0 - conf_level) / 2.0);
    let [var_b, cov_bc, var_c] = f.vcov_bc;
    let z_effect = |est: f64, se: f64| {
        let z = est / se;
        MediationEffect {
            estimate: est,
            std_error: se,
            statistic: z,
            p_value: 2.0 * normal.cdf(-z.abs()),
            conf_low: est - zq * se,
            conf_high: est + zq * se,
        }
    };
    let t_effect = |est: f64, se: f64, df: f64| -> MediationEffect {
        let Ok(t) = StudentsT::new(0.0, 1.0, df) else {
            return MediationEffect {
                estimate: est,
                ..MediationEffect::nan()
            };
        };
        let tq = t.inverse_cdf(1.0 - (1.0 - conf_level) / 2.0);
        let stat = est / se;
        MediationEffect {
            estimate: est,
            std_error: se,
            statistic: stat,
            p_value: 2.0 * t.cdf(-stat.abs()),
            conf_low: est - tq * se,
            conf_high: est + tq * se,
        }
    };

    let (a, b, cp) = (f.a, f.b, f.c_prime);
    let acme_se = delta.abs() * (a * a * var_b + b * b * f.var_a).sqrt();
    let acme = z_effect(a * b * delta, acme_se);
    let ade = t_effect(cp * delta, delta.abs() * var_c.sqrt(), f.df_y);

    // Total effect: c of y ~ x + c (equal to a*b + c' on the same rows).
    let all: Vec<usize> = (0..d.y.len()).collect();
    let k = d.c.len();
    let xt = Mat::from_fn(all.len(), 1 + k, |i, j| {
        if j == 0 {
            d.x[i]
        } else {
            d.c[j - 1][i]
        }
    });
    let yt = Col::from_fn(all.len(), |i| d.y[i]);
    let ft = OlsRegressor::builder()
        .with_intercept(true)
        .compute_inference(true)
        .build()
        .fit(&xt, &yt)
        .map_err(|e| MediationError::NotIdentified(format!("total-effect model: {e}")))?;
    let rt = ft.result();
    let se_c = rt.std_errors.as_ref().map(|s| s[0]).unwrap_or(f64::NAN);
    let total = t_effect(
        (a * b + cp) * delta,
        delta.abs() * se_c,
        (all.len() - rt.rank - 1) as f64,
    );

    // Proportion mediated ab / (ab + c'): delta method, a independent of (b, c').
    let t = a * b + cp;
    let (ga, gb, gc) = (b * cp / (t * t), a * cp / (t * t), -a * b / (t * t));
    let var_p = ga * ga * f.var_a + gb * gb * var_b + gc * gc * var_c + 2.0 * gb * gc * cov_bc;
    let prop = z_effect(a * b / t, var_p.sqrt());
    Ok((acme, ade, total, prop))
}

struct Draws {
    acme: Vec<f64>,
    ade: Vec<f64>,
    total: Vec<f64>,
    prop: Vec<f64>,
}

impl Draws {
    fn with_capacity(n: usize) -> Self {
        Self {
            acme: Vec::with_capacity(n),
            ade: Vec::with_capacity(n),
            total: Vec::with_capacity(n),
            prop: Vec::with_capacity(n),
        }
    }

    fn push(&mut self, a: f64, b: f64, c_prime: f64, delta: f64) {
        let acme = a * b * delta;
        let ade = c_prime * delta;
        let total = acme + ade;
        self.acme.push(acme);
        self.ade.push(ade);
        self.total.push(total);
        self.prop.push(acme / total);
    }
}

fn bootstrap_draws(d: &Data, o: &MediationOptions, delta: f64) -> Result<Draws, MediationError> {
    let n = d.y.len();
    let mut rng = SplitMix64::new(o.seed);
    let mut draws = Draws::with_capacity(o.n_sims);
    let mut idx = vec![0usize; n];
    for _ in 0..o.n_sims {
        for v in idx.iter_mut() {
            *v = rng.below(n);
        }
        // A degenerate resample (e.g. a constant treatment) is skipped.
        if let Some(p) = fit_paths(d, &idx, false) {
            draws.push(p.a, p.b, p.c_prime, delta);
        }
    }
    if draws.acme.len() < 2 {
        return Err(MediationError::BootstrapFailed {
            fitted: draws.acme.len(),
            requested: o.n_sims,
        });
    }
    Ok(draws)
}

fn quasi_bayesian_draws(
    f: &Paths,
    o: &MediationOptions,
    delta: f64,
) -> Result<Draws, MediationError> {
    let [var_b, cov_bc, var_c] = f.vcov_bc;
    if !(f.var_a >= 0.0 && var_b > 0.0 && var_c >= 0.0) {
        return Err(MediationError::NotIdentified(
            "the coefficient covariance is not available".into(),
        ));
    }
    // Cholesky factor of the (b, c') covariance.
    let l11 = var_b.sqrt();
    let l21 = cov_bc / l11;
    let l22 = (var_c - l21 * l21).max(0.0).sqrt();
    let sd_a = f.var_a.sqrt();
    let mut rng = SplitMix64::new(o.seed);
    let mut draws = Draws::with_capacity(o.n_sims);
    for _ in 0..o.n_sims {
        let (z1, z2, z3) = (rng.normal(), rng.normal(), rng.normal());
        let a = f.a + sd_a * z1;
        let b = f.b + l11 * z2;
        let cp = f.c_prime + l21 * z2 + l22 * z3;
        draws.push(a, b, cp, delta);
    }
    Ok(draws)
}

/// R's `quantile(x, p, type = 7)` of sorted data.
fn quantile_type7(sorted: &[f64], p: f64) -> f64 {
    if !p.is_finite() {
        return f64::NAN;
    }
    let n = sorted.len();
    let h = (n - 1) as f64 * p.clamp(0.0, 1.0);
    let lo = h.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    sorted[lo] + (h - lo as f64) * (sorted[hi] - sorted[lo])
}

/// Adjusted tail probabilities of `mediate`'s BCa interval (its `BC.CI`).
fn bca_levels(d: &[f64], mean: f64, low: f64, high: f64) -> (f64, f64) {
    let normal = Normal::new(0.0, 1.0).expect("standard normal");
    let nd = d.len() as f64;
    let z0 = normal.inverse_cdf(d.iter().filter(|&&v| v < mean).count() as f64 / nd);
    let u: Vec<f64> = d.iter().map(|v| (nd - 1.0) * (mean - v)).collect();
    let top: f64 = u.iter().map(|v| v.powi(3)).sum();
    let under = 6.0 * u.iter().map(|v| v * v).sum::<f64>().powf(1.5);
    let acc = top / under;
    let adj = |q: f64| {
        let zq = normal.inverse_cdf(q);
        normal.cdf(z0 + (z0 + zq) / (1.0 - acc * (z0 + zq)))
    };
    (adj(low), adj(high))
}

fn median(d: &[f64]) -> f64 {
    let mut s: Vec<f64> = d.iter().copied().filter(|v| !v.is_nan()).collect();
    if s.is_empty() {
        return f64::NAN;
    }
    s.sort_by(|a, b| a.total_cmp(b));
    quantile_type7(&s, 0.5)
}

#[derive(Debug, Clone)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform integer in `0..n` (rejection sampling, no modulo bias).
    fn below(&mut self, n: usize) -> usize {
        let span = n as u64;
        let bound = u64::MAX - (u64::MAX % span);
        loop {
            let r = self.next_u64();
            if r < bound {
                return (r % span) as usize;
            }
        }
    }

    /// Uniform in the open interval (0, 1).
    fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    /// Standard normal (Box–Muller, one draw per call).
    fn normal(&mut self) -> f64 {
        let (u1, u2) = (self.uniform(), self.uniform());
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut rng = SplitMix64::new(7);
        let n = 80;
        let x: Vec<f64> = (0..n).map(|i| (i % 2) as f64).collect();
        let c: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
        let m: Vec<f64> = (0..n)
            .map(|i| 0.3 + 0.9 * x[i] + 0.2 * c[i] + rng.normal())
            .collect();
        let y: Vec<f64> = (0..n)
            .map(|i| 1.0 + 0.5 * x[i] + 0.7 * m[i] - 0.3 * c[i] + rng.normal())
            .collect();
        (y, m, x, c)
    }

    #[test]
    fn effects_decompose_and_are_deterministic() {
        let (y, m, x, c) = data();
        let o = MediationOptions {
            n_sims: 200,
            ..Default::default()
        };
        let r1 = mediate(&y, &m, &x, std::slice::from_ref(&c), &o).unwrap();
        let r2 = mediate(&y, &m, &x, &[c], &o).unwrap();
        // NaN statistics compare unequal, so compare the debug output.
        assert_eq!(format!("{r1:?}"), format!("{r2:?}"));
        assert!((r1.acme.estimate + r1.ade.estimate - r1.total.estimate).abs() < 1e-12);
        assert!((r1.acme.estimate - r1.a * r1.b).abs() < 1e-12);
        assert!(r1.acme.conf_low < r1.acme.estimate && r1.acme.estimate < r1.acme.conf_high);
        assert_eq!(r1.n_obs, 80);
        assert_eq!(r1.n_sims, 200);
    }

    #[test]
    fn non_finite_rows_dropped_from_both_models() {
        let (mut y, m, x, c) = data();
        let o = MediationOptions {
            method: MediationMethod::Sobel,
            ..Default::default()
        };
        let full = mediate(&y[1..], &m[1..], &x[1..], &[&c[1..]], &o).unwrap();
        y[0] = f64::NAN;
        let dropped = mediate(&y, &m, &x, &[&c[..]], &o).unwrap();
        assert_eq!(dropped.n_obs, 79);
        assert_eq!(full.acme, dropped.acme);
    }

    #[test]
    fn contrast_scales_effects() {
        let (y, m, x, c) = data();
        let base = MediationOptions {
            method: MediationMethod::Sobel,
            ..Default::default()
        };
        let r1 = mediate(&y, &m, &x, std::slice::from_ref(&c), &base).unwrap();
        let r2 = mediate(
            &y,
            &m,
            &x,
            &[c],
            &MediationOptions {
                treat_value: 3.0,
                control_value: 1.0,
                ..base
            },
        )
        .unwrap();
        assert!((r2.acme.estimate - 2.0 * r1.acme.estimate).abs() < 1e-12);
        assert!((r2.prop_mediated.estimate - r1.prop_mediated.estimate).abs() < 1e-12);
        assert!((r2.acme.p_value - r1.acme.p_value).abs() < 1e-12);
    }

    #[test]
    fn errors() {
        let (y, m, x, _) = data();
        let constant = vec![1.0; y.len()];
        assert!(matches!(
            mediate(
                &y,
                &m,
                &constant,
                &[] as &[Vec<f64>],
                &MediationOptions::default()
            ),
            Err(MediationError::NotIdentified(_))
        ));
        assert!(matches!(
            mediate(
                &y[..3],
                &m[..3],
                &x[..3],
                &[] as &[Vec<f64>],
                &MediationOptions::default()
            ),
            Err(MediationError::InsufficientData {
                usable: 3,
                needed: 4
            })
        ));
        assert!(matches!(
            mediate(
                &y,
                &m[1..],
                &x,
                &[] as &[Vec<f64>],
                &MediationOptions::default()
            ),
            Err(MediationError::DimensionMismatch { field: "m", .. })
        ));
        let bad = MediationOptions {
            treat_value: 0.0,
            ..Default::default()
        };
        assert!(matches!(
            mediate(&y, &m, &x, &[] as &[Vec<f64>], &bad),
            Err(MediationError::InvalidValue {
                field: "treat_value",
                ..
            })
        ));
    }

    #[test]
    fn option_and_input_errors() {
        let (y, m, x, c) = data();
        let none: &[Vec<f64>] = &[];
        for (o, field) in [
            (
                MediationOptions {
                    conf_level: 1.0,
                    ..Default::default()
                },
                "conf_level",
            ),
            (
                MediationOptions {
                    n_sims: 1,
                    ..Default::default()
                },
                "n_sims",
            ),
            (
                MediationOptions {
                    treat_value: f64::NAN,
                    ..Default::default()
                },
                "treat_value",
            ),
        ] {
            assert!(matches!(
                mediate(&y, &m, &x, none, &o),
                Err(MediationError::InvalidValue { field: f, .. }) if f == field
            ));
        }
        assert!(matches!(
            mediate(&y, &m, &x, &[&c[1..]], &MediationOptions::default()),
            Err(MediationError::DimensionMismatch {
                field: "covariates",
                ..
            })
        ));
        assert!(matches!(
            mediate(&y, &m, &x[1..], none, &MediationOptions::default()),
            Err(MediationError::DimensionMismatch { field: "x", .. })
        ));
        // A mediator collinear with the treatment leaves b unidentified.
        assert!(matches!(
            mediate(&y, &x, &x, none, &MediationOptions::default()),
            Err(MediationError::NotIdentified(_))
        ));
    }

    #[test]
    fn bootstrap_fails_when_resamples_are_degenerate() {
        // Treatment 1 in a single row: almost every resample misses it or
        // repeats it, so the treatment is constant and a is not identified.
        let n = 6;
        let x: Vec<f64> = (0..n).map(|i| if i == 0 { 1.0 } else { 0.0 }).collect();
        let m = [0.9, 0.1, 0.3, 0.2, 0.4, 0.0];
        let y = [2.0, 0.2, 0.5, 0.1, 0.9, 0.3];
        let failed = (0..20u64).any(|seed| {
            let o = MediationOptions {
                n_sims: 2,
                seed,
                ..Default::default()
            };
            matches!(
                mediate(&y, &m, &x, &[] as &[Vec<f64>], &o),
                Err(MediationError::BootstrapFailed { requested: 2, .. })
            )
        });
        assert!(failed);
    }

    #[test]
    fn quasi_bayesian_and_bca_run() {
        let (y, m, x, c) = data();
        for (method, boot_ci) in [
            (MediationMethod::QuasiBayesian, BootCi::Percentile),
            (MediationMethod::Bootstrap, BootCi::Bca),
        ] {
            let o = MediationOptions {
                method,
                boot_ci,
                n_sims: 100,
                ..Default::default()
            };
            let r = mediate(&y, &m, &x, std::slice::from_ref(&c), &o).unwrap();
            assert_eq!(r.n_sims, 100);
            assert_eq!(r.method, method);
            assert!(r.acme.conf_low < r.acme.conf_high);
            assert!(r.acme.statistic.is_nan());
        }
    }

    #[test]
    fn from_draws_edge_cases() {
        let e = MediationEffect::from_draws(1.0, &[f64::NAN, 2.0], 0.95, BootCi::Percentile);
        assert_eq!(e.estimate, 1.0);
        assert!(e.std_error.is_nan() && e.conf_low.is_nan());
        let e = MediationEffect::from_draws(0.0, &[-1.0, 1.0, 2.0], 0.95, BootCi::Percentile);
        assert_eq!(e.p_value, 1.0);
        let e = MediationEffect::from_draws(1.0, &[1.0, 2.0, 3.0, -1.0], 0.5, BootCi::Percentile);
        assert!((e.p_value - 0.5).abs() < 1e-12);
        assert!(median(&[f64::NAN]).is_nan());
        assert!((median(&[3.0, 1.0, 2.0]) - 2.0).abs() < 1e-12);
        assert!(quantile_type7(&[1.0, 2.0], f64::NAN).is_nan());
    }

    #[test]
    fn quantile_type7_matches_r() {
        // quantile(c(1, 2, 4, 8), c(0.025, 0.5, 0.975))
        let s = [1.0, 2.0, 4.0, 8.0];
        assert!((quantile_type7(&s, 0.025) - 1.075).abs() < 1e-12);
        assert!((quantile_type7(&s, 0.5) - 3.0).abs() < 1e-12);
        assert!((quantile_type7(&s, 0.975) - 7.7).abs() < 1e-12);
    }
}
