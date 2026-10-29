//! Transforming the response and back (RMS Ch 16): the Box–Cox profile
//! likelihood of the power `λ` (`MASS::boxcox`), additive regression with
//! optimal transformations of both sides ([`areg`], `Hmisc::areg`), and
//! Duan's smearing estimator of the mean or quantiles of `Y` on the original
//! scale from a model of `g(Y)` (`Hmisc::smearingEst`).
//!
//! ```rust,ignore
//! use anofox_regression::transform::{box_cox, BoxCoxOptions, smearing_estimate, Smearing};
//!
//! let prof = box_cox(&x, &y, &BoxCoxOptions::default())?;
//! println!("lambda = {} ({:?}, {:?})", prof.lambda_hat, prof.ci_lower, prof.ci_upper);
//!
//! // a model of sqrt(y): smear the residuals back through z ↦ z²
//! let mean_y = smearing_estimate(&lp, &residuals, |z| z * z, Smearing::Mean);
//! ```

mod areg;

pub use areg::{areg, Areg, AregBootstrap, AregOptions, AregType};

use crate::diagnostics::accuracy::quantile7;
use crate::solvers::RegressionError;
use faer::{Col, Mat};
use statrs::distribution::{ChiSquared, ContinuousCDF};

/// The Box–Cox transform `(y^λ − 1)/λ`, `log y` at `λ = 0`.
pub fn box_cox_transform(y: f64, lambda: f64) -> f64 {
    if lambda == 0.0 {
        y.ln()
    } else {
        (y.powf(lambda) - 1.0) / lambda
    }
}

/// The inverse Box–Cox transform `(λz + 1)^{1/λ}`, `exp z` at `λ = 0`.
pub fn box_cox_inverse(z: f64, lambda: f64) -> f64 {
    if lambda == 0.0 {
        z.exp()
    } else {
        (lambda * z + 1.0).powf(1.0 / lambda)
    }
}

/// Options of [`box_cox`].
#[derive(Debug, Clone, PartialEq)]
pub struct BoxCoxOptions {
    /// Grid of `λ` at which the profile is reported (MASS default
    /// `seq(-2, 2, 1/10)`). `λ̂` and the interval are searched within its
    /// range.
    pub lambdas: Vec<f64>,
    /// Below this `|λ|` the transform uses MASS's series for `log y`.
    pub eps: f64,
    /// Prepend an intercept column to `x`.
    pub with_intercept: bool,
    /// Level of the likelihood-ratio interval of `λ`.
    pub confidence_level: f64,
}

impl Default for BoxCoxOptions {
    fn default() -> Self {
        Self {
            lambdas: (0..=40).map(|k| -2.0 + k as f64 / 10.0).collect(),
            eps: 1.0 / 50.0,
            with_intercept: true,
            confidence_level: 0.95,
        }
    }
}

/// Result of [`box_cox`].
#[derive(Debug, Clone, PartialEq)]
pub struct BoxCoxProfile {
    /// The grid of `λ`.
    pub lambdas: Vec<f64>,
    /// Profile log-likelihood at each grid value, `−n/2 log RSS(λ)` of the
    /// geometric-mean-scaled transform (MASS's scale, no constant).
    pub log_likelihood: Vec<f64>,
    /// The maximising `λ` within the grid range.
    pub lambda_hat: f64,
    /// Profile log-likelihood at `λ̂`.
    pub max_log_likelihood: f64,
    /// Lower limit `{λ : ℓ(λ) ≥ ℓ(λ̂) − χ²₁/2}`; `None` if the profile stays
    /// above the cut down to the grid's lower end.
    pub ci_lower: Option<f64>,
    /// Upper limit, `None` if above the cut up to the grid's upper end.
    pub ci_upper: Option<f64>,
}

/// The data of a Box–Cox profile: `y` divided by its geometric mean, and
/// the column space of the design.
struct BoxCoxProblem {
    q: Mat<f64>,
    y: Vec<f64>,
    eps: f64,
}

impl BoxCoxProblem {
    fn new(x: &Mat<f64>, y: &Col<f64>, options: &BoxCoxOptions) -> Result<Self, RegressionError> {
        let n = y.nrows();
        if x.nrows() != n {
            return Err(RegressionError::DimensionMismatch {
                x_rows: x.nrows(),
                y_len: n,
            });
        }
        if y.iter().any(|v| !(*v > 0.0 && v.is_finite())) {
            return Err(RegressionError::NumericalError(
                "the Box–Cox response must be positive".to_string(),
            ));
        }
        let design = if options.with_intercept {
            Mat::from_fn(
                n,
                x.ncols() + 1,
                |i, j| {
                    if j == 0 {
                        1.0
                    } else {
                        x[(i, j - 1)]
                    }
                },
            )
        } else {
            x.to_owned()
        };
        if n <= design.ncols() {
            return Err(RegressionError::InsufficientObservations {
                needed: design.ncols() + 1,
                got: n,
            });
        }
        let log_gm = y.iter().map(|v| v.ln()).sum::<f64>() / n as f64;
        Ok(Self {
            q: design.qr().compute_thin_Q(),
            y: y.iter().map(|v| v / log_gm.exp()).collect(),
            eps: options.eps,
        })
    }

    /// `−n/2 log RSS(λ)` as `MASS::boxcox.default`.
    fn log_likelihood(&self, lambda: f64) -> f64 {
        let n = self.y.len();
        let yt = Col::from_fn(n, |i| {
            let y = self.y[i];
            if lambda.abs() > self.eps {
                (y.powf(lambda) - 1.0) / lambda
            } else {
                let ly = y.ln();
                let a = lambda * ly;
                ly * (1.0 + a / 2.0 * (1.0 + a / 3.0 * (1.0 + a / 4.0)))
            }
        });
        let fit = &self.q * (self.q.transpose() * &yt);
        let rss: f64 = (0..n).map(|i| (yt[i] - fit[i]).powi(2)).sum();
        -(n as f64) / 2.0 * rss.ln()
    }
}

/// Box–Cox profile log-likelihood of `λ` for the linear model of `y`
/// (positive) on `x`, as `MASS::boxcox(lm(y ~ x))`, with the maximising
/// `λ̂` and its likelihood-ratio interval.
pub fn box_cox(
    x: &Mat<f64>,
    y: &Col<f64>,
    options: &BoxCoxOptions,
) -> Result<BoxCoxProfile, RegressionError> {
    if options.lambdas.is_empty() || options.lambdas.iter().any(|l| !l.is_finite()) {
        return Err(RegressionError::NumericalError(
            "the λ grid must be non-empty and finite".to_string(),
        ));
    }
    if !(options.confidence_level > 0.0 && options.confidence_level < 1.0) {
        return Err(RegressionError::NumericalError(
            "confidence_level must be in (0, 1)".to_string(),
        ));
    }
    let problem = BoxCoxProblem::new(x, y, options)?;
    let ll = |l: f64| problem.log_likelihood(l);
    let log_likelihood: Vec<f64> = options.lambdas.iter().map(|&l| ll(l)).collect();
    let lo = options
        .lambdas
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let hi = options
        .lambdas
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let lambda_hat = maximise(&ll, lo, hi, &options.lambdas, &log_likelihood);
    let max_ll = ll(lambda_hat);
    let chi = ChiSquared::new(1.0).expect("one degree of freedom");
    let cut = max_ll - chi.inverse_cdf(options.confidence_level) / 2.0;
    let crossing = |end: f64| (ll(end) < cut).then(|| bisect(&|l| ll(l) - cut, lambda_hat, end));
    Ok(BoxCoxProfile {
        lambdas: options.lambdas.clone(),
        log_likelihood,
        lambda_hat,
        max_log_likelihood: max_ll,
        ci_lower: crossing(lo),
        ci_upper: crossing(hi),
    })
}

/// Golden-section maximum of `f` on the grid cell around the best grid
/// value (clamped to `[lo, hi]`).
fn maximise(f: &dyn Fn(f64) -> f64, lo: f64, hi: f64, grid: &[f64], values: &[f64]) -> f64 {
    let best = (0..grid.len())
        .max_by(|&a, &b| values[a].total_cmp(&values[b]))
        .expect("non-empty grid");
    let mut sorted = grid.to_vec();
    sorted.sort_by(f64::total_cmp);
    let k = sorted.partition_point(|g| *g < grid[best]);
    let mut a = if k > 0 { sorted[k - 1] } else { lo };
    let mut b = sorted.get(k + 1).copied().unwrap_or(hi);
    const INV_PHI: f64 = 0.618_033_988_749_895;
    let (mut c, mut d) = (b - INV_PHI * (b - a), a + INV_PHI * (b - a));
    let (mut fc, mut fd) = (f(c), f(d));
    while (b - a).abs() > 1e-12 * (1.0 + a.abs()) {
        if fc > fd {
            b = d;
            (d, fd) = (c, fc);
            c = b - INV_PHI * (b - a);
            fc = f(c);
        } else {
            a = c;
            (c, fc) = (d, fd);
            d = a + INV_PHI * (b - a);
            fd = f(d);
        }
    }
    0.5 * (a + b)
}

/// Root of `g` between `a` (where `g > 0`) and `b` (where `g < 0`).
fn bisect(g: &dyn Fn(f64) -> f64, mut a: f64, mut b: f64) -> f64 {
    for _ in 0..200 {
        let m = 0.5 * (a + b);
        if g(m) > 0.0 {
            a = m;
        } else {
            b = m;
        }
        if (b - a).abs() <= 1e-13 * (1.0 + a.abs()) {
            break;
        }
    }
    0.5 * (a + b)
}

/// The statistic of [`smearing_estimate`] (`smearingEst(statistic = )`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Smearing {
    /// `mean_i g⁻¹(lp + r_i)`: Duan's smearing estimate of `E[Y | x]`.
    Mean,
    /// `g⁻¹(lp + median r)`.
    Median,
    /// `g⁻¹(lp + q-quantile of r)` (R type-7 quantile).
    Quantile(f64),
    /// The naive back-transform `g⁻¹(lp)`.
    Fitted,
}

/// Duan's smearing estimator: estimates of the mean or a quantile of `Y` on
/// the original scale at linear predictors `lp` of a model of `g(Y)`, from
/// its `residuals` and the inverse transform `inverse = g⁻¹` (monotone
/// increasing for the quantiles). Errors on empty residuals or a quantile
/// outside `[0, 1]`.
pub fn smearing_estimate(
    lp: &[f64],
    residuals: &[f64],
    inverse: impl Fn(f64) -> f64,
    statistic: Smearing,
) -> Result<Vec<f64>, RegressionError> {
    if residuals.is_empty() {
        return Err(RegressionError::NumericalError(
            "smearing needs residuals".to_string(),
        ));
    }
    if let Smearing::Quantile(q) = statistic {
        if !(0.0..=1.0).contains(&q) {
            return Err(RegressionError::NumericalError(
                "the smearing quantile must be in [0, 1]".to_string(),
            ));
        }
    }
    let shift = |q: f64| {
        let mut r = residuals.to_vec();
        r.sort_by(f64::total_cmp);
        quantile7(&r, q)
    };
    Ok(match statistic {
        Smearing::Mean => lp
            .iter()
            .map(|&l| {
                residuals.iter().map(|&r| inverse(l + r)).sum::<f64>() / residuals.len() as f64
            })
            .collect(),
        Smearing::Median | Smearing::Quantile(_) => {
            let q = match statistic {
                Smearing::Quantile(q) => q,
                _ => 0.5,
            };
            let s = shift(q);
            lp.iter().map(|&l| inverse(l + s)).collect()
        }
        Smearing::Fitted => lp.iter().map(|&l| inverse(l)).collect(),
    })
}
