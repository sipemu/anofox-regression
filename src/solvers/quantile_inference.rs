//! Inference for quantile regression and the regression-quantile process.
//!
//! Standard errors follow `quantreg::summary.rq`:
//!
//! - [`QuantileSe::Iid`] (`se = "iid"`): `Σ = τ(1 − τ) · s² · (XᵀX)⁻¹`, with the
//!   sparsity `s = 1/f(F⁻¹(τ))` estimated, as quantreg, by the slope of a
//!   median regression of the ordered residuals `r₍pz+1₎ … r₍pz+h+1₎` (the
//!   `h + 1` smallest in absolute value after the `pz` exact zeros, then
//!   sorted) on `i / (n − k)`, `h = max(k + 1, ⌈n · h_HS⌉)`.
//! - [`QuantileSe::Nid`] (`se = "nid"`): the Hendricks–Koenker sandwich
//!   `Σ = τ(1 − τ) · (XᵀFX)⁻¹ XᵀX (XᵀFX)⁻¹` with
//!   `f_i = max(0, 2h / (x_iᵀ(β̂(τ + h) − β̂(τ − h)) − ε))`, `ε = √machine-ε`,
//!   `h = h_HS` halved until `τ ± h ∈ [0, 1]`.
//! - [`QuantileSe::Boot`] (`se = "boot"`, `bsmethod = "xy"`): `r` resamples of
//!   the `(x_i, y_i)` pairs drawn with the crate's seeded generator; `Σ` is the
//!   sample covariance (denominator `r − 1`) of the bootstrap coefficients.
//!   The draws differ from R's for the same seed.
//!
//! `h_HS` is the Hall–Sheather bandwidth of `quantreg::bandwidth.rq` with
//! `alpha = 0.05`:
//! `n^(−1/3) · z_{0.975}^(2/3) · (1.5 φ(Φ⁻¹(τ))² / (2 Φ⁻¹(τ)² + 1))^(1/3)`.
//!
//! As `summary.rq`, `statistic = estimate / std_error` and the p-value is
//! two-sided from Student's t with `n − k` degrees of freedom (`k` counts the
//! intercept). Confidence limits are `estimate ± t_{n−k, 1−(1−level)/2} ·
//! std_error`. Rank-inversion intervals (`se = "rank"`) are not implemented.

use crate::core::RegressionResult;
use crate::solvers::path::SplitMix64;
use crate::solvers::traits::{FittedRegressor, RegressionError, Regressor};
use crate::solvers::QuantileRegressor;
use faer::linalg::solvers::Solve;
use faer::{Col, Mat};
use statrs::distribution::{Continuous, ContinuousCDF, Normal, StudentsT};

/// Standard-error type for quantile regression (`quantreg::summary.rq`'s `se`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantileSe {
    /// Errors iid: sparsity from the ordered residuals (`se = "iid"`).
    Iid,
    /// Locally iid, Hendricks–Koenker sandwich with Hall–Sheather
    /// bandwidth (`se = "nid"`).
    Nid,
    /// `(x, y)`-pair bootstrap with `r` replications and a fixed `seed`
    /// (`se = "boot"`, `bsmethod = "xy"`).
    Boot {
        /// Seed of the resampling generator.
        seed: u64,
        /// Number of bootstrap replications (≥ 2).
        r: usize,
    },
}

/// Coefficient table of a quantile regression fit. Coefficient vectors start
/// with the intercept when the model has one.
#[derive(Debug, Clone)]
pub struct QuantileInference {
    /// Quantile `τ`.
    pub tau: f64,
    /// Standard-error type (`None` when no inference was requested, see
    /// [`quantile_process`]).
    pub se_type: Option<QuantileSe>,
    /// Whether the first entry is the intercept.
    pub has_intercept: bool,
    /// Coefficient estimates.
    pub estimate: Vec<f64>,
    /// Standard errors.
    pub std_error: Vec<f64>,
    /// `estimate / std_error`.
    pub statistic: Vec<f64>,
    /// Two-sided p-values (Student's t, `df_residual` degrees of freedom).
    pub p_value: Vec<f64>,
    /// Lower confidence limits.
    pub conf_low: Vec<f64>,
    /// Upper confidence limits.
    pub conf_high: Vec<f64>,
    /// Confidence level of `conf_low` / `conf_high`.
    pub conf_level: f64,
    /// Residual degrees of freedom `n − k`.
    pub df_residual: f64,
    /// Bandwidth used (`h` of `nid`; the Hall–Sheather bandwidth for `iid`;
    /// `NaN` for the bootstrap).
    pub bandwidth: f64,
    /// Estimated sparsity `1/f(F⁻¹(τ))` (`iid` only, otherwise `NaN`).
    pub sparsity: f64,
    /// Coefficient covariance matrix (`k × k`, `NaN` without inference).
    pub covariance: Mat<f64>,
    /// Bootstrap replications that were used (0 unless `Boot`).
    pub n_boot: usize,
}

/// Hall–Sheather bandwidth of `quantreg::bandwidth.rq(tau, n, hs = TRUE)`
/// (`alpha = 0.05`).
pub fn hall_sheather_bandwidth(tau: f64, n: usize) -> f64 {
    let normal = Normal::new(0.0, 1.0).expect("standard normal");
    let x0 = normal.inverse_cdf(tau);
    let f0 = normal.pdf(x0);
    let z = normal.inverse_cdf(1.0 - 0.05 / 2.0);
    (n as f64).powf(-1.0 / 3.0)
        * z.powf(2.0 / 3.0)
        * ((1.5 * f0 * f0) / (2.0 * x0 * x0 + 1.0)).powf(1.0 / 3.0)
}

/// Fit a quantile regression at `tau` and compute its coefficient table.
pub fn quantile_inference(
    x: &Mat<f64>,
    y: &Col<f64>,
    tau: f64,
    with_intercept: bool,
    se: QuantileSe,
    level: f64,
) -> Result<QuantileInference, RegressionError> {
    let fit = fit_at(x, y, tau, with_intercept)?;
    inference_for_fit(x, y, tau, with_intercept, fit.result(), se, level)
}

/// The regression-quantile process: one fit per `τ` in `taus`, with standard
/// errors and confidence limits when `se` is given (all `NaN` otherwise).
/// This is the table behind `plot(summary(rq(y ~ x, tau = taus)))`.
pub fn quantile_process(
    x: &Mat<f64>,
    y: &Col<f64>,
    taus: &[f64],
    with_intercept: bool,
    se: Option<QuantileSe>,
    level: f64,
) -> Result<Vec<QuantileInference>, RegressionError> {
    if taus.is_empty() {
        return Err(RegressionError::NumericalError(
            "taus must not be empty".to_string(),
        ));
    }
    taus.iter()
        .map(|&tau| {
            let fit = fit_at(x, y, tau, with_intercept)?;
            match se {
                Some(se) => inference_for_fit(x, y, tau, with_intercept, fit.result(), se, level),
                None => Ok(point_only(
                    tau,
                    with_intercept,
                    fit.result(),
                    x.nrows(),
                    level,
                )),
            }
        })
        .collect()
}

fn fit_at(
    x: &Mat<f64>,
    y: &Col<f64>,
    tau: f64,
    with_intercept: bool,
) -> Result<crate::solvers::FittedQuantile, RegressionError> {
    QuantileRegressor::builder()
        .tau(tau)
        .with_intercept(with_intercept)
        .build()
        .fit(x, y)
}

fn coefs(result: &RegressionResult) -> Vec<f64> {
    result
        .intercept
        .into_iter()
        .chain(result.coefficients.iter().copied())
        .collect()
}

fn point_only(
    tau: f64,
    with_intercept: bool,
    result: &RegressionResult,
    n: usize,
    level: f64,
) -> QuantileInference {
    let estimate = coefs(result);
    let k = estimate.len();
    let nan = vec![f64::NAN; k];
    QuantileInference {
        tau,
        se_type: None,
        has_intercept: with_intercept,
        estimate,
        std_error: nan.clone(),
        statistic: nan.clone(),
        p_value: nan.clone(),
        conf_low: nan.clone(),
        conf_high: nan,
        conf_level: level,
        df_residual: n as f64 - k as f64,
        bandwidth: f64::NAN,
        sparsity: f64::NAN,
        covariance: Mat::from_fn(k, k, |_, _| f64::NAN),
        n_boot: 0,
    }
}

fn design(x: &Mat<f64>, with_intercept: bool) -> Mat<f64> {
    let off = usize::from(with_intercept);
    Mat::from_fn(x.nrows(), x.ncols() + off, |i, j| {
        if j < off {
            1.0
        } else {
            x[(i, j - off)]
        }
    })
}

fn spd_inverse(a: &Mat<f64>) -> Result<Mat<f64>, RegressionError> {
    let k = a.nrows();
    let llt = a
        .llt(faer::Side::Lower)
        .map_err(|_| RegressionError::SingularMatrix)?;
    Ok(llt.solve(Mat::<f64>::identity(k, k)))
}

/// Coefficient table of an existing fit (`result`) on `(x, y)`.
pub(crate) fn inference_for_fit(
    x: &Mat<f64>,
    y: &Col<f64>,
    tau: f64,
    with_intercept: bool,
    result: &RegressionResult,
    se: QuantileSe,
    level: f64,
) -> Result<QuantileInference, RegressionError> {
    if x.nrows() != y.nrows() || result.coefficients.nrows() != x.ncols() {
        return Err(RegressionError::DimensionMismatch {
            x_rows: x.nrows(),
            y_len: y.nrows(),
        });
    }
    if !(level > 0.0 && level < 1.0) {
        return Err(RegressionError::NumericalError(format!(
            "level must be in (0, 1), got {level}"
        )));
    }
    let n = x.nrows();
    let xa = design(x, with_intercept);
    let k = xa.ncols();
    if n <= k {
        return Err(RegressionError::InsufficientObservations {
            needed: k + 1,
            got: n,
        });
    }
    let estimate = coefs(result);
    let mut bandwidth = f64::NAN;
    let mut sparsity = f64::NAN;
    let mut n_boot = 0;
    let tt = tau * (1.0 - tau);
    let covariance = match se {
        QuantileSe::Iid => {
            let xtx_inv = spd_inverse(&(xa.transpose() * &xa))?;
            let resid = Col::from_fn(n, |i| {
                y[i] - (0..k).map(|j| xa[(i, j)] * estimate[j]).sum::<f64>()
            });
            let (s, bw) = iid_sparsity(&resid, tau, k)?;
            sparsity = s;
            bandwidth = bw;
            Mat::from_fn(k, k, |a, b| s * s * tt * xtx_inv[(a, b)])
        }
        QuantileSe::Nid => {
            let mut h = hall_sheather_bandwidth(tau, n);
            while tau - h < 0.0 || tau + h > 1.0 {
                h /= 2.0;
            }
            bandwidth = h;
            let hi = coefs(fit_at(x, y, tau + h, with_intercept)?.result());
            let lo = coefs(fit_at(x, y, tau - h, with_intercept)?.result());
            let eps = f64::EPSILON.sqrt();
            let f: Vec<f64> = (0..n)
                .map(|i| {
                    let dy: f64 = (0..k).map(|j| xa[(i, j)] * (hi[j] - lo[j])).sum();
                    (2.0 * h / (dy - eps)).max(0.0)
                })
                .collect();
            let xfx = Mat::from_fn(k, k, |a, b| {
                (0..n).map(|i| f[i] * xa[(i, a)] * xa[(i, b)]).sum()
            });
            let fxx_inv = spd_inverse(&xfx)?;
            let xtx = xa.transpose() * &xa;
            let m = &fxx_inv * &xtx * &fxx_inv;
            Mat::from_fn(k, k, |a, b| tt * m[(a, b)])
        }
        QuantileSe::Boot { seed, r } => {
            if r < 2 {
                return Err(RegressionError::NumericalError(
                    "bootstrap needs r >= 2 replications".to_string(),
                ));
            }
            let mut rng = SplitMix64::new(seed);
            let mut draws: Vec<Vec<f64>> = Vec::with_capacity(r);
            for _ in 0..r {
                let idx: Vec<usize> = (0..n).map(|_| rng.below(n as u64) as usize).collect();
                let xb = Mat::from_fn(n, x.ncols(), |i, j| x[(idx[i], j)]);
                let yb = Col::from_fn(n, |i| y[idx[i]]);
                // A degenerate resample (e.g. a constant column) is skipped.
                if let Ok(fb) = fit_at(&xb, &yb, tau, with_intercept) {
                    let b = coefs(fb.result());
                    if b.iter().all(|v| v.is_finite()) {
                        draws.push(b);
                    }
                }
            }
            n_boot = draws.len();
            if n_boot < 2 {
                return Err(RegressionError::NumericalError(
                    "fewer than 2 bootstrap replications could be fitted".to_string(),
                ));
            }
            let nb = n_boot as f64;
            let mean: Vec<f64> = (0..k)
                .map(|j| draws.iter().map(|d| d[j]).sum::<f64>() / nb)
                .collect();
            Mat::from_fn(k, k, |a, b| {
                draws
                    .iter()
                    .map(|d| (d[a] - mean[a]) * (d[b] - mean[b]))
                    .sum::<f64>()
                    / (nb - 1.0)
            })
        }
    };

    let df = (n - k) as f64;
    let t = StudentsT::new(0.0, 1.0, df)
        .map_err(|e| RegressionError::NumericalError(format!("t distribution: {e}")))?;
    let q = t.inverse_cdf(1.0 - (1.0 - level) / 2.0);
    let std_error: Vec<f64> = (0..k)
        .map(|j| {
            let v = covariance[(j, j)];
            if v >= 0.0 {
                v.sqrt()
            } else {
                f64::NAN
            }
        })
        .collect();
    let statistic: Vec<f64> = (0..k).map(|j| estimate[j] / std_error[j]).collect();
    let p_value = statistic
        .iter()
        .map(|s| {
            if s.is_finite() {
                2.0 * t.sf(s.abs())
            } else {
                f64::NAN
            }
        })
        .collect();
    let conf_low = (0..k).map(|j| estimate[j] - q * std_error[j]).collect();
    let conf_high = (0..k).map(|j| estimate[j] + q * std_error[j]).collect();
    Ok(QuantileInference {
        tau,
        se_type: Some(se),
        has_intercept: with_intercept,
        estimate,
        std_error,
        statistic,
        p_value,
        conf_low,
        conf_high,
        conf_level: level,
        df_residual: df,
        bandwidth,
        sparsity,
        covariance,
        n_boot,
    })
}

/// quantreg's iid sparsity estimate; returns `(sparsity, bandwidth)`.
fn iid_sparsity(resid: &Col<f64>, tau: f64, k: usize) -> Result<(f64, f64), RegressionError> {
    let n = resid.nrows();
    let eps = f64::EPSILON.sqrt();
    let pz = resid.iter().filter(|r| r.abs() < eps).count();
    let bw = hall_sheather_bandwidth(tau, n);
    let h = (k + 1).max((n as f64 * bw).ceil() as usize);
    // ir = (pz + 1):(h + pz + 1), 1-based.
    if h + pz + 1 > n {
        return Err(RegressionError::InsufficientObservations {
            needed: h + pz + 1,
            got: n,
        });
    }
    let mut by_abs: Vec<usize> = (0..n).collect();
    by_abs.sort_by(|&a, &b| resid[a].abs().total_cmp(&resid[b].abs())); // stable, as R's order()
    let ir: Vec<usize> = (pz + 1..=h + pz + 1).collect();
    let mut ord: Vec<f64> = ir.iter().map(|&i| resid[by_abs[i - 1]]).collect();
    ord.sort_by(|a, b| a.total_cmp(b));
    let m = ir.len();
    let xt = Mat::from_fn(m, 1, |i, _| ir[i] as f64 / (n - k) as f64);
    let yo = Col::from_fn(m, |i| ord[i]);
    let lad = fit_at(&xt, &yo, 0.5, true)?;
    Ok((lad.result().coefficients[0], bw))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bandwidth_matches_quantreg() {
        // quantreg::bandwidth.rq(0.5, 100) = 0.2093160469 (Hall-Sheather)
        let h = hall_sheather_bandwidth(0.5, 100);
        assert!((h - 0.209_316_046_9).abs() < 1e-9, "{h}");
    }
}
