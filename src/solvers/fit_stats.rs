//! Shared goodness-of-fit statistics for the linear solvers.
//!
//! One implementation of R², adjusted R², MSE, F and the Gaussian
//! log-likelihood / information criteria, so the linear solvers cannot drift
//! apart. The conventions follow R's `summary.lm` and `logLik.lm`:
//!
//! * **With an intercept** the total sum of squares is centred,
//!   `TSS = Σ wᵢ (yᵢ − ȳ_w)²`, and `df_total = n − 1`.
//! * **Without an intercept** it is uncentred, `TSS = Σ wᵢ yᵢ²`, and
//!   `df_total = n` (as `summary.lm` and statsmodels do for models without a
//!   constant).
//! * `F = ((TSS − RSS) / df_model) / (RSS / df_resid)` with
//!   `df_model = n_params − 1` (intercept) or `n_params` (no intercept).
//! * The log-likelihood is the Gaussian log-likelihood at the **ML** variance
//!   `σ̂² = RSS / n`:
//!   `ℓ = ½ (Σ ln wᵢ − n (ln 2π + 1 − ln n + ln RSS))`
//!   (observations with zero weight are excluded, as `logLik.lm` does), and
//!   the information criteria count the residual variance as a parameter:
//!   `k = n_params + 1`, `AIC = 2k − 2ℓ`, `BIC = k ln n − 2ℓ`,
//!   `AICc = AIC + 2k(k + 1) / (n − k − 1)`.

use faer::Col;
use statrs::distribution::{ContinuousCDF, FisherSnedecor};

/// Goodness-of-fit statistics of a linear fit.
#[derive(Debug, Clone, Copy)]
pub(crate) struct LinearFitStats {
    pub r_squared: f64,
    pub adj_r_squared: f64,
    pub mse: f64,
    pub rmse: f64,
    pub f_statistic: f64,
    pub f_pvalue: f64,
    pub log_likelihood: f64,
    pub aic: f64,
    pub aicc: f64,
    pub bic: f64,
}

impl LinearFitStats {
    /// Copy the statistics into a result.
    pub fn apply(&self, result: &mut crate::core::RegressionResult) {
        result.r_squared = self.r_squared;
        result.adj_r_squared = self.adj_r_squared;
        result.mse = self.mse;
        result.rmse = self.rmse;
        result.f_statistic = self.f_statistic;
        result.f_pvalue = self.f_pvalue;
        result.log_likelihood = self.log_likelihood;
        result.aic = self.aic;
        result.aicc = self.aicc;
        result.bic = self.bic;
    }
}

/// Compute the fit statistics.
///
/// * `residuals` – `y − ŷ` for every row.
/// * `weights` – optional observation weights (WLS); `None` means unit weights.
/// * `has_intercept` – selects the centred / uncentred TSS.
/// * `n_params` – estimated mean parameters including the intercept.
/// * `clamp_r_squared` – clamp R² to `[0, 1]`. Least-squares fits never need
///   it; estimators that do not minimise the RSS (penalised, robust) keep the
///   historical clamp.
pub(crate) fn linear_fit_stats(
    y: &Col<f64>,
    residuals: &Col<f64>,
    weights: Option<&Col<f64>>,
    has_intercept: bool,
    n_params: usize,
    clamp_r_squared: bool,
) -> LinearFitStats {
    let n_rows = y.nrows();
    let w = |i: usize| weights.map_or(1.0, |w| w[i]);

    // Observations with zero weight carry no information (lm drops them).
    let n = (0..n_rows).filter(|&i| w(i) > 0.0).count();
    let sum_w: f64 = (0..n_rows).map(w).sum();

    let tss = if has_intercept {
        let y_mean = if sum_w > 0.0 {
            (0..n_rows).map(|i| w(i) * y[i]).sum::<f64>() / sum_w
        } else {
            f64::NAN
        };
        (0..n_rows).map(|i| w(i) * (y[i] - y_mean).powi(2)).sum()
    } else {
        (0..n_rows).map(|i| w(i) * y[i] * y[i]).sum::<f64>()
    };
    let rss: f64 = (0..n_rows).map(|i| w(i) * residuals[i].powi(2)).sum();

    let r_squared = if tss > 0.0 {
        let r2 = 1.0 - rss / tss;
        if clamp_r_squared {
            r2.clamp(0.0, 1.0)
        } else {
            r2
        }
    } else if rss < 1e-10 {
        1.0
    } else {
        0.0
    };

    let df_total = if has_intercept {
        n as f64 - 1.0
    } else {
        n as f64
    };
    let df_resid = n as f64 - n_params as f64;
    let adj_r_squared = if df_resid > 0.0 && df_total > 0.0 {
        1.0 - (1.0 - r_squared) * df_total / df_resid
    } else {
        f64::NAN
    };

    let mse = if df_resid > 0.0 {
        rss / df_resid
    } else {
        f64::NAN
    };
    let rmse = mse.sqrt();

    let ess = tss - rss;
    let df_model = n_params.saturating_sub(usize::from(has_intercept)) as f64;
    let f_statistic = if df_model > 0.0 && df_resid > 0.0 && mse > 0.0 {
        (ess / df_model) / mse
    } else {
        f64::NAN
    };
    let f_pvalue = if f_statistic.is_finite() && df_model > 0.0 && df_resid > 0.0 {
        FisherSnedecor::new(df_model, df_resid)
            .ok()
            .map_or(f64::NAN, |d| 1.0 - d.cdf(f_statistic))
    } else {
        f64::NAN
    };

    let nf = n as f64;
    let log_likelihood = if rss > 0.0 && n > 0 {
        let sum_log_w: f64 = (0..n_rows).filter(|&i| w(i) > 0.0).map(|i| w(i).ln()).sum();
        0.5 * (sum_log_w - nf * ((2.0 * std::f64::consts::PI).ln() + 1.0 - nf.ln() + rss.ln()))
    } else {
        f64::NAN
    };

    let (aic, aicc, bic) = information_criteria(log_likelihood, n_params as f64 + 1.0, nf);

    LinearFitStats {
        r_squared,
        adj_r_squared,
        mse,
        rmse,
        f_statistic,
        f_pvalue,
        log_likelihood,
        aic,
        aicc,
        bic,
    }
}

/// `(AIC, AICc, BIC)` for a log-likelihood with `k` parameters and `n` rows.
pub(crate) fn information_criteria(log_likelihood: f64, k: f64, n: f64) -> (f64, f64, f64) {
    if !log_likelihood.is_finite() {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    let aic = 2.0 * k - 2.0 * log_likelihood;
    let aicc = if n - k - 1.0 > 0.0 {
        aic + 2.0 * k * (k + 1.0) / (n - k - 1.0)
    } else {
        f64::NAN
    };
    let bic = k * n.ln() - 2.0 * log_likelihood;
    (aic, aicc, bic)
}

/// Poisson log-likelihood `Σ (yᵢ ln μᵢ − μᵢ − ln yᵢ!)`, as R's `logLik(glm)`
/// (equal to `−deviance/2` plus the saturated term).
pub(crate) fn poisson_log_likelihood(y: &[f64], mu: &[f64]) -> f64 {
    use statrs::function::gamma::ln_gamma;
    y.iter()
        .zip(mu)
        .map(|(&yi, &mi)| {
            let a = if yi > 0.0 { yi * mi.ln() } else { 0.0 };
            a - mi - ln_gamma(yi + 1.0)
        })
        .sum()
}

/// Negative-binomial (NB2, size `θ`) log-likelihood
/// `Σ log dnbinom(yᵢ, size = θ, mu = μᵢ)`, as `MASS::glm.nb`.
pub(crate) fn negbin_log_likelihood(y: &[f64], mu: &[f64], theta: f64) -> f64 {
    use statrs::function::gamma::ln_gamma;
    y.iter()
        .zip(mu)
        .map(|(&yi, &mi)| {
            let a = ln_gamma(yi + theta) - ln_gamma(theta) - ln_gamma(yi + 1.0)
                + theta * (theta / (theta + mi)).ln();
            if yi > 0.0 {
                a + yi * (mi / (theta + mi)).ln()
            } else {
                a
            }
        })
        .sum()
}

/// Saturated Poisson log-likelihood `Σ log dpois(yᵢ, yᵢ)`.
pub(crate) fn poisson_saturated_log_likelihood(y: &[f64]) -> f64 {
    poisson_log_likelihood(y, &y.iter().map(|&v| v.max(0.0)).collect::<Vec<_>>())
}

/// Saturated binomial log-likelihood for responses that are proportions
/// `yᵢ` of `mᵢ` trials: `Σ log dbinom(mᵢyᵢ, mᵢ, yᵢ)`; zero for 0/1 data.
pub(crate) fn binomial_saturated_log_likelihood(y: &[f64], trials: Option<&[f64]>) -> f64 {
    use statrs::function::gamma::ln_gamma;
    y.iter()
        .enumerate()
        .map(|(i, &yi)| {
            let m = trials.map_or(1.0, |t| t[i]);
            let k = (m * yi).round();
            let mut v = ln_gamma(m + 1.0) - ln_gamma(k + 1.0) - ln_gamma(m - k + 1.0);
            if yi > 0.0 {
                v += k * yi.ln();
            }
            if yi < 1.0 {
                v += (m - k) * (1.0 - yi).ln();
            }
            v
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uncentred_tss_without_intercept() {
        // y = 2x exactly on x = 1..4 -> R² = 1 either way.
        let y = Col::from_fn(4, |i| 2.0 * (i + 1) as f64);
        let r = Col::from_fn(4, |_| 0.0);
        let s = linear_fit_stats(&y, &r, None, false, 1, false);
        assert_eq!(s.r_squared, 1.0);
        assert!(s.log_likelihood.is_nan());
    }

    #[test]
    fn loglik_uses_ml_variance() {
        let y = Col::from_fn(5, |i| i as f64);
        let r = Col::from_fn(5, |i| if i % 2 == 0 { 1.0 } else { -1.0 });
        let s = linear_fit_stats(&y, &r, None, true, 2, false);
        // RSS = 5, n = 5 -> sigma2_ml = 1 -> ll = -5/2 (ln 2π + 1)
        let expected = -2.5 * ((2.0 * std::f64::consts::PI).ln() + 1.0);
        assert!((s.log_likelihood - expected).abs() < 1e-12);
        assert!((s.aic - (6.0 - 2.0 * expected)).abs() < 1e-12);
    }
}
