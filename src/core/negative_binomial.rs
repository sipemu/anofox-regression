//! Negative Binomial family for overdispersed count data regression.
//!
//! This module provides the negative binomial family for count data that
//! exhibits overdispersion (variance > mean).
//!
//! # Example
//!
//! ```ignore
//! use anofox_regression::NegativeBinomialFamily;
//!
//! // Negative binomial with known theta
//! let nb = NegativeBinomialFamily::new(2.0);
//!
//! // Variance at mu = 5: V(5) = 5 + 5²/2 = 17.5
//! let var = nb.variance(5.0);
//! ```

use super::family::GlmFamily;
use statrs::function::gamma::digamma;

/// Negative Binomial family for overdispersed count data.
///
/// The negative binomial distribution is useful for count data where the
/// variance exceeds the mean (overdispersion). It generalizes the Poisson
/// distribution with an additional dispersion parameter θ (theta).
///
/// # Variance Function
///
/// For negative binomial data with μ = E\[Y\], the variance function is:
/// V(μ) = μ + μ²/θ
///
/// As θ → ∞, this approaches the Poisson variance V(μ) = μ.
///
/// # Unit Deviance
///
/// The unit deviance for negative binomial is:
/// d(y, μ) = 2[y·log(y/μ) - (y + θ)·log((y + θ)/(μ + θ))]
///
/// with appropriate handling for y = 0.
///
/// # Link Function
///
/// The canonical link is log, like Poisson.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NegativeBinomialFamily {
    /// The dispersion parameter (also called "size" in R).
    /// Higher values mean less overdispersion (more Poisson-like).
    pub theta: f64,
}

impl Default for NegativeBinomialFamily {
    fn default() -> Self {
        Self::new(1.0)
    }
}

impl NegativeBinomialFamily {
    /// Create a new negative binomial family with the specified theta.
    ///
    /// # Arguments
    ///
    /// * `theta` - The dispersion parameter (must be positive).
    ///   Smaller theta means more overdispersion.
    pub fn new(theta: f64) -> Self {
        assert!(theta > 0.0, "theta must be positive");
        Self { theta }
    }

    /// Create a negative binomial family that approximates Poisson (high theta).
    pub fn poisson_like() -> Self {
        Self::new(1e6)
    }

    /// Update the theta parameter (used during estimation).
    pub fn with_theta(&self, new_theta: f64) -> Self {
        Self::new(new_theta)
    }

    /// Compute the overdispersion ratio at given mean.
    ///
    /// This is V(μ)/μ = 1 + μ/θ, which should be > 1 for overdispersed data.
    pub fn overdispersion_ratio(&self, mu: f64) -> f64 {
        1.0 + mu / self.theta
    }
}

impl GlmFamily for NegativeBinomialFamily {
    /// Variance function V(μ) = μ + μ²/θ.
    fn variance(&self, mu: f64) -> f64 {
        let mu_safe = mu.max(1e-10);
        mu_safe + mu_safe * mu_safe / self.theta
    }

    /// Log link function (canonical): g(μ) = ln(μ).
    fn link(&self, mu: f64) -> f64 {
        mu.max(1e-10).ln()
    }

    /// Inverse log link: g⁻¹(η) = exp(η).
    fn link_inverse(&self, eta: f64) -> f64 {
        if eta > 30.0 {
            (30.0_f64).exp()
        } else if eta < -30.0 {
            1e-14
        } else {
            eta.exp().max(1e-14)
        }
    }

    /// Derivative of log link: dη/dμ = 1/μ.
    fn link_derivative(&self, mu: f64) -> f64 {
        1.0 / mu.max(1e-10)
    }

    /// Unit deviance for negative binomial.
    ///
    /// d(y, μ) = 2[y·log(y/μ) - (y + θ)·log((y + θ)/(μ + θ))]
    fn unit_deviance(&self, y: f64, mu: f64) -> f64 {
        let mu_safe = mu.max(1e-10);
        let theta = self.theta;

        if y < 1e-10 {
            // d(0, μ) = 2[0 − θ·log(θ/(μ + θ))] = 2θ·log(1 + μ/θ) ≥ 0
            2.0 * theta * (mu_safe / theta).ln_1p()
        } else {
            // Full formula
            let term1 = y * (y / mu_safe).ln();
            let term2 = (y + theta) * ((y + theta) / (mu_safe + theta)).ln();
            2.0 * (term1 - term2)
        }
    }

    /// Initialize μ values for IRLS iteration.
    ///
    /// Uses (y + y_mean) / 2 to ensure positive starting values.
    fn initialize_mu(&self, y: &[f64]) -> Vec<f64> {
        let y_mean = y.iter().sum::<f64>() / y.len() as f64;
        let y_mean = y_mean.max(1e-3);

        y.iter()
            .map(|&yi| {
                let mu = (yi + y_mean) / 2.0;
                mu.max(1e-3)
            })
            .collect()
    }
}

/// Estimate theta from residuals using method of moments.
///
/// Given the Pearson residuals and fitted values, estimate theta.
/// This is a simplified estimator; more accurate methods use MLE.
pub fn estimate_theta_moments(y: &[f64], mu: &[f64]) -> f64 {
    let n = y.len() as f64;

    // Pearson chi-squared
    let chi2: f64 = y
        .iter()
        .zip(mu.iter())
        .map(|(&yi, &mui)| {
            let mui = mui.max(1e-10);
            (yi - mui).powi(2) / mui
        })
        .sum();

    // For Poisson, chi2/n ≈ 1. For NB, chi2/n ≈ 1 + μ_bar/θ
    // So θ ≈ μ_bar / (chi2/n - 1)
    let mu_bar = mu.iter().sum::<f64>() / n;
    let overdispersion = (chi2 / n - 1.0).max(0.01);

    (mu_bar / overdispersion).max(0.1)
}

/// Maximum-likelihood estimate of θ given fitted means (MASS::theta.ml).
///
/// Newton–Raphson on the exact negative binomial profile log-likelihood in θ,
/// started from the moment estimate `n / Σ (y/μ − 1)²` exactly as
/// `MASS::theta.ml` does:
///
/// - score  `U(θ) = Σ [ψ(θ+y) − ψ(θ) + ln θ + 1 − ln(θ+μ) − (y+θ)/(μ+θ)]`
/// - info   `I(θ) = Σ [−ψ'(θ+y) + ψ'(θ) − 1/θ + 2/(μ+θ) − (y+θ)/(μ+θ)²]`
/// - update `θ ← θ + U/I`
pub fn estimate_theta_ml(y: &[f64], mu: &[f64], max_iter: usize, tol: f64) -> f64 {
    let n = y.len() as f64;

    // MASS starting value: n / Σ (y/μ − 1)².
    let denom: f64 = y
        .iter()
        .zip(mu.iter())
        .map(|(&yi, &mui)| (yi / mui.max(1e-10) - 1.0).powi(2))
        .sum();
    let mut theta = if denom > 0.0 && denom.is_finite() {
        n / denom
    } else {
        estimate_theta_moments(y, mu)
    };
    theta = theta.clamp(1e-4, 1e8);

    for _ in 0..max_iter.max(1) {
        let (score, info) = theta_score_and_info(y, mu, theta);
        if !score.is_finite() || !info.is_finite() || info.abs() < 1e-300 {
            break;
        }
        let mut new_theta = theta + score / info;
        if new_theta.is_nan() || new_theta <= 0.0 {
            // Newton overshot into θ ≤ 0: fall back to halving towards 0.
            new_theta = theta / 2.0;
        }
        let new_theta = new_theta.min(1e8);
        let converged = (new_theta - theta).abs() < tol * theta.max(1.0);
        theta = new_theta;
        if converged {
            break;
        }
    }

    theta
}

/// Score and (observed) information of the NB log-likelihood w.r.t. θ.
fn theta_score_and_info(y: &[f64], mu: &[f64], theta: f64) -> (f64, f64) {
    let mut score = 0.0;
    let mut info = 0.0;
    let psi_t = digamma(theta);
    let tri_t = trigamma(theta);

    for (&yi, &mui) in y.iter().zip(mu.iter()) {
        let mui = mui.max(1e-10);
        let mt = mui + theta;
        score += digamma(theta + yi) - psi_t + theta.ln() + 1.0 - mt.ln() - (yi + theta) / mt;
        info += -trigamma(theta + yi) + tri_t - 1.0 / theta + 2.0 / mt - (yi + theta) / (mt * mt);
    }

    (score, info)
}

/// Trigamma function ψ'(x) for x > 0 (recurrence + asymptotic expansion).
fn trigamma(mut x: f64) -> f64 {
    let mut acc = 0.0;
    while x < 10.0 {
        acc += 1.0 / (x * x);
        x += 1.0;
    }
    let x2 = 1.0 / (x * x);
    // ψ'(x) ~ 1/x + 1/(2x²) + 1/(6x³) − 1/(30x⁵) + 1/(42x⁷) − 1/(30x⁹) + 5/(66x¹¹)
    acc + 1.0 / x
        + x2 / 2.0
        + (1.0 / x)
            * x2
            * (1.0 / 6.0
                + x2 * (-1.0 / 30.0 + x2 * (1.0 / 42.0 + x2 * (-1.0 / 30.0 + x2 * (5.0 / 66.0)))))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trigamma_known_values() {
        let pi2 = std::f64::consts::PI.powi(2);
        assert!((trigamma(1.0) - pi2 / 6.0).abs() < 1e-12);
        assert!((trigamma(0.5) - pi2 / 2.0).abs() < 1e-12);
        // ψ'(x+1) = ψ'(x) − 1/x²
        assert!((trigamma(7.3) - (trigamma(6.3) - 1.0 / (6.3 * 6.3))).abs() < 1e-13);
    }

    #[test]
    fn test_variance() {
        let nb = NegativeBinomialFamily::new(2.0);

        // V(μ) = μ + μ²/θ
        // At μ = 4, θ = 2: V = 4 + 16/2 = 12
        assert!((nb.variance(4.0) - 12.0).abs() < 1e-10);

        // At μ = 1, θ = 2: V = 1 + 1/2 = 1.5
        assert!((nb.variance(1.0) - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_approaches_poisson() {
        let nb_high_theta = NegativeBinomialFamily::new(1e6);

        // With very high theta, variance ≈ μ (Poisson)
        let mu = 5.0;
        let var = nb_high_theta.variance(mu);
        assert!((var - mu).abs() / mu < 0.01);
    }

    #[test]
    fn test_link_roundtrip() {
        let nb = NegativeBinomialFamily::new(1.0);

        for mu in [0.5, 1.0, 2.0, 5.0, 10.0] {
            let eta = nb.link(mu);
            let mu_back = nb.link_inverse(eta);
            assert!((mu - mu_back).abs() < 1e-8, "Failed for mu={}", mu);
        }
    }

    #[test]
    fn test_unit_deviance_perfect_fit() {
        let nb = NegativeBinomialFamily::new(2.0);

        // Perfect prediction: y = μ
        let dev = nb.unit_deviance(5.0, 5.0);
        assert!(dev.abs() < 1e-10);
    }

    #[test]
    fn test_unit_deviance_zero() {
        let nb = NegativeBinomialFamily::new(2.0);

        // y = 0, μ = 1, θ = 2
        // d(0, 1) = -2θ·ln(θ/(μ+θ)) = 2 * 2 * ln(3/2) ≈ 1.62 (unit deviances are ≥ 0)
        let dev = nb.unit_deviance(0.0, 1.0);
        let expected = 2.0 * 2.0 * (3.0 / 2.0_f64).ln();
        assert!((dev - expected).abs() < 1e-12);
        assert!(dev > 0.0);
        // Continuous with the general formula as y → 0+.
        let near_zero = nb.unit_deviance(1e-9, 1.0);
        assert!((near_zero - dev).abs() < 1e-6);
    }

    #[test]
    fn test_deviance() {
        let nb = NegativeBinomialFamily::new(2.0);

        // Perfect fit
        let y = vec![1.0, 2.0, 3.0, 4.0];
        let mu = vec![1.0, 2.0, 3.0, 4.0];
        let dev = nb.deviance(&y, &mu);
        assert!(dev < 1e-8);
    }

    #[test]
    fn test_initialize_mu() {
        let nb = NegativeBinomialFamily::new(1.0);
        let y = vec![0.0, 1.0, 5.0, 10.0];
        let mu_init = nb.initialize_mu(&y);

        // All values should be positive
        for &mu in &mu_init {
            assert!(mu > 0.0);
        }
    }

    #[test]
    fn test_overdispersion_ratio() {
        let nb = NegativeBinomialFamily::new(2.0);

        // At μ = 4, θ = 2: ratio = 1 + 4/2 = 3
        assert!((nb.overdispersion_ratio(4.0) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_estimate_theta_moments() {
        // Create overdispersed data
        let y = vec![0.0, 1.0, 0.0, 5.0, 2.0, 0.0, 8.0, 1.0, 0.0, 3.0];
        let mu = vec![2.0; 10]; // Constant mean

        let theta = estimate_theta_moments(&y, &mu);

        // Theta should be positive
        assert!(theta > 0.0);
    }

    #[test]
    fn test_irls_weight() {
        let nb = NegativeBinomialFamily::new(2.0);

        // At μ = 2, θ = 2:
        // V(2) = 2 + 4/2 = 4
        // dη/dμ = 1/2 = 0.5
        // w = 1/(V * (dη/dμ)²) = 1/(4 * 0.25) = 1
        let w = nb.irls_weight(2.0);
        assert!((w - 1.0).abs() < 1e-10);
    }
}
