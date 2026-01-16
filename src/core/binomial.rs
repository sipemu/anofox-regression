//! Binomial family for logistic regression and binary outcome models.
//!
//! This module provides the binomial family with logit, probit, and
//! complementary log-log link functions for binary response regression.
//!
//! # Example
//!
//! ```ignore
//! use regress::BinomialFamily;
//!
//! // Logistic regression (default)
//! let logistic = BinomialFamily::logistic();
//!
//! // Probit regression
//! let probit = BinomialFamily::probit();
//!
//! // Complementary log-log
//! let cloglog = BinomialFamily::cloglog();
//! ```

use super::family::GlmFamily;
use super::link::BinomialLink;

/// Binomial family for binary outcome regression.
///
/// Supports logistic regression (logit link), probit regression (probit link),
/// and complementary log-log regression.
///
/// # Variance Function
///
/// For binomial data with μ = E\[Y\], the variance function is:
/// V(μ) = μ(1 - μ)
///
/// # Unit Deviance
///
/// The unit deviance for binomial is:
/// d(y, μ) = 2[y·log(y/μ) + (1-y)·log((1-y)/(1-μ))]
///
/// with appropriate handling for y = 0 or y = 1.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BinomialFamily {
    /// The link function to use.
    pub link: BinomialLink,
}

impl Default for BinomialFamily {
    fn default() -> Self {
        Self::logistic()
    }
}

impl BinomialFamily {
    /// Create a new binomial family with the specified link.
    pub fn new(link: BinomialLink) -> Self {
        Self { link }
    }

    /// Create a logistic regression family (logit link - canonical).
    pub fn logistic() -> Self {
        Self {
            link: BinomialLink::Logit,
        }
    }

    /// Create a probit regression family.
    pub fn probit() -> Self {
        Self {
            link: BinomialLink::Probit,
        }
    }

    /// Create a complementary log-log regression family.
    ///
    /// Useful for modeling rare events or asymmetric responses.
    pub fn cloglog() -> Self {
        Self {
            link: BinomialLink::Cloglog,
        }
    }

    /// Check if the current link is the canonical link (logit).
    pub fn is_canonical_link(&self) -> bool {
        self.link == BinomialLink::Logit
    }
}

impl GlmFamily for BinomialFamily {
    /// Variance function V(μ) = μ(1-μ).
    fn variance(&self, mu: f64) -> f64 {
        let mu_clamped = mu.clamp(1e-10, 1.0 - 1e-10);
        mu_clamped * (1.0 - mu_clamped)
    }

    fn link(&self, mu: f64) -> f64 {
        self.link.link(mu)
    }

    fn link_inverse(&self, eta: f64) -> f64 {
        self.link.link_inverse(eta)
    }

    fn link_derivative(&self, mu: f64) -> f64 {
        self.link.link_derivative(mu)
    }

    /// Check if μ is valid for binomial family: 0 < μ < 1.
    fn valid_mu(&self, mu: f64) -> bool {
        mu.is_finite() && mu > 0.0 && mu < 1.0
    }

    /// Clamp μ to valid range for binomial: (ε, 1-ε).
    fn clamp_mu(&self, mu: f64) -> f64 {
        mu.clamp(1e-10, 1.0 - 1e-10)
    }

    /// Unit deviance: d(y,μ) = 2[y·log(y/μ) + (1-y)·log((1-y)/(1-μ))].
    ///
    /// Uses limit values for y = 0 or y = 1 to avoid numerical issues.
    fn unit_deviance(&self, y: f64, mu: f64) -> f64 {
        let mu_clamped = mu.clamp(1e-10, 1.0 - 1e-10);

        let term1 = if y > 1e-10 {
            y * (y / mu_clamped).ln()
        } else {
            0.0
        };

        let term2 = if y < 1.0 - 1e-10 {
            (1.0 - y) * ((1.0 - y) / (1.0 - mu_clamped)).ln()
        } else {
            0.0
        };

        // Guard against numerical precision issues that could give small negative values
        (2.0 * (term1 + term2)).max(0.0)
    }

    /// Initialize μ values for IRLS iteration.
    ///
    /// Uses (y + 0.5) / 2 to start with values away from 0 and 1.
    fn initialize_mu(&self, y: &[f64]) -> Vec<f64> {
        y.iter()
            .map(|&yi| {
                // Push toward 0.5 to avoid boundary issues
                (yi + 0.5) / 2.0
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_logistic_family() {
        let fam = BinomialFamily::logistic();
        assert!(fam.is_canonical_link());
        assert_eq!(fam.link, BinomialLink::Logit);
    }

    #[test]
    fn test_probit_family() {
        let fam = BinomialFamily::probit();
        assert!(!fam.is_canonical_link());
        assert_eq!(fam.link, BinomialLink::Probit);
    }

    #[test]
    fn test_cloglog_family() {
        let fam = BinomialFamily::cloglog();
        assert!(!fam.is_canonical_link());
        assert_eq!(fam.link, BinomialLink::Cloglog);
    }

    #[test]
    fn test_variance() {
        let fam = BinomialFamily::logistic();

        // V(0.5) = 0.5 * 0.5 = 0.25
        assert!((fam.variance(0.5) - 0.25).abs() < 1e-10);

        // V(0.2) = 0.2 * 0.8 = 0.16
        assert!((fam.variance(0.2) - 0.16).abs() < 1e-10);

        // V(0.8) = 0.8 * 0.2 = 0.16
        assert!((fam.variance(0.8) - 0.16).abs() < 1e-10);
    }

    #[test]
    fn test_link_roundtrip() {
        let families = [
            BinomialFamily::logistic(),
            BinomialFamily::probit(),
            BinomialFamily::cloglog(),
        ];

        for fam in &families {
            for mu in [0.1, 0.3, 0.5, 0.7, 0.9] {
                let eta = fam.link(mu);
                let mu_back = fam.link_inverse(eta);
                assert!(
                    (mu - mu_back).abs() < 1e-6,
                    "Roundtrip failed for {:?} at mu={}",
                    fam.link,
                    mu
                );
            }
        }
    }

    #[test]
    fn test_unit_deviance() {
        let fam = BinomialFamily::logistic();

        // Perfect prediction: y = μ
        assert!(fam.unit_deviance(0.5, 0.5).abs() < 1e-10);

        // y = 1, μ = 0.9: d = 2 * ln(1/0.9) ≈ 0.211
        let dev = fam.unit_deviance(1.0, 0.9);
        let expected = 2.0 * (1.0 / 0.9_f64).ln();
        assert!((dev - expected).abs() < 1e-6);

        // y = 0, μ = 0.1: d = 2 * ln(1/0.9) ≈ 0.211
        let dev = fam.unit_deviance(0.0, 0.1);
        let expected = 2.0 * (1.0 / 0.9_f64).ln();
        assert!((dev - expected).abs() < 1e-6);
    }

    #[test]
    fn test_deviance() {
        let fam = BinomialFamily::logistic();

        // Perfect fit
        let y = vec![0.0, 0.0, 1.0, 1.0];
        let mu = vec![0.0001, 0.0001, 0.9999, 0.9999];
        let dev = fam.deviance(&y, &mu);
        assert!(dev < 0.01); // Very small deviance
    }

    #[test]
    fn test_initialize_mu() {
        let fam = BinomialFamily::logistic();
        let y = vec![0.0, 1.0, 0.0, 1.0];
        let mu_init = fam.initialize_mu(&y);

        // All values should be between 0 and 1, pushed toward 0.5
        for &mu in &mu_init {
            assert!(mu > 0.0 && mu < 1.0);
            assert!(mu > 0.2 && mu < 0.8); // Pushed toward center
        }
    }

    #[test]
    fn test_irls_weight_logistic() {
        let fam = BinomialFamily::logistic();

        // At μ = 0.5 with logit link:
        // V(μ) = 0.25, dη/dμ = 4
        // w = 1/(0.25 * 16) = 0.25
        let w = fam.irls_weight(0.5);
        assert!((w - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_working_response() {
        let fam = BinomialFamily::logistic();
        let y = 1.0;
        let mu = 0.5;
        let eta = fam.link(mu);

        let z = fam.working_response(y, mu, eta);
        // z = 0 + (1 - 0.5) * 4 = 2
        let expected = eta + (y - mu) * fam.link_derivative(mu);
        assert!((z - expected).abs() < 1e-10);
    }

    #[test]
    fn test_null_deviance() {
        let fam = BinomialFamily::logistic();
        let y = vec![0.0, 0.0, 1.0, 1.0];
        let null_dev = fam.null_deviance(&y);

        // With y_mean = 0.5, null deviance should be positive
        assert!(null_dev > 0.0);
    }
}
