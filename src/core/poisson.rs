//! Poisson family for count data regression.
//!
//! This module provides the Poisson family with log (canonical), identity,
//! and square root link functions for count data regression.
//!
//! # Example
//!
//! ```ignore
//! use regress_rs::PoissonFamily;
//!
//! // Poisson regression with log link (canonical, default)
//! let poisson = PoissonFamily::log();
//!
//! // Poisson with identity link
//! let poisson_id = PoissonFamily::identity();
//!
//! // Poisson with sqrt link
//! let poisson_sqrt = PoissonFamily::sqrt();
//! ```

use super::family::GlmFamily;
use super::poisson_link::PoissonLink;

/// Poisson family for count data regression.
///
/// Supports Poisson regression with log link (canonical), identity link,
/// and square root link functions.
///
/// # Variance Function
///
/// For Poisson data with μ = E\[Y\], the variance function is:
/// V(μ) = μ
///
/// # Unit Deviance
///
/// The unit deviance for Poisson is:
/// d(y, μ) = 2[y·log(y/μ) - (y - μ)]
///
/// with d(0, μ) = 2μ for y = 0.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PoissonFamily {
    /// The link function to use.
    pub link: PoissonLink,
}

impl Default for PoissonFamily {
    fn default() -> Self {
        Self::log()
    }
}

impl PoissonFamily {
    /// Create a new Poisson family with the specified link.
    pub fn new(link: PoissonLink) -> Self {
        Self { link }
    }

    /// Create a Poisson family with log link (canonical).
    pub fn log() -> Self {
        Self {
            link: PoissonLink::Log,
        }
    }

    /// Create a Poisson family with identity link.
    pub fn identity() -> Self {
        Self {
            link: PoissonLink::Identity,
        }
    }

    /// Create a Poisson family with square root link.
    pub fn sqrt() -> Self {
        Self {
            link: PoissonLink::Sqrt,
        }
    }

    /// Check if the current link is the canonical link (log).
    pub fn is_canonical_link(&self) -> bool {
        self.link == PoissonLink::Log
    }
}

impl GlmFamily for PoissonFamily {
    /// Variance function V(μ) = μ.
    fn variance(&self, mu: f64) -> f64 {
        mu.max(1e-10)
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

    /// Unit deviance: d(y, μ) = 2[y·log(y/μ) - (y - μ)].
    ///
    /// For y = 0: d(0, μ) = 2μ.
    fn unit_deviance(&self, y: f64, mu: f64) -> f64 {
        let mu_clamped = mu.max(1e-10);

        if y < 1e-10 {
            // d(0, μ) = 2μ
            2.0 * mu_clamped
        } else {
            // d(y, μ) = 2[y·log(y/μ) - (y - μ)]
            2.0 * (y * (y / mu_clamped).ln() - (y - mu_clamped))
        }
    }

    /// Initialize μ values for IRLS iteration.
    ///
    /// Uses (y + y_mean) / 2 to ensure positive starting values.
    fn initialize_mu(&self, y: &[f64]) -> Vec<f64> {
        let y_mean = y.iter().sum::<f64>() / y.len() as f64;
        let y_mean = y_mean.max(1e-3); // Ensure positive mean

        y.iter()
            .map(|&yi| {
                // Push toward mean to avoid boundary issues
                let mu = (yi + y_mean) / 2.0;
                mu.max(1e-3) // Ensure positive
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_family() {
        let fam = PoissonFamily::log();
        assert!(fam.is_canonical_link());
        assert_eq!(fam.link, PoissonLink::Log);
    }

    #[test]
    fn test_identity_family() {
        let fam = PoissonFamily::identity();
        assert!(!fam.is_canonical_link());
        assert_eq!(fam.link, PoissonLink::Identity);
    }

    #[test]
    fn test_sqrt_family() {
        let fam = PoissonFamily::sqrt();
        assert!(!fam.is_canonical_link());
        assert_eq!(fam.link, PoissonLink::Sqrt);
    }

    #[test]
    fn test_variance() {
        let fam = PoissonFamily::log();

        // V(μ) = μ
        assert!((fam.variance(1.0) - 1.0).abs() < 1e-10);
        assert!((fam.variance(5.0) - 5.0).abs() < 1e-10);
        assert!((fam.variance(10.0) - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_link_roundtrip() {
        let families = [
            PoissonFamily::log(),
            PoissonFamily::identity(),
            PoissonFamily::sqrt(),
        ];

        for fam in &families {
            for mu in [0.5, 1.0, 2.0, 5.0, 10.0] {
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
    fn test_unit_deviance_perfect_fit() {
        let fam = PoissonFamily::log();

        // Perfect prediction: y = μ
        // d(y, y) = 2[y·log(1) - 0] = 0
        assert!(fam.unit_deviance(5.0, 5.0).abs() < 1e-10);
        assert!(fam.unit_deviance(1.0, 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_unit_deviance_zero() {
        let fam = PoissonFamily::log();

        // y = 0, μ = 1: d = 2 * 1 = 2
        let dev = fam.unit_deviance(0.0, 1.0);
        assert!((dev - 2.0).abs() < 1e-10);

        // y = 0, μ = 5: d = 2 * 5 = 10
        let dev = fam.unit_deviance(0.0, 5.0);
        assert!((dev - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_unit_deviance_nonzero() {
        let fam = PoissonFamily::log();

        // y = 5, μ = 4
        // d = 2 * [5 * ln(5/4) - (5 - 4)]
        //   = 2 * [5 * ln(1.25) - 1]
        //   = 2 * [5 * 0.2231 - 1]
        //   ≈ 0.2315
        let dev = fam.unit_deviance(5.0, 4.0);
        let expected = 2.0 * (5.0 * (5.0_f64 / 4.0).ln() - 1.0);
        assert!((dev - expected).abs() < 1e-6);
    }

    #[test]
    fn test_deviance() {
        let fam = PoissonFamily::log();

        // Perfect fit
        let y = vec![1.0, 2.0, 3.0, 4.0];
        let mu = vec![1.0, 2.0, 3.0, 4.0];
        let dev = fam.deviance(&y, &mu);
        assert!(dev < 1e-8); // Zero deviance for perfect fit
    }

    #[test]
    fn test_initialize_mu() {
        let fam = PoissonFamily::log();
        let y = vec![0.0, 1.0, 5.0, 10.0];
        let mu_init = fam.initialize_mu(&y);

        // All values should be positive
        for &mu in &mu_init {
            assert!(mu > 0.0);
        }

        // Mean of y is 4, so values should be pushed toward 4
        assert!(mu_init[0] > 0.0); // Was 0, now positive
    }

    #[test]
    fn test_irls_weight_log() {
        let fam = PoissonFamily::log();

        // At μ = 1 with log link:
        // V(μ) = 1, dη/dμ = 1
        // w = 1/(1 * 1) = 1
        let w = fam.irls_weight(1.0);
        assert!((w - 1.0).abs() < 1e-10);

        // At μ = 4 with log link:
        // V(μ) = 4, dη/dμ = 0.25
        // w = 1/(4 * 0.0625) = 1/0.25 = 4
        let w = fam.irls_weight(4.0);
        assert!((w - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_working_response() {
        let fam = PoissonFamily::log();
        let y = 5.0;
        let mu = 4.0;
        let eta = fam.link(mu);

        let z = fam.working_response(y, mu, eta);
        let expected = eta + (y - mu) * fam.link_derivative(mu);
        assert!((z - expected).abs() < 1e-10);
    }

    #[test]
    fn test_null_deviance() {
        let fam = PoissonFamily::log();
        let y = vec![1.0, 2.0, 3.0, 4.0];
        let null_dev = fam.null_deviance(&y);

        // Null deviance should be positive
        assert!(null_dev > 0.0);
    }
}
