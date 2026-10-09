//! Link functions for binomial GLM.
//!
//! Provides logit, probit, and complementary log-log link functions
//! for binary outcome regression models.

use std::f64::consts::{FRAC_1_SQRT_2, PI};

/// Link function types for binomial regression.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BinomialLink {
    /// Logit link (canonical): g(μ) = log(μ/(1-μ))
    #[default]
    Logit,
    /// Probit link: g(μ) = Φ⁻¹(μ) where Φ is the standard normal CDF
    Probit,
    /// Complementary log-log link: g(μ) = log(-log(1-μ))
    Cloglog,
}

impl BinomialLink {
    /// Lower-case name of the link (`"logit"`, `"probit"`, `"cloglog"`).
    pub fn as_str(&self) -> &'static str {
        match self {
            BinomialLink::Logit => "logit",
            BinomialLink::Probit => "probit",
            BinomialLink::Cloglog => "cloglog",
        }
    }

    /// Compute the link function g(μ).
    ///
    /// Transforms the probability μ ∈ (0,1) to the linear predictor η ∈ ℝ.
    #[inline]
    pub fn link(&self, mu: f64) -> f64 {
        // Clamp μ to avoid numerical issues
        let mu_clamped = mu.clamp(1e-10, 1.0 - 1e-10);

        match self {
            BinomialLink::Logit => {
                // logit(μ) = log(μ/(1-μ))
                (mu_clamped / (1.0 - mu_clamped)).ln()
            }
            BinomialLink::Probit => {
                // Φ⁻¹(μ) - inverse standard normal CDF
                probit(mu_clamped)
            }
            BinomialLink::Cloglog => {
                // log(-log(1-μ))
                (-((1.0 - mu_clamped).ln())).ln()
            }
        }
    }

    /// Compute the inverse link function g⁻¹(η) = μ.
    ///
    /// Transforms the linear predictor η ∈ ℝ to probability μ ∈ (0,1).
    #[inline]
    pub fn link_inverse(&self, eta: f64) -> f64 {
        match self {
            BinomialLink::Logit => {
                // logistic(η) = 1 / (1 + exp(-η))
                // Numerically stable for large |η|
                if eta > 30.0 {
                    1.0 - 1e-14
                } else if eta < -30.0 {
                    1e-14
                } else {
                    1.0 / (1.0 + (-eta).exp())
                }
            }
            BinomialLink::Probit => {
                // Φ(η) - standard normal CDF
                let result = standard_normal_cdf(eta);
                result.clamp(1e-14, 1.0 - 1e-14)
            }
            BinomialLink::Cloglog => {
                // 1 - exp(-exp(η))
                let result = if eta > 10.0 {
                    1.0 - 1e-14
                } else if eta < -30.0 {
                    1e-14
                } else {
                    1.0 - (-eta.exp()).exp()
                };
                result.clamp(1e-14, 1.0 - 1e-14)
            }
        }
    }

    /// Compute derivative of link function dη/dμ.
    #[inline]
    pub fn link_derivative(&self, mu: f64) -> f64 {
        // Clamp μ to avoid division by zero
        let mu_clamped = mu.clamp(1e-10, 1.0 - 1e-10);

        match self {
            BinomialLink::Logit => {
                // d/dμ log(μ/(1-μ)) = 1/(μ(1-μ))
                1.0 / (mu_clamped * (1.0 - mu_clamped))
            }
            BinomialLink::Probit => {
                // d/dμ Φ⁻¹(μ) = 1/φ(Φ⁻¹(μ)) where φ is standard normal PDF
                let z = probit(mu_clamped);
                let pdf = standard_normal_pdf(z);
                if pdf < 1e-14 {
                    1e14 // Cap at large value
                } else {
                    1.0 / pdf
                }
            }
            BinomialLink::Cloglog => {
                // d/dμ log(-log(1-μ)) = 1/((1-μ)(-log(1-μ)))
                let one_minus_mu = 1.0 - mu_clamped;
                let neg_log = -one_minus_mu.ln();
                if neg_log < 1e-14 {
                    1e14 // Cap at large value
                } else {
                    1.0 / (one_minus_mu * neg_log)
                }
            }
        }
    }

    /// Compute derivative of inverse link function dμ/dη.
    #[inline]
    pub fn link_inverse_derivative(&self, eta: f64) -> f64 {
        match self {
            BinomialLink::Logit => {
                // d/dη (1/(1+exp(-η))) = exp(-η)/(1+exp(-η))² = μ(1-μ)
                let mu = self.link_inverse(eta);
                mu * (1.0 - mu)
            }
            BinomialLink::Probit => {
                // d/dη Φ(η) = φ(η) (standard normal PDF)
                standard_normal_pdf(eta)
            }
            BinomialLink::Cloglog => {
                // d/dη (1 - exp(-exp(η))) = exp(η - exp(η))
                if eta > 10.0 {
                    0.0
                } else if eta < -30.0 {
                    eta.exp()
                } else {
                    (eta - eta.exp()).exp()
                }
            }
        }
    }
}

/// Standard normal CDF Φ(x) using error function approximation.
#[inline]
fn standard_normal_cdf(x: f64) -> f64 {
    0.5 * statrs::function::erf::erfc(-x * FRAC_1_SQRT_2)
}

/// Standard normal PDF φ(x) = exp(-x²/2) / √(2π)
#[inline]
fn standard_normal_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * PI).sqrt()
}

/// Probit function (inverse standard normal CDF), `Φ⁻¹(p) = −√2 · erfc⁻¹(2p)`.
fn probit(p: f64) -> f64 {
    // For extreme values, use limits
    if p <= 1e-300 {
        return -38.0;
    }
    if p >= 1.0 - 1e-16 {
        return 8.2;
    }
    if p <= 1e-16 {
        return -8.2;
    }
    -std::f64::consts::SQRT_2 * statrs::function::erf::erfc_inv(2.0 * p)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_logit_link() {
        let link = BinomialLink::Logit;

        // Test at μ = 0.5 -> η = 0
        assert!((link.link(0.5) - 0.0).abs() < 1e-10);

        // Test at μ = 0.731... -> η ≈ 1.0
        let mu = 1.0 / (1.0 + (-1.0_f64).exp());
        assert!((link.link(mu) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_logit_inverse() {
        let link = BinomialLink::Logit;

        // η = 0 -> μ = 0.5
        assert!((link.link_inverse(0.0) - 0.5).abs() < 1e-10);

        // η = 1 -> μ = 1/(1+e^-1) ≈ 0.731
        let expected = 1.0 / (1.0 + (-1.0_f64).exp());
        assert!((link.link_inverse(1.0) - expected).abs() < 1e-10);
    }

    #[test]
    fn test_logit_roundtrip() {
        let link = BinomialLink::Logit;

        for mu in [0.1, 0.3, 0.5, 0.7, 0.9] {
            let eta = link.link(mu);
            let mu_back = link.link_inverse(eta);
            assert!((mu - mu_back).abs() < 1e-8, "Failed for mu={}", mu);
        }
    }

    #[test]
    fn test_probit_roundtrip() {
        let link = BinomialLink::Probit;

        for mu in [0.1, 0.3, 0.5, 0.7, 0.9] {
            let eta = link.link(mu);
            let mu_back = link.link_inverse(eta);
            assert!((mu - mu_back).abs() < 1e-6, "Failed for mu={}", mu);
        }
    }

    #[test]
    fn test_cloglog_roundtrip() {
        let link = BinomialLink::Cloglog;

        for mu in [0.1, 0.3, 0.5, 0.7, 0.9] {
            let eta = link.link(mu);
            let mu_back = link.link_inverse(eta);
            assert!((mu - mu_back).abs() < 1e-8, "Failed for mu={}", mu);
        }
    }

    #[test]
    fn test_logit_derivative() {
        let link = BinomialLink::Logit;

        // At μ = 0.5: derivative = 1/(0.5 * 0.5) = 4
        assert!((link.link_derivative(0.5) - 4.0).abs() < 1e-10);

        // At μ = 0.2: derivative = 1/(0.2 * 0.8) = 6.25
        assert!((link.link_derivative(0.2) - 6.25).abs() < 1e-10);
    }

    #[test]
    fn test_logit_inverse_derivative() {
        let link = BinomialLink::Logit;

        // At η = 0: μ = 0.5, dμ/dη = 0.5 * 0.5 = 0.25
        assert!((link.link_inverse_derivative(0.0) - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_numerical_stability() {
        let link = BinomialLink::Logit;

        // Extreme values should not panic or produce NaN
        assert!(link.link(1e-15).is_finite());
        assert!(link.link(1.0 - 1e-15).is_finite());
        assert!(link.link_inverse(50.0).is_finite());
        assert!(link.link_inverse(-50.0).is_finite());
    }

    #[test]
    fn test_probit_at_half() {
        let link = BinomialLink::Probit;

        // Φ⁻¹(0.5) = 0
        assert!(link.link(0.5).abs() < 1e-6);
        // Φ(0) = 0.5
        assert!((link.link_inverse(0.0) - 0.5).abs() < 1e-8);
    }

    #[test]
    fn test_cloglog_properties() {
        let link = BinomialLink::Cloglog;

        // cloglog(0.5) = log(-log(0.5)) = log(log(2)) ≈ -0.3665
        let expected = (0.5_f64.ln().abs()).ln().copysign(-1.0);
        assert!((link.link(0.5) - expected).abs() < 0.01);
    }

    // ==================== Additional tests for coverage ====================

    #[test]
    fn test_cloglog_extreme_eta() {
        let link = BinomialLink::Cloglog;

        // Very large eta (> 10) should return value close to 1
        let result_high = link.link_inverse(15.0);
        assert!(result_high > 0.99);
        assert!(result_high.is_finite());

        // Very small eta (< -30) should return value close to 0
        let result_low = link.link_inverse(-35.0);
        assert!(result_low < 0.01);
        assert!(result_low.is_finite());
    }

    #[test]
    fn test_probit_derivative_extreme() {
        let link = BinomialLink::Probit;

        // At extreme μ values, pdf becomes very small, derivative caps at 1e14
        // Test near boundaries
        let deriv_low = link.link_derivative(1e-9);
        assert!(deriv_low.is_finite());
        assert!(deriv_low > 0.0);

        let deriv_high = link.link_derivative(1.0 - 1e-9);
        assert!(deriv_high.is_finite());
        assert!(deriv_high > 0.0);
    }

    #[test]
    fn test_cloglog_derivative_extreme() {
        let link = BinomialLink::Cloglog;

        // At μ very close to 1, -log(1-μ) becomes very small, derivative caps
        let deriv_high = link.link_derivative(1.0 - 1e-12);
        assert!(deriv_high.is_finite());
        assert!(deriv_high > 0.0);

        // At μ very close to 0
        let deriv_low = link.link_derivative(1e-12);
        assert!(deriv_low.is_finite());
        assert!(deriv_low > 0.0);
    }

    #[test]
    fn test_cloglog_inverse_derivative_extreme() {
        let link = BinomialLink::Cloglog;

        // Very large eta (> 10) should return ~0
        let deriv_high = link.link_inverse_derivative(15.0);
        assert!(deriv_high.abs() < 1e-5);

        // Very small eta (< -30) should return exp(eta)
        let deriv_low = link.link_inverse_derivative(-35.0);
        assert!(deriv_low.is_finite());
        assert!((deriv_low - (-35.0_f64).exp()).abs() < 1e-20);
    }

    #[test]
    fn test_probit_extreme_values() {
        // Test extreme probability values to hit far tail region
        let link = BinomialLink::Probit;

        // Very small probability (far tail)
        let eta_very_low = link.link(1e-10);
        assert!(eta_very_low < -5.0);
        assert!(eta_very_low.is_finite());

        // Very high probability (far tail)
        let eta_very_high = link.link(1.0 - 1e-10);
        assert!(eta_very_high > 5.0);
        assert!(eta_very_high.is_finite());
    }

    #[test]
    fn test_probit_boundary_conditions() {
        // Test the exact boundary conditions in probit function
        // p <= 1e-300 returns -38.0
        let result_very_low = probit(1e-310);
        assert!((result_very_low - (-38.0)).abs() < 1e-10);

        // p >= 1.0 - 1e-16 returns 8.2
        let result_very_high = probit(1.0 - 1e-17);
        assert!((result_very_high - 8.2).abs() < 1e-10);

        // p <= 1e-16 returns -8.2
        let result_low = probit(1e-17);
        assert!((result_low - (-8.2)).abs() < 1e-10);
    }

    #[test]
    fn test_probit_far_tail_region() {
        // Test probabilities that hit the far tail region (r > 5.0)
        // Need q = min(p, 1-p) such that r = sqrt(-ln(q)) > 5
        // This means q < exp(-25) ≈ 1.4e-11
        let result1 = probit(1e-12);
        assert!(result1.is_finite());
        assert!(result1 < -6.0); // Should be very negative

        let result2 = probit(1.0 - 1e-12);
        assert!(result2.is_finite());
        assert!(result2 > 6.0); // Should be very positive
    }

    #[test]
    fn test_probit_inverse_derivative() {
        let link = BinomialLink::Probit;

        // At η = 0: dμ/dη = φ(0) = 1/√(2π) ≈ 0.3989
        let expected = 1.0 / (2.0 * std::f64::consts::PI).sqrt();
        assert!((link.link_inverse_derivative(0.0) - expected).abs() < 1e-6);
    }

    #[test]
    fn test_probit_matches_qnorm_across_regions() {
        // R: qnorm(c(0.2, 0.1, 0.075, 0.05, 0.9, 1e-10))
        let cases = [
            (0.2, -0.841_621_233_572_914_3),
            (0.1, -1.281_551_565_544_600_5),
            (0.075, -1.439_531_470_938_455_6),
            (0.05, -1.644_853_626_951_472_2),
            (0.9, 1.281_551_565_544_600_5),
            (1e-10, -6.361_340_902_404_056),
        ];
        for (p, want) in cases {
            let got = probit(p);
            assert!(
                (got - want).abs() < 1e-12,
                "probit({p}) = {got}, qnorm = {want}"
            );
        }
    }
}
