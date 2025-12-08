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
    0.5 * (1.0 + erf(x * FRAC_1_SQRT_2))
}

/// Standard normal PDF φ(x) = exp(-x²/2) / √(2π)
#[inline]
fn standard_normal_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * PI).sqrt()
}

/// Probit function (inverse standard normal CDF).
///
/// Uses Wichura's AS 241 algorithm for high accuracy.
#[allow(clippy::excessive_precision)]
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

    // Symmetry: use q = min(p, 1-p) and adjust sign
    let q = if p < 0.5 { p } else { 1.0 - p };

    if q > 0.425 {
        // Central region: use rational approximation
        let r = 0.180625 - (0.5 - p) * (0.5 - p);
        let num = ((((((2.5090809287301226727e3 * r + 3.3430575583588128105e4) * r
            + 6.7265770927008700853e4)
            * r
            + 4.5921953931549871457e4)
            * r
            + 1.3731693765509461125e4)
            * r
            + 1.9715909503065514427e3)
            * r
            + 1.3314166764078193025e2)
            * r
            + 3.3871328727963666080;
        let den = ((((((5.2264952788528545610e3 * r + 2.8729085735721942674e4) * r
            + 3.9307895513773136620e4)
            * r
            + 2.1213794301586595867e4)
            * r
            + 5.3941960214247511077e3)
            * r
            + 6.8718700749205790830e2)
            * r
            + 4.2313330701600911252e1)
            * r
            + 1.0;
        return (p - 0.5) * num / den;
    }

    // Tail region
    let r = (-q.ln()).sqrt();

    let result = if r <= 5.0 {
        // Intermediate region
        let r = r - 1.6;
        let num = ((((((7.74545014278341407640e-4 * r + 2.27238449892691845833e-2) * r
            + 2.41780725177450611770e-1)
            * r
            + 1.27045825245236838258)
            * r
            + 3.64784832476320460504)
            * r
            + 5.76949722146069140550)
            * r
            + 4.63033784615654529590)
            * r
            + 1.42343711074968357734;
        let den = ((((((1.05075007164441684324e-9 * r + 5.47593808499534494600e-4) * r
            + 1.51986665636164571966e-2)
            * r
            + 1.48103976427480074590e-1)
            * r
            + 6.89767334985100004550e-1)
            * r
            + 1.67638483018380384940)
            * r
            + 2.05319162663775882187)
            * r
            + 1.0;
        num / den
    } else {
        // Far tail
        let r = r - 5.0;
        let num = ((((((2.01033439929228813265e-7 * r + 2.71155556874348757815e-5) * r
            + 1.24266094738807843860e-3)
            * r
            + 2.65321895265761230930e-2)
            * r
            + 2.96560571828504891230e-1)
            * r
            + 1.78482653991729133580)
            * r
            + 5.46378491116411436990)
            * r
            + 6.65790464350110377720;
        let den = ((((((2.04426310338993978564e-15 * r + 1.42151175831644588870e-7) * r
            + 1.84631831751005468180e-5)
            * r
            + 7.86869131145613259100e-4)
            * r
            + 1.48753612908506148525e-2)
            * r
            + 1.36929880922735805310e-1)
            * r
            + 5.99832206555887937690e-1)
            * r
            + 1.0;
        num / den
    };

    if p < 0.5 {
        -result
    } else {
        result
    }
}

/// Error function approximation (Abramowitz and Stegun 7.1.26).
fn erf(x: f64) -> f64 {
    // Constants
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x_abs = x.abs();

    let t = 1.0 / (1.0 + p * x_abs);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x_abs * x_abs).exp();

    sign * y
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
}
