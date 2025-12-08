//! Link functions for Poisson GLM.
//!
//! Provides log (canonical), identity, and square root link functions
//! for count data regression models.

/// Link function types for Poisson regression.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PoissonLink {
    /// Log link (canonical): g(μ) = ln(μ)
    #[default]
    Log,
    /// Identity link: g(μ) = μ
    Identity,
    /// Square root link: g(μ) = √μ
    Sqrt,
}

impl PoissonLink {
    /// Compute the link function g(μ).
    ///
    /// Transforms the mean μ > 0 to the linear predictor η ∈ ℝ.
    #[inline]
    pub fn link(&self, mu: f64) -> f64 {
        // Clamp μ to avoid numerical issues
        let mu_clamped = mu.max(1e-10);

        match self {
            PoissonLink::Log => mu_clamped.ln(),
            PoissonLink::Identity => mu_clamped,
            PoissonLink::Sqrt => mu_clamped.sqrt(),
        }
    }

    /// Compute the inverse link function g⁻¹(η) = μ.
    ///
    /// Transforms the linear predictor η to the mean μ > 0.
    #[inline]
    pub fn link_inverse(&self, eta: f64) -> f64 {
        match self {
            PoissonLink::Log => {
                // exp(η), clamped for numerical stability
                if eta > 30.0 {
                    (30.0_f64).exp()
                } else if eta < -30.0 {
                    1e-14
                } else {
                    eta.exp().max(1e-14)
                }
            }
            PoissonLink::Identity => {
                // μ = η, must be positive
                eta.max(1e-14)
            }
            PoissonLink::Sqrt => {
                // μ = η², must be positive
                let eta_pos = eta.max(1e-7);
                eta_pos * eta_pos
            }
        }
    }

    /// Compute derivative of link function dη/dμ.
    #[inline]
    pub fn link_derivative(&self, mu: f64) -> f64 {
        // Clamp μ to avoid division by zero
        let mu_clamped = mu.max(1e-10);

        match self {
            PoissonLink::Log => {
                // d/dμ ln(μ) = 1/μ
                1.0 / mu_clamped
            }
            PoissonLink::Identity => {
                // d/dμ μ = 1
                1.0
            }
            PoissonLink::Sqrt => {
                // d/dμ √μ = 1/(2√μ)
                0.5 / mu_clamped.sqrt()
            }
        }
    }

    /// Compute derivative of inverse link function dμ/dη.
    #[inline]
    pub fn link_inverse_derivative(&self, eta: f64) -> f64 {
        match self {
            PoissonLink::Log => {
                // d/dη exp(η) = exp(η) = μ
                self.link_inverse(eta)
            }
            PoissonLink::Identity => {
                // d/dη η = 1
                1.0
            }
            PoissonLink::Sqrt => {
                // d/dη η² = 2η
                2.0 * eta.max(1e-7)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_link() {
        let link = PoissonLink::Log;

        // Test at μ = 1 -> η = 0
        assert!((link.link(1.0) - 0.0).abs() < 1e-10);

        // Test at μ = e -> η = 1
        assert!((link.link(std::f64::consts::E) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_log_inverse() {
        let link = PoissonLink::Log;

        // η = 0 -> μ = 1
        assert!((link.link_inverse(0.0) - 1.0).abs() < 1e-10);

        // η = 1 -> μ = e
        assert!((link.link_inverse(1.0) - std::f64::consts::E).abs() < 1e-10);
    }

    #[test]
    fn test_log_roundtrip() {
        let link = PoissonLink::Log;

        for mu in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0] {
            let eta = link.link(mu);
            let mu_back = link.link_inverse(eta);
            assert!((mu - mu_back).abs() < 1e-8, "Failed for mu={}", mu);
        }
    }

    #[test]
    fn test_identity_roundtrip() {
        let link = PoissonLink::Identity;

        for mu in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0] {
            let eta = link.link(mu);
            let mu_back = link.link_inverse(eta);
            assert!((mu - mu_back).abs() < 1e-8, "Failed for mu={}", mu);
        }
    }

    #[test]
    fn test_sqrt_roundtrip() {
        let link = PoissonLink::Sqrt;

        for mu in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0] {
            let eta = link.link(mu);
            let mu_back = link.link_inverse(eta);
            assert!((mu - mu_back).abs() < 1e-8, "Failed for mu={}", mu);
        }
    }

    #[test]
    fn test_log_derivative() {
        let link = PoissonLink::Log;

        // At μ = 1: derivative = 1/1 = 1
        assert!((link.link_derivative(1.0) - 1.0).abs() < 1e-10);

        // At μ = 2: derivative = 1/2 = 0.5
        assert!((link.link_derivative(2.0) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_identity_derivative() {
        let link = PoissonLink::Identity;

        // Derivative is always 1
        assert!((link.link_derivative(1.0) - 1.0).abs() < 1e-10);
        assert!((link.link_derivative(5.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_sqrt_derivative() {
        let link = PoissonLink::Sqrt;

        // At μ = 1: derivative = 1/(2*1) = 0.5
        assert!((link.link_derivative(1.0) - 0.5).abs() < 1e-10);

        // At μ = 4: derivative = 1/(2*2) = 0.25
        assert!((link.link_derivative(4.0) - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_log_inverse_derivative() {
        let link = PoissonLink::Log;

        // At η = 0: dμ/dη = exp(0) = 1
        assert!((link.link_inverse_derivative(0.0) - 1.0).abs() < 1e-10);

        // At η = 1: dμ/dη = exp(1) = e
        assert!((link.link_inverse_derivative(1.0) - std::f64::consts::E).abs() < 1e-10);
    }

    #[test]
    fn test_numerical_stability() {
        let link = PoissonLink::Log;

        // Extreme values should not panic or produce NaN
        assert!(link.link(1e-15).is_finite());
        assert!(link.link(1e15).is_finite());
        assert!(link.link_inverse(50.0).is_finite());
        assert!(link.link_inverse(-50.0).is_finite());
    }
}
