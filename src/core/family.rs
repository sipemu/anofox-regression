//! GLM family definitions for generalized linear models.
//!
//! This module provides variance and link functions for GLM families,
//! following R's `glm()` implementation.
//!
//! # Reference
//!
//! - Dunn, P.K. and Smyth, G.K. (2018). "Generalized linear models with examples in R".
//!   Springer, New York, NY. Chapter 12.
//! - R package `statmod`: <https://cran.r-project.org/web/packages/statmod/index.html>

/// Trait for GLM family definitions.
///
/// A GLM family specifies:
/// - A variance function V(μ) relating variance to the mean
/// - A link function g(μ) = η relating the mean to the linear predictor
/// - Methods for IRLS (Iteratively Reweighted Least Squares) fitting
pub trait GlmFamily {
    /// Compute the variance function V(μ).
    ///
    /// The variance of Y given μ is `Var[Y] = φ * V(μ)` where φ is the dispersion.
    fn variance(&self, mu: f64) -> f64;

    /// Compute the link function g(μ) = η.
    fn link(&self, mu: f64) -> f64;

    /// Compute the inverse link function g⁻¹(η) = μ.
    fn link_inverse(&self, eta: f64) -> f64;

    /// Compute derivative of link function dη/dμ.
    fn link_derivative(&self, mu: f64) -> f64;

    /// Compute IRLS weight for observation.
    ///
    /// Weight = 1 / (V(μ) * (dη/dμ)²)
    fn irls_weight(&self, mu: f64) -> f64 {
        let v = self.variance(mu);
        let link_deriv = self.link_derivative(mu);

        if v.abs() < 1e-14 || link_deriv.abs() < 1e-14 {
            return 1e-10;
        }

        1.0 / (v * link_deriv * link_deriv)
    }

    /// Compute working response (adjusted dependent variable) for IRLS.
    ///
    /// z = η + (y - μ) * (dη/dμ)
    fn working_response(&self, y: f64, mu: f64, eta: f64) -> f64 {
        let link_deriv = self.link_derivative(mu);
        eta + (y - mu) * link_deriv
    }

    /// Compute unit deviance d(y, μ) for a single observation.
    fn unit_deviance(&self, y: f64, mu: f64) -> f64;

    /// Compute total deviance: D = Σ d(yᵢ, μᵢ).
    fn deviance(&self, y: &[f64], mu: &[f64]) -> f64 {
        y.iter()
            .zip(mu.iter())
            .map(|(&yi, &mui)| self.unit_deviance(yi, mui))
            .sum()
    }

    /// Initialize μ values for IRLS iteration.
    fn initialize_mu(&self, y: &[f64]) -> Vec<f64>;

    /// Compute null deviance (deviance of intercept-only model).
    fn null_deviance(&self, y: &[f64]) -> f64 {
        let y_mean: f64 = y.iter().sum::<f64>() / y.len() as f64;
        y.iter().map(|&yi| self.unit_deviance(yi, y_mean)).sum()
    }
}

/// Tweedie family for generalized linear models.
///
/// The Tweedie family is parameterized by:
/// - `var_power` (p): The power of the variance function V(μ) = μ^p
/// - `link_power` (q): The power of the link function g(μ) = μ^q (Box-Cox)
///
/// # Special Cases
///
/// | var_power | Distribution |
/// |-----------|--------------|
/// | 0 | Normal (Gaussian) |
/// | 1 | Poisson |
/// | (1, 2) | Compound Poisson-Gamma |
/// | 2 | Gamma |
/// | 3 | Inverse-Gaussian |
///
/// Values of var_power between 0 and 1 are not allowed (no valid distribution).
#[derive(Debug, Clone, Copy)]
pub struct TweedieFamily {
    /// Power of the variance function: V(μ) = μ^var_power
    pub var_power: f64,
    /// Power of the link function (Box-Cox): g(μ) = μ^link_power for link_power != 0
    /// For link_power = 0, the link is log: g(μ) = log(μ)
    pub link_power: f64,
}

impl Default for TweedieFamily {
    fn default() -> Self {
        // Default: Compound Poisson-Gamma with log link
        Self {
            var_power: 1.5,
            link_power: 0.0,
        }
    }
}

impl TweedieFamily {
    /// Create a new Tweedie family with the given parameters.
    ///
    /// # Arguments
    ///
    /// * `var_power` - Power of variance function. Common values:
    ///   - 0: Normal
    ///   - 1: Poisson
    ///   - 1.5: Compound Poisson-Gamma (good for zero-inflated continuous)
    ///   - 2: Gamma
    ///   - 3: Inverse-Gaussian
    /// * `link_power` - Power of link function:
    ///   - 0: Log link (most common)
    ///   - 1: Identity link
    ///   - -1: Inverse link
    ///
    /// # Panics
    ///
    /// Panics if var_power is in (0, 1), which has no valid distribution.
    pub fn new(var_power: f64, link_power: f64) -> Self {
        assert!(
            !(var_power > 0.0 && var_power < 1.0),
            "var_power in (0, 1) is not allowed (no valid distribution)"
        );
        Self {
            var_power,
            link_power,
        }
    }

    /// Create a Gaussian (Normal) family.
    pub fn gaussian() -> Self {
        Self::new(0.0, 1.0) // var_power=0 (constant variance), identity link
    }

    /// Create a Poisson family with log link.
    pub fn poisson() -> Self {
        Self::new(1.0, 0.0) // var_power=1, log link
    }

    /// Create a Gamma family with log link.
    pub fn gamma() -> Self {
        Self::new(2.0, 0.0) // var_power=2, log link
    }

    /// Create an Inverse-Gaussian family with log link.
    pub fn inverse_gaussian() -> Self {
        Self::new(3.0, 0.0) // var_power=3, log link
    }

    /// Create a compound Poisson-Gamma family (for zero-inflated continuous data).
    ///
    /// This is useful for data with exact zeros and positive continuous values,
    /// such as insurance claims or rainfall amounts.
    pub fn compound_poisson_gamma(var_power: f64) -> Self {
        assert!(
            var_power > 1.0 && var_power < 2.0,
            "Compound Poisson-Gamma requires var_power in (1, 2)"
        );
        Self::new(var_power, 0.0) // log link
    }

    // ========== Variance Function ==========

    /// Compute the variance function V(μ) = μ^var_power.
    ///
    /// The variance of Y given μ is `Var[Y] = φ * V(μ)` where φ is the dispersion.
    #[inline]
    pub fn variance(&self, mu: f64) -> f64 {
        if self.var_power == 0.0 {
            1.0 // Normal: V(μ) = 1
        } else if self.var_power == 1.0 {
            mu // Poisson: V(μ) = μ
        } else if self.var_power == 2.0 {
            mu * mu // Gamma: V(μ) = μ²
        } else {
            mu.powf(self.var_power)
        }
    }

    /// Compute derivative of variance function: dV/dμ = var_power * μ^(var_power-1).
    #[inline]
    pub fn variance_derivative(&self, mu: f64) -> f64 {
        if self.var_power == 0.0 {
            0.0
        } else if self.var_power == 1.0 {
            1.0
        } else if self.var_power == 2.0 {
            2.0 * mu
        } else {
            self.var_power * mu.powf(self.var_power - 1.0)
        }
    }

    // ========== Link Function ==========

    /// Compute the link function g(μ).
    ///
    /// For link_power != 0: g(μ) = μ^link_power (Box-Cox)
    /// For link_power = 0: g(μ) = log(μ)
    #[inline]
    pub fn link(&self, mu: f64) -> f64 {
        if self.link_power == 0.0 {
            mu.ln() // Log link
        } else if self.link_power == 1.0 {
            mu // Identity link
        } else if self.link_power == -1.0 {
            1.0 / mu // Inverse link
        } else {
            mu.powf(self.link_power)
        }
    }

    /// Compute the inverse link function g^(-1)(η) = μ.
    ///
    /// For link_power != 0: μ = η^(1/link_power)
    /// For link_power = 0: μ = exp(η)
    #[inline]
    pub fn link_inverse(&self, eta: f64) -> f64 {
        if self.link_power == 0.0 {
            eta.exp() // Inverse of log
        } else if self.link_power == 1.0 {
            eta // Inverse of identity
        } else if self.link_power == -1.0 {
            1.0 / eta // Inverse of inverse
        } else {
            eta.powf(1.0 / self.link_power)
        }
    }

    /// Compute derivative of inverse link: dμ/dη.
    ///
    /// For link_power != 0: dμ/dη = (1/link_power) * η^(1/link_power - 1)
    /// For link_power = 0: dμ/dη = exp(η) = μ
    #[inline]
    pub fn link_inverse_derivative(&self, eta: f64) -> f64 {
        if self.link_power == 0.0 {
            eta.exp() // d/dη exp(η) = exp(η)
        } else if self.link_power == 1.0 {
            1.0 // d/dη η = 1
        } else if self.link_power == -1.0 {
            -1.0 / (eta * eta) // d/dη (1/η) = -1/η²
        } else {
            (1.0 / self.link_power) * eta.powf(1.0 / self.link_power - 1.0)
        }
    }

    /// Compute derivative of link function: dη/dμ.
    ///
    /// This is the reciprocal of link_inverse_derivative.
    #[inline]
    pub fn link_derivative(&self, mu: f64) -> f64 {
        if self.link_power == 0.0 {
            1.0 / mu // d/dμ log(μ) = 1/μ
        } else if self.link_power == 1.0 {
            1.0 // d/dμ μ = 1
        } else if self.link_power == -1.0 {
            -1.0 / (mu * mu) // d/dμ (1/μ) = -1/μ²
        } else {
            self.link_power * mu.powf(self.link_power - 1.0)
        }
    }

    // ========== IRLS Weights and Working Response ==========

    /// Compute IRLS weight for observation.
    ///
    /// Weight = 1 / (V(μ) * (dη/dμ)²)
    ///
    /// This is used in iteratively reweighted least squares.
    #[inline]
    pub fn irls_weight(&self, mu: f64) -> f64 {
        let v = self.variance(mu);
        let link_deriv = self.link_derivative(mu);

        if v.abs() < 1e-14 || link_deriv.abs() < 1e-14 {
            return 1e-10; // Avoid division by zero
        }

        1.0 / (v * link_deriv * link_deriv)
    }

    /// Compute working response (adjusted dependent variable) for IRLS.
    ///
    /// z = η + (y - μ) * (dη/dμ)
    ///
    /// where η = g(μ) is the linear predictor.
    #[inline]
    pub fn working_response(&self, y: f64, mu: f64, eta: f64) -> f64 {
        let link_deriv = self.link_derivative(mu);
        eta + (y - mu) * link_deriv
    }

    // ========== Deviance ==========

    /// Compute unit deviance d(y, μ) for a single observation.
    ///
    /// The deviance is D = 2 * Σ d(yᵢ, μᵢ).
    ///
    /// For Tweedie with var_power p:
    /// - p = 0 (Normal): d(y,μ) = (y - μ)²
    /// - p = 1 (Poisson): d(y,μ) = 2 * (y * log(y/μ) - (y - μ))
    /// - p = 2 (Gamma): d(y,μ) = 2 * (-log(y/μ) + (y - μ)/μ)
    /// - General: Uses Tweedie deviance formula
    pub fn unit_deviance(&self, y: f64, mu: f64) -> f64 {
        let p = self.var_power;

        if p == 0.0 {
            // Normal
            (y - mu).powi(2)
        } else if (p - 1.0).abs() < 1e-10 {
            // Poisson
            if y > 0.0 {
                2.0 * (y * (y / mu).ln() - (y - mu))
            } else {
                2.0 * mu
            }
        } else if (p - 2.0).abs() < 1e-10 {
            // Gamma
            2.0 * (-(y / mu).ln() + (y - mu) / mu)
        } else {
            // General Tweedie
            // d(y,μ) = 2 * (y^(2-p)/((1-p)*(2-p)) - y*μ^(1-p)/(1-p) + μ^(2-p)/(2-p))
            let term1 = if y > 0.0 {
                y.powf(2.0 - p) / ((1.0 - p) * (2.0 - p))
            } else {
                0.0
            };
            let term2 = if y > 0.0 {
                -y * mu.powf(1.0 - p) / (1.0 - p)
            } else {
                0.0
            };
            let term3 = mu.powf(2.0 - p) / (2.0 - p);

            2.0 * (term1 + term2 + term3)
        }
    }

    /// Compute total deviance: D = 2 * Σ d(yᵢ, μᵢ).
    pub fn deviance(&self, y: &[f64], mu: &[f64]) -> f64 {
        y.iter()
            .zip(mu.iter())
            .map(|(&yi, &mui)| self.unit_deviance(yi, mui))
            .sum()
    }

    /// Compute null deviance (deviance of intercept-only model).
    pub fn null_deviance(&self, y: &[f64]) -> f64 {
        let y_mean: f64 = y.iter().sum::<f64>() / y.len() as f64;
        y.iter().map(|&yi| self.unit_deviance(yi, y_mean)).sum()
    }

    // ========== Initialization ==========

    /// Initialize μ values for IRLS iteration.
    ///
    /// Different strategies for different families to ensure valid starting values.
    #[allow(clippy::redundant_guards)]
    pub fn initialize_mu(&self, y: &[f64]) -> Vec<f64> {
        let n = y.len();
        let y_mean: f64 = y.iter().sum::<f64>() / n as f64;

        match self.var_power {
            p if p == 0.0 => {
                // Normal: μ = y is fine
                y.to_vec()
            }
            p if p >= 1.0 => {
                // Poisson, Gamma, etc.: Need μ > 0
                // Use (y + y_mean) / 2 to avoid zeros
                y.iter()
                    .map(|&yi| {
                        let mu = (yi + y_mean) / 2.0;
                        if mu <= 0.0 {
                            y_mean.max(0.1)
                        } else {
                            mu
                        }
                    })
                    .collect()
            }
            _ => y.to_vec(),
        }
    }

    /// Check if variance power is valid.
    pub fn is_valid(&self) -> bool {
        // var_power in (0, 1) is not allowed
        !(self.var_power > 0.0 && self.var_power < 1.0)
    }

    /// Get the canonical link power for this variance function.
    ///
    /// The canonical link satisfies θ = η where θ is the natural parameter.
    pub fn canonical_link_power(&self) -> f64 {
        1.0 - self.var_power
    }

    /// Check if the current link is the canonical link.
    pub fn is_canonical_link(&self) -> bool {
        (self.link_power - self.canonical_link_power()).abs() < 1e-10
    }
}

impl GlmFamily for TweedieFamily {
    fn variance(&self, mu: f64) -> f64 {
        TweedieFamily::variance(self, mu)
    }

    fn link(&self, mu: f64) -> f64 {
        TweedieFamily::link(self, mu)
    }

    fn link_inverse(&self, eta: f64) -> f64 {
        TweedieFamily::link_inverse(self, eta)
    }

    fn link_derivative(&self, mu: f64) -> f64 {
        TweedieFamily::link_derivative(self, mu)
    }

    fn unit_deviance(&self, y: f64, mu: f64) -> f64 {
        TweedieFamily::unit_deviance(self, y, mu)
    }

    fn initialize_mu(&self, y: &[f64]) -> Vec<f64> {
        TweedieFamily::initialize_mu(self, y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaussian_family() {
        let fam = TweedieFamily::gaussian();
        assert!((fam.var_power - 0.0).abs() < 1e-10);
        assert!((fam.link_power - 1.0).abs() < 1e-10);

        // Variance is constant
        assert!((fam.variance(1.0) - 1.0).abs() < 1e-10);
        assert!((fam.variance(5.0) - 1.0).abs() < 1e-10);

        // Identity link
        assert!((fam.link(3.0) - 3.0).abs() < 1e-10);
        assert!((fam.link_inverse(3.0) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_poisson_family() {
        let fam = TweedieFamily::poisson();
        assert!((fam.var_power - 1.0).abs() < 1e-10);

        // Variance = μ
        assert!((fam.variance(2.0) - 2.0).abs() < 1e-10);
        assert!((fam.variance(5.0) - 5.0).abs() < 1e-10);

        // Log link
        assert!((fam.link(std::f64::consts::E) - 1.0).abs() < 1e-10);
        assert!((fam.link_inverse(1.0) - std::f64::consts::E).abs() < 1e-10);
    }

    #[test]
    fn test_gamma_family() {
        let fam = TweedieFamily::gamma();
        assert!((fam.var_power - 2.0).abs() < 1e-10);

        // Variance = μ²
        assert!((fam.variance(2.0) - 4.0).abs() < 1e-10);
        assert!((fam.variance(3.0) - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_inverse_gaussian_family() {
        let fam = TweedieFamily::inverse_gaussian();
        assert!((fam.var_power - 3.0).abs() < 1e-10);

        // Variance = μ³
        assert!((fam.variance(2.0) - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_compound_poisson_gamma() {
        let fam = TweedieFamily::compound_poisson_gamma(1.5);
        assert!(fam.var_power > 1.0 && fam.var_power < 2.0);

        // Variance = μ^1.5
        let mu: f64 = 4.0;
        let expected_var = mu.powf(1.5);
        assert!((fam.variance(mu) - expected_var).abs() < 1e-10);
    }

    #[test]
    fn test_irls_weight() {
        let fam = TweedieFamily::poisson();
        let mu = 2.0;

        let weight = fam.irls_weight(mu);
        // For Poisson with log link: w = μ
        assert!((weight - mu).abs() < 1e-10);
    }

    #[test]
    fn test_working_response() {
        let fam = TweedieFamily::poisson();
        let y = 3.0;
        let mu = 2.0;
        let eta = fam.link(mu);

        let z = fam.working_response(y, mu, eta);
        // z = log(μ) + (y - μ) / μ
        let expected = eta + (y - mu) / mu;
        assert!((z - expected).abs() < 1e-10);
    }

    #[test]
    fn test_unit_deviance_normal() {
        let fam = TweedieFamily::gaussian();
        let y = 3.0;
        let mu = 2.0;
        let dev = fam.unit_deviance(y, mu);
        assert!((dev - 1.0).abs() < 1e-10); // (3-2)² = 1
    }

    #[test]
    fn test_unit_deviance_poisson() {
        let fam = TweedieFamily::poisson();
        let y = 3.0;
        let mu = 2.0;
        let dev = fam.unit_deviance(y, mu);
        let expected = 2.0 * (y * (y / mu).ln() - (y - mu));
        assert!((dev - expected).abs() < 1e-10);
    }

    #[test]
    fn test_deviance() {
        let fam = TweedieFamily::gaussian();
        let y = vec![1.0, 2.0, 3.0];
        let mu = vec![1.0, 2.0, 3.0];
        let dev = fam.deviance(&y, &mu);
        assert!(dev.abs() < 1e-10); // Perfect fit
    }

    #[test]
    fn test_canonical_link() {
        // Normal: canonical link_power = 1 - 0 = 1 (identity)
        let normal = TweedieFamily::gaussian();
        assert!(normal.is_canonical_link());

        // Poisson: canonical link_power = 1 - 1 = 0 (log)
        let poisson = TweedieFamily::poisson();
        assert!(poisson.is_canonical_link());

        // Gamma: canonical link_power = 1 - 2 = -1 (inverse)
        // Our gamma uses log link, so NOT canonical
        let gamma = TweedieFamily::gamma();
        assert!(!gamma.is_canonical_link());
    }

    #[test]
    #[should_panic(expected = "var_power in (0, 1) is not allowed")]
    fn test_invalid_var_power() {
        TweedieFamily::new(0.5, 0.0);
    }

    #[test]
    fn test_link_inverse_roundtrip() {
        let families = vec![
            TweedieFamily::gaussian(),
            TweedieFamily::poisson(),
            TweedieFamily::gamma(),
            TweedieFamily::inverse_gaussian(),
            TweedieFamily::compound_poisson_gamma(1.5),
        ];

        for fam in families {
            let mu = 2.5;
            let eta = fam.link(mu);
            let mu_back = fam.link_inverse(eta);
            assert!(
                (mu - mu_back).abs() < 1e-10,
                "Roundtrip failed for var_power={}",
                fam.var_power
            );
        }
    }
}
