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
/// - Validity checks for μ and η (following R's glm.fit)
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

    /// Check if a mean value μ is valid for this family.
    ///
    /// This follows R's `validmu()` function in glm families.
    /// Returns `true` if μ is in the valid range for the family.
    ///
    /// # Examples
    ///
    /// - Poisson: μ > 0
    /// - Binomial: 0 < μ < 1
    /// - Gamma: μ > 0
    fn valid_mu(&self, mu: f64) -> bool {
        mu.is_finite() && mu > 0.0
    }

    /// Check if a linear predictor η is valid for this family.
    ///
    /// This follows R's `valideta()` function in glm families.
    /// Returns `true` if η is finite (default behavior).
    fn valid_eta(&self, eta: f64) -> bool {
        eta.is_finite()
    }

    /// Check validity of all μ values in a slice.
    ///
    /// Returns `true` if all values are valid.
    fn all_valid_mu(&self, mu: &[f64]) -> bool {
        mu.iter().all(|&m| self.valid_mu(m))
    }

    /// Check validity of all η values in a slice.
    ///
    /// Returns `true` if all values are valid.
    fn all_valid_eta(&self, eta: &[f64]) -> bool {
        eta.iter().all(|&e| self.valid_eta(e))
    }

    /// Clamp μ to valid range for this family.
    ///
    /// Default implementation clamps to (1e-10, 1e10).
    /// Override for family-specific bounds (e.g., Binomial: (ε, 1-ε)).
    fn clamp_mu(&self, mu: f64) -> f64 {
        mu.clamp(1e-10, 1e10)
    }

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

    /// Check if μ is valid for Tweedie family.
    ///
    /// - var_power = 0 (Gaussian): any finite μ
    /// - var_power > 0 (Poisson, Gamma, etc.): μ > 0
    fn valid_mu(&self, mu: f64) -> bool {
        if !mu.is_finite() {
            return false;
        }
        if self.var_power == 0.0 {
            true // Gaussian: any finite μ
        } else {
            mu > 0.0 // Poisson, Gamma, etc.: μ > 0
        }
    }

    /// Clamp μ to valid range for Tweedie family.
    fn clamp_mu(&self, mu: f64) -> f64 {
        if self.var_power == 0.0 {
            mu.clamp(-1e10, 1e10) // Gaussian: any value
        } else {
            mu.clamp(1e-10, 1e10) // Poisson, Gamma: μ > 0
        }
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

    // === Tests for variance_derivative ===

    #[test]
    fn test_variance_derivative_normal() {
        // Tests line 196: var_power == 0 (Normal)
        let fam = TweedieFamily::gaussian();

        // dV/dμ = 0 for Normal (constant variance)
        assert!((fam.variance_derivative(1.0) - 0.0).abs() < 1e-10);
        assert!((fam.variance_derivative(5.0) - 0.0).abs() < 1e-10);
        assert!((fam.variance_derivative(100.0) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_variance_derivative_poisson() {
        // Tests line 198: var_power == 1 (Poisson)
        let fam = TweedieFamily::poisson();

        // dV/dμ = 1 for Poisson
        assert!((fam.variance_derivative(0.5) - 1.0).abs() < 1e-10);
        assert!((fam.variance_derivative(2.0) - 1.0).abs() < 1e-10);
        assert!((fam.variance_derivative(10.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_variance_derivative_gamma() {
        // Tests line 200: var_power == 2 (Gamma)
        let fam = TweedieFamily::gamma();

        // dV/dμ = 2μ for Gamma
        assert!((fam.variance_derivative(1.0) - 2.0).abs() < 1e-10);
        assert!((fam.variance_derivative(3.0) - 6.0).abs() < 1e-10);
        assert!((fam.variance_derivative(5.0) - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_variance_derivative_general() {
        // Tests line 202: general case
        let fam = TweedieFamily::inverse_gaussian(); // var_power = 3
        let mu: f64 = 2.0;

        // dV/dμ = p * μ^(p-1) = 3 * 2^2 = 12
        let expected = 3.0 * mu.powf(2.0);
        assert!((fam.variance_derivative(mu) - expected).abs() < 1e-10);

        // Compound Poisson-Gamma (p = 1.5)
        let fam2 = TweedieFamily::compound_poisson_gamma(1.5);
        let expected2 = 1.5 * mu.powf(0.5);
        assert!((fam2.variance_derivative(mu) - expected2).abs() < 1e-10);
    }

    // === Tests for link_inverse_derivative ===

    #[test]
    fn test_link_inverse_derivative_log() {
        // Tests line 249: link_power == 0 (log link)
        let fam = TweedieFamily::poisson(); // log link
        let eta: f64 = 2.0;

        // d/dη exp(η) = exp(η)
        let expected = eta.exp();
        assert!((fam.link_inverse_derivative(eta) - expected).abs() < 1e-10);
    }

    #[test]
    fn test_link_inverse_derivative_identity() {
        // Tests line 251: link_power == 1 (identity link)
        let fam = TweedieFamily::gaussian(); // identity link

        // d/dη η = 1
        assert!((fam.link_inverse_derivative(5.0) - 1.0).abs() < 1e-10);
        assert!((fam.link_inverse_derivative(-3.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_link_inverse_derivative_inverse() {
        // Tests line 253: link_power == -1 (inverse link)
        let fam = TweedieFamily::new(2.0, -1.0); // Gamma with inverse link
        let eta = 2.0;

        // d/dη (1/η) = -1/η²
        let expected = -1.0 / (eta * eta);
        assert!((fam.link_inverse_derivative(eta) - expected).abs() < 1e-10);
    }

    #[test]
    fn test_link_inverse_derivative_general() {
        // Tests line 255: general power link
        let fam = TweedieFamily::new(1.0, 0.5); // Power link with q=0.5
        let eta: f64 = 4.0;

        // d/dη η^(1/q) = (1/q) * η^(1/q - 1) = 2 * 4^1 = 8
        let expected = (1.0 / 0.5) * eta.powf(1.0 / 0.5 - 1.0);
        assert!((fam.link_inverse_derivative(eta) - expected).abs() < 1e-10);
    }

    // === Tests for null_deviance ===

    #[test]
    fn test_null_deviance_normal() {
        // Tests lines 360-363
        let fam = TweedieFamily::gaussian();
        let y = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let y_mean: f64 = 3.0;

        // Null deviance = sum((y - y_mean)^2) = 10
        let expected: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();
        let null_dev = fam.null_deviance(&y);

        assert!((null_dev - expected).abs() < 1e-10);
    }

    #[test]
    fn test_null_deviance_poisson() {
        let fam = TweedieFamily::poisson();
        let y = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let y_mean: f64 = 3.0;

        // Null deviance = sum(2 * (y * log(y/y_mean) - (y - y_mean)))
        let expected: f64 = y
            .iter()
            .map(|&yi| {
                if yi > 0.0 {
                    2.0 * (yi * (yi / y_mean).ln() - (yi - y_mean))
                } else {
                    2.0 * y_mean
                }
            })
            .sum();

        let null_dev = fam.null_deviance(&y);
        assert!((null_dev - expected).abs() < 1e-10);
    }

    #[test]
    fn test_null_deviance_gamma() {
        let fam = TweedieFamily::gamma();
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        // Gamma deviance = 2 * sum(-log(y/mu) + (y-mu)/mu)
        let null_dev = fam.null_deviance(&y);

        assert!(null_dev.is_finite());
        assert!(null_dev >= 0.0);
    }

    // === Tests for canonical_link_power ===

    #[test]
    fn test_canonical_link_power() {
        // Tests lines 407-409
        // Canonical link power = 1 - var_power

        let normal = TweedieFamily::gaussian();
        assert!((normal.canonical_link_power() - 1.0).abs() < 1e-10); // 1 - 0 = 1

        let poisson = TweedieFamily::poisson();
        assert!((poisson.canonical_link_power() - 0.0).abs() < 1e-10); // 1 - 1 = 0

        let gamma = TweedieFamily::gamma();
        assert!((gamma.canonical_link_power() - (-1.0)).abs() < 1e-10); // 1 - 2 = -1

        let ig = TweedieFamily::inverse_gaussian();
        assert!((ig.canonical_link_power() - (-2.0)).abs() < 1e-10); // 1 - 3 = -2
    }

    // === Tests for is_valid ===

    #[test]
    fn test_is_valid() {
        // Tests lines 399-402

        // Valid cases
        assert!(TweedieFamily::gaussian().is_valid());
        assert!(TweedieFamily::poisson().is_valid());
        assert!(TweedieFamily::gamma().is_valid());
        assert!(TweedieFamily::new(0.0, 1.0).is_valid()); // p = 0
        assert!(TweedieFamily::new(1.0, 0.0).is_valid()); // p = 1
        assert!(TweedieFamily::new(1.5, 0.0).is_valid()); // p in (1, 2)
        assert!(TweedieFamily::new(3.0, 0.0).is_valid()); // p > 2
        assert!(TweedieFamily::new(-1.0, 1.0).is_valid()); // p < 0
    }

    // === Tests for unit_deviance edge cases ===

    #[test]
    fn test_unit_deviance_gamma_case() {
        // Tests lines 329-331: Gamma specific formula
        let fam = TweedieFamily::gamma();
        let y: f64 = 2.0;
        let mu: f64 = 3.0;

        // d(y, μ) = 2 * (-log(y/μ) + (y-μ)/μ)
        let expected = 2.0 * (-(y / mu).ln() + (y - mu) / mu);
        let dev = fam.unit_deviance(y, mu);

        assert!((dev - expected).abs() < 1e-10);
    }

    #[test]
    fn test_unit_deviance_general_tweedie() {
        // Tests lines 333-348: general Tweedie formula
        let fam = TweedieFamily::inverse_gaussian(); // p = 3
        let y: f64 = 2.0;
        let mu: f64 = 1.5;
        let p: f64 = 3.0;

        let term1 = y.powf(2.0 - p) / ((1.0 - p) * (2.0 - p));
        let term2 = -y * mu.powf(1.0 - p) / (1.0 - p);
        let term3 = mu.powf(2.0 - p) / (2.0 - p);
        let expected = 2.0 * (term1 + term2 + term3);

        let dev = fam.unit_deviance(y, mu);
        assert!((dev - expected).abs() < 1e-10);
    }

    #[test]
    fn test_unit_deviance_zero_y_poisson() {
        // Tests y = 0 case in Poisson
        let fam = TweedieFamily::poisson();
        let dev = fam.unit_deviance(0.0, 2.0);

        // When y = 0: d(0, μ) = 2μ
        assert!((dev - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_unit_deviance_zero_y_general_tweedie() {
        // Tests y = 0 case in general Tweedie
        let fam = TweedieFamily::compound_poisson_gamma(1.5);
        let mu = 2.0;
        let p = 1.5;

        let dev = fam.unit_deviance(0.0, mu);

        // When y = 0: only term3 contributes
        let expected = 2.0 * mu.powf(2.0 - p) / (2.0 - p);
        assert!((dev - expected).abs() < 1e-10);
    }

    // === Tests for IRLS methods with different families ===

    #[test]
    fn test_irls_weight_gaussian() {
        let fam = TweedieFamily::gaussian();
        let mu = 5.0;

        // For Gaussian with identity link: w = 1 / (V(μ) * (dη/dμ)²) = 1 / (1 * 1²) = 1
        let weight = fam.irls_weight(mu);
        assert!((weight - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_irls_weight_gamma() {
        let fam = TweedieFamily::gamma(); // log link
        let mu = 2.0;

        // V(μ) = μ² = 4, dη/dμ = 1/μ = 0.5
        // w = 1 / (4 * 0.25) = 1
        let weight = fam.irls_weight(mu);
        assert!((weight - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_irls_weight_inverse_gaussian() {
        let fam = TweedieFamily::inverse_gaussian(); // log link
        let mu = 2.0;

        // V(μ) = μ³ = 8, dη/dμ = 1/μ = 0.5
        // w = 1 / (8 * 0.25) = 0.5
        let weight = fam.irls_weight(mu);
        assert!((weight - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_working_response_gaussian() {
        let fam = TweedieFamily::gaussian();
        let y = 5.0;
        let mu = 3.0;
        let eta = fam.link(mu); // = 3.0 (identity)

        // z = η + (y - μ) * dη/dμ = 3 + (5-3) * 1 = 5
        let z = fam.working_response(y, mu, eta);
        assert!((z - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_working_response_gamma() {
        let fam = TweedieFamily::gamma();
        let y = 4.0;
        let mu = 2.0;
        let eta = fam.link(mu); // = log(2)

        // z = log(2) + (4-2) * (1/2) = log(2) + 1
        let expected = mu.ln() + (y - mu) / mu;
        let z = fam.working_response(y, mu, eta);
        assert!((z - expected).abs() < 1e-10);
    }

    // === Tests for link_derivative branches ===

    #[test]
    fn test_link_derivative_all_branches() {
        let mu = 2.0;

        // Log link (link_power = 0)
        let log_link = TweedieFamily::poisson();
        assert!((log_link.link_derivative(mu) - 0.5).abs() < 1e-10); // 1/μ

        // Identity link (link_power = 1)
        let id_link = TweedieFamily::gaussian();
        assert!((id_link.link_derivative(mu) - 1.0).abs() < 1e-10);

        // Inverse link (link_power = -1)
        let inv_link = TweedieFamily::new(2.0, -1.0);
        assert!((inv_link.link_derivative(mu) - (-0.25)).abs() < 1e-10); // -1/μ²

        // General power link (link_power = 0.5)
        let pow_link = TweedieFamily::new(1.0, 0.5);
        let expected = 0.5 * mu.powf(-0.5); // q * μ^(q-1)
        assert!((pow_link.link_derivative(mu) - expected).abs() < 1e-10);
    }

    // === Tests for link function branches ===

    #[test]
    fn test_link_all_branches() {
        let mu = 2.0;

        // Log link (link_power = 0)
        let log_link = TweedieFamily::poisson();
        assert!((log_link.link(mu) - mu.ln()).abs() < 1e-10);

        // Identity link (link_power = 1)
        let id_link = TweedieFamily::gaussian();
        assert!((id_link.link(mu) - mu).abs() < 1e-10);

        // Inverse link (link_power = -1)
        let inv_link = TweedieFamily::new(2.0, -1.0);
        assert!((inv_link.link(mu) - 0.5).abs() < 1e-10); // 1/μ

        // General power link (link_power = 0.5)
        let pow_link = TweedieFamily::new(1.0, 0.5);
        assert!((pow_link.link(mu) - mu.powf(0.5)).abs() < 1e-10);
    }

    // === Tests for link_inverse branches ===

    #[test]
    fn test_link_inverse_all_branches() {
        let eta = 2.0;

        // Log link (link_power = 0)
        let log_link = TweedieFamily::poisson();
        assert!((log_link.link_inverse(eta) - eta.exp()).abs() < 1e-10);

        // Identity link (link_power = 1)
        let id_link = TweedieFamily::gaussian();
        assert!((id_link.link_inverse(eta) - eta).abs() < 1e-10);

        // Inverse link (link_power = -1)
        let inv_link = TweedieFamily::new(2.0, -1.0);
        assert!((inv_link.link_inverse(eta) - 0.5).abs() < 1e-10); // 1/η

        // General power link (link_power = 0.5)
        let pow_link = TweedieFamily::new(1.0, 0.5);
        assert!((pow_link.link_inverse(eta) - eta.powf(2.0)).abs() < 1e-10); // η^(1/0.5)
    }

    // === Tests for initialize_mu ===

    #[test]
    fn test_initialize_mu_normal() {
        let fam = TweedieFamily::gaussian();
        let y = vec![1.0, 2.0, 3.0, -1.0, 5.0];

        let mu = fam.initialize_mu(&y);

        // For normal, mu should be y
        assert_eq!(mu.len(), y.len());
        for i in 0..y.len() {
            assert!((mu[i] - y[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_initialize_mu_poisson() {
        let fam = TweedieFamily::poisson();
        let y = vec![0.0, 1.0, 2.0, 3.0, 4.0];

        let mu = fam.initialize_mu(&y);

        // For Poisson, mu should be positive
        assert_eq!(mu.len(), y.len());
        for (i, &mu_i) in mu.iter().enumerate() {
            assert!(mu_i > 0.0, "mu[{}] should be positive", i);
        }

        // First observation has y=0, mu should be (0 + 2)/2 = 1.0
        assert!((mu[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_initialize_mu_with_negative() {
        let fam = TweedieFamily::poisson();
        let y = vec![-1.0, -2.0, 0.0, 1.0, 2.0];

        let mu = fam.initialize_mu(&y);

        // All mu values should be positive for Poisson
        for (i, &mu_i) in mu.iter().enumerate() {
            assert!(mu_i > 0.0, "mu[{}] = {} should be positive", i, mu_i);
        }
    }

    // === Tests for variance branches ===

    #[test]
    fn test_variance_all_branches() {
        let mu = 2.0;

        // Normal (var_power = 0)
        let normal = TweedieFamily::gaussian();
        assert!((normal.variance(mu) - 1.0).abs() < 1e-10);

        // Poisson (var_power = 1)
        let poisson = TweedieFamily::poisson();
        assert!((poisson.variance(mu) - mu).abs() < 1e-10);

        // Gamma (var_power = 2)
        let gamma = TweedieFamily::gamma();
        assert!((gamma.variance(mu) - mu * mu).abs() < 1e-10);

        // General (var_power = 3)
        let ig = TweedieFamily::inverse_gaussian();
        assert!((ig.variance(mu) - mu.powf(3.0)).abs() < 1e-10);
    }

    // === Test for GlmFamily trait implementation ===

    #[test]
    fn test_glm_family_trait() {
        let fam = TweedieFamily::poisson();

        // Test trait methods work correctly
        let mu = 2.0;
        assert!((GlmFamily::variance(&fam, mu) - fam.variance(mu)).abs() < 1e-10);
        assert!((GlmFamily::link(&fam, mu) - fam.link(mu)).abs() < 1e-10);
        assert!((GlmFamily::link_inverse(&fam, fam.link(mu)) - mu).abs() < 1e-10);
        assert!((GlmFamily::link_derivative(&fam, mu) - fam.link_derivative(mu)).abs() < 1e-10);
    }

    // === Test for deviance with different families ===

    #[test]
    fn test_deviance_misfit() {
        let fam = TweedieFamily::gaussian();
        let y = vec![1.0, 2.0, 3.0];
        let mu = vec![2.0, 2.0, 2.0]; // Not a perfect fit

        let dev = fam.deviance(&y, &mu);
        // (1-2)² + (2-2)² + (3-2)² = 1 + 0 + 1 = 2
        assert!((dev - 2.0).abs() < 1e-10);
    }
}
