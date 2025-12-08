//! GLM-specific residual types.
//!
//! Provides Pearson, deviance, and working residuals for generalized linear models.
//!
//! # Residual Types
//!
//! - **Response**: Raw residuals `(y - μ)`
//! - **Pearson**: `(y - μ) / sqrt(V(μ))`
//! - **Deviance**: `sign(y - μ) * sqrt(d_i)` where `d_i` is unit deviance
//! - **Working**: `(y - μ) * (dη/dμ)` - used in IRLS
//!
//! # Reference
//!
//! McCullagh, P. and Nelder, J.A. (1989). Generalized Linear Models, 2nd ed.

use crate::core::GlmFamily;
use faer::Col;

/// Compute response residuals: y - μ.
///
/// The simplest residual type, but variance is not constant.
pub fn response_residuals(y: &Col<f64>, mu: &Col<f64>) -> Col<f64> {
    let n = y.nrows();
    Col::from_fn(n, |i| y[i] - mu[i])
}

/// Compute Pearson residuals: (y - μ) / sqrt(V(μ)).
///
/// Standardized by the variance function, approximately homoscedastic.
pub fn pearson_residuals<F: GlmFamily>(y: &Col<f64>, mu: &Col<f64>, family: &F) -> Col<f64> {
    let n = y.nrows();
    Col::from_fn(n, |i| {
        let v = family.variance(mu[i]);
        if v < 1e-14 {
            0.0
        } else {
            (y[i] - mu[i]) / v.sqrt()
        }
    })
}

/// Compute deviance residuals: sign(y - μ) * sqrt(d_i).
///
/// Each residual contributes d_i² to the total deviance.
/// These residuals have better distributional properties than Pearson residuals.
pub fn deviance_residuals<F: GlmFamily>(y: &Col<f64>, mu: &Col<f64>, family: &F) -> Col<f64> {
    let n = y.nrows();
    Col::from_fn(n, |i| {
        let d_i = family.unit_deviance(y[i], mu[i]);
        let sign = if y[i] >= mu[i] { 1.0 } else { -1.0 };
        sign * d_i.sqrt()
    })
}

/// Compute working residuals: (y - μ) * (dη/dμ).
///
/// Used in the IRLS algorithm. Related to the working response by:
/// z = η + working_residual
pub fn working_residuals<F: GlmFamily>(y: &Col<f64>, mu: &Col<f64>, family: &F) -> Col<f64> {
    let n = y.nrows();
    Col::from_fn(n, |i| {
        let link_deriv = family.link_derivative(mu[i]);
        (y[i] - mu[i]) * link_deriv
    })
}

/// Compute standardized Pearson residuals: r_P / sqrt(φ * (1 - h_ii)).
///
/// Adjusts for both dispersion and leverage.
pub fn standardized_pearson_residuals<F: GlmFamily>(
    y: &Col<f64>,
    mu: &Col<f64>,
    family: &F,
    leverage: &Col<f64>,
    dispersion: f64,
) -> Col<f64> {
    let pearson = pearson_residuals(y, mu, family);
    let n = y.nrows();

    Col::from_fn(n, |i| {
        let h_ii = leverage[i];
        let scale = (dispersion * (1.0 - h_ii)).max(1e-14).sqrt();
        pearson[i] / scale
    })
}

/// Compute standardized deviance residuals: r_D / sqrt(φ * (1 - h_ii)).
///
/// Often preferred over standardized Pearson residuals.
pub fn standardized_deviance_residuals<F: GlmFamily>(
    y: &Col<f64>,
    mu: &Col<f64>,
    family: &F,
    leverage: &Col<f64>,
    dispersion: f64,
) -> Col<f64> {
    let deviance = deviance_residuals(y, mu, family);
    let n = y.nrows();

    Col::from_fn(n, |i| {
        let h_ii = leverage[i];
        let scale = (dispersion * (1.0 - h_ii)).max(1e-14).sqrt();
        deviance[i] / scale
    })
}

/// Compute Pearson's chi-squared statistic: Σ (y - μ)² / V(μ).
///
/// Used for dispersion estimation and goodness-of-fit testing.
pub fn pearson_chi_squared<F: GlmFamily>(y: &[f64], mu: &[f64], family: &F) -> f64 {
    y.iter()
        .zip(mu.iter())
        .map(|(&yi, &mui)| {
            let v = family.variance(mui);
            if v < 1e-14 {
                0.0
            } else {
                (yi - mui).powi(2) / v
            }
        })
        .sum()
}

/// Estimate dispersion parameter φ using Pearson's method.
///
/// φ̂ = X² / (n - p) where X² is Pearson's chi-squared.
pub fn estimate_dispersion_pearson<F: GlmFamily>(
    y: &[f64],
    mu: &[f64],
    family: &F,
    n_params: usize,
) -> f64 {
    let n = y.len();
    if n <= n_params {
        return 1.0;
    }

    let chi_sq = pearson_chi_squared(y, mu, family);
    let df = (n - n_params) as f64;

    chi_sq / df
}

/// Estimate dispersion parameter φ using deviance method.
///
/// φ̂ = D / (n - p) where D is the deviance.
pub fn estimate_dispersion_deviance<F: GlmFamily>(
    y: &[f64],
    mu: &[f64],
    family: &F,
    n_params: usize,
) -> f64 {
    let n = y.len();
    if n <= n_params {
        return 1.0;
    }

    let deviance = family.deviance(y, mu);
    let df = (n - n_params) as f64;

    deviance / df
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BinomialFamily, TweedieFamily};

    #[test]
    fn test_response_residuals() {
        let y = Col::from_fn(5, |i| i as f64);
        let mu = Col::from_fn(5, |i| (i as f64) + 0.5);

        let resid = response_residuals(&y, &mu);

        for i in 0..5 {
            assert!((resid[i] - (-0.5)).abs() < 1e-10);
        }
    }

    #[test]
    fn test_pearson_residuals_gaussian() {
        let fam = TweedieFamily::gaussian();
        let y = Col::from_fn(5, |i| i as f64);
        let mu = Col::from_fn(5, |i| (i as f64) + 0.5);

        let resid = pearson_residuals(&y, &mu, &fam);

        // For Gaussian, V(μ) = 1, so Pearson = response
        for i in 0..5 {
            assert!((resid[i] - (-0.5)).abs() < 1e-10);
        }
    }

    #[test]
    fn test_pearson_residuals_binomial() {
        let fam = BinomialFamily::logistic();
        let y = Col::from_fn(4, |i| if i < 2 { 0.0 } else { 1.0 });
        let mu = Col::from_fn(4, |_| 0.5);

        let resid = pearson_residuals(&y, &mu, &fam);

        // At μ = 0.5: V(μ) = 0.25, sqrt = 0.5
        // Pearson = (y - 0.5) / 0.5 = ±1
        for i in 0..4 {
            assert!((resid[i].abs() - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_deviance_residuals_gaussian() {
        let fam = TweedieFamily::gaussian();
        let y = Col::from_fn(5, |i| i as f64);
        let mu = Col::from_fn(5, |i| i as f64 + 0.5);

        let resid = deviance_residuals(&y, &mu, &fam);

        // For Gaussian, unit_deviance = (y-μ)², so deviance residual = y - μ
        for i in 0..5 {
            assert!((resid[i] - (-0.5)).abs() < 1e-10);
        }
    }

    #[test]
    fn test_working_residuals_gaussian() {
        let fam = TweedieFamily::gaussian();
        let y = Col::from_fn(5, |i| i as f64);
        let mu = Col::from_fn(5, |i| i as f64 + 0.5);

        let resid = working_residuals(&y, &mu, &fam);

        // For Gaussian with identity link, dη/dμ = 1, so working = response
        for i in 0..5 {
            assert!((resid[i] - (-0.5)).abs() < 1e-10);
        }
    }

    #[test]
    fn test_pearson_chi_squared() {
        let fam = TweedieFamily::gaussian();
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mu = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        // Perfect fit
        let chi_sq = pearson_chi_squared(&y, &mu, &fam);
        assert!(chi_sq < 1e-10);

        // With residuals
        let mu_off = vec![1.5, 2.5, 3.5, 4.5, 5.5];
        let chi_sq = pearson_chi_squared(&y, &mu_off, &fam);
        // Each contributes 0.25, total = 1.25
        assert!((chi_sq - 1.25).abs() < 1e-10);
    }

    #[test]
    fn test_estimate_dispersion() {
        let fam = TweedieFamily::gaussian();
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mu = vec![1.2, 2.1, 2.9, 4.0, 5.1];

        let phi = estimate_dispersion_pearson(&y, &mu, &fam, 2);
        // Should be positive
        assert!(phi > 0.0);
    }

    #[test]
    fn test_standardized_residuals() {
        let fam = TweedieFamily::gaussian();
        let y = Col::from_fn(5, |i| i as f64);
        let mu = Col::from_fn(5, |i| i as f64 + 0.5);
        let leverage = Col::from_fn(5, |_| 0.2);
        let dispersion = 1.0;

        let std_resid = standardized_pearson_residuals(&y, &mu, &fam, &leverage, dispersion);

        // Check that residuals are scaled
        let scale = (1.0 * 0.8_f64).sqrt();
        for i in 0..5 {
            assert!((std_resid[i] - (-0.5 / scale)).abs() < 1e-10);
        }
    }
}
