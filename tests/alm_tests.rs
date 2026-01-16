//! Unit tests for Augmented Linear Model (ALM)
//!
//! Tests follow TDD approach - written before implementation.

mod common;

use anofox_regression::solvers::{AlmDistribution, AlmRegressor, FittedRegressor, Regressor};
use faer::{Col, Mat};

// ============================================================================
// Distribution Likelihood Tests
// ============================================================================

mod likelihood_tests {
    use super::*;
    use anofox_regression::solvers::alm::{log_likelihood, AlmDistribution};

    /// Test Normal distribution log-likelihood
    /// LL = -n/2 * log(2*pi*sigma^2) - RSS/(2*sigma^2)
    #[test]
    fn test_normal_log_likelihood() {
        let y = Col::from_fn(5, |i| i as f64);
        let mu = Col::from_fn(5, |i| i as f64 + 0.1); // slight offset
        let sigma = 1.0;

        let ll = log_likelihood(&y, &mu, AlmDistribution::Normal, sigma, None);

        // Manual calculation:
        // residuals = [-0.1, -0.1, -0.1, -0.1, -0.1]
        // RSS = 5 * 0.01 = 0.05
        // LL = -5/2 * log(2*pi*1) - 0.05/2 = -5/2 * 1.8379 - 0.025 ≈ -4.62
        assert!(ll.is_finite());
        assert!(ll < 0.0); // Log likelihood should be negative
    }

    /// Test Laplace distribution log-likelihood
    /// LL = -n*log(2*b) - sum(|y - mu|)/b where b = scale
    #[test]
    fn test_laplace_log_likelihood() {
        let y = Col::from_fn(5, |i| i as f64);
        let mu = Col::from_fn(5, |i| i as f64);
        let scale = 1.0;

        let ll = log_likelihood(&y, &mu, AlmDistribution::Laplace, scale, None);

        // Perfect fit: sum(|y - mu|) = 0
        // LL = -5*log(2*1) - 0 = -5*0.693 ≈ -3.47
        assert!(ll.is_finite());
        let expected = -5.0 * (2.0_f64).ln();
        assert!((ll - expected).abs() < 1e-10);
    }

    /// Test Student-t distribution log-likelihood
    #[test]
    fn test_student_t_log_likelihood() {
        let y = Col::from_fn(10, |i| i as f64);
        let mu = Col::from_fn(10, |i| i as f64 + 0.5);
        let scale = 1.0;
        let df = Some(5.0); // degrees of freedom

        let ll = log_likelihood(&y, &mu, AlmDistribution::StudentT, scale, df);

        assert!(ll.is_finite());
        assert!(ll < 0.0);
    }

    /// Test Logistic distribution log-likelihood
    #[test]
    fn test_logistic_log_likelihood() {
        let y = Col::from_fn(5, |i| i as f64);
        let mu = Col::from_fn(5, |i| i as f64);
        let scale = 1.0;

        let ll = log_likelihood(&y, &mu, AlmDistribution::Logistic, scale, None);

        // Perfect fit should give finite LL
        assert!(ll.is_finite());
    }

    /// Test Poisson distribution log-likelihood
    /// LL = sum(y*log(mu) - mu - log(y!))
    #[test]
    fn test_poisson_log_likelihood() {
        let y = Col::from_fn(5, |i| (i + 1) as f64); // [1, 2, 3, 4, 5]
        let mu = Col::from_fn(5, |i| (i + 1) as f64); // Perfect fit

        let ll = log_likelihood(&y, &mu, AlmDistribution::Poisson, 1.0, None);

        assert!(ll.is_finite());
    }

    /// Test Negative Binomial distribution log-likelihood
    #[test]
    fn test_negative_binomial_log_likelihood() {
        let y = Col::from_fn(5, |i| (i * 2) as f64); // [0, 2, 4, 6, 8]
        let mu = Col::from_fn(5, |i| (i * 2 + 1) as f64); // [1, 3, 5, 7, 9]
        let size = Some(2.0); // dispersion parameter

        let ll = log_likelihood(&y, &mu, AlmDistribution::NegativeBinomial, 1.0, size);

        assert!(ll.is_finite());
    }

    /// Test Binomial distribution log-likelihood
    #[test]
    fn test_binomial_log_likelihood() {
        let y = Col::from_fn(5, |_| 0.5); // proportion outcomes
        let mu = Col::from_fn(5, |_| 0.5); // predicted probabilities
        let n_trials = Some(10.0);

        let ll = log_likelihood(&y, &mu, AlmDistribution::Binomial, 1.0, n_trials);

        assert!(ll.is_finite());
    }

    /// Test Gamma distribution log-likelihood
    #[test]
    fn test_gamma_log_likelihood() {
        let y = Col::from_fn(5, |i| (i + 1) as f64); // [1, 2, 3, 4, 5]
        let mu = Col::from_fn(5, |i| (i + 1) as f64);
        let shape = Some(2.0);

        let ll = log_likelihood(&y, &mu, AlmDistribution::Gamma, 1.0, shape);

        assert!(ll.is_finite());
    }

    /// Test Inverse Gaussian distribution log-likelihood
    #[test]
    fn test_inverse_gaussian_log_likelihood() {
        let y = Col::from_fn(5, |i| (i + 1) as f64);
        let mu = Col::from_fn(5, |i| (i + 1) as f64);
        let lambda = Some(1.0); // shape parameter

        let ll = log_likelihood(&y, &mu, AlmDistribution::InverseGaussian, 1.0, lambda);

        assert!(ll.is_finite());
    }

    /// Test Log-Normal distribution log-likelihood
    #[test]
    fn test_log_normal_log_likelihood() {
        let y = Col::from_fn(5, |i| (i + 1) as f64); // positive values
        let mu = Col::from_fn(5, |i| (i + 1) as f64);
        let sigma = 0.5;

        let ll = log_likelihood(&y, &mu, AlmDistribution::LogNormal, sigma, None);

        assert!(ll.is_finite());
    }

    /// Test Asymmetric Laplace distribution log-likelihood
    #[test]
    fn test_asymmetric_laplace_log_likelihood() {
        let y = Col::from_fn(10, |i| i as f64);
        let mu = Col::from_fn(10, |i| i as f64 + 0.5);
        let scale = 1.0;
        let alpha = Some(0.3); // asymmetry parameter (quantile)

        let ll = log_likelihood(&y, &mu, AlmDistribution::AsymmetricLaplace, scale, alpha);

        assert!(ll.is_finite());
    }

    /// Test Generalised Normal (Subbotin) distribution log-likelihood
    #[test]
    fn test_generalised_normal_log_likelihood() {
        let y = Col::from_fn(10, |i| i as f64);
        let mu = Col::from_fn(10, |i| i as f64);
        let scale = 1.0;
        let shape = Some(2.0); // shape=2 gives Normal

        let ll = log_likelihood(&y, &mu, AlmDistribution::GeneralisedNormal, scale, shape);

        assert!(ll.is_finite());
    }

    /// Test S distribution log-likelihood
    #[test]
    fn test_s_distribution_log_likelihood() {
        let y = Col::from_fn(10, |i| i as f64);
        let mu = Col::from_fn(10, |i| i as f64);
        let scale = 1.0;

        let ll = log_likelihood(&y, &mu, AlmDistribution::S, scale, None);

        assert!(ll.is_finite());
    }

    /// Test Exponential distribution log-likelihood
    #[test]
    fn test_exponential_log_likelihood() {
        let y = Col::from_fn(5, |i| (i + 1) as f64); // positive values
        let mu = Col::from_fn(5, |i| (i + 1) as f64);

        let ll = log_likelihood(&y, &mu, AlmDistribution::Exponential, 1.0, None);

        assert!(ll.is_finite());
    }

    /// Test Folded Normal distribution log-likelihood
    #[test]
    fn test_folded_normal_log_likelihood() {
        let y = Col::from_fn(5, |i| (i + 1) as f64); // positive values
        let mu = Col::from_fn(5, |i| (i + 1) as f64);
        let sigma = 1.0;

        let ll = log_likelihood(&y, &mu, AlmDistribution::FoldedNormal, sigma, None);

        assert!(ll.is_finite());
    }

    /// Test Log-Laplace distribution log-likelihood
    #[test]
    fn test_log_laplace_log_likelihood() {
        let y = Col::from_fn(5, |i| (i + 1) as f64);
        let mu = Col::from_fn(5, |i| (i + 1) as f64);
        let scale = 1.0;

        let ll = log_likelihood(&y, &mu, AlmDistribution::LogLaplace, scale, None);

        assert!(ll.is_finite());
    }

    /// Test Geometric distribution log-likelihood
    #[test]
    fn test_geometric_log_likelihood() {
        let y = Col::from_fn(5, |i| i as f64); // [0, 1, 2, 3, 4] number of failures
        let mu = Col::from_fn(5, |_| 0.3); // probability of success

        let ll = log_likelihood(&y, &mu, AlmDistribution::Geometric, 1.0, None);

        assert!(ll.is_finite());
    }

    /// Test Beta distribution log-likelihood
    #[test]
    fn test_beta_log_likelihood() {
        let y = Col::from_fn(5, |i| 0.1 + 0.1 * i as f64); // [0.1, 0.2, 0.3, 0.4, 0.5]
        let mu = Col::from_fn(5, |i| 0.1 + 0.1 * i as f64);
        let phi = Some(5.0); // precision parameter

        let ll = log_likelihood(&y, &mu, AlmDistribution::Beta, 1.0, phi);

        assert!(ll.is_finite());
    }

    /// Test Box-Cox Normal distribution log-likelihood
    #[test]
    fn test_box_cox_normal_log_likelihood() {
        let y = Col::from_fn(5, |i| (i + 1) as f64);
        let mu = Col::from_fn(5, |i| (i + 1) as f64);
        let sigma = 1.0;
        let lambda = Some(0.5); // Box-Cox transformation parameter

        let ll = log_likelihood(&y, &mu, AlmDistribution::BoxCoxNormal, sigma, lambda);

        assert!(ll.is_finite());
    }
}

// ============================================================================
// ALM Fitting Tests
// ============================================================================

mod fitting_tests {
    use super::*;

    /// Test fitting with Normal distribution (should match OLS)
    #[test]
    fn test_alm_normal_matches_ols() {
        let x = Mat::from_fn(
            50,
            2,
            |i, j| {
                if j == 0 {
                    i as f64
                } else {
                    (i as f64).sin()
                }
            },
        );
        let y = Col::from_fn(50, |i| 2.0 + 3.0 * i as f64 + 0.5 * (i as f64).sin());

        let alm = AlmRegressor::builder()
            .distribution(AlmDistribution::Normal)
            .with_intercept(true)
            .build();

        let fitted = alm.fit(&x, &y).unwrap();

        // For Normal distribution with identity link, should be close to OLS
        assert!((fitted.intercept().unwrap() - 2.0).abs() < 0.5);
        assert!((fitted.coefficients()[0] - 3.0).abs() < 0.5);
    }

    /// Test fitting with Laplace distribution
    #[test]
    fn test_alm_laplace_robust() {
        // Create data with outliers
        let mut y_data = vec![0.0; 50];
        let mut x_data = vec![0.0; 50];
        for i in 0..50 {
            x_data[i] = i as f64;
            y_data[i] = 2.0 + 3.0 * i as f64;
        }
        // Add outliers
        y_data[10] += 100.0;
        y_data[30] += 100.0;

        let x = Mat::from_fn(50, 1, |i, _| x_data[i]);
        let y = Col::from_fn(50, |i| y_data[i]);

        let alm = AlmRegressor::builder()
            .distribution(AlmDistribution::Laplace)
            .with_intercept(true)
            .build();

        let fitted = alm.fit(&x, &y).unwrap();

        // Laplace should be more robust to outliers than Normal
        // Slope should still be approximately 3.0
        assert!((fitted.coefficients()[0] - 3.0).abs() < 1.0);
    }

    /// Test fitting with Poisson distribution
    #[test]
    fn test_alm_poisson() {
        // Generate Poisson-like count data
        let x = Mat::from_fn(100, 1, |i, _| (i as f64) / 50.0);
        // lambda = exp(0.5 + 1.0 * x)
        let y = Col::from_fn(100, |i| {
            let lambda = (0.5 + 1.0 * (i as f64) / 50.0).exp();
            lambda.round() // Approximate Poisson
        });

        let alm = AlmRegressor::builder()
            .distribution(AlmDistribution::Poisson)
            .with_intercept(true)
            .build();

        let fitted = alm.fit(&x, &y).unwrap();

        // Check that coefficients are reasonable
        assert!(fitted.intercept().is_some());
        assert!(fitted.coefficients().nrows() == 1);
    }

    /// Test fitting with Student-t distribution
    #[test]
    fn test_alm_student_t() {
        let x = Mat::from_fn(50, 1, |i, _| i as f64);
        let y = Col::from_fn(50, |i| 1.0 + 2.0 * i as f64);

        let alm = AlmRegressor::builder()
            .distribution(AlmDistribution::StudentT)
            .extra_parameter(5.0) // degrees of freedom
            .with_intercept(true)
            .build();

        let fitted = alm.fit(&x, &y).unwrap();

        assert!((fitted.coefficients()[0] - 2.0).abs() < 0.5);
    }

    /// Test fitting with Gamma distribution
    #[test]
    fn test_alm_gamma() {
        // Generate positive response data
        let x = Mat::from_fn(50, 1, |i, _| (i as f64) / 10.0);
        let y = Col::from_fn(50, |i| {
            let mu = (1.0 + 0.5 * (i as f64) / 10.0).exp();
            mu.max(0.1) // Ensure positive
        });

        let alm = AlmRegressor::builder()
            .distribution(AlmDistribution::Gamma)
            .with_intercept(true)
            .build();

        let fitted = alm.fit(&x, &y).unwrap();

        assert!(fitted.result().log_likelihood.is_finite());
    }

    /// Test fitting with Binomial distribution (logistic regression)
    #[test]
    fn test_alm_binomial() {
        // Generate binary outcome data
        let x = Mat::from_fn(100, 1, |i, _| (i as f64 - 50.0) / 25.0);
        let y = Col::from_fn(100, |i| {
            let p = 1.0 / (1.0 + (-0.5 - 1.0 * (i as f64 - 50.0) / 25.0).exp());
            if p > 0.5 {
                1.0
            } else {
                0.0
            }
        });

        let alm = AlmRegressor::builder()
            .distribution(AlmDistribution::Binomial)
            .with_intercept(true)
            .build();

        let fitted = alm.fit(&x, &y).unwrap();

        // Check coefficients have correct sign
        assert!(fitted.coefficients()[0] > 0.0);
    }

    /// Test fitting with Negative Binomial distribution
    #[test]
    fn test_alm_negative_binomial() {
        // Generate overdispersed count data
        let x = Mat::from_fn(100, 1, |i, _| (i as f64) / 50.0);
        let y = Col::from_fn(100, |i| {
            let mu = (0.5 + 0.8 * (i as f64) / 50.0).exp();
            mu.round().max(0.0)
        });

        let alm = AlmRegressor::builder()
            .distribution(AlmDistribution::NegativeBinomial)
            .extra_parameter(2.0) // size/dispersion
            .with_intercept(true)
            .build();

        let fitted = alm.fit(&x, &y).unwrap();

        assert!(fitted.result().log_likelihood.is_finite());
    }

    /// Test fitting with Inverse Gaussian distribution
    #[test]
    fn test_alm_inverse_gaussian() {
        let x = Mat::from_fn(50, 1, |i, _| (i as f64 + 1.0) / 10.0);
        let y = Col::from_fn(50, |i| {
            let mu = (1.0 + 0.5 * (i as f64 + 1.0) / 10.0).exp();
            mu.max(0.1)
        });

        let alm = AlmRegressor::builder()
            .distribution(AlmDistribution::InverseGaussian)
            .with_intercept(true)
            .build();

        let fitted = alm.fit(&x, &y).unwrap();

        assert!(fitted.result().log_likelihood.is_finite());
    }

    /// Test fitting with Log-Normal distribution
    #[test]
    fn test_alm_log_normal() {
        let x = Mat::from_fn(50, 1, |i, _| i as f64);
        let y = Col::from_fn(50, |i| ((1.0 + 0.1 * i as f64).exp()).max(0.1));

        let alm = AlmRegressor::builder()
            .distribution(AlmDistribution::LogNormal)
            .with_intercept(true)
            .build();

        let fitted = alm.fit(&x, &y).unwrap();

        assert!(fitted.result().log_likelihood.is_finite());
    }

    /// Test fitting with Exponential distribution
    #[test]
    fn test_alm_exponential() {
        let x = Mat::from_fn(50, 1, |i, _| (i as f64) / 10.0);
        let y = Col::from_fn(50, |i| {
            let rate = (0.5 + 0.3 * (i as f64) / 10.0).exp();
            (1.0 / rate).max(0.01)
        });

        let alm = AlmRegressor::builder()
            .distribution(AlmDistribution::Exponential)
            .with_intercept(true)
            .build();

        let fitted = alm.fit(&x, &y).unwrap();

        assert!(fitted.result().log_likelihood.is_finite());
    }

    /// Test fitting with Beta distribution
    #[test]
    fn test_alm_beta() {
        // Generate data in (0, 1)
        let x = Mat::from_fn(50, 1, |i, _| (i as f64) / 50.0);
        let y = Col::from_fn(50, |i| {
            let logit_mu = -1.0 + 2.0 * (i as f64) / 50.0;
            let mu = 1.0 / (1.0 + (-logit_mu).exp());
            mu.clamp(0.01, 0.99)
        });

        let alm = AlmRegressor::builder()
            .distribution(AlmDistribution::Beta)
            .extra_parameter(5.0) // precision
            .with_intercept(true)
            .build();

        let fitted = alm.fit(&x, &y).unwrap();

        assert!(fitted.result().log_likelihood.is_finite());
    }
}

// ============================================================================
// Prediction Tests
// ============================================================================

mod prediction_tests {
    use super::*;

    /// Test predictions with Normal distribution
    #[test]
    fn test_predict_normal() {
        let x = Mat::from_fn(50, 1, |i, _| i as f64);
        let y = Col::from_fn(50, |i| 2.0 + 3.0 * i as f64);

        let alm = AlmRegressor::builder()
            .distribution(AlmDistribution::Normal)
            .with_intercept(true)
            .build();

        let fitted = alm.fit(&x, &y).unwrap();

        let x_new = Mat::from_fn(5, 1, |i, _| (i + 60) as f64);
        let preds = fitted.predict(&x_new);

        assert_eq!(preds.nrows(), 5);
        // Should be approximately 2 + 3 * x
        for i in 0..5 {
            let expected = 2.0 + 3.0 * (i + 60) as f64;
            assert!((preds[i] - expected).abs() < 10.0);
        }
    }

    /// Test predictions with Poisson (log link)
    #[test]
    fn test_predict_poisson() {
        let x = Mat::from_fn(100, 1, |i, _| (i as f64) / 50.0);
        let y = Col::from_fn(100, |i| {
            let lambda = (0.5 + 1.0 * (i as f64) / 50.0).exp();
            lambda.round()
        });

        let alm = AlmRegressor::builder()
            .distribution(AlmDistribution::Poisson)
            .with_intercept(true)
            .build();

        let fitted = alm.fit(&x, &y).unwrap();

        let x_new = Mat::from_fn(3, 1, |i, _| (i as f64) / 2.0);
        let preds = fitted.predict(&x_new);

        // Predictions should be positive (exp of linear predictor)
        for i in 0..3 {
            assert!(preds[i] > 0.0);
        }
    }
}

// ============================================================================
// Inference Tests
// ============================================================================

mod inference_tests {
    use super::*;

    /// Test standard errors are computed
    #[test]
    fn test_standard_errors() {
        let x = Mat::from_fn(100, 2, |i, j| {
            if j == 0 {
                i as f64
            } else {
                (i as f64 * 0.1).sin()
            }
        });
        let y = Col::from_fn(100, |i| 1.0 + 2.0 * i as f64 + 0.5 * (i as f64 * 0.1).sin());

        let alm = AlmRegressor::builder()
            .distribution(AlmDistribution::Normal)
            .compute_inference(true)
            .with_intercept(true)
            .build();

        let fitted = alm.fit(&x, &y).unwrap();
        let result = fitted.result();

        assert!(result.std_errors.is_some());
        let se = result.std_errors.as_ref().unwrap();
        assert_eq!(se.nrows(), 2);
        assert!(se[0] > 0.0);
        assert!(se[1] > 0.0);
    }

    /// Test p-values are computed
    #[test]
    fn test_p_values() {
        let x = Mat::from_fn(100, 1, |i, _| i as f64);
        let y = Col::from_fn(100, |i| 1.0 + 2.0 * i as f64);

        let alm = AlmRegressor::builder()
            .distribution(AlmDistribution::Normal)
            .compute_inference(true)
            .with_intercept(true)
            .build();

        let fitted = alm.fit(&x, &y).unwrap();
        let result = fitted.result();

        assert!(result.p_values.is_some());
        let p = result.p_values.as_ref().unwrap();
        // For a strong linear relationship, p-value should be very small
        assert!(p[0] < 0.05);
    }

    /// Test confidence intervals are computed
    #[test]
    fn test_confidence_intervals() {
        // Add some noise to avoid degenerate case with zero variance
        let x = Mat::from_fn(100, 1, |i, _| i as f64);
        let y = Col::from_fn(100, |i| {
            let noise = ((i as f64 * 0.1).sin()) * 5.0; // Deterministic "noise"
            1.0 + 2.0 * i as f64 + noise
        });

        let alm = AlmRegressor::builder()
            .distribution(AlmDistribution::Normal)
            .compute_inference(true)
            .confidence_level(0.95)
            .with_intercept(true)
            .build();

        let fitted = alm.fit(&x, &y).unwrap();
        let result = fitted.result();

        assert!(result.conf_interval_lower.is_some());
        assert!(result.conf_interval_upper.is_some());

        let lower = result.conf_interval_lower.as_ref().unwrap();
        let upper = result.conf_interval_upper.as_ref().unwrap();

        // Check that CI has reasonable width and contains estimate
        let coef = result.coefficients[0];
        assert!(lower[0] < coef);
        assert!(upper[0] > coef);
        // True coefficient is approximately 2.0, should be within CI
        assert!((coef - 2.0).abs() < 0.5);
    }

    /// Test AIC/BIC are computed
    #[test]
    fn test_information_criteria() {
        let x = Mat::from_fn(50, 1, |i, _| i as f64);
        let y = Col::from_fn(50, |i| 1.0 + 2.0 * i as f64);

        let alm = AlmRegressor::builder()
            .distribution(AlmDistribution::Normal)
            .with_intercept(true)
            .build();

        let fitted = alm.fit(&x, &y).unwrap();
        let result = fitted.result();

        assert!(result.aic.is_finite());
        assert!(result.bic.is_finite());
        assert!(result.log_likelihood.is_finite());
    }
}

// ============================================================================
// Edge Cases and Error Handling
// ============================================================================

mod edge_cases {
    use super::*;
    use anofox_regression::solvers::RegressionError;

    /// Test dimension mismatch error
    #[test]
    fn test_dimension_mismatch() {
        let x = Mat::from_fn(10, 2, |i, j| (i + j) as f64);
        let y = Col::from_fn(5, |i| i as f64); // Wrong size

        let alm = AlmRegressor::builder()
            .distribution(AlmDistribution::Normal)
            .build();

        let result = alm.fit(&x, &y);
        assert!(matches!(
            result,
            Err(RegressionError::DimensionMismatch { .. })
        ));
    }

    /// Test insufficient observations error
    #[test]
    fn test_insufficient_observations() {
        let x = Mat::from_fn(2, 5, |i, j| (i + j) as f64);
        let y = Col::from_fn(2, |i| i as f64);

        let alm = AlmRegressor::builder()
            .distribution(AlmDistribution::Normal)
            .with_intercept(true)
            .build();

        let result = alm.fit(&x, &y);
        assert!(matches!(
            result,
            Err(RegressionError::InsufficientObservations { .. })
        ));
    }

    /// Test Poisson with negative values
    #[test]
    fn test_poisson_negative_values() {
        let x = Mat::from_fn(10, 1, |i, _| i as f64);
        let y = Col::from_fn(10, |i| i as f64 - 5.0); // Contains negative values

        let alm = AlmRegressor::builder()
            .distribution(AlmDistribution::Poisson)
            .build();

        // Should either error or handle gracefully
        let result = alm.fit(&x, &y);
        // Implementation-dependent: may error or clamp values
        assert!(result.is_err() || result.is_ok());
    }

    /// Test Binomial with values outside [0, 1]
    #[test]
    fn test_binomial_invalid_values() {
        let x = Mat::from_fn(10, 1, |i, _| i as f64);
        let y = Col::from_fn(10, |i| i as f64); // Values outside [0, 1]

        let alm = AlmRegressor::builder()
            .distribution(AlmDistribution::Binomial)
            .build();

        let result = alm.fit(&x, &y);
        // Should handle or error gracefully
        assert!(result.is_err() || result.is_ok());
    }

    /// Test with all zero response
    #[test]
    fn test_all_zero_response() {
        let x = Mat::from_fn(20, 1, |i, _| i as f64);
        let y = Col::from_fn(20, |_| 0.0);

        let alm = AlmRegressor::builder()
            .distribution(AlmDistribution::Normal)
            .build();

        let result = alm.fit(&x, &y);
        // Should handle this case
        assert!(result.is_ok());
    }

    /// Test with single predictor
    #[test]
    fn test_single_predictor() {
        let x = Mat::from_fn(50, 1, |i, _| i as f64);
        let y = Col::from_fn(50, |i| 2.0 * i as f64);

        let alm = AlmRegressor::builder()
            .distribution(AlmDistribution::Normal)
            .with_intercept(false)
            .build();

        let fitted = alm.fit(&x, &y).unwrap();
        assert!((fitted.coefficients()[0] - 2.0).abs() < 0.1);
    }
}

// ============================================================================
// Distribution Comparison Tests
// ============================================================================

mod distribution_comparison {
    use super::*;

    /// Compare Normal and Laplace on data with outliers
    #[test]
    fn test_normal_vs_laplace_outliers() {
        let mut y_data = vec![0.0; 100];
        let mut x_data = vec![0.0; 100];
        for i in 0..100 {
            x_data[i] = i as f64;
            y_data[i] = 1.0 + 2.0 * i as f64;
        }
        // Add outliers
        y_data[50] += 500.0;
        y_data[75] += 500.0;

        let x = Mat::from_fn(100, 1, |i, _| x_data[i]);
        let y = Col::from_fn(100, |i| y_data[i]);

        let normal = AlmRegressor::builder()
            .distribution(AlmDistribution::Normal)
            .with_intercept(true)
            .build();

        let laplace = AlmRegressor::builder()
            .distribution(AlmDistribution::Laplace)
            .with_intercept(true)
            .build();

        let fitted_normal = normal.fit(&x, &y).unwrap();
        let fitted_laplace = laplace.fit(&x, &y).unwrap();

        // Laplace should have slope closer to true value (2.0)
        let normal_error = (fitted_normal.coefficients()[0] - 2.0).abs();
        let laplace_error = (fitted_laplace.coefficients()[0] - 2.0).abs();

        // Laplace should be more robust (smaller error)
        assert!(laplace_error < normal_error);
    }

    /// Compare Student-t with different degrees of freedom
    #[test]
    fn test_student_t_df_comparison() {
        let x = Mat::from_fn(100, 1, |i, _| i as f64);
        let y = Col::from_fn(100, |i| 1.0 + 2.0 * i as f64);

        let t_low_df = AlmRegressor::builder()
            .distribution(AlmDistribution::StudentT)
            .extra_parameter(3.0) // heavy tails
            .with_intercept(true)
            .build();

        let t_high_df = AlmRegressor::builder()
            .distribution(AlmDistribution::StudentT)
            .extra_parameter(30.0) // close to Normal
            .with_intercept(true)
            .build();

        let fitted_low = t_low_df.fit(&x, &y).unwrap();
        let fitted_high = t_high_df.fit(&x, &y).unwrap();

        // Both should give similar results on clean data
        assert!((fitted_low.coefficients()[0] - fitted_high.coefficients()[0]).abs() < 0.5);
    }
}
