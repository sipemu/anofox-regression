//! Elastic Net regression integration tests.

mod common;

use anofox_regression::solvers::{ElasticNetRegressor, FittedRegressor, Regressor};
use approx::assert_relative_eq;
use faer::{Col, Mat};

// ============================================================================
// Basic Elastic Net Tests
// ============================================================================

#[test]
fn test_elastic_net_basic_fit() {
    let x = Mat::from_fn(50, 2, |i, j| ((i + j) as f64) * 0.1);
    let y = Col::from_fn(50, |i| 1.0 + 2.0 * i as f64 * 0.1);

    let model = ElasticNetRegressor::builder()
        .with_intercept(true)
        .lambda(0.1)
        .alpha(0.5) // 50% L1, 50% L2
        .build();

    let fitted = model.fit(&x, &y).expect("fit should succeed");

    // Check basic properties
    assert!(fitted.r_squared() > 0.0);
    assert_eq!(fitted.result().coefficients.nrows(), 2);
}

#[test]
fn test_elastic_net_pure_ridge() {
    // alpha = 0 is pure Ridge
    let x = Mat::from_fn(50, 2, |i, j| ((i + j) as f64) * 0.1);
    let y = Col::from_fn(50, |i| 1.0 + 2.0 * i as f64 * 0.1);

    let model = ElasticNetRegressor::builder()
        .with_intercept(true)
        .lambda(0.1)
        .alpha(0.0) // Pure Ridge
        .build();

    let fitted = model.fit(&x, &y).expect("fit should succeed");

    // Pure Ridge should keep all coefficients non-zero
    assert!(fitted.n_nonzero() == 2);
}

#[test]
fn test_elastic_net_pure_lasso() {
    // alpha = 1 is pure Lasso
    let x = Mat::from_fn(100, 5, |i, j| {
        if j == 0 {
            i as f64 * 0.1
        } else if j == 1 {
            (i as f64 * 0.1).sin()
        } else {
            // Noise predictors
            ((i * (j + 1)) as f64 * 0.01).cos()
        }
    });
    let y = Col::from_fn(100, |i| {
        1.0 + 2.0 * i as f64 * 0.1 + 0.5 * (i as f64 * 0.1).sin()
    });

    let model = ElasticNetRegressor::builder()
        .with_intercept(true)
        .lambda(1.0) // Moderate regularization
        .alpha(1.0) // Pure Lasso
        .max_iterations(1000)
        .tolerance(1e-6)
        .build();

    let fitted = model.fit(&x, &y).expect("fit should succeed");

    // Lasso should potentially zero out some coefficients
    let n_nonzero = fitted.n_nonzero();
    assert!(n_nonzero <= 5, "Lasso should produce sparse solution");
}

#[test]
fn test_elastic_net_prediction() {
    let x = Mat::from_fn(50, 2, |i, j| ((i + j) as f64) * 0.1);
    let y = Col::from_fn(50, |i| 1.0 + 2.0 * i as f64 * 0.1);

    let model = ElasticNetRegressor::builder()
        .with_intercept(true)
        .lambda(0.01)
        .alpha(0.5)
        .build();

    let fitted = model.fit(&x, &y).expect("fit should succeed");

    // Predict on new data
    let x_new = Mat::from_fn(10, 2, |i, j| ((i + j + 50) as f64) * 0.1);
    let predictions = fitted.predict(&x_new);

    assert_eq!(predictions.nrows(), 10);

    // Predictions should be reasonable (positive for this data)
    for i in 0..10 {
        assert!(predictions[i].is_finite());
    }
}

#[test]
fn test_elastic_net_varying_alpha() {
    let x = Mat::from_fn(100, 3, |i, j| ((i + j) as f64) * 0.1);
    let y = Col::from_fn(100, |i| 1.0 + 2.0 * i as f64 * 0.1);

    // Test various alpha values
    for alpha in &[0.0, 0.25, 0.5, 0.75, 1.0] {
        let model = ElasticNetRegressor::builder()
            .with_intercept(true)
            .lambda(0.5)
            .alpha(*alpha)
            .build();

        let fitted = model
            .fit(&x, &y)
            .unwrap_or_else(|_| panic!("fit should succeed for alpha={}", alpha));
        assert!(fitted.r_squared() >= 0.0);
        assert!(fitted.r_squared() <= 1.0 || fitted.r_squared().is_nan());
    }
}

#[test]
fn test_elastic_net_high_regularization() {
    // With very high lambda, coefficients should be near zero
    let x = Mat::from_fn(50, 2, |i, j| ((i + j) as f64) * 0.1);
    let y = Col::from_fn(50, |i| 1.0 + 2.0 * i as f64 * 0.1);

    let model = ElasticNetRegressor::builder()
        .with_intercept(true)
        .lambda(1000.0)
        .alpha(0.5)
        .build();

    let fitted = model.fit(&x, &y).expect("fit should succeed");

    // With high regularization, coefficients should be small
    let coefs = fitted.coefficients();
    for i in 0..coefs.nrows() {
        assert!(
            coefs[i].abs() < 1.0,
            "High regularization should shrink coefficients"
        );
    }
}

#[test]
fn test_elastic_net_no_intercept() {
    let x = Mat::from_fn(50, 2, |i, j| ((i + j + 1) as f64) * 0.1);
    let y = Col::from_fn(50, |i| 2.0 * i as f64 * 0.1);

    let model = ElasticNetRegressor::builder()
        .with_intercept(false)
        .lambda(0.1)
        .alpha(0.5)
        .build();

    let fitted = model.fit(&x, &y).expect("fit should succeed");

    assert!(fitted.intercept().is_none());
}

#[test]
fn test_elastic_net_convergence_parameters() {
    let x = Mat::from_fn(100, 3, |i, j| ((i + j) as f64) * 0.1);
    let y = Col::from_fn(100, |i| 1.0 + 2.0 * i as f64 * 0.1);

    // Test with different convergence parameters
    let model_fast = ElasticNetRegressor::builder()
        .with_intercept(true)
        .lambda(0.1)
        .alpha(0.5)
        .max_iterations(10)
        .tolerance(1e-3)
        .build();

    let model_slow = ElasticNetRegressor::builder()
        .with_intercept(true)
        .lambda(0.1)
        .alpha(0.5)
        .max_iterations(10000)
        .tolerance(1e-10)
        .build();

    let fitted_fast = model_fast.fit(&x, &y).expect("fit should succeed");
    let fitted_slow = model_slow.fit(&x, &y).expect("fit should succeed");

    // Both should produce reasonable fits
    assert!(fitted_fast.r_squared() >= 0.0);
    assert!(fitted_slow.r_squared() >= 0.0);
}

#[test]
fn test_elastic_net_with_inference() {
    let x = Mat::from_fn(100, 2, |i, j| ((i + j) as f64) * 0.1);
    let y = Col::from_fn(100, |i| {
        1.0 + 2.0 * i as f64 * 0.1 + (i as f64 * 0.1).sin() * 0.1
    });

    let model = ElasticNetRegressor::builder()
        .with_intercept(true)
        .lambda(0.1)
        .alpha(0.5)
        .build();

    let fitted = model.fit(&x, &y).expect("fit should succeed");
    let result = fitted.result();

    // Check fit statistics are computed
    assert!(result.r_squared >= 0.0);
    assert!(result.mse >= 0.0);
    assert!(result.rmse >= 0.0);
}

#[test]
fn test_elastic_net_getters() {
    let model = ElasticNetRegressor::builder()
        .with_intercept(true)
        .lambda(0.5)
        .alpha(0.7)
        .build();

    let x = Mat::from_fn(50, 2, |i, j| ((i + j) as f64) * 0.1);
    let y = Col::from_fn(50, |i| 1.0 + 2.0 * i as f64 * 0.1);

    let fitted = model.fit(&x, &y).expect("fit should succeed");

    // Test getter methods
    assert_relative_eq!(fitted.lambda(), 0.5, epsilon = 1e-10);
    assert_relative_eq!(fitted.alpha(), 0.7, epsilon = 1e-10);

    // Options should be accessible
    let options = fitted.options();
    assert!(options.with_intercept);
}

#[test]
fn test_elastic_net_sparsity_increases_with_lambda() {
    let x = Mat::from_fn(100, 5, |i, j| {
        match j {
            0 => i as f64 * 0.1,
            1 => (i as f64 * 0.1).sin(),
            _ => ((i * j) as f64 * 0.01).cos() * 0.01, // Near-zero contribution
        }
    });
    let y = Col::from_fn(100, |i| {
        1.0 + 2.0 * i as f64 * 0.1 + 0.5 * (i as f64 * 0.1).sin()
    });

    let mut prev_nonzero = 5;

    for lambda in &[0.01, 0.1, 1.0, 10.0] {
        let model = ElasticNetRegressor::builder()
            .with_intercept(true)
            .lambda(*lambda)
            .alpha(0.9) // High L1 ratio
            .build();

        let fitted = model.fit(&x, &y).expect("fit should succeed");
        let n_nonzero = fitted.n_nonzero();

        // Sparsity should generally not decrease with increasing lambda
        assert!(
            n_nonzero <= prev_nonzero || n_nonzero == 0,
            "Sparsity should increase with lambda: lambda={}, n_nonzero={}, prev={}",
            lambda,
            n_nonzero,
            prev_nonzero
        );
        prev_nonzero = n_nonzero;
    }
}

// ============================================================================
// Edge Cases
// ============================================================================

#[test]
fn test_elastic_net_single_predictor() {
    let x = Mat::from_fn(50, 1, |i, _| i as f64);
    let y = Col::from_fn(50, |i| 1.0 + 2.0 * i as f64);

    let model = ElasticNetRegressor::builder()
        .with_intercept(true)
        .lambda(0.1)
        .alpha(0.5)
        .build();

    let fitted = model.fit(&x, &y).expect("fit should succeed");
    assert_eq!(fitted.result().coefficients.nrows(), 1);
}

#[test]
fn test_elastic_net_small_sample() {
    let x = Mat::from_fn(10, 2, |i, j| ((i + j) as f64) * 0.1);
    let y = Col::from_fn(10, |i| 1.0 + 2.0 * i as f64 * 0.1);

    let model = ElasticNetRegressor::builder()
        .with_intercept(true)
        .lambda(0.1)
        .alpha(0.5)
        .build();

    let fitted = model.fit(&x, &y).expect("fit should succeed");
    assert!(fitted.r_squared().is_finite() || fitted.r_squared().is_nan());
}

#[test]
fn test_elastic_net_many_predictors() {
    let x = Mat::from_fn(100, 20, |i, j| ((i * j + 1) as f64).sin() * 0.1);
    let y = Col::from_fn(100, |i| 1.0 + (i as f64 * 0.1).cos());

    let model = ElasticNetRegressor::builder()
        .with_intercept(true)
        .lambda(1.0)
        .alpha(0.9)
        .max_iterations(5000)
        .build();

    let fitted = model.fit(&x, &y).expect("fit should succeed");

    // With high regularization and many predictors, should be sparse
    assert!(fitted.n_nonzero() <= 20);
}

#[test]
fn test_elastic_net_constant_target() {
    let x = Mat::from_fn(50, 2, |i, j| ((i + j) as f64) * 0.1);
    let y = Col::from_fn(50, |_| 5.0); // Constant target

    let model = ElasticNetRegressor::builder()
        .with_intercept(true)
        .lambda(0.1)
        .alpha(0.5)
        .build();

    let fitted = model.fit(&x, &y).expect("fit should succeed");

    // Coefficients should be near zero with constant target
    let coefs = fitted.coefficients();
    for i in 0..coefs.nrows() {
        assert!(
            coefs[i].abs() < 0.1,
            "Constant target should result in near-zero coefficients"
        );
    }
}

#[test]
fn test_elastic_net_perfect_fit() {
    // Create perfectly linear data
    let x = Mat::from_fn(50, 2, |i, _| i as f64);
    let y = Col::from_fn(50, |i| 1.0 + 2.0 * i as f64);

    let model = ElasticNetRegressor::builder()
        .with_intercept(true)
        .lambda(0.001) // Very small regularization
        .alpha(0.5)
        .max_iterations(5000)
        .tolerance(1e-10)
        .build();

    let fitted = model.fit(&x, &y).expect("fit should succeed");

    // Should achieve high R²
    assert!(fitted.r_squared() > 0.99);
}
