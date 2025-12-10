//! # Ridge Regression (L2 Regularization)
//!
//! Ridge regression adds an L2 penalty to the OLS objective function,
//! shrinking coefficients towards zero to reduce overfitting.
//!
//! ## When to Use
//! - Multicollinearity among predictors
//! - More predictors than observations (p > n)
//! - Prevent overfitting in complex models
//! - When all predictors should be retained (vs. Lasso)
//!
//! ## Key Features
//! - L2 penalty: lambda * ||beta||^2
//! - Shrinks coefficients but never to exactly zero
//! - Falls back to OLS when lambda = 0
//! - Bias-variance tradeoff via lambda
//!
//! Run with: `cargo run --example ridge`

use anofox_regression::solvers::{FittedRegressor, Regressor, RidgeRegressor};
use faer::{Col, Mat};

fn main() {
    println!("=== Ridge Regression (L2 Regularization) ===\n");

    basic_ridge();
    multicollinearity_example();
    regularization_path();
    compare_ols_ridge();
}

/// Basic ridge regression
fn basic_ridge() {
    println!("--- Basic Ridge Regression ---\n");

    let n = 50;
    let x = Mat::from_fn(n, 2, |i, j| {
        if j == 0 {
            (i as f64) * 0.2
        } else {
            ((i as f64) * 0.3).sin()
        }
    });

    let y = Col::from_fn(n, |i| {
        let noise = ((i as f64 * 0.7).sin()) * 0.3;
        1.0 + 2.0 * x[(i, 0)] - 0.5 * x[(i, 1)] + noise
    });

    // Ridge with lambda = 0.1
    let model = RidgeRegressor::builder()
        .with_intercept(true)
        .lambda(0.1)
        .build();

    let fitted = model.fit(&x, &y).expect("fit should succeed");

    println!("True model: y = 1 + 2*x1 - 0.5*x2 + noise");
    println!("Lambda = 0.1\n");
    println!("Intercept: {:.4}", fitted.intercept().unwrap_or(0.0));
    println!("Coefficient x1: {:.4}", fitted.coefficients()[0]);
    println!("Coefficient x2: {:.4}", fitted.coefficients()[1]);
    println!("R-squared: {:.4}", fitted.r_squared());
    println!();
}

/// Ridge helps with multicollinear predictors
fn multicollinearity_example() {
    println!("--- Handling Multicollinearity ---\n");

    let n = 100;
    // Create correlated predictors: x2 ≈ x1 + small noise
    let x = Mat::from_fn(n, 3, |i, j| match j {
        0 => (i as f64) * 0.1,
        1 => (i as f64) * 0.1 + ((i as f64 * 0.5).sin()) * 0.05, // Highly correlated with x1
        2 => ((i as f64) * 0.15).cos(),
        _ => 0.0,
    });

    let y = Col::from_fn(n, |i| {
        let noise = ((i as f64 * 0.8).sin()) * 0.2;
        2.0 + 1.5 * x[(i, 0)] + 0.5 * x[(i, 1)] - 1.0 * x[(i, 2)] + noise
    });

    println!("Data with highly correlated predictors (x1 and x2):\n");

    // Try different lambda values
    let lambdas = [0.0, 0.01, 0.1, 1.0, 10.0];

    println!(
        "{:<10} {:>10} {:>10} {:>10} {:>10}",
        "Lambda", "x1", "x2", "x3", "R-squared"
    );
    println!("{}", "-".repeat(55));

    for &lambda in &lambdas {
        let model = RidgeRegressor::builder()
            .with_intercept(true)
            .lambda(lambda)
            .build();

        let fitted = model.fit(&x, &y).expect("fit should succeed");

        println!(
            "{:<10.3} {:>10.4} {:>10.4} {:>10.4} {:>10.4}",
            lambda,
            fitted.coefficients()[0],
            fitted.coefficients()[1],
            fitted.coefficients()[2],
            fitted.r_squared()
        );
    }

    println!("\nNote: With lambda=0 (OLS), coefficients may be unstable.");
    println!("      Ridge regularization stabilizes the estimates.");
    println!();
}

/// Show how coefficients change with lambda
fn regularization_path() {
    println!("--- Regularization Path ---\n");

    let n = 80;
    let x = Mat::from_fn(n, 4, |i, j| match j {
        0 => (i as f64) * 0.1,
        1 => ((i as f64) * 0.2).sin(),
        2 => ((i as f64) * 0.15).cos(),
        3 => ((i as f64) * 0.1).powi(2) / 100.0,
        _ => 0.0,
    });

    let y = Col::from_fn(n, |i| {
        let noise = ((i as f64 * 0.9).sin()) * 0.3;
        1.0 + 3.0 * x[(i, 0)] - 2.0 * x[(i, 1)] + 1.5 * x[(i, 2)] - 0.5 * x[(i, 3)] + noise
    });

    println!("True coefficients: [3.0, -2.0, 1.5, -0.5]");
    println!("Showing how coefficients shrink as lambda increases:\n");

    println!(
        "{:<12} {:>8} {:>8} {:>8} {:>8} {:>10}",
        "Lambda", "x1", "x2", "x3", "x4", "R-squared"
    );
    println!("{}", "-".repeat(60));

    let lambdas = [0.0, 0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0];

    for &lambda in &lambdas {
        let model = RidgeRegressor::builder()
            .with_intercept(true)
            .lambda(lambda)
            .build();

        let fitted = model.fit(&x, &y).expect("fit should succeed");

        println!(
            "{:<12.3} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>10.4}",
            lambda,
            fitted.coefficients()[0],
            fitted.coefficients()[1],
            fitted.coefficients()[2],
            fitted.coefficients()[3],
            fitted.r_squared()
        );
    }

    println!("\nNote: As lambda increases, coefficients shrink towards zero,");
    println!("      but never reach exactly zero (unlike Lasso).");
    println!();
}

/// Compare OLS and Ridge predictions
fn compare_ols_ridge() {
    println!("--- OLS vs Ridge Comparison ---\n");

    use anofox_regression::solvers::OlsRegressor;

    // Small sample with many predictors (prone to overfitting)
    let n = 30;
    let p = 10;

    let x = Mat::from_fn(n, p, |i, j| {
        ((i as f64 * 0.3 + j as f64 * 0.7).sin()) + ((j as f64) * 0.1)
    });

    // True model uses only first 3 predictors
    let y = Col::from_fn(n, |i| {
        let noise = ((i as f64 * 1.1).sin()) * 0.5;
        2.0 + 1.0 * x[(i, 0)] + 0.5 * x[(i, 1)] - 0.3 * x[(i, 2)] + noise
    });

    // Split into train/test
    let n_train = 20;
    let x_train = Mat::from_fn(n_train, p, |i, j| x[(i, j)]);
    let y_train = Col::from_fn(n_train, |i| y[i]);
    let x_test = Mat::from_fn(n - n_train, p, |i, j| x[(i + n_train, j)]);
    let y_test = Col::from_fn(n - n_train, |i| y[i + n_train]);

    // Fit OLS
    let ols_model = OlsRegressor::builder().with_intercept(true).build();
    let ols_fitted = ols_model
        .fit(&x_train, &y_train)
        .expect("OLS fit should succeed");

    // Fit Ridge with different lambdas
    let lambdas = [0.1, 1.0, 10.0];

    println!("Small sample (n=20) with many predictors (p=10)");
    println!("True model uses only x1, x2, x3\n");

    println!("{:<15} {:>12} {:>12}", "Method", "Train R2", "Test MSE");
    println!("{}", "-".repeat(42));

    // OLS performance
    let ols_train_r2 = ols_fitted.r_squared();
    let ols_pred = ols_fitted.predict(&x_test);
    let ols_test_mse: f64 = (0..y_test.nrows())
        .map(|i| (y_test[i] - ols_pred[i]).powi(2))
        .sum::<f64>()
        / y_test.nrows() as f64;

    println!(
        "{:<15} {:>12.4} {:>12.4}",
        "OLS", ols_train_r2, ols_test_mse
    );

    // Ridge performance
    for &lambda in &lambdas {
        let ridge_model = RidgeRegressor::builder()
            .with_intercept(true)
            .lambda(lambda)
            .build();

        let ridge_fitted = ridge_model
            .fit(&x_train, &y_train)
            .expect("Ridge fit should succeed");

        let ridge_train_r2 = ridge_fitted.r_squared();
        let ridge_pred = ridge_fitted.predict(&x_test);
        let ridge_test_mse: f64 = (0..y_test.nrows())
            .map(|i| (y_test[i] - ridge_pred[i]).powi(2))
            .sum::<f64>()
            / y_test.nrows() as f64;

        println!(
            "{:<15} {:>12.4} {:>12.4}",
            format!("Ridge({})", lambda),
            ridge_train_r2,
            ridge_test_mse
        );
    }

    println!("\nNote: OLS may overfit (high train R2, high test error).");
    println!("      Ridge regularization often improves test performance.");
}
