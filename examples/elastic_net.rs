//! # Elastic Net Regression (L1 + L2 Regularization)
//!
//! Elastic Net combines Lasso (L1) and Ridge (L2) penalties,
//! providing variable selection while handling correlated predictors.
//!
//! ## When to Use
//! - Feature selection needed (L1 drives coefficients to zero)
//! - Correlated predictors present (L2 handles groups)
//! - More predictors than observations (p > n)
//! - Want sparsity with stability
//!
//! ## Key Features
//! - Penalty: lambda * (alpha * ||beta||_1 + (1-alpha) * ||beta||_2^2)
//! - alpha = 1: Pure Lasso (L1)
//! - alpha = 0: Pure Ridge (L2)
//! - Coordinate descent optimization
//!
//! Run with: `cargo run --example elastic_net`

use anofox_regression::solvers::{ElasticNetRegressor, FittedRegressor, Regressor};
use faer::{Col, Mat};

fn main() {
    println!("=== Elastic Net Regression ===\n");

    basic_elastic_net();
    lasso_vs_ridge_vs_elastic();
    feature_selection();
    sparsity_pattern();
}

/// Basic elastic net regression
fn basic_elastic_net() {
    println!("--- Basic Elastic Net ---\n");

    let n = 100;
    let x = Mat::from_fn(n, 3, |i, j| match j {
        0 => (i as f64) * 0.1,
        1 => ((i as f64) * 0.15).sin(),
        2 => ((i as f64) * 0.08).cos(),
        _ => 0.0,
    });

    let y = Col::from_fn(n, |i| {
        let noise = ((i as f64 * 0.7).sin()) * 0.3;
        2.0 + 3.0 * x[(i, 0)] - 1.5 * x[(i, 1)] + 0.0 * x[(i, 2)] + noise // x3 not used
    });

    // Elastic Net with alpha = 0.5 (50% L1, 50% L2)
    let model = ElasticNetRegressor::builder()
        .with_intercept(true)
        .lambda(0.1)
        .alpha(0.5)
        .build();

    let fitted = model.fit(&x, &y).expect("fit should succeed");

    println!("True model: y = 2 + 3*x1 - 1.5*x2 + 0*x3 + noise");
    println!("Lambda = 0.1, Alpha = 0.5 (50% L1, 50% L2)\n");
    println!("Intercept: {:.4}", fitted.intercept().unwrap_or(0.0));
    println!("Coefficient x1: {:.4}", fitted.coefficients()[0]);
    println!("Coefficient x2: {:.4}", fitted.coefficients()[1]);
    println!(
        "Coefficient x3: {:.4} (should be ~0)",
        fitted.coefficients()[2]
    );
    println!("R-squared: {:.4}", fitted.r_squared());
    println!();
}

/// Compare Lasso, Ridge, and Elastic Net
fn lasso_vs_ridge_vs_elastic() {
    println!("--- Lasso vs Ridge vs Elastic Net ---\n");

    let n = 100;
    let x = Mat::from_fn(n, 4, |i, j| match j {
        0 => (i as f64) * 0.1,
        1 => ((i as f64) * 0.12).sin(),
        2 => 0.5, // Constant (should be zeroed by Lasso)
        3 => ((i as f64) * 0.1).cos(),
        _ => 0.0,
    });

    let y = Col::from_fn(n, |i| {
        let noise = ((i as f64 * 0.8).sin()) * 0.2;
        1.0 + 2.0 * x[(i, 0)] - 1.0 * x[(i, 1)] + 0.0 * x[(i, 2)] + 0.5 * x[(i, 3)] + noise
    });

    let lambda = 0.5;

    // Lasso (alpha = 1)
    let lasso = ElasticNetRegressor::builder()
        .with_intercept(true)
        .lambda(lambda)
        .alpha(1.0)
        .build();
    let lasso_fit = lasso.fit(&x, &y).expect("lasso fit");

    // Ridge (alpha = 0)
    let ridge = ElasticNetRegressor::builder()
        .with_intercept(true)
        .lambda(lambda)
        .alpha(0.0)
        .build();
    let ridge_fit = ridge.fit(&x, &y).expect("ridge fit");

    // Elastic Net (alpha = 0.5)
    let enet = ElasticNetRegressor::builder()
        .with_intercept(true)
        .lambda(lambda)
        .alpha(0.5)
        .build();
    let enet_fit = enet.fit(&x, &y).expect("enet fit");

    println!("True coefficients: [2.0, -1.0, 0.0, 0.5]");
    println!("Lambda = {}\n", lambda);

    println!(
        "{:<15} {:>8} {:>8} {:>8} {:>8} {:>10}",
        "Method", "x1", "x2", "x3", "x4", "R-squared"
    );
    println!("{}", "-".repeat(65));

    println!(
        "{:<15} {:>8.4} {:>8.4} {:>8.4} {:>8.4} {:>10.4}",
        "True", 2.0, -1.0, 0.0, 0.5, 1.0
    );
    println!(
        "{:<15} {:>8.4} {:>8.4} {:>8.4} {:>8.4} {:>10.4}",
        "Lasso (a=1)",
        lasso_fit.coefficients()[0],
        lasso_fit.coefficients()[1],
        lasso_fit.coefficients()[2],
        lasso_fit.coefficients()[3],
        lasso_fit.r_squared()
    );
    println!(
        "{:<15} {:>8.4} {:>8.4} {:>8.4} {:>8.4} {:>10.4}",
        "Ridge (a=0)",
        ridge_fit.coefficients()[0],
        ridge_fit.coefficients()[1],
        ridge_fit.coefficients()[2],
        ridge_fit.coefficients()[3],
        ridge_fit.r_squared()
    );
    println!(
        "{:<15} {:>8.4} {:>8.4} {:>8.4} {:>8.4} {:>10.4}",
        "Elastic (a=0.5)",
        enet_fit.coefficients()[0],
        enet_fit.coefficients()[1],
        enet_fit.coefficients()[2],
        enet_fit.coefficients()[3],
        enet_fit.r_squared()
    );

    // Count non-zero coefficients
    let count_nonzero =
        |coefs: &Col<f64>| -> usize { coefs.iter().filter(|&&c| c.abs() > 1e-6).count() };

    println!("\nNon-zero coefficients:");
    println!("  Lasso:       {}", count_nonzero(lasso_fit.coefficients()));
    println!("  Ridge:       {}", count_nonzero(ridge_fit.coefficients()));
    println!("  Elastic Net: {}", count_nonzero(enet_fit.coefficients()));

    println!("\nNote: Lasso produces sparser models (more zeros).");
    println!("      Ridge keeps all coefficients non-zero.");
    println!("      Elastic Net balances both properties.");
    println!();
}

/// Feature selection with varying alpha
fn feature_selection() {
    println!("--- Feature Selection with Alpha ---\n");

    let n = 100;
    // 8 predictors, but only 3 are relevant
    let x = Mat::from_fn(n, 8, |i, j| match j {
        0 => (i as f64) * 0.1,                      // Relevant
        1 => ((i as f64) * 0.2).sin(),              // Relevant
        2 => ((i as f64) * 0.15).cos(),             // Relevant
        _ => ((i as f64 * (j as f64)).sin()) * 0.5, // Noise predictors
    });

    let y = Col::from_fn(n, |i| {
        let noise = ((i as f64 * 0.9).sin()) * 0.3;
        1.0 + 2.5 * x[(i, 0)] - 1.2 * x[(i, 1)] + 0.8 * x[(i, 2)] + noise
    });

    println!("8 predictors, only first 3 are relevant");
    println!("True: [2.5, -1.2, 0.8, 0, 0, 0, 0, 0]\n");

    let lambda = 0.3;
    let alphas = [0.0, 0.25, 0.5, 0.75, 1.0];

    println!(
        "{:<8} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6} {:>8}",
        "Alpha", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "Non-zero"
    );
    println!("{}", "-".repeat(85));

    for &alpha in &alphas {
        let model = ElasticNetRegressor::builder()
            .with_intercept(true)
            .lambda(lambda)
            .alpha(alpha)
            .build();

        let fitted = model.fit(&x, &y).expect("fit should succeed");
        let coefs = fitted.coefficients();
        let n_nonzero = coefs.iter().filter(|&&c| c.abs() > 1e-6).count();

        println!(
            "{:<8.2} {:>6.2} {:>6.2} {:>6.2} {:>6.2} {:>6.2} {:>6.2} {:>6.2} {:>6.2} {:>8}",
            alpha,
            coefs[0],
            coefs[1],
            coefs[2],
            coefs[3],
            coefs[4],
            coefs[5],
            coefs[6],
            coefs[7],
            n_nonzero
        );
    }

    println!("\nNote: Higher alpha (more L1) produces sparser solutions.");
    println!();
}

/// Show how sparsity changes with lambda
fn sparsity_pattern() {
    println!("--- Sparsity vs Lambda ---\n");

    let n = 100;
    let p = 10;

    let x = Mat::from_fn(n, p, |i, j| {
        ((i as f64 * 0.1 + j as f64 * 0.3).sin()) + (j as f64) * 0.05
    });

    // Only first 4 predictors are relevant
    let y = Col::from_fn(n, |i| {
        let noise = ((i as f64 * 1.1).sin()) * 0.2;
        2.0 + 1.5 * x[(i, 0)] - 1.0 * x[(i, 1)] + 0.7 * x[(i, 2)] - 0.4 * x[(i, 3)] + noise
    });

    let alpha = 0.8; // Mostly Lasso
    let lambdas = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0];

    println!("10 predictors, first 4 relevant. Alpha = {}", alpha);
    println!("True: [1.5, -1.0, 0.7, -0.4, 0, 0, 0, 0, 0, 0]\n");

    println!(
        "{:<10} {:>10} {:>10} {:>10}",
        "Lambda", "Non-zero", "R-squared", "Correct*"
    );
    println!("{}", "-".repeat(45));

    for &lambda in &lambdas {
        let model = ElasticNetRegressor::builder()
            .with_intercept(true)
            .lambda(lambda)
            .alpha(alpha)
            .build();

        let fitted = model.fit(&x, &y).expect("fit should succeed");
        let coefs = fitted.coefficients();

        let n_nonzero = coefs.iter().filter(|&&c| c.abs() > 1e-6).count();

        // Check if we correctly identified the 4 relevant features
        let relevant_nonzero = (0..4).filter(|&i| coefs[i].abs() > 1e-6).count();
        let irrelevant_zero = (4..p).filter(|&i| coefs[i].abs() < 1e-6).count();
        let correct = relevant_nonzero == 4 && irrelevant_zero == 6;

        println!(
            "{:<10.3} {:>10} {:>10.4} {:>10}",
            lambda,
            n_nonzero,
            fitted.r_squared(),
            if correct { "Yes" } else { "No" }
        );
    }

    println!("\n* Correct = all 4 relevant features selected, all 6 noise features zeroed");
    println!("\nNote: Very small lambda: all features selected (overfitting risk)");
    println!("      Very large lambda: too many features zeroed (underfitting)");
    println!("      Optimal lambda: correct feature selection");
}
