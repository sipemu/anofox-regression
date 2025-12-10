//! # Weighted Least Squares (WLS) Regression
//!
//! WLS extends OLS by allowing different weights for each observation.
//! This is useful when observations have different variances (heteroscedasticity).
//!
//! ## When to Use
//! - Heteroscedastic errors (non-constant variance)
//! - Different reliability of observations
//! - Inverse variance weighting for known variances
//! - Aggregated data with different group sizes
//!
//! ## Key Features
//! - Observation-specific weights
//! - Falls back to OLS when weights are equal
//! - All OLS diagnostics and inference available
//!
//! Run with: `cargo run --example wls`

use anofox_regression::core::IntervalType;
use anofox_regression::solvers::{FittedRegressor, Regressor, WlsRegressor};
use faer::{Col, Mat};

fn main() {
    println!("=== Weighted Least Squares (WLS) Regression ===\n");

    basic_wls();
    heteroscedastic_data();
    inverse_variance_weighting();
    compare_ols_wls();
}

/// Basic WLS with simple weights
fn basic_wls() {
    println!("--- Basic WLS ---\n");

    let n = 50;
    let x = Mat::from_fn(n, 1, |i, _| i as f64);
    let y = Col::from_fn(n, |i| {
        let noise = ((i as f64 * 0.7).sin()) * 0.5;
        2.0 + 1.5 * (i as f64) + noise
    });

    // Higher weights for later observations (more reliable)
    let weights = Col::from_fn(n, |i| 1.0 + (i as f64) * 0.1);

    let model = WlsRegressor::builder()
        .with_intercept(true)
        .weights(weights)
        .build();

    let fitted = model.fit(&x, &y).expect("fit should succeed");

    println!("Model: y = 2 + 1.5*x + noise (later observations weighted more)");
    println!("Intercept: {:.4}", fitted.intercept().unwrap_or(0.0));
    println!("Slope:     {:.4}", fitted.coefficients()[0]);
    println!("R-squared: {:.4}", fitted.r_squared());
    println!();
}

/// WLS for heteroscedastic data (variance increases with x)
fn heteroscedastic_data() {
    println!("--- Heteroscedastic Data ---\n");

    // Generate data where variance increases with x
    let n = 100;
    let x = Mat::from_fn(n, 1, |i, _| (i as f64) * 0.1 + 1.0);

    // Noise variance proportional to x
    let y = Col::from_fn(n, |i| {
        let xi = (i as f64) * 0.1 + 1.0;
        let noise_scale = xi.sqrt(); // Variance = x
        let noise = ((i as f64 * 1.3).sin()) * noise_scale;
        5.0 + 3.0 * xi + noise
    });

    // Optimal weights are inverse variance: w = 1/Var = 1/x
    let weights = Col::from_fn(n, |i| {
        let xi = (i as f64) * 0.1 + 1.0;
        1.0 / xi
    });

    // Fit with WLS (correct for heteroscedasticity)
    let wls_model = WlsRegressor::builder()
        .with_intercept(true)
        .weights(weights)
        .build();
    let wls_fitted = wls_model.fit(&x, &y).expect("fit should succeed");

    println!("True model: y = 5 + 3*x, with Var(error) proportional to x");
    println!("Using inverse variance weights: w_i = 1/x_i\n");

    println!("WLS estimates:");
    println!("  Intercept: {:.4}", wls_fitted.intercept().unwrap_or(0.0));
    println!("  Slope:     {:.4}", wls_fitted.coefficients()[0]);
    println!("  R-squared: {:.4}", wls_fitted.r_squared());
    println!();
}

/// Inverse variance weighting when variances are known
fn inverse_variance_weighting() {
    println!("--- Inverse Variance Weighting ---\n");

    // Simulate data from different sources with known variances
    let group_sizes = [20, 30, 50];
    let group_variances = [1.0, 4.0, 9.0]; // Known variances

    let n: usize = group_sizes.iter().sum();
    let mut x_data = Vec::with_capacity(n);
    let mut y_data = Vec::with_capacity(n);
    let mut w_data = Vec::with_capacity(n);

    let mut idx = 0;
    for (group, (&size, &var)) in group_sizes.iter().zip(group_variances.iter()).enumerate() {
        for _ in 0..size {
            let xi = (idx as f64) * 0.1;
            // Noise scaled by sqrt(variance)
            let noise = ((idx as f64 * 0.9).sin()) * (var as f64).sqrt();
            let yi = 1.0 + 2.0 * xi + noise;

            x_data.push(xi);
            y_data.push(yi);
            w_data.push(1.0 / var); // Inverse variance weight

            idx += 1;
        }
        println!(
            "Group {}: {} observations, variance = {:.1}, weight = {:.3}",
            group + 1,
            size,
            var,
            1.0 / var
        );
    }

    let x = Mat::from_fn(n, 1, |i, _| x_data[i]);
    let y = Col::from_fn(n, |i| y_data[i]);
    let weights = Col::from_fn(n, |i| w_data[i]);

    let model = WlsRegressor::builder()
        .with_intercept(true)
        .weights(weights)
        .build();
    let fitted = model.fit(&x, &y).expect("fit should succeed");

    println!("\nTrue model: y = 1 + 2*x");
    println!("\nWLS estimates with inverse variance weighting:");
    println!("  Intercept: {:.4}", fitted.intercept().unwrap_or(0.0));
    println!("  Slope:     {:.4}", fitted.coefficients()[0]);
    println!("  R-squared: {:.4}", fitted.r_squared());
    println!();
}

/// Compare OLS and WLS on heteroscedastic data
fn compare_ols_wls() {
    println!("--- OLS vs WLS Comparison ---\n");

    use anofox_regression::solvers::OlsRegressor;

    let n = 100;
    let x = Mat::from_fn(n, 2, |i, j| {
        if j == 0 {
            (i as f64) * 0.1
        } else {
            ((i as f64) * 0.2).sin()
        }
    });

    // Generate heteroscedastic data
    let y = Col::from_fn(n, |i| {
        let noise_scale = 1.0 + (i as f64) * 0.05; // Increasing variance
        let noise = ((i as f64 * 0.8).sin()) * noise_scale;
        3.0 + 2.0 * x[(i, 0)] - 1.0 * x[(i, 1)] + noise
    });

    // Inverse variance weights
    let weights = Col::from_fn(n, |i| {
        let variance = (1.0 + (i as f64) * 0.05).powi(2);
        1.0 / variance
    });

    // Fit OLS (ignoring heteroscedasticity)
    let ols_model = OlsRegressor::builder().with_intercept(true).build();
    let ols_fitted = ols_model.fit(&x, &y).expect("OLS fit should succeed");

    // Fit WLS (accounting for heteroscedasticity)
    let wls_model = WlsRegressor::builder()
        .with_intercept(true)
        .weights(weights)
        .build();
    let wls_fitted = wls_model.fit(&x, &y).expect("WLS fit should succeed");

    println!("True model: y = 3 + 2*x1 - 1*x2, with increasing variance\n");

    println!(
        "{:<12} {:>12} {:>12} {:>12}",
        "Method", "Intercept", "x1", "x2"
    );
    println!("{}", "-".repeat(50));
    println!("{:<12} {:>12.4} {:>12.4} {:>12.4}", "True", 3.0, 2.0, -1.0);
    println!(
        "{:<12} {:>12.4} {:>12.4} {:>12.4}",
        "OLS",
        ols_fitted.intercept().unwrap_or(0.0),
        ols_fitted.coefficients()[0],
        ols_fitted.coefficients()[1]
    );
    println!(
        "{:<12} {:>12.4} {:>12.4} {:>12.4}",
        "WLS",
        wls_fitted.intercept().unwrap_or(0.0),
        wls_fitted.coefficients()[0],
        wls_fitted.coefficients()[1]
    );

    println!("\nModel fit comparison:");
    println!("  OLS R-squared: {:.4}", ols_fitted.r_squared());
    println!("  WLS R-squared: {:.4}", wls_fitted.r_squared());

    // Prediction comparison
    let x_new = Mat::from_fn(3, 2, |i, j| if j == 0 { 5.0 + (i as f64) } else { 0.5 });

    let ols_pred = ols_fitted.predict_with_interval(&x_new, Some(IntervalType::Prediction), 0.95);
    let wls_pred = wls_fitted.predict_with_interval(&x_new, Some(IntervalType::Prediction), 0.95);

    println!("\nPredictions at new x values:");
    println!(
        "{:<8} {:>10} {:>10} {:>12} {:>12}",
        "x1", "OLS fit", "WLS fit", "OLS width", "WLS width"
    );
    println!("{}", "-".repeat(60));

    for i in 0..3 {
        let ols_width = ols_pred.upper[i] - ols_pred.lower[i];
        let wls_width = wls_pred.upper[i] - wls_pred.lower[i];
        println!(
            "{:<8.1} {:>10.3} {:>10.3} {:>12.3} {:>12.3}",
            x_new[(i, 0)],
            ols_pred.fit[i],
            wls_pred.fit[i],
            ols_width,
            wls_width
        );
    }

    println!("\nNote: WLS often provides more efficient estimates when weights");
    println!("      correctly reflect the inverse variance of observations.");
}
