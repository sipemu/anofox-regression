//! # Tweedie Regression
//!
//! Tweedie regression models data from the Tweedie family of distributions,
//! which includes Gaussian, Poisson, Gamma, and compound Poisson-Gamma
//! distributions as special cases.
//!
//! ## When to Use
//! - Insurance claims (many zeros + continuous positives)
//! - Revenue data with zero-inflation
//! - Positive continuous data (Gamma)
//! - Count data (Poisson)
//! - Flexible variance modeling
//!
//! ## Key Features
//! - Variance function: V(mu) = mu^var_power
//! - var_power = 0: Gaussian (Normal)
//! - var_power = 1: Poisson
//! - var_power in (1,2): Compound Poisson-Gamma
//! - var_power = 2: Gamma
//! - var_power = 3: Inverse Gaussian
//!
//! Run with: `cargo run --example tweedie`

use anofox_regression::solvers::{FittedRegressor, Regressor, TweedieRegressor};
use faer::{Col, Mat};

fn main() {
    println!("=== Tweedie Regression ===\n");

    gaussian_tweedie();
    gamma_tweedie();
    compound_poisson_gamma();
    variance_power_comparison();
}

/// Tweedie with var_power=0 (Gaussian)
fn gaussian_tweedie() {
    println!("--- Gaussian (var_power = 0) ---\n");

    let n = 80;
    let x = Mat::from_fn(n, 2, |i, j| {
        if j == 0 {
            (i as f64) * 0.1
        } else {
            ((i as f64) * 0.15).sin()
        }
    });

    let y = Col::from_fn(n, |i| {
        let noise = ((i as f64 * 0.7).sin()) * 0.5;
        2.0 + 1.5 * x[(i, 0)] - 0.8 * x[(i, 1)] + noise
    });

    let model = TweedieRegressor::gaussian().build();
    let fitted = model.fit(&x, &y).expect("fit should succeed");

    println!("True model: y = 2 + 1.5*x1 - 0.8*x2 + noise");
    println!("Using Gaussian Tweedie (identity link, var_power=0)\n");
    println!(
        "Intercept: {:.4} (true: 2.0)",
        fitted.intercept().unwrap_or(0.0)
    );
    println!(
        "Coefficient x1: {:.4} (true: 1.5)",
        fitted.coefficients()[0]
    );
    println!(
        "Coefficient x2: {:.4} (true: -0.8)",
        fitted.coefficients()[1]
    );
    println!("R-squared: {:.4}", fitted.r_squared());

    println!("\nNote: Gaussian Tweedie is equivalent to OLS regression.");
    println!();
}

/// Tweedie with var_power=2 (Gamma)
fn gamma_tweedie() {
    println!("--- Gamma (var_power = 2) ---\n");

    let n = 100;
    let x = Mat::from_fn(n, 1, |i, _| (i as f64) * 0.05 + 0.1);

    // Generate positive continuous data with variance proportional to mu^2
    let y = Col::from_fn(n, |i| {
        let eta = 0.5 + 0.3 * x[(i, 0)];
        let mu = eta.exp(); // Log link inverse
                            // Gamma-like noise: CV roughly constant
        let noise = ((i as f64 * 0.9).sin()) * (mu * 0.2);
        (mu + noise).max(0.1)
    });

    let model = TweedieRegressor::gamma().build();
    let fitted = model.fit(&x, &y).expect("fit should succeed");
    let result = fitted.result();

    println!("True model: log(mu) = 0.5 + 0.3*x (Gamma distribution)\n");
    println!(
        "Intercept: {:.4} (true: 0.5)",
        fitted.intercept().unwrap_or(0.0)
    );
    println!("Coefficient: {:.4} (true: 0.3)", fitted.coefficients()[0]);
    println!("AIC: {:.4}", result.aic);

    // Interpretation
    println!("\nInterpretation (multiplicative effects):");
    let effect = fitted.coefficients()[0].exp();
    println!("  1 unit increase in x multiplies mu by {:.4}", effect);
    println!();
}

/// Compound Poisson-Gamma (insurance claims)
fn compound_poisson_gamma() {
    println!("--- Compound Poisson-Gamma (var_power = 1.5) ---\n");

    let n = 150;
    let x = Mat::from_fn(n, 2, |i, j| {
        if j == 0 {
            (i as f64) * 0.02 // Risk factor 1
        } else {
            ((i as f64) * 0.05).sin() * 0.5 + 0.5 // Risk factor 2
        }
    });

    // Simulate insurance-like data: many zeros, positive values when claims occur
    let y = Col::from_fn(n, |i| {
        let eta = 0.3 + 0.2 * x[(i, 0)] + 0.1 * x[(i, 1)];
        let mu = eta.exp();

        // Probability of claim (simplified)
        let claim_prob = (mu / (1.0 + mu)).min(0.7);
        let has_claim = ((i as f64 * 0.7).sin() + 1.0) / 2.0 < claim_prob;

        if has_claim {
            // Claim amount
            mu * (1.0 + ((i as f64 * 1.1).sin()) * 0.3)
        } else {
            0.0
        }
    });

    // Count zeros
    let n_zeros = y.iter().filter(|&&yi| yi == 0.0).count();
    println!("Data summary:");
    println!("  Total observations: {}", n);
    println!(
        "  Zero values: {} ({:.1}%)",
        n_zeros,
        100.0 * n_zeros as f64 / n as f64
    );

    // Fit with var_power = 1.5 (between Poisson and Gamma)
    let model = TweedieRegressor::builder()
        .var_power(1.5)
        .link_power(0.0) // Log link
        .build();

    let fitted = model.fit(&x, &y).expect("fit should succeed");
    let result = fitted.result();

    println!("\nCompound Poisson-Gamma model (var_power = 1.5):");
    println!("  Intercept: {:.4}", fitted.intercept().unwrap_or(0.0));
    println!("  Risk factor 1: {:.4}", fitted.coefficients()[0]);
    println!("  Risk factor 2: {:.4}", fitted.coefficients()[1]);
    println!("  AIC: {:.4}", result.aic);

    println!("\nNote: Compound Poisson-Gamma naturally handles");
    println!("      zero-inflated positive continuous data (e.g., insurance claims).");
    println!();
}

/// Compare different variance powers
fn variance_power_comparison() {
    println!("--- Variance Power Comparison ---\n");

    let n = 100;
    let x = Mat::from_fn(n, 1, |i, _| (i as f64) * 0.1);

    // Generate positive data
    let y = Col::from_fn(n, |i| {
        let mu = (1.0 + 0.2 * x[(i, 0)]).exp();
        let noise = ((i as f64 * 0.8).sin()) * (mu * 0.3);
        (mu + noise).max(0.1)
    });

    // Compare different variance powers
    let var_powers = [0.0, 1.0, 1.5, 2.0, 3.0];
    let names = ["Gaussian", "Poisson", "CP-Gamma", "Gamma", "Inv-Gauss"];

    println!(
        "{:<12} {:>8} {:>12} {:>12} {:>12}",
        "Model", "VarPow", "Intercept", "Slope", "AIC"
    );
    println!("{}", "-".repeat(60));

    for (&vp, &name) in var_powers.iter().zip(names.iter()) {
        let model = TweedieRegressor::builder()
            .var_power(vp)
            .link_power(if vp == 0.0 { 1.0 } else { 0.0 }) // Identity for Gaussian, log otherwise
            .build();

        let fitted = model.fit(&x, &y).expect("fit");

        println!(
            "{:<12} {:>8.1} {:>12.4} {:>12.4} {:>12.2}",
            name,
            vp,
            fitted.intercept().unwrap_or(0.0),
            fitted.coefficients()[0],
            fitted.result().aic
        );
    }

    println!("\nVariance functions V(mu) = mu^p:");
    println!("  p=0: Var = constant (Gaussian)");
    println!("  p=1: Var = mu (Poisson)");
    println!("  p=1.5: Var = mu^1.5 (Compound Poisson-Gamma)");
    println!("  p=2: Var = mu^2 (Gamma, constant CV)");
    println!("  p=3: Var = mu^3 (Inverse Gaussian)");

    println!("\nNote: Choose var_power based on how variance changes with mean.");
    println!("      Lower AIC suggests better fit for the data.");
}
