//! # Bounded Least Squares (BLS) / Non-Negative Least Squares (NNLS)
//!
//! BLS extends OLS by constraining coefficients to lie within specified bounds.
//! NNLS is a special case where all coefficients must be non-negative.
//!
//! ## When to Use
//! - Physical constraints require non-negative coefficients
//! - Domain knowledge imposes bounds on parameters
//! - Mixture/composition models (proportions must be >= 0)
//! - Portfolio allocation (no short selling)
//!
//! ## Key Features
//! - Box constraints: lower <= beta <= upper
//! - NNLS preset for non-negative constraints
//! - Lawson-Hanson active set algorithm
//! - Falls back to OLS when unconstrained
//!
//! Run with: `cargo run --example bls`

use anofox_regression::solvers::{BlsRegressor, FittedRegressor, Regressor};
use faer::{Col, Mat};

fn main() {
    println!("=== Bounded Least Squares (BLS) ===\n");

    non_negative_least_squares();
    box_constraints();
    mixture_model();
    compare_constrained_unconstrained();
}

/// Non-negative least squares (NNLS)
fn non_negative_least_squares() {
    println!("--- Non-Negative Least Squares (NNLS) ---\n");

    let n = 50;
    let x = Mat::from_fn(n, 3, |i, j| match j {
        0 => (i as f64) * 0.1,
        1 => ((i as f64) * 0.15).sin().abs(),
        2 => ((i as f64) * 0.1).cos().abs(),
        _ => 0.0,
    });

    // True model has one negative coefficient
    let y = Col::from_fn(n, |i| {
        let noise = ((i as f64 * 0.7).sin()) * 0.2;
        1.0 + 2.0 * x[(i, 0)] - 0.5 * x[(i, 1)] + 1.5 * x[(i, 2)] + noise
    });

    // Fit with NNLS constraint
    let model = BlsRegressor::nnls().with_intercept(true).build();
    let fitted = model.fit(&x, &y).expect("fit should succeed");

    println!("True model: y = 1 + 2*x1 - 0.5*x2 + 1.5*x3 + noise");
    println!("NNLS constraint: all coefficients >= 0\n");
    println!("Intercept: {:.4}", fitted.intercept().unwrap_or(0.0));
    println!(
        "Coefficient x1: {:.4} (true: 2.0)",
        fitted.coefficients()[0]
    );
    println!(
        "Coefficient x2: {:.4} (true: -0.5, constrained to 0)",
        fitted.coefficients()[1]
    );
    println!(
        "Coefficient x3: {:.4} (true: 1.5)",
        fitted.coefficients()[2]
    );
    println!("R-squared: {:.4}", fitted.r_squared());

    println!("\nNote: x2 coefficient is forced to 0 (or positive) due to NNLS constraint.");
    println!();
}

/// Custom box constraints
fn box_constraints() {
    println!("--- Custom Box Constraints ---\n");

    let n = 60;
    let x = Mat::from_fn(n, 2, |i, j| {
        if j == 0 {
            (i as f64) * 0.1
        } else {
            ((i as f64) * 0.2).sin()
        }
    });

    let y = Col::from_fn(n, |i| {
        let noise = ((i as f64 * 0.8).sin()) * 0.3;
        2.0 + 3.0 * x[(i, 0)] + 2.0 * x[(i, 1)] + noise
    });

    // Constraint: 0 <= x1 <= 2, -1 <= x2 <= 1
    let model = BlsRegressor::builder()
        .with_intercept(true)
        .lower_bounds(vec![0.0, -1.0])
        .upper_bounds(vec![2.0, 1.0])
        .build();

    let fitted = model.fit(&x, &y).expect("fit should succeed");

    println!("True model: y = 2 + 3*x1 + 2*x2 + noise");
    println!("Constraints: 0 <= x1 <= 2, -1 <= x2 <= 1\n");
    println!("Intercept: {:.4}", fitted.intercept().unwrap_or(0.0));
    println!(
        "Coefficient x1: {:.4} (true: 3.0, bounded to [0, 2])",
        fitted.coefficients()[0]
    );
    println!(
        "Coefficient x2: {:.4} (true: 2.0, bounded to [-1, 1])",
        fitted.coefficients()[1]
    );
    println!("R-squared: {:.4}", fitted.r_squared());

    println!("\nNote: Coefficients are capped at their constraint bounds.");
    println!();
}

/// Mixture model where coefficients are proportions
fn mixture_model() {
    println!("--- Mixture Model (Proportions) ---\n");

    // Simulate mixture of 4 components
    let n = 100;
    let n_components = 4;

    // Component spectra/signatures
    let components = Mat::from_fn(n, n_components, |i, j| match j {
        0 => (1.0 + (i as f64 * 0.1).sin()).max(0.0),
        1 => (1.5 - (i as f64 * 0.08).cos()).max(0.0),
        2 => ((i as f64 * 0.05).exp() / 100.0).min(2.0),
        3 => (2.0 - (i as f64 * 0.02)).max(0.1),
        _ => 0.0,
    });

    // True mixture proportions (sum approximately to 1)
    let true_props = [0.3, 0.4, 0.2, 0.1];

    // Observed mixture
    let y = Col::from_fn(n, |i| {
        let noise = ((i as f64 * 1.1).sin()) * 0.05;
        let mut val = 0.0;
        for j in 0..n_components {
            val += true_props[j] * components[(i, j)];
        }
        val + noise
    });

    // Use NNLS to recover proportions (must be non-negative)
    let model = BlsRegressor::nnls().with_intercept(false).build();
    let fitted = model.fit(&components, &y).expect("fit should succeed");

    println!("Recovering mixture proportions from observed signal");
    println!("(all proportions must be >= 0)\n");

    println!(
        "{:<12} {:>12} {:>12}",
        "Component", "True Prop", "Estimated"
    );
    println!("{}", "-".repeat(40));

    for i in 0..n_components {
        println!(
            "{:<12} {:>12.4} {:>12.4}",
            format!("Component {}", i + 1),
            true_props[i],
            fitted.coefficients()[i]
        );
    }

    let est_sum: f64 = fitted.coefficients().iter().sum();
    let true_sum: f64 = true_props.iter().sum();
    println!("\nSum of proportions:");
    println!("  True: {:.4}", true_sum);
    println!("  Estimated: {:.4}", est_sum);
    println!();
}

/// Compare constrained vs unconstrained solutions
fn compare_constrained_unconstrained() {
    println!("--- Constrained vs Unconstrained ---\n");

    use anofox_regression::solvers::OlsRegressor;

    let n = 80;
    let x = Mat::from_fn(n, 3, |i, j| match j {
        0 => (i as f64) * 0.1,
        1 => ((i as f64) * 0.12).sin(),
        2 => ((i as f64) * 0.08).cos(),
        _ => 0.0,
    });

    // Model where one true coefficient is negative
    let y = Col::from_fn(n, |i| {
        let noise = ((i as f64 * 0.7).sin()) * 0.3;
        1.0 + 1.5 * x[(i, 0)] - 0.8 * x[(i, 1)] + 0.3 * x[(i, 2)] + noise
    });

    // Unconstrained OLS
    let ols_model = OlsRegressor::builder().with_intercept(true).build();
    let ols_fitted = ols_model.fit(&x, &y).expect("OLS fit");

    // NNLS
    let nnls_model = BlsRegressor::nnls().with_intercept(true).build();
    let nnls_fitted = nnls_model.fit(&x, &y).expect("NNLS fit");

    // Custom bounds
    let bls_model = BlsRegressor::builder()
        .with_intercept(true)
        .lower_bounds(vec![0.0, -0.5, 0.0])
        .upper_bounds(vec![f64::INFINITY, 0.5, f64::INFINITY])
        .build();
    let bls_fitted = bls_model.fit(&x, &y).expect("BLS fit");

    println!("True model: y = 1 + 1.5*x1 - 0.8*x2 + 0.3*x3 + noise\n");

    println!(
        "{:<15} {:>10} {:>10} {:>10} {:>12}",
        "Method", "x1", "x2", "x3", "R-squared"
    );
    println!("{}", "-".repeat(60));

    println!(
        "{:<15} {:>10.4} {:>10.4} {:>10.4} {:>12.4}",
        "True", 1.5, -0.8, 0.3, 1.0
    );
    println!(
        "{:<15} {:>10.4} {:>10.4} {:>10.4} {:>12.4}",
        "OLS",
        ols_fitted.coefficients()[0],
        ols_fitted.coefficients()[1],
        ols_fitted.coefficients()[2],
        ols_fitted.r_squared()
    );
    println!(
        "{:<15} {:>10.4} {:>10.4} {:>10.4} {:>12.4}",
        "NNLS (>=0)",
        nnls_fitted.coefficients()[0],
        nnls_fitted.coefficients()[1],
        nnls_fitted.coefficients()[2],
        nnls_fitted.r_squared()
    );
    println!(
        "{:<15} {:>10.4} {:>10.4} {:>10.4} {:>12.4}",
        "BLS (custom)",
        bls_fitted.coefficients()[0],
        bls_fitted.coefficients()[1],
        bls_fitted.coefficients()[2],
        bls_fitted.r_squared()
    );

    println!("\nNote: OLS has best R-squared but may violate constraints.");
    println!("      NNLS forces x2 to 0 (can't be negative).");
    println!("      BLS with custom bounds [-0.5, 0.5] for x2 is a compromise.");
}
