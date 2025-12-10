//! # Augmented Linear Model (ALM)
//!
//! ALM provides a unified framework for regression with many distribution
//! families and flexible loss functions for robust estimation.
//!
//! ## When to Use
//! - Non-normal error distributions
//! - Heavy-tailed data requiring robust estimation
//! - Skewed positive data (Gamma, Log-Normal)
//! - Count data (Poisson, Negative Binomial)
//! - Bounded data (Beta for proportions)
//!
//! ## Key Features
//! - 27+ distribution families
//! - Multiple loss functions (MLE, MAE, MSE, ROLE)
//! - Flexible link functions
//! - Maximum likelihood estimation
//!
//! Run with: `cargo run --example alm`

use anofox_regression::solvers::{
    AlmDistribution, AlmLoss, AlmRegressor, FittedRegressor, Regressor,
};
use faer::{Col, Mat};

fn main() {
    println!("=== Augmented Linear Model (ALM) ===\n");

    normal_distribution();
    laplace_robust();
    student_t_heavy_tails();
    gamma_positive_data();
    loss_functions();
}

/// Normal distribution (equivalent to OLS)
fn normal_distribution() {
    println!("--- Normal Distribution ---\n");

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

    let model = AlmRegressor::builder()
        .distribution(AlmDistribution::Normal)
        .with_intercept(true)
        .build();

    let fitted = model.fit(&x, &y).expect("fit should succeed");

    println!("True model: y = 2 + 1.5*x1 - 0.8*x2 + N(0, 0.5) noise\n");
    println!("Intercept: {:.4}", fitted.intercept().unwrap_or(0.0));
    println!("Coefficient x1: {:.4}", fitted.coefficients()[0]);
    println!("Coefficient x2: {:.4}", fitted.coefficients()[1]);
    println!("Scale (sigma): {:.4}", fitted.scale());
    println!("Log-likelihood: {:.4}", fitted.result().log_likelihood);

    println!("\nNote: Normal ALM is equivalent to OLS regression.");
    println!();
}

/// Laplace distribution for robust regression (LAD)
fn laplace_robust() {
    println!("--- Laplace Distribution (Robust to Outliers) ---\n");

    let n = 100;
    let x = Mat::from_fn(n, 1, |i, _| (i as f64) * 0.1);

    // Generate data with outliers
    let y = Col::from_fn(n, |i| {
        let base = 1.0 + 2.0 * x[(i, 0)];
        let noise = ((i as f64 * 0.8).sin()) * 0.3;

        // Add outliers at specific positions
        let outlier = if i % 20 == 0 { 5.0 } else { 0.0 };
        base + noise + outlier
    });

    // Fit with Normal (non-robust)
    let normal_model = AlmRegressor::builder()
        .distribution(AlmDistribution::Normal)
        .build();
    let normal_fit = normal_model.fit(&x, &y).expect("normal fit");

    // Fit with Laplace (robust)
    let laplace_model = AlmRegressor::builder()
        .distribution(AlmDistribution::Laplace)
        .build();
    let laplace_fit = laplace_model.fit(&x, &y).expect("laplace fit");

    println!("True model: y = 1 + 2*x + noise (with outliers)\n");

    println!("{:<12} {:>12} {:>12}", "Distribution", "Intercept", "Slope");
    println!("{}", "-".repeat(40));
    println!("{:<12} {:>12.4} {:>12.4}", "True", 1.0, 2.0);
    println!(
        "{:<12} {:>12.4} {:>12.4}",
        "Normal",
        normal_fit.intercept().unwrap_or(0.0),
        normal_fit.coefficients()[0]
    );
    println!(
        "{:<12} {:>12.4} {:>12.4}",
        "Laplace",
        laplace_fit.intercept().unwrap_or(0.0),
        laplace_fit.coefficients()[0]
    );

    println!("\nNote: Laplace (LAD regression) is less affected by outliers.");
    println!();
}

/// Student-t for heavy-tailed data
fn student_t_heavy_tails() {
    println!("--- Student-t Distribution (Heavy Tails) ---\n");

    let n = 100;
    let x = Mat::from_fn(n, 1, |i, _| (i as f64 - 50.0) * 0.1);

    // Generate heavy-tailed data
    let y = Col::from_fn(n, |i| {
        let base = 0.5 + 1.5 * x[(i, 0)];
        // Heavy-tailed noise
        let noise = ((i as f64 * 0.6).sin()) * 0.5 * (1.0 + (i % 10) as f64 * 0.2);
        base + noise
    });

    // Fit with different distributions
    let normal = AlmRegressor::builder()
        .distribution(AlmDistribution::Normal)
        .build();
    let normal_fit = normal.fit(&x, &y).expect("normal");

    let student_3 = AlmRegressor::builder()
        .distribution(AlmDistribution::StudentT)
        .extra_parameter(3.0) // df = 3
        .build();
    let student_3_fit = student_3.fit(&x, &y).expect("student-t df=3");

    let student_5 = AlmRegressor::builder()
        .distribution(AlmDistribution::StudentT)
        .extra_parameter(5.0) // df = 5
        .build();
    let student_5_fit = student_5.fit(&x, &y).expect("student-t df=5");

    println!("Comparison of distributions for heavy-tailed data:\n");
    println!(
        "{:<15} {:>10} {:>10} {:>12}",
        "Distribution", "Intercept", "Slope", "Scale"
    );
    println!("{}", "-".repeat(50));

    println!(
        "{:<15} {:>10.4} {:>10.4} {:>12.4}",
        "Normal",
        normal_fit.intercept().unwrap_or(0.0),
        normal_fit.coefficients()[0],
        normal_fit.scale()
    );
    println!(
        "{:<15} {:>10.4} {:>10.4} {:>12.4}",
        "Student-t (df=3)",
        student_3_fit.intercept().unwrap_or(0.0),
        student_3_fit.coefficients()[0],
        student_3_fit.scale()
    );
    println!(
        "{:<15} {:>10.4} {:>10.4} {:>12.4}",
        "Student-t (df=5)",
        student_5_fit.intercept().unwrap_or(0.0),
        student_5_fit.coefficients()[0],
        student_5_fit.scale()
    );

    println!("\nNote: Lower df gives heavier tails, more robust to outliers.");
    println!("      As df -> infinity, Student-t approaches Normal.");
    println!();
}

/// Gamma distribution for positive continuous data
fn gamma_positive_data() {
    println!("--- Gamma Distribution (Positive Data) ---\n");

    let n = 100;
    let x = Mat::from_fn(n, 1, |i, _| (i as f64) * 0.05 + 0.1);

    // Generate positive data with variance proportional to mean squared
    let y = Col::from_fn(n, |i| {
        let eta = 0.5 + 0.3 * x[(i, 0)];
        let mu = eta.exp();
        let noise = ((i as f64 * 0.9).sin()) * (mu * 0.2);
        (mu + noise).max(0.01)
    });

    let model = AlmRegressor::builder()
        .distribution(AlmDistribution::Gamma)
        .with_intercept(true)
        .build();

    let fitted = model.fit(&x, &y).expect("fit should succeed");

    println!("Positive continuous data with Gamma distribution\n");
    println!("True model: log(mu) = 0.5 + 0.3*x\n");
    println!(
        "Intercept: {:.4} (true: 0.5)",
        fitted.intercept().unwrap_or(0.0)
    );
    println!("Coefficient: {:.4} (true: 0.3)", fitted.coefficients()[0]);
    println!("Scale: {:.4}", fitted.scale());
    println!("AIC: {:.4}", fitted.result().aic);

    // Show available distributions
    println!("\nOther available distributions for positive data:");
    println!("  - LogNormal: for log-transformed normal errors");
    println!("  - Exponential: for constant hazard rate");
    println!("  - InverseGaussian: for highly skewed positive data");
    println!();
}

/// Different loss functions
fn loss_functions() {
    println!("--- Loss Functions ---\n");

    let n = 100;
    let x = Mat::from_fn(n, 1, |i, _| (i as f64) * 0.1);

    // Data with outliers
    let y = Col::from_fn(n, |i| {
        let base = 1.0 + 1.5 * x[(i, 0)];
        let noise = ((i as f64 * 0.7).sin()) * 0.3;
        let outlier = if i % 15 == 0 { 4.0 } else { 0.0 };
        base + noise + outlier
    });

    println!("Data with outliers - comparing loss functions:\n");

    // MLE (default)
    let mle = AlmRegressor::builder()
        .distribution(AlmDistribution::Normal)
        .loss(AlmLoss::Likelihood)
        .build();
    let mle_fit = mle.fit(&x, &y).expect("mle");

    // MSE
    let mse = AlmRegressor::builder()
        .distribution(AlmDistribution::Normal)
        .loss(AlmLoss::MSE)
        .build();
    let mse_fit = mse.fit(&x, &y).expect("mse");

    // MAE (robust)
    let mae = AlmRegressor::builder()
        .distribution(AlmDistribution::Normal)
        .loss(AlmLoss::MAE)
        .build();
    let mae_fit = mae.fit(&x, &y).expect("mae");

    // ROLE (trimmed likelihood)
    let role = AlmRegressor::builder()
        .distribution(AlmDistribution::Normal)
        .role_trim(0.1) // Trim 10% worst observations
        .build();
    let role_fit = role.fit(&x, &y).expect("role");

    println!(
        "{:<20} {:>12} {:>12}",
        "Loss Function", "Intercept", "Slope"
    );
    println!("{}", "-".repeat(48));
    println!("{:<20} {:>12.4} {:>12.4}", "True", 1.0, 1.5);
    println!(
        "{:<20} {:>12.4} {:>12.4}",
        "Likelihood (MLE)",
        mle_fit.intercept().unwrap_or(0.0),
        mle_fit.coefficients()[0]
    );
    println!(
        "{:<20} {:>12.4} {:>12.4}",
        "MSE",
        mse_fit.intercept().unwrap_or(0.0),
        mse_fit.coefficients()[0]
    );
    println!(
        "{:<20} {:>12.4} {:>12.4}",
        "MAE (robust)",
        mae_fit.intercept().unwrap_or(0.0),
        mae_fit.coefficients()[0]
    );
    println!(
        "{:<20} {:>12.4} {:>12.4}",
        "ROLE (10% trim)",
        role_fit.intercept().unwrap_or(0.0),
        role_fit.coefficients()[0]
    );

    println!("\nLoss functions:");
    println!("  Likelihood: Maximum likelihood (default, efficient)");
    println!("  MSE: Mean squared error (same as OLS for Normal)");
    println!("  MAE: Mean absolute error (robust to outliers)");
    println!("  HAM: Half absolute moment (very robust)");
    println!("  ROLE: Trimmed likelihood (removes worst observations)");
}
