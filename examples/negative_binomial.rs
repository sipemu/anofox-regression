//! # Negative Binomial Regression
//!
//! Negative Binomial regression extends Poisson regression to handle
//! overdispersed count data where variance exceeds the mean.
//!
//! ## When to Use
//! - Count data with overdispersion (Var > Mean)
//! - Clustered or heterogeneous count data
//! - Poisson model shows poor fit due to excess variance
//! - Zero-inflated counts (partial solution)
//!
//! ## Key Features
//! - Variance = mu + mu^2/theta (overdispersion parameter)
//! - As theta -> infinity, approaches Poisson
//! - Can estimate theta automatically or fix it
//! - Log link for count data
//!
//! Run with: `cargo run --example negative_binomial`

use anofox_regression::solvers::{FittedRegressor, NegativeBinomialRegressor, Regressor};
use faer::{Col, Mat};

fn main() {
    println!("=== Negative Binomial Regression ===\n");

    basic_negative_binomial();
    estimate_vs_fixed_theta();
    overdispersion_detection();
    compare_poisson_negbin();
}

/// Basic negative binomial regression
fn basic_negative_binomial() {
    println!("--- Basic Negative Binomial Regression ---\n");

    let n = 100;
    let x = Mat::from_fn(n, 2, |i, j| {
        if j == 0 {
            (i as f64) * 0.05
        } else {
            ((i as f64) * 0.1).sin() * 0.5 + 0.5
        }
    });

    // Generate overdispersed count data
    // True model: log(mu) = 0.5 + 0.4*x1 + 0.3*x2
    let y = Col::from_fn(n, |i| {
        let eta = 0.5 + 0.4 * x[(i, 0)] + 0.3 * x[(i, 1)];
        let mu = eta.exp();
        // Add overdispersion noise
        let extra_var = ((i as f64 * 1.3).sin()) * (mu * 0.5);
        (mu + extra_var).round().max(0.0)
    });

    let model = NegativeBinomialRegressor::builder()
        .estimate_theta(true)
        .build();

    let fitted = model.fit(&x, &y).expect("fit should succeed");
    let result = fitted.result();

    println!("True model: log(mu) = 0.5 + 0.4*x1 + 0.3*x2 (with overdispersion)\n");
    println!(
        "Intercept: {:.4} (true: 0.5)",
        fitted.intercept().unwrap_or(0.0)
    );
    println!(
        "Coefficient x1: {:.4} (true: 0.4)",
        fitted.coefficients()[0]
    );
    println!(
        "Coefficient x2: {:.4} (true: 0.3)",
        fitted.coefficients()[1]
    );
    println!("\nTheta (dispersion): {:.4}", fitted.theta);
    println!("AIC: {:.4}", result.aic);
    println!();
}

/// Compare estimated vs fixed theta
fn estimate_vs_fixed_theta() {
    println!("--- Estimated vs Fixed Theta ---\n");

    let n = 80;
    let x = Mat::from_fn(n, 1, |i, _| (i as f64) * 0.1);

    // Generate overdispersed counts
    let y = Col::from_fn(n, |i| {
        let mu = (1.0 + 0.3 * x[(i, 0)]).exp();
        let noise = ((i as f64 * 0.9).sin()) * (mu * 0.8);
        (mu + noise).round().max(0.0)
    });

    // Estimate theta
    let model_est = NegativeBinomialRegressor::builder()
        .estimate_theta(true)
        .build();
    let fit_est = model_est.fit(&x, &y).expect("fit with estimated theta");

    // Fixed theta values
    let thetas = [0.5, 1.0, 2.0, 5.0, 10.0, 100.0];

    println!(
        "{:<15} {:>12} {:>12} {:>12}",
        "Theta", "Intercept", "Slope", "AIC"
    );
    println!("{}", "-".repeat(55));

    println!(
        "{:<15} {:>12.4} {:>12.4} {:>12.4}",
        format!("Est ({:.2})", fit_est.theta),
        fit_est.intercept().unwrap_or(0.0),
        fit_est.coefficients()[0],
        fit_est.result().aic
    );

    for &theta in &thetas {
        let model = NegativeBinomialRegressor::with_theta(theta).build();
        let fitted = model.fit(&x, &y).expect("fit with fixed theta");

        println!(
            "{:<15} {:>12.4} {:>12.4} {:>12.4}",
            format!("Fixed ({})", theta),
            fitted.intercept().unwrap_or(0.0),
            fitted.coefficients()[0],
            fitted.result().aic
        );
    }

    println!("\nNote: As theta increases, model approaches Poisson.");
    println!("      Lower AIC indicates better fit.");
    println!();
}

/// Detecting overdispersion
fn overdispersion_detection() {
    println!("--- Overdispersion Detection ---\n");

    use anofox_regression::solvers::PoissonRegressor;

    let n = 100;
    let x = Mat::from_fn(n, 1, |i, _| (i as f64) * 0.08);

    // Generate highly overdispersed data
    let y = Col::from_fn(n, |i| {
        let mu = (0.8 + 0.25 * x[(i, 0)]).exp();
        // High variance noise
        let noise = ((i as f64 * 0.7).sin()) * (mu * 1.5);
        (mu + noise).round().max(0.0)
    });

    // Fit Poisson
    let poisson_model = PoissonRegressor::log().build();
    let poisson_fit = poisson_model.fit(&x, &y).expect("poisson fit");

    // Fit Negative Binomial
    let negbin_model = NegativeBinomialRegressor::builder()
        .estimate_theta(true)
        .build();
    let negbin_fit = negbin_model.fit(&x, &y).expect("negbin fit");

    // Calculate mean and variance of y
    let y_mean: f64 = y.iter().sum::<f64>() / n as f64;
    let y_var: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum::<f64>() / (n - 1) as f64;

    println!("Data summary:");
    println!("  Mean(y):     {:.4}", y_mean);
    println!("  Var(y):      {:.4}", y_var);
    println!(
        "  Var/Mean:    {:.4} (>1 indicates overdispersion)",
        y_var / y_mean
    );

    // Overdispersion ratio from fitted Poisson residuals
    let poisson_residuals = &poisson_fit.result().residuals;
    let poisson_fitted = &poisson_fit.result().fitted_values;
    let pearson_chi_sq: f64 = poisson_residuals
        .iter()
        .zip(poisson_fitted.iter())
        .map(|(&r, &mu)| r.powi(2) / mu.max(0.001))
        .sum();
    let dispersion_ratio = pearson_chi_sq / (n - 2) as f64;

    println!(
        "  Dispersion:  {:.4} (Pearson chi-sq / df)",
        dispersion_ratio
    );

    println!("\nModel comparison:");
    println!("{:<20} {:>12} {:>12}", "Model", "AIC", "Deviance/df");
    println!("{}", "-".repeat(48));

    let poisson_dev = poisson_fit.result().log_likelihood * -2.0;
    let negbin_dev = negbin_fit.result().log_likelihood * -2.0;

    println!(
        "{:<20} {:>12.4} {:>12.4}",
        "Poisson",
        poisson_fit.result().aic,
        poisson_dev / (n - 2) as f64
    );
    println!(
        "{:<20} {:>12.4} {:>12.4}",
        format!("NegBin (theta={:.2})", negbin_fit.theta),
        negbin_fit.result().aic,
        negbin_dev / (n - 3) as f64
    );

    println!("\nNote: Large Var/Mean ratio suggests overdispersion.");
    println!("      Negative Binomial often has lower AIC with overdispersed data.");
    println!();
}

/// Compare Poisson and Negative Binomial models
fn compare_poisson_negbin() {
    println!("--- Poisson vs Negative Binomial ---\n");

    use anofox_regression::core::IntervalType;
    use anofox_regression::solvers::PoissonRegressor;

    let n = 100;
    let x = Mat::from_fn(n, 1, |i, _| (i as f64) * 0.1);

    // Moderately overdispersed data
    let y = Col::from_fn(n, |i| {
        let mu = (1.5 + 0.2 * x[(i, 0)]).exp();
        let noise = ((i as f64 * 1.1).sin()) * (mu * 0.6);
        (mu + noise).round().max(0.0)
    });

    // Fit both models
    let poisson = PoissonRegressor::log().build();
    let poisson_fit = poisson.fit(&x, &y).expect("poisson");

    let negbin = NegativeBinomialRegressor::builder()
        .estimate_theta(true)
        .build();
    let negbin_fit = negbin.fit(&x, &y).expect("negbin");

    println!("Coefficient comparison:\n");
    println!("{:<15} {:>12} {:>12}", "Model", "Intercept", "Slope");
    println!("{}", "-".repeat(42));
    println!(
        "{:<15} {:>12.4} {:>12.4}",
        "Poisson",
        poisson_fit.intercept().unwrap_or(0.0),
        poisson_fit.coefficients()[0]
    );
    println!(
        "{:<15} {:>12.4} {:>12.4}",
        "Neg Binomial",
        negbin_fit.intercept().unwrap_or(0.0),
        negbin_fit.coefficients()[0]
    );

    // Prediction comparison
    let x_new = Mat::from_fn(5, 1, |i, _| (5 + i * 2) as f64);

    let pois_pred = poisson_fit.predict_with_interval(&x_new, Some(IntervalType::Confidence), 0.95);
    let nb_pred = negbin_fit.predict_with_interval(&x_new, Some(IntervalType::Confidence), 0.95);

    println!("\nPredictions with 95% CI:\n");
    println!(
        "{:>6} {:>10} {:>10} {:>10} {:>10}",
        "x", "Pois.Fit", "Pois.Width", "NB.Fit", "NB.Width"
    );
    println!("{}", "-".repeat(52));

    for i in 0..5 {
        let pois_width = pois_pred.upper[i] - pois_pred.lower[i];
        let nb_width = nb_pred.upper[i] - nb_pred.lower[i];
        println!(
            "{:>6.1} {:>10.2} {:>10.2} {:>10.2} {:>10.2}",
            x_new[(i, 0)],
            pois_pred.fit[i],
            pois_width,
            nb_pred.fit[i],
            nb_width
        );
    }

    println!("\nNote: Negative Binomial often has wider confidence intervals,");
    println!("      better reflecting uncertainty with overdispersed data.");
}
