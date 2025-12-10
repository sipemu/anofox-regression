//! # Ordinary Least Squares (OLS) Regression
//!
//! OLS is the most common linear regression method, minimizing the sum of squared
//! residuals to find the best-fitting line through data points.
//!
//! ## When to Use
//! - Linear relationship between predictors and response
//! - Homoscedastic errors (constant variance)
//! - No severe multicollinearity
//! - Normally distributed errors (for inference)
//!
//! ## Key Features
//! - Automatic detection of collinear/constant columns
//! - Confidence and prediction intervals
//! - Statistical inference (t-tests, F-test)
//! - R-squared and adjusted R-squared
//!
//! Run with: `cargo run --example ols`

use anofox_regression::core::IntervalType;
use anofox_regression::solvers::{FittedRegressor, OlsRegressor, Regressor};
use faer::{Col, Mat};

fn main() {
    println!("=== Ordinary Least Squares (OLS) Regression ===\n");

    simple_regression();
    multiple_regression();
    prediction_intervals();
    model_diagnostics();
}

/// Simple linear regression with one predictor
fn simple_regression() {
    println!("--- Simple Linear Regression ---\n");

    // Generate data: y = 2 + 3*x + noise
    let n = 50;
    let noise = [
        0.5, -0.3, 0.8, -0.2, 0.1, -0.6, 0.4, -0.1, 0.3, -0.5, 0.2, -0.4, 0.6, -0.3, 0.1, -0.2,
        0.4, -0.5, 0.3, -0.1, 0.5, -0.3, 0.2, -0.4, 0.1, -0.6, 0.4, -0.2, 0.3, -0.5, 0.2, -0.4,
        0.6, -0.3, 0.1, -0.2, 0.4, -0.5, 0.3, -0.1, 0.5, -0.3, 0.2, -0.4, 0.1, -0.6, 0.4, -0.2,
        0.3, -0.5,
    ];

    let x = Mat::from_fn(n, 1, |i, _| i as f64 * 0.2);
    let y = Col::from_fn(n, |i| 2.0 + 3.0 * (i as f64 * 0.2) + noise[i]);

    // Fit model
    let model = OlsRegressor::builder().with_intercept(true).build();
    let fitted = model.fit(&x, &y).expect("fit should succeed");

    // Display results
    println!("True model: y = 2 + 3*x + noise");
    println!(
        "Estimated intercept: {:.4}",
        fitted.intercept().unwrap_or(0.0)
    );
    println!("Estimated slope: {:.4}", fitted.coefficients()[0]);
    println!("R-squared: {:.4}", fitted.r_squared());
    println!("Adjusted R-squared: {:.4}", fitted.result().adj_r_squared);
    println!();
}

/// Multiple linear regression with several predictors
fn multiple_regression() {
    println!("--- Multiple Linear Regression ---\n");

    // Generate data: y = 1 + 2*x1 - 0.5*x2 + 0.8*x3 + noise
    // Use independent predictors to avoid collinearity
    let n = 100;
    let x = Mat::from_fn(n, 3, |i, j| match j {
        0 => (i as f64) * 0.1,                     // x1: linear trend
        1 => ((i as f64) * 0.3).sin() * 2.0,       // x2: oscillating (different frequency)
        2 => ((i as f64) * 0.7).cos() * 3.0 + 1.5, // x3: different oscillation, offset
        _ => 0.0,
    });

    let y = Col::from_fn(n, |i| {
        let noise = ((i as f64 * 1.3).sin()) * 0.2;
        1.0 + 2.0 * x[(i, 0)] - 0.5 * x[(i, 1)] + 0.8 * x[(i, 2)] + noise
    });

    // Fit model with inference
    let model = OlsRegressor::builder()
        .with_intercept(true)
        .confidence_level(0.95)
        .build();

    let fitted = model.fit(&x, &y).expect("fit should succeed");
    let result = fitted.result();

    println!("True model: y = 1 + 2*x1 - 0.5*x2 + 0.8*x3 + noise\n");

    if let Some(intercept_se) = result.intercept_std_error {
        println!(
            "Estimated intercept: {:.4} (SE: {:.4})",
            fitted.intercept().unwrap_or(0.0),
            intercept_se
        );
    }

    let var_names = ["x1", "x2", "x3"];
    let true_coefs = [2.0, -0.5, 0.8];

    println!("\nCoefficients:");
    println!(
        "{:<6} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "Var", "True", "Est", "SE", "t-stat", "p-value"
    );
    println!("{}", "-".repeat(66));

    let std_errors = result.std_errors.as_ref();
    let t_stats = result.t_statistics.as_ref();
    let p_vals = result.p_values.as_ref();

    for (i, name) in var_names.iter().enumerate() {
        let se = std_errors.map(|s| s[i]).unwrap_or(f64::NAN);
        let t = t_stats.map(|s| s[i]).unwrap_or(f64::NAN);
        let p = p_vals.map(|s| s[i]).unwrap_or(f64::NAN);

        println!(
            "{:<6} {:>10.4} {:>10.4} {:>10.4} {:>10.4} {:>10.4}",
            name,
            true_coefs[i],
            fitted.coefficients()[i],
            se,
            t,
            p
        );
    }

    println!("\nModel fit:");
    println!("  R-squared:          {:.4}", fitted.r_squared());
    println!("  Adjusted R-squared: {:.4}", result.adj_r_squared);
    println!(
        "  F-statistic:        {:.4} (p-value: {:.6})",
        result.f_statistic, result.f_pvalue
    );
    println!();
}

/// Demonstrate prediction with confidence and prediction intervals
fn prediction_intervals() {
    println!("--- Prediction Intervals ---\n");

    // Simple model for clarity
    let x = Mat::from_fn(30, 1, |i, _| i as f64);
    let noise = [
        0.5, -0.3, 0.8, -0.2, 0.1, -0.6, 0.4, -0.1, 0.3, -0.5, 0.2, -0.4, 0.6, -0.3, 0.1, -0.2,
        0.4, -0.5, 0.3, -0.1, 0.5, -0.3, 0.2, -0.4, 0.1, -0.6, 0.4, -0.2, 0.3, -0.5,
    ];
    let y = Col::from_fn(30, |i| 5.0 + 2.0 * (i as f64) + noise[i]);

    let model = OlsRegressor::builder()
        .with_intercept(true)
        .confidence_level(0.95)
        .build();
    let fitted = model.fit(&x, &y).expect("fit should succeed");

    // Predict on new data
    let x_new = Mat::from_fn(5, 1, |i, _| (30 + i * 2) as f64);

    // Confidence intervals (for mean response)
    let conf = fitted.predict_with_interval(&x_new, Some(IntervalType::Confidence), 0.95);

    // Prediction intervals (for individual observations)
    let pred = fitted.predict_with_interval(&x_new, Some(IntervalType::Prediction), 0.95);

    println!("Predictions for new x values (95% intervals):\n");
    println!(
        "{:>6} {:>10} {:>12} {:>12} {:>12} {:>12}",
        "x", "Fitted", "Conf.Lower", "Conf.Upper", "Pred.Lower", "Pred.Upper"
    );
    println!("{}", "-".repeat(76));

    for i in 0..5 {
        println!(
            "{:>6.0} {:>10.3} {:>12.3} {:>12.3} {:>12.3} {:>12.3}",
            x_new[(i, 0)],
            conf.fit[i],
            conf.lower[i],
            conf.upper[i],
            pred.lower[i],
            pred.upper[i]
        );
    }

    println!("\nNote: Prediction intervals are wider than confidence intervals");
    println!("      because they account for individual observation variability.");
    println!();
}

/// Display model diagnostics
fn model_diagnostics() {
    println!("--- Model Diagnostics ---\n");

    // Generate data with known properties
    let n = 50;
    let x = Mat::from_fn(
        n,
        2,
        |i, j| if j == 0 { i as f64 } else { (i as f64).sqrt() },
    );

    let y = Col::from_fn(n, |i| {
        let noise = ((i as f64 * 1.7).sin()) * 0.5;
        3.0 + 1.5 * x[(i, 0)] - 2.0 * x[(i, 1)] + noise
    });

    let model = OlsRegressor::builder().with_intercept(true).build();
    let fitted = model.fit(&x, &y).expect("fit should succeed");
    let result = fitted.result();

    println!("Regression diagnostics:");
    println!("  Number of observations: {}", result.n_observations);
    println!("  Number of parameters:   {}", result.n_parameters);
    println!("  Residual DF:            {}", result.residual_df());
    println!("  RMSE:                   {:.4}", result.rmse);
    println!();

    println!("Goodness of fit:");
    println!("  R-squared:              {:.4}", fitted.r_squared());
    println!("  Adjusted R-squared:     {:.4}", result.adj_r_squared);
    println!("  AIC:                    {:.4}", result.aic);
    println!("  AICc:                   {:.4}", result.aicc);
    println!("  BIC:                    {:.4}", result.bic);
    println!();

    println!("Residuals summary:");
    let residuals = &result.residuals;
    let min_resid = residuals.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_resid = residuals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mean_resid: f64 = residuals.iter().sum::<f64>() / n as f64;

    println!("  Min:  {:.4}", min_resid);
    println!("  Max:  {:.4}", max_resid);
    println!("  Mean: {:.4} (should be ~0)", mean_resid);
    println!();

    // Check for aliased columns
    if result.aliased.iter().any(|&a| a) {
        println!("Warning: Some coefficients are aliased (collinear):");
        for (i, &aliased) in result.aliased.iter().enumerate() {
            if aliased {
                println!("  Coefficient {} is aliased", i);
            }
        }
    } else {
        println!("No aliased (collinear) coefficients detected.");
    }
}
