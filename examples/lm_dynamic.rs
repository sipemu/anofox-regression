//! # Dynamic Linear Models (LmDynamic)
//!
//! Dynamic linear models allow regression coefficients to vary over time,
//! using information criteria to weight different model specifications.
//!
//! ## When to Use
//! - Time-varying relationships
//! - Structural breaks in data
//! - Regime changes
//! - Non-stationary parameters
//!
//! ## Key Features
//! - Information criteria weighting (AIC, AICc, BIC)
//! - LOWESS smoothing of weights
//! - Multiple model specifications
//! - Time-varying coefficient estimates
//!
//! Run with: `cargo run --example lm_dynamic`

use anofox_regression::solvers::{InformationCriterion, LmDynamicRegressor, Regressor};
use faer::{Col, Mat};

fn main() {
    println!("=== Dynamic Linear Models ===\n");

    basic_dynamic_model();
    time_varying_coefficients();
    information_criteria();
}

/// Basic dynamic linear model
fn basic_dynamic_model() {
    println!("--- Basic Dynamic Model ---\n");

    let n = 100;
    let x = Mat::from_fn(n, 2, |i, j| {
        if j == 0 {
            (i as f64) * 0.1
        } else {
            ((i as f64) * 0.15).sin()
        }
    });

    // Relationship that changes over time
    let y = Col::from_fn(n, |i| {
        let t = i as f64 / n as f64;
        // Coefficients change: beta1 goes from 2 to 1, beta2 from 0.5 to 1.5
        let beta1 = 2.0 - t * 1.0;
        let beta2 = 0.5 + t * 1.0;
        let noise = ((i as f64 * 0.7).sin()) * 0.3;
        1.0 + beta1 * x[(i, 0)] + beta2 * x[(i, 1)] + noise
    });

    let model = LmDynamicRegressor::builder()
        .ic(InformationCriterion::AICc)
        .lowess_span(0.3)
        .build();

    let fitted = model.fit(&x, &y).expect("fit should succeed");

    println!("True model: coefficients vary linearly from start to end");
    println!("  beta1: 2.0 -> 1.0");
    println!("  beta2: 0.5 -> 1.5\n");

    // Get dynamic coefficients
    let dyn_coefs = fitted.dynamic_coefficients();

    println!("Sample of estimated time-varying coefficients:\n");
    println!(
        "{:>8} {:>12} {:>12} {:>12} {:>12}",
        "Time", "True b1", "Est b1", "True b2", "Est b2"
    );
    println!("{}", "-".repeat(60));

    for i in [0, 25, 50, 75, 99] {
        let t = i as f64 / n as f64;
        let true_b1 = 2.0 - t * 1.0;
        let true_b2 = 0.5 + t * 1.0;
        // Dynamic coefficients include intercept at column 0
        let est_b1 = if dyn_coefs.ncols() > 1 {
            dyn_coefs[(i, 1)]
        } else {
            f64::NAN
        };
        let est_b2 = if dyn_coefs.ncols() > 2 {
            dyn_coefs[(i, 2)]
        } else {
            f64::NAN
        };

        println!(
            "{:>8} {:>12.4} {:>12.4} {:>12.4} {:>12.4}",
            i, true_b1, est_b1, true_b2, est_b2
        );
    }

    println!("\nNote: Dynamic models track time-varying relationships.");
    println!();
}

/// Examining time-varying coefficients
fn time_varying_coefficients() {
    println!("--- Time-Varying Coefficients ---\n");

    // Create data with a structural break
    let n = 150;
    let x = Mat::from_fn(n, 1, |i, _| (i as f64) * 0.1);

    // Structural break at observation 75
    let y = Col::from_fn(n, |i| {
        let coef = if i < 75 { 2.0 } else { 0.5 }; // Coefficient drops
        let noise = ((i as f64 * 0.9).sin()) * 0.3;
        1.0 + coef * x[(i, 0)] + noise
    });

    let model = LmDynamicRegressor::builder()
        .ic(InformationCriterion::AICc)
        .lowess_span(0.2)
        .build();

    let fitted = model.fit(&x, &y).expect("fit should succeed");
    let dyn_coefs = fitted.dynamic_coefficients();

    println!("Data with structural break at t=75:");
    println!("  Coefficient before: 2.0");
    println!("  Coefficient after:  0.5\n");

    println!("Estimated coefficients around the break:\n");
    println!("{:>8} {:>15} {:>15}", "Time", "True Coef", "Estimated Coef");
    println!("{}", "-".repeat(42));

    for i in [50, 60, 70, 80, 90, 100] {
        let true_coef = if i < 75 { 2.0 } else { 0.5 };
        let est_coef = if dyn_coefs.ncols() > 1 {
            dyn_coefs[(i, 1)]
        } else {
            f64::NAN
        };

        println!("{:>8} {:>15.4} {:>15.4}", i, true_coef, est_coef);
    }

    println!("\nNote: The dynamic model detects the structural break");
    println!("      and adjusts coefficients accordingly.");
    println!();
}

/// Different information criteria
fn information_criteria() {
    println!("--- Information Criteria Comparison ---\n");

    let n = 80;
    let x = Mat::from_fn(n, 1, |i, _| (i as f64) * 0.1);

    let y = Col::from_fn(n, |i| {
        let t = i as f64 / n as f64;
        let coef = 1.5 + t * 0.5; // Slowly increasing coefficient
        let noise = ((i as f64 * 0.8).sin()) * 0.4;
        2.0 + coef * x[(i, 0)] + noise
    });

    let criteria = [
        InformationCriterion::AIC,
        InformationCriterion::AICc,
        InformationCriterion::BIC,
    ];
    let names = ["AIC", "AICc", "BIC"];

    println!("Comparing information criteria for model weighting:\n");

    for (&ic, &name) in criteria.iter().zip(names.iter()) {
        let model = LmDynamicRegressor::builder()
            .ic(ic)
            .lowess_span(0.3)
            .build();

        let fitted = model.fit(&x, &y).expect("fit");
        let dyn_coefs = fitted.dynamic_coefficients();

        // Get coefficients at start, middle, end
        let coef_start = if dyn_coefs.ncols() > 1 {
            dyn_coefs[(0, 1)]
        } else {
            f64::NAN
        };
        let coef_mid = if dyn_coefs.ncols() > 1 {
            dyn_coefs[(n / 2, 1)]
        } else {
            f64::NAN
        };
        let coef_end = if dyn_coefs.ncols() > 1 {
            dyn_coefs[(n - 1, 1)]
        } else {
            f64::NAN
        };

        println!(
            "{}: coef at t=0: {:.4}, t=40: {:.4}, t=79: {:.4}",
            name, coef_start, coef_mid, coef_end
        );
    }

    println!("\nTrue coefficients: 1.5 (t=0), 1.75 (t=40), 2.0 (t=79)");

    println!("\nInformation criteria:");
    println!("  AIC:  Standard Akaike IC");
    println!("  AICc: Corrected AIC (better for small samples)");
    println!("  BIC:  Bayesian IC (penalizes complexity more)");
}
