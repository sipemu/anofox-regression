//! # Binomial/Logistic Regression
//!
//! Binomial regression models binary outcomes (0/1) or proportions,
//! predicting the probability of success given predictors.
//!
//! ## When to Use
//! - Binary outcomes (yes/no, success/failure)
//! - Proportion data (successes out of trials)
//! - Classification with probability outputs
//! - Risk/probability modeling
//!
//! ## Key Features
//! - Logit link (logistic regression): log(p/(1-p)) = X*beta
//! - Probit link: Phi^(-1)(p) = X*beta
//! - Complementary log-log link: log(-log(1-p)) = X*beta
//!
//! Run with: `cargo run --example binomial`

use anofox_regression::solvers::{BinomialRegressor, FittedRegressor, Regressor};
use faer::{Col, Mat};

fn main() {
    println!("=== Binomial/Logistic Regression ===\n");

    logistic_regression();
    link_function_comparison();
    predict_probabilities();
    classification_example();
}

/// Basic logistic regression
fn logistic_regression() {
    println!("--- Logistic Regression ---\n");

    let n = 100;
    let x = Mat::from_fn(n, 2, |i, j| {
        if j == 0 {
            (i as f64 - 50.0) * 0.1 // Centered predictor
        } else {
            ((i as f64) * 0.15).sin()
        }
    });

    // Generate binary outcomes
    // True model: logit(p) = -0.5 + 1.0*x1 + 0.5*x2
    let y = Col::from_fn(n, |i| {
        let eta = -0.5 + 1.0 * x[(i, 0)] + 0.5 * x[(i, 1)];
        let p = 1.0 / (1.0 + (-eta).exp());
        // Deterministic threshold with noise
        let threshold = 0.5 + ((i as f64 * 0.7).sin()) * 0.2;
        if p > threshold {
            1.0
        } else {
            0.0
        }
    });

    let model = BinomialRegressor::logistic().build();
    let fitted = model.fit(&x, &y).expect("fit should succeed");
    let result = fitted.result();

    println!("True model: logit(p) = -0.5 + 1.0*x1 + 0.5*x2\n");
    println!(
        "Intercept: {:.4} (true: -0.5)",
        fitted.intercept().unwrap_or(0.0)
    );
    println!(
        "Coefficient x1: {:.4} (true: 1.0)",
        fitted.coefficients()[0]
    );
    println!(
        "Coefficient x2: {:.4} (true: 0.5)",
        fitted.coefficients()[1]
    );
    println!("\nDeviance: {:.4}", result.log_likelihood * -2.0);
    println!("AIC: {:.4}", result.aic);

    // Odds ratio interpretation
    println!("\nOdds Ratio interpretation:");
    let or_x1 = fitted.coefficients()[0].exp();
    println!(
        "  OR for x1: {:.4} (1 unit increase multiplies odds by {:.2}x)",
        or_x1, or_x1
    );
    println!();
}

/// Compare logit, probit, and cloglog links
fn link_function_comparison() {
    println!("--- Link Function Comparison ---\n");

    let n = 120;
    let x = Mat::from_fn(n, 1, |i, _| (i as f64 - 60.0) * 0.05);

    // Generate binary data
    let y = Col::from_fn(n, |i| {
        let p = 1.0 / (1.0 + (-0.3 - 0.8 * x[(i, 0)]).exp());
        let threshold = 0.5 + ((i as f64 * 0.9).sin()) * 0.15;
        if p > threshold {
            1.0
        } else {
            0.0
        }
    });

    // Logit link (logistic regression)
    let logit_model = BinomialRegressor::logistic().build();
    let logit_fit = logit_model.fit(&x, &y).expect("logit fit");

    // Probit link
    let probit_model = BinomialRegressor::probit().build();
    let probit_fit = probit_model.fit(&x, &y).expect("probit fit");

    // Complementary log-log link
    let cloglog_model = BinomialRegressor::cloglog().build();
    let cloglog_fit = cloglog_model.fit(&x, &y).expect("cloglog fit");

    println!(
        "{:<12} {:>12} {:>12} {:>12}",
        "Link", "Intercept", "Slope", "AIC"
    );
    println!("{}", "-".repeat(52));

    println!(
        "{:<12} {:>12.4} {:>12.4} {:>12.4}",
        "Logit",
        logit_fit.intercept().unwrap_or(0.0),
        logit_fit.coefficients()[0],
        logit_fit.result().aic
    );
    println!(
        "{:<12} {:>12.4} {:>12.4} {:>12.4}",
        "Probit",
        probit_fit.intercept().unwrap_or(0.0),
        probit_fit.coefficients()[0],
        probit_fit.result().aic
    );
    println!(
        "{:<12} {:>12.4} {:>12.4} {:>12.4}",
        "CLogLog",
        cloglog_fit.intercept().unwrap_or(0.0),
        cloglog_fit.coefficients()[0],
        cloglog_fit.result().aic
    );

    println!("\nNote: Coefficients differ across link functions but give similar predictions.");
    println!("      Probit coefficients are roughly logit/1.7 due to scaling.");
    println!();
}

/// Predict probabilities
fn predict_probabilities() {
    println!("--- Predicting Probabilities ---\n");

    use anofox_regression::core::IntervalType;

    let n = 80;
    let x = Mat::from_fn(n, 1, |i, _| (i as f64 - 40.0) * 0.1);

    let y = Col::from_fn(n, |i| {
        let p = 1.0 / (1.0 + (-(0.2 + 0.5 * x[(i, 0)])).exp());
        let threshold = 0.5 + ((i as f64 * 1.1).sin()) * 0.2;
        if p > threshold {
            1.0
        } else {
            0.0
        }
    });

    let model = BinomialRegressor::logistic().build();
    let fitted = model.fit(&x, &y).expect("fit should succeed");

    // Predict on new data
    let x_new = Mat::from_fn(7, 1, |i, _| -3.0 + (i as f64));

    // Get probabilities
    let probabilities = fitted.predict_probability(&x_new);

    // With confidence intervals on probability scale
    let pred_ci = fitted.predict_with_interval(&x_new, Some(IntervalType::Confidence), 0.95);

    println!("Predicted probabilities for new x values:\n");
    println!(
        "{:>8} {:>12} {:>12} {:>12}",
        "x", "P(Y=1)", "Lower 95%", "Upper 95%"
    );
    println!("{}", "-".repeat(48));

    for i in 0..7 {
        println!(
            "{:>8.1} {:>12.4} {:>12.4} {:>12.4}",
            x_new[(i, 0)],
            probabilities[i],
            pred_ci.lower[i],
            pred_ci.upper[i]
        );
    }

    println!("\nNote: Probabilities are bounded between 0 and 1.");
    println!();
}

/// Classification example with decision threshold
fn classification_example() {
    println!("--- Classification Example ---\n");

    // Simulate a classification problem
    let n = 150;
    let x = Mat::from_fn(n, 2, |i, j| {
        let base = if j == 0 {
            (i as f64 - 75.0) * 0.08
        } else {
            ((i as f64) * 0.1).sin() * 2.0
        };
        base + ((i as f64 * (j as f64 + 1.0) * 0.5).sin()) * 0.3
    });

    // Generate classes with some overlap
    let y = Col::from_fn(n, |i| {
        let score = 0.5 * x[(i, 0)] + 0.3 * x[(i, 1)];
        let noise = ((i as f64 * 0.8).sin()) * 0.5;
        if score + noise > 0.0 {
            1.0
        } else {
            0.0
        }
    });

    // Split into train/test
    let n_train = 100;
    let x_train = Mat::from_fn(n_train, 2, |i, j| x[(i, j)]);
    let y_train = Col::from_fn(n_train, |i| y[i]);
    let x_test = Mat::from_fn(n - n_train, 2, |i, j| x[(i + n_train, j)]);
    let y_test = Col::from_fn(n - n_train, |i| y[i + n_train]);

    let model = BinomialRegressor::logistic().build();
    let fitted = model.fit(&x_train, &y_train).expect("fit should succeed");

    // Predict on test set
    let test_probs = fitted.predict_probability(&x_test);

    // Classification metrics at threshold = 0.5
    let threshold = 0.5;
    let mut tp = 0;
    let mut tn = 0;
    let mut fp = 0;
    let mut fn_ = 0;

    for i in 0..y_test.nrows() {
        let predicted = if test_probs[i] > threshold { 1.0 } else { 0.0 };
        let actual = y_test[i];

        if predicted == 1.0 && actual == 1.0 {
            tp += 1;
        } else if predicted == 0.0 && actual == 0.0 {
            tn += 1;
        } else if predicted == 1.0 && actual == 0.0 {
            fp += 1;
        } else {
            fn_ += 1;
        }
    }

    let accuracy = (tp + tn) as f64 / y_test.nrows() as f64;
    let precision = if tp + fp > 0 {
        tp as f64 / (tp + fp) as f64
    } else {
        0.0
    };
    let recall = if tp + fn_ > 0 {
        tp as f64 / (tp + fn_) as f64
    } else {
        0.0
    };

    println!("Classification on test set (threshold = {}):\n", threshold);
    println!("Confusion Matrix:");
    println!("                Predicted");
    println!("              0       1");
    println!("Actual  0    {:>3}     {:>3}", tn, fp);
    println!("        1    {:>3}     {:>3}", fn_, tp);

    println!("\nMetrics:");
    println!("  Accuracy:  {:.2}%", accuracy * 100.0);
    println!("  Precision: {:.2}%", precision * 100.0);
    println!("  Recall:    {:.2}%", recall * 100.0);

    // Show different thresholds
    println!("\nAccuracy at different thresholds:");
    for &t in &[0.3, 0.4, 0.5, 0.6, 0.7] {
        let correct: usize = (0..y_test.nrows())
            .filter(|&i| {
                let pred = if test_probs[i] > t { 1.0 } else { 0.0 };
                (pred - y_test[i]).abs() < 0.01
            })
            .count();
        println!(
            "  Threshold {:.1}: {:.2}%",
            t,
            100.0 * correct as f64 / y_test.nrows() as f64
        );
    }
}
