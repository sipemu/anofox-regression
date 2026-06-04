//! # Logistic Regression
//!
//! Binary classification via the logit link. Returns class labels from
//! `predict`, probabilities from `predict_proba`, and accuracy from
//! `score` — matching sklearn's `LogisticRegression` API.
//!
//! Run with: `cargo run --example logistic`

use anofox_regression::solvers::{LogisticRegression, Penalty};
use faer::{Col, Mat};

fn main() {
    println!("=== Logistic Regression ===\n");

    // Two-feature dataset linearly separable by sign of x1 + 0.5 x2 - 1.
    let n = 80;
    let x = Mat::from_fn(n, 2, |i, j| {
        let t = i as f64 / 10.0;
        if j == 0 {
            (t * 0.7).sin()
        } else {
            (t * 0.3).cos()
        }
    });
    let y = Col::from_fn(n, |i| {
        let logit = x[(i, 0)] + 0.5 * x[(i, 1)] - 0.3;
        if logit > 0.0 {
            1.0
        } else {
            0.0
        }
    });

    let fitted = LogisticRegression::builder()
        .penalty(Penalty::L2(1.0))
        .with_intercept(true)
        .threshold(0.5)
        .max_iterations(100)
        .tolerance(1e-8)
        .build()
        .fit(&x, &y)
        .expect("Logistic fit failed");

    println!("Intercept: {:.4}", fitted.intercept().unwrap());
    println!(
        "Coefficients: [{:.4}, {:.4}]",
        fitted.coefficients()[0],
        fitted.coefficients()[1]
    );
    println!("IRLS iterations: {}", fitted.n_iter());

    let proba = fitted.predict_proba(&x);
    let preds = fitted.predict(&x);
    let acc = fitted.score(&x, &y);

    println!("\nAccuracy on training data: {:.3}", acc);
    println!(
        "First 5 P(Y=1|X): [{:.3}, {:.3}, {:.3}, {:.3}, {:.3}]",
        proba[0], proba[1], proba[2], proba[3], proba[4]
    );
    println!(
        "First 5 predicted labels: [{}, {}, {}, {}, {}]",
        preds[0] as u8, preds[1] as u8, preds[2] as u8, preds[3] as u8, preds[4] as u8,
    );
}
