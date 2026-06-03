//! BayesianRidge and ARDRegression validation against scikit-learn 1.5.2.
//!
//! Reference values come from
//! `validation/python/generate_bayesian_validation.py`.

#![allow(dead_code)]

use anofox_regression::solvers::{ArdRegression, BayesianRidge, FittedRegressor, Regressor};
use faer::{Col, Mat};

include!("fixtures/bayesian_validation.rs");

#[test]
fn bayesian_ridge_matches_sklearn() {
    let n = N_BR;
    let p = P_BR;
    let x = Mat::from_fn(n, p, |i, j| X_BR_FLAT[i * p + j]);
    let y = Col::from_fn(n, |i| Y_BR[i]);

    let fitted = BayesianRidge::builder()
        .fit_intercept(true)
        .max_iter(300)
        .tolerance(1e-3)
        .build()
        .fit(&x, &y)
        .unwrap();

    let intercept = fitted.result().intercept.unwrap();
    let coefs = &fitted.result().coefficients;

    let tol = 5e-3; // evidence updates land at the same fixed point modulo
                    // sklearn-vs-faer ordering of floating ops
    assert!(
        (intercept - EXPECTED_INTERCEPT_BR).abs() < tol,
        "intercept {} vs sklearn {}",
        intercept,
        EXPECTED_INTERCEPT_BR
    );
    for j in 0..p {
        assert!(
            (coefs[j] - EXPECTED_COEFS_BR[j]).abs() < tol,
            "coef[{}] {} vs sklearn {}",
            j,
            coefs[j],
            EXPECTED_COEFS_BR[j]
        );
    }

    // α, λ from the evidence maximisation.
    assert!(
        (fitted.alpha() - EXPECTED_ALPHA_BR).abs() < 0.5,
        "alpha {} vs sklearn {}",
        fitted.alpha(),
        EXPECTED_ALPHA_BR
    );
    assert!(
        (fitted.lambda() - EXPECTED_LAMBDA_BR).abs() < 0.05,
        "lambda {} vs sklearn {}",
        fitted.lambda(),
        EXPECTED_LAMBDA_BR
    );
}

#[test]
fn ard_matches_sklearn() {
    let n = N_ARD;
    let p = P_ARD;
    let x = Mat::from_fn(n, p, |i, j| X_ARD_FLAT[i * p + j]);
    let y = Col::from_fn(n, |i| Y_ARD[i]);

    let fitted = ArdRegression::builder()
        .fit_intercept(true)
        .max_iter(300)
        .tolerance(1e-3)
        .threshold_lambda(10_000.0)
        .build()
        .fit(&x, &y)
        .unwrap();

    let intercept = fitted.result().intercept.unwrap();
    let coefs = &fitted.result().coefficients;

    let coef_tol = 5e-2;
    assert!(
        (intercept - EXPECTED_INTERCEPT_ARD).abs() < coef_tol,
        "intercept {} vs sklearn {}",
        intercept,
        EXPECTED_INTERCEPT_ARD
    );
    for j in 0..p {
        assert!(
            (coefs[j] - EXPECTED_COEFS_ARD[j]).abs() < coef_tol,
            "coef[{}] {} vs sklearn {}",
            j,
            coefs[j],
            EXPECTED_COEFS_ARD[j]
        );
    }

    // Both implementations should prune features 3 and 4 (true coefs zero).
    // We just check the pruning happened — exact λ values vary because
    // λ_j growth depends on the precise iteration count when each feature
    // crosses threshold_lambda.
    let lambdas = fitted.lambdas();
    assert!(lambdas[3] > 10.0 || coefs[3].abs() < 0.05);
    assert!(lambdas[4] > 10.0 || coefs[4].abs() < 0.05);
}
