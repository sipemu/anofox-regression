//! WASM tests for anofox-regression-js.

#![cfg(target_arch = "wasm32")]

use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

use anofox_regression_js::regressors::{
    FittedIsotonic, FittedOls, FittedPoisson, FittedQuantile, FittedRidge, IsotonicRegressor,
    OlsRegressor, PoissonRegressor, QuantileRegressor, RidgeRegressor,
};

#[wasm_bindgen_test]
fn test_ols_basic() {
    let ols = OlsRegressor::new();

    // Simple linear data: y = 2 + 3*x
    let x: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y: Vec<f64> = vec![5.0, 8.0, 11.0, 14.0, 17.0];

    let fitted = ols.fit(&x, 5, 1, &y).expect("OLS fit should succeed");

    let r_squared = fitted.get_r_squared();
    assert!(r_squared > 0.99, "R-squared should be close to 1.0");

    let intercept = fitted.get_intercept().expect("Should have intercept");
    assert!(
        (intercept - 2.0).abs() < 0.1,
        "Intercept should be close to 2.0"
    );

    let coefficients = fitted.get_coefficients();
    assert!(
        (coefficients[0] - 3.0).abs() < 0.1,
        "Coefficient should be close to 3.0"
    );
}

#[wasm_bindgen_test]
fn test_ols_predict() {
    let ols = OlsRegressor::new();

    let x: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y: Vec<f64> = vec![5.0, 8.0, 11.0, 14.0, 17.0];

    let fitted = ols.fit(&x, 5, 1, &y).expect("OLS fit should succeed");

    // Predict for x = 6
    let x_new: Vec<f64> = vec![6.0];
    let predictions = fitted.predict(&x_new, 1).expect("Predict should succeed");

    // Expected: 2 + 3*6 = 20
    assert!(
        (predictions[0] - 20.0).abs() < 0.1,
        "Prediction should be close to 20.0"
    );
}

#[wasm_bindgen_test]
fn test_ridge_basic() {
    let mut ridge = RidgeRegressor::new();
    ridge.set_lambda(0.1);

    let x: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y: Vec<f64> = vec![5.0, 8.0, 11.0, 14.0, 17.0];

    let fitted = ridge.fit(&x, 5, 1, &y).expect("Ridge fit should succeed");

    let r_squared = fitted.get_r_squared();
    assert!(r_squared > 0.9, "R-squared should be high for this data");
}

#[wasm_bindgen_test]
fn test_quantile_median() {
    let mut quantile = QuantileRegressor::new();
    quantile.set_tau(0.5); // Median

    let x: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y: Vec<f64> = vec![5.0, 8.0, 11.0, 14.0, 17.0];

    let fitted = quantile
        .fit(&x, 5, 1, &y)
        .expect("Quantile fit should succeed");

    let coefficients = fitted.get_coefficients();
    assert!(coefficients.len() == 1, "Should have one coefficient");
}

#[wasm_bindgen_test]
fn test_isotonic_increasing() {
    let isotonic = IsotonicRegressor::new();

    let x: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y: Vec<f64> = vec![1.0, 3.0, 2.0, 4.0, 5.0]; // Has violations

    let fitted = isotonic.fit(&x, &y).expect("Isotonic fit should succeed");

    assert!(fitted.is_increasing(), "Should be increasing");

    let r_squared = fitted.get_r_squared();
    assert!(r_squared > 0.0, "R-squared should be positive");
}

#[wasm_bindgen_test]
fn test_poisson_basic() {
    let poisson = PoissonRegressor::new();

    // Count data
    let x: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y: Vec<f64> = vec![2.0, 5.0, 10.0, 20.0, 35.0];

    let fitted = poisson
        .fit(&x, 5, 1, &y)
        .expect("Poisson fit should succeed");

    let deviance = fitted.get_deviance();
    assert!(deviance >= 0.0, "Deviance should be non-negative");
}

#[wasm_bindgen_test]
fn test_version() {
    let version = anofox_regression_js::version();
    assert!(!version.is_empty(), "Version should not be empty");
    assert!(version.contains('.'), "Version should contain dots");
}
