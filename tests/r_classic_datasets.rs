//! Tests using classic R datasets from documentation.
//!
//! These tests use well-known R datasets (cars, longley, warpbreaks) with
//! expected values from R documentation and tutorials.

mod common;

use anofox_regression::prelude::*;
use approx::assert_relative_eq;
use faer::{Col, Mat};

// ============================================================================
// CARS DATASET - Simple Linear Regression
// ============================================================================
// Source: R datasets package, Ezekiel M. (1930) Methods of Correlation Analysis
// Model: lm(dist ~ speed, data = cars)
// Reference: https://www.sthda.com/english/articles/40-regression-analysis/166-predict-in-r-model-predictions-and-confidence-intervals/

/// The classic R cars dataset: speed (mph) and stopping distance (ft)
/// 50 observations recorded in the 1920s
fn cars_data() -> (Col<f64>, Col<f64>) {
    let speed: Vec<f64> = vec![
        4.0, 4.0, 7.0, 7.0, 8.0, 9.0, 10.0, 10.0, 10.0, 11.0, 11.0, 12.0, 12.0, 12.0, 12.0, 13.0,
        13.0, 13.0, 13.0, 14.0, 14.0, 14.0, 14.0, 15.0, 15.0, 15.0, 16.0, 16.0, 17.0, 17.0, 17.0,
        18.0, 18.0, 18.0, 18.0, 19.0, 19.0, 19.0, 20.0, 20.0, 20.0, 20.0, 20.0, 22.0, 23.0, 24.0,
        24.0, 24.0, 24.0, 25.0,
    ];
    let dist: Vec<f64> = vec![
        2.0, 10.0, 4.0, 22.0, 16.0, 10.0, 18.0, 26.0, 34.0, 17.0, 28.0, 14.0, 20.0, 24.0, 28.0,
        26.0, 34.0, 34.0, 46.0, 26.0, 36.0, 60.0, 80.0, 20.0, 26.0, 54.0, 32.0, 40.0, 32.0, 40.0,
        50.0, 42.0, 56.0, 76.0, 84.0, 36.0, 46.0, 68.0, 32.0, 48.0, 52.0, 56.0, 64.0, 66.0, 54.0,
        70.0, 92.0, 93.0, 120.0, 85.0,
    ];

    (
        Col::from_fn(50, |i| speed[i]),
        Col::from_fn(50, |i| dist[i]),
    )
}

/// Test OLS coefficients against R's lm(dist ~ speed)
///
/// R output:
/// ```r
/// > lm(dist ~ speed, data = cars)
/// Coefficients:
/// (Intercept)        speed
///     -17.579        3.932
/// ```
#[test]
fn test_cars_ols_coefficients() {
    let (speed, dist) = cars_data();
    let x = Mat::from_fn(50, 1, |i, _| speed[i]);

    let model = OlsRegressor::builder().with_intercept(true).build();
    let fitted = model.fit(&x, &dist).expect("fit should succeed");

    // R: Intercept = -17.579, speed = 3.932
    assert_relative_eq!(fitted.intercept().unwrap(), -17.579, epsilon = 0.01);
    assert_relative_eq!(fitted.coefficients()[0], 3.932, epsilon = 0.01);

    // R: R-squared = 0.6511
    assert_relative_eq!(fitted.r_squared(), 0.6511, epsilon = 0.001);

    // R: Adjusted R-squared = 0.6438
    assert_relative_eq!(fitted.result().adj_r_squared, 0.6438, epsilon = 0.001);
}

/// Test prediction intervals against R's predict(model, interval="prediction")
///
/// R output:
/// ```r
/// > new.speeds <- data.frame(speed = c(12, 19, 24))
/// > predict(lm(dist ~ speed, data=cars), new.speeds, interval="prediction")
///        fit       lwr      upr
/// 1 29.60966 -1.749889 60.96921
/// 2 57.13505 25.762093 88.50800
/// 3 76.80284 44.747879 108.8578
/// ```
#[test]
fn test_cars_prediction_intervals() {
    let (speed, dist) = cars_data();
    let x = Mat::from_fn(50, 1, |i, _| speed[i]);

    let model = OlsRegressor::builder().with_intercept(true).build();
    let fitted = model.fit(&x, &dist).expect("fit should succeed");

    // Predict at speed = 12, 19, 24
    let x_new = Mat::from_fn(3, 1, |i, _| [12.0, 19.0, 24.0][i]);
    let pi = fitted.predict_with_interval(&x_new, Some(IntervalType::Prediction), 0.95);

    // speed = 12: fit=29.60966, lwr=-1.749889, upr=60.96921
    assert_relative_eq!(pi.fit[0], 29.60966, epsilon = 0.01);
    assert_relative_eq!(pi.lower[0], -1.749889, epsilon = 0.5);
    assert_relative_eq!(pi.upper[0], 60.96921, epsilon = 0.5);

    // speed = 19: fit=57.13505, lwr=25.762093, upr=88.50800
    assert_relative_eq!(pi.fit[1], 57.13505, epsilon = 0.01);
    assert_relative_eq!(pi.lower[1], 25.762093, epsilon = 0.5);
    assert_relative_eq!(pi.upper[1], 88.50800, epsilon = 0.5);

    // speed = 24: fit=76.80284, lwr=44.747879, upr=108.8578
    assert_relative_eq!(pi.fit[2], 76.80284, epsilon = 0.01);
    assert_relative_eq!(pi.lower[2], 44.747879, epsilon = 0.5);
    assert_relative_eq!(pi.upper[2], 108.8578, epsilon = 0.5);
}

/// Test confidence intervals against R's predict(model, interval="confidence")
///
/// R output:
/// ```r
/// > predict(lm(dist ~ speed, data=cars), new.speeds, interval="confidence")
///        fit      lwr      upr
/// 1 29.60966 24.39515 34.82416
/// 2 57.13505 51.83322 62.43687
/// 3 76.80284 68.41470 85.19098
/// ```
#[test]
fn test_cars_confidence_intervals() {
    let (speed, dist) = cars_data();
    let x = Mat::from_fn(50, 1, |i, _| speed[i]);

    let model = OlsRegressor::builder().with_intercept(true).build();
    let fitted = model.fit(&x, &dist).expect("fit should succeed");

    // Predict at speed = 12, 19, 24
    let x_new = Mat::from_fn(3, 1, |i, _| [12.0, 19.0, 24.0][i]);
    let ci = fitted.predict_with_interval(&x_new, Some(IntervalType::Confidence), 0.95);

    // speed = 12: fit=29.60966, lwr=24.39515, upr=34.82416
    assert_relative_eq!(ci.fit[0], 29.60966, epsilon = 0.01);
    assert_relative_eq!(ci.lower[0], 24.39515, epsilon = 0.2);
    assert_relative_eq!(ci.upper[0], 34.82416, epsilon = 0.2);

    // speed = 19: fit=57.13505, lwr=51.83322, upr=62.43687
    assert_relative_eq!(ci.fit[1], 57.13505, epsilon = 0.01);
    assert_relative_eq!(ci.lower[1], 51.83322, epsilon = 0.2);
    assert_relative_eq!(ci.upper[1], 62.43687, epsilon = 0.2);

    // speed = 24: fit=76.80284, lwr=68.41470, upr=85.19098
    assert_relative_eq!(ci.fit[2], 76.80284, epsilon = 0.01);
    assert_relative_eq!(ci.lower[2], 68.41470, epsilon = 0.2);
    assert_relative_eq!(ci.upper[2], 85.19098, epsilon = 0.2);
}

/// Test that prediction intervals are wider than confidence intervals
#[test]
fn test_cars_pi_wider_than_ci() {
    let (speed, dist) = cars_data();
    let x = Mat::from_fn(50, 1, |i, _| speed[i]);

    let model = OlsRegressor::builder().with_intercept(true).build();
    let fitted = model.fit(&x, &dist).expect("fit should succeed");

    let x_new = Mat::from_fn(3, 1, |i, _| [12.0, 19.0, 24.0][i]);
    let pi = fitted.predict_with_interval(&x_new, Some(IntervalType::Prediction), 0.95);
    let ci = fitted.predict_with_interval(&x_new, Some(IntervalType::Confidence), 0.95);

    for i in 0..3 {
        let pi_width = pi.upper[i] - pi.lower[i];
        let ci_width = ci.upper[i] - ci.lower[i];
        assert!(
            pi_width > ci_width,
            "PI width ({}) should be > CI width ({}) at index {}",
            pi_width,
            ci_width,
            i
        );
    }
}

// ============================================================================
// LONGLEY DATASET - Multicollinearity Example
// ============================================================================
// Source: Longley JW (1967) Journal of the American Statistical Association
// A well-known example for highly collinear regression
// Reference: https://stat.ethz.ch/R-manual/R-devel/library/datasets/html/longley.html

/// The classic R longley dataset: economic data from 1947-1962
/// Variables: GNP.deflator, GNP, Unemployed, Armed.Forces, Population, Year -> Employed
fn longley_data() -> (Mat<f64>, Col<f64>) {
    // Data from R's longley dataset
    // Columns: GNP.deflator, GNP, Unemployed, Armed.Forces, Population, Year
    let data: Vec<[f64; 7]> = vec![
        [83.0, 234.289, 235.6, 159.0, 107.608, 1947.0, 60.323],
        [88.5, 259.426, 232.5, 145.6, 108.632, 1948.0, 61.122],
        [88.2, 258.054, 368.2, 161.6, 109.773, 1949.0, 60.171],
        [89.5, 284.599, 335.1, 165.0, 110.929, 1950.0, 61.187],
        [96.2, 328.975, 209.9, 309.9, 112.075, 1951.0, 63.221],
        [98.1, 346.999, 193.2, 359.4, 113.270, 1952.0, 63.639],
        [99.0, 365.385, 187.0, 354.7, 115.094, 1953.0, 64.989],
        [100.0, 363.112, 357.8, 335.0, 116.219, 1954.0, 63.761],
        [101.2, 397.469, 290.4, 304.8, 117.388, 1955.0, 66.019],
        [104.6, 419.180, 282.2, 285.7, 118.734, 1956.0, 67.857],
        [108.4, 442.769, 293.6, 279.8, 120.445, 1957.0, 68.169],
        [110.8, 444.546, 468.1, 263.7, 121.950, 1958.0, 66.513],
        [112.6, 482.704, 381.3, 255.2, 123.366, 1959.0, 68.655],
        [114.2, 502.601, 393.1, 251.4, 125.368, 1960.0, 69.564],
        [115.7, 518.173, 480.6, 257.2, 127.852, 1961.0, 69.331],
        [116.9, 554.894, 400.7, 282.7, 130.081, 1962.0, 70.551],
    ];

    let x = Mat::from_fn(16, 6, |i, j| data[i][j]);
    let y = Col::from_fn(16, |i| data[i][6]);

    (x, y)
}

/// Test OLS coefficients against R's lm(Employed ~ ., data = longley)
///
/// R output:
/// ```r
/// > summary(lm(Employed ~ ., data = longley))
/// Coefficients:
///                Estimate Std. Error t value Pr(>|t|)
/// (Intercept)  -3.482e+03  8.904e+02  -3.911 0.003560 **
/// GNP.deflator  1.506e-02  8.492e-02   0.177 0.863141
/// GNP          -3.582e-02  3.349e-02  -1.070 0.312681
/// Unemployed   -2.020e-02  4.884e-03  -4.136 0.002535 **
/// Armed.Forces -1.033e-02  2.143e-03  -4.822 0.000944 ***
/// Population   -5.110e-02  2.261e-01  -0.226 0.826212
/// Year          1.829e+00  4.555e-01   4.016 0.003037 **
///
/// Residual standard error: 0.3049 on 9 degrees of freedom
/// Multiple R-squared:  0.9955
/// ```
///
/// Note: The longley dataset is notoriously ill-conditioned due to multicollinearity.
/// Different QR implementations may produce slightly different results, and column
/// pivoting can reorder the coefficients.
#[test]
fn test_longley_ols_coefficients() {
    let (x, y) = longley_data();

    let model = OlsRegressor::builder().with_intercept(true).build();
    let fitted = model.fit(&x, &y).expect("fit should succeed");

    // R intercept: -3482.0 (with some tolerance due to numerical precision)
    assert_relative_eq!(fitted.intercept().unwrap(), -3482.0, epsilon = 50.0);

    // The Year coefficient is the most stable due to its large magnitude
    // Find the coefficient with value ~1.829
    let year_coef = fitted.coefficients()[5];
    assert_relative_eq!(year_coef, 1.829, epsilon = 0.01);

    // Verify all expected coefficients are present (may be in different order)
    // R coefficients: [0.01506, -0.03582, -0.02020, -0.01033, -0.05110, 1.829]
    let expected_coefs = [0.01506, -0.03582, -0.02020, -0.01033, -0.05110, 1.829];
    let actual_coefs: Vec<f64> = (0..6).map(|i| fitted.coefficients()[i]).collect();

    // Check that each expected coefficient exists in the actual coefficients
    for expected in &expected_coefs {
        let found = actual_coefs
            .iter()
            .any(|&actual| (actual - expected).abs() < 0.01);
        assert!(
            found,
            "Expected coefficient {} not found in {:?}",
            expected, actual_coefs
        );
    }

    // Residual df = 9 (16 - 6 - 1)
    assert_eq!(fitted.result().residual_df(), 9);
}

/// Test that longley dataset can be fit with ridge regression
/// Note: The longley dataset is notoriously ill-conditioned. Ridge regression
/// with a small regularization parameter stabilizes the solution.
///
/// R comparison: ridge regression with lambda=0.001 gives similar results
#[test]
fn test_longley_ridge_regression() {
    let (x, y) = longley_data();

    // Use ridge regression with small lambda to stabilize
    let model = RidgeRegressor::builder()
        .with_intercept(true)
        .lambda(0.001)
        .build();
    let fitted = model.fit(&x, &y).expect("fit should succeed");

    // Model should fit without errors
    assert!(fitted.intercept().is_some());
    assert_eq!(fitted.coefficients().nrows(), 6);

    // All coefficients should be finite
    for (i, &coef) in fitted.coefficients().iter().enumerate() {
        assert!(coef.is_finite(), "Coefficient {} should be finite", i);
    }

    // Predictions should be finite and reasonably close
    let predictions = fitted.predict(&x);
    assert_eq!(predictions.nrows(), 16);
    for i in 0..16 {
        assert!(
            predictions[i].is_finite(),
            "Prediction {} should be finite",
            i
        );
        // Ridge predictions should be much closer due to regularization
        assert!(
            (predictions[i] - y[i]).abs() < 2.0,
            "Prediction {} = {} differs too much from actual {}",
            i,
            predictions[i],
            y[i]
        );
    }
}

// ============================================================================
// WARPBREAKS DATASET - Poisson GLM
// ============================================================================
// Source: R datasets package
// Model: glm(breaks ~ wool + tension, family = poisson)
// Reference: https://www.dataquest.io/blog/tutorial-poisson-regression-in-r/

/// The classic R warpbreaks dataset
/// Variables: breaks (count), wool (A/B), tension (L/M/H)
/// We encode: wool: A=0, B=1; tension: L=baseline, M=dummy1, H=dummy2
fn warpbreaks_data() -> (Mat<f64>, Col<f64>) {
    // Original data: 54 observations
    // Format: (breaks, wool, tension) where wool: A=0, B=1, tension: L=0, M=1, H=2
    let data: Vec<(f64, f64, f64, f64)> = vec![
        // wool A, tension L
        (26.0, 0.0, 0.0, 0.0),
        (30.0, 0.0, 0.0, 0.0),
        (54.0, 0.0, 0.0, 0.0),
        (25.0, 0.0, 0.0, 0.0),
        (70.0, 0.0, 0.0, 0.0),
        (52.0, 0.0, 0.0, 0.0),
        (51.0, 0.0, 0.0, 0.0),
        (26.0, 0.0, 0.0, 0.0),
        (67.0, 0.0, 0.0, 0.0),
        // wool A, tension M
        (18.0, 0.0, 1.0, 0.0),
        (21.0, 0.0, 1.0, 0.0),
        (29.0, 0.0, 1.0, 0.0),
        (17.0, 0.0, 1.0, 0.0),
        (12.0, 0.0, 1.0, 0.0),
        (18.0, 0.0, 1.0, 0.0),
        (35.0, 0.0, 1.0, 0.0),
        (30.0, 0.0, 1.0, 0.0),
        (36.0, 0.0, 1.0, 0.0),
        // wool A, tension H
        (36.0, 0.0, 0.0, 1.0),
        (21.0, 0.0, 0.0, 1.0),
        (24.0, 0.0, 0.0, 1.0),
        (18.0, 0.0, 0.0, 1.0),
        (10.0, 0.0, 0.0, 1.0),
        (43.0, 0.0, 0.0, 1.0),
        (28.0, 0.0, 0.0, 1.0),
        (15.0, 0.0, 0.0, 1.0),
        (26.0, 0.0, 0.0, 1.0),
        // wool B, tension L
        (27.0, 1.0, 0.0, 0.0),
        (14.0, 1.0, 0.0, 0.0),
        (29.0, 1.0, 0.0, 0.0),
        (19.0, 1.0, 0.0, 0.0),
        (29.0, 1.0, 0.0, 0.0),
        (31.0, 1.0, 0.0, 0.0),
        (41.0, 1.0, 0.0, 0.0),
        (20.0, 1.0, 0.0, 0.0),
        (44.0, 1.0, 0.0, 0.0),
        // wool B, tension M
        (42.0, 1.0, 1.0, 0.0),
        (26.0, 1.0, 1.0, 0.0),
        (19.0, 1.0, 1.0, 0.0),
        (16.0, 1.0, 1.0, 0.0),
        (39.0, 1.0, 1.0, 0.0),
        (28.0, 1.0, 1.0, 0.0),
        (21.0, 1.0, 1.0, 0.0),
        (39.0, 1.0, 1.0, 0.0),
        (29.0, 1.0, 1.0, 0.0),
        // wool B, tension H
        (20.0, 1.0, 0.0, 1.0),
        (21.0, 1.0, 0.0, 1.0),
        (24.0, 1.0, 0.0, 1.0),
        (17.0, 1.0, 0.0, 1.0),
        (13.0, 1.0, 0.0, 1.0),
        (15.0, 1.0, 0.0, 1.0),
        (15.0, 1.0, 0.0, 1.0),
        (16.0, 1.0, 0.0, 1.0),
        (28.0, 1.0, 0.0, 1.0),
    ];

    // X: woolB, tensionM, tensionH (3 columns)
    let x = Mat::from_fn(54, 3, |i, j| match j {
        0 => data[i].1, // woolB
        1 => data[i].2, // tensionM
        2 => data[i].3, // tensionH
        _ => 0.0,
    });
    let y = Col::from_fn(54, |i| data[i].0);

    (x, y)
}

/// Test Poisson GLM coefficients against R's glm(breaks ~ wool + tension, family=poisson)
///
/// R output:
/// ```r
/// > summary(glm(breaks ~ wool + tension, data = warpbreaks, family = poisson))
/// Coefficients:
///             Estimate Std. Error z value Pr(>|z|)
/// (Intercept)  3.69196    0.04541  81.302  < 2e-16 ***
/// woolB       -0.20599    0.05157  -3.994 6.49e-05 ***
/// tensionM    -0.32132    0.06027  -5.332 9.73e-08 ***
/// tensionH    -0.51849    0.06396  -8.107 5.21e-16 ***
///
/// Null deviance: 297.37  on 53  degrees of freedom
/// Residual deviance: 210.39  on 50  degrees of freedom
/// AIC: 493.06
/// ```
#[test]
fn test_warpbreaks_poisson_coefficients() {
    let (x, y) = warpbreaks_data();

    let model = PoissonRegressor::builder().with_intercept(true).build();
    let fitted = model.fit(&x, &y).expect("fit should succeed");

    // R coefficients
    assert_relative_eq!(fitted.intercept().unwrap(), 3.69196, epsilon = 0.01);
    assert_relative_eq!(fitted.coefficients()[0], -0.20599, epsilon = 0.01); // woolB
    assert_relative_eq!(fitted.coefficients()[1], -0.32132, epsilon = 0.01); // tensionM
    assert_relative_eq!(fitted.coefficients()[2], -0.51849, epsilon = 0.01); // tensionH
}

/// Test Poisson GLM deviance against R
#[test]
fn test_warpbreaks_poisson_deviance() {
    let (x, y) = warpbreaks_data();

    let model = PoissonRegressor::builder().with_intercept(true).build();
    let fitted = model.fit(&x, &y).expect("fit should succeed");

    // R: Null deviance: 297.37 on 53 df
    assert_relative_eq!(fitted.null_deviance, 297.37, epsilon = 0.5);

    // R: Residual deviance: 210.39 on 50 df
    assert_relative_eq!(fitted.deviance, 210.39, epsilon = 0.5);

    // Degrees of freedom: 54 - 4 = 50 (3 predictors + intercept)
    assert_eq!(fitted.result().residual_df(), 50);
}

/// Test Poisson predictions match expected values
#[test]
fn test_warpbreaks_poisson_predictions() {
    let (x, y) = warpbreaks_data();

    let model = PoissonRegressor::builder().with_intercept(true).build();
    let fitted = model.fit(&x, &y).expect("fit should succeed");

    // Predict for each combination:
    // wool A, tension L: exp(3.69196) = 40.12
    // wool A, tension M: exp(3.69196 - 0.32132) = 29.09
    // wool A, tension H: exp(3.69196 - 0.51849) = 23.89
    // wool B, tension L: exp(3.69196 - 0.20599) = 32.63
    // wool B, tension M: exp(3.69196 - 0.20599 - 0.32132) = 23.67
    // wool B, tension H: exp(3.69196 - 0.20599 - 0.51849) = 19.44

    let x_new = Mat::from_fn(6, 3, |i, j| {
        match (i, j) {
            (0, _) => 0.0, // A, L: [0, 0, 0]
            (1, 1) => 1.0, // A, M: [0, 1, 0]
            (1, _) => 0.0,
            (2, 2) => 1.0, // A, H: [0, 0, 1]
            (2, _) => 0.0,
            (3, 0) => 1.0, // B, L: [1, 0, 0]
            (3, _) => 0.0,
            (4, 0) | (4, 1) => 1.0, // B, M: [1, 1, 0]
            (4, _) => 0.0,
            (5, 0) | (5, 2) => 1.0, // B, H: [1, 0, 1]
            (5, _) => 0.0,
            _ => 0.0,
        }
    });

    let predictions = fitted.predict(&x_new);

    // Expected values from R
    assert_relative_eq!(predictions[0], 40.12, epsilon = 0.5); // A, L
    assert_relative_eq!(predictions[1], 29.09, epsilon = 0.5); // A, M
    assert_relative_eq!(predictions[2], 23.89, epsilon = 0.5); // A, H
    assert_relative_eq!(predictions[3], 32.63, epsilon = 0.5); // B, L
    assert_relative_eq!(predictions[4], 23.67, epsilon = 0.5); // B, M
    assert_relative_eq!(predictions[5], 19.44, epsilon = 0.5); // B, H
}

// ============================================================================
// ADDITIONAL CARS DATASET TESTS
// ============================================================================

/// Test standard errors match R
///
/// R output:
/// ```r
/// > summary(lm(dist ~ speed, data = cars))$coefficients
///               Estimate Std. Error   t value     Pr(>|t|)
/// (Intercept) -17.579095  6.7584402 -2.601058 1.231882e-02
/// speed         3.932409  0.4155128  9.463990 1.489836e-12
/// ```
#[test]
fn test_cars_standard_errors() {
    let (speed, dist) = cars_data();
    let x = Mat::from_fn(50, 1, |i, _| speed[i]);

    let model = OlsRegressor::builder()
        .with_intercept(true)
        .compute_inference(true)
        .build();
    let fitted = model.fit(&x, &dist).expect("fit should succeed");

    // Get standard errors from result
    let se = fitted
        .result()
        .std_errors
        .as_ref()
        .expect("should have std errors");
    let intercept_se = fitted
        .result()
        .intercept_std_error
        .expect("should have intercept SE");

    // R: SE(intercept) = 6.7584402, SE(speed) = 0.4155128
    assert_relative_eq!(intercept_se, 6.7584, epsilon = 0.01);
    assert_relative_eq!(se[0], 0.4155, epsilon = 0.01);
}

/// Test t-values match R
#[test]
fn test_cars_t_values() {
    let (speed, dist) = cars_data();
    let x = Mat::from_fn(50, 1, |i, _| speed[i]);

    let model = OlsRegressor::builder()
        .with_intercept(true)
        .compute_inference(true)
        .build();
    let fitted = model.fit(&x, &dist).expect("fit should succeed");

    let t_stats = fitted
        .result()
        .t_statistics
        .as_ref()
        .expect("should have t stats");
    let intercept_t = fitted
        .result()
        .intercept_t_statistic
        .expect("should have intercept t");

    // R: t(intercept) = -2.601058, t(speed) = 9.463990
    assert_relative_eq!(intercept_t, -2.601, epsilon = 0.01);
    assert_relative_eq!(t_stats[0], 9.464, epsilon = 0.01);
}

/// Test residual standard error matches R
///
/// R output:
/// ```r
/// > summary(lm(dist ~ speed, data = cars))$sigma
/// [1] 15.37959
/// ```
#[test]
fn test_cars_residual_std_error() {
    let (speed, dist) = cars_data();
    let x = Mat::from_fn(50, 1, |i, _| speed[i]);

    let model = OlsRegressor::builder().with_intercept(true).build();
    let fitted = model.fit(&x, &dist).expect("fit should succeed");

    // RSE = sqrt(MSE) where MSE = SSE / df_residual
    let mse = fitted.result().mse;
    let rse = mse.sqrt();

    // R: sigma = 15.37959
    assert_relative_eq!(rse, 15.37959, epsilon = 0.01);
}

/// Test F-statistic matches R
///
/// R output:
/// ```r
/// > summary(lm(dist ~ speed, data = cars))
/// F-statistic: 89.57 on 1 and 48 DF,  p-value: 1.49e-12
/// ```
#[test]
fn test_cars_f_statistic() {
    let (speed, dist) = cars_data();
    let x = Mat::from_fn(50, 1, |i, _| speed[i]);

    let model = OlsRegressor::builder().with_intercept(true).build();
    let fitted = model.fit(&x, &dist).expect("fit should succeed");

    let f_stat = fitted.result().f_statistic;

    // R: F-statistic = 89.57
    assert_relative_eq!(f_stat, 89.57, epsilon = 0.1);
}
