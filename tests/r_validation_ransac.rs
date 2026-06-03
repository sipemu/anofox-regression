//! RANSAC validation against scikit-learn 1.5.2.
//!
//! Reference values come from
//! `validation/python/generate_ransac_validation.py`. Fixtures are
//! constructed with a clean inlier/outlier gap so the consensus inlier set
//! is unique regardless of subsample-ordering — both implementations should
//! converge to identical coefficients after the final OLS refit.

#![allow(dead_code)]

use anofox_regression::solvers::{FittedRegressor, RansacRegressor, Regressor};
use faer::{Col, Mat};

include!("fixtures/ransac_validation.rs");

#[test]
fn ransac_univariate_matches_sklearn() {
    let n = N_RANSAC_UNI;
    let x = Mat::from_fn(n, 1, |i, _| X_RANSAC_UNI[i]);
    let y = Col::from_fn(n, |i| Y_RANSAC_UNI[i]);

    let fitted = RansacRegressor::builder()
        .min_samples(2)
        .residual_threshold(0.5)
        .max_trials(500)
        .random_state(42)
        .build()
        .fit(&x, &y)
        .expect("RANSAC fit failed");

    let intercept = fitted.result().intercept.unwrap();
    let coef = fitted.result().coefficients[0];

    // Both implementations refit OLS on the same inlier set, so the
    // coefficients must match to numerical precision.
    assert!(
        (intercept - EXPECTED_INTERCEPT_RANSAC_UNI).abs() < 1e-9,
        "intercept {} vs sklearn {}",
        intercept,
        EXPECTED_INTERCEPT_RANSAC_UNI
    );
    assert!(
        (coef - EXPECTED_COEF_RANSAC_UNI).abs() < 1e-9,
        "coef {} vs sklearn {}",
        coef,
        EXPECTED_COEF_RANSAC_UNI
    );

    let mask = fitted.inlier_mask();
    assert_eq!(mask.len(), EXPECTED_INLIER_MASK_RANSAC_UNI.len());
    for i in 0..mask.len() {
        assert_eq!(
            mask[i], EXPECTED_INLIER_MASK_RANSAC_UNI[i],
            "inlier-mask mismatch at index {}",
            i
        );
    }
    assert_eq!(fitted.n_inliers(), EXPECTED_N_INLIERS_RANSAC_UNI);
}

#[test]
fn ransac_multivariate_matches_sklearn() {
    let n = N_RANSAC_MULTI;
    let p = P_RANSAC_MULTI;
    let x = Mat::from_fn(n, p, |i, j| X_RANSAC_MULTI_FLAT[i * p + j]);
    let y = Col::from_fn(n, |i| Y_RANSAC_MULTI[i]);

    let fitted = RansacRegressor::builder()
        .min_samples(3)
        .residual_threshold(0.3)
        .max_trials(2000)
        .random_state(42)
        .build()
        .fit(&x, &y)
        .expect("RANSAC fit failed");

    let intercept = fitted.result().intercept.unwrap();
    let coefs = &fitted.result().coefficients;

    assert!(
        (intercept - EXPECTED_INTERCEPT_RANSAC_MULTI).abs() < 1e-9,
        "intercept {} vs sklearn {}",
        intercept,
        EXPECTED_INTERCEPT_RANSAC_MULTI
    );
    for j in 0..p {
        assert!(
            (coefs[j] - EXPECTED_COEFS_RANSAC_MULTI[j]).abs() < 1e-9,
            "coef[{}] {} vs sklearn {}",
            j,
            coefs[j],
            EXPECTED_COEFS_RANSAC_MULTI[j]
        );
    }
    assert_eq!(fitted.n_inliers(), EXPECTED_N_INLIERS_RANSAC_MULTI);
}
