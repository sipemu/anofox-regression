//! Issue #50: Theil–Sen on a rank-deficient design must alias the redundant
//! column (NaN coefficient, like OLS / R `lm`) instead of returning ~1e14
//! coefficients from singular subsample systems.
//!
//! Data (the DuckDB extension's `collin` table): i = 1..40,
//! x1 = (i %% 7) + 0.5, x2 = 2·x1,
//! y = 3 + 0.8 x1 − 0.3 ((3i %% 11)/2) + ((37i %% 13)/13) − 0.5.
//! R: `coef(lm(y ~ x1 + I(2*x1)))` = 2.2671090 0.7815128 NA (aliased).

use anofox_regression::solvers::{FittedRegressor, Regressor, TheilSenRegressor};
use faer::{Col, Mat};

fn data() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let x1: Vec<f64> = (1..=40).map(|i| (i % 7) as f64 + 0.5).collect();
    let x2b: Vec<f64> = (1..=40).map(|i| ((i * 3) % 11) as f64 / 2.0).collect();
    let y: Vec<f64> = (1..=40)
        .map(|i| 3.0 + 0.8 * x1[i - 1] - 0.3 * x2b[i - 1] + ((i * 37) % 13) as f64 / 13.0 - 0.5)
        .collect();
    (x1, x2b, y)
}

#[test]
fn collinear_columns_are_aliased() {
    let (x1, _, yv) = data();
    let n = x1.len();
    let x = Mat::from_fn(n, 2, |i, j| if j == 0 { x1[i] } else { 2.0 * x1[i] });
    let y = Col::from_fn(n, |i| yv[i]);
    let fit = TheilSenRegressor::builder().build().fit(&x, &y).unwrap();

    let reduced = TheilSenRegressor::builder()
        .build()
        .fit(&Mat::from_fn(n, 1, |i, _| x1[i]), &y)
        .unwrap();

    let r = fit.result();
    assert!(r.coefficients[1].is_nan(), "x2 must be aliased");
    assert_eq!(r.aliased, vec![false, true]);
    // Identical to the Theil–Sen fit on x1 alone (pairwise median slope).
    assert!((r.coefficients[0] - reduced.result().coefficients[0]).abs() < 1e-12);
    assert!((r.intercept.unwrap() - reduced.result().intercept.unwrap()).abs() < 1e-12);
    assert!((r.coefficients[0] - 0.769_230_769_230_769_2).abs() < 1e-12);
    assert!((r.intercept.unwrap() - 2.265_384_615_384_616).abs() < 1e-12);

    let pred = fit.predict(&x);
    let pred_red = reduced.predict(&Mat::from_fn(n, 1, |i, _| x1[i]));
    for i in 0..n {
        assert!((pred[i] - pred_red[i]).abs() < 1e-12);
    }
}

#[test]
fn multivariate_with_a_linear_combination_is_aliased() {
    // x3 = x1 + x2: the spatial-median path on (x1, x2), x3 aliased.
    let (x1, x2, yv) = data();
    let n = x1.len();
    let x = Mat::from_fn(n, 3, |i, j| match j {
        0 => x1[i],
        1 => x2[i],
        _ => x1[i] + x2[i],
    });
    let y = Col::from_fn(n, |i| yv[i]);
    let fit = TheilSenRegressor::builder().build().fit(&x, &y).unwrap();
    let reduced = TheilSenRegressor::builder()
        .build()
        .fit(
            &Mat::from_fn(n, 2, |i, j| if j == 0 { x1[i] } else { x2[i] }),
            &y,
        )
        .unwrap();
    let r = fit.result();
    assert_eq!(r.aliased, vec![false, false, true]);
    assert!(r.coefficients[2].is_nan());
    for j in 0..2 {
        assert!((r.coefficients[j] - reduced.result().coefficients[j]).abs() < 1e-12);
        assert!(r.coefficients[j].abs() < 10.0);
    }
    assert!(fit.predict(&x).iter().all(|v| v.is_finite()));
}

#[test]
fn constant_column_with_intercept_is_aliased() {
    let (x1, x2, yv) = data();
    let n = x1.len();
    let x = Mat::from_fn(n, 3, |i, j| match j {
        0 => x1[i],
        1 => 3.0,
        _ => x2[i],
    });
    let y = Col::from_fn(n, |i| yv[i]);
    let fit = TheilSenRegressor::builder().build().fit(&x, &y).unwrap();
    let r = fit.result();
    assert_eq!(r.aliased, vec![false, true, false]);
    assert!(r.coefficients[1].is_nan());
    assert!(r.coefficients[0].is_finite() && r.coefficients[2].is_finite());
    assert!((r.coefficients[0] - 0.8).abs() < 0.2);
    assert!((r.coefficients[2] + 0.3).abs() < 0.2);
}

#[test]
fn full_rank_design_is_unchanged() {
    let (x1, x2, yv) = data();
    let n = x1.len();
    let x = Mat::from_fn(n, 2, |i, j| if j == 0 { x1[i] } else { x2[i] });
    let y = Col::from_fn(n, |i| yv[i]);
    let r = TheilSenRegressor::builder().build().fit(&x, &y).unwrap();
    assert!(r.result().aliased.iter().all(|a| !a));
    assert!(r.result().coefficients.iter().all(|c| c.is_finite()));
}
