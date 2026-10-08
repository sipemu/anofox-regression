//! Issue #39 (R reference values from R 4.6.1 / MASS / lme4): constant columns are dropped even when with_intercept(false) (then the
//! constant column *is* the intercept); and the absolute 1e-10 tolerance drops
//! non-constant tiny-unit columns.
use anofox_regression::solvers::{FittedRegressor, OlsRegressor, Regressor, WlsRegressor};
use faer::{Col, Mat};

const Y: [f64; 8] = [3.2, 4.8, 7.1, 9.0, 10.9, 13.2, 14.8, 17.1];

#[test]
fn ols_no_intercept_keeps_constant_column() {
    let x = Mat::from_fn(8, 2, |i, j| if j == 0 { 2.0 } else { (i + 1) as f64 });
    let y = Col::from_fn(8, |i| Y[i]);
    let r = OlsRegressor::builder()
        .with_intercept(false)
        .build()
        .fit(&x, &y)
        .map(|f| f.result().clone());
    match &r {
        Ok(r) => println!(
            "OLS y~0+c+x1: coef {:?} aliased {:?}  (R: 0.51964285714285685 1.99404761904761929)",
            r.coefficients.iter().collect::<Vec<_>>(),
            r.aliased
        ),
        Err(e) => println!("error {e:?}"),
    }
    let r = r.unwrap();
    println!("  fitted[0] {} rmse {} rank {}  (R: fitted[0] 3.03333333333333144, sigma 0.17666442047089623)", r.fitted_values[0], r.rmse, r.rank);
    assert!(
        (r.coefficients[0] - 0.51964285714285685).abs() < 1e-9,
        "constant column coef {}",
        r.coefficients[0]
    );
    assert!((r.coefficients[1] - 1.99404761904761929).abs() < 1e-9);
    assert!(
        (r.fitted_values[0] - 3.03333333333333144).abs() < 1e-9,
        "fitted[0] {}",
        r.fitted_values[0]
    );
    assert!(
        (r.rmse - 0.17666442047089623).abs() < 1e-9,
        "rmse {}",
        r.rmse
    );
}

#[test]
fn wls_no_intercept_keeps_constant_column() {
    let x = Mat::from_fn(8, 2, |i, j| if j == 0 { 2.0 } else { (i + 1) as f64 });
    let y = Col::from_fn(8, |i| Y[i]);
    let r = WlsRegressor::builder()
        .with_intercept(false)
        .weights(Col::from_fn(8, |_| 1.0))
        .build()
        .fit(&x, &y)
        .map(|f| f.result().clone());
    match &r {
        Ok(r) => println!(
            "WLS(w=1) y~0+c+x1: coef {:?} aliased {:?}",
            r.coefficients.iter().collect::<Vec<_>>(),
            r.aliased
        ),
        Err(e) => println!("error {e:?}"),
    }
    let r = r.unwrap();
    assert!(
        (r.coefficients[0] - 0.51964285714285685).abs() < 1e-9,
        "constant column coef {}",
        r.coefficients[0]
    );
}

#[test]
fn ols_no_intercept_single_constant_column() {
    let x = Mat::from_fn(8, 1, |_, _| 2.0);
    let y = Col::from_fn(8, |i| Y[i]);
    let r = OlsRegressor::builder()
        .with_intercept(false)
        .build()
        .fit(&x, &y);
    match &r {
        Ok(f) => println!(
            "OLS y~0+c: coef {:?}  (R: 5.0062499999999996)",
            f.result().coefficients[0]
        ),
        Err(e) => println!("OLS y~0+c: error {e:?}  (R: 5.0062499999999996)"),
    }
    assert!((r.unwrap().result().coefficients[0] - 5.0062499999999996).abs() < 1e-9);
}

#[test]
fn ols_keeps_tiny_unit_column() {
    let x = Mat::from_fn(8, 1, |i, _| (i + 1) as f64 * 1e-12);
    let y = Col::from_fn(8, |i| Y[i]);
    let f = OlsRegressor::builder()
        .with_intercept(true)
        .build()
        .fit(&x, &y)
        .unwrap();
    let r = f.result();
    println!("OLS y~x*1e-12: intercept {:?} coef {} aliased {:?}  (R: 1.0392857142857126 1.9940476190476194e12)", r.intercept, r.coefficients[0], r.aliased);
    assert!(!r.aliased[0]);
    assert!((r.coefficients[0] / 1.9940476190476194e12 - 1.0).abs() < 1e-9);
}
