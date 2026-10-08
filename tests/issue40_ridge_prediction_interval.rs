//! Issue #40 (R reference values from R 4.6.1 / MASS / lme4): FittedRidge::predict_with_interval uses (X'X + lambda P)^-1 as the variance
//! factor; the variance of x0' beta_ridge is s^2 x0' A X'X A x0.
use anofox_regression::core::IntervalType;
use anofox_regression::solvers::{FittedRegressor, Regressor, RidgeRegressor};
use faer::{Col, Mat};

#[test]
fn ridge_confidence_interval_se_matches_sandwich() {
    let n = 15;
    let x = Mat::from_fn(n, 2, |k, j| {
        let i = k + 1;
        if j == 0 {
            i as f64 / 3.0
        } else {
            ((i * 5) % 7) as f64 / 2.0
        }
    });
    let y = Col::from_fn(n, |k| {
        let i = k + 1;
        2.0 + 1.5 * x[(k, 0)] - 0.8 * x[(k, 1)] + (((i * 3) % 5) as f64 - 2.0) * 0.5
    });
    let f = RidgeRegressor::builder()
        .lambda(1.0)
        .with_intercept(true)
        .build()
        .fit(&x, &y)
        .unwrap();
    let x0 = Mat::from_fn(2, 2, |i, j| match (i, j) {
        (0, _) => 0.0,
        (1, 0) => 10.0,
        _ => 1.0,
    });
    let c = f.predict_with_interval(&x0, Some(IntervalType::Confidence), 0.95);
    let p = f.predict_with_interval(&x0, Some(IntervalType::Prediction), 0.95);
    println!(
        "fit {:?} conf se {:?} pred se {:?}",
        c.fit.iter().collect::<Vec<_>>(),
        c.se.iter().collect::<Vec<_>>(),
        p.se.iter().collect::<Vec<_>>()
    );
    println!("R: fit 2.2194604600796008 15.2843413427768624; conf se 0.5341843123743818 1.0193578906380427; pred se 0.94890503469367726 1.28614089216001437");
    assert!((c.fit[1] - 15.2843413427768624).abs() < 1e-9);
    assert!(
        (c.se[0] - 0.5341843123743818).abs() < 1e-6,
        "conf se[0] {}",
        c.se[0]
    );
    assert!(
        (c.se[1] - 1.0193578906380427).abs() < 1e-6,
        "conf se[1] {}",
        c.se[1]
    );
    assert!(
        (p.se[1] - 1.28614089216001437).abs() < 1e-6,
        "pred se[1] {}",
        p.se[1]
    );
}
