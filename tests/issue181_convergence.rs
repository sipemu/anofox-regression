//! #181: Huber and quantile regression report whether they converged.

use anofox_regression::solvers::{HuberRegressor, QuantileRegressor, Regressor};
use faer::{Col, Mat};

fn data() -> (Mat<f64>, Col<f64>) {
    let n = 60;
    let x = Mat::from_fn(n, 2, |i, j| ((i * (j + 3)) as f64 * 0.41).sin() * 2.0);
    let y = Col::from_fn(n, |i| {
        let outlier = if i % 17 == 0 { 15.0 } else { 0.0 };
        1.0 + x[(i, 0)] - 0.5 * x[(i, 1)] + ((i * 7919) % 11) as f64 / 10.0 + outlier
    });
    (x, y)
}

#[test]
fn huber_reports_convergence() {
    let (x, y) = data();
    let f = HuberRegressor::builder().build().fit(&x, &y).unwrap();
    assert!(f.converged());
    assert!(f.iterations() >= 1 && f.iterations() < 100);
    // one IRLS step is not enough with outliers
    let f = HuberRegressor::builder()
        .max_iterations(1)
        .build()
        .fit(&x, &y)
        .unwrap();
    assert!(!f.converged());
    assert_eq!(f.iterations(), 1);
}

#[test]
fn quantile_reports_the_lp_optimum() {
    let (x, y) = data();
    for tau in [0.1, 0.5, 0.9] {
        let f = QuantileRegressor::builder()
            .tau(tau)
            .build()
            .fit(&x, &y)
            .unwrap();
        assert!(f.converged(), "tau {tau}");
        assert!(f.iterations() >= 1);
    }
    // a duplicated column: the pivots cannot certify a vertex
    let xd = Mat::from_fn(x.nrows(), 2, |i, _| x[(i, 0)]);
    let f = QuantileRegressor::builder().build().fit(&xd, &y).unwrap();
    assert!(!f.converged());
}
