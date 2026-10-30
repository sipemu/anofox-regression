//! #176: `NaAction` (default `Omit`) is applied before every fit of the
//! estimators configured by `RegressionOptions`.

use anofox_regression::solvers::{
    BinomialRegressor, FittedRegressor, OlsRegressor, PoissonRegressor, Regressor, RidgeRegressor,
};
use faer::{Col, Mat};

fn data() -> (Mat<f64>, Col<f64>, Mat<f64>, Col<f64>) {
    let n = 40;
    let x = Mat::from_fn(n, 2, |i, j| ((i * (j + 2) * 7) % 13) as f64 / 4.0);
    let y = Col::from_fn(n, |i| ((i * 5) % 4) as f64);
    let (mut xn, mut yn) = (x.clone(), y.clone());
    xn[(3, 1)] = f64::NAN;
    yn[10] = f64::NAN;
    let keep: Vec<usize> = (0..n).filter(|&i| i != 3 && i != 10).collect();
    let xc = Mat::from_fn(keep.len(), 2, |i, j| x[(keep[i], j)]);
    let yc = Col::from_fn(keep.len(), |i| y[keep[i]]);
    (xn, yn, xc, yc)
}

fn same(a: &Col<f64>, b: &Col<f64>) {
    assert_eq!(a.nrows(), b.nrows());
    for i in 0..a.nrows() {
        assert!((a[i] - b[i]).abs() < 1e-12, "{} vs {}", a[i], b[i]);
    }
}

#[test]
fn omit_equals_fitting_the_complete_cases() {
    let (xn, yn, xc, yc) = data();
    let ols = OlsRegressor::builder().build();
    same(
        &ols.fit(&xn, &yn).unwrap().result().coefficients,
        &ols.fit(&xc, &yc).unwrap().result().coefficients,
    );
    let ridge = RidgeRegressor::builder().lambda(0.5).build();
    same(
        &ridge.fit(&xn, &yn).unwrap().result().coefficients,
        &ridge.fit(&xc, &yc).unwrap().result().coefficients,
    );
    let pois = PoissonRegressor::log().build();
    same(
        &pois.fit(&xn, &yn).unwrap().result().coefficients,
        &pois.fit(&xc, &yc).unwrap().result().coefficients,
    );
    let yb = Col::from_fn(yn.nrows(), |i| {
        if yn[i].is_nan() {
            f64::NAN
        } else {
            f64::from(u8::from(yn[i] > 1.0))
        }
    });
    let ybc = Col::from_fn(yc.nrows(), |i| f64::from(u8::from(yc[i] > 1.0)));
    let logit = BinomialRegressor::logistic().build();
    same(
        &logit.fit(&xn, &yb).unwrap().result().coefficients,
        &logit.fit(&xc, &ybc).unwrap().result().coefficients,
    );
}

#[test]
fn row_aligned_offsets_cannot_be_omitted() {
    let (xn, yn, _, _) = data();
    let pois = PoissonRegressor::log()
        .offset(Col::zeros(yn.nrows()))
        .build();
    assert!(pois.fit(&xn, &yn).is_err());
}
