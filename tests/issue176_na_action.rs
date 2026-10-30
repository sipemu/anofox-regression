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

/// Every estimator configured by `RegressionOptions` fits the complete
/// cases under the default `Omit` (same fit as the hand-cleaned data).
#[test]
fn every_options_estimator_omits_incomplete_rows() {
    use anofox_regression::solvers::lm_dynamic::LmDynamicRegressor;
    use anofox_regression::solvers::{
        AlmRegressor, BlsRegressor, ElasticNetRegressor, NegativeBinomialRegressor, RlsRegressor,
        TweedieRegressor, WlsRegressor,
    };
    let (xn, yn, xc, yc) = data();
    // counts shifted to be positive for Tweedie / Gamma-type fits
    let shift = |y: &Col<f64>| Col::from_fn(y.nrows(), |i| y[i] + 1.0);
    fn check<R: Regressor>(r: R, xn: &Mat<f64>, yn: &Col<f64>, xc: &Mat<f64>, yc: &Col<f64>) {
        let a = r.fit(xn, yn).unwrap();
        let b = r.fit(xc, yc).unwrap();
        assert_eq!(a.result().n_observations, b.result().n_observations);
        same(&a.result().coefficients, &b.result().coefficients);
    }
    check(WlsRegressor::builder().build(), &xn, &yn, &xc, &yc);
    check(BlsRegressor::builder().build(), &xn, &yn, &xc, &yc);
    check(
        ElasticNetRegressor::builder().lambda(0.1).build(),
        &xn,
        &yn,
        &xc,
        &yc,
    );
    check(RlsRegressor::builder().build(), &xn, &yn, &xc, &yc);
    check(
        NegativeBinomialRegressor::builder().build(),
        &xn,
        &yn,
        &xc,
        &yc,
    );
    check(
        TweedieRegressor::gamma().build(),
        &xn,
        &shift(&yn),
        &xc,
        &shift(&yc),
    );
    check(AlmRegressor::builder().build(), &xn, &yn, &xc, &yc);
    check(LmDynamicRegressor::builder().build(), &xn, &yn, &xc, &yc);
}

#[test]
fn fail_errors_and_weights_cannot_be_omitted() {
    use anofox_regression::core::{NaAction, RegressionOptionsBuilder};
    use anofox_regression::solvers::WlsRegressor;
    let (xn, yn, _, _) = data();
    let fail = RegressionOptionsBuilder::default()
        .na_action(NaAction::Fail)
        .build()
        .unwrap();
    assert!(OlsRegressor::new(fail).fit(&xn, &yn).is_err());
    let w = WlsRegressor::builder()
        .weights(Col::from_fn(yn.nrows(), |_| 1.0))
        .build();
    assert!(w.fit(&xn, &yn).is_err());
}
