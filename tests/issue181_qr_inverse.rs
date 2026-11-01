//! #181: `(X'X)⁻¹` for standard errors and intervals comes from a QR of the
//! design, not from inverting `X'X`, which squares the condition number.
//! Reference: R 4.6, `predict(lm(y ~ x + … + I(x^6)), se.fit = TRUE)` for
//! `x = 1:30` (κ(X) = 3.2e9), `set.seed(3); y <- 1 + 0.5x − 0.01x² + rnorm(30)`.

use anofox_regression::core::IntervalType;
use anofox_regression::solvers::{FittedRegressor, OlsRegressor, Regressor};
use faer::{Col, Mat};

#[rustfmt::skip]
const Y: [f64; 30] = [0.52806658408011731, 1.6674742771215334, 2.6687882162412517, 1.6878681140848673, 3.4457828262863752, 3.6701239446016318, 4.0954177316122715, 5.4766102127152658, 3.4711425844220116, 6.2673687220898877, 4.5452184038648591, 4.428781429164264, 5.0936415099670231, 6.2926523696462144, 6.4020457066555956, 6.1323435703215798, 5.6569826690918772, 6.1117571885515298, 8.1143136242805856, 7.1998116079829693, 6.5115162781400278, 6.2176992665224606, 7.0062718203380054, 5.573525159969841, 6.765544890849009, 6.4989273392784046, 8.3706157792413354, 8.1720671249342338, 7.0179215259134402, 5.8632177019006777];
const SE_FIT: [f64; 3] = [
    0.44819891362819803,
    0.34196688009756804,
    0.54498959831181082,
];

fn poly(x: &[f64]) -> Mat<f64> {
    Mat::from_fn(x.len(), 6, |i, j| x[i].powi(j as i32 + 1))
}

#[test]
fn prediction_se_of_an_ill_conditioned_polynomial_matches_r() {
    let x: Vec<f64> = (1..=30).map(f64::from).collect();
    let f = OlsRegressor::builder()
        .compute_inference(true)
        .build()
        .fit(&poly(&x), &Col::from_fn(30, |i| Y[i]))
        .unwrap();
    let p = f.predict_with_interval(
        &poly(&[2.5, 15.5, 29.5]),
        Some(IntervalType::Confidence),
        0.95,
    );
    for (k, want) in SE_FIT.iter().enumerate() {
        let got = p.se[k];
        // QR of the design: ≤ 7e-10; inverting X'X gave up to 2e-9
        assert!(
            (got - want).abs() < 1e-9 * want,
            "se[{k}]: {got} vs R {want}"
        );
    }
}
