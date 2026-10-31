//! #205: `NaAction::Exclude` pads residuals and fitted values back to the
//! input rows with NaN, as R's `na.exclude`; `Omit` records the dropped rows.
//!
//! Reference values: `tests/r_scripts/generate_na_exclude_validation.R`.

use anofox_regression::core::{
    NaAction, PoissonFamily, RegressionOptions, RegressionOptionsBuilder,
};
use anofox_regression::solvers::{FittedRegressor, OlsRegressor, PoissonRegressor, Regressor};
use faer::{Col, Mat};

const X1: [f64; N] = [
    4.9039999999999999,
    0.59799999999999998,
    2.601,
    f64::NAN,
    3.2810000000000001,
    1.9299999999999999,
    0.216,
    4.3330000000000002,
    4.3529999999999998,
    4.8860000000000001,
    0.84399999999999997,
    0.97799999999999998,
    1.8080000000000001,
    0.871,
    2.5049999999999999,
];
const X2: [f64; N] = [
    -0.70899999999999996,
    -0.97699999999999998,
    -0.76900000000000002,
    1.905,
    0.14000000000000001,
    -1.4550000000000001,
    -0.59499999999999997,
    -0.97499999999999998,
    f64::NAN,
    0.53700000000000003,
    -0.42099999999999999,
    2.0579999999999998,
    0.36799999999999999,
    -1.653,
    0.78100000000000003,
];
const Y: [f64; N] = [
    5.2480000000000002,
    1.512,
    3.4209999999999998,
    2.5720000000000001,
    3.4470000000000001,
    2.7189999999999999,
    1.0960000000000001,
    4.2279999999999998,
    3.9489999999999998,
    4.3899999999999997,
    1.258,
    f64::NAN,
    1.49,
    2.4729999999999999,
    1.877,
];
const YC: [f64; N] = [
    3.0,
    1.0,
    0.0,
    5.0,
    4.0,
    1.0,
    2.0,
    3.0,
    6.0,
    4.0,
    1.0,
    f64::NAN,
    1.0,
    1.0,
    3.0,
];
const OLS_RESID: [f64; N] = [
    0.18848917281914435,
    -0.027948866482144563,
    0.29719538630446685,
    f64::NAN,
    0.32191686218187798,
    -0.26855828600790083,
    0.12791134967866863,
    -0.51225365257972644,
    f64::NAN,
    0.14314877013865326,
    -0.13706986130386845,
    f64::NAN,
    -0.22656861116223548,
    0.26651352358825681,
    -0.17277578717519221,
];
const OLS_FITTED: [f64; N] = [
    5.0595108271808558,
    1.5399488664821446,
    3.1238046136955329,
    f64::NAN,
    3.1250831378181223,
    2.9875582860079009,
    0.96808865032133151,
    4.7402536525797263,
    f64::NAN,
    4.2468512298613463,
    1.3950698613038686,
    f64::NAN,
    1.7165686111622356,
    2.2064864764117429,
    2.049775787175192,
];
const POIS_RESID: [f64; N] = [
    0.050722893952042991,
    0.011244787672787138,
    -1.6880169781694576,
    f64::NAN,
    1.3575070995921301,
    -0.15997804681373973,
    0.97901637989309021,
    0.62889033633223113,
    f64::NAN,
    -0.3646891977218738,
    -0.24988717066696009,
    f64::NAN,
    -1.0127932875392425,
    0.14987836117073705,
    0.2981048222982654,
];
const POIS_FITTED: [f64; N] = [
    2.949277106047957,
    0.98875521232721286,
    1.6880169781694576,
    f64::NAN,
    2.6424929004078699,
    1.1599780468137397,
    1.0209836201069098,
    2.3711096636677689,
    f64::NAN,
    4.3646891977218738,
    1.2498871706669601,
    f64::NAN,
    2.0127932875392425,
    0.85012163882926295,
    2.7018951777017346,
];

const N: usize = 15;

fn options(action: NaAction) -> RegressionOptions {
    RegressionOptionsBuilder::default()
        .na_action(action)
        .tolerance(1e-12)
        .build()
        .unwrap()
}

fn x() -> Mat<f64> {
    Mat::from_fn(N, 2, |i, j| if j == 0 { X1[i] } else { X2[i] })
}

fn check(got: &Col<f64>, want: &[f64; N], tol: f64, what: &str) {
    assert_eq!(got.nrows(), N, "{what}: padded length");
    for i in 0..N {
        if want[i].is_nan() {
            assert!(got[i].is_nan(), "{what}[{i}] should be NA");
        } else {
            assert!(
                (got[i] - want[i]).abs() < tol,
                "{what}[{i}]: {} vs R {}",
                got[i],
                want[i]
            );
        }
    }
}

#[test]
fn exclude_pads_ols_residuals_and_fitted_values() {
    let fit = OlsRegressor::new(options(NaAction::Exclude))
        .fit(&x(), &Col::from_fn(N, |i| Y[i]))
        .unwrap();
    let r = fit.result();
    assert_eq!(r.residuals.nrows(), N - 3);
    assert_eq!(r.n_na_removed(), 3);
    check(&r.residuals_expanded(), &OLS_RESID, 1e-10, "residuals");
    check(&r.fitted_expanded(), &OLS_FITTED, 1e-10, "fitted");
}

#[test]
fn exclude_pads_glm_residuals_and_fitted_values() {
    let fit = PoissonRegressor::new(options(NaAction::Exclude), PoissonFamily::log())
        .fit(&x(), &Col::from_fn(N, |i| YC[i]))
        .unwrap();
    let r = fit.result();
    check(&r.residuals_expanded(), &POIS_RESID, 1e-8, "residuals");
    check(&r.fitted_expanded(), &POIS_FITTED, 1e-8, "fitted");
}

#[test]
fn omit_records_but_does_not_pad() {
    let fit = OlsRegressor::new(options(NaAction::Omit))
        .fit(&x(), &Col::from_fn(N, |i| Y[i]))
        .unwrap();
    let r = fit.result();
    assert!(r.had_na_removed());
    assert_eq!(r.n_na_removed(), 3);
    assert_eq!(r.residuals_expanded().nrows(), N - 3);
}
