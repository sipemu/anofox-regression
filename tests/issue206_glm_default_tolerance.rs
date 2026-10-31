//! #206: the default IRLS tolerance is R's `glm.control(epsilon = 1e-8)` on
//! the relative deviance change (`glm.fit`'s criterion). Fits at the default
//! agree with fully converged R `glm` (`epsilon = 1e-15`) to < 1e-5 relative
//! (non-canonical links converge linearly; with the old default 1e-6 the
//! error reached 5e-5 here), and with `tolerance(1e-15)` to < 1e-7.
//!
//! Reference values: `tests/r_scripts/generate_glm_default_tolerance_validation.R`.

use anofox_regression::solvers::FittedRegressor;
use anofox_regression::solvers::{
    BinomialRegressor, PoissonRegressor, Regressor, TweedieRegressor,
};
use faer::{Col, Mat};

const X: [f64; N] = [
    0.71999999999999997,
    -0.058999999999999997,
    0.69199999999999995,
    -0.83299999999999996,
    -0.93200000000000005,
    -0.50600000000000001,
    -0.13,
    0.38800000000000001,
    -0.22600000000000001,
    0.88700000000000001,
    -0.85599999999999998,
    0.374,
    0.074999999999999997,
    0.502,
    0.24399999999999999,
    -0.46300000000000002,
    -0.55800000000000005,
    0.78300000000000003,
    0.10199999999999999,
    0.017999999999999999,
    0.192,
    0.58399999999999996,
    0.33300000000000002,
    -0.44700000000000001,
    0.83199999999999996,
    0.13300000000000001,
    0.217,
    -0.65500000000000003,
    0.63600000000000001,
    0.88,
    0.189,
    -0.52100000000000002,
    -0.076999999999999999,
    0.72399999999999998,
    -0.59999999999999998,
    0.67600000000000005,
    -0.68500000000000005,
    -0.098000000000000004,
    0.096000000000000002,
    0.70299999999999996,
    -0.023,
    0.021000000000000001,
    0.61599999999999999,
    -0.39500000000000002,
    0.096000000000000002,
    -0.77400000000000002,
    0.81299999999999994,
    -0.71999999999999997,
    0.0080000000000000002,
    0.26800000000000002,
];
const YG: [f64; N] = [
    2.3195999999999999,
    0.36209999999999998,
    6.2659000000000002,
    0.56740000000000002,
    2.4590999999999998,
    2.9249999999999998,
    0.1067,
    0.92210000000000003,
    1.9173,
    1.6175999999999999,
    1.4269000000000001,
    0.50390000000000001,
    4.4950999999999999,
    3.1817000000000002,
    2.8664000000000001,
    2.1031,
    1.228,
    2.2252999999999998,
    2.8631000000000002,
    1.2833000000000001,
    3.2629000000000001,
    7.6275000000000004,
    6.4188000000000001,
    1.7310000000000001,
    4.1535000000000002,
    1.9923,
    1.6549,
    1.9214,
    2.3437000000000001,
    6.7526999999999999,
    0.67369999999999997,
    0.64229999999999998,
    1.2806999999999999,
    8.1403999999999996,
    0.51390000000000002,
    1.1107,
    0.1444,
    1.6686000000000001,
    0.65400000000000003,
    5.8346999999999998,
    0.76959999999999995,
    0.85389999999999999,
    0.81559999999999999,
    0.254,
    3.5421,
    1.8749,
    2.1049000000000002,
    0.89449999999999996,
    0.37009999999999998,
    0.60250000000000004,
];
const YB: [f64; N] = [
    1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0,
    0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0,
    1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0,
];
const YP: [f64; N] = [
    4.0, 2.0, 6.0, 1.0, 2.0, 3.0, 0.0, 5.0, 2.0, 2.0, 1.0, 5.0, 1.0, 6.0, 2.0, 1.0, 2.0, 5.0, 6.0,
    3.0, 4.0, 3.0, 4.0, 4.0, 1.0, 4.0, 6.0, 2.0, 3.0, 7.0, 4.0, 2.0, 6.0, 6.0, 2.0, 6.0, 2.0, 1.0,
    5.0, 6.0, 1.0, 3.0, 7.0, 2.0, 7.0, 5.0, 9.0, 2.0, 3.0, 7.0,
];
const GAMMA_LOG: [f64; 2] = [0.67417110917062228, 0.74658662516868268];
const BIN_CLOGLOG: [f64; 2] = [-0.53094999099899964, 1.356063786180346];
const BIN_PROBIT: [f64; 2] = [-0.085395808612797774, 1.2039469043161577];
const POIS_IDENTITY: [f64; 2] = [3.5211434776494084, 2.1402053383260169];
const POIS_LOG: [f64; 2] = [1.201484172145427, 0.63449061110683413];

const N: usize = 50;

fn x() -> Mat<f64> {
    Mat::from_fn(N, 1, |i, _| X[i])
}

fn y(v: &[f64; N]) -> Col<f64> {
    Col::from_fn(N, |i| v[i])
}

fn check(fit: &impl FittedRegressor, want: &[f64; 2], tol: f64, what: &str) {
    let got = [
        fit.result().intercept.unwrap(),
        fit.result().coefficients[0],
    ];
    for k in 0..2 {
        let rel = (got[k] - want[k]).abs() / want[k].abs();
        assert!(
            rel < tol,
            "{what} coef {k}: {} vs converged R {} (rel {rel:.1e})",
            got[k],
            want[k]
        );
    }
}

#[test]
fn default_tolerance_fits_match_converged_r() {
    let tol = 1e-5;
    let f = TweedieRegressor::gamma()
        .build()
        .fit(&x(), &y(&YG))
        .unwrap();
    check(&f, &GAMMA_LOG, tol, "Gamma(log)");
    let f = BinomialRegressor::cloglog()
        .build()
        .fit(&x(), &y(&YB))
        .unwrap();
    check(&f, &BIN_CLOGLOG, tol, "binomial(cloglog)");
    let f = BinomialRegressor::probit()
        .build()
        .fit(&x(), &y(&YB))
        .unwrap();
    check(&f, &BIN_PROBIT, tol, "binomial(probit)");
    let f = PoissonRegressor::identity()
        .build()
        .fit(&x(), &y(&YP))
        .unwrap();
    check(&f, &POIS_IDENTITY, tol, "poisson(identity)");
    let f = PoissonRegressor::log().build().fit(&x(), &y(&YP)).unwrap();
    check(&f, &POIS_LOG, 1e-10, "poisson(log)");
}

#[test]
fn tight_tolerance_fits_match_converged_r() {
    let tol = 1e-8;
    let f = TweedieRegressor::gamma()
        .tolerance(1e-15)
        .build()
        .fit(&x(), &y(&YG))
        .unwrap();
    check(&f, &GAMMA_LOG, tol, "Gamma(log)");
    let f = BinomialRegressor::cloglog()
        .tolerance(1e-15)
        .build()
        .fit(&x(), &y(&YB))
        .unwrap();
    check(&f, &BIN_CLOGLOG, tol, "binomial(cloglog)");
    let f = BinomialRegressor::probit()
        .tolerance(1e-15)
        .build()
        .fit(&x(), &y(&YB))
        .unwrap();
    check(&f, &BIN_PROBIT, tol, "binomial(probit)");
    let f = PoissonRegressor::identity()
        .tolerance(1e-15)
        .build()
        .fit(&x(), &y(&YP))
        .unwrap();
    check(&f, &POIS_IDENTITY, tol, "poisson(identity)");
}
