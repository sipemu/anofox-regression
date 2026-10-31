//! #189: `PoissonRegressor` uses dispersion 1 (`glm(family = poisson)`), not
//! `max(1, Pearson χ²/df)`; `estimate_dispersion(true)` is quasi-Poisson.
//! #190: the null deviance of the IRLS GLMs (Poisson, negative binomial,
//! Tweedie / Gamma, binomial) follows `glm.fit` with an offset, without an
//! intercept and with prior weights.
//!
//! Reference values: `tests/r_scripts/generate_glm_null_deviance_validation.R`
//! (R 4.x, MASS; `glm.control(epsilon = 1e-15)`).

use anofox_regression::solvers::{
    BinomialRegressor, FittedRegressor, NegativeBinomialRegressor, PoissonRegressor, Regressor,
    TweedieRegressor,
};
use faer::{Col, Mat};

const X1: [f64; N] = [
    0.40999999999999998,
    0.186,
    0.17399999999999999,
    0.39900000000000002,
    -0.54200000000000004,
    -0.91300000000000003,
    -0.19900000000000001,
    -0.091999999999999998,
    -0.041000000000000002,
    -0.61799999999999999,
    0.69099999999999995,
    0.46500000000000002,
    0.33800000000000002,
    0.60399999999999998,
    -0.311,
    -0.69199999999999995,
    0.377,
    -0.88100000000000001,
    0.060999999999999999,
    0.157,
    0.75600000000000001,
    0.17699999999999999,
    -0.311,
    0.66300000000000003,
    -0.76400000000000001,
    0.79700000000000004,
    0.71899999999999997,
    0.61499999999999999,
    0.437,
    -0.56000000000000005,
    -0.626,
    -0.28999999999999998,
    0.051999999999999998,
    -0.81599999999999995,
    0.26100000000000001,
    0.30499999999999999,
    0.021999999999999999,
    0.13800000000000001,
    0.13500000000000001,
    0.41699999999999998,
];
const X2: [f64; N] = [
    1.585,
    0.621,
    1.425,
    0.042999999999999997,
    1.881,
    0.82999999999999996,
    0.91100000000000003,
    1.02,
    0.88800000000000001,
    1.649,
    1.6040000000000001,
    0.10100000000000001,
    0.57799999999999996,
    0.186,
    0.48199999999999998,
    1.4870000000000001,
    0.081000000000000003,
    0.76400000000000001,
    1.9099999999999999,
    0.063,
    1.605,
    0.97199999999999998,
    1.1779999999999999,
    0.46500000000000002,
    0.78700000000000003,
    1.6040000000000001,
    1.1579999999999999,
    0.055,
    1.5920000000000001,
    0.88400000000000001,
    0.26700000000000002,
    0.40999999999999998,
    0.92600000000000005,
    0.51100000000000001,
    1.1439999999999999,
    1.1639999999999999,
    1.577,
    1.7689999999999999,
    0.19,
    1.77,
];
const OFF: [f64; N] = [
    0.73716406597671957,
    1.0557047876568333,
    0.52532006991644331,
    -0.23445731121448313,
    0.9887974226609032,
    0.3839009301923238,
    0.0079681696491768813,
    0.75612197972133366,
    0.71049581889455837,
    0.90825856017689077,
    0.9058362739702962,
    0.9891693772867276,
    0.17981842657583616,
    1.013780353151412,
    1.0888985964804707,
    0.74193734472937733,
    0.16636153721522529,
    0.49408629976169266,
    0.60103189165213966,
    0.33504264381161847,
    -0.14966077455440627,
    0.1292723357041391,
    0.61572603359136047,
    0.83768033129224362,
    -0.59602046982922252,
    0.52945108788915551,
    0.67803354274989713,
    -0.18152187662339034,
    0.37706563358646639,
    0.60976557162089429,
    -0.57270102748407825,
    -0.069350078134793172,
    0.50621501120830747,
    -0.23825718912425789,
    1.082482906738226,
    0.13627761829254775,
    0.80513568250799261,
    0.9442949282837525,
    0.60048349569652593,
    -0.3133418192323586,
];
const Y: [f64; N] = [
    2.0, 7.0, 5.0, 3.0, 2.0, 0.0, 1.0, 3.0, 9.0, 0.0, 0.0, 4.0, 2.0, 3.0, 4.0, 0.0, 0.0, 3.0, 2.0,
    0.0, 0.0, 0.0, 6.0, 5.0, 0.0, 2.0, 8.0, 4.0, 2.0, 3.0, 0.0, 2.0, 0.0, 0.0, 10.0, 1.0, 2.0, 1.0,
    0.0, 0.0,
];
const YG: [f64; N] = [
    0.54349999999999998,
    0.1943,
    1.9731000000000001,
    0.52910000000000001,
    5.1048999999999998,
    1.8326,
    0.87270000000000003,
    1.9988999999999999,
    2.6898,
    0.47660000000000002,
    3.9716,
    3.3066,
    2.5007000000000001,
    9.4115000000000002,
    1.0742,
    2.0598999999999998,
    3.1124999999999998,
    2.0918000000000001,
    4.3661000000000003,
    2.8555000000000001,
    0.077799999999999994,
    1.6197999999999999,
    4.7111999999999998,
    4.5556000000000001,
    0.39539999999999997,
    0.5534,
    4.5129000000000001,
    0.92900000000000005,
    1.8197000000000001,
    3.2852999999999999,
    0.25580000000000003,
    0.1331,
    3.3176000000000001,
    0.56989999999999996,
    6.1041999999999996,
    2.7206999999999999,
    3.9306000000000001,
    5.1574999999999998,
    3.0859999999999999,
    1.5094000000000001,
];
const M: [f64; N] = [
    8.0, 12.0, 12.0, 12.0, 5.0, 13.0, 6.0, 7.0, 8.0, 10.0, 13.0, 7.0, 13.0, 11.0, 14.0, 14.0, 14.0,
    5.0, 9.0, 9.0, 6.0, 8.0, 12.0, 15.0, 7.0, 12.0, 15.0, 11.0, 14.0, 6.0, 8.0, 6.0, 8.0, 9.0,
    15.0, 8.0, 6.0, 10.0, 10.0, 5.0,
];
const P: [f64; N] = [
    0.625,
    0.41666666666666669,
    0.33333333333333331,
    0.75,
    0.20000000000000001,
    0.30769230769230771,
    0.33333333333333331,
    0.2857142857142857,
    0.25,
    0.29999999999999999,
    0.46153846153846156,
    0.7142857142857143,
    0.53846153846153844,
    0.63636363636363635,
    0.42857142857142855,
    0.35714285714285715,
    0.35714285714285715,
    0.20000000000000001,
    0.44444444444444442,
    0.22222222222222221,
    0.66666666666666663,
    0.375,
    0.41666666666666669,
    0.46666666666666667,
    0.2857142857142857,
    0.66666666666666663,
    0.59999999999999998,
    0.72727272727272729,
    0.42857142857142855,
    0.5,
    0.125,
    0.33333333333333331,
    0.375,
    0.1111111111111111,
    0.46666666666666667,
    0.25,
    0.66666666666666663,
    0.69999999999999996,
    0.5,
    0.20000000000000001,
];
const POIS_COEF: [f64; 3] = [
    0.94964530760236021,
    0.53189154566507002,
    -0.13986533599114764,
];
const POIS_SE: [f64; 3] = [
    0.18758711106004866,
    0.22145236529397813,
    0.16923382433920306,
];
const POIS_Z: [f64; 3] = [5.0624230110264268, 2.4018327596500662, -0.82646206535408184];
const POIS_P: [f64; 3] = [
    4.1396125411343394e-07,
    0.016313164583742431,
    0.4085420284951285,
];
const QPOIS_DISP: f64 = 2.9640133035163916;
const QPOIS_SE: [f64; 3] = [
    0.32295578578422257,
    0.38125925732923788,
    0.29135819839584154,
];
const QPOIS_P: [f64; 3] = [
    0.005621414311435818,
    0.1713067671480632,
    0.63402057602830519,
];
const PEARSON_OVER_DF: f64 = 2.9640132948656603;
const ND_POIS: f64 = 120.12556841789343;
const ND_POIS_OFF: f64 = 95.051600690519365;
const ND_POIS_NOINT: f64 = 176.21556598984222;
const ND_POIS_OFF_NOINT: f64 = 104.14817488989422;
const ND_POIS_SQRT_OFF: f64 = 96.700860253296739;
const ND_NB_OFF: f64 = 45.992595866162922;
const ND_NB_NOINT: f64 = 85.630549606922301;
const ND_GAMMA_OFF: f64 = 24.161059333562999;
const ND_GAMMA_NOINT: f64 = 82.27836757954266;
const ND_BIN_W: f64 = 45.628437177452234;
const ND_BIN_W_OFF: f64 = 46.176629862820263;
const ND_BIN_W_NOINT: f64 = 51.262778259600516;

const N: usize = 40;

fn x(cols: &[&[f64; N]]) -> Mat<f64> {
    Mat::from_fn(N, cols.len(), |i, j| cols[j][i])
}

fn col(v: &[f64; N]) -> Col<f64> {
    Col::from_fn(N, |i| v[i])
}

fn close(a: f64, b: f64, tol: f64, what: &str) {
    assert!(
        (a - b).abs() <= tol * (1.0 + b.abs()),
        "{what}: {a} vs R {b}"
    );
}

#[test]
fn poisson_overdispersed_uses_dispersion_one() {
    let fit = PoissonRegressor::log()
        .compute_inference(true)
        .tolerance(1e-12)
        .build()
        .fit(&x(&[&X1, &X2]), &col(&Y))
        .unwrap();
    assert_eq!(fit.dispersion, 1.0);
    let r = fit.result();
    let se = r.std_errors.as_ref().unwrap();
    let p = r.p_values.as_ref().unwrap();
    close(r.intercept.unwrap(), POIS_COEF[0], 1e-8, "b0");
    close(r.intercept_std_error.unwrap(), POIS_SE[0], 1e-8, "se0");
    close(r.intercept_p_value.unwrap(), POIS_P[0], 1e-8, "p0");
    for j in 0..2 {
        close(r.coefficients[j], POIS_COEF[j + 1], 1e-8, "coef");
        close(se[j], POIS_SE[j + 1], 1e-8, "se");
        close(p[j], POIS_P[j + 1], 1e-8, "p");
    }
    close(fit.null_deviance, ND_POIS, 1e-10, "null deviance");
}

#[test]
fn poisson_estimate_dispersion_is_quasipoisson() {
    let fit = PoissonRegressor::log()
        .compute_inference(true)
        .estimate_dispersion(true)
        .tolerance(1e-12)
        .build()
        .fit(&x(&[&X1, &X2]), &col(&Y))
        .unwrap();
    close(fit.dispersion, PEARSON_OVER_DF, 1e-8, "dispersion");
    close(fit.dispersion, QPOIS_DISP, 1e-7, "quasipoisson dispersion");
    let r = fit.result();
    let se = r.std_errors.as_ref().unwrap();
    let p = r.p_values.as_ref().unwrap();
    close(r.intercept_std_error.unwrap(), QPOIS_SE[0], 1e-7, "se0");
    close(
        r.intercept_p_value.unwrap(),
        QPOIS_P[0],
        1e-7,
        "p0 (t test)",
    );
    for j in 0..2 {
        close(se[j], QPOIS_SE[j + 1], 1e-7, "se");
        close(p[j], QPOIS_P[j + 1], 1e-7, "p (t test)");
    }
}

#[test]
fn poisson_null_deviance_offset_and_no_intercept() {
    let xm = x(&[&X1, &X2]);
    let y = col(&Y);
    let off = col(&OFF);
    let expo_sqrt = Col::from_fn(N, |i| OFF[i].exp().sqrt());
    let cases = [
        (true, Some(off.clone()), ND_POIS_OFF, "offset"),
        (false, None, ND_POIS_NOINT, "no intercept"),
        (
            false,
            Some(off.clone()),
            ND_POIS_OFF_NOINT,
            "offset, no intercept",
        ),
    ];
    for (intercept, offset, expected, what) in cases {
        let mut b = PoissonRegressor::log()
            .with_intercept(intercept)
            .tolerance(1e-12);
        if let Some(o) = offset {
            b = b.offset(o);
        }
        let fit = b.build().fit(&xm, &y).unwrap();
        close(fit.null_deviance, expected, 1e-9, what);
    }
    let fit = PoissonRegressor::sqrt()
        .offset(expo_sqrt)
        .tolerance(1e-12)
        .build()
        .fit(&xm, &y)
        .unwrap();
    close(
        fit.null_deviance,
        ND_POIS_SQRT_OFF,
        1e-9,
        "sqrt link, offset",
    );
}

#[test]
fn negative_binomial_null_deviance() {
    let xm = x(&[&X1, &X2]);
    let y = col(&Y);
    let fit = NegativeBinomialRegressor::with_theta(1.5)
        .offset(col(&OFF))
        .tolerance(1e-12)
        .build()
        .fit(&xm, &y)
        .unwrap();
    close(fit.null_deviance, ND_NB_OFF, 1e-9, "offset");
    let fit = NegativeBinomialRegressor::with_theta(1.5)
        .with_intercept(false)
        .tolerance(1e-12)
        .build()
        .fit(&xm, &y)
        .unwrap();
    close(fit.null_deviance, ND_NB_NOINT, 1e-9, "no intercept");
}

#[test]
fn gamma_null_deviance() {
    let yg = col(&YG);
    let fit = TweedieRegressor::gamma()
        .offset(col(&OFF))
        .tolerance(1e-12)
        .build()
        .fit(&x(&[&X1]), &yg)
        .unwrap();
    close(fit.null_deviance, ND_GAMMA_OFF, 1e-9, "offset");
    let fit = TweedieRegressor::gamma()
        .with_intercept(false)
        .tolerance(1e-12)
        .build()
        .fit(&x(&[&X1, &X2]), &yg)
        .unwrap();
    close(fit.null_deviance, ND_GAMMA_NOINT, 1e-9, "no intercept");
}

#[test]
fn binomial_null_deviance_weights_offset_no_intercept() {
    let p = col(&P);
    let m = col(&M);
    let fit = BinomialRegressor::logistic()
        .weights(m.clone())
        .tolerance(1e-12)
        .build()
        .fit(&x(&[&X1]), &p)
        .unwrap();
    close(fit.null_deviance, ND_BIN_W, 1e-10, "weights");
    let fit = BinomialRegressor::logistic()
        .weights(m.clone())
        .offset(Col::from_fn(N, |i| 0.1 * X2[i]))
        .tolerance(1e-12)
        .build()
        .fit(&x(&[&X1]), &p)
        .unwrap();
    close(fit.null_deviance, ND_BIN_W_OFF, 1e-9, "weights, offset");
    let fit = BinomialRegressor::logistic()
        .weights(m)
        .with_intercept(false)
        .tolerance(1e-12)
        .build()
        .fit(&x(&[&X1]), &p)
        .unwrap();
    close(
        fit.null_deviance,
        ND_BIN_W_NOINT,
        1e-10,
        "weights, no intercept",
    );
    // R² = 1 − deviance / null deviance uses the corrected null deviance.
    close(
        fit.result().r_squared,
        1.0 - fit.deviance / ND_BIN_W_NOINT,
        1e-10,
        "r_squared",
    );
}
