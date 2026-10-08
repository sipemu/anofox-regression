//! Issue #49: negative binomial standard errors use dispersion 1, like
//! `MASS::glm.nb` / `summary(glm(family = negative.binomial(θ)), dispersion = 1)`.
//! Before the fix the covariance was scaled by `max(1, Pearson χ²/df)`, which
//! inflated SEs whenever χ²/df > 1 (here at θ = 10: χ²/df = 1.97).
//!
//! References (R 4.6.1, MASS 7.3.65):
//! ```r
//! f <- glm(nb ~ z1 + z2, family = negative.binomial(theta),
//!          control = glm.control(epsilon = 1e-14, maxit = 100))
//! coef(summary(f, dispersion = 1))
//! predict(f, se.fit = TRUE, dispersion = 1, type = "link")$se.fit[1:3]
//! ```
//! At the glm.nb ML theta (1.8294684732585813) these equal `summary(glm.nb(nb ~ z1 + z2))`.

use anofox_regression::prelude::*;
use anofox_regression::solvers::NegativeBinomialRegressor;
use faer::{Col, Mat};

const Z1: [f64; 30] = [
    0.63, -0.79, -0.44, -0.34, 0.7, 0.52, 0.88, 0.08, -0.42, -0.57, -0.84, 0.66, 0.49, -0.66,
    -0.27, -0.19, 0.05, -0.92, 0.31, -0.52, 0.66, -0.99, 0.42, 0.91, 0.06, 0.55, 0.79, 0.2, -0.93,
    0.13,
];
const Z2: [f64; 30] = [
    1.68, 0.53, 0.07, 0.42, 1.87, 0.08, 1.07, 0.33, 1.41, 1.73, 0.6, 1.9, 0.58, 1.11, 0.81, 1.43,
    0.08, 0.38, 1.36, 0.97, 1.17, 1.32, 0.7, 1.26, 0.45, 1.35, 0.97, 1.77, 0.16, 0.63,
];
const NB: [f64; 30] = [
    0., 2., 0., 2., 0., 1., 4., 5., 2., 7., 2., 4., 2., 5., 0., 8., 8., 2., 7., 2., 9., 0., 10.,
    6., 3., 3., 2., 7., 0., 10.,
];

struct Ref {
    theta: f64,
    coef: [f64; 3],
    se: [f64; 3],
    z: [f64; 3],
    p: [f64; 3],
    pred_se_link: [f64; 3],
}

const REFS: [Ref; 2] = [
    Ref {
        theta: 1.8294684732585813,
        coef: [1.141310475630e+00, 4.445075425644e-01, 1.563917335269e-01],
        se: [3.403116503810e-01, 2.953927095121e-01, 3.114970571954e-01],
        z: [3.353721432552e+00, 1.504802008481e+00, 5.020648828433e-01],
        p: [7.973258884584e-04, 1.323749872448e-01, 6.156218814779e-01],
        pred_se_link: [2.843149548512e-01, 2.969421509047e-01, 3.195353396532e-01],
    },
    Ref {
        theta: 10.0,
        coef: [1.186252064427e+00, 3.862941878080e-01, 1.145930039276e-01],
        se: [2.301565921406e-01, 2.005140224941e-01, 2.087673148773e-01],
        z: [5.154108571883e+00, 1.926519567076e+00, 5.489029927647e-01],
        p: [2.548402690564e-07, 5.403953102318e-02, 5.830720242408e-01],
        pred_se_link: [1.841027998536e-01, 2.078876640008e-01, 2.193592549675e-01],
    },
];

fn close(a: f64, b: f64, tol: f64, what: &str) {
    assert!(
        (a - b).abs() <= tol * (1.0 + b.abs()),
        "{what}: {a} vs R {b}"
    );
}

#[test]
fn negbin_fixed_theta_inference_matches_mass() {
    let x = Mat::from_fn(30, 2, |i, j| if j == 0 { Z1[i] } else { Z2[i] });
    let y = Col::from_fn(30, |i| NB[i]);
    for r in &REFS {
        let fit = NegativeBinomialRegressor::with_theta(r.theta)
            .compute_inference(true)
            .tolerance(1e-12)
            .build()
            .fit(&x, &y)
            .unwrap();
        assert_eq!(fit.dispersion, 1.0);
        let res = fit.result();
        let tag = |s: &str| format!("theta={} {s}", r.theta);
        close(res.intercept.unwrap(), r.coef[0], 1e-6, &tag("intercept"));
        close(res.intercept_std_error.unwrap(), r.se[0], 1e-6, &tag("se0"));
        close(res.intercept_t_statistic.unwrap(), r.z[0], 1e-6, &tag("z0"));
        close(res.intercept_p_value.unwrap(), r.p[0], 1e-6, &tag("p0"));
        let se = res.std_errors.as_ref().unwrap();
        let z = res.t_statistics.as_ref().unwrap();
        let p = res.p_values.as_ref().unwrap();
        for j in 0..2 {
            close(res.coefficients[j], r.coef[j + 1], 1e-6, &tag("coef"));
            close(se[j], r.se[j + 1], 1e-6, &tag("se"));
            close(z[j], r.z[j + 1], 1e-6, &tag("z"));
            close(p[j], r.p[j + 1], 1e-6, &tag("p"));
        }
        let pred = fit.predict_with_se(&x, PredictionType::Link, None, 0.95);
        for i in 0..3 {
            close(pred.se[i], r.pred_se_link[i], 1e-6, &tag("pred se"));
        }
    }
}

/// `estimate_dispersion(true)` reproduces R's `summary.glm` default for
/// `glm(family = negative.binomial(θ))`, which estimates the Pearson dispersion
/// (here 0.8951608253, below 1 — the old `max(1, ·)` floor could not reach it):
/// `coef(summary(glm(nb ~ z1 + z2, family = negative.binomial(1.8294684732585813))))`
/// SE = 0.32197885489695210 0.27947972467922549 0.29471652136293919.
#[test]
fn negbin_estimate_dispersion_matches_summary_glm() {
    let x = Mat::from_fn(30, 2, |i, j| if j == 0 { Z1[i] } else { Z2[i] });
    let y = Col::from_fn(30, |i| NB[i]);
    let fit = NegativeBinomialRegressor::with_theta(1.8294684732585813)
        .compute_inference(true)
        .tolerance(1e-12)
        .estimate_dispersion(true)
        .build()
        .fit(&x, &y)
        .unwrap();
    close(fit.dispersion, 0.89516082533230956, 1e-6, "dispersion");
    let res = fit.result();
    let se = res.std_errors.as_ref().unwrap();
    close(
        res.intercept_std_error.unwrap(),
        0.32197885489695210,
        1e-6,
        "se0",
    );
    close(se[0], 0.27947972467922549, 1e-6, "se1");
    close(se[1], 0.29471652136293919, 1e-6, "se2");
}
