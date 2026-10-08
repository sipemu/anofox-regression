//! Issue #42 (R reference values from R 4.6.1 / MASS / lme4): log_likelihood is -deviance/2 for the Poisson (and NegBin) GLM and for the
//! Poisson/Binomial GLMM; R's logLik(glm) / lme4's logLik(glmer) include the saturated term
//! sum(log f(y | mu = y)).
use anofox_regression::solvers::{FittedRegressor, GlmmRegressor, PoissonRegressor, Regressor};
use faer::{Col, Mat};

const G: [usize; 48] = [
    0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5,
    5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7,
];
const Q1: [f64; 48] = [
    0.79, 0.62, 0.52, 0.61, 0.33, 0.03, -0.64, 0.13, -0.46, 0.44, -0.61, 0.85, -0.90, 0.93, -0.82,
    0.81, -0.63, -0.59, -0.42, 0.60, -0.05, 0.54, 0.98, 0.62, 0.27, 0.69, 0.00, 0.33, 0.02, 0.05,
    -0.31, 0.62, -0.57, -0.15, -0.24, 0.57, -0.27, 0.46, -0.24, 0.43, -0.93, 0.00, 0.44, 0.96,
    0.90, -0.41, -0.48, 0.85,
];
const PC: [f64; 48] = [
    1., 4., 1., 2., 2., 0., 1., 1., 2., 1., 1., 3., 0., 3., 2., 5., 2., 0., 0., 4., 2., 4., 7., 3.,
    1., 1., 2., 1., 3., 2., 1., 7., 0., 4., 5., 6., 4., 3., 2., 5., 2., 6., 4., 4., 5., 5., 4., 6.,
];

#[test]
fn poisson_glmm_loglik_matches_lme4() {
    let x = Mat::from_fn(48, 1, |i, _| Q1[i]);
    let y = Col::from_fn(48, |i| PC[i]);
    let f = GlmmRegressor::poisson()
        .with_intercept(true)
        .build()
        .fit(&x, &y, &G)
        .unwrap();
    println!("fixef {:?} logLik {}  (lme4 nAGQ=0: 0.84950417889667618 0.62925677676935809, logLik -88.372144804214017; saturated term -61.606327025805719)",
             f.fixed_effects(), f.log_likelihood());
    assert!(
        (f.fixed_effects()[1] - 0.62925677676935809).abs() < 1e-3,
        "same fit as lme4"
    );
    assert!(
        (f.log_likelihood() + 88.372144804214017).abs() < 1e-2,
        "log_likelihood {}",
        f.log_likelihood()
    );
}

#[test]
fn poisson_glm_loglik_matches_r() {
    let yv = [0.0, 1.0, 0.0, 3.0, 2.0, 0.0, 5.0, 1.0, 4.0, 0.0];
    let x = Mat::from_fn(10, 1, |i, _| (i + 1) as f64);
    let y = Col::from_fn(10, |i| yv[i]);
    let f = PoissonRegressor::log()
        .with_intercept(true)
        .build()
        .fit(&x, &y)
        .unwrap();
    let r = f.result();
    println!("Poisson GLM logLik {} AIC {}  (R: logLik -18.068639277609133, AIC 40.137278555218266; -deviance/2 = -9.8926852884654242)", r.log_likelihood, r.aic);
    assert!(
        (r.log_likelihood + 18.068639277609133).abs() < 1e-6,
        "log_likelihood {}",
        r.log_likelihood
    );
    assert!((r.aic - 40.137278555218266).abs() < 1e-5, "aic {}", r.aic);
}

fn nb_data() -> (Mat<f64>, Col<f64>) {
    let yv = [0.0, 1.0, 0.0, 3.0, 2.0, 0.0, 5.0, 1.0, 4.0, 0.0];
    (
        Mat::from_fn(10, 1, |i, _| (i + 1) as f64),
        Col::from_fn(10, |i| yv[i]),
    )
}

#[test]
fn negbin_fixed_theta_loglik_matches_r() {
    // R: glm(y ~ x, family = negative.binomial(2)); theta is a known family
    // parameter, so AIC counts only the 2 coefficients.
    let (x, y) = nb_data();
    let f = anofox_regression::solvers::NegativeBinomialRegressor::with_theta(2.0)
        .with_intercept(true)
        .build()
        .fit(&x, &y)
        .unwrap();
    let r = f.result();
    assert!(
        (r.log_likelihood + 16.875485994829198).abs() < 1e-6,
        "{}",
        r.log_likelihood
    );
    assert!((r.aic - 37.750971989658396).abs() < 1e-5, "{}", r.aic);
}

#[test]
fn negbin_estimated_theta_loglik_matches_glm_nb() {
    // R: MASS::glm.nb(y ~ x): logLik, AIC (theta counted), BIC
    let (x, y) = nb_data();
    let f = anofox_regression::solvers::NegativeBinomialRegressor::builder()
        .with_intercept(true)
        .estimate_theta(true)
        .build()
        .fit(&x, &y)
        .unwrap();
    let r = f.result();
    assert!(
        (r.log_likelihood + 16.824451939052764).abs() < 1e-5,
        "{}",
        r.log_likelihood
    );
    assert!((r.aic - 39.648903878105529).abs() < 1e-4, "{}", r.aic);
    assert!((r.bic - 40.556659157087665).abs() < 1e-4, "{}", r.bic);
}
