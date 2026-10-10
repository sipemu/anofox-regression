//! Model description and information criteria of the family-function GLM results,
//! the AFT result and the GLMM (consumed by the anofox-statistics DuckDB extension
//! for its stable model structs).
//!
//! GLMM reference values from R 4.6.1 / lme4 2.0.1 on the issue #42 data:
//!   m1 <- glmer(pc ~ q1 + (1 | g), family = poisson, nAGQ = 0)
//!   m2 <- lmer(yg ~ q1 + (1 | g))            # REML, yg <- pc + 2 * q1
//!   m3 <- lmer(yg ~ q1 + (1 + q1 | g))
//!   attr(logLik(m), "df"); logLik(m); AIC(m); BIC(m)
use anofox_regression::core::{HasModelInfo, ModelInfo};
use anofox_regression::solvers::aft::{fit_aft, AftDistribution, AftOptions};
use anofox_regression::solvers::penalized_glm::{
    fit_binomial, fit_gamma, fit_logistic, fit_negbinomial, fit_poisson, fit_tweedie,
    PenalizedGlmFamily, PenalizedGlmRegressor,
};
use anofox_regression::solvers::penalized_glm::{
    BinomialOptions, GammaOptions, LogisticOptions, NegBinomialOptions, PoissonOptions,
    TweedieOptions,
};
use anofox_regression::solvers::GlmmRegressor;
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

fn info(model_type: &'static str, family: &'static str, link: &'static str) -> ModelInfo {
    ModelInfo::new(model_type, Some(family), link)
}

fn x_cols() -> Vec<Vec<f64>> {
    vec![Q1.to_vec()]
}

#[test]
fn glm_family_results_describe_the_model() {
    let x = x_cols();
    let counts = PC.to_vec();
    let positive: Vec<f64> = PC.iter().map(|v| v + 0.5).collect();
    let binary: Vec<f64> = PC.iter().map(|&v| f64::from(u8::from(v >= 3.0))).collect();

    let poisson = fit_poisson(&counts, &x, &PoissonOptions::default()).unwrap();
    assert_eq!(poisson.model_info(), info("glm", "poisson", "log"));

    let binomial = fit_binomial(&binary, &x, &BinomialOptions::default()).unwrap();
    assert_eq!(binomial.model_info(), info("glm", "binomial", "logit"));

    let logistic = fit_logistic(&binary, &x, &LogisticOptions::default()).unwrap();
    assert_eq!(logistic.fit.model_info(), info("glm", "binomial", "logit"));

    let negbin = fit_negbinomial(&counts, &x, &NegBinomialOptions::default()).unwrap();
    assert_eq!(negbin.model_info(), info("glm", "negative_binomial", "log"));

    let gamma = fit_gamma(&positive, &x, &GammaOptions::default()).unwrap();
    assert_eq!(gamma.model_info(), info("glm", "gamma", "log"));

    let tweedie = fit_tweedie(
        &positive,
        &x,
        &TweedieOptions {
            power: 1.5,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(tweedie.model_info(), info("glm", "tweedie", "log"));
}

#[test]
fn glm_result_and_regressor_agree() {
    let xm = Mat::from_fn(48, 1, |i, _| Q1[i]);
    let y = Col::from_fn(48, |i| PC[i]);
    let fitted = PenalizedGlmRegressor::builder()
        .family(PenalizedGlmFamily::Poisson(
            anofox_regression::core::PoissonLink::Log,
        ))
        .build()
        .fit(&xm, &y)
        .unwrap();
    assert_eq!(fitted.model_info(), fitted.result().model_info());
}

#[test]
fn aft_result_describes_the_model() {
    let time: Vec<f64> = PC.iter().map(|v| v + 1.0).collect();
    let event: Vec<f64> = (0..48).map(|i| f64::from(u8::from(i % 4 != 0))).collect();
    let r = fit_aft(
        &time,
        &x_cols(),
        &event,
        &AftOptions {
            dist: AftDistribution::LogNormal,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(r.model_info(), info("aft", "lognormal", "log"));
}

fn close(actual: f64, expected: f64, tol: f64, what: &str) {
    assert!(
        (actual - expected).abs() < tol,
        "{what}: {actual} vs {expected}"
    );
}

#[test]
fn poisson_glmm_information_criteria_match_lme4() {
    let x = Mat::from_fn(48, 1, |i, _| Q1[i]);
    let y = Col::from_fn(48, |i| PC[i]);
    let f = GlmmRegressor::poisson()
        .with_intercept(true)
        .build()
        .fit(&x, &y, &G)
        .unwrap();
    assert_eq!(f.n_parameters(), 3);
    assert_eq!(f.n_observations(), 48);
    close(f.aic(), 182.7442896084, 2e-2, "aic");
    close(f.bic(), 188.3578926412, 2e-2, "bic");
}

#[test]
fn gaussian_lmm_information_criteria_match_lme4() {
    let x = Mat::from_fn(48, 1, |i, _| Q1[i]);
    let y = Col::from_fn(48, |i| PC[i] + 2.0 * Q1[i]);
    let f = GlmmRegressor::gaussian()
        .with_intercept(true)
        .build()
        .fit(&x, &y, &G)
        .unwrap();
    assert_eq!(f.n_parameters(), 4);
    close(f.log_likelihood(), -89.5996683689, 1e-4, "logLik");
    close(f.aic(), 187.1993367378, 1e-3, "aic");
    close(f.bic(), 194.6841407814, 1e-3, "bic");

    // A random slope adds two covariance parameters (variance and covariance).
    let s = GlmmRegressor::gaussian()
        .with_intercept(true)
        .random_slopes(vec![0])
        .build()
        .fit(&x, &y, &G)
        .unwrap();
    assert_eq!(s.n_parameters(), 6);
    close(s.aic(), 186.9209557285, 1e-2, "aic (slope)");
    close(s.bic(), 198.1481617940, 1e-2, "bic (slope)");
}
