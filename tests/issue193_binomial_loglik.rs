//! #193: binomial log-likelihood / AIC / BIC with proportion responses and
//! trial weights include the log binomial coefficient, as R's `logLik.glm`.
//!
//! Reference values: `tests/r_scripts/generate_binomial_loglik_validation.R`.

use anofox_regression::solvers::{BinomialRegressor, FittedRegressor, Regressor};
use faer::{Col, Mat};

const X: [f64; N] = [
    0.089999999999999997,
    -1.1160000000000001,
    -0.59399999999999997,
    -0.98799999999999999,
    1.173,
    0.22800000000000001,
    -0.115,
    0.77500000000000002,
    0.54800000000000004,
    -0.95999999999999996,
    -1.2150000000000001,
    -1.268,
    -0.20300000000000001,
    0.089999999999999997,
    0.76300000000000001,
    -0.59999999999999998,
    0.71399999999999997,
    -0.28599999999999998,
    0.83599999999999997,
    0.33200000000000002,
    1.2629999999999999,
    -1.3819999999999999,
    0.76700000000000002,
    0.65400000000000003,
    -0.70799999999999996,
    0.47099999999999997,
    0.49099999999999999,
    -0.68100000000000005,
    -0.94699999999999995,
    -0.021999999999999999,
];
const M: [f64; N] = [
    17.0, 11.0, 17.0, 20.0, 12.0, 13.0, 4.0, 9.0, 12.0, 20.0, 18.0, 6.0, 11.0, 11.0, 8.0, 3.0,
    20.0, 17.0, 15.0, 13.0, 16.0, 9.0, 7.0, 12.0, 3.0, 19.0, 3.0, 7.0, 20.0, 15.0,
];
const S: [f64; N] = [
    6.0, 0.0, 7.0, 5.0, 6.0, 5.0, 1.0, 2.0, 7.0, 2.0, 2.0, 1.0, 4.0, 5.0, 4.0, 1.0, 12.0, 3.0, 8.0,
    7.0, 7.0, 0.0, 4.0, 6.0, 1.0, 12.0, 2.0, 2.0, 6.0, 4.0,
];
const YB: [f64; N] = [
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0,
    0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0,
];
const WB: [f64; N] = [
    3.0, 4.0, 3.0, 2.0, 4.0, 2.0, 3.0, 4.0, 1.0, 1.0, 3.0, 4.0, 3.0, 3.0, 3.0, 4.0, 3.0, 1.0, 2.0,
    3.0, 2.0, 1.0, 4.0, 2.0, 1.0, 3.0, 3.0, 1.0, 2.0, 2.0,
];
const PROP_LOGIT_LOGLIK: f64 = -51.152765256180913;
const PROP_LOGIT_AIC: f64 = 106.30553051236183;
const PROP_LOGIT_BIC: f64 = 109.10792527568614;
const PROP_PROBIT_LOGLIK: f64 = -50.979046535287551;
const PROP_PROBIT_AIC: f64 = 105.9580930705751;
const PROP_PROBIT_BIC: f64 = 108.76048783389942;
const CASEW_LOGLIK: f64 = -40.346235252271484;
const CASEW_AIC: f64 = 84.692470504542968;
const CASEW_BIC: f64 = 87.494865267867283;
const PLAIN_LOGLIK: f64 = -17.936548154008172;
const PLAIN_AIC: f64 = 39.873096308016343;
const PLAIN_BIC: f64 = 42.675491071340652;

const N: usize = 30;

fn check(fit: &impl FittedRegressor, ll: f64, aic: f64, bic: f64, what: &str) {
    let r = fit.result();
    for (got, want, name) in [
        (r.log_likelihood, ll, "logLik"),
        (r.aic, aic, "AIC"),
        (r.bic, bic, "BIC"),
    ] {
        assert!(
            (got - want).abs() < 1e-8 * (1.0 + want.abs()),
            "{what} {name}: {got} vs R {want}"
        );
    }
}

fn x() -> Mat<f64> {
    Mat::from_fn(N, 1, |i, _| X[i])
}

#[test]
fn proportions_with_trial_weights_match_r() {
    let p = Col::from_fn(N, |i| S[i] / M[i]);
    let m = Col::from_fn(N, |i| M[i]);
    let fit = BinomialRegressor::logistic()
        .weights(m.clone())
        .tolerance(1e-12)
        .build()
        .fit(&x(), &p)
        .unwrap();
    check(
        &fit,
        PROP_LOGIT_LOGLIK,
        PROP_LOGIT_AIC,
        PROP_LOGIT_BIC,
        "logit",
    );
    let fit = BinomialRegressor::probit()
        .weights(m)
        .tolerance(1e-12)
        .build()
        .fit(&x(), &p)
        .unwrap();
    check(
        &fit,
        PROP_PROBIT_LOGLIK,
        PROP_PROBIT_AIC,
        PROP_PROBIT_BIC,
        "probit",
    );
}

#[test]
fn binary_response_with_and_without_case_weights_match_r() {
    let y = Col::from_fn(N, |i| YB[i]);
    let fit = BinomialRegressor::logistic()
        .weights(Col::from_fn(N, |i| WB[i]))
        .tolerance(1e-12)
        .build()
        .fit(&x(), &y)
        .unwrap();
    check(&fit, CASEW_LOGLIK, CASEW_AIC, CASEW_BIC, "case weights");
    // 0/1 response: log-likelihood is −deviance/2 (Σ w log p̂).
    assert!((fit.result().log_likelihood + fit.deviance / 2.0).abs() < 1e-10);
    let fit = BinomialRegressor::logistic()
        .tolerance(1e-12)
        .build()
        .fit(&x(), &y)
        .unwrap();
    check(&fit, PLAIN_LOGLIK, PLAIN_AIC, PLAIN_BIC, "unweighted");
}
