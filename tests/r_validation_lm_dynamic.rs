//! LmDynamicRegressor validation against `greybox::lmDynamic` (normal
//! distribution, `bruteforce = TRUE`, explicit lowess span).
//!
//! Reference values come from `tests/r_scripts/generate_lm_dynamic_validation.R`.

use anofox_regression::solvers::{
    FittedRegressor, InformationCriterion, LmDynamicRegressor, Regressor,
};
use faer::{Col, Mat};

include!("fixtures/lm_dynamic_validation.rs");

const TOL: f64 = 1e-8;

#[allow(clippy::too_many_arguments)]
fn check(
    n: usize,
    p: usize,
    x_flat: &[f64],
    y: &[f64],
    ic: &str,
    span: Option<f64>,
    dynamic: &[f64],
    weights: &[f64],
    coef: &[f64],
    loglik: f64,
) {
    let x = Mat::from_fn(n, p, |i, j| x_flat[i * p + j]);
    let y = Col::from_fn(n, |i| y[i]);
    let ic = match ic {
        "AIC" => InformationCriterion::AIC,
        "BIC" => InformationCriterion::BIC,
        _ => InformationCriterion::AICc,
    };
    let mut builder = LmDynamicRegressor::builder().ic(ic);
    builder = match span {
        Some(f) => builder.lowess_span(f),
        None => builder.no_smoothing(),
    };
    let fitted = builder.build().fit(&x, &y).unwrap();

    let w = fitted
        .smoothed_weights()
        .unwrap_or_else(|| fitted.model_weights());
    let n_models = 1 << p;
    assert_eq!(w.ncols(), n_models, "candidate models");
    for i in 0..n {
        for m in 0..n_models {
            let want = weights[i * n_models + m];
            assert!(
                (w[(i, m)] - want).abs() <= TOL,
                "weight[{i}, {m}] = {} vs greybox {want}",
                w[(i, m)]
            );
        }
    }

    let d = fitted.dynamic_coefficients();
    for i in 0..n {
        for j in 0..=p {
            let want = dynamic[i * (p + 1) + j];
            assert!(
                (d[(i, j)] - want).abs() <= TOL * want.abs().max(1.0),
                "coefficient[{i}, {j}] = {} vs greybox {want}",
                d[(i, j)]
            );
        }
    }

    let r = fitted.result();
    assert!((r.intercept.unwrap() - coef[0]).abs() <= TOL * coef[0].abs().max(1.0));
    for j in 0..p {
        assert!(
            (r.coefficients[j] - coef[j + 1]).abs() <= TOL * coef[j + 1].abs().max(1.0),
            "mean coefficient {j}: {} vs greybox {}",
            r.coefficients[j],
            coef[j + 1]
        );
    }
    assert!(
        (r.log_likelihood - loglik).abs() <= 1e-8 * loglik.abs(),
        "log-likelihood {} vs greybox {loglik}",
        r.log_likelihood
    );
}

#[test]
fn lm_dynamic_matches_greybox_without_smoothing() {
    check(
        N_STEP_AICC,
        P_STEP_AICC,
        &X_STEP_AICC,
        &Y_STEP_AICC,
        IC_STEP_AICC,
        SPAN_STEP_AICC,
        &EXPECTED_DYNAMIC_STEP_AICC,
        &EXPECTED_WEIGHTS_STEP_AICC,
        &EXPECTED_COEF_STEP_AICC,
        EXPECTED_LOGLIK_STEP_AICC,
    );
}

#[test]
fn lm_dynamic_matches_greybox_aic_lowess() {
    check(
        N_STEP_AIC_F05,
        P_STEP_AIC_F05,
        &X_STEP_AIC_F05,
        &Y_STEP_AIC_F05,
        IC_STEP_AIC_F05,
        SPAN_STEP_AIC_F05,
        &EXPECTED_DYNAMIC_STEP_AIC_F05,
        &EXPECTED_WEIGHTS_STEP_AIC_F05,
        &EXPECTED_COEF_STEP_AIC_F05,
        EXPECTED_LOGLIK_STEP_AIC_F05,
    );
}

#[test]
fn lm_dynamic_matches_greybox_bic_three_regressors() {
    check(
        N_THREE_BIC_F03,
        P_THREE_BIC_F03,
        &X_THREE_BIC_F03,
        &Y_THREE_BIC_F03,
        IC_THREE_BIC_F03,
        SPAN_THREE_BIC_F03,
        &EXPECTED_DYNAMIC_THREE_BIC_F03,
        &EXPECTED_WEIGHTS_THREE_BIC_F03,
        &EXPECTED_COEF_THREE_BIC_F03,
        EXPECTED_LOGLIK_THREE_BIC_F03,
    );
}

#[test]
fn lm_dynamic_matches_greybox_aicc_wide_span() {
    check(
        N_THREE_AICC_F09,
        P_THREE_AICC_F09,
        &X_THREE_AICC_F09,
        &Y_THREE_AICC_F09,
        IC_THREE_AICC_F09,
        SPAN_THREE_AICC_F09,
        &EXPECTED_DYNAMIC_THREE_AICC_F09,
        &EXPECTED_WEIGHTS_THREE_AICC_F09,
        &EXPECTED_COEF_THREE_AICC_F09,
        EXPECTED_LOGLIK_THREE_AICC_F09,
    );
}

#[test]
fn lm_dynamic_predict_uses_mean_coefficients() {
    let x = Mat::from_fn(N_STEP_AICC, P_STEP_AICC, |i, j| {
        X_STEP_AICC[i * P_STEP_AICC + j]
    });
    let y = Col::from_fn(N_STEP_AICC, |i| Y_STEP_AICC[i]);
    let fitted = LmDynamicRegressor::builder()
        .no_smoothing()
        .build()
        .fit(&x, &y)
        .unwrap();
    let pred = fitted.predict(&x);
    let c = &EXPECTED_COEF_STEP_AICC;
    for i in 0..N_STEP_AICC {
        let want = c[0] + c[1] * x[(i, 0)] + c[2] * x[(i, 1)];
        assert!((pred[i] - want).abs() < 1e-8);
    }
}
