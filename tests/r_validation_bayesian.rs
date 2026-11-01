//! BayesianRidge and ARDRegression validation against scikit-learn 1.9.1.
//!
//! Reference values come from
//! `validation/python/generate_bayesian_validation.py`. Both estimators
//! follow sklearn's iteration step for step, so the coefficients,
//! hyperparameters, posterior covariance, iteration counts and posterior
//! predictive standard deviations agree to rounding.

#![allow(dead_code)]

use anofox_regression::core::IntervalType;
use anofox_regression::solvers::{ArdRegression, BayesianRidge, FittedRegressor, Regressor};
use faer::{Col, Mat};

include!("fixtures/bayesian_validation.rs");

const TOL: f64 = 1e-8;

fn assert_close(what: &str, got: f64, want: f64, tol: f64) {
    let scale = want.abs().max(1.0);
    assert!(
        (got - want).abs() <= tol * scale,
        "{what}: {got} vs sklearn {want}"
    );
}

fn assert_all_close(what: &str, got: &[f64], want: &[f64], tol: f64) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (j, (g, w)) in got.iter().zip(want).enumerate() {
        assert_close(&format!("{what}[{j}]"), *g, *w, tol);
    }
}

fn matrix(flat: &[f64], n: usize, p: usize) -> Mat<f64> {
    Mat::from_fn(n, p, |i, j| flat[i * p + j])
}

fn flat(m: &Mat<f64>) -> Vec<f64> {
    let mut v = Vec::with_capacity(m.nrows() * m.ncols());
    for i in 0..m.nrows() {
        for j in 0..m.ncols() {
            v.push(m[(i, j)]);
        }
    }
    v
}

fn col(c: &Col<f64>) -> Vec<f64> {
    (0..c.nrows()).map(|i| c[i]).collect()
}

#[allow(clippy::too_many_arguments)]
fn check_bayesian_ridge(
    x_flat: &[f64],
    y: &[f64],
    n: usize,
    p: usize,
    fit_intercept: bool,
    intercept: Option<f64>,
    coefs: &[f64],
    alpha: f64,
    lambda: f64,
    score: f64,
    sigma: &[f64],
    n_iter: usize,
    query: (&[f64], usize),
    pred: &[f64],
    std: &[f64],
) {
    let x = matrix(x_flat, n, p);
    let y = Col::from_fn(n, |i| y[i]);
    let fitted = BayesianRidge::builder()
        .fit_intercept(fit_intercept)
        .build()
        .fit(&x, &y)
        .unwrap();

    if let Some(want) = intercept {
        assert_close("intercept", fitted.result().intercept.unwrap(), want, TOL);
    } else {
        assert!(fitted.result().intercept.is_none());
    }
    assert_all_close("coef", &col(&fitted.result().coefficients), coefs, TOL);
    assert_close("alpha", fitted.alpha(), alpha, TOL);
    assert_close("lambda", fitted.lambda(), lambda, TOL);
    // sklearn evaluates its final score with the last loop iteration's
    // coefficients but the refit's residual sum of squares; we use the refit
    // for both, which differs in the 7th significant digit.
    assert_close(
        "log marginal likelihood",
        fitted.log_marginal_likelihood(),
        score,
        1e-6,
    );
    assert_all_close("sigma", &flat(fitted.sigma()), sigma, TOL);
    assert_eq!(fitted.n_iter(), n_iter, "n_iter");

    let xq = matrix(query.0, query.1, p);
    assert_all_close("predict", &col(&fitted.predict(&xq)), pred, TOL);
    assert_all_close("predict std", &col(&fitted.predict_std(&xq)), std, TOL);
}

#[test]
fn bayesian_ridge_matches_sklearn() {
    check_bayesian_ridge(
        &X_BR_FLAT,
        &Y_BR,
        N_BR,
        P_BR,
        true,
        Some(EXPECTED_INTERCEPT_BR),
        &EXPECTED_COEFS_BR,
        EXPECTED_ALPHA_BR,
        EXPECTED_LAMBDA_BR,
        EXPECTED_SCORE_BR,
        &EXPECTED_SIGMA_BR,
        EXPECTED_N_ITER_BR,
        (&X_QUERY_BR_FLAT, N_QUERY_BR),
        &EXPECTED_PRED_BR,
        &EXPECTED_STD_BR,
    );
}

#[test]
fn bayesian_ridge_more_features_than_rows_matches_sklearn() {
    check_bayesian_ridge(
        &X_BRW_FLAT,
        &Y_BRW,
        N_BRW,
        P_BRW,
        true,
        Some(EXPECTED_INTERCEPT_BRW),
        &EXPECTED_COEFS_BRW,
        EXPECTED_ALPHA_BRW,
        EXPECTED_LAMBDA_BRW,
        EXPECTED_SCORE_BRW,
        &EXPECTED_SIGMA_BRW,
        EXPECTED_N_ITER_BRW,
        (&X_QUERY_BRW_FLAT, N_QUERY_BRW),
        &EXPECTED_PRED_BRW,
        &EXPECTED_STD_BRW,
    );
}

#[test]
fn bayesian_ridge_without_intercept_matches_sklearn() {
    check_bayesian_ridge(
        &X_BRN_FLAT,
        &Y_BRN,
        N_BRN,
        P_BRN,
        false,
        None,
        &EXPECTED_COEFS_BRN,
        EXPECTED_ALPHA_BRN,
        EXPECTED_LAMBDA_BRN,
        EXPECTED_SCORE_BRN,
        &EXPECTED_SIGMA_BRN,
        EXPECTED_N_ITER_BRN,
        (&X_QUERY_BRN_FLAT, N_QUERY_BRN),
        &EXPECTED_PRED_BRN,
        &EXPECTED_STD_BRN,
    );
}

#[test]
fn bayesian_ridge_intervals_use_posterior_predictive_std() {
    let x = matrix(&X_BR_FLAT, N_BR, P_BR);
    let y = Col::from_fn(N_BR, |i| Y_BR[i]);
    let fitted = BayesianRidge::builder().build().fit(&x, &y).unwrap();
    let xq = matrix(&X_QUERY_BR_FLAT, N_QUERY_BR, P_BR);

    let z = 1.959_963_984_540_054;
    let pi = fitted.predict_with_interval(&xq, Some(IntervalType::Prediction), 0.95);
    for i in 0..N_QUERY_BR {
        assert_close("se", pi.se[i], EXPECTED_STD_BR[i], TOL);
        assert_close(
            "lower",
            pi.lower[i],
            EXPECTED_PRED_BR[i] - z * EXPECTED_STD_BR[i],
            TOL,
        );
        assert_close(
            "upper",
            pi.upper[i],
            EXPECTED_PRED_BR[i] + z * EXPECTED_STD_BR[i],
            TOL,
        );
    }

    // The interval for the mean drops the noise variance 1/alpha.
    let ci = fitted.predict_with_interval(&xq, Some(IntervalType::Confidence), 0.95);
    for (i, &want) in EXPECTED_STD_BR.iter().enumerate().take(N_QUERY_BR) {
        let mean_var = want.powi(2) - 1.0 / EXPECTED_ALPHA_BR;
        assert_close("mean se", ci.se[i], mean_var.max(0.0).sqrt(), 1e-6);
        assert!(ci.upper[i] - ci.lower[i] < pi.upper[i] - pi.lower[i]);
    }

    // No interval requested: point predictions only.
    let none = fitted.predict_with_interval(&xq, None, 0.95);
    assert_all_close("point", &col(&none.fit), &EXPECTED_PRED_BR, TOL);
}

#[test]
fn bayesian_fit_statistics_are_not_reported_as_zero() {
    let x = matrix(&X_BR_FLAT, N_BR, P_BR);
    let y = Col::from_fn(N_BR, |i| Y_BR[i]);
    let fitted = BayesianRidge::builder().build().fit(&x, &y).unwrap();
    let r = fitted.result();
    assert!(r.r_squared > 0.9);
    for v in [r.log_likelihood, r.aic, r.bic, r.f_statistic, r.rmse] {
        assert!(v.is_nan(), "expected NaN, got {v}");
    }
}

#[test]
fn ard_matches_sklearn() {
    let x = matrix(&X_ARD_FLAT, N_ARD, P_ARD);
    let y = Col::from_fn(N_ARD, |i| Y_ARD[i]);

    let fitted = ArdRegression::builder()
        .fit_intercept(true)
        .max_iter(300)
        .tolerance(1e-3)
        .threshold_lambda(10_000.0)
        .build()
        .fit(&x, &y)
        .unwrap();

    assert_close(
        "intercept",
        fitted.result().intercept.unwrap(),
        EXPECTED_INTERCEPT_ARD,
        TOL,
    );
    assert_all_close(
        "coef",
        &col(&fitted.result().coefficients),
        &EXPECTED_COEFS_ARD,
        TOL,
    );
    assert_close("alpha", fitted.alpha(), EXPECTED_ALPHA_ARD, TOL);
    assert_all_close("lambda", fitted.lambdas(), &EXPECTED_LAMBDAS_ARD, 1e-6);
    let kept: Vec<f64> = (0..P_ARD)
        .map(|j| f64::from(u8::from(fitted.kept_features().contains(&j))))
        .collect();
    assert_eq!(kept, EXPECTED_KEPT_ARD.to_vec(), "kept features");
    assert_all_close("sigma", &flat(fitted.sigma()), &EXPECTED_SIGMA_ARD, TOL);
    assert_eq!(fitted.n_iter(), EXPECTED_N_ITER_ARD, "n_iter");

    let xq = matrix(&X_QUERY_ARD_FLAT, N_QUERY_ARD, P_ARD);
    assert_all_close(
        "predict",
        &col(&fitted.predict(&xq)),
        &EXPECTED_PRED_ARD,
        TOL,
    );
    assert_all_close(
        "predict std",
        &col(&fitted.predict_std(&xq)),
        &EXPECTED_STD_ARD,
        TOL,
    );
    let pi = fitted.predict_with_interval(&xq, Some(IntervalType::Prediction), 0.9);
    let z = 1.644_853_626_951_472_2;
    for i in 0..N_QUERY_ARD {
        assert_close(
            "ard upper",
            pi.upper[i],
            EXPECTED_PRED_ARD[i] + z * EXPECTED_STD_ARD[i],
            TOL,
        );
    }
}

// Posterior SDs and 90% central credible intervals (#92). The coefficient SDs
// are sklearn's `sqrt(diag(sigma_))`; the intercept SD is checked against the
// (0, 0) element of the inverted full posterior precision of (b0, β) with a
// flat prior on b0, computed independently in the generator.

fn check_credible(
    ci: &anofox_regression::solvers::CredibleIntervals,
    sd: &[f64],
    lo: &[f64],
    hi: &[f64],
    intercept: Option<(f64, f64, f64)>,
) {
    assert_eq!(ci.level, 0.9);
    assert_all_close("posterior sd", &ci.sd, sd, TOL);
    assert_all_close("credible lower", &ci.lower, lo, TOL);
    assert_all_close("credible upper", &ci.upper, hi, TOL);
    match intercept {
        Some((b0_sd, b0_lo, b0_hi)) => {
            assert_close("intercept sd", ci.intercept_sd.unwrap(), b0_sd, TOL);
            let (l, h) = ci.intercept_interval.unwrap();
            assert_close("intercept lower", l, b0_lo, TOL);
            assert_close("intercept upper", h, b0_hi, TOL);
        }
        None => {
            assert!(ci.intercept_sd.is_none());
            assert!(ci.intercept_interval.is_none());
        }
    }
}

/// (x row-major, y, n, p, fit_intercept)
type Case<'a> = (&'a [f64], &'a [f64], usize, usize, bool);

#[test]
fn bayesian_ridge_credible_intervals_match_sklearn_posterior() {
    let cases: [Case; 3] = [
        (&X_BR_FLAT, &Y_BR, N_BR, P_BR, true),
        (&X_BRW_FLAT, &Y_BRW, N_BRW, P_BRW, true),
        (&X_BRN_FLAT, &Y_BRN, N_BRN, P_BRN, false),
    ];
    let expected = [
        (
            &EXPECTED_POST_SD_BR[..],
            &EXPECTED_CRED_LO_BR[..],
            &EXPECTED_CRED_HI_BR[..],
            Some((
                EXPECTED_INTERCEPT_SD_BR,
                EXPECTED_INTERCEPT_CRED_LO_BR,
                EXPECTED_INTERCEPT_CRED_HI_BR,
            )),
        ),
        (
            &EXPECTED_POST_SD_BRW[..],
            &EXPECTED_CRED_LO_BRW[..],
            &EXPECTED_CRED_HI_BRW[..],
            Some((
                EXPECTED_INTERCEPT_SD_BRW,
                EXPECTED_INTERCEPT_CRED_LO_BRW,
                EXPECTED_INTERCEPT_CRED_HI_BRW,
            )),
        ),
        (
            &EXPECTED_POST_SD_BRN[..],
            &EXPECTED_CRED_LO_BRN[..],
            &EXPECTED_CRED_HI_BRN[..],
            None,
        ),
    ];
    for ((x_flat, y, n, p, intercept), (sd, lo, hi, b0)) in cases.into_iter().zip(expected) {
        let x = matrix(x_flat, n, p);
        let y = Col::from_fn(n, |i| y[i]);
        let fitted = BayesianRidge::builder()
            .fit_intercept(intercept)
            .build()
            .fit(&x, &y)
            .unwrap();
        check_credible(&fitted.credible_intervals(0.9), sd, lo, hi, b0);
    }
}

#[test]
fn ard_credible_intervals_match_sklearn_posterior() {
    let x = matrix(&X_ARD_FLAT, N_ARD, P_ARD);
    let y = Col::from_fn(N_ARD, |i| Y_ARD[i]);
    let fitted = ArdRegression::builder().build().fit(&x, &y).unwrap();
    let ci = fitted.credible_intervals(0.9);
    check_credible(
        &ci,
        &EXPECTED_POST_SD_ARD,
        &EXPECTED_CRED_LO_ARD,
        &EXPECTED_CRED_HI_ARD,
        Some((
            EXPECTED_INTERCEPT_SD_ARD,
            EXPECTED_INTERCEPT_CRED_LO_ARD,
            EXPECTED_INTERCEPT_CRED_HI_ARD,
        )),
    );
    // Pruned features: SD 0 and the degenerate interval [0, 0].
    for (j, &kept) in EXPECTED_KEPT_ARD.iter().enumerate() {
        if kept == 0.0 {
            assert_eq!((ci.sd[j], ci.lower[j], ci.upper[j]), (0.0, 0.0, 0.0));
        }
    }
}
