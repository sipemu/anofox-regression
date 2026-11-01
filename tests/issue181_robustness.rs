//! #181: every estimator rejects bad input with an error, never a panic or
//! NaN coefficients in `Ok`. Estimators with an `NaAction` drop NaN rows
//! (`na.omit`, the default) but reject infinite values, as R's `lm.fit`
//! rejects `NA/NaN/Inf in 'x'`; all others reject any non-finite value.

use anofox_regression::inference::{
    compute_hc_standard_errors, compute_prediction_intervals, CoefficientInference, HcType,
};
use anofox_regression::solvers::aft::FittedAft;
use anofox_regression::solvers::penalized_glm::FittedPenalizedGlm;
use anofox_regression::solvers::*;
use faer::{Col, Mat};
use std::panic::{catch_unwind, AssertUnwindSafe};

const N: usize = 40;

fn data(p: usize, binary: bool) -> (Mat<f64>, Col<f64>) {
    let x = Mat::from_fn(N, p, |i, j| {
        ((i * (j + 2)) as f64 * 0.37).sin() + i as f64 * 0.05
    });
    let y = Col::from_fn(N, |i| {
        let v = if (i * 7) % 3 == 0 { 1.0 } else { 0.0 } + 0.5 * x[(i, 0)].abs() + 0.2;
        if binary {
            f64::from(u8::from(v > 0.8))
        } else {
            v
        }
    });
    (x, y)
}

/// The bad inputs: a NaN or ±∞ in x or y, mismatched lengths, no rows.
fn cases(p: usize, binary: bool) -> Vec<(&'static str, Mat<f64>, Col<f64>)> {
    let (x, y) = data(p, binary);
    let with_x = |v: f64| {
        let mut x = x.clone();
        x[(3, 0)] = v;
        x
    };
    let with_y = |v: f64| {
        let mut y = y.clone();
        y[5] = v;
        y
    };
    vec![
        ("x NaN", with_x(f64::NAN), y.clone()),
        ("x Inf", with_x(f64::INFINITY), y.clone()),
        ("y NaN", x.clone(), with_y(f64::NAN)),
        ("y -Inf", x.clone(), with_y(f64::NEG_INFINITY)),
        ("length", x.clone(), Col::from_fn(N - 1, |i| y[i])),
        ("empty", Mat::zeros(0, p), Col::zeros(0)),
    ]
}

/// `Ok(n)` for a fit on `n` rows, `Err` otherwise; a panic fails the test.
type Outcome = Result<usize, String>;

fn check(
    name: &str,
    na_action: bool,
    fit: impl Fn(&Mat<f64>, &Col<f64>) -> Outcome,
    p: usize,
    binary: bool,
) {
    for (what, x, y) in cases(p, binary) {
        let out = catch_unwind(AssertUnwindSafe(|| fit(&x, &y)))
            .unwrap_or_else(|_| panic!("{name}, {what}: panicked"));
        let nan = what.ends_with("NaN");
        match out {
            Ok(n) if na_action && nan => {
                assert_eq!(n, N - 1, "{name}, {what}: NaN row not omitted")
            }
            Ok(n) => panic!("{name}, {what}: accepted (n = {n})"),
            Err(e) => assert!(
                !(na_action && nan),
                "{name}, {what}: NaN row not omitted: {e}"
            ),
        }
    }
}

fn regressor<R: Regressor>(r: R) -> impl Fn(&Mat<f64>, &Col<f64>) -> Outcome {
    move |x, y| match r.fit(x, y) {
        Ok(f) => {
            let res = f.result();
            let finite = res.coefficients.iter().all(|v| v.is_finite())
                && res.intercept.is_none_or(f64::is_finite);
            // NaN coefficients in `Ok` count as zero rows, which no case accepts
            Ok(if finite { res.n_observations } else { 0 })
        }
        Err(e) => Err(e.to_string()),
    }
}

#[test]
fn estimators_with_na_action_omit_nan_and_reject_inf() {
    check(
        "Alm",
        true,
        regressor(AlmRegressor::builder().build()),
        2,
        false,
    );
    check(
        "Binomial",
        true,
        regressor(BinomialRegressor::builder().build()),
        2,
        true,
    );
    check(
        "Bls",
        true,
        regressor(BlsRegressor::builder().build()),
        2,
        false,
    );
    check(
        "ElasticNet",
        true,
        regressor(ElasticNetRegressor::builder().lambda(0.1).build()),
        2,
        false,
    );
    check(
        "Gamma",
        true,
        regressor(GammaRegressor::builder().build()),
        2,
        false,
    );
    check(
        "LmDynamic",
        true,
        regressor(LmDynamicRegressor::builder().build()),
        2,
        false,
    );
    check(
        "NegativeBinomial",
        true,
        regressor(NegativeBinomialRegressor::builder().build()),
        2,
        false,
    );
    check(
        "Ols",
        true,
        regressor(OlsRegressor::builder().build()),
        2,
        false,
    );
    check(
        "Poisson",
        true,
        regressor(PoissonRegressor::builder().build()),
        2,
        false,
    );
    check(
        "Ridge",
        true,
        regressor(RidgeRegressor::builder().lambda(0.1).build()),
        2,
        false,
    );
    check(
        "Rls",
        true,
        regressor(RlsRegressor::builder().build()),
        2,
        false,
    );
    check(
        "Tweedie",
        true,
        regressor(TweedieRegressor::builder().build()),
        2,
        false,
    );
    check(
        "Wls",
        true,
        regressor(WlsRegressor::builder().build()),
        2,
        false,
    );
}

#[test]
fn estimators_without_na_action_reject_non_finite_input() {
    check(
        "Ard",
        false,
        regressor(ArdRegression::builder().build()),
        2,
        false,
    );
    check(
        "BayesianRidge",
        false,
        regressor(BayesianRidge::builder().build()),
        2,
        false,
    );
    check(
        "Huber",
        false,
        regressor(HuberRegressor::builder().build()),
        2,
        false,
    );
    check(
        "Isotonic",
        false,
        regressor(IsotonicRegressor::builder().build()),
        1,
        false,
    );
    check(
        "Lars",
        false,
        regressor(LarsRegressor::builder().build()),
        2,
        false,
    );
    check(
        "PassiveAggressive",
        false,
        regressor(PassiveAggressiveRegressor::builder().build()),
        2,
        false,
    );
    check(
        "Pls",
        false,
        regressor(PlsRegressor::builder().build()),
        2,
        false,
    );
    check(
        "PSpline",
        false,
        regressor(PSplineRegressor::builder().build()),
        1,
        false,
    );
    check(
        "Quantile",
        false,
        regressor(QuantileRegressor::builder().build()),
        2,
        false,
    );
    check(
        "Ransac",
        false,
        regressor(RansacRegressor::builder().build()),
        2,
        false,
    );
    check(
        "TheilSen",
        false,
        regressor(TheilSenRegressor::builder().build()),
        2,
        false,
    );
}

#[test]
fn other_fit_entry_points_reject_non_finite_input() {
    let ok = |n: usize| -> Outcome { Ok(n) };
    check(
        "Logistic",
        false,
        |x, y| {
            LogisticRegression::builder()
                .build()
                .fit(x, y)
                .map(|_| ok(y.nrows()))
                .map_err(|e| e.to_string())?
        },
        2,
        true,
    );
    check(
        "Isotonic fit_1d",
        false,
        |x, y| {
            let x = Col::from_fn(x.nrows(), |i| x[(i, 0)]);
            IsotonicRegressor::builder()
                .build()
                .fit_1d(&x, y)
                .map(|_| ok(y.nrows()))
                .map_err(|e| e.to_string())?
        },
        1,
        false,
    );
    let ordinal = |x: &Mat<f64>, y: &Col<f64>| {
        let y = Col::from_fn(y.nrows(), |i| {
            if y[i].is_finite() {
                (i % 3) as f64
            } else {
                y[i]
            }
        });
        OrdinalRegressor::builder()
            .build()
            .fit(x, &y)
            .map(|_| ok(y.nrows()))
            .map_err(|e| e.to_string())?
    };
    check("Ordinal", false, ordinal, 2, false);
    let cr = |x: &Mat<f64>, y: &Col<f64>| {
        let y = Col::from_fn(y.nrows(), |i| {
            if y[i].is_finite() {
                (i % 3) as f64
            } else {
                y[i]
            }
        });
        ContinuationRatioRegressor::builder()
            .build()
            .fit(x, &y)
            .map(|_| ok(y.nrows()))
            .map_err(|e| e.to_string())?
    };
    check("ContinuationRatio", false, cr, 2, false);
    let group = |n: usize| (0..n).map(|i| i % 5).collect::<Vec<_>>();
    check(
        "Glmm",
        false,
        |x, y| {
            GlmmRegressor::gaussian()
                .build()
                .fit(x, y, &group(x.nrows()))
                .map(|_| ok(y.nrows()))
                .map_err(|e| e.to_string())?
        },
        2,
        false,
    );
    check(
        "Gls",
        false,
        |x, y| {
            GlsRegressor::builder()
                .build()
                .fit(x, y, &group(x.nrows()), None)
                .map(|_| ok(y.nrows()))
                .map_err(|e| e.to_string())?
        },
        2,
        false,
    );
}

#[test]
fn nan_degrees_of_freedom_do_not_panic() {
    let t = Col::from_fn(2, |i| i as f64 + 1.0);
    assert!(CoefficientInference::p_values(&t, f64::NAN)
        .iter()
        .all(|p| p.is_nan()));
    let x = Mat::from_fn(2, 1, |i, _| i as f64);
    let p = compute_prediction_intervals(
        &x,
        &Mat::identity(2, 2),
        &Col::zeros(2),
        1.0,
        f64::NAN,
        0.95,
        anofox_regression::core::IntervalType::Confidence,
        true,
    );
    assert!(p.lower.iter().all(|v| v.is_nan()));
}

#[test]
fn hc_rejects_a_wrong_aliased_length() {
    let x = Mat::from_fn(10, 2, |i, j| (i * (j + 1)) as f64);
    let r = Col::from_fn(10, |i| (i as f64).sin());
    assert!(compute_hc_standard_errors(&x, &r, &[false], true, HcType::HC1).is_err());
}

/// Fitted models can be shared across threads (a compile-time check).
#[test]
fn fitted_models_are_send_and_sync() {
    fn send_sync<T: Send + Sync>() {}
    send_sync::<FittedAft>();
    send_sync::<FittedAlm>();
    send_sync::<FittedArd>();
    send_sync::<FittedBayesianRidge>();
    send_sync::<FittedBinomial>();
    send_sync::<FittedBls>();
    send_sync::<FittedContinuationRatio>();
    send_sync::<FittedElasticNet>();
    send_sync::<FittedGamma>();
    send_sync::<FittedGlmm>();
    send_sync::<FittedGls>();
    send_sync::<FittedHuber>();
    send_sync::<FittedIsotonic>();
    send_sync::<FittedLars>();
    send_sync::<FittedLmDynamic>();
    send_sync::<FittedLogistic>();
    send_sync::<FittedNegativeBinomial>();
    send_sync::<FittedOls>();
    send_sync::<FittedOrdinal>();
    send_sync::<FittedPassiveAggressive>();
    send_sync::<FittedPenalizedGlm>();
    send_sync::<FittedPls>();
    send_sync::<FittedPoisson>();
    send_sync::<FittedPSpline>();
    send_sync::<FittedQuantile>();
    send_sync::<FittedRansac>();
    send_sync::<FittedRidge>();
    send_sync::<FittedRls>();
    send_sync::<FittedTheilSen>();
    send_sync::<FittedTweedie>();
    send_sync::<FittedWls>();
}
