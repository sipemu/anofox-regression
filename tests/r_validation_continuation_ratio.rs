#![allow(clippy::excessive_precision)]

//! #110: the continuation-ratio model (RMS §13.4) against `rms::cr.setup` +
//! `lrm`. Reference values come from
//! `tests/r_scripts/generate_continuation_ratio_validation.R`.

use anofox_regression::solvers::{cr_setup, ContinuationRatioRegressor, FittedRegressor};
use faer::{Col, Mat};

mod fixture {
    include!("fixtures/continuation_ratio_validation.rs");
}
use fixture::*;

const N: usize = 150;

fn close(name: &str, a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() <= tol * (1.0 + b.abs()), "{name}: {a} vs {b}");
}

#[test]
fn cr_setup_matches_rms() {
    let s = cr_setup(&Col::from_fn(SMALL_Y.len(), |i| SMALL_Y[i])).unwrap();
    assert_eq!(s.levels, vec![1.0, 2.0, 3.0, 4.0]);
    let as_usize = |v: &[f64]| v.iter().map(|&x| x as usize).collect::<Vec<_>>();
    assert_eq!(s.y.iter().copied().collect::<Vec<_>>(), SMALL_CR_Y.to_vec());
    assert_eq!(s.cohort, as_usize(&SMALL_COHORT));
    assert_eq!(s.subs, as_usize(&SMALL_SUBS));
    assert_eq!(s.reps, as_usize(&SMALL_REPS));
}

#[test]
fn continuation_ratio_matches_lrm_on_cr_setup() {
    let x = Mat::from_fn(N, 3, |i, j| X[i * 3 + j]);
    let y = Col::from_fn(N, |i| Y_CAT[i]);
    let xnew = Mat::from_fn(3, 3, |i, j| XNEW[i * 3 + j]);
    let cases: [(&str, &[usize], &[f64], &[f64], &[f64], &[f64]); 2] = [
        (
            "equal slopes",
            &[],
            &EQUAL_COEF,
            &EQUAL_VCOV,
            &EQUAL_LL,
            &EQUAL_COND,
        ),
        (
            "cohort-specific x1",
            &[0],
            &X1_COEF,
            &X1_VCOV,
            &X1_LL,
            &X1_COND,
        ),
    ];
    for (id, specific, coef, vcov, ll, cond) in cases {
        let f = ContinuationRatioRegressor::builder()
            .cohort_specific(specific)
            .build()
            .fit(&x, &y)
            .unwrap();
        let c = f.coefficients();
        assert_eq!(c.nrows(), coef.len(), "{id}");
        let d = coef.len();
        let v = f.vcov();
        for a in 0..d {
            close(&format!("{id} coef {a}"), c[a], coef[a], 1e-7);
            for b in 0..d {
                close(
                    &format!("{id} vcov {a},{b}"),
                    v[(a, b)],
                    vcov[a * d + b],
                    1e-6,
                );
            }
        }
        close(&format!("{id} ll"), f.log_likelihood(), ll[0], 1e-9);
        let h = f.conditional_probabilities(&xnew);
        let p = f.category_probabilities(&xnew);
        for i in 0..3 {
            for j in 0..4 {
                close(
                    &format!("{id} P(stop {j} | reach) row {i}"),
                    h[(i, j)],
                    cond[i * 4 + j],
                    1e-7,
                );
            }
            let total: f64 = (0..5).map(|j| p[(i, j)]).sum();
            close("probabilities sum to 1", total, 1.0, 1e-12);
            close("P(Y = y1) = h1", p[(i, 0)], h[(i, 0)], 1e-15);
        }
    }
}

#[test]
fn continuation_ratio_rejects_bad_input() {
    let x = Mat::from_fn(N, 3, |i, j| X[i * 3 + j]);
    let y = Col::from_fn(N, |i| Y_CAT[i]);
    let two = Col::from_fn(N, |i| (Y_CAT[i] > 2.0) as u8 as f64);
    assert!(ContinuationRatioRegressor::builder()
        .build()
        .fit(&x, &two)
        .is_err());
    let short = Col::from_fn(10, |i| Y_CAT[i]);
    assert!(ContinuationRatioRegressor::builder()
        .build()
        .fit(&x, &short)
        .is_err());
    assert!(ContinuationRatioRegressor::builder()
        .cohort_specific(&[5])
        .build()
        .fit(&x, &y)
        .is_err());
    let mut y_nan = y.clone();
    y_nan[0] = f64::NAN;
    assert!(cr_setup(&y_nan).is_err());
    let f = ContinuationRatioRegressor::builder()
        .build()
        .fit(&x, &y)
        .unwrap();
    assert_eq!(f.levels(), &[1.0, 2.0, 3.0, 4.0, 5.0]);
    assert_eq!(f.n_features(), 3);
    assert!(f.logistic().result().intercept.is_some());
}

#[test]
fn case_weights_match_weighted_lrm_and_glm() {
    use anofox_regression::solvers::{BinomialRegressor, Regressor};
    let x = Mat::from_fn(N, 3, |i, j| X[i * 3 + j]);
    let y = Col::from_fn(N, |i| Y_CAT[i]);
    let f = ContinuationRatioRegressor::builder()
        .weights(&W)
        .build()
        .fit(&x, &y)
        .unwrap();
    let k = WEQUAL_COEF.len();
    let coef = f.coefficients();
    let v = f.vcov();
    for a in 0..k {
        close(&format!("coef {a}"), coef[a], WEQUAL_COEF[a], 1e-9);
        assert!((coef[a] - GLMW_COEF[a]).abs() < 1e-9);
        for b in 0..k {
            close(
                &format!("vcov {a},{b}"),
                v[(a, b)],
                WEQUAL_VCOV[b * k + a],
                1e-8,
            );
            close(
                &format!("glm vcov {a},{b}"),
                v[(a, b)],
                GLMW_VCOV[b * k + a],
                1e-8,
            );
        }
    }
    close("loglik", f.log_likelihood(), WEQUAL_LL[0], 1e-10);
    close("deviance", f.logistic().deviance, GLMW_DEV[0], 1e-10);
    close(
        "null deviance",
        f.logistic().null_deviance,
        GLMW_DEV[1],
        1e-10,
    );

    // weights of the wrong length, and negative weights
    assert!(ContinuationRatioRegressor::builder()
        .weights(&W[..10])
        .build()
        .fit(&x, &y)
        .is_err());
    let neg = BinomialRegressor::logistic()
        .weights(Col::from_fn(N, |i| if i == 0 { -1.0 } else { 1.0 }))
        .build()
        .fit(
            &x,
            &Col::from_fn(N, |i| f64::from(u8::from(Y_CAT[i] > 2.0))),
        );
    assert!(neg.is_err());
}
