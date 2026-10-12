//! R validation of quantile-regression inference
//! (`FittedQuantile::inference`, `quantile_inference::quantile_process`)
//! against `quantreg::summary.rq(se = "iid" | "nid")` and `rq(tau = 1:9/10)`;
//! the xy-pair bootstrap gets a statistical sanity check.
//!
//! Reference values: `tests/r_scripts/generate_quantile_inference_validation.R`.

#[path = "fixtures/quantile_inference_validation.rs"]
mod refs;

use anofox_regression::solvers::quantile_inference::{
    quantile_inference, quantile_process, QuantileSe,
};
use anofox_regression::solvers::{QuantileRegressor, Regressor};
use faer::{Col, Mat};
use refs::*;

fn data() -> (Mat<f64>, Col<f64>) {
    (
        Mat::from_fn(N, 2, |i, j| if j == 0 { X1[i] } else { X2[i] }),
        Col::from_fn(N, |i| Y[i]),
    )
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .fold(0.0_f64, |m, (u, v)| m.max((u - v).abs()))
}

#[test]
fn iid_and_nid_match_summary_rq() {
    let (x, y) = data();
    let refs: [[(&[f64; 3], &[f64; 3], &[f64; 3], &[f64; 3]); 2]; 3] = [
        [
            (&T1_IID_COEF, &T1_IID_SE, &T1_IID_T, &T1_IID_P),
            (&T1_NID_COEF, &T1_NID_SE, &T1_NID_T, &T1_NID_P),
        ],
        [
            (&T2_IID_COEF, &T2_IID_SE, &T2_IID_T, &T2_IID_P),
            (&T2_NID_COEF, &T2_NID_SE, &T2_NID_T, &T2_NID_P),
        ],
        [
            (&T3_IID_COEF, &T3_IID_SE, &T3_IID_T, &T3_IID_P),
            (&T3_NID_COEF, &T3_NID_SE, &T3_NID_T, &T3_NID_P),
        ],
    ];
    for (k, &tau) in TAUS.iter().enumerate() {
        let fit = QuantileRegressor::builder()
            .tau(tau)
            .build()
            .fit(&x, &y)
            .unwrap();
        for (s, se) in [QuantileSe::Iid, QuantileSe::Nid].into_iter().enumerate() {
            let (coef, sev, t, p) = refs[k][s];
            let inf = fit.inference(&x, &y, se, 0.95).unwrap();
            let dc = max_abs_diff(&inf.estimate, coef);
            let ds = max_abs_diff(&inf.std_error, sev);
            let dt = max_abs_diff(&inf.statistic, t);
            let dp = max_abs_diff(&inf.p_value, p);
            eprintln!("tau={tau} {se:?}: max|Δcoef| = {dc:.2e}, max|Δse| = {ds:.2e}, max|Δt| = {dt:.2e}, max|Δp| = {dp:.2e}");
            assert!(dc < 1e-8, "coef {dc}");
            assert!(ds < 1e-8, "se {ds}");
            assert!(dt < 1e-6, "t {dt}");
            assert!(dp < 1e-8, "p {dp}");
            // CI is estimate ± t_{n-k} se
            for j in 0..3 {
                assert!(inf.conf_low[j] < inf.estimate[j] && inf.estimate[j] < inf.conf_high[j]);
            }
        }
    }
}

#[test]
fn process_matches_rq_taus() {
    let (x, y) = data();
    let proc_ = quantile_process(&x, &y, &PROCESS_TAUS, true, Some(QuantileSe::Nid), 0.9).unwrap();
    let est: Vec<f64> = proc_.iter().flat_map(|r| r.estimate.clone()).collect();
    let d = max_abs_diff(&est, &PROCESS_COEF);
    eprintln!("process (9 taus): max|Δcoef| = {d:.2e}");
    assert!(d < 1e-8, "{d}");
    assert!(proc_
        .iter()
        .all(|r| r.std_error.iter().all(|s| s.is_finite() && *s > 0.0)));
    let no_se = quantile_process(&x, &y, &[0.5], true, None, 0.95).unwrap();
    assert!(no_se[0].std_error.iter().all(|s| s.is_nan()));
}

#[test]
fn bootstrap_is_seeded_and_close_to_r() {
    let (x, y) = data();
    let a = quantile_inference(
        &x,
        &y,
        0.5,
        true,
        QuantileSe::Boot { seed: 7, r: 400 },
        0.95,
    )
    .unwrap();
    let b = quantile_inference(
        &x,
        &y,
        0.5,
        true,
        QuantileSe::Boot { seed: 7, r: 400 },
        0.95,
    )
    .unwrap();
    assert_eq!(a.std_error, b.std_error);
    assert_eq!(a.n_boot, 400);
    // Different RNG than R: compare to R's R = 4000 bootstrap SE within 20 %.
    for j in 0..3 {
        let rel = (a.std_error[j] / BOOT_MEDIAN_SE[j] - 1.0).abs();
        eprintln!(
            "boot se[{j}] = {:.4} vs R {:.4} (rel {rel:.3})",
            a.std_error[j], BOOT_MEDIAN_SE[j]
        );
        assert!(rel < 0.2, "boot se {j}: rel diff {rel}");
    }
}

#[test]
fn weighted_fit_is_rejected() {
    let (x, y) = data();
    let fit = QuantileRegressor::builder()
        .tau(0.5)
        .weights(Col::from_fn(N, |_| 1.0))
        .build()
        .fit(&x, &y)
        .unwrap();
    assert!(fit.inference(&x, &y, QuantileSe::Iid, 0.95).is_err());
}
