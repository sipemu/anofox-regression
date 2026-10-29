#![allow(clippy::excessive_precision, clippy::type_complexity, dead_code)]

//! #105: cluster-robust covariance against `sandwich::vcovCL`,
//! `rms::robcov` and `survreg(cluster =)`.
//!
//! Reference values come from `tests/r_scripts/generate_cluster_validation.R`.

use anofox_regression::core::{BinomialFamily, PoissonFamily, TweedieFamily};
use anofox_regression::inference::{glm_cluster_vcov, ClusterOptions, ClusterType};
use anofox_regression::solvers::aft::{AftDistribution, AftRegressor};
use anofox_regression::solvers::{OlsRegressor, Regressor, WlsRegressor};
use faer::{Col, Mat};

include!("fixtures/cluster_validation.rs");

fn design() -> Mat<f64> {
    Mat::from_fn(X1.len(), 2, |i, j| if j == 0 { X1[i] } else { X2[i] })
}

fn col(v: &[f64]) -> Col<f64> {
    Col::from_fn(v.len(), |i| v[i])
}

fn assert_mat(name: &str, got: &Mat<f64>, want: &[f64], tol: f64) {
    let k = got.nrows();
    assert_eq!(k * k, want.len(), "{name}: size");
    for i in 0..k {
        for j in 0..k {
            let (a, b) = (got[(i, j)], want[i * k + j]);
            assert!(
                (a - b).abs() <= tol * (1e-6 + b.abs()),
                "{name}[{i},{j}]: {a} vs {b}"
            );
        }
    }
}

fn opts(kind: ClusterType, cadjust: bool) -> ClusterOptions {
    ClusterOptions { kind, cadjust }
}

const TYPES: [ClusterType; 4] = [
    ClusterType::HC0,
    ClusterType::HC1,
    ClusterType::HC2,
    ClusterType::HC3,
];

#[test]
fn ols_matches_vcovcl_for_every_type() {
    let x = design();
    let fit = OlsRegressor::builder().build().fit(&x, &col(&Y)).unwrap();
    let want = [&LM_HC0, &LM_HC1, &LM_HC2, &LM_HC3];
    let want_noadj = [&LM_HC0_NOADJ, &LM_HC1_NOADJ, &LM_HC2_NOADJ, &LM_HC3_NOADJ];
    for (k, t) in TYPES.into_iter().enumerate() {
        let v = fit.cluster_vcov(&x, &[&G1], &opts(t, true)).unwrap();
        assert_mat(&format!("{t:?}"), &v, want[k], 1e-10);
        let v = fit.cluster_vcov(&x, &[&G1], &opts(t, false)).unwrap();
        assert_mat(&format!("{t:?} noadj"), &v, want_noadj[k], 1e-10);
    }
    assert_eq!(ClusterOptions::default(), opts(ClusterType::HC1, true));
}

#[test]
fn two_way_clustering_matches_vcovcl() {
    let x = design();
    let fit = OlsRegressor::builder().build().fit(&x, &col(&Y)).unwrap();
    let v = fit
        .cluster_vcov(&x, &[&G1, &G2], &ClusterOptions::cr1())
        .unwrap();
    assert_mat("two-way HC1", &v, &LM_TWOWAY_HC1, 1e-10);
    let v = fit
        .cluster_vcov(&x, &[&G1, &G2], &ClusterOptions::hc0())
        .unwrap();
    assert_mat("two-way HC0", &v, &LM_TWOWAY_HC0, 1e-10);
}

#[test]
fn wls_and_no_intercept_match_vcovcl() {
    let x = design();
    let fit = WlsRegressor::builder()
        .weights(col(&W))
        .build()
        .fit(&x, &col(&Y))
        .unwrap();
    let want = [&WLS_HC0, &WLS_HC1, &WLS_HC2, &WLS_HC3];
    for (k, t) in TYPES.into_iter().enumerate() {
        let v = fit.cluster_vcov(&x, &[&G1], &opts(t, true)).unwrap();
        assert_mat(&format!("WLS {t:?}"), &v, want[k], 1e-10);
    }
    let fit0 = OlsRegressor::builder()
        .with_intercept(false)
        .build()
        .fit(&x, &col(&Y))
        .unwrap();
    let v = fit0
        .cluster_vcov(&x, &[&G1], &ClusterOptions::cr1())
        .unwrap();
    assert_mat("no intercept", &v, &LM0_HC1, 1e-10);
}

#[test]
fn robcov_is_hc0_without_adjustment() {
    let x = design();
    let fit = OlsRegressor::builder().build().fit(&x, &col(&Y)).unwrap();
    let v = fit
        .cluster_vcov(&x, &[&G1], &ClusterOptions::robcov())
        .unwrap();
    assert_mat("robcov", &v, &ROBCOV, 1e-10);
}

/// R's `glm` keeps the working weights of its last IRLS iterate, which for a
/// non-canonical link (probit) lags the final means by ~1e-7.
const GLM_TOL: f64 = 1e-6;

#[test]
fn glms_match_vcovcl() {
    let x = design();
    let logit = BinomialFamily::logistic();
    let probit = BinomialFamily::probit();
    let pois = PoissonFamily::log();
    let gamma = TweedieFamily::gamma();
    let cases: [(
        &str,
        &dyn anofox_regression::core::GlmFamily,
        &[f64],
        &[f64],
        &[f64],
        &[f64],
    ); 4] = [
        ("poisson", &pois, &YC, &POIS_MU, &POIS_HC0, &POIS_HC1),
        ("logit", &logit, &YB, &LOGIT_MU, &LOGIT_HC0, &LOGIT_HC1),
        ("probit", &probit, &YB, &PROBIT_MU, &PROBIT_HC0, &PROBIT_HC1),
        ("gamma", &gamma, &YG, &GAMMA_MU, &GAMMA_HC0, &GAMMA_HC1),
    ];
    for (name, fam, y, mu, hc0, hc1) in cases {
        let v =
            glm_cluster_vcov(fam, &x, true, y, mu, None, &[&G1], &ClusterOptions::hc0()).unwrap();
        assert_mat(&format!("{name} HC0"), &v, hc0, GLM_TOL);
        let v =
            glm_cluster_vcov(fam, &x, true, y, mu, None, &[&G1], &ClusterOptions::cr1()).unwrap();
        assert_mat(&format!("{name} HC1"), &v, hc1, GLM_TOL);
    }
    let v = glm_cluster_vcov(
        &pois,
        &x,
        true,
        &YC,
        &POIS_MU,
        None,
        &[&G1, &G2],
        &ClusterOptions::hc0(),
    )
    .unwrap();
    assert_mat("poisson two-way", &v, &POIS_TWOWAY, 1e-8);
    let v = glm_cluster_vcov(
        &pois,
        &x,
        true,
        &YC,
        &POISW_MU,
        Some(&W),
        &[&G1],
        &ClusterOptions::hc0(),
    )
    .unwrap();
    assert_mat("poisson weighted", &v, &POISW_HC0, 1e-8);
}

#[test]
fn aft_matches_survreg_cluster() {
    let x = design();
    let event: Vec<bool> = EV.iter().map(|&e| e == 1.0).collect();
    let fit = AftRegressor::builder()
        .distribution(AftDistribution::Weibull)
        .compute_inference(true)
        .build()
        .fit(&x, &col(&TIME), &event)
        .unwrap();
    let mut coef = vec![
        fit.intercept().unwrap(),
        fit.coefficients()[0],
        fit.coefficients()[1],
    ];
    coef.push(fit.scale().ln());
    for (j, (a, b)) in coef.iter().zip(&AFT_COEF).enumerate() {
        assert!(
            (a - b).abs() < 1e-6 * (1.0 + b.abs()),
            "coef {j}: {a} vs {b}"
        );
    }
    let naive = fit.inference().unwrap().vcov.clone().unwrap();
    assert_mat("naive", &naive, &AFT_NAIVE, 1e-5);
    let v = fit
        .cluster_vcov(&x, &col(&TIME), &event, &[&G1], &ClusterOptions::robcov())
        .unwrap();
    assert_mat("robust", &v, &AFT_ROBUST, 1e-5);
}

#[test]
fn fit_without_residuals_is_refused() {
    let x = design();
    let s = anofox_regression::solvers::SufficientStats::from_data(&x, &col(&Y), None).unwrap();
    let fit = OlsRegressor::builder().build().fit_from_state(&s).unwrap();
    assert!(fit
        .cluster_vcov(&x, &[&G1], &ClusterOptions::cr1())
        .is_err());
}
