#![allow(clippy::excessive_precision, dead_code)]

//! 0.12 covariance options:
//!
//! * `FittedOls::hc_vcov` / `FittedWls::hc_vcov` against `sandwich::vcovHC`;
//! * `GlmResult::cluster_vcov` (the `fit_*` entry points) against
//!   `sandwich::vcovCL(glm)`, and with priors against `rms::robcov` on a
//!   penalized `lrm` and the penalized sandwich of a Gamma GLM;
//! * `AftResult::cluster_vcov` (`fit_aft`) against `survreg(cluster =)` and
//!   the penalized sandwich of a Weibull AFT.
//!
//! Reference values: `tests/r_scripts/generate_cluster_validation.R` (data,
//! unpenalized GLM / AFT) and `tests/r_scripts/generate_covariance_012_validation.R`.

use anofox_regression::inference::{ClusterError, ClusterOptions, ClusterType, HcType};
use anofox_regression::solvers::aft::{fit_aft, AftDistribution, AftOptions, AftRegressor};
use anofox_regression::solvers::penalized_glm::{
    fit_gamma, fit_logistic, fit_poisson, GammaOptions, GlmPriorOptions, LogisticOptions,
    PoissonOptions, PriorSpec, VcovType,
};
use anofox_regression::solvers::{OlsRegressor, Regressor, WlsRegressor};
use faer::{Col, Mat};

include!("fixtures/cluster_validation.rs");
include!("fixtures/covariance_012_validation.rs");

fn design() -> Mat<f64> {
    Mat::from_fn(X1.len(), 2, |i, j| if j == 0 { X1[i] } else { X2[i] })
}

fn columns() -> Vec<Vec<f64>> {
    vec![X1.to_vec(), X2.to_vec()]
}

fn col(v: &[f64]) -> Col<f64> {
    Col::from_fn(v.len(), |i| v[i])
}

fn assert_mat(name: &str, got: &Mat<f64>, want: &[f64], tol: f64) {
    let k = got.nrows();
    assert_eq!(k * k, want.len(), "{name}: size");
    let scale = want.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    for i in 0..k {
        for j in 0..k {
            let (a, b) = (got[(i, j)], want[i * k + j]);
            assert!(
                (a - b).abs() <= tol * scale,
                "{name}[{i},{j}]: {a} vs R {b}"
            );
        }
    }
}

fn assert_vec(name: &str, got: &[f64], want: &[f64], tol: f64) {
    for (j, (a, b)) in got.iter().zip(want).enumerate() {
        assert!(
            (a - b).abs() <= tol * (1.0 + b.abs()),
            "{name}[{j}]: {a} vs R {b}"
        );
    }
}

#[test]
fn ols_and_wls_hc_vcov_match_vcovhc() {
    let x = design();
    let ols = OlsRegressor::builder().build().fit(&x, &col(&Y)).unwrap();
    let wls = WlsRegressor::builder()
        .weights(col(&W))
        .build()
        .fit(&x, &col(&Y))
        .unwrap();
    let cases: [(HcType, &[f64], &[f64]); 4] = [
        (HcType::HC0, &HC_OLS_HC0, &HC_WLS_HC0),
        (HcType::HC1, &HC_OLS_HC1, &HC_WLS_HC1),
        (HcType::HC2, &HC_OLS_HC2, &HC_WLS_HC2),
        (HcType::HC3, &HC_OLS_HC3, &HC_WLS_HC3),
    ];
    for (kind, want_ols, want_wls) in cases {
        let v = ols.hc_vcov(&x, kind).unwrap();
        assert_mat(&format!("OLS {kind:?}"), &v, want_ols, 1e-10);
        // the HC standard errors are the square roots of its diagonal
        let se = ols.hc_standard_errors(&x, kind).unwrap();
        assert!((se.intercept_std_error.unwrap() - v[(0, 0)].sqrt()).abs() < 1e-14);
        assert!((se.std_errors[1] - v[(2, 2)].sqrt()).abs() < 1e-14);
        let v = wls.hc_vcov(&x, kind).unwrap();
        assert_mat(&format!("WLS {kind:?}"), &v, want_wls, 1e-10);
    }
}

#[test]
fn hc_vcov_puts_zero_rows_for_aliased_columns() {
    let n = X1.len();
    let x = Mat::from_fn(n, 3, |i, j| match j {
        0 => X1[i],
        1 => X2[i],
        _ => 2.0 * X1[i],
    });
    let fit = OlsRegressor::builder().build().fit(&x, &col(&Y)).unwrap();
    let v = fit.hc_vcov(&x, HcType::HC3).unwrap();
    assert_eq!(v.nrows(), 4);
    assert!((0..4).all(|j| v[(3, j)] == 0.0 && v[(j, 3)] == 0.0));
    let sub = Mat::from_fn(3, 3, |a, b| v[(a, b)]);
    assert_mat("aliased HC3", &sub, &HC_OLS_HC3, 1e-10);
    assert!(fit.hc_vcov(&design(), HcType::HC0).is_err());
}

#[test]
fn glm_entry_points_match_vcovcl() {
    let x = columns();
    let inference = |o: PoissonOptions| o.compute_inference(true).tolerance(1e-12);
    let pois = fit_poisson(&YC, &x, &inference(PoissonOptions::default())).unwrap();
    let v = pois
        .cluster_vcov(&YC, &x, &[&G1], &ClusterOptions::hc0())
        .unwrap();
    assert_mat("poisson HC0", &v, &POIS_HC0, 1e-8);
    let v = pois
        .cluster_vcov(&YC, &x, &[&G1], &ClusterOptions::cr1())
        .unwrap();
    assert_mat("poisson HC1", &v, &POIS_HC1, 1e-8);
    let v = pois
        .cluster_vcov(&YC, &x, &[&G1, &G2], &ClusterOptions::hc0())
        .unwrap();
    assert_mat("poisson two-way", &v, &POIS_TWOWAY, 1e-8);

    let logit = fit_logistic(
        &YB,
        &x,
        &LogisticOptions::default()
            .compute_inference(true)
            .tolerance(1e-12),
    )
    .unwrap();
    let v = logit
        .fit
        .cluster_vcov(&YB, &x, &[&G1], &ClusterOptions::hc0())
        .unwrap();
    assert_mat("logit HC0", &v, &LOGIT_HC0, 1e-8);

    // Gamma: the Pearson dispersion cancels.
    let gamma = fit_gamma(
        &YG,
        &x,
        &GammaOptions::default()
            .compute_inference(true)
            .tolerance(1e-12),
    )
    .unwrap();
    let v = gamma
        .cluster_vcov(&YG, &x, &[&G1], &ClusterOptions::hc0())
        .unwrap();
    assert_mat("gamma HC0", &v, &GAMMA_HC0, 1e-7);
    let v = gamma
        .cluster_vcov(&YG, &x, &[&G1], &ClusterOptions::cr1())
        .unwrap();
    assert_mat("gamma HC1", &v, &GAMMA_HC1, 1e-7);
}

#[test]
fn glm_cluster_vcov_drops_the_rows_the_fit_dropped() {
    // A NaN response row is dropped by the fit and by the covariance; its
    // cluster code is ignored.
    let mut y = YC.to_vec();
    let mut x = columns();
    let mut g = G1.to_vec();
    y.push(f64::NAN);
    x[0].push(1.0);
    x[1].push(0.5);
    g.push(99);
    let opts = PoissonOptions::default()
        .compute_inference(true)
        .tolerance(1e-12);
    let fit = fit_poisson(&y, &x, &opts).unwrap();
    let v = fit
        .cluster_vcov(&y, &x, &[&g], &ClusterOptions::hc0())
        .unwrap();
    assert_mat("poisson with a dropped row", &v, &POIS_HC0, 1e-8);
}

fn logit_priors(vcov: VcovType) -> LogisticOptions {
    LogisticOptions::default()
        .compute_inference(true)
        .tolerance(1e-13)
        .prior_opts(
            GlmPriorOptions::default()
                .priors(vec![
                    PriorSpec::flat(),
                    PriorSpec::normal(0.0, 0.5),
                    PriorSpec::normal(0.0, 0.3),
                ])
                .vcov(vcov),
        )
}

#[test]
fn penalized_logistic_matches_robcov_of_penalized_lrm() {
    let x = columns();
    for vcov in [VcovType::Laplace, VcovType::Sandwich, VcovType::Naive] {
        let fit = fit_logistic(&YB, &x, &logit_priors(vcov)).unwrap().fit;
        let mut coef = vec![fit.core.intercept.unwrap()];
        coef.extend(&fit.core.coefficients);
        assert_vec("penalized logit coef", &coef, &PEN_LOGIT_COEF, 1e-8);
        if vcov == VcovType::Laplace {
            let v = fit.inference.as_ref().unwrap().vcov.clone().unwrap();
            assert_mat("penalized logit vcov", &v, &PEN_LOGIT_VCOV, 1e-8);
        }
        // The bread is the penalized information whatever vcov the fit reports.
        let v = fit
            .cluster_vcov(&YB, &x, &[&G1], &ClusterOptions::robcov())
            .unwrap();
        assert_mat(
            &format!("penalized logit robcov ({vcov:?})"),
            &v,
            &PEN_LOGIT_ROBCOV,
            1e-8,
        );
    }
}

#[test]
fn penalized_gamma_uses_the_penalized_bread() {
    let x = columns();
    let opts = GammaOptions::default()
        .compute_inference(true)
        .tolerance(1e-13)
        .prior_opts(GlmPriorOptions::default().priors(vec![
            PriorSpec::normal(0.0, 0.2),
            PriorSpec::normal(0.0, 0.1),
        ]));
    let fit = fit_gamma(&YG, &x, &opts).unwrap();
    let mut coef = vec![fit.core.intercept.unwrap()];
    coef.extend(&fit.core.coefficients);
    assert_vec("penalized gamma coef", &coef, &PEN_GAMMA_COEF, 1e-8);
    let v = fit
        .cluster_vcov(&YG, &x, &[&G1], &ClusterOptions::hc0())
        .unwrap();
    assert_mat("penalized gamma HC0", &v, &PEN_GAMMA_HC0, 1e-7);
}

#[test]
fn glm_cluster_vcov_rejects_bad_requests() {
    let x = columns();
    let no_inf = fit_poisson(&YC, &x, &PoissonOptions::default()).unwrap();
    assert!(matches!(
        no_inf.cluster_vcov(&YC, &x, &[&G1], &ClusterOptions::hc0()),
        Err(ClusterError::Unsupported(_))
    ));
    let fit = fit_poisson(&YC, &x, &PoissonOptions::default().compute_inference(true)).unwrap();
    let hc2 = ClusterOptions::hc0().kind(ClusterType::HC2);
    assert!(fit.cluster_vcov(&YC, &x, &[&G1], &hc2).is_err());
    assert!(matches!(
        fit.cluster_vcov(&YC, &x, &[], &ClusterOptions::hc0()),
        Err(ClusterError::NoClusters)
    ));
    assert!(fit
        .cluster_vcov(&YC, &x, &[&G1[..10]], &ClusterOptions::hc0())
        .is_err());
    assert!(fit
        .cluster_vcov(&YC, &x[..1], &[&G1], &ClusterOptions::hc0())
        .is_err());
}

fn aft_options() -> AftOptions {
    AftOptions::default()
        .dist(AftDistribution::Weibull)
        .compute_inference(true)
        .tolerance(1e-12)
}

#[test]
fn fit_aft_cluster_vcov_matches_survreg() {
    let x = columns();
    let fit = fit_aft(&TIME, &x, &EV, &aft_options()).unwrap();
    let v = fit
        .cluster_vcov(&TIME, &x, &EV, &[&G1], &ClusterOptions::robcov())
        .unwrap();
    assert_mat("fit_aft robust", &v, &AFT_ROBUST, 1e-5);
    let scores = fit.score_contributions(&TIME, &x, &EV).unwrap();
    assert_eq!((scores.nrows(), scores.ncols()), (TIME.len(), 4));
    // the scores sum to (nearly: the fit stops on the log-likelihood change)
    // zero at the MLE
    for j in 0..4 {
        let s: f64 = (0..scores.nrows()).map(|i| scores[(i, j)]).sum();
        assert!(s.abs() < 1e-3, "score sum {j}: {s}");
    }
    // a row with a missing time is skipped, as in the fit
    let (mut t, mut e, mut xx, mut g) = (TIME.to_vec(), EV.to_vec(), x.clone(), G1.to_vec());
    t.push(f64::NAN);
    e.push(1.0);
    xx[0].push(0.0);
    xx[1].push(0.0);
    g.push(77);
    let fit2 = fit_aft(&t, &xx, &e, &aft_options()).unwrap();
    let v2 = fit2
        .cluster_vcov(&t, &xx, &e, &[&g], &ClusterOptions::robcov())
        .unwrap();
    assert_mat("fit_aft robust, dropped row", &v2, &AFT_ROBUST, 1e-5);
    assert!(fit
        .cluster_vcov(&TIME, &x[..1], &EV, &[&G1], &ClusterOptions::robcov())
        .is_err());
    let no_inf = fit_aft(&TIME, &x, &EV, &aft_options().compute_inference(false)).unwrap();
    assert!(no_inf
        .cluster_vcov(&TIME, &x, &EV, &[&G1], &ClusterOptions::robcov())
        .is_err());
}

#[test]
fn penalized_aft_cluster_vcov_uses_the_penalized_bread() {
    let x = columns();
    let priors = vec![PriorSpec::normal(0.0, 0.1), PriorSpec::normal(0.0, 0.2)];
    for vcov in [VcovType::Laplace, VcovType::Sandwich] {
        let opts = aft_options().priors(priors.clone()).vcov(vcov);
        let fit = fit_aft(&TIME, &x, &EV, &opts).unwrap();
        let mut coef = vec![fit.core.intercept.unwrap()];
        coef.extend(&fit.core.coefficients);
        coef.push(fit.core.scale.ln());
        assert_vec("penalized AFT coef", &coef, &PEN_AFT_COEF, 1e-7);
        let v = fit
            .cluster_vcov(&TIME, &x, &EV, &[&G1], &ClusterOptions::robcov())
            .unwrap();
        assert_mat(
            &format!("penalized AFT ({vcov:?})"),
            &v,
            &PEN_AFT_ROBUST,
            1e-6,
        );

        // FittedAft uses the same bread
        let event: Vec<bool> = EV.iter().map(|&e| e == 1.0).collect();
        let fitted = AftRegressor::builder()
            .distribution(AftDistribution::Weibull)
            .compute_inference(true)
            .tolerance(1e-12)
            .priors(priors.clone())
            .vcov_type(vcov)
            .build()
            .fit(&design(), &col(&TIME), &event)
            .unwrap();
        let v = fitted
            .cluster_vcov(
                &design(),
                &col(&TIME),
                &event,
                &[&G1],
                &ClusterOptions::robcov(),
            )
            .unwrap();
        assert_mat(
            &format!("FittedAft penalized ({vcov:?})"),
            &v,
            &PEN_AFT_ROBUST,
            1e-6,
        );
    }
}

#[test]
fn a_coefficient_at_an_l1_kink_has_no_sandwich() {
    let x = columns();
    let opts = PoissonOptions::default()
        .compute_inference(true)
        .prior_opts(
            GlmPriorOptions::default()
                .priors(vec![PriorSpec::flat(), PriorSpec::laplace(0.0, 1e-4)]),
        );
    let fit = fit_poisson(&YC, &x, &opts).unwrap();
    assert_eq!(fit.core.coefficients[1], 0.0);
    assert!(matches!(
        fit.cluster_vcov(&YC, &x, &[&G1], &ClusterOptions::hc0()),
        Err(ClusterError::Unsupported(_))
    ));
}

#[test]
fn fitted_aft_from_a_fit_aft_result_matches_the_regressor() {
    let x = columns();
    let r = fit_aft(&TIME, &x, &EV, &aft_options()).unwrap();
    let from = anofox_regression::solvers::aft::FittedAft::from_result(r).unwrap();
    let event: Vec<bool> = EV.iter().map(|&e| e == 1.0).collect();
    let fitted = AftRegressor::builder()
        .distribution(AftDistribution::Weibull)
        .compute_inference(true)
        .tolerance(1e-12)
        .build()
        .fit(&design(), &col(&TIME), &event)
        .unwrap();
    let d = design();
    assert_eq!(from.predict_median(&d), fitted.predict_median(&d));
    assert_eq!(from.scale(), fitted.scale());
    let a = from
        .cluster_vcov(&d, &col(&TIME), &event, &[&G1], &ClusterOptions::robcov())
        .unwrap();
    assert_mat("from_result robust", &a, &AFT_ROBUST, 1e-5);
}
