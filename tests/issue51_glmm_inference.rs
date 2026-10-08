//! Issue #51: Wald inference for GLMM fixed effects — z = β̂/SE,
//! two-sided normal p-values and Wald confidence intervals — against lme4.
//!
//! Reference values come from `tests/r_scripts/generate_glmm_inference_validation.R`
//! (R 4.6.1, lme4 2.0.1), which reads the scenario data from
//! `tests/fixtures/glmm_validation.rs`:
//! `coef(summary(m))`, `2 * pnorm(-abs(z))` and
//! `confint(m, parm = "beta_", method = "Wald", level = 0.95 / 0.90)`.

#![allow(dead_code)]
#![allow(clippy::approx_constant)]
#![allow(clippy::excessive_precision)]

use anofox_regression::solvers::{FittedGlmm, GlmmRegressor};
use faer::{Col, Mat};

include!("fixtures/glmm_validation.rs");

const LMM_A_Z: [f64; 2] = [4.451695054970e+00, 9.554977878286e+00];
const LMM_A_P: [f64; 2] = [8.519509552922e-06, 1.236103260560e-21];
const LMM_A_CI95_LO: [f64; 2] = [5.704827690382e-01, 3.776760859016e-01];
const LMM_A_CI95_HI: [f64; 2] = [1.467951757072e+00, 5.726017401650e-01];
const LMM_A_CI90_LO: [f64; 2] = [6.426274043414e-01, 3.933455301114e-01];
const LMM_A_CI90_HI: [f64; 2] = [1.395807121769e+00, 5.569322959551e-01];
const POIS_C_Z: [f64; 2] = [-2.957854950925e+00, 1.320888650099e+00];
const POIS_C_P: [f64; 2] = [3.097878580377e-03, 1.865384947364e-01];
const POIS_C_CI95_LO: [f64; 2] = [-1.223934653084e+00, -9.936421971247e-02];
const POIS_C_CI95_HI: [f64; 2] = [-2.483526436791e-01, 5.101110064237e-01];
const POIS_C_CI90_LO: [f64; 2] = [-1.145510764108e+00, -5.037047340794e-02];
const POIS_C_CI90_HI: [f64; 2] = [-3.267765326549e-01, 4.611172601192e-01];
const BIN_D_Z: [f64; 2] = [-2.016711503189e+00, 2.758992792362e+00];
const BIN_D_P: [f64; 2] = [4.372562878786e-02, 5.797981374722e-03];
const BIN_D_CI95_LO: [f64; 2] = [-1.823455297529e+00, 1.934433533612e-01];
const BIN_D_CI95_HI: [f64; 2] = [-2.602087191227e-02, 1.142450452792e+00];
const BIN_D_CI90_LO: [f64; 2] = [-1.678965345401e+00, 2.697309710966e-01];
const BIN_D_CI90_HI: [f64; 2] = [-1.705108240404e-01, 1.066162835057e+00];

struct Expected<'a> {
    z: &'a [f64],
    p: &'a [f64],
    ci95: (&'a [f64], &'a [f64]),
    ci90: (&'a [f64], &'a [f64]),
}

/// `tol` is the absolute tolerance on z and on the CI bounds (inherited from
/// the estimate / SE agreement with lme4); p-values are compared on the
/// same z-scale via a relative tolerance.
fn check(tag: &str, fit: &FittedGlmm, e: &Expected, tol: f64) {
    let z = fit.z_values();
    let p = fit.p_values();
    let (lo95, hi95) = fit.conf_int(0.95);
    let (lo90, hi90) = fit.conf_int(0.90);
    for j in 0..e.z.len() {
        assert!(
            (z[j] - e.z[j]).abs() < tol * (1.0 + e.z[j].abs()),
            "{tag} z[{j}] {} vs {}",
            z[j],
            e.z[j]
        );
        assert!(
            (p[j] - e.p[j]).abs() < 10.0 * tol * e.p[j].max(1e-3),
            "{tag} p[{j}] {} vs {}",
            p[j],
            e.p[j]
        );
        assert!(
            (lo95[j] - e.ci95.0[j]).abs() < tol,
            "{tag} lo95[{j}] {} vs {}",
            lo95[j],
            e.ci95.0[j]
        );
        assert!(
            (hi95[j] - e.ci95.1[j]).abs() < tol,
            "{tag} hi95[{j}] {} vs {}",
            hi95[j],
            e.ci95.1[j]
        );
        assert!(
            (lo90[j] - e.ci90.0[j]).abs() < tol,
            "{tag} lo90[{j}] {} vs {}",
            lo90[j],
            e.ci90.0[j]
        );
        assert!(
            (hi90[j] - e.ci90.1[j]).abs() < tol,
            "{tag} hi90[{j}] {} vs {}",
            hi90[j],
            e.ci90.1[j]
        );

        // Internal consistency with the point estimates and SEs.
        let b = fit.fixed_effects()[j];
        let se = fit.std_errors()[j];
        assert!((z[j] - b / se).abs() < 1e-12 * (1.0 + z[j].abs()));
        assert!(((lo95[j] + hi95[j]) / 2.0 - b).abs() < 1e-12 * (1.0 + b.abs()));
        assert!(((hi95[j] - lo95[j]) / (2.0 * se) - 1.959_963_984_540_054).abs() < 1e-9);
    }
}

fn mats(xs: &[f64], ys: &[f64]) -> (Mat<f64>, Col<f64>) {
    (
        Mat::from_fn(xs.len(), 1, |i, _| xs[i]),
        Col::from_fn(ys.len(), |i| ys[i]),
    )
}

#[test]
fn lmm_wald_inference_matches_lme4() {
    let (x, y) = mats(&X_LMM_A, &Y_LMM_A);
    let fit = GlmmRegressor::gaussian()
        .reml(true)
        .build()
        .fit(&x, &y, &G_LMM_A)
        .unwrap();
    let e = Expected {
        z: &LMM_A_Z,
        p: &LMM_A_P,
        ci95: (&LMM_A_CI95_LO, &LMM_A_CI95_HI),
        ci90: (&LMM_A_CI90_LO, &LMM_A_CI90_HI),
    };
    check("lmm", &fit, &e, 2e-3);
}

#[test]
fn poisson_glmm_wald_inference_matches_glmer() {
    let (x, y) = mats(&X_POIS_C, &Y_POIS_C);
    let fit = GlmmRegressor::poisson()
        .build()
        .fit(&x, &y, &G_POIS_C)
        .unwrap();
    let e = Expected {
        z: &POIS_C_Z,
        p: &POIS_C_P,
        ci95: (&POIS_C_CI95_LO, &POIS_C_CI95_HI),
        ci90: (&POIS_C_CI90_LO, &POIS_C_CI90_HI),
    };
    check("poisson", &fit, &e, 1e-2);
}

#[test]
fn binomial_glmm_wald_inference_matches_glmer() {
    let (x, y) = mats(&X_BIN_D, &Y_BIN_D);
    let fit = GlmmRegressor::binomial()
        .build()
        .fit(&x, &y, &G_BIN_D)
        .unwrap();
    let e = Expected {
        z: &BIN_D_Z,
        p: &BIN_D_P,
        ci95: (&BIN_D_CI95_LO, &BIN_D_CI95_HI),
        ci90: (&BIN_D_CI90_LO, &BIN_D_CI90_HI),
    };
    check("binomial", &fit, &e, 1e-2);
}

#[test]
fn conf_int_rejects_invalid_level() {
    let (x, y) = mats(&X_LMM_A, &Y_LMM_A);
    let fit = GlmmRegressor::gaussian()
        .build()
        .fit(&x, &y, &G_LMM_A)
        .unwrap();
    for level in [0.0, 1.0, -0.5, 1.5, f64::NAN] {
        let (lo, hi) = fit.conf_int(level);
        assert_eq!(lo.len(), 2);
        assert!(lo.iter().chain(&hi).all(|v| v.is_nan()), "level {level}");
    }
}
