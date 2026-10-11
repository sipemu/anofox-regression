//! R validation of `diagnostics::augment_glm` / `FittedPenalizedGlm::augment`:
//! per-observation diagnostics of fits of the penalized GLM engine
//! (`penalized_glm::fit_*`) against R's `glm` / `MASS::glm.nb`
//! (`fitted`, `residuals(type = "deviance" | "pearson" | "response")`,
//! `rstandard`, `rstudent`, `hatvalues`, `cooks.distance`, `dffits`), and
//! against `diagnostics::augment` on the dedicated GLM regressors.
//!
//! Reference values: `tests/r_scripts/generate_augment_penalized_glm_validation.R`
//! (30 rows, 2 features, row 30 with leverage ~0.82).

#[path = "fixtures/augment_penalized_glm_validation.rs"]
mod refs;

use anofox_regression::diagnostics::{
    augment, augment_glm, augment_glm_with, Augment, ResidualType,
};
use anofox_regression::prelude::*;
use anofox_regression::solvers::penalized_glm::{
    fit_binomial, fit_gamma, fit_logistic, fit_negbinomial, fit_poisson, fit_tweedie,
    BinomialOptions, GammaOptions, GlmEngineError, GlmResult, LogisticOptions, NegBinomialOptions,
    PenalizedGlmFamily, PenalizedGlmRegressor, PoissonOptions, PriorSpec, TweedieOptions,
};
use faer::{Col, Mat};
use refs::*;

const TOL: f64 = 1e-6;
const MAX_ITER: u32 = 200;
const ETOL: f64 = 1e-15;

fn x_cols() -> Vec<Vec<f64>> {
    vec![X1.to_vec(), X2.to_vec()]
}

fn x_mat() -> Mat<f64> {
    Mat::from_fn(X1.len(), 2, |i, j| if j == 0 { X1[i] } else { X2[i] })
}

fn col(v: &[f64]) -> Col<f64> {
    Col::from_fn(v.len(), |i| v[i])
}

/// Largest absolute difference, asserting a relative tolerance per element.
fn cmp(label: &str, got: &Col<f64>, want: &[f64], tol: f64) -> f64 {
    assert_eq!(got.nrows(), want.len(), "{label}: length");
    let mut max = 0.0_f64;
    for (i, &w) in want.iter().enumerate() {
        let d = (got[i] - w).abs();
        assert!(
            d <= tol * w.abs().max(1.0),
            "{label}[{i}]: got {:.12e}, R {w:.12e}",
            got[i]
        );
        max = max.max(d);
    }
    max
}

/// Compares all three residual types and every diagnostic; returns the max abs
/// difference over everything.
fn check(label: &str, fit: &GlmResult, y: &[f64], x: &[Vec<f64>], r: &AugRef, tol: f64) -> f64 {
    let a = augment_glm(fit, y, x).unwrap();
    assert_eq!(a.resid_type, ResidualType::Deviance);
    let mut m = 0.0_f64;
    m = m.max(cmp(&format!("{label} fitted"), &a.fitted, r.fitted, tol));
    m = m.max(cmp(&format!("{label} resid dev"), &a.residual, r.dev, tol));
    m = m.max(cmp(
        &format!("{label} rstandard dev"),
        &a.std_residual,
        r.rstd_dev,
        tol,
    ));
    m = m.max(cmp(
        &format!("{label} rstudent"),
        &a.stud_residual,
        r.rstud,
        tol,
    ));
    m = m.max(cmp(&format!("{label} hat"), &a.leverage, r.hat, tol));
    m = m.max(cmp(&format!("{label} cooks"), &a.cooks_d, r.cook, tol));
    m = m.max(cmp(&format!("{label} dffits"), &a.dffits, r.dffits, tol));
    let p = augment_glm_with(fit, y, x, ResidualType::Pearson).unwrap();
    m = m.max(cmp(
        &format!("{label} resid pearson"),
        &p.residual,
        r.pear,
        tol,
    ));
    m = m.max(cmp(
        &format!("{label} rstandard pearson"),
        &p.std_residual,
        r.rstd_pear,
        tol,
    ));
    let q = augment_glm_with(fit, y, x, ResidualType::Response).unwrap();
    m = m.max(cmp(
        &format!("{label} resid response"),
        &q.residual,
        r.resp,
        tol,
    ));
    eprintln!("{label}: max |augment_glm - R| = {m:.3e}");
    m
}

fn assert_same(label: &str, a: &Augment, b: &Augment, tol: f64) {
    let pairs = [
        ("fitted", &a.fitted, &b.fitted),
        ("residual", &a.residual, &b.residual),
        ("std_residual", &a.std_residual, &b.std_residual),
        ("stud_residual", &a.stud_residual, &b.stud_residual),
        ("leverage", &a.leverage, &b.leverage),
        ("cooks_d", &a.cooks_d, &b.cooks_d),
        ("dffits", &a.dffits, &b.dffits),
    ];
    for (name, u, v) in pairs {
        assert_eq!(u.nrows(), v.nrows(), "{label} {name}: length");
        for i in 0..u.nrows() {
            let (p, q) = (u[i], v[i]);
            assert!(
                (p.is_nan() && q.is_nan()) || (p - q).abs() <= tol * q.abs().max(1.0),
                "{label} {name}[{i}]: {p:.12e} vs {q:.12e}"
            );
        }
    }
}

fn poisson_opts() -> PoissonOptions {
    PoissonOptions {
        max_iterations: MAX_ITER,
        tolerance: ETOL,
        ..Default::default()
    }
}

// ------------------------------------------------------------ R validation

#[test]
fn poisson_matches_r() {
    let fit = fit_poisson(&Y_POIS, &x_cols(), &poisson_opts()).unwrap();
    check("Poisson", &fit, &Y_POIS, &x_cols(), &POIS, TOL);
}

#[test]
fn poisson_with_offset_matches_r() {
    let mut x = x_cols();
    x.push(LOG_EXPO.to_vec());
    let opts = PoissonOptions {
        offset_column: Some(3),
        ..poisson_opts()
    };
    let fit = fit_poisson(&Y_OFF, &x, &opts).unwrap();
    assert_eq!(fit.offset_column, Some(3));
    check("Poisson+offset", &fit, &Y_OFF, &x, &POFF, TOL);

    // The builder (offset held outside x) gives the same diagnostics.
    let b = PenalizedGlmRegressor::builder()
        .family(PenalizedGlmFamily::poisson())
        .max_iterations(MAX_ITER)
        .tolerance(ETOL)
        .offset(col(&LOG_EXPO))
        .build()
        .fit(&x_mat(), &col(&Y_OFF))
        .unwrap();
    let via_builder = b.augment(&x_mat(), &col(&Y_OFF)).unwrap();
    assert_same(
        "builder vs fit_poisson",
        &via_builder,
        &augment_glm(&fit, &Y_OFF, &x).unwrap(),
        1e-12,
    );
}

#[test]
fn quasipoisson_matches_r() {
    let opts = PoissonOptions {
        estimate_dispersion: true,
        ..poisson_opts()
    };
    let fit = fit_poisson(&Y_POIS, &x_cols(), &opts).unwrap();
    check("quasipoisson", &fit, &Y_POIS, &x_cols(), &QPOIS, TOL);
}

#[test]
fn binomial_matches_r() {
    let opts = BinomialOptions {
        max_iterations: MAX_ITER,
        tolerance: ETOL,
        ..Default::default()
    };
    let fit = fit_binomial(&Y_BIN, &x_cols(), &opts).unwrap();
    check("binomial", &fit, &Y_BIN, &x_cols(), &BIN, TOL);
}

#[test]
fn binomial_proportions_match_r() {
    let opts = BinomialOptions {
        max_iterations: MAX_ITER,
        tolerance: ETOL,
        ..Default::default()
    };
    let fit = fit_binomial(&Y_PROP, &x_cols(), &opts).unwrap();
    check("binomial proportions", &fit, &Y_PROP, &x_cols(), &PROP, TOL);
}

#[test]
fn logistic_matches_r() {
    let opts = LogisticOptions {
        max_iterations: MAX_ITER,
        tolerance: ETOL,
        ..Default::default()
    };
    let fit = fit_logistic(&Y_BIN, &x_cols(), &opts).unwrap();
    check("logistic", &fit.fit, &Y_BIN, &x_cols(), &BIN, TOL);
}

#[test]
fn negative_binomial_estimated_theta_matches_glm_nb() {
    let opts = NegBinomialOptions {
        max_iterations: MAX_ITER,
        tolerance: ETOL,
        ..Default::default()
    };
    let fit = fit_negbinomial(&Y_NB, &x_cols(), &opts).unwrap();
    let theta = fit.core.dispersion.unwrap();
    assert!(
        (theta - NB_THETA).abs() < 1e-4 * NB_THETA,
        "theta {theta} vs R {NB_THETA}"
    );
    // theta is estimated by an alternation that stops at MASS's tolerance, so the
    // diagnostics agree to the precision of theta.
    check("NB (theta estimated)", &fit, &Y_NB, &x_cols(), &NB, 1e-4);
}

#[test]
fn negative_binomial_fixed_theta_matches_r() {
    let opts = NegBinomialOptions {
        alpha: Some(2.0),
        max_iterations: MAX_ITER,
        tolerance: ETOL,
        ..Default::default()
    };
    let fit = fit_negbinomial(&Y_NB, &x_cols(), &opts).unwrap();
    check("NB (theta = 2)", &fit, &Y_NB, &x_cols(), &NBF, TOL);
}

#[test]
fn gamma_matches_r() {
    let opts = GammaOptions {
        max_iterations: MAX_ITER,
        tolerance: ETOL,
        ..Default::default()
    };
    let fit = fit_gamma(&Y_GAMMA, &x_cols(), &opts).unwrap();
    check("Gamma", &fit, &Y_GAMMA, &x_cols(), &GAMMA, TOL);
}

#[test]
fn tweedie_matches_r() {
    if !HAVE_TWEEDIE {
        eprintln!("statmod not installed when the fixture was generated: skipped");
        return;
    }
    let opts = TweedieOptions {
        power: 1.5,
        max_iterations: MAX_ITER,
        tolerance: ETOL,
        ..Default::default()
    };
    let fit = fit_tweedie(&Y_TW, &x_cols(), &opts).unwrap();
    check("Tweedie p=1.5", &fit, &Y_TW, &x_cols(), &TW, TOL);
}

// ------------------------------------------------- dropped rows / columns

#[test]
fn a_constant_or_collinear_column_does_not_change_the_diagnostics() {
    let n = X1.len();
    let constant = vec![3.0; n];
    let collinear: Vec<f64> = (0..n).map(|i| 2.0 * X1[i] - X2[i]).collect();
    let x = vec![X1.to_vec(), constant, X2.to_vec(), collinear];
    let fit = fit_poisson(&Y_POIS, &x, &poisson_opts()).unwrap();
    assert!(
        fit.core.coefficients[1].is_nan(),
        "constant column is dropped"
    );
    check(
        "Poisson + constant + collinear",
        &fit,
        &Y_POIS,
        &x,
        &POIS,
        TOL,
    );
}

#[test]
fn dropped_rows_are_nan_and_the_rest_unchanged() {
    let mut y = Y_POIS.to_vec();
    let mut x = x_cols();
    y.push(f64::NAN);
    x[0].push(1.0);
    x[1].push(0.5);
    y.insert(0, 4.0);
    x[0].insert(0, f64::INFINITY);
    x[1].insert(0, 0.0);
    let fit = fit_poisson(&y, &x, &poisson_opts()).unwrap();
    let a = augment_glm(&fit, &y, &x).unwrap();
    let n = y.len();
    assert_eq!(a.fitted.nrows(), n);
    for i in [0, n - 1] {
        assert!(a.fitted[i].is_nan() && a.leverage[i].is_nan() && a.cooks_d[i].is_nan());
    }
    let inner = |v: &Col<f64>| Col::from_fn(n - 2, |i| v[i + 1]);
    cmp("dropped rows: hat", &inner(&a.leverage), POIS.hat, TOL);
    cmp("dropped rows: cooks", &inner(&a.cooks_d), POIS.cook, TOL);
    cmp(
        "dropped rows: rstudent",
        &inner(&a.stud_residual),
        POIS.rstud,
        TOL,
    );
}

// --------------------------------------------------------------- errors

#[test]
fn a_penalized_fit_is_rejected() {
    let ridge = PoissonOptions {
        lambda: 0.5,
        ..poisson_opts()
    };
    let fit = fit_poisson(&Y_POIS, &x_cols(), &ridge).unwrap();
    assert!(fit.penalized);
    assert!(matches!(
        augment_glm(&fit, &Y_POIS, &x_cols()),
        Err(GlmEngineError::InvalidInput(_))
    ));

    let b = PenalizedGlmRegressor::builder()
        .family(PenalizedGlmFamily::poisson())
        .priors(vec![PriorSpec::normal(0.0, 1.0), PriorSpec::flat()])
        .build()
        .fit(&x_mat(), &col(&Y_POIS))
        .unwrap();
    assert!(b.result().penalized);
    assert!(b.augment(&x_mat(), &col(&Y_POIS)).is_err());

    // Flat priors are not a penalty.
    let mut flat = poisson_opts();
    flat.prior_opts.priors = vec![PriorSpec::flat(), PriorSpec::flat()];
    let fit = fit_poisson(&Y_POIS, &x_cols(), &flat).unwrap();
    assert!(!fit.penalized);
    assert!(augment_glm(&fit, &Y_POIS, &x_cols()).is_ok());
}

#[test]
fn mismatched_data_is_rejected() {
    let fit = fit_poisson(&Y_POIS, &x_cols(), &poisson_opts()).unwrap();
    assert!(augment_glm(&fit, &Y_POIS[..29], &x_cols()).is_err());
    assert!(augment_glm(&fit, &Y_POIS, &[X1.to_vec()]).is_err());
}

// ----------------------------------- equal to the dedicated regressors

#[test]
fn equals_augment_on_the_dedicated_regressors() {
    let x = x_mat();
    let tol = 1e-6;

    let y = col(&Y_POIS);
    let ded = PoissonRegressor::log()
        .with_intercept(true)
        .max_iterations(MAX_ITER as usize)
        .tolerance(ETOL)
        .build()
        .fit(&x, &y)
        .unwrap();
    let eng = fit_poisson(&Y_POIS, &x_cols(), &poisson_opts()).unwrap();
    assert_same(
        "Poisson",
        &augment_glm(&eng, &Y_POIS, &x_cols()).unwrap(),
        &augment(&ded, &x, &y, None).unwrap(),
        tol,
    );

    let y = col(&Y_BIN);
    let ded = BinomialRegressor::logistic()
        .with_intercept(true)
        .max_iterations(MAX_ITER as usize)
        .tolerance(ETOL)
        .build()
        .fit(&x, &y)
        .unwrap();
    let eng = fit_binomial(
        &Y_BIN,
        &x_cols(),
        &BinomialOptions {
            max_iterations: MAX_ITER,
            tolerance: ETOL,
            ..Default::default()
        },
    )
    .unwrap();
    assert_same(
        "Binomial",
        &augment_glm_with(&eng, &Y_BIN, &x_cols(), ResidualType::Pearson).unwrap(),
        &anofox_regression::diagnostics::augment_with(&ded, &x, &y, None, ResidualType::Pearson)
            .unwrap(),
        tol,
    );

    let ded = LogisticRegression::builder()
        .with_intercept(true)
        .max_iterations(MAX_ITER as usize)
        .tolerance(ETOL)
        .build()
        .fit(&x, &y)
        .unwrap();
    let eng = fit_logistic(
        &Y_BIN,
        &x_cols(),
        &LogisticOptions {
            max_iterations: MAX_ITER,
            tolerance: ETOL,
            ..Default::default()
        },
    )
    .unwrap();
    assert_same(
        "Logistic",
        &augment_glm(&eng.fit, &Y_BIN, &x_cols()).unwrap(),
        &augment(&ded, &x, &y, None).unwrap(),
        tol,
    );

    let y = col(&Y_NB);
    let ded = NegativeBinomialRegressor::builder()
        .with_intercept(true)
        .estimate_theta(false)
        .theta(2.0)
        .max_iterations(MAX_ITER as usize)
        .tolerance(ETOL)
        .build()
        .fit(&x, &y)
        .unwrap();
    let eng = fit_negbinomial(
        &Y_NB,
        &x_cols(),
        &NegBinomialOptions {
            alpha: Some(2.0),
            max_iterations: MAX_ITER,
            tolerance: ETOL,
            ..Default::default()
        },
    )
    .unwrap();
    assert_same(
        "NB (theta = 2)",
        &augment_glm(&eng, &Y_NB, &x_cols()).unwrap(),
        &augment(&ded, &x, &y, None).unwrap(),
        tol,
    );

    let ded = NegativeBinomialRegressor::builder()
        .with_intercept(true)
        .max_iterations(MAX_ITER as usize)
        .tolerance(ETOL)
        .build()
        .fit(&x, &y)
        .unwrap();
    let eng = fit_negbinomial(
        &Y_NB,
        &x_cols(),
        &NegBinomialOptions {
            max_iterations: MAX_ITER,
            tolerance: ETOL,
            ..Default::default()
        },
    )
    .unwrap();
    assert_same(
        "NB (theta estimated)",
        &augment_glm(&eng, &Y_NB, &x_cols()).unwrap(),
        &augment(&ded, &x, &y, None).unwrap(),
        1e-4,
    );

    let y = col(&Y_GAMMA);
    let ded = GammaRegressor::builder()
        .with_intercept(true)
        .max_iterations(MAX_ITER as usize)
        .tolerance(ETOL)
        .build()
        .fit(&x, &y)
        .unwrap();
    let eng = fit_gamma(
        &Y_GAMMA,
        &x_cols(),
        &GammaOptions {
            max_iterations: MAX_ITER,
            tolerance: ETOL,
            ..Default::default()
        },
    )
    .unwrap();
    assert_same(
        "Gamma",
        &augment_glm(&eng, &Y_GAMMA, &x_cols()).unwrap(),
        &augment(&ded, &x, &y, None).unwrap(),
        tol,
    );

    let ytw: Vec<f64> = if HAVE_TWEEDIE {
        Y_TW.to_vec()
    } else {
        Y_GAMMA.to_vec()
    };
    let y = col(&ytw);
    let ded = TweedieRegressor::builder()
        .var_power(1.5)
        .link_power(0.0)
        .with_intercept(true)
        .max_iterations(MAX_ITER as usize)
        .tolerance(ETOL)
        .build()
        .fit(&x, &y)
        .unwrap();
    let eng = fit_tweedie(
        &ytw,
        &x_cols(),
        &TweedieOptions {
            power: 1.5,
            max_iterations: MAX_ITER,
            tolerance: ETOL,
            ..Default::default()
        },
    )
    .unwrap();
    assert_same(
        "Tweedie",
        &augment_glm(&eng, &ytw, &x_cols()).unwrap(),
        &augment(&ded, &x, &y, None).unwrap(),
        tol,
    );
}
