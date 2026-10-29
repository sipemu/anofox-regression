#![allow(clippy::excessive_precision, dead_code)]

//! #101: LR and score tests for nested GLMs and profile-likelihood CIs
//! against `glm`, `anova(..., test = "LRT" / "Rao")`,
//! `statmod::glm.scoretest` and `confint()`.
//!
//! Reference values come from `tests/r_scripts/generate_likelihood_validation.R`.

use anofox_regression::core::{BinomialFamily, GlmFamily, PoissonFamily, TweedieFamily};
use anofox_regression::inference::{
    fit_glm, glm_score_test, lr_test, model_lr_test, profile_confint, Dispersion, GlmData,
};
use faer::Mat;

include!("fixtures/likelihood_validation.rs");

fn close(name: &str, a: f64, b: f64, tol: f64) {
    let ok = (a - b).abs() <= tol * (1.0 + b.abs());
    assert!(ok, "{name}: {a} vs {b}");
}

fn design(cols: &[&[f64]]) -> Mat<f64> {
    Mat::from_fn(X1.len(), cols.len(), |i, j| cols[j][i])
}

struct Case<'a> {
    id: &'a str,
    family: &'a dyn GlmFamily,
    y: &'a [f64],
    weights: Option<&'a [f64]>,
    offset: Option<&'a [f64]>,
    dispersion: Dispersion,
    coef: &'a [f64],
    vcov: &'a [f64],
    dev: f64,
    nulldev: f64,
    disp: f64,
    lr: f64,
    lr_p: f64,
    rao: f64,
    rao_p: f64,
    score_z: &'a [f64],
    ci95: &'a [f64],
    ci90: &'a [f64],
}

fn run(c: &Case) {
    let full_x = design(&[&X1, &X2, &X3]);
    let red_x = design(&[&X1]);
    let add_x = design(&[&X2, &X3]);
    let full = GlmData {
        x: &full_x,
        with_intercept: true,
        y: c.y,
        weights: c.weights,
        offset: c.offset,
    };
    let red = GlmData { x: &red_x, ..full };
    let f = fit_glm(c.family, &full, c.dispersion).unwrap();
    let r = fit_glm(c.family, &red, c.dispersion).unwrap();
    let id = c.id;
    for (j, (a, b)) in f.coefficients.iter().zip(c.coef).enumerate() {
        close(&format!("{id} coef[{j}]"), *a, *b, 1e-9);
    }
    let k = c.coef.len();
    for a in 0..k {
        for b in 0..k {
            close(
                &format!("{id} vcov[{a},{b}]"),
                f.vcov[(a, b)],
                c.vcov[a * k + b],
                1e-6,
            );
        }
    }
    close(&format!("{id} deviance"), f.deviance, c.dev, 1e-10);
    close(
        &format!("{id} null deviance"),
        f.null_deviance,
        c.nulldev,
        1e-10,
    );
    close(&format!("{id} dispersion"), f.dispersion, c.disp, 1e-8);

    let t = lr_test(&r, &f);
    assert_eq!(t.df, 2);
    close(&format!("{id} LR"), t.statistic, c.lr, 1e-8);
    close(&format!("{id} LR p"), t.p_value, c.lr_p, 1e-6);
    let m = model_lr_test(&f).unwrap();
    close(
        &format!("{id} model LR"),
        m.statistic,
        (c.nulldev - c.dev) / c.disp,
        1e-8,
    );
    assert_eq!(m.df, 3);

    // anova(test = "Rao") scales by the larger model's dispersion and
    // reports the unscaled statistic.
    let s = glm_score_test(c.family, &red, &add_x, f.dispersion).unwrap();
    close(&format!("{id} Rao"), s.test.statistic, c.rao / c.disp, 1e-7);
    close(&format!("{id} Rao p"), s.test.p_value, c.rao_p, 1e-6);
    // glm.scoretest uses the reduced fit's dispersion.
    let z = glm_score_test(c.family, &red, &add_x, r.dispersion)
        .unwrap()
        .z;
    for (j, (a, b)) in z.iter().zip(c.score_z).enumerate() {
        close(&format!("{id} score z[{j}]"), *a, *b, 1e-7);
    }

    for (level, want) in [(0.95, c.ci95), (0.9, c.ci90)] {
        let ci = profile_confint(c.family, &full, c.dispersion, level).unwrap();
        for (j, (lo, hi)) in ci.iter().enumerate() {
            close(&format!("{id} {level} lower[{j}]"), *lo, want[2 * j], 1e-6);
            close(
                &format!("{id} {level} upper[{j}]"),
                *hi,
                want[2 * j + 1],
                1e-6,
            );
        }
    }
}

#[test]
fn binomial_matches_r() {
    run(&Case {
        id: "binomial",
        family: &BinomialFamily::logistic(),
        y: &YB,
        weights: None,
        offset: None,
        dispersion: Dispersion::Fixed(1.0),
        coef: &BIN_COEF,
        vcov: &BIN_VCOV,
        dev: BIN_DEV,
        nulldev: BIN_NULLDEV,
        disp: BIN_DISP,
        lr: BIN_LR,
        lr_p: BIN_LR_P,
        rao: BIN_RAO,
        rao_p: BIN_RAO_P,
        score_z: &BIN_SCORE_Z,
        ci95: &BIN_CI95,
        ci90: &BIN_CI90,
    });
}

#[test]
fn weighted_poisson_with_offset_matches_r() {
    let off: Vec<f64> = EXPO.iter().map(|v| v.ln()).collect();
    run(&Case {
        id: "poisson",
        family: &PoissonFamily::log(),
        y: &YP,
        weights: Some(&W),
        offset: Some(&off),
        dispersion: Dispersion::Fixed(1.0),
        coef: &POIS_COEF,
        vcov: &POIS_VCOV,
        dev: POIS_DEV,
        nulldev: POIS_NULLDEV,
        disp: POIS_DISP,
        lr: POIS_LR,
        lr_p: POIS_LR_P,
        rao: POIS_RAO,
        rao_p: POIS_RAO_P,
        score_z: &POIS_SCORE_Z,
        ci95: &POIS_CI95,
        ci90: &POIS_CI90,
    });
}

#[test]
fn gamma_with_estimated_dispersion_matches_r() {
    run(&Case {
        id: "gamma",
        family: &TweedieFamily::gamma(),
        y: &YG,
        weights: None,
        offset: None,
        dispersion: Dispersion::Estimated,
        coef: &GAMMA_COEF,
        vcov: &GAMMA_VCOV,
        dev: GAMMA_DEV,
        nulldev: GAMMA_NULLDEV,
        disp: GAMMA_DISP,
        lr: GAMMA_LR,
        lr_p: GAMMA_LR_P,
        rao: GAMMA_RAO,
        rao_p: GAMMA_RAO_P,
        score_z: &GAMMA_SCORE_Z,
        ci95: &GAMMA_CI95,
        ci90: &GAMMA_CI90,
    });
}

/// The `LikelihoodFit` trait on the crate's own GLM fits gives R's LR tests.
#[test]
fn likelihood_fit_trait_on_crate_fits() {
    use anofox_regression::solvers::{BinomialRegressor, GammaRegressor, Regressor};
    use faer::Col;
    let full_x = design(&[&X1, &X2, &X3]);
    let red_x = design(&[&X1]);
    let yb = Col::from_fn(YB.len(), |i| YB[i]);
    let fb = BinomialRegressor::logistic()
        .build()
        .fit(&full_x, &yb)
        .unwrap();
    let rb = BinomialRegressor::logistic()
        .build()
        .fit(&red_x, &yb)
        .unwrap();
    let t = lr_test(&rb, &fb);
    assert_eq!(t.df, 2);
    close("binomial LR (crate fit)", t.statistic, BIN_LR, 1e-6);
    close(
        "binomial model LR",
        model_lr_test(&fb).unwrap().statistic,
        BIN_NULLDEV - BIN_DEV,
        1e-6,
    );

    let yg = Col::from_fn(YG.len(), |i| YG[i]);
    let fg = GammaRegressor::builder().build().fit(&full_x, &yg).unwrap();
    let rg = GammaRegressor::builder().build().fit(&red_x, &yg).unwrap();
    close(
        "gamma LR (crate fit)",
        lr_test(&rg, &fg).statistic,
        GAMMA_LR,
        1e-5,
    );
}
