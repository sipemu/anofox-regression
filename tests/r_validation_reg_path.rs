//! R validation of the ridge / elastic-net paths and k-fold CV
//! (`solvers::path`, `RidgeRegressor::path`, `ElasticNetRegressor::path`,
//! `FittedLars::path`).
//!
//! Reference values: `tests/r_scripts/generate_reg_path_validation.R`
//! (glmnet's documented Gaussian objective and default grid, cv.glmnet's
//! aggregates with fold ids passed in, and grpreg as an independent
//! lasso solver; glmnet itself was not installed on the generating
//! machine).

#[path = "fixtures/reg_path_validation.rs"]
mod refs;

use anofox_regression::solvers::path::{
    cv_path, fit_path, kfold_ids, lambda_grid, PathOptions, RegPath,
};
use anofox_regression::solvers::{
    ElasticNetRegressor, FittedRegressor, LarsRegressor, Regressor, RidgeRegressor,
};
use faer::{Col, Mat};
use refs::*;

fn data() -> (Mat<f64>, Col<f64>) {
    (
        Mat::from_fn(N, P, |i, j| X[i * P + j]),
        Col::from_fn(N, |i| Y[i]),
    )
}

fn max_abs_diff(a: impl IntoIterator<Item = f64>, b: &[f64]) -> f64 {
    a.into_iter()
        .zip(b)
        .fold(0.0_f64, |m, (u, v)| m.max((u - v).abs()))
}

/// Max abs difference of coefficients (column-major p x L) and intercepts.
fn path_diff(path: &RegPath, beta: &[f64], a0: &[f64]) -> (f64, f64) {
    let l = path.len();
    let coefs = (0..l).flat_map(|c| (0..P).map(move |j| (j, c)));
    let b = max_abs_diff(coefs.map(|(j, c)| path.coefficients[(j, c)]), beta);
    let a = max_abs_diff(path.intercepts.iter().copied(), a0);
    (b, a)
}

fn check_path(name: &str, opts: PathOptions, lam: &[f64], beta: &[f64], a0: &[f64], tol: f64) {
    let (x, y) = data();
    let grid = lambda_grid(&x, &y, &opts, lam.len(), None).unwrap();
    let dg = max_abs_diff(grid.iter().copied(), lam);
    let path = fit_path(&x, &y, lam, &opts).unwrap();
    let (db, da) = path_diff(&path, beta, a0);
    eprintln!("{name}: max|Δlambda| = {dg:.2e}, max|Δbeta| = {db:.2e}, max|Δa0| = {da:.2e}");
    assert!(dg <= 1e-10 * lam[0], "{name}: lambda grid {dg}");
    assert!(db < tol && da < tol, "{name}: beta {db}, a0 {da}");
}

fn check_cv(
    name: &str,
    opts: PathOptions,
    lam: &[f64],
    cvm: &[f64],
    cvsd: &[f64],
    lmin: f64,
    l1se: f64,
) {
    let (x, y) = data();
    let cv = cv_path(&x, &y, lam, &FOLDID, &opts).unwrap();
    let dm = max_abs_diff(cv.cv_mean.iter().copied(), cvm);
    let ds = max_abs_diff(cv.cv_se.iter().copied(), cvsd);
    eprintln!(
        "{name} CV: max|Δcvm| = {dm:.2e}, max|Δcvsd| = {ds:.2e}, lambda_min {} vs {lmin}, lambda_1se {} vs {l1se}",
        cv.lambda_min, cv.lambda_1se
    );
    assert!(dm < 1e-7 && ds < 1e-7, "{name}: cvm {dm}, cvsd {ds}");
    assert_eq!(cv.lambda_min, lmin, "{name}: lambda_min");
    assert_eq!(cv.lambda_1se, l1se, "{name}: lambda_1se");
    assert_eq!(cv.n_folds, 5);
}

#[test]
fn ridge_path_matches_glmnet_objective() {
    check_path(
        "ridge",
        PathOptions::ridge(),
        &RIDGE_LAMBDA,
        &RIDGE_BETA,
        &RIDGE_A0,
        1e-9,
    );
    check_path(
        "ridge std",
        PathOptions::ridge().with_standardize(true),
        &RIDGE_STD_LAMBDA,
        &RIDGE_STD_BETA,
        &RIDGE_STD_A0,
        1e-9,
    );
}

#[test]
fn elastic_net_path_matches_glmnet_objective() {
    let tol = 1e-6;
    check_path(
        "enet",
        PathOptions::elastic_net(0.5),
        &ENET_LAMBDA,
        &ENET_BETA,
        &ENET_A0,
        tol,
    );
    check_path(
        "enet std",
        PathOptions::elastic_net(0.5).with_standardize(true),
        &ENET_STD_LAMBDA,
        &ENET_STD_BETA,
        &ENET_STD_A0,
        tol,
    );
    check_path(
        "lasso std",
        PathOptions::elastic_net(1.0).with_standardize(true),
        &LASSO_STD_LAMBDA,
        &LASSO_STD_BETA,
        &LASSO_STD_A0,
        tol,
    );
}

#[test]
fn lasso_path_matches_grpreg() {
    let (x, _) = data();
    let ys = Col::from_fn(N, |i| YS[i]);
    {
        let (name, alpha, lam, beta, a0) = (
            "grpreg lasso",
            1.0,
            &GRP_LASSO_LAMBDA[..],
            &GRP_LASSO_BETA[..],
            &GRP_LASSO_A0[..],
        );
        let opts = PathOptions::elastic_net(alpha).with_standardize(true);
        let path = fit_path(&x, &ys, lam, &opts).unwrap();
        let (db, da) = path_diff(&path, beta, a0);
        eprintln!("{name}: max|Δbeta| = {db:.2e}, max|Δa0| = {da:.2e}");
        assert!(db < 1e-5 && da < 1e-5, "{name}: beta {db}, a0 {da}");
    }
}

#[test]
fn cv_matches_cv_glmnet_aggregates() {
    check_cv(
        "ridge",
        PathOptions::ridge(),
        &RIDGE_LAMBDA,
        &RIDGE_CVM,
        &RIDGE_CVSD,
        RIDGE_LMIN,
        RIDGE_L1SE,
    );
    check_cv(
        "ridge std",
        PathOptions::ridge().with_standardize(true),
        &RIDGE_STD_LAMBDA,
        &RIDGE_STD_CVM,
        &RIDGE_STD_CVSD,
        RIDGE_STD_LMIN,
        RIDGE_STD_L1SE,
    );
    check_cv(
        "enet",
        PathOptions::elastic_net(0.5),
        &ENET_LAMBDA,
        &ENET_CVM,
        &ENET_CVSD,
        ENET_LMIN,
        ENET_L1SE,
    );
    check_cv(
        "enet std",
        PathOptions::elastic_net(0.5).with_standardize(true),
        &ENET_STD_LAMBDA,
        &ENET_STD_CVM,
        &ENET_STD_CVSD,
        ENET_STD_LMIN,
        ENET_STD_L1SE,
    );
}

#[test]
fn regressor_path_methods_delegate() {
    let (x, y) = data();
    let ridge = RidgeRegressor::builder().lambda(1.0).build();
    let p = ridge.path(&x, &y, &RIDGE_LAMBDA).unwrap();
    let (db, _) = path_diff(&p, &RIDGE_BETA, &RIDGE_A0);
    assert!(db < 1e-9);
    let enet = ElasticNetRegressor::builder()
        .lambda(1.0)
        .alpha(0.5)
        .build();
    let p = enet.path(&x, &y, &ENET_LAMBDA).unwrap();
    let (db, _) = path_diff(&p, &ENET_BETA, &ENET_A0);
    assert!(db < 1e-6);
    let cv = enet.cv_path(&x, &y, &ENET_LAMBDA, &FOLDID).unwrap();
    assert_eq!(cv.lambda_min, ENET_LMIN);
}

/// With unit-variance y and no standardisation, the ridge path equals
/// `RidgeRegressor` with `LambdaScaling::Glmnet` at every lambda.
#[test]
fn ridge_path_matches_ridge_regressor_glmnet_scaling() {
    let (x, _) = data();
    let ys = Col::from_fn(N, |i| YS[i]);
    let lams = [5.0, 0.5, 0.01];
    let path = fit_path(&x, &ys, &lams, &PathOptions::ridge()).unwrap();
    for (l, &lam) in lams.iter().enumerate() {
        let f = RidgeRegressor::builder()
            .lambda(lam)
            .lambda_scaling(anofox_regression::LambdaScaling::Glmnet)
            .build()
            .fit(&x, &ys)
            .unwrap();
        for j in 0..P {
            assert!((path.coefficients[(j, l)] - f.result().coefficients[j]).abs() < 1e-9);
        }
        assert!((path.intercepts[l] - f.result().intercept.unwrap()).abs() < 1e-9);
    }
}

/// KKT conditions of the elastic-net objective at every lambda of the path.
#[test]
fn elastic_net_path_satisfies_kkt() {
    let (x, y) = data();
    let opts = PathOptions::elastic_net(0.3);
    let lam = lambda_grid(&x, &y, &opts, 40, None).unwrap();
    let path = fit_path(&x, &y, &lam, &opts).unwrap();
    let n = N as f64;
    let ym = y.iter().sum::<f64>() / n;
    let sy = (y.iter().map(|v| (v - ym).powi(2)).sum::<f64>() / n).sqrt();
    let fitted = path.predict(&x);
    for (l, &lambda) in lam.iter().enumerate() {
        for j in 0..P {
            // gradient of 1/(2n) RSS + lambda[(1-a)/(2 sy) b^2 + a |b|]
            let g: f64 = (0..N)
                .map(|i| x[(i, j)] * (y[i] - fitted[(i, l)]))
                .sum::<f64>()
                / n;
            let b = path.coefficients[(j, l)];
            let ridge = lambda * (1.0 - 0.3) / sy * b;
            let l1 = lambda * 0.3;
            if b != 0.0 {
                assert!(
                    (g - ridge - l1 * b.signum()).abs() < 1e-6 * (1.0 + l1),
                    "l={l} j={j}"
                );
            } else {
                assert!(g.abs() <= l1 * (1.0 + 1e-6) + 1e-9, "l={l} j={j}");
            }
        }
    }
}

#[test]
fn lars_path_has_the_same_shape() {
    let (x, y) = data();
    let fit = LarsRegressor::builder().build().fit(&x, &y).unwrap();
    let path = fit.path();
    assert_eq!(path.coefficients.nrows(), P);
    assert_eq!(path.len(), fit.alphas().len());
    let last = path.len() - 1;
    // The end of the LARS path is the OLS fit.
    for j in 0..P {
        assert!((path.coefficients[(j, last)] - fit.result().coefficients[j]).abs() < 1e-8);
    }
    assert!((path.intercepts[last] - fit.result().intercept.unwrap()).abs() < 1e-8);
    assert_eq!(path.df[0], 0);
}

#[test]
fn seeded_folds_are_reproducible() {
    let (x, y) = data();
    let opts = PathOptions::elastic_net(0.5);
    let lam = lambda_grid(&x, &y, &opts, 20, None).unwrap();
    let folds = kfold_ids(N, 10, 99);
    let a = cv_path(&x, &y, &lam, &folds, &opts).unwrap();
    let b = cv_path(&x, &y, &lam, &kfold_ids(N, 10, 99), &opts).unwrap();
    assert_eq!(a.cv_mean, b.cv_mean);
    assert_eq!(a.n_folds, 10);
    assert!(a.lambda_1se >= a.lambda_min);
}
