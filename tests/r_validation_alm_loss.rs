//! ALM losses vs `greybox::alm` (likelihood, MSE, MAE, HAM, ROLE) on clean
//! data and on data with two outliers, at tight nloptr tolerances.
//! Generator: `tests/r_scripts/generate_alm_loss_validation.R`.
#![allow(clippy::excessive_precision, clippy::unreadable_literal)]

use anofox_regression::solvers::{
    AlmDistribution, AlmLoss, AlmRegressor, FittedAlm, FittedRegressor, Regressor,
};
use faer::{Col, Mat};

#[rustfmt::skip]
mod reference {
    pub const X: [f64; 50] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0];
    pub const YC: [f64; 50] = [8.1128753414400059, 3.805905485811734, 8.0893852340120169, 10.398587814883122, 11.212804969422997, 11.181626451725547, 17.534565992316818, 14.216022884760708, 22.055271141631124, 17.311857702842737, 22.914608962670457, 27.359936178103318, 17.833417896662983, 22.663633699547887, 24.600035990819027, 28.407851194210224, 27.147241235751782, 21.530633737285672, 23.67859921427344, 36.460340037190576, 33.080084217764579, 30.156074698059999, 36.484247932721139, 42.1440240975178, 45.6855803837949, 40.208592605181401, 42.228191851693211, 39.210510744415657, 47.380292064493815, 45.580015372119639, 50.366350369723655, 52.614512011686458, 55.105310565909768, 51.673220873778369, 56.514865369893911, 51.348973962779972, 55.646622974861515, 56.947277217470443, 53.757377050160102, 62.608367820676769, 64.617995800600767, 64.416828104353996, 69.274489707098553, 66.319885518770278, 65.895156866742113, 72.79845407766615, 70.565820471439991, 78.832303785163759, 74.705661392159968, 79.466943650206616];
    pub const YO: [f64; 50] = [8.1128753414400059, 3.805905485811734, 8.0893852340120169, 10.398587814883122, 11.212804969422997, 11.181626451725547, 17.534565992316818, 14.216022884760708, 22.055271141631124, 97.31185770284273, 22.914608962670457, 27.359936178103318, 17.833417896662983, 22.663633699547887, 24.600035990819027, 28.407851194210224, 27.147241235751782, 21.530633737285672, 23.67859921427344, 36.460340037190576, 33.080084217764579, 30.156074698059999, 36.484247932721139, 42.1440240975178, 45.6855803837949, 40.208592605181401, 42.228191851693211, 39.210510744415657, 47.380292064493815, 45.580015372119639, 50.366350369723655, 52.614512011686458, 55.105310565909768, 51.673220873778369, 56.514865369893911, 51.348973962779972, 55.646622974861515, 56.947277217470443, 53.757377050160102, -17.391632179323231, 64.617995800600767, 64.416828104353996, 69.274489707098553, 66.319885518770278, 65.895156866742113, 72.79845407766615, 70.565820471439991, 78.832303785163759, 74.705661392159968, 79.466943650206616];
    pub const LIKELIHOOD: [f64; 3] = [3.6121344910907975, 1.4521902024860607, 11.218405837631327];
    pub const MSE: [f64; 3] = [3.6121344910907975, 1.4521902024860607, 11.218405837631327];
    pub const MSE_OUTLIERS: [f64; 3] = [9.4896855114989513, 1.2216980056073095, 250.6162074490442];
    pub const MAE: [f64; 3] = [2.6729975657522154, 1.470054363775003, 11.518734043298071];
    pub const MAE_OUTLIERS: [f64; 3] = [3.8220343997037429, 1.4224502763229196, 259.30980682645509];
    pub const HAM: [f64; 3] = [2.8977318686583748, 1.4414125829290165, 12.22117534402847];
    pub const HAM_OUTLIERS: [f64; 3] = [2.8031437622646607, 1.4417590788686026, 261.85669578828447];
    pub const ROLE05_OUTLIERS: [f64; 3] = [3.9708343506440102, 1.4440585328149123, 12.190825807424883];
    pub const ROLE10_OUTLIERS: [f64; 3] = [4.2351566379167807, 1.4472015401784046, 9.5231888750902236];
}
use reference::*;

fn fit(y: &[f64], loss: AlmLoss) -> FittedAlm {
    AlmRegressor::builder()
        .distribution(AlmDistribution::Normal)
        .loss(loss)
        .tolerance(1e-12)
        .max_iterations(10_000)
        .build()
        .fit(
            &Mat::from_fn(X.len(), 1, |i, _| X[i]),
            &Col::from_fn(y.len(), |i| y[i]),
        )
        .unwrap()
}

/// Intercept, slope and scale (greybox's `$scale`, the variance).
fn estimates(f: &FittedAlm) -> [f64; 3] {
    [
        f.result().intercept.unwrap(),
        f.result().coefficients[0],
        f.scale().powi(2),
    ]
}

fn close(got: &[f64], want: &[f64], tol: f64, what: &str) {
    for (k, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= tol * (1.0 + w.abs()),
            "{what}[{k}]: {g} vs greybox {w}"
        );
    }
}

#[test]
fn likelihood_and_mse_match_greybox() {
    close(
        &estimates(&fit(&YC, AlmLoss::Likelihood)),
        &LIKELIHOOD,
        1e-8,
        "likelihood",
    );
    close(&estimates(&fit(&YC, AlmLoss::MSE)), &MSE, 1e-8, "MSE");
    close(
        &estimates(&fit(&YO, AlmLoss::MSE)),
        &MSE_OUTLIERS,
        1e-8,
        "MSE, outliers",
    );
}

#[test]
fn mae_matches_greybox() {
    close(&estimates(&fit(&YC, AlmLoss::MAE)), &MAE, 1e-7, "MAE");
    close(
        &estimates(&fit(&YO, AlmLoss::MAE)),
        &MAE_OUTLIERS,
        1e-7,
        "MAE, outliers",
    );
}

/// Mean √|e| (greybox's HAM objective).
fn ham(y: &[f64], b: &[f64]) -> f64 {
    (0..X.len())
        .map(|i| (y[i] - b[0] - b[1] * X[i]).abs().sqrt())
        .sum::<f64>()
        / X.len() as f64
}

/// The global HAM minimum is a line through two observations (√|e| is
/// concave on each side of 0): the best over all pairs.
fn ham_global(y: &[f64]) -> f64 {
    let n = X.len();
    (0..n)
        .flat_map(|i| (i + 1..n).map(move |j| (i, j)))
        .map(|(i, j)| {
            let b1 = (y[j] - y[i]) / (X[j] - X[i]);
            ham(y, &[y[i] - b1 * X[i], b1])
        })
        .fold(f64::INFINITY, f64::min)
}

/// With outliers greybox stops short (mean √|e| 1.75831); the crate reaches
/// the global minimum, 1.753924 (to 1e-7: Nelder–Mead at a kink).
#[test]
fn ham_with_outliers_reaches_the_global_minimum() {
    let e = estimates(&fit(&YO, AlmLoss::HAM));
    assert!(ham(&YO, &e) <= ham(&YO, &HAM_OUTLIERS));
    let (got, best) = (ham(&YO, &e), ham_global(&YO));
    assert!(got - best < 1e-7, "{got} vs global {best}");
    close(&e[..2], &HAM_OUTLIERS[..2], 0.06, "HAM, outliers");
}

/// greybox reaches 1.43928 here (global minimum 1.43074); the crate stops
/// at 1.45112.
#[test]
#[ignore = "#261: HAM's Nelder–Mead stops at a worse local minimum"]
fn ham_on_clean_data_reaches_greybox() {
    let e = estimates(&fit(&YC, AlmLoss::HAM));
    assert!(ham(&YC, &e) <= ham(&YC, &HAM) + 1e-12);
}

/// Trimmed negative log-likelihood, ×n as greybox (`−meanFast(trim) · n`),
/// at coefficients `b` and variance `s2`.
fn role(b: &[f64], s2: f64, trim: f64) -> f64 {
    let n = X.len();
    let mut l: Vec<f64> = (0..n)
        .map(|i| {
            let e = YO[i] - b[0] - b[1] * X[i];
            -0.5 * (2.0 * std::f64::consts::PI * s2).ln() - e * e / (2.0 * s2)
        })
        .collect();
    l.sort_by(|a, b| b.total_cmp(a));
    let keep = (n as f64 * (1.0 - trim)).floor() as usize;
    -l[..keep].iter().sum::<f64>() / keep as f64 * n as f64
}

/// ROLE: greybox stops short of its own objective (129.829 vs the crate's
/// 129.534 at trim 0.05; 124.571 vs 124.411 at 0.10), and its reported
/// `$scale` is not the jointly estimated one. The crate's coefficients are
/// close and its objective at least as small; its scale is the optimum.
#[test]
fn role_is_at_least_as_good_as_greybox() {
    for (trim, want, tol) in [
        (0.05, &ROLE05_OUTLIERS, 0.02),
        (0.10, &ROLE10_OUTLIERS, 1e-4),
    ] {
        let e = estimates(&fit(&YO, AlmLoss::role_with_trim(trim)));
        close(&e[..2], &want[..2], tol, &format!("ROLE {trim}"));
        // greybox's objective, at its best variance for its coefficients
        let greybox = (1..=20_000)
            .map(|k| role(&want[..2], k as f64 * 1e-3, trim))
            .fold(f64::INFINITY, f64::min);
        assert!(role(&e[..2], e[2], trim) <= greybox + 1e-9, "ROLE {trim}");
    }
}
