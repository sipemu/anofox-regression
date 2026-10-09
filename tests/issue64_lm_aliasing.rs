#![allow(clippy::needless_range_loop)]

//! Issue #64: OLS/WLS alias rank-deficient designs like R's `lm` (LINPACK
//! `dqrdc2`, limited pivoting, tol = 1e-7 relative) for every solver: columns
//! are kept in order and a later column that is a linear combination of the
//! intercept and earlier kept columns gets a NaN coefficient and is flagged
//! aliased. References from `tests/r_scripts/issue64_lm_aliasing.R`.

use anofox_regression::solvers::{FittedRegressor, OlsRegressor, Regressor, WlsRegressor};
use anofox_regression::SolverType;
use faer::{Col, Mat};

const N: usize = 60;
const SOLVERS: [SolverType; 3] = [SolverType::Qr, SolverType::Svd, SolverType::Cholesky];

fn t(i: usize) -> f64 {
    i as f64
}
fn x1() -> Vec<f64> {
    (0..N).map(|i| t(i) / 10.0).collect()
}
fn x2() -> Vec<f64> {
    (0..N).map(|i| t(i).sin()).collect()
}
fn x3() -> Vec<f64> {
    (0..N).map(|i| (t(i) / 3.0).cos()).collect()
}
fn y() -> Col<f64> {
    Col::from_fn(N, |i| {
        2.0 + 1.5 * t(i) / 10.0 - 0.8 * t(i).sin() + 0.3 * (7.0 * t(i)).cos()
    })
}
fn w() -> Col<f64> {
    Col::from_fn(N, |i| 1.0 + (i % 5) as f64 / 2.0)
}
fn lin(a: &[(f64, &Vec<f64>)]) -> Vec<f64> {
    (0..N)
        .map(|i| a.iter().map(|(c, v)| c * v[i]).sum())
        .collect()
}

struct RRef<'a> {
    intercept: Option<f64>,
    coef: &'a [f64], // NaN = NA (aliased)
    se: &'a [f64],   // non-aliased coefficient SEs (excluding intercept)
    sigma: f64,
    r2: f64,
    adj: f64,
    rank: usize, // R's rank (includes intercept)
}

fn check(
    tag: &str,
    cols: &[Vec<f64>],
    weights: Option<Col<f64>>,
    with_intercept: bool,
    r: &RRef,
    tol: f64,
) {
    let x = Mat::from_fn(N, cols.len(), |i, j| cols[j][i]);
    let y = y();
    for solver in SOLVERS {
        // Column scaling must not change the aliasing decision.
        for scale in [1.0, 1e-6, 1e6] {
            let xs = Mat::from_fn(N, cols.len(), |i, j| {
                x[(i, j)] * if j % 2 == 1 { scale } else { 1.0 }
            });
            let result = match &weights {
                None => OlsRegressor::builder()
                    .with_intercept(with_intercept)
                    .solve_method(solver)
                    .build()
                    .fit(&xs, &y)
                    .unwrap()
                    .result()
                    .clone(),
                Some(wv) => WlsRegressor::builder()
                    .with_intercept(with_intercept)
                    .weights(wv.clone())
                    .solve_method(solver)
                    .build()
                    .fit(&xs, &y)
                    .unwrap()
                    .result()
                    .clone(),
            };
            let ctx = format!("{tag} solver={solver:?} scale={scale}");
            let mut k = 0;
            for j in 0..cols.len() {
                let c_scale = if j % 2 == 1 { scale } else { 1.0 };
                if r.coef[j].is_nan() {
                    assert!(result.aliased[j], "{ctx}: column {j} should be aliased");
                    assert!(result.coefficients[j].is_nan(), "{ctx}: coef {j} not NaN");
                } else {
                    assert!(!result.aliased[j], "{ctx}: column {j} wrongly aliased");
                    let b = result.coefficients[j] * c_scale;
                    assert!(
                        (b - r.coef[j]).abs() <= tol * r.coef[j].abs().max(1.0),
                        "{ctx}: coef {j} = {b}, R = {}",
                        r.coef[j]
                    );
                    let se = result.std_errors.as_ref().unwrap()[j] * c_scale;
                    assert!(
                        (se - r.se[k]).abs() <= tol * r.se[k].max(1.0),
                        "{ctx}: se {j} = {se}, R = {}",
                        r.se[k]
                    );
                    k += 1;
                }
            }
            if let Some(b0) = r.intercept {
                let got = result.intercept.unwrap();
                assert!(
                    (got - b0).abs() <= tol * b0.abs().max(1.0),
                    "{ctx}: intercept {got}"
                );
            }
            assert!(
                (result.rmse - r.sigma).abs() < 1e-9,
                "{ctx}: sigma {}",
                result.rmse
            );
            assert!((result.r_squared - r.r2).abs() < 1e-9, "{ctx}: r2");
            assert!((result.adj_r_squared - r.adj).abs() < 1e-9, "{ctx}: adj r2");
            assert_eq!(result.n_parameters, r.rank, "{ctx}: rank");
        }
    }
}

const NA: f64 = f64::NAN;

#[test]
fn exact_duplicate_aliases_later_column() {
    let (a, b) = (x1(), x1());
    check(
        "dup",
        &[a, b],
        None,
        true,
        &RRef {
            intercept: Some(2.00330174321490),
            coef: &[1.48861560234839, NA],
            se: &[0.0442450700963615],
            sigma: 0.593527454881709,
            r2: 0.951259322529991,
            adj: 0.950418966021888,
            rank: 2,
        },
        1e-9,
    );
}

#[test]
fn scaled_copy_aliases_later_column() {
    let a = x1();
    let b = lin(&[(2.0, &a)]);
    check(
        "scaled",
        &[a, b, x2()],
        None,
        true,
        &RRef {
            intercept: Some(2.023318557879104),
            coef: &[1.490290680634032, NA, -0.772169023108856],
            se: &[0.0161213612630192, 0.0396174535357292],
            sigma: 0.216257642550628,
            r2: 0.993640840379574,
            adj: 0.993417711971839,
            rank: 3,
        },
        1e-9,
    );
}

#[test]
fn linear_combination_aliases_last_column() {
    let (a, b) = (x1(), x2());
    let c = lin(&[(1.0, &a), (1.0, &b)]);
    check(
        "lincomb",
        &[b, a, c],
        None,
        true,
        &RRef {
            intercept: Some(2.023318557879102),
            coef: &[-0.772169023108856, 1.490290680634033, NA],
            se: &[0.0396174535357292, 0.0161213612630192],
            sigma: 0.216257642550628,
            r2: 0.993640840379574,
            adj: 0.993417711971839,
            rank: 3,
        },
        1e-9,
    );
}

#[test]
fn combination_of_three_columns() {
    let (a, b, c) = (x1(), x2(), x3());
    let d = lin(&[(0.5, &a), (-2.0, &b), (3.0, &c)]);
    check(
        "lincomb3",
        &[a, d, b, c],
        None,
        true,
        &RRef {
            intercept: Some(2.02308343087296327),
            coef: &[
                1.49135718575485443,
                -0.00179633314960569,
                -0.77548876747366879,
                NA,
            ],
            se: &[0.0180404486690316, 0.0131556408328023, 0.0467775214837986],
            sigma: 0.218143659513665,
            r2: 0.993642956872802,
            adj: 0.993302400990988,
            rank: 4,
        },
        1e-8,
    );
}

/// e: unit vector orthogonal to (1, x1, x2), as in the R script.
fn near_direction() -> Vec<f64> {
    let (a, b) = (x1(), x2());
    let e0: Vec<f64> = (0..N).map(|i| (13.0 * t(i)).sin()).collect();
    let x = Mat::from_fn(N, 2, |i, j| if j == 0 { a[i] } else { b[i] });
    let ec = Col::from_fn(N, |i| e0[i]);
    let f = OlsRegressor::builder().build().fit(&x, &ec).unwrap();
    let r = &f.result().residuals;
    let nr = r.iter().map(|v| v * v).sum::<f64>().sqrt();
    (0..N).map(|i| r[i] / nr).collect()
}

fn near(ratio: f64) -> Vec<Vec<f64>> {
    let (a, b) = (x1(), x2());
    let base = lin(&[(1.0, &a), (1.0, &b)]);
    let nb = base.iter().map(|v| v * v).sum::<f64>().sqrt();
    let e = near_direction();
    let c: Vec<f64> = (0..N).map(|i| base[i] + ratio * nb * e[i]).collect();
    vec![a, b, c]
}

#[test]
fn near_collinear_below_tolerance_is_aliased() {
    check(
        "below",
        &near(0.5e-7),
        None,
        true,
        &RRef {
            intercept: Some(2.023318557879104),
            coef: &[1.490290680634032, -0.772169023108856, NA],
            se: &[0.0161213612630192, 0.0396174535357292],
            sigma: 0.216257642550628,
            r2: 0.993640840379574,
            adj: 0.993417711971839,
            rank: 3,
        },
        1e-9,
    );
}

#[test]
fn near_collinear_above_tolerance_is_kept() {
    for solver in [SolverType::Qr, SolverType::Svd] {
        let cols = near(2e-7);
        let x = Mat::from_fn(N, 3, |i, j| cols[j][i]);
        let f = OlsRegressor::builder()
            .solve_method(solver)
            .build()
            .fit(&x, &y())
            .unwrap();
        let r = f.result();
        assert!(r.aliased.iter().all(|a| !a), "{solver:?}");
        // R: 30169.122326, 30166.859866, -30167.632035 (condition ~1e7).
        let expect = [
            30169.12232610578940,
            30166.85986640204646,
            -30167.63203542516567,
        ];
        for j in 0..3 {
            assert!(
                (r.coefficients[j] - expect[j]).abs() < 1e-4 * expect[j].abs(),
                "{solver:?} coef {j} = {}",
                r.coefficients[j]
            );
        }
        assert!((r.rmse - 0.217068502408162).abs() < 1e-8);
        assert_eq!(r.n_parameters, 4);
    }
}

#[test]
fn wls_scaled_copy_aliases_later_column() {
    let a = x1();
    let b = lin(&[(2.0, &a)]);
    check(
        "wls_scaled",
        &[a, b, x2()],
        Some(w()),
        true,
        &RRef {
            intercept: Some(2.008988048468000),
            coef: &[1.494367216183389, NA, -0.774931009626785],
            se: &[0.0163679078787441, 0.0402659270793710],
            sigma: 0.310386301013356,
            r2: 0.993499086258852,
            adj: 0.99327098402232,
            rank: 3,
        },
        1e-9,
    );
}

#[test]
fn wls_no_intercept_duplicate() {
    let (a, b) = (x1(), x2());
    let c = lin(&[(3.0, &b)]);
    check(
        "wls_dup_noint",
        &[a, b, c],
        Some(w()),
        false,
        &RRef {
            intercept: None,
            coef: &[1.996234614755434, -0.727862022303558, NA],
            se: &[0.0389559048292182, 0.1913613798585877],
            sigma: 1.47589618501478,
            r2: 0.978398095055174,
            adj: 0.977653201781215,
            rank: 2,
        },
        1e-9,
    );
}

#[test]
fn ols_no_intercept_linear_combination() {
    let (a, b) = (x1(), x2());
    let c = lin(&[(1.0, &a), (1.0, &b)]);
    check(
        "noint_lincomb",
        &[b, a, c],
        None,
        false,
        &RRef {
            intercept: None,
            coef: &[-0.745108849562454, 2.000135444454152, NA],
            se: &[0.1947938711280890, 0.0401719767834559],
            sigma: 1.06349509858203,
            r2: 0.977164928142048,
            adj: 0.976377511871084,
            rank: 2,
        },
        1e-9,
    );
}
