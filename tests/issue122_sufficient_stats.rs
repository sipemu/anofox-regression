#![allow(clippy::excessive_precision)]

//! #122: mergeable sufficient-statistic states for OLS / WLS / ridge.
//!
//! A fit from a merged state must equal `fit()` on the concatenated rows (to
//! 1e-12 relative) for any split into parts and any merge order, an empty
//! state must be the identity, large offsets must keep the batch precision,
//! and the state must round-trip through its byte encoding.

use anofox_regression::core::{IntervalType, LambdaScaling, RegressionResult};
use anofox_regression::inference::HcType;
use anofox_regression::solvers::{
    FittedRegressor, OlsRegressor, Regressor, RidgeRegressor, SufficientStats, WlsRegressor,
};
use faer::{Col, Mat};

/// Small deterministic generator (no `rand` dependency).
struct Lcg(u64);

impl Lcg {
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 11
    }
    fn uniform(&mut self) -> f64 {
        self.next_u64() as f64 / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        let (u, v) = (self.uniform().max(1e-300), self.uniform());
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

struct Data {
    x: Mat<f64>,
    y: Col<f64>,
    w: Col<f64>,
}

fn data(n: usize, p: usize, seed: u64) -> Data {
    let mut r = Lcg(seed);
    let x = Mat::from_fn(n, p, |_, j| 3.0 * j as f64 + r.normal() * (1.0 + j as f64));
    let y = Col::from_fn(n, |i| {
        1.5 + (0..p).map(|j| (j as f64 - 1.0) * x[(i, j)]).sum::<f64>() + r.normal()
    });
    let w = Col::from_fn(n, |_| 0.2 + 2.0 * r.uniform());
    Data { x, y, w }
}

/// State of `rows` of `d`, pushed row by row.
fn state_of(d: &Data, rows: &[usize], weighted: bool) -> SufficientStats {
    let mut s = SufficientStats::new(d.x.ncols());
    let mut row = vec![0.0; d.x.ncols()];
    for &i in rows {
        for (j, v) in row.iter_mut().enumerate() {
            *v = d.x[(i, j)];
        }
        let w = if weighted { d.w[i] } else { 1.0 };
        s.push_weighted(&row, d.y[i], w).unwrap();
    }
    s
}

/// Split the rows into `k` random parts (some possibly empty), build a state
/// per part (alternating push / two-pass construction), round-trip each
/// through bytes and merge them in a random order and grouping.
fn split_and_merge(d: &Data, k: usize, weighted: bool, r: &mut Lcg) -> SufficientStats {
    let n = d.x.nrows();
    let mut parts: Vec<Vec<usize>> = vec![Vec::new(); k];
    for i in 0..n {
        parts[r.below(k)].push(i);
    }
    let mut states: Vec<SufficientStats> = parts
        .iter()
        .enumerate()
        .map(|(idx, rows)| {
            let s = if idx % 2 == 0 {
                state_of(d, rows, weighted)
            } else {
                let x = Mat::from_fn(rows.len(), d.x.ncols(), |i, j| d.x[(rows[i], j)]);
                let y = Col::from_fn(rows.len(), |i| d.y[rows[i]]);
                let w = Col::from_fn(rows.len(), |i| if weighted { d.w[rows[i]] } else { 1.0 });
                SufficientStats::from_data(&x, &y, weighted.then_some(&w)).unwrap()
            };
            SufficientStats::from_bytes(&s.to_bytes()).unwrap()
        })
        .collect();
    while states.len() > 1 {
        let a = states.swap_remove(r.below(states.len()));
        let b = states.swap_remove(r.below(states.len()));
        states.push(a.merged(&b).unwrap());
    }
    states.pop().unwrap()
}

fn close(a: f64, b: f64, rel: f64) -> bool {
    (a.is_nan() && b.is_nan()) || a == b || (a - b).abs() <= rel * a.abs().max(b.abs()).max(1e-300)
}

fn assert_vec(name: &str, a: Option<&Col<f64>>, b: Option<&Col<f64>>, rel: f64) {
    match (a, b) {
        (None, None) => {}
        (Some(a), Some(b)) => {
            assert_eq!(a.nrows(), b.nrows(), "{name} length");
            for j in 0..a.nrows() {
                assert!(close(a[j], b[j], rel), "{name}[{j}]: {} vs {}", a[j], b[j]);
            }
        }
        _ => panic!("{name}: presence differs"),
    }
}

fn assert_same_fit(state: &RegressionResult, batch: &RegressionResult, rel: f64) {
    assert_eq!(state.aliased, batch.aliased, "aliased");
    assert_eq!(state.rank, batch.rank, "rank");
    assert_eq!(state.n_parameters, batch.n_parameters, "n_parameters");
    assert_eq!(state.n_observations, batch.n_observations, "n_observations");
    assert_vec(
        "coefficients",
        Some(&state.coefficients),
        Some(&batch.coefficients),
        rel,
    );
    let scalars = [
        ("intercept", state.intercept, batch.intercept),
        ("r_squared", Some(state.r_squared), Some(batch.r_squared)),
        (
            "adj_r_squared",
            Some(state.adj_r_squared),
            Some(batch.adj_r_squared),
        ),
        ("mse", Some(state.mse), Some(batch.mse)),
        ("rmse", Some(state.rmse), Some(batch.rmse)),
        (
            "f_statistic",
            Some(state.f_statistic),
            Some(batch.f_statistic),
        ),
        (
            "log_likelihood",
            Some(state.log_likelihood),
            Some(batch.log_likelihood),
        ),
        ("aic", Some(state.aic), Some(batch.aic)),
        ("aicc", Some(state.aicc), Some(batch.aicc)),
        ("bic", Some(state.bic), Some(batch.bic)),
        (
            "intercept_se",
            state.intercept_std_error,
            batch.intercept_std_error,
        ),
        (
            "intercept_t",
            state.intercept_t_statistic,
            batch.intercept_t_statistic,
        ),
    ];
    for (name, a, b) in scalars {
        match (a, b) {
            (Some(a), Some(b)) => assert!(close(a, b, rel), "{name}: {a} vs {b}"),
            (None, None) => {}
            _ => panic!("{name}: presence differs"),
        }
    }
    assert_vec(
        "std_errors",
        state.std_errors.as_ref(),
        batch.std_errors.as_ref(),
        rel,
    );
    assert_vec(
        "t",
        state.t_statistics.as_ref(),
        batch.t_statistics.as_ref(),
        rel,
    );
    assert_vec(
        "ci_lower",
        state.conf_interval_lower.as_ref(),
        batch.conf_interval_lower.as_ref(),
        rel,
    );
    assert_vec(
        "ci_upper",
        state.conf_interval_upper.as_ref(),
        batch.conf_interval_upper.as_ref(),
        rel,
    );
    // Tail probabilities amplify relative rounding by ~t²; compare loosely.
    assert_vec(
        "p_values",
        state.p_values.as_ref(),
        batch.p_values.as_ref(),
        1e-8,
    );
    assert!(close(state.f_pvalue, batch.f_pvalue, 1e-8) || batch.f_pvalue < 1e-300);
    assert_eq!(state.residuals.nrows(), 0, "a state fit keeps no residuals");
}

fn assert_same_intervals(a: &impl FittedRegressor, b: &impl FittedRegressor, x: &Mat<f64>) {
    for kind in [IntervalType::Confidence, IntervalType::Prediction] {
        let pa = a.predict_with_interval(x, Some(kind), 0.9);
        let pb = b.predict_with_interval(x, Some(kind), 0.9);
        assert_vec("se", Some(&pa.se), Some(&pb.se), 1e-11);
        // Bounds can cross zero; compare on the scale of the interval.
        for i in 0..x.nrows() {
            let scale = pb.fit[i].abs() + (pb.upper[i] - pb.lower[i]).abs();
            for (name, a, b) in [
                ("fit", pa.fit[i], pb.fit[i]),
                ("lower", pa.lower[i], pb.lower[i]),
                ("upper", pa.upper[i], pb.upper[i]),
            ] {
                assert!((a - b).abs() <= 1e-12 * scale, "{name}[{i}]: {a} vs {b}");
            }
        }
    }
}

const REL: f64 = 1e-12;

#[test]
fn ols_state_merge_matches_batch_for_random_splits() {
    let d = data(200, 4, 1);
    let mut r = Lcg(99);
    for with_intercept in [true, false] {
        let model = OlsRegressor::builder()
            .with_intercept(with_intercept)
            .compute_inference(true)
            .build();
        let batch = model.fit(&d.x, &d.y).unwrap();
        for k in 1..=7 {
            for _ in 0..3 {
                let s = split_and_merge(&d, k, false, &mut r);
                assert!(!s.is_weighted());
                let fit = model.fit_from_state(&s).unwrap();
                assert_same_fit(fit.result(), batch.result(), REL);
                assert_same_intervals(&fit, &batch, &d.x);
            }
        }
    }
}

#[test]
fn wls_state_merge_matches_batch_including_zero_weights() {
    let mut d = data(150, 3, 2);
    for i in (0..150).step_by(17) {
        d.w[i] = 0.0;
    }
    let mut r = Lcg(7);
    for with_intercept in [true, false] {
        let model = WlsRegressor::builder()
            .with_intercept(with_intercept)
            .compute_inference(true)
            .weights(d.w.clone())
            .build();
        let batch = model.fit(&d.x, &d.y).unwrap();
        for k in 1..=6 {
            let s = split_and_merge(&d, k, true, &mut r);
            assert_eq!(s.n_observations(), 150);
            assert_eq!(s.n_positive(), 150 - 9);
            let fit = WlsRegressor::builder()
                .with_intercept(with_intercept)
                .compute_inference(true)
                .build()
                .fit_from_state(&s)
                .unwrap();
            assert_same_fit(fit.result(), batch.result(), REL);
            assert_same_intervals(&fit, &batch, &d.x);
            // OLS on a weighted state is the WLS fit.
            let ols = OlsRegressor::builder()
                .with_intercept(with_intercept)
                .build()
                .fit_from_state(&s)
                .unwrap();
            assert_same_fit(ols.result(), batch.result(), REL);
        }
    }
}

#[test]
fn ridge_one_state_serves_a_lambda_grid() {
    let d = data(120, 5, 3);
    let mut r = Lcg(11);
    let s = split_and_merge(&d, 4, false, &mut r);
    for with_intercept in [true, false] {
        for scaling in [LambdaScaling::Raw, LambdaScaling::Glmnet] {
            for lambda in [0.0, 1e-3, 0.5, 10.0] {
                let model = RidgeRegressor::builder()
                    .with_intercept(with_intercept)
                    .lambda(lambda)
                    .lambda_scaling(scaling)
                    .compute_inference(true)
                    .build();
                let batch = model.fit(&d.x, &d.y).unwrap();
                let fit = model.fit_from_state(&s).unwrap();
                assert_same_fit(fit.result(), batch.result(), 1e-11);
                assert_same_intervals(&fit, &batch, &d.x);
                let (a, b) = (
                    fit.variance_factor().unwrap(),
                    batch.variance_factor().unwrap(),
                );
                for i in 0..a.nrows() {
                    for j in 0..a.ncols() {
                        assert!(close(a[(i, j)], b[(i, j)], 1e-11), "M[{i},{j}]");
                    }
                }
            }
        }
    }
}

#[test]
fn aliasing_matches_lm() {
    // Columns: x0, 2·x0 (aliased), a constant (aliased with intercept),
    // x3, x0 + x3 (aliased).
    let base = data(80, 2, 4);
    let n = 80;
    let x = Mat::from_fn(n, 5, |i, j| match j {
        0 => base.x[(i, 0)],
        1 => 2.0 * base.x[(i, 0)],
        2 => 7.5,
        3 => base.x[(i, 1)],
        _ => base.x[(i, 0)] + base.x[(i, 1)],
    });
    let d = Data {
        x,
        y: base.y.clone(),
        w: base.w.clone(),
    };
    for with_intercept in [true, false] {
        let model = OlsRegressor::builder()
            .with_intercept(with_intercept)
            .compute_inference(true)
            .build();
        let batch = model.fit(&d.x, &d.y).unwrap();
        let s = state_of(&d, &(0..n).collect::<Vec<_>>(), false);
        let fit = model.fit_from_state(&s).unwrap();
        let expected = if with_intercept {
            vec![false, true, true, false, true]
        } else {
            vec![false, true, false, false, true]
        };
        assert_eq!(fit.result().aliased, expected);
        assert_same_fit(fit.result(), batch.result(), 1e-11);
        assert_same_intervals(&fit, &batch, &d.x);
    }
}

#[test]
fn large_offset_keeps_batch_precision() {
    // x = 1e9 + noise: raw cross-products would lose ~18 digits. The slopes
    // must match a fit on the exactly shifted data.
    let mut r = Lcg(5);
    let n = 500;
    let noise = Mat::from_fn(n, 2, |_, _| r.normal());
    let offset = 1e9;
    let x = Mat::from_fn(n, 2, |i, j| offset + noise[(i, j)]);
    let shifted = Mat::from_fn(n, 2, |i, j| x[(i, j)] - offset);
    let y = Col::from_fn(n, |i| {
        3.0 + 0.5 * shifted[(i, 0)] - 2.0 * shifted[(i, 1)] + r.normal()
    });
    let d = Data {
        x: x.clone(),
        y: y.clone(),
        w: Col::from_fn(n, |_| 1.0),
    };
    // Like R's lm (tol = 1e-7), the default aliasing test drops a column
    // whose centred norm is 1e-9 of its raw norm, in both paths.
    let default = OlsRegressor::builder().build();
    let s = split_and_merge(&d, 3, false, &mut r);
    assert_eq!(
        default.fit_from_state(&s).unwrap().result().aliased,
        vec![true, true]
    );
    assert_eq!(
        default.fit(&x, &y).unwrap().result().aliased,
        vec![true, true]
    );

    // With lm(tol = 1e-12) both columns are kept.
    let model = OlsRegressor::builder()
        .compute_inference(true)
        .alias_tolerance(1e-12)
        .build();
    let reference = model.fit(&shifted, &y).unwrap();
    let batch = model.fit(&x, &y).unwrap();
    let s = split_and_merge(&d, 5, false, &mut r);
    let fit = model.fit_from_state(&s).unwrap();
    for j in 0..2 {
        let (a, b, c) = (
            fit.result().coefficients[j],
            reference.result().coefficients[j],
            batch.result().coefficients[j],
        );
        let batch_err = (c - b).abs();
        assert!(
            (a - b).abs() <= (1e-9 * b.abs()).max(2.0 * batch_err),
            "slope {j}: {a} vs {b}"
        );
        let (sa, sb) = (
            fit.result().std_errors.as_ref().unwrap()[j],
            reference.result().std_errors.as_ref().unwrap()[j],
        );
        assert!(close(sa, sb, 1e-9), "se {j}: {sa} vs {sb}");
    }
    assert!(close(
        fit.result().r_squared,
        reference.result().r_squared,
        1e-9
    ));
    assert!(close(fit.result().mse, reference.result().mse, 1e-9));
}

#[test]
fn row_level_inference_is_refused() {
    let d = data(60, 2, 6);
    let s = state_of(&d, &(0..60).collect::<Vec<_>>(), false);
    let fit = OlsRegressor::builder().build().fit_from_state(&s).unwrap();
    assert!(fit.hc_standard_errors(&d.x, HcType::HC0).is_err());
    assert!(fit.hc_inference(&d.x, HcType::HC3).is_err());
    assert_eq!(fit.result().fitted_values.nrows(), 0);
}

#[test]
fn too_few_rows_and_empty_state_error() {
    let model = OlsRegressor::builder().build();
    assert!(model.fit_from_state(&SufficientStats::new(2)).is_err());
    let mut s = SufficientStats::new(2);
    s.push(&[1.0, 2.0], 1.0).unwrap();
    s.push(&[2.0, 1.0], 2.0).unwrap();
    assert!(model.fit_from_state(&s).is_err());
}

/// R reference (R 4.x):
///
/// ```r
/// i <- 0:11; x1 <- sin(i); x2 <- cos(0.7 * i) * i / 10
/// y <- 1 + 2 * x1 - 3 * x2 + 0.1 * sin(3.3 * i); w <- 1 + (i %% 4)
/// f <- lm(y ~ x1 + x2, weights = w); s <- summary(f)
/// ```
#[test]
fn wls_state_matches_r_lm() {
    let n = 12;
    let x = Mat::from_fn(n, 2, |i, j| {
        let t = i as f64;
        if j == 0 {
            t.sin()
        } else {
            (0.7 * t).cos() * t / 10.0
        }
    });
    let y = Col::from_fn(n, |i| {
        let t = i as f64;
        1.0 + 2.0 * x[(i, 0)] - 3.0 * x[(i, 1)] + 0.1 * (3.3 * t).sin()
    });
    let w = Col::from_fn(n, |i| 1.0 + (i % 4) as f64);
    let mut a = SufficientStats::from_data(
        &Mat::from_fn(5, 2, |i, j| x[(i, j)]),
        &Col::from_fn(5, |i| y[i]),
        Some(&Col::from_fn(5, |i| w[i])),
    )
    .unwrap();
    let mut b = SufficientStats::new(2);
    for i in 5..n {
        b.push_weighted(&[x[(i, 0)], x[(i, 1)]], y[i], w[i])
            .unwrap();
    }
    a.merge(&b).unwrap();
    let fit = WlsRegressor::builder()
        .compute_inference(true)
        .build()
        .fit_from_state(&a)
        .unwrap();
    let r = fit.result();
    let tol = 1e-8;
    assert!(
        close(r.intercept.unwrap(), R_COEF[0], tol),
        "{:?}",
        r.intercept
    );
    assert!(close(r.coefficients[0], R_COEF[1], tol));
    assert!(close(r.coefficients[1], R_COEF[2], tol));
    assert!(close(r.intercept_std_error.unwrap(), R_SE[0], tol));
    let se = r.std_errors.as_ref().unwrap();
    assert!(close(se[0], R_SE[1], tol));
    assert!(close(se[1], R_SE[2], tol));
    assert!(close(r.r_squared, R_R2, tol));
    assert!(close(r.f_statistic, R_F, tol));
    assert!(close(r.log_likelihood, R_LOGLIK, tol));
    assert!(close(r.aic, R_AIC, tol));
}

const R_COEF: [f64; 3] = [0.98163117620587037, 2.0037976499469847, -2.9894386342533945];
const R_SE: [f64; 3] = [
    0.025754419273227136,
    0.036007677920755841,
    0.064980703088864944,
];
const R_R2: f64 = 0.99800020260570876;
const R_F: f64 = 2245.7279545147808;
const R_LOGLIK: f64 = 13.452741804137343;
const R_AIC: f64 = -18.905483608274686;
