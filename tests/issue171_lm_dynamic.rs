//! #171: `LmDynamicRegressor` against OLS on anofox-forecast's design (an
//! intercept column, a trend and two Fourier pairs, plus the regressor's
//! own intercept).
//!
//! On the noise-free series `y = 2t + 10 sin(2πt/7) + 5` several candidate
//! models fit exactly, so the IC-weighted fit and OLS both forecast the
//! truth and coincide. They differed before 0.5.13 only because OLS gave
//! wrong coefficients there (the column-pivot unpermute bug fixed in
//! 0.5.13, off by up to 18.7 in these forecasts). With noise the dynamic
//! fit must differ from OLS.

use anofox_regression::solvers::lm_dynamic::{InformationCriterion, LmDynamicRegressor};
use anofox_regression::solvers::{FittedRegressor, OlsRegressor, Regressor};
use faer::{Col, Mat};

const W: f64 = 2.0 * std::f64::consts::PI / 7.0;

fn design(n: usize) -> Mat<f64> {
    Mat::from_fn(n, 6, |i, j| {
        let t = i as f64;
        match j {
            0 => 1.0,
            1 => t,
            2 => (W * t).sin(),
            3 => (W * t).cos(),
            4 => (2.0 * W * t).sin(),
            _ => (2.0 * W * t).cos(),
        }
    })
}

fn truth(t: f64) -> f64 {
    2.0 * t + 10.0 * (W * t).sin() + 5.0
}

/// Forecasts of the dynamic fit and of OLS for the 14 steps after `n = 200`.
fn forecasts(noise: f64) -> (Vec<f64>, Vec<f64>) {
    let n = 200;
    let x = design(n);
    // deterministic pseudo-noise in [-1, 1)
    let y = Col::from_fn(n, |i| {
        truth(i as f64) + noise * (((i * 7919) % 1000) as f64 / 500.0 - 1.0)
    });
    let dynamic = LmDynamicRegressor::builder()
        .with_intercept(true)
        .ic(InformationCriterion::AICc)
        .lowess_span(0.3)
        .build()
        .fit(&x, &y)
        .unwrap();
    let ols = OlsRegressor::builder()
        .with_intercept(true)
        .build()
        .fit(&x, &y)
        .unwrap();
    let xf = design(n + 14);
    let xh = Mat::from_fn(14, 6, |i, j| xf[(n + i, j)]);
    let (a, b) = (dynamic.predict(&xh), ols.predict(&xh));
    (
        (0..14).map(|i| a[i]).collect(),
        (0..14).map(|i| b[i]).collect(),
    )
}

#[test]
fn exact_series_both_forecast_the_truth() {
    let (d, o) = forecasts(0.0);
    for (k, (a, b)) in d.iter().zip(&o).enumerate() {
        let t = truth((200 + k) as f64);
        assert!((a - t).abs() < 1e-8, "dynamic {k}: {a} vs {t}");
        assert!((b - t).abs() < 1e-8, "ols {k}: {b} vs {t}");
    }
}

#[test]
fn noisy_series_dynamic_differs_from_ols() {
    let (d, o) = forecasts(3.0);
    let diff: f64 = d
        .iter()
        .zip(&o)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt();
    assert!(
        diff > 1e-6,
        "dynamic collapsed to OLS on a noisy series: {diff}"
    );
}
