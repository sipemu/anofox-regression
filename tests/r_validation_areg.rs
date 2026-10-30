#![allow(clippy::excessive_precision, clippy::large_const_arrays)]

//! #112: additive regression with optimal transformations of both sides
//! (`Hmisc::areg`, RMS §16.1). Reference values come from
//! `tests/r_scripts/generate_areg_validation.R`: `log y` additive in a
//! linear `x1`, a sine of `x2` and a three-level factor.

use anofox_regression::transform::{areg, AregOptions, AregType};
use faer::Mat;

mod fixture {
    include!("fixtures/areg_validation.rs");
}
use fixture::*;

const N: usize = 150;
const P: usize = 3;

fn close(name: &str, a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() <= tol * (1.0 + b.abs()), "{name}: {a} vs {b}");
}

fn all_close(name: &str, a: &[f64], b: &[f64], tol: f64) {
    assert_eq!(a.len(), b.len(), "{name}: length");
    for (k, (u, v)) in a.iter().zip(b).enumerate() {
        close(&format!("{name}[{k}]"), *u, *v, tol);
    }
}

fn x() -> Mat<f64> {
    Mat::from_fn(N, P, |i, j| X[i * P + j])
}

fn smooth() -> AregOptions {
    AregOptions {
        nk: 4,
        x_types: Some(vec![
            AregType::Spline,
            AregType::Spline,
            AregType::Categorical,
        ]),
        y_type: AregType::Spline,
    }
}

fn v(c: &faer::Col<f64>) -> Vec<f64> {
    c.iter().copied().collect()
}

fn tx(m: &Mat<f64>) -> Vec<f64> {
    (0..N * P).map(|k| m[(k / P, k % P)]).collect()
}

#[test]
fn smooth_response_matches_hmisc() {
    let f = areg(&x(), &Y, &smooth()).unwrap();
    all_close("xcoef", &v(&f.x_coefficients), &S_XCOEF, 1e-8);
    all_close("ycoef", &v(&f.y_coefficients), &S_YCOEF, 1e-8);
    close("r2", f.r_squared, S_STATS[0], 1e-10);
    close("mad", f.mad.unwrap(), S_STATS[1], 1e-8);
    close("med", f.med.unwrap(), S_STATS[2], 1e-8);
    all_close("ty", &v(&f.transformed_y), &S_TY, 1e-8);
    all_close("lp", &v(&f.linear_predictors), &S_LP, 1e-8);
    all_close("tx", &tx(&f.transformed_x), &S_TX, 1e-8);
    assert_eq!(f.x_df, vec![3, 3, 2]);
    assert_eq!(f.y_df, 3);
    all_close("y knots", f.y_knots().unwrap(), &S_YKNOTS, 1e-12);
    let res: Vec<f64> = (0..N).map(|i| S_TY[i] - S_LP[i]).collect();
    all_close("residuals", &v(&f.residuals), &res, 1e-8);

    all_close("ytrans", &f.transform_y(&S_YTRANS_AT), &S_YTRANS, 1e-8);
    let inv = f.inverse_y(&S_LP[..8]).unwrap();
    all_close("yinv", &inv, &S_YINV, 1e-8);
    all_close("predict", &f.predict(&x()), &S_LP, 1e-8);
}

#[test]
fn linear_response_is_least_squares() {
    let opts = AregOptions {
        nk: 5,
        x_types: Some(vec![
            AregType::Spline,
            AregType::Linear,
            AregType::Categorical,
        ]),
        y_type: AregType::Linear,
    };
    let f = areg(&x(), &Y, &opts).unwrap();
    all_close("xcoef", &v(&f.x_coefficients), &L_XCOEF, 1e-8);
    // Hmisc reports ycoefficients = 1; ours keep the intercept slot
    assert_eq!(L_YCOEF, [1.0]);
    assert_eq!(v(&f.y_coefficients), vec![0.0, 1.0]);
    close("r2", f.r_squared, L_STATS[0], 1e-10);
    close("mad", f.mad.unwrap(), L_STATS[1], 1e-8);
    close("med", f.med.unwrap(), L_STATS[2], 1e-8);
    all_close("ty", &v(&f.transformed_y), &L_TY, 1e-12);
    all_close("lp", &v(&f.linear_predictors), &L_LP, 1e-8);
    all_close("tx", &tx(&f.transformed_x), &L_TX, 1e-8);
    assert_eq!(f.x_df, vec![4, 1, 2]);
    // a linear response cannot be bootstrapped here
    assert!(f.bootstrap(&[vec![0; N]]).is_err());
}

#[test]
fn bootstrap_optimism_matches_hmisc() {
    let f = areg(&x(), &Y, &smooth()).unwrap();
    let b = BOOT_IDX.len() / N;
    let idx: Vec<Vec<usize>> = (0..b)
        .map(|r| (0..N).map(|i| BOOT_IDX[r * N + i] as usize).collect())
        .collect();
    let boot = f.bootstrap(&idx).unwrap();
    let got = [
        boot.r2_optimism,
        boot.r2_corrected,
        boot.mad_optimism,
        boot.mad_corrected,
        boot.med_optimism,
        boot.med_corrected,
    ];
    all_close("bootstrap", &got, &BOOT_STATS, 1e-8);
    assert!(f.bootstrap(&[]).is_err());
    assert!(f.bootstrap(&[vec![N]]).is_err());
}

#[test]
fn rejects_bad_input() {
    let x = x();
    assert!(areg(&x, &Y[..N - 1], &AregOptions::default()).is_err());
    let wrong_types = AregOptions {
        x_types: Some(vec![AregType::Spline]),
        ..AregOptions::default()
    };
    assert!(areg(&x, &Y, &wrong_types).is_err());
    let cat_y = AregOptions {
        y_type: AregType::Categorical,
        ..AregOptions::default()
    };
    assert!(areg(&x, &Y, &cat_y).is_err());
    let constant = Mat::from_fn(N, 1, |_, _| 1.0);
    let one_level = AregOptions {
        x_types: Some(vec![AregType::Categorical]),
        ..AregOptions::default()
    };
    assert!(areg(&constant, &Y, &one_level).is_err());
    // nk = 0: every spline is linear
    let lin = AregOptions {
        nk: 0,
        ..AregOptions::default()
    };
    let f = areg(&x, &Y, &lin).unwrap();
    assert_eq!((f.x_df.clone(), f.y_df), (vec![1, 1, 1], 1));
}

#[test]
fn non_monotone_response_transformation_has_no_inverse() {
    // x = (y − 5.5)²: the optimal transformation of y is U-shaped
    let n = 60;
    let y: Vec<f64> = (0..n)
        .map(|i| 1.0 + 9.0 * i as f64 / (n - 1) as f64)
        .collect();
    let x = Mat::from_fn(n, 1, |i, _| {
        (y[i] - 5.5).powi(2) + 0.01 * ((i * 7) % 5) as f64
    });
    let f = areg(&x, &y, &AregOptions::default()).unwrap();
    assert!(f.r_squared > 0.9);
    assert!(f.inverse_y(&[0.0]).is_none());
    assert!(f.mad.is_none() && f.med.is_none());
    assert!(f.bootstrap(&[(0..n).collect()]).is_err());
}
