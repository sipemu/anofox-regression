//! #179: vector accessors return faer columns — views of stored values,
//! owned columns for computed ones — with one entry per coefficient,
//! observation or group.

use anofox_regression::solvers::penalized_glm::{PenalizedGlmFamily, PenalizedGlmRegressor};
use anofox_regression::solvers::*;
use faer::{Col, Mat};

const N: usize = 48;

fn x() -> Mat<f64> {
    Mat::from_fn(N, 2, |i, j| ((i * (j + 3)) as f64 * 0.41).sin())
}

#[test]
fn glmm_accessors() {
    let x = x();
    let y = Col::from_fn(N, |i| {
        1.0 + x[(i, 0)] + (i % 6) as f64 * 0.3 + ((i * 7) % 5) as f64 / 10.0
    });
    let g: Vec<usize> = (0..N).map(|i| i % 6).collect();
    let f = GlmmRegressor::gaussian().build().fit(&x, &y, &g).unwrap();
    assert_eq!(f.fixed_effects().nrows(), 3);
    assert_eq!(f.std_errors().nrows(), 3);
    assert_eq!(f.slopes().nrows(), 2);
    assert_eq!(f.z_values().nrows(), 3);
    assert_eq!(f.p_values().nrows(), 3);
    assert_eq!(f.random_effects().nrows(), 6);
    assert_eq!(f.random_effects_se().nrows(), 6);
    assert_eq!(f.random_sd().nrows(), 1);
    assert_eq!(f.fitted_values().nrows(), N);
    assert_eq!(f.fitted_population().nrows(), N);
    assert_eq!(f.residuals().nrows(), N);
    assert_eq!(f.pearson_residuals().nrows(), N);
    let r = f.residuals();
    assert!((0..N).all(|i| (r[i] - (y[i] - f.fitted_values()[i])).abs() < 1e-12));
}

#[test]
fn other_accessors() {
    let x = x();
    let y = Col::from_fn(N, |i| {
        1.0 + x[(i, 0)] - 0.5 * x[(i, 1)] + ((i * 7919) % 11) as f64 / 10.0
    });
    let counts = Col::from_fn(N, |i| {
        ((0.4 + 0.6 * x[(i, 0)]).exp() + (i % 3) as f64).floor()
    });
    let p = PenalizedGlmRegressor::builder()
        .family(PenalizedGlmFamily::poisson())
        .build();
    assert_eq!(p.fit(&x, &counts).unwrap().fitted_values().nrows(), N);
    let br = BayesianRidgeRegressor::builder()
        .build()
        .fit(&x, &y)
        .unwrap();
    assert_eq!(br.sigma_diag().nrows(), 2);
    let ard = ArdRegressor::builder().build().fit(&x, &y).unwrap();
    assert!(ard.sigma_diag().nrows() <= 2);
    assert_eq!(ard.lambdas().nrows(), 2);
    let lars = LarsRegressor::builder().build().fit(&x, &y).unwrap();
    assert!(lars.alphas().nrows() >= 2);
    let mut s = SufficientStats::new(2);
    for i in 0..N {
        s.push(&[x[(i, 0)], x[(i, 1)]], y[i]);
    }
    assert_eq!(s.mean_x().nrows(), 2);
    assert_eq!(s.centered_xy().nrows(), 2);
}
