//! Issue #310: scikit-learn's joint coefficient / scale Huber estimator
//! (`HuberScale::Joint`), against `sklearn.linear_model.HuberRegressor`
//! (L-BFGS-B, `max_iter=100000`, `tol=1e-12`) on the first 120 rows of the
//! diabetes data and on a heavy-tailed synthetic set with outliers.
//!
//! Reference values: `validation/python/generate_huber_joint_validation.py`.
//! The objective is jointly convex, so the optimum is unique; ours reaches
//! an objective at least as low as sklearn's.

use anofox_regression::solvers::{
    FittedRegressor, HuberRegressor, HuberScale, IterativeFit, Regressor,
};
use faer::{Col, Mat};

include!("fixtures/huber_joint_validation.rs");

fn data(which: u8) -> (Mat<f64>, Col<f64>) {
    let (x, y, p): (&[f64], &[f64], usize) = if which == 0 {
        (&X_DIAB, &Y_DIAB, 10)
    } else {
        (&X_SYN, &Y_SYN, 2)
    };
    let n = y.len();
    (
        Mat::from_fn(n, p, |i, j| x[i * p + j]),
        Col::from_fn(n, |i| y[i]),
    )
}

/// sklearn's objective at `(coef, intercept, scale)`.
fn objective(
    x: &Mat<f64>,
    y: &Col<f64>,
    eps: f64,
    alpha: f64,
    coef: &[f64],
    b0: f64,
    s: f64,
) -> f64 {
    let mut f = y.nrows() as f64 * s + alpha * coef.iter().map(|c| c * c).sum::<f64>();
    for i in 0..y.nrows() {
        let r = y[i] - b0 - (0..coef.len()).map(|j| x[(i, j)] * coef[j]).sum::<f64>();
        f += if r.abs() <= eps * s {
            r * r / s
        } else {
            2.0 * eps * r.abs() - eps * eps * s
        };
    }
    f
}

/// Gradient of sklearn's objective in `(b0, coef, scale)`.
fn gradient(
    x: &Mat<f64>,
    y: &Col<f64>,
    eps: f64,
    alpha: f64,
    coef: &[f64],
    b0: f64,
    s: f64,
) -> Vec<f64> {
    let p = coef.len();
    let mut g = vec![0.0; p + 2];
    g[p + 1] = y.nrows() as f64;
    for i in 0..y.nrows() {
        let r = y[i] - b0 - (0..p).map(|j| x[(i, j)] * coef[j]).sum::<f64>();
        let dr = if r.abs() <= eps * s {
            g[p + 1] -= r * r / (s * s);
            -2.0 * r / s
        } else {
            g[p + 1] -= eps * eps;
            -2.0 * eps * r.signum()
        };
        g[0] += dr;
        for j in 0..p {
            g[j + 1] += dr * x[(i, j)];
        }
    }
    for j in 0..p {
        g[j + 1] += 2.0 * alpha * coef[j];
    }
    g
}

#[test]
fn joint_estimator_matches_sklearn() {
    for &(which, eps, alpha, intercept, coef, b0, scale, n_out, obj) in CASES {
        let what = format!("data {which} eps {eps} alpha {alpha} intercept {intercept}");
        let (x, y) = data(which);
        let f = HuberRegressor::builder()
            .epsilon(eps)
            .alpha(alpha)
            .with_intercept(intercept)
            .scale_method(HuberScale::Joint)
            .tolerance(1e-10)
            .build()
            .fit(&x, &y)
            .unwrap();
        assert!(f.converged(), "{what}: not converged");
        let r = f.result();
        let got: Vec<f64> = r.coefficients.iter().copied().collect();
        let got_b0 = r.intercept.unwrap_or(0.0);

        // The optimum: at least as low as sklearn's objective, zero gradient.
        let ours = objective(&x, &y, eps, alpha, &got, got_b0, f.scale());
        assert!(
            ours <= obj * (1.0 + 1e-12),
            "{what}: objective {ours} vs sklearn {obj}"
        );
        let g = gradient(&x, &y, eps, alpha, &got, got_b0, f.scale());
        let g = if intercept { &g[..] } else { &g[1..] };
        assert!(g.iter().all(|v| v.abs() < 1e-6), "{what}: gradient {g:?}");

        let rel = |a: f64, b: f64| (a - b).abs() / b.abs().max(1.0);
        if which == 1 {
            // Well-conditioned data: sklearn converges, parameters agree.
            for (j, (u, w)) in got.iter().zip(coef).enumerate() {
                assert!(rel(*u, *w) < 5e-5, "{what}: coef[{j}] {u} vs sklearn {w}");
            }
            assert!(rel(got_b0, b0) < 5e-5, "{what}: intercept {got_b0} vs {b0}");
            assert!(
                rel(f.scale(), scale) < 5e-5,
                "{what}: scale {} vs {scale}",
                f.scale()
            );
            assert_eq!(f.n_outliers(), n_out, "{what}: outliers");
        } else {
            // Diabetes (features of scale 0.05): the objective is flat in the
            // coefficients and L-BFGS-B stops short of the optimum (its
            // objective is higher), so the fits agree in predictions and
            // scale rather than digit for digit.
            let sd = {
                let m = y.iter().sum::<f64>() / y.nrows() as f64;
                (y.iter().map(|v| (v - m).powi(2)).sum::<f64>() / y.nrows() as f64).sqrt()
            };
            for i in 0..y.nrows() {
                let fo = got_b0 + (0..got.len()).map(|j| x[(i, j)] * got[j]).sum::<f64>();
                let fs = b0 + (0..coef.len()).map(|j| x[(i, j)] * coef[j]).sum::<f64>();
                assert!(
                    (fo - fs).abs() < 5e-3 * sd,
                    "{what}: prediction {i}: {fo} vs {fs}"
                );
            }
            assert!(
                rel(f.scale(), scale) < 1e-3,
                "{what}: scale {} vs {scale}",
                f.scale()
            );
        }
    }
}

#[test]
fn mad_scale_stays_the_default() {
    let (x, y) = data(1);
    let mad = HuberRegressor::builder().build().fit(&x, &y).unwrap();
    let explicit = HuberRegressor::builder()
        .scale_method(HuberScale::Mad)
        .build()
        .fit(&x, &y)
        .unwrap();
    assert_eq!(mad.scale(), explicit.scale());
    let joint = HuberRegressor::builder()
        .scale_method(HuberScale::Joint)
        .build()
        .fit(&x, &y)
        .unwrap();
    assert!((joint.scale() - mad.scale()).abs() > 1e-3);
}
