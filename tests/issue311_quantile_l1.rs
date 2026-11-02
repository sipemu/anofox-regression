//! Issue #311: L1 penalty (`alpha`) for `QuantileRegressor`, against
//! scikit-learn's `QuantileRegressor(solver="highs")`.
//!
//! Reference values: `validation/python/generate_quantile_l1_validation.py`.
//! The LP can have several optimal vertices, so the objective is compared
//! first; the coefficients are compared where the optimum is unique.

use anofox_regression::solvers::{FittedRegressor, IterativeFit, QuantileRegressor, Regressor};
use faer::{Col, Mat};

include!("fixtures/quantile_l1_validation.rs");

const N: usize = 50;
const P: usize = 4;

fn x() -> Mat<f64> {
    Mat::from_fn(N, P, |i, j| X[i * P + j])
}

/// sklearn's objective `(1/Σw) Σ wᵢ ρ_τ(rᵢ) + alpha ‖β‖₁`.
fn objective(tau: f64, alpha: f64, w: &[f64], coef: &[f64], b0: f64) -> f64 {
    let x = x();
    let mut loss = 0.0;
    for i in 0..N {
        let r = Y[i] - b0 - (0..P).map(|j| x[(i, j)] * coef[j]).sum::<f64>();
        loss += w[i] * if r >= 0.0 { tau * r } else { (tau - 1.0) * r };
    }
    loss / w.iter().sum::<f64>() + alpha * coef.iter().map(|c| c.abs()).sum::<f64>()
}

#[test]
fn l1_quantile_matches_sklearn() {
    let ones = [1.0; N];
    for &(tau, alpha, intercept, weighted, coef, b0, obj) in CASES {
        let what = format!("tau {tau} alpha {alpha} intercept {intercept} weighted {weighted}");
        let w: &[f64] = if weighted { &W } else { &ones };
        let mut b = QuantileRegressor::builder()
            .tau(tau)
            .alpha(alpha)
            .with_intercept(intercept);
        if weighted {
            b = b.weights(Col::from_fn(N, |i| W[i]));
        }
        let f = b.build().fit(&x(), &Col::from_fn(N, |i| Y[i])).unwrap();
        assert!(f.converged(), "{what}: not certified optimal");
        let r = f.result();
        let got: Vec<f64> = r.coefficients.iter().copied().collect();
        let got_b0 = r.intercept.unwrap_or(0.0);
        let ours = objective(tau, alpha, w, &got, got_b0);
        assert!(
            (ours - obj).abs() < 1e-9 * obj.max(1.0),
            "{what}: objective {ours} vs sklearn {obj}"
        );
        // Exact zeros where sklearn has them, the same values elsewhere
        // (both are vertices of the same LP).
        for (j, (g, s)) in got.iter().zip(coef).enumerate() {
            if *s == 0.0 {
                assert_eq!(*g, 0.0, "{what}: coef[{j}] should be exactly 0, got {g}");
            }
            assert!((g - s).abs() < 1e-8, "{what}: coef[{j}] {g} vs sklearn {s}");
        }
        assert!(
            (got_b0 - b0).abs() < 1e-8,
            "{what}: intercept {got_b0} vs sklearn {b0}"
        );
        assert_eq!(f.alpha(), alpha);
    }
}

#[test]
fn penalized_fit_rejects_inference_and_bad_alpha() {
    use anofox_regression::solvers::quantile_inference::QuantileSe;
    let (xm, y) = (x(), Col::from_fn(N, |i| Y[i]));
    let f = QuantileRegressor::builder()
        .alpha(0.05)
        .build()
        .fit(&xm, &y)
        .unwrap();
    assert!(f.inference(&xm, &y, QuantileSe::Iid, 0.95).is_err());
    for bad in [-1.0, f64::NAN] {
        assert!(QuantileRegressor::builder()
            .alpha(bad)
            .build()
            .fit(&xm, &y)
            .is_err());
    }
}
