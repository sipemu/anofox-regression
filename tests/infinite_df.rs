//! 0.12: an infinite residual d.f. is the normal reference (it used to hang
//! in statrs' `StudentsT::inverse_cdf`).

use anofox_regression::inference::CoefficientInference;
use faer::Col;
use std::sync::mpsc;
use std::time::Duration;

#[test]
fn infinite_df_uses_normal_quantiles_and_does_not_hang() {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let b = Col::from_fn(2, |j| [1.0, -0.5][j]);
        let se = Col::from_fn(2, |j| [0.5, 0.25][j]);
        let (lo, hi) = CoefficientInference::confidence_intervals(&b, &se, f64::INFINITY, 0.95);
        let t = CoefficientInference::t_statistics(&b, &se);
        let p = CoefficientInference::p_values(&t, f64::INFINITY);
        tx.send((lo[0], hi[0], p[0])).unwrap();
    });
    let (lo, hi, p) = rx
        .recv_timeout(Duration::from_secs(20))
        .expect("confidence_intervals / p_values with df = inf must not hang");
    // R: 1 + c(-1, 1) * qnorm(0.975) * 0.5; 2 * pnorm(-2)
    let z = 1.959963984540054;
    assert!((lo - (1.0 - z * 0.5)).abs() < 1e-9 && (hi - (1.0 + z * 0.5)).abs() < 1e-9);
    assert!((p - 0.04550026389635842).abs() < 1e-10, "{p}");
}
