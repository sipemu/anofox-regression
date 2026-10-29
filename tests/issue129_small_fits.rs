//! #129: the deferred (X'X)⁻¹ of a fit without inference gives the same
//! variance factor and intervals as the one computed during a fit with
//! inference.

use anofox_regression::core::IntervalType;
use anofox_regression::solvers::{FittedRegressor, OlsRegressor, Regressor};
use faer::{Col, Mat};

#[test]
fn deferred_factor_matches_eager_factor() {
    let x = Mat::from_fn(25, 3, |i, j| {
        ((i * 7 + j * 13) % 11) as f64 + 0.3 * (i as f64 * 0.7 + j as f64).sin()
    });
    let y = Col::from_fn(25, |i| {
        1.0 + 0.5 * x[(i, 0)] - 0.2 * x[(i, 2)] + (i as f64 * 1.3).cos()
    });
    for with_intercept in [true, false] {
        let eager = OlsRegressor::builder()
            .with_intercept(with_intercept)
            .compute_inference(true)
            .build()
            .fit(&x, &y)
            .unwrap();
        let lazy = OlsRegressor::builder()
            .with_intercept(with_intercept)
            .compute_inference(false)
            .build()
            .fit(&x, &y)
            .unwrap();
        let (a, b) = (
            eager.variance_factor().unwrap(),
            lazy.variance_factor().unwrap(),
        );
        for i in 0..a.nrows() {
            for j in 0..a.ncols() {
                assert!((a[(i, j)] - b[(i, j)]).abs() <= 1e-12 * a[(i, j)].abs().max(1e-12));
            }
        }
        for kind in [IntervalType::Confidence, IntervalType::Prediction] {
            let (pa, pb) = (
                eager.predict_with_interval(&x, Some(kind), 0.9),
                lazy.predict_with_interval(&x, Some(kind), 0.9),
            );
            for i in 0..x.nrows() {
                assert!(
                    (pa.lower[i] - pb.lower[i]).abs() < 1e-11
                        && (pa.upper[i] - pb.upper[i]).abs() < 1e-11
                );
            }
        }
        assert!(lazy.result().std_errors.is_none());
    }
}

#[test]
fn alias_test_handles_an_empty_design() {
    let x = Mat::<f64>::zeros(0, 2);
    assert_eq!(
        anofox_regression::utils::lm_alias_columns(&x, None, &[false, false], 1e-7),
        vec![true, true]
    );
}
