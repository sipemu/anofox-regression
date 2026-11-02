//! Issue #309: a public B-spline basis reproducing scikit-learn's
//! `SplineTransformer` (uniform / quantile / explicit knots, every
//! extrapolation mode, with and without the bias column).
//!
//! Reference values: `validation/python/generate_spline_transformer_validation.py`.

use anofox_regression::splines::{BSpline, KnotPlacement, SplineError, SplineExtrapolation};

include!("fixtures/spline_transformer_validation.rs");

fn extrapolation(name: &str) -> SplineExtrapolation {
    match name {
        "constant" => SplineExtrapolation::Constant,
        "linear" => SplineExtrapolation::Linear,
        "continue" => SplineExtrapolation::Continue,
        "periodic" => SplineExtrapolation::Periodic,
        _ => SplineExtrapolation::Error,
    }
}

fn spline(n_knots: usize, degree: usize, knots: u8) -> BSpline {
    match knots {
        0 => BSpline::from_data(&X_TRAIN, n_knots, degree, KnotPlacement::Uniform),
        1 => BSpline::from_data(&X_TRAIN, n_knots, degree, KnotPlacement::Quantile),
        _ => BSpline::new(&EXPLICIT_KNOTS, degree),
    }
    .unwrap()
}

#[test]
fn matches_sklearn_spline_transformer() {
    assert!(CASES.len() > 100);
    for &(n_knots, degree, knots, extrap, bias, n_cols, want) in CASES {
        let what = format!("n_knots={n_knots} degree={degree} knots={knots} {extrap} bias={bias}");
        let s = spline(n_knots, degree, knots)
            .with_extrapolation(extrapolation(extrap))
            .with_bias(bias);
        assert_eq!(s.n_columns(), n_cols, "{what}: columns");
        let q: Vec<f64> = if extrap == "error" {
            let t = s.knots();
            let (lo, hi) = (t[degree], t[t.len() - degree - 1]);
            let mut inside: Vec<f64> = X_QUERY
                .iter()
                .copied()
                .filter(|v| lo <= *v && *v <= hi)
                .collect();
            inside.resize(X_QUERY.len(), lo);
            inside
        } else {
            X_QUERY.to_vec()
        };
        let b = s.basis(&q, "x").unwrap();
        assert_eq!(b.columns.nrows(), q.len());
        for i in 0..q.len() {
            for j in 0..n_cols {
                let (got, w) = (b.columns[(i, j)], want[i * n_cols + j]);
                assert!(
                    (got - w).abs() < 1e-12,
                    "{what}: x={} col {j}: {got} vs sklearn {w}",
                    q[i]
                );
            }
        }
    }
}

#[test]
fn knot_vectors_and_names() {
    let s = BSpline::new(&[0.0, 1.0, 3.0], 2).unwrap();
    assert_eq!(s.knots(), vec![-2.0, -1.0, 0.0, 1.0, 3.0, 5.0, 7.0]);
    let p = s.clone().with_extrapolation(SplineExtrapolation::Periodic);
    assert_eq!(p.knots(), vec![-3.0, -2.0, 0.0, 1.0, 3.0, 4.0, 6.0]);
    assert_eq!(p.n_columns(), 2);
    let b = s.with_bias(false).basis(&[0.5, f64::NAN], "age").unwrap();
    assert_eq!(b.names, ["age_sp_0", "age_sp_1", "age_sp_2"]);
    assert!(b.nonlinear.iter().all(|f| *f));
    assert!(b.columns[(1, 0)].is_nan());
}

#[test]
fn errors() {
    assert_eq!(
        BSpline::new(&[1.0], 3),
        Err(SplineError::TooFewBaseKnots(1))
    );
    assert_eq!(
        BSpline::new(&[0.0, 2.0, 1.0], 3),
        Err(SplineError::UnsortedKnots)
    );
    assert_eq!(
        BSpline::new(&[0.0, f64::INFINITY], 3),
        Err(SplineError::NonFiniteKnots)
    );
    assert_eq!(
        BSpline::from_data(&[2.0; 10], 4, 3, KnotPlacement::Uniform),
        Err(SplineError::UnsortedKnots)
    );
    assert_eq!(
        BSpline::from_data(&[f64::NAN], 4, 3, KnotPlacement::Uniform),
        Err(SplineError::TooFewObservations)
    );
    let s = BSpline::new(&[0.0, 1.0, 2.0], 3).unwrap();
    assert_eq!(
        s.clone()
            .with_extrapolation(SplineExtrapolation::Periodic)
            .basis(&[0.5], "x"),
        Err(SplineError::PeriodicDegree {
            degree: 3,
            n_knots: 3
        })
    );
    assert_eq!(
        s.with_extrapolation(SplineExtrapolation::Error)
            .basis(&[2.5], "x"),
        Err(SplineError::OutsideKnots(2.5))
    );
}
