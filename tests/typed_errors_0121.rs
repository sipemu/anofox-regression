#![allow(clippy::excessive_precision, dead_code)]

//! 0.12.1: typed errors for degenerate data (instead of message text), the
//! categorical level check of `ModelSpec::design`, and AFT models rebuilt
//! from stored parts.

use anofox_regression::design::DesignError;
use anofox_regression::effects::ModelSpec;
use anofox_regression::inference::ClusterOptions;
use anofox_regression::reduction::{
    principal_components, spearman2, spearman2_categorical, transcan, varclus, Linkage, Similarity,
    SimilarityTransform, TranscanOptions,
};
use anofox_regression::solvers::aft::{fit_aft, AftDistribution, AftOptions, AftResult, FittedAft};
use anofox_regression::solvers::{DegenerateData, RegressionError};
use anofox_regression::splines::SplineError;
use anofox_regression::transform::{areg, box_cox, AregOptions, BoxCoxOptions};
use faer::{Col, Mat};

include!("fixtures/cluster_validation.rs");

fn degenerate(r: Result<impl std::fmt::Debug, RegressionError>) -> DegenerateData {
    match r {
        Err(RegressionError::Degenerate(d)) => d,
        other => panic!("expected a Degenerate error, got {other:?}"),
    }
}

#[test]
fn reduction_and_transform_errors_are_typed() {
    let x: Vec<f64> = (0..20).map(|i| i as f64).collect();
    let flat = vec![3.0; 20];
    assert_eq!(
        degenerate(spearman2(&x, &flat, 1)),
        DegenerateData::ConstantResponse
    );
    assert_eq!(
        degenerate(spearman2_categorical(&[1; 20], &x)),
        DegenerateData::TooFewLevels
    );
    let m = Mat::from_fn(20, 3, |i, j| {
        if j == 1 {
            5.0
        } else {
            (i * (j + 2) % 7) as f64
        }
    });
    assert_eq!(
        degenerate(varclus(
            &m,
            Similarity::Spearman,
            SimilarityTransform::Square,
            Linkage::Average
        )),
        DegenerateData::SimilarityNotComputable
    );
    assert_eq!(
        degenerate(principal_components(&m, true)),
        DegenerateData::ConstantColumn
    );
    let mut sparse = Mat::from_fn(20, 2, |i, j| (i + j) as f64);
    for i in 2..20 {
        sparse[(i, 0)] = f64::NAN;
    }
    assert_eq!(
        degenerate(transcan(
            &sparse,
            &[false, false],
            &TranscanOptions::default()
        )),
        DegenerateData::TooFewObservedValues
    );
    let y = Col::from_fn(20, |i| if i == 3 { -1.0 } else { 1.0 + i as f64 });
    let xb = Mat::from_fn(20, 1, |i, _| i as f64);
    assert_eq!(
        degenerate(box_cox(&xb, &y, &BoxCoxOptions::default())),
        DegenerateData::NonPositiveResponse
    );
}

#[test]
fn areg_with_too_few_rows_for_spline_knots_is_a_spline_error() {
    // 5 rows: default knots need 6 non-missing observations
    let x = Mat::from_fn(5, 1, |i, _| [0.3, 1.1, 2.0, 2.9, 4.2][i]);
    let y = [1.0, 2.5, 2.0, 4.0, 5.5];
    let err = areg(&x, &y, &AregOptions::default().nk(3)).unwrap_err();
    assert!(
        matches!(
            err,
            RegressionError::Spline(SplineError::TooFewObservations)
        ),
        "{err:?}"
    );
}

#[test]
fn model_spec_design_rejects_unknown_levels() {
    let mut s = ModelSpec::new();
    s.linear("x", &[1.0, 2.0, 3.0, 4.0]);
    s.categorical("g", &[1.0, 2.0, 2.0, 3.0]);
    assert!(s.design(&[&[1.0, 2.0], &[1.0, 3.0]]).is_ok());
    let err = s.design(&[&[1.0, 2.0], &[1.0, 4.0]]).unwrap_err();
    assert_eq!(
        err,
        DesignError::UnknownLevel {
            predictor: "g".into(),
            value: 4.0
        }
    );
    assert!(matches!(
        s.design(&[&[1.0], &[f64::NAN]]),
        Err(DesignError::UnknownLevel { .. })
    ));
    assert!(matches!(s.design(&[&[1.0]]), Err(DesignError::Invalid(_))));
}

fn columns() -> Vec<Vec<f64>> {
    vec![X1.to_vec(), X2.to_vec()]
}

#[test]
fn aft_from_parts_predicts_as_the_fit() {
    let x = columns();
    for dist in [AftDistribution::Weibull, AftDistribution::Exponential] {
        let opts = AftOptions::default()
            .dist(dist)
            .compute_inference(true)
            .tolerance(1e-12);
        let fit = fit_aft(&TIME, &x, &EV, &opts).unwrap();
        let vcov = fit.inference.as_ref().unwrap().vcov.clone().unwrap();
        let rebuilt = AftResult::from_parts(
            dist,
            fit.core.intercept,
            fit.core.coefficients.clone(),
            fit.core.scale,
            Some(&vcov),
        )
        .unwrap();
        let row = [0.4, 0.7];
        let times = [0.5, 1.0, 3.0, 7.0];
        assert_eq!(
            rebuilt.survival_curve(&row, &times),
            fit.survival_curve(&row, &times)
        );
        assert_eq!(
            rebuilt.quantiles(&row, &[0.1, 0.5, 0.9]),
            fit.quantiles(&row, &[0.1, 0.5, 0.9])
        );
        let a = fit.inference.as_ref().unwrap();
        let b = rebuilt.inference.as_ref().unwrap();
        for (u, v) in a.std_errors.iter().zip(&b.std_errors) {
            assert!((u - v).abs() < 1e-12 * u.abs());
        }
        for (u, v) in a.p_values.iter().zip(&b.p_values) {
            assert!((u - v).abs() < 1e-10, "{u} vs {v}");
        }
        let v0 = fit
            .cluster_vcov(&TIME, &x, &EV, &[&G1], &ClusterOptions::robcov())
            .unwrap();
        let v1 = rebuilt
            .cluster_vcov(&TIME, &x, &EV, &[&G1], &ClusterOptions::robcov())
            .unwrap();
        for r in 0..v0.nrows() {
            for c in 0..v0.ncols() {
                assert!((v0[(r, c)] - v1[(r, c)]).abs() < 1e-9 * v0[(r, r)].abs().max(1e-12));
            }
        }
        // FittedAft from the parts, without a covariance
        let fitted = FittedAft::from_parts(
            dist,
            fit.core.intercept,
            fit.core.coefficients.clone(),
            fit.core.scale,
            None,
        )
        .unwrap();
        let xm = Mat::from_fn(1, 2, |_, j| row[j]);
        let q = fitted.predict_median(&xm)[0];
        assert_eq!(q, fit.quantiles(&row, &[0.5])[0]);
        assert!(fitted.inference().is_none());
    }
    // invalid parts
    assert!(
        AftResult::from_parts(AftDistribution::Weibull, Some(1.0), vec![0.2], -1.0, None).is_err()
    );
    assert!(AftResult::from_parts(
        AftDistribution::Exponential,
        Some(1.0),
        vec![0.2],
        2.0,
        None
    )
    .is_err());
    let small = Mat::<f64>::identity(2, 2);
    assert!(AftResult::from_parts(
        AftDistribution::Weibull,
        Some(1.0),
        vec![0.2],
        1.0,
        Some(&small)
    )
    .is_err());
}
