//! 0.12: rebuilding fitted models from their parts (`from_parts`,
//! `FittedOrdinal::from_information`). A model rebuilt from the parts of a fit
//! predicts exactly as the fit; what goes through the covariance agrees to
//! rounding.

use anofox_regression::core::IntervalType;
use anofox_regression::inference::ClusterOptions;
use anofox_regression::solvers::{
    ContinuationRatioRegressor, FittedContinuationRatio, FittedMultinomialLogistic, FittedOrdinal,
    FittedPSpline, FittedRegressor, MultinomialLogisticRegressor, OrdinalLink,
    OrdinalQuantileMethod, OrdinalRegressor, PSplineRegressor, Regressor,
};
use faer::{Col, Mat};

/// Deterministic pseudo-random numbers in (0, 1).
fn uniform(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        })
        .collect()
}

fn predictors(n: usize) -> Mat<f64> {
    let u = uniform(2 * n, 7);
    Mat::from_fn(
        n,
        2,
        |i, j| {
            if j == 0 {
                4.0 * u[i] - 2.0
            } else {
                u[n + i]
            }
        },
    )
}

/// An ordered response with `k` levels from a logistic latent variable.
fn ordered_response(x: &Mat<f64>, k: usize) -> Col<f64> {
    let e = uniform(x.nrows(), 11);
    Col::from_fn(x.nrows(), |i| {
        let latent = 0.9 * x[(i, 0)] - 1.2 * x[(i, 1)] + (e[i] / (1.0 - e[i])).ln();
        let cut = ((latent + 3.0) / 6.0 * k as f64).floor();
        cut.clamp(0.0, (k - 1) as f64)
    })
}

fn new_rows() -> Mat<f64> {
    Mat::from_fn(5, 2, |i, j| {
        if j == 0 {
            -2.5 + 1.2 * i as f64
        } else {
            0.2 * i as f64
        }
    })
}

fn assert_mat_eq(name: &str, a: &Mat<f64>, b: &Mat<f64>) {
    assert_eq!(
        (a.nrows(), a.ncols()),
        (b.nrows(), b.ncols()),
        "{name}: shape"
    );
    for i in 0..a.nrows() {
        for j in 0..a.ncols() {
            assert_eq!(a[(i, j)].to_bits(), b[(i, j)].to_bits(), "{name}[{i},{j}]");
        }
    }
}

fn assert_col_close(name: &str, a: &Col<f64>, b: &Col<f64>, tol: f64) {
    assert_eq!(a.nrows(), b.nrows(), "{name}: length");
    for i in 0..a.nrows() {
        assert!(
            (a[i] - b[i]).abs() <= tol * (1.0 + b[i].abs()),
            "{name}[{i}]: {} vs {}",
            a[i],
            b[i]
        );
    }
}

fn assert_mat_close(name: &str, a: &Mat<f64>, b: &Mat<f64>, tol: f64) {
    for i in 0..a.nrows() {
        for j in 0..a.ncols() {
            assert!(
                (a[(i, j)] - b[(i, j)]).abs() <= tol * (1.0 + b[(i, j)].abs()),
                "{name}[{i},{j}]: {} vs {}",
                a[(i, j)],
                b[(i, j)]
            );
        }
    }
}

#[test]
fn multinomial_from_parts_predicts_as_the_fit() {
    let x = predictors(150);
    for k in [2, 4] {
        let y = ordered_response(&x, k);
        let fit = MultinomialLogisticRegressor::builder()
            .lambda(0.1)
            .build()
            .fit(&x, &y)
            .unwrap();
        let rebuilt = FittedMultinomialLogistic::from_parts(
            fit.classes().to_vec(),
            fit.coefficients().clone(),
            fit.intercepts().clone(),
        )
        .unwrap();
        let xn = new_rows();
        assert_mat_eq(
            "proba",
            &rebuilt.predict_proba(&xn),
            &fit.predict_proba(&xn),
        );
        assert_mat_eq(
            "decision",
            &rebuilt.decision_function(&xn),
            &fit.decision_function(&xn),
        );
        assert_eq!(
            rebuilt.predict(&x).iter().collect::<Vec<_>>(),
            fit.predict(&x).iter().collect::<Vec<_>>()
        );
        assert_eq!(rebuilt.score(&x, &y), fit.score(&x, &y));
        assert!(rebuilt.log_likelihood().is_nan());
    }
}

#[test]
fn multinomial_from_parts_validates() {
    let b = Mat::<f64>::zeros(3, 2);
    let c = Col::<f64>::zeros(3);
    assert!(
        FittedMultinomialLogistic::from_parts(vec![0.0, 1.0, 2.0], b.clone(), c.clone()).is_ok()
    );
    // two classes have one row
    assert!(FittedMultinomialLogistic::from_parts(vec![0.0, 1.0], b.clone(), c.clone()).is_err());
    assert!(FittedMultinomialLogistic::from_parts(vec![0.0], b.clone(), c.clone()).is_err());
    assert!(
        FittedMultinomialLogistic::from_parts(vec![1.0, 0.0, 2.0], b.clone(), c.clone()).is_err()
    );
    assert!(
        FittedMultinomialLogistic::from_parts(vec![0.0, 1.0, 2.0], b.clone(), Col::zeros(2))
            .is_err()
    );
    let mut bad = b.clone();
    bad[(0, 0)] = f64::NAN;
    assert!(FittedMultinomialLogistic::from_parts(vec![0.0, 1.0, 2.0], bad, c).is_err());
}

fn check_ordinal_round_trip(fit: &FittedOrdinal, rebuilt: &FittedOrdinal, tol: f64) {
    let xn = new_rows();
    assert_mat_eq(
        "category probabilities",
        &rebuilt.category_probabilities(&xn),
        &fit.category_probabilities(&xn),
    );
    assert_mat_eq(
        "exceedance",
        &rebuilt.exceedance_probabilities(&xn),
        &fit.exceedance_probabilities(&xn),
    );
    let mean = |f: &FittedOrdinal| f.predict_mean(&xn);
    assert_eq!(
        mean(rebuilt).iter().collect::<Vec<_>>(),
        mean(fit).iter().collect::<Vec<_>>()
    );
    for method in [
        OrdinalQuantileMethod::Interpolated,
        OrdinalQuantileMethod::Discrete,
    ] {
        let q = |f: &FittedOrdinal| f.predict_quantile(&xn, 0.5, method);
        assert_eq!(
            q(rebuilt).iter().collect::<Vec<_>>(),
            q(fit).iter().collect::<Vec<_>>()
        );
        let (e1, l1, u1) = rebuilt.predict_quantile_ci(&xn, 0.5, method, 0.9);
        let (e0, l0, u0) = fit.predict_quantile_ci(&xn, 0.5, method, 0.9);
        assert_col_close("quantile", &e1, &e0, tol);
        assert_col_close("quantile lower", &l1, &l0, tol);
        assert_col_close("quantile upper", &u1, &u0, tol);
    }
    let (m1, s1) = rebuilt.predict_mean_with_se(&xn);
    let (m0, s0) = fit.predict_mean_with_se(&xn);
    assert_col_close("mean", &m1, &m0, 0.0);
    assert_col_close("mean se", &s1, &s0, tol);
    assert_mat_close("vcov", &rebuilt.vcov(), &fit.vcov(), tol);
    assert_col_close("se", &rebuilt.std_errors(), &fit.std_errors(), tol);
    assert_col_close("z", &rebuilt.z_values(), &fit.z_values(), tol);
    assert_col_close("p", &rebuilt.p_values(), &fit.p_values(), tol);
}

#[test]
fn ordinal_from_parts_predicts_as_the_fit() {
    let x = predictors(200);
    let y = ordered_response(&x, 5);
    for (link, penalty) in [
        (OrdinalLink::Logistic, None),
        (OrdinalLink::Probit, None),
        (OrdinalLink::Logistic, Some(2.0)),
    ] {
        let mut b = OrdinalRegressor::builder().link(link);
        if let Some(l) = penalty {
            b = b.penalty(l);
        }
        let fit = b.build().fit(&x, &y).unwrap();
        let rebuilt = FittedOrdinal::from_parts(
            fit.link(),
            fit.levels().to_vec(),
            fit.intercepts().clone(),
            fit.coefficients().clone(),
            &fit.vcov(),
        )
        .unwrap();
        check_ordinal_round_trip(&fit, &rebuilt, 1e-9);
        assert!(rebuilt.log_likelihood().is_nan() && rebuilt.aic().is_nan());
        // the compact form round-trips exactly
        let compact = FittedOrdinal::from_information(
            fit.link(),
            fit.levels().to_vec(),
            fit.intercepts().clone(),
            fit.coefficients().clone(),
            fit.information(),
        )
        .unwrap();
        check_ordinal_round_trip(&fit, &compact, 0.0);
        assert_eq!(compact.information(), fit.information());
        // and so does the cluster-robust covariance on the fit's data
        let codes: Vec<usize> = (0..x.nrows()).map(|i| i % 17).collect();
        let v0 = fit
            .cluster_vcov(&x, &y, &[&codes], &ClusterOptions::robcov())
            .unwrap();
        let v1 = rebuilt
            .cluster_vcov(&x, &y, &[&codes], &ClusterOptions::robcov())
            .unwrap();
        assert_mat_close("cluster vcov", &v1, &v0, 1e-8);
    }
}

#[test]
fn ordinal_wald_statistics() {
    let x = predictors(200);
    let y = ordered_response(&x, 4);
    let fit = OrdinalRegressor::builder().build().fit(&x, &y).unwrap();
    let (b, se, z, p) = (
        fit.coefficients(),
        fit.std_errors(),
        fit.z_values(),
        fit.p_values(),
    );
    for j in 0..2 {
        assert_eq!(z[j], b[j] / se[j]);
        let tail = 0.5 * statrs::function::erf::erfc(z[j].abs() / std::f64::consts::SQRT_2);
        assert!(
            (p[j] - 2.0 * tail).abs() < 1e-12,
            "p {j}: {} vs {}",
            p[j],
            2.0 * tail
        );
    }
}

#[test]
fn ordinal_from_parts_validates() {
    let x = predictors(100);
    let y = ordered_response(&x, 4);
    let fit = OrdinalRegressor::builder().build().fit(&x, &y).unwrap();
    let (levels, a, b, v) = (
        fit.levels().to_vec(),
        fit.intercepts().clone(),
        fit.coefficients().clone(),
        fit.vcov(),
    );
    let link = OrdinalLink::Logistic;
    let parts = |l: Vec<f64>, a: Col<f64>, v: &Mat<f64>| {
        FittedOrdinal::from_parts(link, l, a, b.clone(), v)
    };
    assert!(parts(levels.clone(), a.clone(), &v).is_ok());
    let mut rev = levels.clone();
    rev.reverse();
    assert!(parts(rev, a.clone(), &v).is_err());
    let increasing = Col::from_fn(a.nrows(), |j| a[a.nrows() - 1 - j]);
    assert!(parts(levels.clone(), increasing, &v).is_err());
    assert!(parts(levels[1..].to_vec(), a.clone(), &v).is_err());
    let small = Mat::from_fn(2, 2, |i, j| v[(i, j)]);
    assert!(parts(levels.clone(), a.clone(), &small).is_err());
    let neg = Mat::from_fn(v.nrows(), v.ncols(), |i, j| -v[(i, j)]);
    assert!(parts(levels.clone(), a.clone(), &neg).is_err());
    // a dense matrix whose inverse is not tridiagonal in the intercepts
    let k = v.nrows();
    let dense = Mat::from_fn(k, k, |i, j| if i == j { 2.0 } else { 0.5 });
    assert!(parts(levels.clone(), a.clone(), &dense).is_err());
    let mut info = fit.information();
    info.intercept_diagonal.pop();
    assert!(FittedOrdinal::from_information(link, levels, a, b.clone(), info).is_err());
}

#[test]
fn continuation_ratio_from_parts_predicts_as_the_fit() {
    let x = predictors(200);
    let y = ordered_response(&x, 4);
    for cohort in [vec![], vec![1]] {
        let fit = ContinuationRatioRegressor::builder()
            .cohort_specific(&cohort)
            .build()
            .fit(&x, &y)
            .unwrap();
        let rebuilt = FittedContinuationRatio::from_parts(
            fit.levels().to_vec(),
            fit.n_features(),
            fit.cohort_specific(),
            &fit.coefficients(),
            Some(&fit.vcov()),
        )
        .unwrap();
        let xn = new_rows();
        assert_mat_eq(
            "cr category",
            &rebuilt.category_probabilities(&xn),
            &fit.category_probabilities(&xn),
        );
        assert_mat_eq(
            "cr conditional",
            &rebuilt.conditional_probabilities(&xn),
            &fit.conditional_probabilities(&xn),
        );
        assert_mat_eq("cr vcov", &rebuilt.vcov(), &fit.vcov());
        let bare = FittedContinuationRatio::from_parts(
            fit.levels().to_vec(),
            fit.n_features(),
            fit.cohort_specific(),
            &fit.coefficients(),
            None,
        )
        .unwrap();
        assert!(bare.vcov()[(0, 0)].is_nan());
        assert_mat_eq(
            "cr category (no vcov)",
            &bare.category_probabilities(&xn),
            &fit.category_probabilities(&xn),
        );
    }
}

#[test]
fn continuation_ratio_from_parts_validates() {
    let c = Col::<f64>::zeros(5);
    let ok = FittedContinuationRatio::from_parts(vec![0.0, 1.0, 2.0, 3.0], 2, &[], &c, None);
    assert!(ok.is_ok());
    assert!(FittedContinuationRatio::from_parts(vec![0.0, 1.0], 2, &[], &c, None).is_err());
    assert!(
        FittedContinuationRatio::from_parts(vec![0.0, 1.0, 2.0, 3.0], 2, &[2], &c, None).is_err()
    );
    assert!(FittedContinuationRatio::from_parts(
        vec![0.0, 1.0, 2.0, 3.0],
        2,
        &[],
        &Col::zeros(4),
        None
    )
    .is_err());
    let v = Mat::<f64>::zeros(4, 4);
    assert!(
        FittedContinuationRatio::from_parts(vec![0.0, 1.0, 2.0, 3.0], 2, &[], &c, Some(&v))
            .is_err()
    );
}

#[test]
fn pspline_from_parts_predicts_as_the_fit() {
    let n = 80;
    let u = uniform(n, 3);
    let x = Mat::from_fn(n, 1, |i, _| 6.0 * i as f64 / (n - 1) as f64);
    let y = Col::from_fn(n, |i| x[(i, 0)].sin() + 0.3 * (u[i] - 0.5));
    for model in [
        PSplineRegressor::new().with_n_basis(12),
        PSplineRegressor::new()
            .with_n_basis(20)
            .with_penalty_order(3),
    ] {
        let fit = model.fit(&x, &y).unwrap();
        let rebuilt = FittedPSpline::from_parts(
            fit.knots().to_vec(),
            fit.coefficients().to_vec(),
            &fit.covariance(),
            fit.sigma2(),
            fit.edf(),
            fit.n_observations(),
        )
        .unwrap();
        let xn = Mat::from_fn(7, 1, |i, _| -0.5 + i as f64);
        let p0 = fit.predict(&xn);
        let p1 = rebuilt.predict(&xn);
        assert_eq!(p0.iter().collect::<Vec<_>>(), p1.iter().collect::<Vec<_>>());
        for kind in [IntervalType::Confidence, IntervalType::Prediction] {
            let a = fit.predict_with_interval(&xn, Some(kind), 0.95);
            let b = rebuilt.predict_with_interval(&xn, Some(kind), 0.95);
            assert_col_close("se", &b.se, &a.se, 1e-9);
            assert_col_close("lower", &b.lower, &a.lower, 1e-9);
            assert_col_close("upper", &b.upper, &a.upper, 1e-9);
        }
        assert_mat_close("covariance", &rebuilt.covariance(), &fit.covariance(), 1e-9);
        assert!(rebuilt.lambda().is_nan() && rebuilt.result().aic.is_nan());
    }
}

#[test]
fn pspline_from_parts_validates() {
    let n = 40;
    let x = Mat::from_fn(n, 1, |i, _| i as f64);
    let y = Col::from_fn(n, |i| (i as f64 * 0.3).sin());
    let fit = PSplineRegressor::new()
        .with_n_basis(10)
        .fit(&x, &y)
        .unwrap();
    let (k, b, v) = (
        fit.knots().to_vec(),
        fit.coefficients().to_vec(),
        fit.covariance(),
    );
    let (s2, edf) = (fit.sigma2(), fit.edf());
    assert!(FittedPSpline::from_parts(k.clone(), b.clone(), &v, s2, edf, n).is_ok());
    assert!(FittedPSpline::from_parts(k[1..].to_vec(), b.clone(), &v, s2, edf, n).is_err());
    assert!(FittedPSpline::from_parts(k.clone(), b.clone(), &v, -1.0, edf, n).is_err());
    assert!(FittedPSpline::from_parts(k.clone(), b.clone(), &v, s2, n as f64, n).is_err());
    let neg = Mat::from_fn(v.nrows(), v.ncols(), |i, j| -v[(i, j)]);
    assert!(FittedPSpline::from_parts(k.clone(), b.clone(), &neg, s2, edf, n).is_err());
    let mut rev = k;
    rev.reverse();
    assert!(FittedPSpline::from_parts(rev, b, &v, s2, edf, n).is_err());
}
