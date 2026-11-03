//! Issue #321: `Serialize` / `Deserialize` for the fitted models (feature
//! `serde`). Every model goes fit → serialize → deserialize through JSON and
//! MessagePack (with field names, as anofox-ml stores it) and must give
//! bit-identical predictions, intervals and results, and re-serialize to the
//! same bytes.

#![cfg(feature = "serde")]

use anofox_regression::core::IntervalType;
use anofox_regression::solvers::*;
use faer::{Col, Mat};
use serde::de::DeserializeOwned;
use serde::Serialize;

const N: usize = 40;

fn x() -> Mat<f64> {
    Mat::from_fn(N, 3, |i, j| {
        ((i * (j + 2)) as f64 * 0.37).sin() + 0.1 * j as f64
    })
}

fn y() -> Col<f64> {
    let x = x();
    Col::from_fn(N, |i| {
        1.0 + 2.0 * x[(i, 0)] - x[(i, 1)]
            + 0.5 * x[(i, 2)]
            + ((i * 7) % 5) as f64 * 0.1
            + if i == 11 { 4.0 } else { 0.0 }
    })
}

/// A design with an exactly collinear column (aliased, NaN coefficient).
fn x_aliased() -> Mat<f64> {
    let x = x();
    Mat::from_fn(N, 4, |i, j| if j < 3 { x[(i, j)] } else { 2.0 * x[(i, 0)] })
}

fn new_x() -> Mat<f64> {
    Mat::from_fn(7, 3, |i, j| (i as f64 - 3.0) * 0.4 + 0.2 * j as f64)
}

/// The JSON and MessagePack round trips of `t`; each must re-serialize to
/// exactly the same bytes.
fn round_trips<T: Serialize + DeserializeOwned>(t: &T) -> [T; 2] {
    let json = serde_json::to_string(t).unwrap();
    let from_json: T = serde_json::from_str(&json).unwrap();
    assert_eq!(
        serde_json::to_string(&from_json).unwrap(),
        json,
        "JSON is not stable"
    );
    let mp = rmp_serde::to_vec_named(t).unwrap();
    let from_mp: T = rmp_serde::from_slice(&mp).unwrap();
    assert_eq!(
        rmp_serde::to_vec_named(&from_mp).unwrap(),
        mp,
        "MessagePack is not stable"
    );
    [from_json, from_mp]
}

fn same(a: &Col<f64>, b: &Col<f64>, what: &str) {
    assert_eq!(a.nrows(), b.nrows(), "{what}: length");
    for i in 0..a.nrows() {
        assert!(
            a[i].to_bits() == b[i].to_bits(),
            "{what}[{i}]: {} vs {}",
            a[i],
            b[i]
        );
    }
}

/// Predictions, both interval kinds and the serialized result agree.
fn check_regressor<F: FittedRegressor + Serialize + DeserializeOwned>(f: &F, xn: &Mat<f64>) {
    let result_json = serde_json::to_string(f.result()).unwrap();
    for g in round_trips(f) {
        same(&f.predict(xn), &g.predict(xn), "predict");
        for kind in [IntervalType::Confidence, IntervalType::Prediction] {
            let (a, b) = (
                f.predict_with_interval(xn, Some(kind), 0.9),
                g.predict_with_interval(xn, Some(kind), 0.9),
            );
            same(&a.lower, &b.lower, "lower");
            same(&a.upper, &b.upper, "upper");
            same(&a.se, &b.se, "se");
        }
        assert_eq!(serde_json::to_string(g.result()).unwrap(), result_json);
    }
}

#[test]
fn ols_round_trip_keeps_inference_and_intervals() {
    let f = OlsRegressor::builder()
        .compute_inference(true)
        .build()
        .fit(&x(), &y())
        .unwrap();
    check_regressor(&f, &new_x());
    for g in round_trips(&f) {
        assert_eq!(
            format!("{:?}", g.variance_factor()),
            format!("{:?}", f.variance_factor())
        );
        assert!(g.result().std_errors.is_some());
    }
}

#[test]
fn aliased_nan_coefficients_survive_json() {
    let f = OlsRegressor::builder()
        .compute_inference(true)
        .build()
        .fit(&x_aliased(), &y())
        .unwrap();
    assert!(f.result().coefficients[3].is_nan());
    let json = serde_json::to_string(&f).unwrap();
    assert!(json.contains("\"NaN\""));
    let xn = Mat::from_fn(5, 4, |i, j| (i + j) as f64 * 0.3);
    check_regressor(&f, &xn);
    for g in round_trips(&f) {
        assert!(g.result().coefficients[3].is_nan());
        assert_eq!(g.result().aliased, f.result().aliased);
    }
}

#[test]
fn wls_ridge_elastic_net_round_trip() {
    let w = Col::from_fn(N, |i| 0.5 + (i % 4) as f64 * 0.5);
    let wls = WlsRegressor::builder()
        .weights(w)
        .compute_inference(true)
        .build()
        .fit(&x(), &y())
        .unwrap();
    check_regressor(&wls, &new_x());
    let ridge = RidgeRegressor::builder()
        .lambda(0.3)
        .compute_inference(true)
        .build()
        .fit(&x(), &y())
        .unwrap();
    check_regressor(&ridge, &new_x());
    let enet = ElasticNetRegressor::builder()
        .lambda(0.05)
        .alpha(0.5)
        .build()
        .fit(&x(), &y())
        .unwrap();
    check_regressor(&enet, &new_x());
    for g in round_trips(&enet) {
        assert_eq!(g.lambda(), enet.lambda());
    }
}

#[test]
fn huber_quantile_isotonic_round_trip() {
    let huber = HuberRegressor::builder()
        .compute_inference(true)
        .build()
        .fit(&x(), &y())
        .unwrap();
    check_regressor(&huber, &new_x());
    for g in round_trips(&huber) {
        assert_eq!(g.scale(), huber.scale());
        assert_eq!(g.outliers(), huber.outliers());
        assert_eq!(g.iterations(), huber.iterations());
    }
    let quantile = QuantileRegressor::builder()
        .tau(0.3)
        .build()
        .fit(&x(), &y())
        .unwrap();
    check_regressor(&quantile, &new_x());
    for g in round_trips(&quantile) {
        assert_eq!(g.tau(), quantile.tau());
        assert_eq!(g.check_loss(), quantile.check_loss());
        assert_eq!(g.converged(), quantile.converged());
    }
    let x1 = Mat::from_fn(N, 1, |i, _| i as f64);
    let iso = IsotonicRegressor::builder().build().fit(&x1, &y()).unwrap();
    let xn = Mat::from_fn(9, 1, |i, _| i as f64 * 6.0 - 5.0);
    check_regressor(&iso, &xn);
}

#[test]
fn corrupt_input_is_rejected() {
    let f = OlsRegressor::builder().build().fit(&x(), &y()).unwrap();
    let json = serde_json::to_string(&f).unwrap();
    let aliased = OlsRegressor::builder()
        .build()
        .fit(&x_aliased(), &y())
        .unwrap();
    let with_nan = serde_json::to_string(&aliased).unwrap();
    let bad_float = with_nan.replacen("\"NaN\"", "\"nan?\"", 1);
    assert!(serde_json::from_str::<FittedOls>(&bad_float).is_err());
    let bad = json.replacen("\"nrows\":", "\"nrows\":99,\"x\":", 1);
    assert!(serde_json::from_str::<FittedOls>(&bad).is_err());
    assert!(serde_json::from_str::<FittedOls>("{}").is_err());
}

fn counts() -> Col<f64> {
    let x = x();
    Col::from_fn(N, |i| {
        (0.5 + 0.6 * x[(i, 0)] - 0.3 * x[(i, 1)]).exp().round() + (i % 3) as f64
    })
}

fn binary() -> Col<f64> {
    let x = x();
    Col::from_fn(N, |i| {
        f64::from(u8::from(
            x[(i, 0)] - 0.5 * x[(i, 1)] + ((i * 5) % 7) as f64 * 0.15 > 0.6,
        ))
    })
}

#[test]
fn glms_round_trip() {
    let (xm, xn) = (x(), new_x());
    let w = Col::from_fn(N, |i| 0.5 + (i % 3) as f64 * 0.5);
    let poisson = PoissonRegressor::log()
        .weights(w.clone())
        .build()
        .fit(&xm, &counts())
        .unwrap();
    check_regressor(&poisson, &xn);
    for g in round_trips(&poisson) {
        same(
            &g.pearson_residuals(),
            &poisson.pearson_residuals(),
            "pearson",
        );
        assert_eq!(g.deviance, poisson.deviance);
    }
    let binomial = BinomialRegressor::logistic()
        .weights(w)
        .build()
        .fit(&xm, &binary())
        .unwrap();
    check_regressor(&binomial, &xn);
    let pos = Col::from_fn(N, |i| counts()[i] + 0.5);
    let tweedie = TweedieRegressor::builder()
        .var_power(1.5)
        .link_power(0.0)
        .build()
        .fit(&xm, &pos)
        .unwrap();
    check_regressor(&tweedie, &xn);
    let gamma = GammaRegressor::builder().build().fit(&xm, &pos).unwrap();
    check_regressor(&gamma, &xn);
    let nb = NegativeBinomialRegressor::builder()
        .build()
        .fit(&xm, &counts())
        .unwrap();
    check_regressor(&nb, &xn);
    let logistic = LogisticRegressor::builder()
        .build()
        .fit(&xm, &binary())
        .unwrap();
    for g in round_trips(&logistic) {
        same(&g.predict_proba(&xn), &logistic.predict_proba(&xn), "proba");
        same(&g.predict(&xn), &logistic.predict(&xn), "labels");
    }
}

#[test]
fn bayesian_lars_multinomial_round_trip() {
    let (xm, xn) = (x(), new_x());
    let br = BayesianRidgeRegressor::builder()
        .build()
        .fit(&xm, &y())
        .unwrap();
    check_regressor(&br, &xn);
    for g in round_trips(&br) {
        same(&g.predict_std(&xn), &br.predict_std(&xn), "std");
    }
    let ard = ArdRegressor::builder().build().fit(&xm, &y()).unwrap();
    check_regressor(&ard, &xn);
    let lars = LarsRegressor::builder()
        .method(LarsMethod::Lasso)
        .build()
        .fit(&xm, &y())
        .unwrap();
    check_regressor(&lars, &xn);
    for g in round_trips(&lars) {
        assert_eq!(g.active(), lars.active());
        assert_eq!(g.coefs_path(), lars.coefs_path());
    }
    let classes = Col::from_fn(N, |i| (i % 3) as f64);
    let m = MultinomialLogisticRegressor::builder()
        .l1_ratio(0.5)
        .build()
        .fit(&xm, &classes)
        .unwrap();
    for g in round_trips(&m) {
        let (a, b) = (m.predict_proba(&xn), g.predict_proba(&xn));
        for i in 0..a.nrows() {
            same(
                &a.row(i).transpose().to_owned(),
                &b.row(i).transpose().to_owned(),
                "proba",
            );
        }
        assert_eq!(g.classes(), m.classes());
    }
}

#[test]
fn spline_bases_round_trip() {
    use anofox_regression::splines::{BSpline, KnotPlacement, Lsp, Rcs, SplineExtrapolation};
    let x: Vec<f64> = (0..30).map(|i| i as f64 * 0.37).collect();
    let q = [-1.0, 0.5, 4.0, 12.0];
    let bs = BSpline::from_data(&x, 5, 3, KnotPlacement::Quantile)
        .unwrap()
        .with_extrapolation(SplineExtrapolation::Linear);
    let rcs = Rcs::from_data(&x, 4).unwrap();
    let lsp = Lsp::new(&[2.0, 5.0]).unwrap();
    for g in round_trips(&bs) {
        assert_eq!(g, bs);
        assert_eq!(g.basis(&q, "x").unwrap(), bs.basis(&q, "x").unwrap());
    }
    for g in round_trips(&rcs) {
        assert_eq!(g, rcs);
    }
    for g in round_trips(&lsp) {
        assert_eq!(g, lsp);
    }
}
