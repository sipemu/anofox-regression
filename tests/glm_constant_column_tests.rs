//! GLMs with a constant feature column and an intercept: the column is
//! perfectly collinear with the intercept, so (like OLS and R's `glm`) it is
//! dropped, reported as aliased with a NaN coefficient, and ignored when
//! predicting. Previously the pivoted IRLS solve zeroed the intercept and let
//! the constant column absorb it (with garbage standard errors).
//!
//! R reference (the constant column x2 = 3 is NA in R; values from the
//! equivalent fit without it, glm.control(epsilon = 1e-14)):
//! ```r
//! i <- 0:39; x1 <- i/10
//! yc <- ((i*7) %% 5) + (i %/% 10); yb <- as.numeric((i*13) %% 7 < 3 + i %/% 10)
//! glm(yc ~ x1, family = poisson)                 # 0.6174470257183958 0.2961752642655828
//!                                                # SE 0.1961387339541893 0.0757780579809437
//! glm(yb ~ x1, family = binomial)                # -0.662579885905986 0.706677175500495
//!                                                # SE 0.661219745366984 0.330884234056606
//! glm(yb ~ x1, family = binomial("probit"))      # -0.399419861846433 0.426750132759850
//! MASS: glm(yc ~ x1, family = negative.binomial(2))  # 0.5967397109352058 0.3058390434732420
//! predict(poisson fit, data.frame(x1 = c(0, 1)), type = "response", se.fit = TRUE)
//! # fit 1.85418829915909 2.49333778809046 ; se 0.363678145509735 0.328779533046553
//! ```

use anofox_regression::prelude::*;
use faer::{Col, Mat};

const N: usize = 40;

fn x_with_constant() -> Mat<f64> {
    Mat::from_fn(N, 2, |i, j| if j == 0 { i as f64 / 10.0 } else { 3.0 })
}
fn y_count() -> Col<f64> {
    Col::from_fn(N, |i| ((i * 7) % 5) as f64 + (i / 10) as f64)
}
fn y_binary() -> Col<f64> {
    Col::from_fn(N, |i| if (i * 13) % 7 < 3 + i / 10 { 1.0 } else { 0.0 })
}

fn check(r: &RegressionResult, b0: f64, b1: f64, tol: f64) {
    assert!(
        (r.intercept.unwrap() - b0).abs() < tol,
        "intercept {:?} vs {b0}",
        r.intercept
    );
    assert!(
        (r.coefficients[0] - b1).abs() < tol,
        "coef {} vs {b1}",
        r.coefficients[0]
    );
    assert!(r.coefficients[1].is_nan(), "constant column must be NaN");
    assert_eq!(r.aliased, vec![false, true]);
    if let Some(se) = &r.std_errors {
        assert!(se[0].is_finite() && se[1].is_nan());
    }
}

#[test]
fn poisson_constant_column_is_aliased() {
    let x = x_with_constant();
    let f = PoissonRegressor::log()
        .with_intercept(true)
        .compute_inference(true)
        .build()
        .fit(&x, &y_count())
        .unwrap();
    check(f.result(), 0.6174470257183958, 0.2961752642655828, 1e-8);
    assert!((f.result().std_errors.as_ref().unwrap()[0] - 0.0757780579809437).abs() < 1e-8);
    assert!((f.result().intercept_std_error.unwrap() - 0.1961387339541893).abs() < 1e-8);

    let xn = Mat::from_fn(2, 2, |i, j| if j == 0 { i as f64 } else { 3.0 });
    let pr = f.predict_with_se(
        &xn,
        PredictionType::Response,
        Some(IntervalType::Confidence),
        0.95,
    );
    assert!((pr.fit[0] - 1.85418829915909).abs() < 1e-8);
    assert!((pr.fit[1] - 2.49333778809046).abs() < 1e-8);
    assert!((pr.se[0] - 0.363678145509735).abs() < 1e-7);
    assert!((pr.se[1] - 0.328779533046553).abs() < 1e-7);
    let p = f.predict(&xn);
    assert!((p[1] - 2.49333778809046).abs() < 1e-8);
}

#[test]
fn binomial_constant_column_is_aliased() {
    let x = x_with_constant();
    let f = BinomialRegressor::logistic()
        .with_intercept(true)
        .compute_inference(true)
        .build()
        .fit(&x, &y_binary())
        .unwrap();
    check(f.result(), -0.662579885905986, 0.706677175500495, 1e-7);
    assert!((f.result().std_errors.as_ref().unwrap()[0] - 0.330884234056606).abs() < 1e-7);
    assert!((f.result().intercept_std_error.unwrap() - 0.661219745366984).abs() < 1e-6);

    let f = BinomialRegressor::probit()
        .with_intercept(true)
        .build()
        .fit(&x, &y_binary())
        .unwrap();
    check(f.result(), -0.399419861846433, 0.426750132759850, 1e-7);
}

#[test]
fn negative_binomial_constant_column_is_aliased() {
    let f = NegativeBinomialRegressor::with_theta(2.0)
        .with_intercept(true)
        .compute_inference(true)
        .build()
        .fit(&x_with_constant(), &y_count())
        .unwrap();
    check(f.result(), 0.5967397109352058, 0.3058390434732420, 1e-7);
}

#[test]
fn gamma_and_tweedie_constant_column_match_fit_without_it() {
    let x = x_with_constant();
    let x1 = Mat::from_fn(N, 1, |i, _| x[(i, 0)]);
    let y = Col::from_fn(N, |i| 1.0 + ((i * 7) % 5) as f64 + i as f64 / 10.0);

    let with_const = GammaRegressor::builder()
        .with_intercept(true)
        .compute_inference(true)
        .build()
        .fit(&x, &y)
        .unwrap();
    let without = GammaRegressor::builder()
        .with_intercept(true)
        .compute_inference(true)
        .build()
        .fit(&x1, &y)
        .unwrap();
    let (a, b) = (with_const.result(), without.result());
    check(a, b.intercept.unwrap(), b.coefficients[0], 1e-12);
    assert!((a.std_errors.as_ref().unwrap()[0] - b.std_errors.as_ref().unwrap()[0]).abs() < 1e-12);

    let tw = TweedieRegressor::builder()
        .var_power(1.5)
        .with_intercept(true)
        .build()
        .fit(&x, &y)
        .unwrap();
    let tw1 = TweedieRegressor::builder()
        .var_power(1.5)
        .with_intercept(true)
        .build()
        .fit(&x1, &y)
        .unwrap();
    check(
        tw.result(),
        tw1.result().intercept.unwrap(),
        tw1.result().coefficients[0],
        1e-12,
    );
    let xn = Mat::from_fn(3, 2, |i, j| if j == 0 { i as f64 } else { 3.0 });
    let xn1 = Mat::from_fn(3, 1, |i, _| i as f64);
    let (p, p1) = (tw.predict(&xn), tw1.predict(&xn1));
    for i in 0..3 {
        assert!((p[i] - p1[i]).abs() < 1e-12);
    }
}

#[test]
fn all_constant_columns_give_intercept_only_fit() {
    let x = Mat::from_fn(N, 1, |_, _| 3.0);
    let yb = y_binary();
    let f = BinomialRegressor::logistic()
        .with_intercept(true)
        .build()
        .fit(&x, &yb)
        .unwrap();
    let p_hat: f64 = (0..N).map(|i| yb[i]).sum::<f64>() / N as f64;
    assert!((f.result().intercept.unwrap() - (p_hat / (1.0 - p_hat)).ln()).abs() < 1e-8);
    assert!(f.result().coefficients[0].is_nan());
    assert!((f.predict(&Mat::from_fn(1, 1, |_, _| 3.0))[0] - p_hat).abs() < 1e-8);
}

#[test]
fn constant_column_without_intercept_is_kept() {
    // Without an intercept the constant column *is* the intercept.
    let x = x_with_constant();
    let f = PoissonRegressor::log()
        .with_intercept(false)
        .build()
        .fit(&x, &y_count())
        .unwrap();
    let r = f.result();
    assert!((r.coefficients[0] - 0.2961752642655828).abs() < 1e-7);
    assert!((3.0 * r.coefficients[1] - 0.6174470257183958).abs() < 1e-7);
}
