//! Issue #43 (R reference values from R 4.6.1 / MASS / lme4): GammaRegressor (Tweedie p = 2) standard errors use the deviance-based
//! dispersion deviance/df; R summary.glm / vcov(glm) use the Pearson estimate.
use anofox_regression::solvers::{FittedRegressor, GammaRegressor, Regressor};
use faer::{Col, Mat};

#[test]
fn gamma_standard_errors_match_summary_glm() {
    let n = 40;
    let x = Mat::from_fn(n, 1, |i, _| i as f64 / 10.0);
    let y = Col::from_fn(n, |i| 1.0 + ((i * 7) % 5) as f64 + i as f64 / 10.0);
    let f = GammaRegressor::builder()
        .with_intercept(true)
        .compute_inference(true)
        .build()
        .fit(&x, &y)
        .unwrap();
    let r = f.result();
    let se = r.std_errors.as_ref().unwrap()[0];
    println!("coef {:?} {} SE slope {} intercept {:?}  (R vcov(glm(Gamma(log))): 0.10228431537393909 0.045137636902525077; deviance/df gives 0.109622973089476272 0.048376155595296524)",
             r.intercept, r.coefficients[0], se, r.intercept_std_error);
    println!(
        "log_likelihood {} aic {}  (R logLik(glm) -74.779819882157284, AIC 155.559639764314568)",
        r.log_likelihood, r.aic
    );
    assert!((r.coefficients[0] - 0.22705616796414885).abs() < 1e-5);
    assert!((se - 0.045137636902525077).abs() < 1e-6, "slope SE {se}");
    assert!((r.intercept_std_error.unwrap() - 0.10228431537393909).abs() < 1e-6);
    assert!((f.inner().dispersion - 0.10859375).abs() < 1e-6);
    // R logLik(glm(..., Gamma(log))) and AIC
    assert!(
        (r.log_likelihood + 74.779819882157284).abs() < 1e-5,
        "{}",
        r.log_likelihood
    );
    assert!((r.aic - 155.559639764314568).abs() < 1e-4, "{}", r.aic);
}
