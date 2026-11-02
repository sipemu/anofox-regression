//! #181: Huber and quantile regression report whether they converged.

use anofox_regression::solvers::IterativeFit;
use anofox_regression::solvers::{HuberRegressor, QuantileRegressor, Regressor};
use faer::{Col, Mat};

fn data() -> (Mat<f64>, Col<f64>) {
    let n = 60;
    let x = Mat::from_fn(n, 2, |i, j| ((i * (j + 3)) as f64 * 0.41).sin() * 2.0);
    let y = Col::from_fn(n, |i| {
        let outlier = if i % 17 == 0 { 15.0 } else { 0.0 };
        1.0 + x[(i, 0)] - 0.5 * x[(i, 1)] + ((i * 7919) % 11) as f64 / 10.0 + outlier
    });
    (x, y)
}

#[test]
fn huber_reports_convergence() {
    let (x, y) = data();
    let f = HuberRegressor::builder().build().fit(&x, &y).unwrap();
    assert!(f.converged());
    assert!(f.iterations() >= 1 && f.iterations() < 100);
    // one IRLS step is not enough with outliers
    let f = HuberRegressor::builder()
        .max_iterations(1)
        .build()
        .fit(&x, &y)
        .unwrap();
    assert!(!f.converged());
    assert_eq!(f.iterations(), 1);
}

#[test]
fn quantile_reports_the_lp_optimum() {
    let (x, y) = data();
    for tau in [0.1, 0.5, 0.9] {
        let f = QuantileRegressor::builder()
            .tau(tau)
            .build()
            .fit(&x, &y)
            .unwrap();
        assert!(f.converged(), "tau {tau}");
        assert!(f.iterations() >= 1);
    }
    // a duplicated column: the pivots cannot certify a vertex
    let xd = Mat::from_fn(x.nrows(), 2, |i, _| x[(i, 0)]);
    let f = QuantileRegressor::builder().build().fit(&xd, &y).unwrap();
    assert!(!f.converged());
}

/// #179: every iterative fit reports through `IterativeFit`, and
/// `ensure_converged` is the opt-in strict policy.
#[test]
fn iterative_fit_is_implemented_everywhere() {
    use anofox_regression::solvers::aft::AftRegressor;
    use anofox_regression::solvers::penalized_glm::{PenalizedGlmFamily, PenalizedGlmRegressor};
    use anofox_regression::solvers::*;
    fn check<F: IterativeFit>(name: &str, f: F) {
        assert!(f.converged(), "{name} did not converge");
        assert!(f.iterations() >= 1, "{name}: no iterations");
        assert!(f.ensure_converged().is_ok(), "{name}");
    }
    let n = 60;
    let x = Mat::from_fn(n, 2, |i, j| ((i * (j + 3)) as f64 * 0.41).sin());
    let counts = Col::from_fn(n, |i| {
        ((0.4 + 0.6 * x[(i, 0)]).exp() + (i % 3) as f64).floor()
    });
    let pos = Col::from_fn(n, |i| 1.0 + counts[i]);
    let binary = Col::from_fn(n, |i| {
        f64::from(u8::from(x[(i, 0)] + 0.3 * ((i * 7) % 5) as f64 > 0.5))
    });
    let cont = Col::from_fn(n, |i| {
        1.0 + x[(i, 0)] - 0.5 * x[(i, 1)] + ((i * 7919) % 11) as f64 / 10.0
    });
    let ordered = Col::from_fn(n, |i| (counts[i] as usize % 3) as f64);
    check(
        "poisson",
        PoissonRegressor::log().build().fit(&x, &counts).unwrap(),
    );
    check(
        "binomial",
        BinomialRegressor::logistic()
            .build()
            .fit(&x, &binary)
            .unwrap(),
    );
    check(
        "negbin",
        NegativeBinomialRegressor::builder()
            .build()
            .fit(&x, &counts)
            .unwrap(),
    );
    check(
        "tweedie",
        TweedieRegressor::builder()
            .var_power(1.5)
            .build()
            .fit(&x, &pos)
            .unwrap(),
    );
    check(
        "gamma",
        GammaRegressor::builder().build().fit(&x, &pos).unwrap(),
    );
    check(
        "logistic",
        LogisticRegressor::builder()
            .build()
            .fit(&x, &binary)
            .unwrap(),
    );
    check(
        "bayesian ridge",
        BayesianRidgeRegressor::builder()
            .build()
            .fit(&x, &cont)
            .unwrap(),
    );
    check(
        "ard",
        ArdRegressor::builder().build().fit(&x, &cont).unwrap(),
    );
    check(
        "passive-aggressive",
        PassiveAggressiveRegressor::builder()
            .build()
            .fit(&x, &cont)
            .unwrap(),
    );
    check(
        "ordinal",
        OrdinalRegressor::builder()
            .build()
            .fit(&x, &ordered)
            .unwrap(),
    );
    check(
        "continuation ratio",
        ContinuationRatioRegressor::builder()
            .build()
            .fit(&x, &ordered)
            .unwrap(),
    );
    let g: Vec<usize> = (0..n).map(|i| i % 6).collect();
    check(
        "glmm",
        GlmmRegressor::poisson()
            .build()
            .fit(&x, &counts, &g)
            .unwrap(),
    );
    check(
        "penalized glm",
        PenalizedGlmRegressor::builder()
            .family(PenalizedGlmFamily::poisson())
            .build()
            .fit(&x, &counts)
            .unwrap(),
    );
    let time = Col::from_fn(n, |i| 1.0 + (i % 9) as f64 + x[(i, 0)].abs());
    let events: Vec<bool> = (0..n).map(|i| i % 4 != 0).collect();
    check(
        "aft",
        AftRegressor::builder()
            .build()
            .fit(&x, &time, &events)
            .unwrap(),
    );

    // the strict policy turns a stopped-early fit into an error
    let early = BayesianRidgeRegressor::builder()
        .max_iter(1)
        .build()
        .fit(&x, &cont)
        .unwrap();
    assert!(!early.converged());
    assert!(matches!(
        early.ensure_converged(),
        Err(RegressionError::ConvergenceFailed { iterations: 1 })
    ));
    let early = ArdRegressor::builder()
        .max_iter(1)
        .build()
        .fit(&x, &cont)
        .unwrap();
    assert!(!early.converged());
    let early = PassiveAggressiveRegressor::builder()
        .max_iter(1)
        .build()
        .fit(&x, &cont)
        .unwrap();
    assert!(!early.converged());
}
