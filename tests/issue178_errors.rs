//! #178: bad input and out-of-range options are reported as structured
//! errors — `InvalidParameter` naming the option, `InvalidInput` for the
//! data — never as the `NumericalError` catch-all.

use anofox_regression::diagnostics::{breusch_pagan, partial_residuals_glm};
use anofox_regression::inference::{
    compute_hc_standard_errors, compute_variance_factor, compute_xtwx_inverse_reduced,
    compute_xtx_inverse_reduced, HcType,
};
use anofox_regression::solvers::aft::AftRegressor;
use anofox_regression::solvers::path::{cv_path, fit_path, lambda_grid, PathOptions};
use anofox_regression::solvers::penalized_glm::{
    fit_poisson, PenalizedGlmRegressor, PoissonOptions,
};
use anofox_regression::solvers::quantile_inference::{
    quantile_inference, quantile_process, QuantileSe,
};
use anofox_regression::solvers::*;
use faer::{Col, Mat};

const N: usize = 30;

fn data() -> (Mat<f64>, Col<f64>) {
    let x = Mat::from_fn(N, 2, |i, j| {
        ((i * (j + 2)) as f64 * 0.37).sin() + i as f64 / 10.0
    });
    let y = Col::from_fn(N, |i| 1.0 + x[(i, 0)] + ((i * 7) % 5) as f64 / 4.0);
    (x, y)
}

#[track_caller]
fn param<T>(r: Result<T, RegressionError>, name: &str) {
    match r {
        Err(RegressionError::InvalidParameter { name: got, .. }) => assert_eq!(got, name),
        Err(e) => panic!("expected InvalidParameter `{name}`, got {e:?}"),
        Ok(_) => panic!("expected InvalidParameter `{name}`, got Ok"),
    }
}

#[track_caller]
fn input<T>(r: Result<T, RegressionError>) {
    match r {
        Err(RegressionError::InvalidInput(_)) => {}
        Err(e) => panic!("expected InvalidInput, got {e:?}"),
        Ok(_) => panic!("expected InvalidInput, got Ok"),
    }
}

/// 0.12.2: data degeneracies are typed.
#[track_caller]
fn degenerate<T>(r: Result<T, RegressionError>, want: DegenerateData) {
    match r {
        Err(RegressionError::Degenerate(d)) => assert_eq!(d, want),
        Err(e) => panic!("expected Degenerate({want:?}), got {e:?}"),
        Ok(_) => panic!("expected Degenerate({want:?}), got Ok"),
    }
}

#[test]
fn regularisation_paths() {
    let (x, y) = data();
    param(
        fit_path(&x, &y, &[0.1], &PathOptions::elastic_net(1.5)),
        "alpha",
    );
    let mut bad_tol = PathOptions::elastic_net(0.5);
    bad_tol.tol = 0.0;
    param(fit_path(&x, &y, &[0.1], &bad_tol), "tolerance");
    let opts = PathOptions::elastic_net(0.5);
    param(fit_path(&x, &y, &[], &opts), "lambdas");
    param(fit_path(&x, &y, &[-1.0], &opts), "lambdas");
    param(lambda_grid(&x, &y, &opts, 0, None), "n_lambda");
    param(
        lambda_grid(&x, &y, &opts, 10, Some(1.5)),
        "lambda_min_ratio",
    );
    degenerate(
        lambda_grid(&x, &Col::from_fn(N, |_| 2.0), &opts, 10, None),
        DegenerateData::NoAssociation,
    );
    input(fit_path(&Mat::zeros(N, 0), &y, &[0.1], &opts));
    let two_folds: Vec<usize> = (0..N).map(|i| i % 2).collect();
    param(cv_path(&x, &y, &[0.1, 0.01], &two_folds, &opts), "k");
}

#[test]
fn estimator_options() {
    let (x, y) = data();
    param(
        HuberRegressor::builder().epsilon(0.5).build().fit(&x, &y),
        "epsilon",
    );
    param(
        RlsRegressor::builder()
            .forgetting_factor(1.5)
            .build()
            .fit(&x, &y),
        "forgetting_factor",
    );
    param(
        TweedieRegressor::builder()
            .var_power(0.5)
            .build()
            .fit(&x, &y),
        "var_power",
    );
    let bls = BlsRegressor::builder()
        .lower_bounds(vec![1.0, 1.0])
        .upper_bounds(vec![0.0, 0.0])
        .build();
    param(bls.fit(&x, &y), "bounds");
    input(IsotonicRegressor::builder().build().fit(&x, &y));
}

#[test]
fn quantile_inference_options() {
    let (x, y) = data();
    param(quantile_process(&x, &y, &[], true, None, 0.95), "taus");
    param(
        quantile_inference(&x, &y, 0.5, true, QuantileSe::Iid, 1.5),
        "level",
    );
    param(
        quantile_inference(&x, &y, 0.5, true, QuantileSe::Boot { seed: 1, r: 1 }, 0.95),
        "r",
    );
}

#[test]
fn glmm_options_and_data() {
    let (x, y) = data();
    let g: Vec<usize> = (0..N).map(|i| i % 5).collect();
    let per_factor = GlmmRegressor::gaussian().random_slopes_per_factor(vec![vec![0], vec![1]]);
    param(
        per_factor.build().fit(&x, &y, &g),
        "random_slopes_per_factor",
    );
    input(
        GlmmRegressor::gaussian()
            .random_intercept(false)
            .build()
            .fit(&x, &y, &g),
    );
    let one_level = vec![0usize; N];
    degenerate(
        GlmmRegressor::gaussian()
            .build()
            .fit_crossed(&x, &y, &[&g, &one_level]),
        DegenerateData::TooFewGroups,
    );
    let offset = Col::from_fn(N, |i| if i == 3 { f64::NAN } else { 0.0 });
    let counts = Col::from_fn(N, |i| (i % 4) as f64);
    input(
        GlmmRegressor::poisson()
            .offset(offset)
            .build()
            .fit(&x, &counts, &g),
    );
}

#[test]
fn sufficient_statistics_and_inference_helpers() {
    let (x, _) = data();
    let nonsquare = Mat::<f64>::zeros(2, 3);
    let (xty, sx) = (Col::zeros(2), Col::zeros(2));
    input(
        OlsRegressor::builder()
            .build()
            .fit_from_moments(&nonsquare, &xty, &sx, 0.0, N),
    );
    input(
        RidgeRegressor::builder()
            .build()
            .fit_from_moments(&nonsquare, &xty, &sx, 0.0, N),
    );
    let all = [true, true];
    input(compute_xtx_inverse_reduced(&x, &all));
    input(compute_xtwx_inverse_reduced(
        &x,
        &Col::from_fn(N, |_| 1.0),
        &all,
    ));
    let few = Mat::from_fn(2, 2, |i, j| (i + j) as f64);
    input(compute_hc_standard_errors(
        &few,
        &Col::zeros(2),
        &[false, false],
        true,
        HcType::HC1,
    ));
    let r = Col::from_fn(N, |i| if i == 0 { f64::NAN } else { 1.0 });
    input(breusch_pagan(&r, &x, true));
    // compute_variance_factor: every argument is checked
    let none = [false, false];
    param(
        compute_variance_factor(&x, None, true, &[false], 0.0),
        "excluded",
    );
    param(
        compute_variance_factor(&x, Some(&Col::zeros(3)), true, &none, 0.0),
        "weights",
    );
    let negative = Col::from_fn(N, |_| -1.0);
    param(
        compute_variance_factor(&x, Some(&negative), true, &none, 0.0),
        "weights",
    );
    param(
        compute_variance_factor(&x, None, true, &none, -1.0),
        "ridge_lambda",
    );
    param(
        compute_variance_factor(&x, None, false, &all, 0.0),
        "excluded",
    );
}

#[test]
fn penalized_glm_aft_and_glm_diagnostics_return_regression_errors() {
    let (x, _) = data();
    let counts = Col::from_fn(N, |i| (i % 4) as f64);
    let short = Col::from_fn(N - 1, |i| counts[i]);
    let pglm = PenalizedGlmRegressor::builder().build();
    assert!(matches!(
        pglm.fit(&x, &short),
        Err(RegressionError::DimensionMismatch { .. })
    ));
    let off = PenalizedGlmRegressor::builder()
        .offset(Col::zeros(N - 1))
        .build();
    assert!(matches!(
        off.fit(&x, &counts),
        Err(RegressionError::DimensionMismatch { .. })
    ));
    let aft = AftRegressor::builder().build();
    let time = Col::from_fn(N, |i| 1.0 + i as f64);
    let events = vec![true; N];
    assert!(matches!(
        aft.fit(&x, &short, &events),
        Err(RegressionError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        aft.fit(&x, &time, &events[1..]),
        Err(RegressionError::DimensionMismatch { .. })
    ));
    let cols: Vec<Vec<f64>> = (0..2)
        .map(|j| (0..N).map(|i| x[(i, j)]).collect())
        .collect();
    let y: Vec<f64> = counts.iter().copied().collect();
    let fit = fit_poisson(&y, &cols, &PoissonOptions::default()).unwrap();
    let mut ragged = cols.clone();
    ragged[1].pop();
    assert!(matches!(
        partial_residuals_glm(&fit, &y, &ragged),
        Err(RegressionError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        anofox_regression::diagnostics::augment_glm(&fit, &y, &ragged),
        Err(RegressionError::DimensionMismatch { .. })
    ));
}
