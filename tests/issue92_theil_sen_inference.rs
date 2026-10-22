//! Theil–Sen coefficient intervals (#92): Sen's exact interval with
//! Kendall's test, and the pairs bootstrap.
//!
//! Exact-interval references come from
//! `tests/r_scripts/generate_theil_sen_ci_validation.R` (R `trend::sens.slope`
//! and `stats::cor.test(method = "kendall")`).

#![allow(dead_code)]

use anofox_regression::solvers::{FittedRegressor, Regressor, TheilSenCiMethod, TheilSenRegressor};
use faer::{Col, Mat};

include!("fixtures/theil_sen_ci_validation.rs");

const TOL: f64 = 1e-12;

fn close(what: &str, got: f64, want: f64, tol: f64) {
    assert!(
        (got - want).abs() <= tol * want.abs().max(1.0),
        "{what}: {got} vs R {want}"
    );
}

fn univariate(x: &[f64], y: &[f64]) -> (Mat<f64>, Col<f64>) {
    (
        Mat::from_fn(x.len(), 1, |i, _| x[i]),
        Col::from_fn(y.len(), |i| y[i]),
    )
}

fn inference(level: f64) -> TheilSenRegressor {
    TheilSenRegressor::builder()
        .compute_inference(true)
        .confidence_level(level)
        .build()
}

#[test]
fn exact_interval_matches_trend_sens_slope() {
    let t: Vec<f64> = (1..=Y_SENS.len()).map(|i| i as f64).collect();
    let (x, y) = univariate(&t, &Y_SENS);
    for (level, lo, hi, slope, z, p) in [
        (
            0.95,
            SENS_LO_95,
            SENS_HI_95,
            SENS_SLOPE_95,
            SENS_Z_95,
            SENS_P_95,
        ),
        (
            0.9,
            SENS_LO_90,
            SENS_HI_90,
            SENS_SLOPE_90,
            SENS_Z_90,
            SENS_P_90,
        ),
    ] {
        let fit = inference(level).fit(&x, &y).unwrap();
        assert_eq!(fit.ci_method(), Some(TheilSenCiMethod::Exact));
        let r = fit.result();
        close("slope", r.coefficients[0], slope, TOL);
        close("lower", r.conf_interval_lower.as_ref().unwrap()[0], lo, TOL);
        close("upper", r.conf_interval_upper.as_ref().unwrap()[0], hi, TOL);
        close("z", r.t_statistics.as_ref().unwrap()[0], z, 1e-10);
        close("p", r.p_values.as_ref().unwrap()[0], p, 1e-8);
        assert_eq!(r.confidence_level, level);
        // No standard error and no intercept interval for the exact method.
        assert!(r.std_errors.is_none());
        assert!(r.intercept_conf_interval.is_none());
        assert!(r.f_statistic.is_nan());
    }
}

#[test]
fn exact_interval_with_ties_matches_r() {
    let (x, y) = univariate(&X_TIES, &Y_TIES);
    let fit = inference(0.95).fit(&x, &y).unwrap();
    let r = fit.result();
    close("slope", r.coefficients[0], TIES_SLOPE, TOL);
    close(
        "lower",
        r.conf_interval_lower.as_ref().unwrap()[0],
        TIES_LO,
        TOL,
    );
    close(
        "upper",
        r.conf_interval_upper.as_ref().unwrap()[0],
        TIES_HI,
        TOL,
    );
    close("z", r.t_statistics.as_ref().unwrap()[0], TIES_Z, 1e-10);
    close("p", r.p_values.as_ref().unwrap()[0], TIES_P, 1e-8);
}

#[test]
fn exact_interval_unattainable_for_tiny_n_is_nan() {
    // n = 4: 6 slopes, C ≈ 1.96·sqrt(8.67) ≈ 5.8 → rank_lo = 0.
    let (x, y) = univariate(&[1.0, 2.0, 3.0, 4.0], &[1.0, 3.0, 2.0, 5.0]);
    let fit = inference(0.95).fit(&x, &y).unwrap();
    let r = fit.result();
    assert!(r.conf_interval_lower.as_ref().unwrap()[0].is_nan());
    assert!(r.conf_interval_upper.as_ref().unwrap()[0].is_nan());
    assert!(r.p_values.as_ref().unwrap()[0].is_finite());
}

#[test]
fn exact_rejects_several_features() {
    let x = Mat::from_fn(20, 2, |i, j| ((i * (j + 3)) % 7) as f64 + i as f64 * 0.1);
    let y = Col::from_fn(20, |i| i as f64);
    let err = TheilSenRegressor::builder()
        .compute_inference(true)
        .ci_method(TheilSenCiMethod::Exact)
        .build()
        .fit(&x, &y)
        .unwrap_err();
    assert!(err.to_string().contains("exact"), "{err}");
}

fn multi_data(n: usize, seed: u64) -> (Mat<f64>, Col<f64>) {
    let mut s = seed;
    let mut rnd = move || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let x = Mat::from_fn(n, 2, |_, _| rnd() * 10.0);
    let y = Col::from_fn(n, |i| {
        1.0 + 0.5 * x[(i, 0)] - 2.0 * x[(i, 1)] + (rnd() - 0.5) * 2.0
    });
    (x, y)
}

#[test]
fn bootstrap_is_deterministic_and_covers_every_parameter() {
    let (x, y) = multi_data(15, 3);
    let model = |seed| {
        TheilSenRegressor::builder()
            .compute_inference(true)
            .n_bootstrap(199)
            .random_state(seed)
            .build()
    };
    let a = model(0).fit(&x, &y).unwrap();
    let b = model(0).fit(&x, &y).unwrap();
    let c = model(1).fit(&x, &y).unwrap();
    assert_eq!(a.ci_method(), Some(TheilSenCiMethod::Bootstrap));
    assert_eq!(a.bootstrap_replicates(), Some(199));
    let ra = a.result();
    let rb = b.result();
    let rc = c.result();
    let se = |r: &anofox_regression::core::RegressionResult| -> Vec<f64> {
        r.std_errors.as_ref().unwrap().iter().copied().collect()
    };
    assert_eq!(se(ra), se(rb));
    assert_ne!(se(ra), se(rc));
    assert_eq!(ra.intercept_conf_interval, rb.intercept_conf_interval);
    for j in 0..2 {
        let (lo, hi) = (
            ra.conf_interval_lower.as_ref().unwrap()[j],
            ra.conf_interval_upper.as_ref().unwrap()[j],
        );
        assert!(lo < ra.coefficients[j] && ra.coefficients[j] < hi);
        assert!(se(ra)[j] > 0.0);
    }
    let (lo, hi) = ra.intercept_conf_interval.unwrap();
    assert!(lo < ra.intercept.unwrap() && ra.intercept.unwrap() < hi);
    assert!(ra.intercept_std_error.unwrap() > 0.0);
    assert!(ra.t_statistics.is_none() && ra.p_values.is_none());

    // Independent of the row order.
    let n = x.nrows();
    let perm = |i: usize| (i * 7 + 3) % n;
    let xp = Mat::from_fn(n, 2, |i, j| x[(perm(i), j)]);
    let yp = Col::from_fn(n, |i| y[perm(i)]);
    let d = model(0).fit(&xp, &yp).unwrap();
    assert_eq!(se(d.result()), se(ra));
    assert_eq!(
        d.result().intercept_conf_interval,
        ra.intercept_conf_interval
    );
}

#[test]
fn bootstrap_on_one_feature_gives_intercept_interval() {
    let t: Vec<f64> = (1..=Y_SENS.len()).map(|i| i as f64).collect();
    let (x, y) = univariate(&t, &Y_SENS);
    let fit = TheilSenRegressor::builder()
        .compute_inference(true)
        .ci_method(TheilSenCiMethod::Bootstrap)
        .build()
        .fit(&x, &y)
        .unwrap();
    let r = fit.result();
    assert_eq!(fit.ci_method(), Some(TheilSenCiMethod::Bootstrap));
    assert!(r.intercept_conf_interval.is_some());
    // The bootstrap and the exact interval roughly agree here.
    let lo = r.conf_interval_lower.as_ref().unwrap()[0];
    let hi = r.conf_interval_upper.as_ref().unwrap()[0];
    assert!(
        (lo - SENS_LO_95).abs() < 0.1 && (hi - SENS_HI_95).abs() < 0.1,
        "{lo} {hi}"
    );
}

#[test]
fn aliased_column_gets_nan_inference() {
    let (x1, y) = multi_data(12, 9);
    let x = Mat::from_fn(
        12,
        3,
        |i, j| if j == 2 { 2.0 * x1[(i, 0)] } else { x1[(i, j)] },
    );
    let fit = TheilSenRegressor::builder()
        .compute_inference(true)
        .n_bootstrap(99)
        .build()
        .fit(&x, &y)
        .unwrap();
    let r = fit.result();
    assert!(r.aliased[2]);
    let se = r.std_errors.as_ref().unwrap();
    assert!(se[0].is_finite() && se[1].is_finite() && se[2].is_nan());
    assert!(r.conf_interval_lower.as_ref().unwrap()[2].is_nan());
}

#[test]
fn inference_options_are_validated() {
    let (x, y) = multi_data(20, 1);
    for builder in [
        TheilSenRegressor::builder().confidence_level(1.0),
        TheilSenRegressor::builder().n_bootstrap(1),
    ] {
        assert!(builder.compute_inference(true).build().fit(&x, &y).is_err());
    }
    // Without inference the options are not used.
    assert!(TheilSenRegressor::builder()
        .n_bootstrap(1)
        .build()
        .fit(&x, &y)
        .is_ok());
}

/// Coverage of the 90% intervals on simulated data (heavy-tailed noise).
/// `cargo test --release --test issue92_theil_sen_inference -- --ignored --nocapture`
#[test]
#[ignore]
fn coverage_simulation() {
    let reps = 400;
    let level = 0.9;
    let mut s: u64 = 12345;
    let mut rnd = move || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 11) as f64 + 0.5) / ((1u64 << 53) as f64)
    };
    // Student-t(3)-like noise via a ratio of uniforms is overkill; use a
    // Laplace draw.
    let mut laplace = move || {
        let u = rnd() - 0.5;
        -u.signum() * (1.0 - 2.0 * u.abs()).ln()
    };
    let (mut exact, mut boot1, mut boot_b0, mut boot2) = (0, 0, 0, [0, 0]);
    for rep in 0..reps {
        let n = 40;
        // One feature: exact and bootstrap.
        let x: Vec<f64> = (0..n).map(|i| i as f64 * 0.25).collect();
        let y: Vec<f64> = x.iter().map(|xi| 2.0 + 0.7 * xi + laplace()).collect();
        let (xm, yc) = univariate(&x, &y);
        let r = inference(level).fit(&xm, &yc).unwrap();
        let (lo, hi) = (
            r.result().conf_interval_lower.as_ref().unwrap()[0],
            r.result().conf_interval_upper.as_ref().unwrap()[0],
        );
        exact += usize::from(lo <= 0.7 && 0.7 <= hi);
        let r = TheilSenRegressor::builder()
            .compute_inference(true)
            .confidence_level(level)
            .ci_method(TheilSenCiMethod::Bootstrap)
            .random_state(rep)
            .build()
            .fit(&xm, &yc)
            .unwrap();
        let rr = r.result();
        let (lo, hi) = (
            rr.conf_interval_lower.as_ref().unwrap()[0],
            rr.conf_interval_upper.as_ref().unwrap()[0],
        );
        boot1 += usize::from(lo <= 0.7 && 0.7 <= hi);
        let (lo, hi) = rr.intercept_conf_interval.unwrap();
        boot_b0 += usize::from(lo <= 2.0 && 2.0 <= hi);

        // Two features: bootstrap.
        let xm2 = Mat::from_fn(n, 2, |i, j| if j == 0 { x[i] } else { (rnd() - 0.5) * 6.0 });
        let y2 = Col::from_fn(n, |i| {
            1.0 + 0.5 * xm2[(i, 0)] - 1.5 * xm2[(i, 1)] + laplace()
        });
        let r = TheilSenRegressor::builder()
            .compute_inference(true)
            .confidence_level(level)
            .n_bootstrap(299)
            .max_subpopulation(2000)
            .random_state(rep)
            .build()
            .fit(&xm2, &y2)
            .unwrap();
        let rr = r.result();
        for (j, beta) in [0.5, -1.5].into_iter().enumerate() {
            let lo = rr.conf_interval_lower.as_ref().unwrap()[j];
            let hi = rr.conf_interval_upper.as_ref().unwrap()[j];
            boot2[j] += usize::from(lo <= beta && beta <= hi);
        }
    }
    let pct = |k: usize| 100.0 * k as f64 / reps as f64;
    println!(
        "nominal {:.0}%: exact slope {:.1}%, bootstrap slope {:.1}%, bootstrap intercept {:.1}%, \
         2-feature bootstrap {:.1}% / {:.1}%",
        100.0 * level,
        pct(exact),
        pct(boot1),
        pct(boot_b0),
        pct(boot2[0]),
        pct(boot2[1])
    );
    for k in [exact, boot1, boot_b0, boot2[0], boot2[1]] {
        assert!((pct(k) - 90.0).abs() < 5.0, "coverage {}", pct(k));
    }
}
