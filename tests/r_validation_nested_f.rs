//! Issue #90: `inference::nested_f_test` against R's
//! `anova(lm(reduced), lm(full))` and `summary(...)$adj.r.squared`. Reference
//! values from `tests/r_scripts/generate_nested_f_validation.R`.

#![allow(clippy::excessive_precision)]

use anofox_regression::inference::{
    nested_f_test, nested_f_test_indices, NestedFTest, NestedFTestError, NestedFTestOptions,
};
use faer::{Col, Mat};

const TOL: f64 = 1e-9;

const X1: [f64; 12] = [
    1.2, 2.5, 3.1, 4.8, 5.0, 6.7, 7.3, 8.1, 9.6, 10.2, 11.5, 12.9,
];
const X2: [f64; 12] = [
    0.5, -1.2, 2.3, 0.8, -0.4, 1.9, -2.1, 0.3, 1.1, -0.7, 2.6, -1.5,
];
const X3: [f64; 12] = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0, 5.0, 8.0];
const Y: [f64; 12] = [
    2.9, 4.1, 7.8, 8.2, 9.1, 14.0, 10.9, 14.6, 17.9, 17.1, 23.0, 20.4,
];

fn close(name: &str, got: f64, want: f64) {
    assert!(
        (got - want).abs() <= TOL * want.abs().max(1.0),
        "{name}: got {got:e}, want {want:e}"
    );
}

/// Design with the given columns (x4 = 2·x1 − x2 is exactly collinear).
fn design(cols: &[&str]) -> Mat<f64> {
    Mat::from_fn(12, cols.len(), |i, j| match cols[j] {
        "x1" => X1[i],
        "x2" => X2[i],
        "x3" => X3[i],
        "x4" => 2.0 * X1[i] - X2[i],
        c => panic!("unknown column {c}"),
    })
}

fn y() -> Col<f64> {
    Col::from_fn(12, |i| Y[i])
}

struct Expected {
    f: f64,
    df1: usize,
    df2: usize,
    p: f64,
    rss_r: f64,
    rss_f: f64,
    adj_r: f64,
    adj_f: f64,
}

fn check(t: &NestedFTest, e: &Expected) {
    close("F", t.f_statistic, e.f);
    assert_eq!((t.df1, t.df2), (e.df1, e.df2), "df");
    close("p", t.p_value, e.p);
    close("rss_reduced", t.rss_reduced, e.rss_r);
    close("rss_full", t.rss_full, e.rss_f);
    close("adj_r2_reduced", t.adj_r_squared_reduced, e.adj_r);
    close("adj_r2_full", t.adj_r_squared_full, e.adj_f);
    close("delta", t.delta_adj_r_squared(), e.adj_f - e.adj_r);
    assert_eq!(t.n_observations, 12);
}

const X1_VS_FULL: Expected = Expected {
    f: 3.91675014697775055e+01,
    df1: 2,
    df2: 8,
    p: 7.37245814150955264e-05,
    rss_r: 2.70743110048637554e+01,
    rss_f: 2.50876794653673052e+00,
    adj_r: 9.33383120598241467e-01,
    adj_f: 9.92283908364675837e-01,
};

#[test]
fn x1_vs_x1_x2_x3() {
    let t = nested_f_test(
        &design(&["x1"]),
        &design(&["x1", "x2", "x3"]),
        &y(),
        &NestedFTestOptions::default(),
    )
    .unwrap();
    check(&t, &X1_VS_FULL);
}

#[test]
fn indices_match_the_explicit_designs() {
    let t = nested_f_test_indices(
        &design(&["x1", "x2", "x3"]),
        &y(),
        &[0],
        &NestedFTestOptions::default(),
    )
    .unwrap();
    check(&t, &X1_VS_FULL);
}

#[test]
fn one_added_parameter_gives_t_squared() {
    let t = nested_f_test(
        &design(&["x1", "x2"]),
        &design(&["x1", "x2", "x3"]),
        &y(),
        &NestedFTestOptions::default(),
    )
    .unwrap();
    check(
        &t,
        &Expected {
            f: 4.68661087541795252e-01,
            df1: 1,
            df2: 8,
            p: 5.12944752624747702e-01,
            rss_r: 2.65573818581346810e+00,
            rss_f: 2.50876794653673052e+00,
            adj_r: 9.92739448335558272e-01,
            adj_f: 9.92283908364675837e-01,
        },
    );
    // summary(lm(y ~ x1 + x2 + x3))$coefficients["x3", "t value"]^2
    close("t^2", t.f_statistic, 4.68661087541797472e-01);
}

#[test]
fn intercept_only_reduced_is_the_overall_f_test() {
    let t = nested_f_test(
        &Mat::zeros(12, 0),
        &design(&["x1", "x2", "x3"]),
        &y(),
        &NestedFTestOptions::default(),
    )
    .unwrap();
    check(
        &t,
        &Expected {
            f: 4.72530731181841475e+02,
            df1: 3,
            df2: 8,
            p: 2.43501816046052847e-09,
            rss_r: 4.47059999999999832e+02,
            rss_f: 2.50876794653673052e+00,
            adj_r: 0.0,
            adj_f: 9.92283908364675837e-01,
        },
    );
    // summary(lm(y ~ x1 + x2 + x3))$fstatistic
    close("overall F", t.f_statistic, 4.72530731181841645e+02);
}

#[test]
fn without_intercept() {
    let options = NestedFTestOptions {
        with_intercept: false,
        ..Default::default()
    };
    let t = nested_f_test(
        &design(&["x1"]),
        &design(&["x1", "x2", "x3"]),
        &y(),
        &options,
    )
    .unwrap();
    check(
        &t,
        &Expected {
            f: 4.22237417729931153e+01,
            df1: 2,
            df2: 9,
            p: 2.67015851514641252e-05,
            rss_r: 2.95305142336220214e+01,
            rss_f: 2.84410685036594302e+00,
            adj_r: 9.86126496973946276e-01,
            adj_f: 9.98366905908049662e-01,
        },
    );
}

#[test]
fn rank_deficient_full_design_uses_the_rank() {
    // R: anova(lm(y ~ x1), lm(y ~ x1 + x2 + x4 + x3)), x4 aliased.
    let t = nested_f_test(
        &design(&["x1"]),
        &design(&["x1", "x2", "x4", "x3"]),
        &y(),
        &NestedFTestOptions::default(),
    )
    .unwrap();
    check(&t, &X1_VS_FULL);
    assert_eq!(t.n_parameters_full, 4);
}

#[test]
fn aliased_additions_give_df1_zero_and_nan() {
    // x4 is in span(x1, x2): the full model adds nothing (R reports NA).
    let t = nested_f_test(
        &design(&["x1", "x2"]),
        &design(&["x1", "x2", "x4"]),
        &y(),
        &NestedFTestOptions::default(),
    )
    .unwrap();
    assert_eq!(t.df1, 0);
    assert!(t.f_statistic.is_nan() && t.p_value.is_nan());
}

#[test]
fn reduced_column_outside_the_full_span_is_an_error() {
    let r = nested_f_test(
        &design(&["x3"]),
        &design(&["x1", "x2"]),
        &y(),
        &NestedFTestOptions::default(),
    );
    assert!(matches!(r, Err(NestedFTestError::NotNested { column: 0 })));
}

#[test]
fn too_few_rows_is_a_fit_error() {
    let x = Mat::from_fn(3, 3, |i, j| ((i + 1) * (j + 2)) as f64 + (i * j) as f64);
    let y = Col::from_fn(3, |i| i as f64);
    let r = nested_f_test_indices(&x, &y, &[0], &NestedFTestOptions::default());
    assert!(matches!(r, Err(NestedFTestError::Fit(_))));
}
