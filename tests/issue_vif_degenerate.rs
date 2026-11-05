//! Degenerate designs in `variance_inflation_factor` / `generalized_vif`
//! (0.11.1): exactly explained columns give `∞` (never a silent 1), a
//! constant column only affects its own term, and `generalized_vif_terms`
//! returns car's `GVIF^(1/(2*Df))`.
//!
//! `car::vif` stops with "there are aliased coefficients" on these designs;
//! the finite references are `car::vif` on the design without the aliased
//! column(s) (`tests/r_scripts/generate_vif_degenerate.R`, car 3.1.5, mtcars).

use anofox_regression::diagnostics::{
    generalized_vif, generalized_vif_terms, variance_inflation_factor, GvifTerm,
};
use faer::Mat;

const WT: [f64; N] = [
    2.6200000000000001,
    2.875,
    2.3199999999999998,
    3.2149999999999999,
    3.4399999999999999,
    3.46,
    3.5699999999999998,
    3.1899999999999999,
    3.1499999999999999,
    3.4399999999999999,
    3.4399999999999999,
    4.0700000000000003,
    3.73,
    3.7799999999999998,
    5.25,
    5.4240000000000004,
    5.3449999999999998,
    2.2000000000000002,
    1.615,
    1.835,
    2.4649999999999999,
    3.52,
    3.4350000000000001,
    3.8399999999999999,
    3.8450000000000002,
    1.9350000000000001,
    2.1400000000000001,
    1.5129999999999999,
    3.1699999999999999,
    2.77,
    3.5699999999999998,
    2.7799999999999998,
];
const HP: [f64; N] = [
    110.0, 110.0, 93.0, 110.0, 175.0, 105.0, 245.0, 62.0, 95.0, 123.0, 123.0, 180.0, 180.0, 180.0,
    205.0, 215.0, 230.0, 66.0, 52.0, 65.0, 97.0, 150.0, 150.0, 245.0, 175.0, 66.0, 91.0, 113.0,
    264.0, 175.0, 335.0, 109.0,
];
const DISP: [f64; N] = [
    160.0,
    160.0,
    108.0,
    258.0,
    360.0,
    225.0,
    360.0,
    146.69999999999999,
    140.80000000000001,
    167.59999999999999,
    167.59999999999999,
    275.80000000000001,
    275.80000000000001,
    275.80000000000001,
    472.0,
    460.0,
    440.0,
    78.700000000000003,
    75.700000000000003,
    71.099999999999994,
    120.09999999999999,
    318.0,
    304.0,
    350.0,
    400.0,
    79.0,
    120.3,
    95.099999999999994,
    351.0,
    145.0,
    301.0,
    121.0,
];
const DRAT: [f64; N] = [
    3.8999999999999999,
    3.8999999999999999,
    3.8500000000000001,
    3.0800000000000001,
    3.1499999999999999,
    2.7599999999999998,
    3.21,
    3.6899999999999999,
    3.9199999999999999,
    3.9199999999999999,
    3.9199999999999999,
    3.0699999999999998,
    3.0699999999999998,
    3.0699999999999998,
    2.9300000000000002,
    3.0,
    3.23,
    4.0800000000000001,
    4.9299999999999997,
    4.2199999999999998,
    3.7000000000000002,
    2.7599999999999998,
    3.1499999999999999,
    3.73,
    3.0800000000000001,
    4.0800000000000001,
    4.4299999999999997,
    3.77,
    4.2199999999999998,
    3.6200000000000001,
    3.54,
    4.1100000000000003,
];
const CYL: [f64; N] = [
    6.0, 6.0, 4.0, 6.0, 8.0, 6.0, 8.0, 4.0, 4.0, 6.0, 6.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 4.0, 4.0,
    4.0, 4.0, 8.0, 8.0, 8.0, 8.0, 4.0, 4.0, 4.0, 8.0, 6.0, 8.0, 4.0,
];
const GEAR: [f64; N] = [
    4.0, 4.0, 4.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0,
    4.0, 3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 5.0, 5.0, 5.0, 5.0, 5.0, 4.0,
];
const VIF_PLAIN: [f64; 4] = [
    5.096600716938724,
    2.8943732350140103,
    8.2094018312665131,
    2.279547155523356,
];
const GVIF: [f64; 4] = [
    3.5411538575031751,
    8.5798552716461138,
    7.4395564411170847,
    5.6713070686135865,
];
const GVIF_ADJ: [f64; 4] = [
    1.8817953814119044,
    1.711472757115494,
    2.727555029897121,
    1.5431949399714917,
];

/// car::vif(lm(y ~ a + b + c)) on a 5-row design.
const SMALL: [[f64; 3]; 5] = [
    [1.0, 2.0, 0.0],
    [2.0, 1.0, 3.0],
    [4.0, 5.0, 1.0],
    [3.0, 2.0, 4.0],
    [7.0, 3.0, 2.0],
];
const VIF_SMALL: [f64; 3] = [1.6060606060606064, 1.9074960127591714, 1.4832535885167468];

const N: usize = 32;

fn close(a: f64, b: f64, what: &str) {
    assert!(
        (a - b).abs() < 1e-9 * (1.0 + b.abs()),
        "{what}: {a} vs car {b}"
    );
}

fn dummy(v: f64, level: f64) -> f64 {
    f64::from(u8::from(v == level))
}

/// wt | cyl6, cyl8 | hp | gear4, gear5, as `model.matrix`.
fn factor_design(i: usize, j: usize) -> f64 {
    match j {
        0 => WT[i],
        1 => dummy(CYL[i], 6.0),
        2 => dummy(CYL[i], 8.0),
        3 => HP[i],
        4 => dummy(GEAR[i], 4.0),
        _ => dummy(GEAR[i], 5.0),
    }
}

#[test]
fn small_valid_design_matches_car() {
    let x = Mat::from_fn(5, 3, |i, j| SMALL[i][j]);
    let vif = variance_inflation_factor(&x);
    for j in 0..3 {
        close(vif[j], VIF_SMALL[j], "VIF");
    }
}

#[test]
fn too_few_rows_is_infinite() {
    // n <= p: every auxiliary regression fits exactly (R² = 1 in R).
    let x45 = Mat::from_fn(4, 5, |i, j| SMALL[i][j % 3] + (i * j * j) as f64);
    let x22 = Mat::from_fn(2, 2, |i, j| (i * 3 + j * j + 1) as f64);
    let x33 = Mat::from_fn(3, 3, |i, j| ((i + 1) * (j + 2) + i * i) as f64);
    for x in [x45, x22, x33] {
        let vif = variance_inflation_factor(&x);
        assert!(vif.iter().all(|v| v.is_infinite()), "{vif:?}");
    }
}

#[test]
fn single_column_and_no_rows() {
    let one = Mat::from_fn(2, 1, |i, _| i as f64);
    assert_eq!(variance_inflation_factor(&one)[0], 1.0);
    let constant = Mat::from_fn(5, 1, |_, _| 3.0);
    assert!(variance_inflation_factor(&constant)[0].is_infinite());
    let empty = Mat::<f64>::zeros(0, 2);
    assert!(variance_inflation_factor(&empty).iter().all(|v| v.is_nan()));
}

#[test]
fn constant_column_only_affects_itself() {
    // wt, const, hp, disp, drat: the others equal car::vif without `const`.
    let x = Mat::from_fn(N, 5, |i, j| [WT[i], 2.5, HP[i], DISP[i], DRAT[i]][j]);
    let vif = variance_inflation_factor(&x);
    assert!(vif[1].is_infinite());
    for (k, j) in [0, 2, 3, 4].into_iter().enumerate() {
        close(vif[j], VIF_PLAIN[k], "VIF");
    }
}

#[test]
fn aliased_columns_only_affect_themselves() {
    // wt, hp, wt + hp, disp, drat: the aliased three are ∞; disp and drat
    // regress on the same span as in car::vif(mpg ~ wt + hp + disp + drat).
    let x = Mat::from_fn(N, 5, |i, j| {
        [WT[i], HP[i], WT[i] + HP[i], DISP[i], DRAT[i]][j]
    });
    let vif = variance_inflation_factor(&x);
    assert!((0..3).all(|j| vif[j].is_infinite()), "{vif:?}");
    close(vif[3], VIF_PLAIN[2], "VIF disp");
    close(vif[4], VIF_PLAIN[3], "VIF drat");
}

#[test]
fn gvif_constant_column_only_affects_its_term() {
    // The factor design of #196 plus a constant column as its own term.
    let x = Mat::from_fn(N, 7, |i, j| if j == 6 { 1.0 } else { factor_design(i, j) });
    let gvif = generalized_vif(&x, &[1, 2, 1, 2, 1]);
    assert!(gvif[4].is_infinite());
    for k in 0..4 {
        close(gvif[k], GVIF[k], "GVIF");
    }
    // The constant column inside the gear term: only that term is ∞.
    let x = Mat::from_fn(N, 7, |i, j| match j {
        4 => 1.0,
        5 | 6 => factor_design(i, j - 1),
        _ => factor_design(i, j),
    });
    let gvif = generalized_vif(&x, &[1, 2, 1, 3]);
    assert!(gvif[3].is_infinite());
    for k in 0..3 {
        close(gvif[k], GVIF[k], "GVIF");
    }
}

#[test]
fn gvif_aliased_term_only_affects_terms_involved() {
    // wt | cyl | hp | gear | wt + hp
    let x = Mat::from_fn(N, 7, |i, j| {
        if j == 6 {
            WT[i] + HP[i]
        } else {
            factor_design(i, j)
        }
    });
    let gvif = generalized_vif(&x, &[1, 2, 1, 2, 1]);
    for k in [0, 2, 4] {
        assert!(gvif[k].is_infinite(), "term {k}: {}", gvif[k]);
    }
    close(gvif[1], GVIF[1], "GVIF cyl");
    close(gvif[3], GVIF[3], "GVIF gear");
}

#[test]
fn gvif_terms_report_adjusted_gvif() {
    let x = Mat::from_fn(N, 6, factor_design);
    let sizes = [1, 2, 1, 2];
    let terms = generalized_vif_terms(&x, &sizes);
    assert_eq!(terms.len(), 4);
    for (k, t) in terms.iter().enumerate() {
        assert_eq!(t.df, sizes[k]);
        close(t.gvif, GVIF[k], "GVIF");
        close(t.adjusted_gvif, GVIF_ADJ[k], "GVIF^(1/(2 df))");
    }
    let x = Mat::from_fn(N, 7, |i, j| if j == 6 { 1.0 } else { factor_design(i, j) });
    let last = generalized_vif_terms(&x, &[1, 2, 1, 2, 1])[4];
    assert!(last.gvif.is_infinite() && last.adjusted_gvif.is_infinite());
    assert_eq!(last.df, 1);
}

#[test]
fn undefined_gvif_is_nan() {
    let x = Mat::from_fn(20, 3, |i, j| ((i * (j + 3)) % 7) as f64 + j as f64);
    assert!(generalized_vif(&x, &[1, 1]).iter().all(|v| v.is_nan()));
    let gvif = generalized_vif(&x, &[1, 0, 2]);
    assert!(gvif[1].is_nan() && gvif[0].is_finite() && gvif[2].is_finite());
    let t: GvifTerm = generalized_vif_terms(&x, &[1, 0, 2])[1];
    assert!(t.gvif.is_nan() && t.adjusted_gvif.is_nan() && t.df == 0);
}
