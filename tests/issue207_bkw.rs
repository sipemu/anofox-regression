//! #207: collinearity diagnostics of Belsley, Kuh & Welsch on the
//! unit-column-length design, and the plain condition number as `kappa`.
//!
//! Reference values: `tests/r_scripts/generate_bkw_validation.R`
//! (`olsrr::ols_eigen_cindex`, `kappa(exact = TRUE)`, mtcars).

use anofox_regression::diagnostics::{
    condition_diagnostic, condition_number, variance_decomposition_proportions, ConditionSeverity,
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
const EIGEN: [f64; 5] = [
    4.6928069138838246,
    0.24030864065589932,
    0.0521534298401451,
    0.01140688855188329,
    0.0033241270682369531,
];
const CINDEX: [f64; 5] = [
    1.0,
    4.4190778178154249,
    9.4858205672553702,
    20.283025821474325,
    37.573143774520446,
];
const PROP: [f64; 25] = [
    0.00023232519808516871,
    0.00071720862855002192,
    0.0025661846126623774,
    0.00110645504111824,
    0.0003775503417537388,
    0.0036813894093758615,
    0.00093942535604093461,
    0.031334561663825623,
    0.03413290375109395,
    0.014825067237532753,
    0.00091920954747242497,
    0.070078981335890786,
    0.73500372222302124,
    0.058394261711095116,
    0.0026259360655648979,
    0.0014476534550052502,
    0.71798346613472863,
    0.20733751084897786,
    0.88572564240377261,
    0.056822691200833747,
    0.99371942239006117,
    0.21028091854478959,
    0.023758020651512925,
    0.020640737092920122,
    0.92534875515431481,
];
const KAPPA: f64 = 4258.3785502265018;
const KAPPA_NOINT: f64 = 669.48898641339906;

const N: usize = 32;

fn x() -> Mat<f64> {
    Mat::from_fn(N, 4, |i, j| [WT[i], HP[i], DISP[i], DRAT[i]][j])
}

fn close(a: f64, b: f64, what: &str) {
    assert!(
        (a - b).abs() < 1e-9 * (1.0 + b.abs()),
        "{what}: {a} vs R {b}"
    );
}

#[test]
fn condition_number_is_kappa() {
    close(condition_number(&x(), true), KAPPA, "kappa");
    close(
        condition_number(&x(), false),
        KAPPA_NOINT,
        "kappa without intercept",
    );
    close(
        condition_diagnostic(&x(), true).condition_number,
        KAPPA,
        "diagnostic kappa",
    );
}

#[test]
fn condition_indices_match_olsrr() {
    let d = condition_diagnostic(&x(), true);
    for k in 0..5 {
        close(d.singular_values[k].powi(2), EIGEN[k], "eigenvalue");
        close(d.condition_indices[k], CINDEX[k], "condition index");
    }
    close(
        d.scaled_condition_number,
        CINDEX[4],
        "scaled condition number",
    );
    // Unit-free: κ = 4258 would be "Severe", the BKW index 37.6 is moderate.
    assert_eq!(d.severity, ConditionSeverity::Moderate);
}

#[test]
fn variance_decomposition_proportions_match_olsrr() {
    let pi = variance_decomposition_proportions(&x(), true);
    // PROP is component-major (olsrr rows); `pi` is coefficient × component.
    for k in 0..5 {
        for j in 0..5 {
            close(pi[(j, k)], PROP[k * 5 + j], &format!("pi[{j}, {k}]"));
        }
    }
}

#[test]
fn diagnostics_do_not_depend_on_units() {
    let mut x2 = x();
    for i in 0..N {
        x2[(i, 1)] *= 1000.0; // hp in milli-units
    }
    let (a, b) = (
        condition_diagnostic(&x(), true),
        condition_diagnostic(&x2, true),
    );
    for k in 0..5 {
        close(
            b.condition_indices[k],
            a.condition_indices[k],
            "scaled indices",
        );
    }
}
