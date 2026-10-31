//! #203: every AID candidate is scored at its maximum-likelihood fit, so its
//! AICc equals `AICc(greybox::alm(y ~ 1, distribution = d))` (ML negative
//! binomial size, ML σ with denominator n, greybox's Gamma fit).
//!
//! Reference values: `tests/r_scripts/generate_aid_ic_validation.R`.

use anofox_regression::solvers::aid::{AidClassifier, DemandDistribution};
use faer::Col;

const YC: [f64; N] = [
    1.0, 9.0, 1.0, 5.0, 3.0, 2.0, 7.0, 2.0, 6.0, 7.0, 2.0, 4.0, 2.0, 1.0, 4.0, 10.0, 3.0, 2.0,
    10.0, 7.0, 5.0, 3.0, 3.0, 3.0, 1.0, 4.0, 5.0, 10.0, 7.0, 2.0, 1.0, 1.0, 7.0, 0.0, 3.0, 2.0,
    1.0, 2.0, 7.0, 2.0, 5.0, 2.0, 4.0, 0.0, 9.0, 5.0, 13.0, 3.0, 4.0, 4.0, 2.0, 1.0, 13.0, 6.0,
    0.0, 4.0, 12.0, 2.0, 11.0, 8.0,
];
const YI: [f64; N] = [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 3.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 5.0, 1.0, 0.0,
    0.0, 1.0, 0.0, 0.0, 2.0, 1.0, 1.0, 0.0, 3.0, 0.0, 0.0, 1.0, 3.0, 1.0, 1.0, 0.0, 0.0, 5.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0,
    0.0, 2.0, 0.0,
];
const YF: [f64; N] = [
    7.657,
    13.388,
    7.2119999999999997,
    8.8040000000000003,
    7.0830000000000002,
    15.98,
    6.0179999999999998,
    4.7000000000000002,
    4.7510000000000003,
    4.5060000000000002,
    6.3200000000000003,
    2.0699999999999998,
    6.0979999999999999,
    6.7240000000000002,
    3.637,
    10.890000000000001,
    4.6109999999999998,
    8.1699999999999999,
    6.2389999999999999,
    2.677,
    4.4249999999999998,
    6.8220000000000001,
    4.3899999999999997,
    5.5979999999999999,
    4.3929999999999998,
    6.0739999999999998,
    3.0710000000000002,
    4.7519999999999998,
    3.8700000000000001,
    4.6150000000000002,
    5.375,
    5.9619999999999997,
    2.843,
    6.0970000000000004,
    7.4550000000000001,
    4.0979999999999999,
    10.616,
    12.246,
    6.1909999999999998,
    1.6140000000000001,
    13.076000000000001,
    5.3419999999999996,
    4.0650000000000004,
    5.0789999999999997,
    3.085,
    8.8670000000000009,
    8.9469999999999992,
    5.8550000000000004,
    4.7270000000000003,
    2.1920000000000002,
    9.7439999999999998,
    4.5979999999999999,
    3.8370000000000002,
    10.785,
    6.4699999999999998,
    8.5619999999999994,
    7.3300000000000001,
    2.403,
    5.3639999999999999,
    5.9660000000000002,
];
const YC_POIS: f64 = 335.75224411824024;
const YC_NB: f64 = 302.06636380847789;
const YC_NORM: f64 = 320.19949298124067;
const YI_POIS: f64 = 148.56558103575071;
const YI_NB: f64 = 138.4514231232927;
const YI_GEOM: f64 = 136.67129891909505;
const YF_NORM: f64 = 303.55227525325091;
const YF_GAMMA: f64 = 290.92323958733294;
const YF_LNORM: f64 = 290.78287040458724;
const YC_NB_SIZE: f64 = 2.7606929791506403;

const N: usize = 60;

fn check(y: &[f64; N], want: &[(DemandDistribution, f64)], what: &str) -> AidClassifier {
    let aid = AidClassifier::builder().detect_anomalies(false).build();
    let c = aid.classify(&Col::from_fn(N, |i| y[i]));
    for &(d, ic) in want {
        let got = c.ic_values[&d];
        assert!(
            (got - ic).abs() < 1e-6 * ic.abs(),
            "{what} {d:?}: {got} vs greybox {ic}"
        );
    }
    aid
}

#[test]
fn regular_count_candidates_match_greybox() {
    use DemandDistribution::*;
    let aid = check(
        &YC,
        &[
            (Poisson, YC_POIS),
            (NegativeBinomial, YC_NB),
            (Normal, YC_NORM),
        ],
        "regular count",
    );
    let c = aid.classify(&Col::from_fn(N, |i| YC[i]));
    assert_eq!(c.distribution, NegativeBinomial);
    let size = c.parameters.shape.unwrap();
    assert!(
        (size - YC_NB_SIZE).abs() < 1e-6 * YC_NB_SIZE,
        "ML size {size} vs {YC_NB_SIZE}"
    );
}

#[test]
fn intermittent_count_candidates_match_greybox() {
    use DemandDistribution::*;
    check(
        &YI,
        &[
            (Poisson, YI_POIS),
            (NegativeBinomial, YI_NB),
            (Geometric, YI_GEOM),
        ],
        "intermittent count",
    );
}

#[test]
fn regular_fractional_candidates_match_greybox() {
    use DemandDistribution::*;
    check(
        &YF,
        &[(Normal, YF_NORM), (Gamma, YF_GAMMA), (LogNormal, YF_LNORM)],
        "regular fractional",
    );
}
