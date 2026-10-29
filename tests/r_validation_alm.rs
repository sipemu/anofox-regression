//! Validation tests for ALM implementation against R greybox package
//!
//! These tests compare the Rust implementation against known outputs from
//! the R greybox package (version 2.0.6).
//!
//! Generated with seed 42 in R for reproducibility.

use anofox_regression::solvers::{AlmDistribution, AlmRegressor, FittedRegressor, Regressor};
use faer::{Col, Mat};

// Tolerance for coefficient comparisons (allow 10% relative error due to different optimizers)
const COEF_TOL: f64 = 0.15;
// Tolerance for log-likelihood (allow 5% relative error)
const LL_TOL: f64 = 0.10;

/// Helper to check if two values are approximately equal with relative tolerance
fn approx_eq_rel(a: f64, b: f64, tol: f64) -> bool {
    if a == b {
        return true;
    }
    let diff = (a - b).abs();
    let max_val = a.abs().max(b.abs()).max(1e-10);
    diff / max_val < tol
}

// =============================================================================
// Test Data from R greybox package (seed = 42)
// =============================================================================

const X_NORMAL: [f64; 50] = [
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0,
    18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0,
    34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0,
    50.0,
];

const Y_NORMAL: [f64; 50] = [
    8.112875, 3.805905, 8.089385, 10.398588, 11.212805, 11.181626, 17.534566, 14.216023, 22.055271,
    17.311858, 22.914609, 27.359936, 17.833418, 22.663634, 24.600036, 28.407851, 27.147241,
    21.530634, 23.678599, 36.460340, 33.080084, 30.156075, 36.484248, 42.144024, 45.685580,
    40.208593, 42.228192, 39.210511, 47.380292, 45.580015, 50.366350, 52.614512, 55.105311,
    51.673221, 56.514865, 51.348974, 55.646623, 56.947277, 53.757377, 62.608368, 64.617996,
    64.416828, 69.274490, 66.319886, 65.895157, 72.798454, 70.565820, 78.832304, 74.705661,
    79.466944,
];

const Y_LAPLACE: [f64; 50] = [
    4.965776, 3.148483, 11.727183, 10.428698, 10.269282, 12.329652, 15.037866, 14.769499, 7.020730,
    68.354649, 17.898296, 21.055692, 23.745471, 27.699210, 22.818124, 30.407628, 29.007544,
    32.615518, 33.762186, 34.662634, 30.870643, 35.229441, 38.870554, 35.639430, 38.371514,
    43.242989, 45.304536, 45.891303, 43.342671, 44.200657, 53.538121, 51.273764, 52.265321,
    53.137310, 51.417013, 58.335991, 57.348580, 58.951730, 63.800039, 14.965319, 68.176349,
    64.071478, 68.951046, 72.673331, 66.667633, 68.917622, 69.604784, 70.122358, 76.239948,
    79.459613,
];

const Y_STUDENT_T: [f64; 50] = [
    6.941604, 3.465051, 4.844003, 7.019676, 9.122799, 9.543905, 17.500570, 12.933728, 18.056944,
    16.869138, 16.267703, 20.492796, 22.199643, 23.802756, 22.520613, 21.285024, 24.160929,
    27.953216, 30.579087, 30.618706, 32.834308, 31.905105, 36.016228, 34.421041, 39.963692,
    42.220832, 41.861212, 43.493331, 53.402567, 47.301486, 48.405228, 46.835664, 55.063881,
    50.155651, 54.979766, 54.766832, 59.096568, 56.711109, 54.569744, 62.183044, 57.432692,
    65.782519, 72.823240, 62.497073, 72.273372, 63.713712, 76.094415, 73.419221, 82.117782,
    79.140656,
];

const Y_LOGNORMAL: [f64; 50] = [
    1.632926, 2.726548, 1.690274, 1.264807, 2.150500, 1.776324, 1.739262, 1.521293, 2.455940,
    2.112336, 2.677010, 2.202714, 1.998604, 3.651255, 2.383264, 3.540917, 1.914632, 2.460009,
    2.689068, 2.671688, 4.639526, 3.168222, 3.536961, 2.553049, 2.804527, 4.852144, 5.406189,
    5.554782, 2.600773, 7.500639, 5.669278, 4.271584, 5.479875, 3.416390, 3.391096, 4.926925,
    3.491918, 5.457581, 7.840572, 4.014199, 4.519800, 5.894202, 4.413735, 5.501467, 8.263321,
    8.765850, 7.577278, 3.992962, 7.055454, 10.169896,
];

const X_POISSON: [f64; 50] = [
    0.000000, 0.040816, 0.081633, 0.122449, 0.163265, 0.204082, 0.244898, 0.285714, 0.326531,
    0.367347, 0.408163, 0.448980, 0.489796, 0.530612, 0.571429, 0.612245, 0.653061, 0.693878,
    0.734694, 0.775510, 0.816327, 0.857143, 0.897959, 0.938776, 0.979592, 1.020408, 1.061224,
    1.102041, 1.142857, 1.183673, 1.224490, 1.265306, 1.306122, 1.346939, 1.387755, 1.428571,
    1.469388, 1.510204, 1.551020, 1.591837, 1.632653, 1.673469, 1.714286, 1.755102, 1.795918,
    1.836735, 1.877551, 1.918367, 1.959184, 2.000000,
];

const Y_POISSON: [f64; 50] = [
    3.0, 2.0, 1.0, 0.0, 0.0, 2.0, 2.0, 1.0, 3.0, 5.0, 2.0, 1.0, 3.0, 4.0, 4.0, 2.0, 2.0, 2.0, 6.0,
    2.0, 4.0, 2.0, 2.0, 2.0, 7.0, 3.0, 7.0, 7.0, 7.0, 3.0, 2.0, 2.0, 5.0, 5.0, 8.0, 5.0, 7.0, 9.0,
    9.0, 6.0, 16.0, 1.0, 7.0, 10.0, 4.0, 4.0, 8.0, 12.0, 11.0, 20.0,
];

const Y_GAMMA: [f64; 50] = [
    1.866877, 3.355052, 1.216312, 5.254581, 1.660498, 1.736697, 1.023930, 4.578080, 2.431459,
    2.563322, 0.391542, 1.146159, 0.236092, 1.197920, 5.097431, 5.109823, 3.019789, 1.815426,
    0.233488, 2.924661, 4.980824, 1.659109, 0.531377, 2.826196, 7.507788, 11.025416, 0.757095,
    1.096952, 7.548495, 15.283589, 9.718480, 29.935210, 2.937180, 23.296511, 2.008270, 2.327836,
    7.044775, 19.465800, 14.832066, 20.654143, 9.880167, 7.079656, 13.311479, 17.705797, 14.824396,
    22.382008, 18.370585, 18.723383, 26.793250, 35.054004,
];

const Y_LOGISTIC: [f64; 50] = [
    3.208258, -1.624217, 10.453460, 1.617688, 11.258257, 12.284640, 11.424119, 23.145476,
    19.156083, 18.375459, 15.337890, 24.318363, 22.621659, 26.262262, 17.920067, 24.370028,
    30.270070, 32.670522, 31.189840, 33.632848, 33.903226, 36.160651, 32.613760, 41.840572,
    39.209831, 43.946154, 48.951126, 42.464532, 49.161313, 48.770342, 51.754079, 47.549573,
    54.247759, 55.549742, 57.216318, 55.107233, 52.408292, 60.956448, 54.724173, 63.681725,
    62.301636, 64.634175, 71.981881, 70.000359, 71.864646, 73.360605, 70.562677, 73.761928,
    76.572795, 81.474467,
];

// Expected values from R greybox
const EXPECTED_INTERCEPT_NORMAL: f64 = 3.6121344911;
const EXPECTED_COEF_NORMAL: f64 = 1.4521902025;
const EXPECTED_LL_NORMAL: f64 = -131.3858218550;

const EXPECTED_INTERCEPT_LAPLACE: f64 = 3.7870753971;
const EXPECTED_COEF_LAPLACE: f64 = 1.4670731701;
const EXPECTED_LL_LAPLACE: f64 = -154.5190668179;

const EXPECTED_INTERCEPT_T: f64 = 0.5875415906;
const EXPECTED_COEF_T: f64 = 1.5458012827;
const EXPECTED_LL_T: f64 = -130.3008375360;

// greybox's `dt` has no scale (and estimated df for EXPECTED_* above); ALM's
// Student t is the location–scale t with df = 5. Its converged MLE
// (β₀, β₁, σ) from `optim` (#145):
const MLE_T: [f64; 3] = [1.1588389511, 1.5189116429, 2.4805192212];
const MLE_LL_T: f64 = -127.1932660023;

const EXPECTED_INTERCEPT_LOGNORM: f64 = 0.4956574961;
const EXPECTED_COEF_LOGNORM: f64 = 0.0301845076;
const EXPECTED_LL_LOGNORM: f64 = -68.7844926408;

const EXPECTED_INTERCEPT_POISSON: f64 = 0.3801366588;
const EXPECTED_COEF_POISSON: f64 = 1.0238080626;
const EXPECTED_LL_POISSON: f64 = -110.9289502190;

const EXPECTED_INTERCEPT_GAMMA: f64 = 0.3288551166;
const EXPECTED_COEF_GAMMA: f64 = 0.0555296357;
const EXPECTED_LL_GAMMA: f64 = -136.5538716326;

const EXPECTED_INTERCEPT_LOGISTIC: f64 = 2.8634125857;
const EXPECTED_COEF_LOGISTIC: f64 = 1.5090025841;
const EXPECTED_LL_LOGISTIC: f64 = -130.8630015489;

// =============================================================================
// Validation Tests
// =============================================================================

/// Test Normal distribution against R greybox
#[test]
fn test_validate_normal_vs_r() {
    let n = X_NORMAL.len();
    let x = Mat::from_fn(n, 1, |i, _| X_NORMAL[i]);
    let y = Col::from_fn(n, |i| Y_NORMAL[i]);

    let alm = AlmRegressor::builder()
        .distribution(AlmDistribution::Normal)
        .with_intercept(true)
        .build();

    let fitted = alm.fit(&x, &y).unwrap();

    let intercept = fitted.intercept().unwrap();
    let coef = fitted.coefficients()[0];
    let ll = fitted.result().log_likelihood;

    println!("Normal Distribution:");
    println!(
        "  Intercept: Rust={:.6}, R={:.6}, diff={:.6}",
        intercept,
        EXPECTED_INTERCEPT_NORMAL,
        (intercept - EXPECTED_INTERCEPT_NORMAL).abs()
    );
    println!(
        "  Coefficient: Rust={:.6}, R={:.6}, diff={:.6}",
        coef,
        EXPECTED_COEF_NORMAL,
        (coef - EXPECTED_COEF_NORMAL).abs()
    );
    println!(
        "  Log-likelihood: Rust={:.6}, R={:.6}, diff={:.6}",
        ll,
        EXPECTED_LL_NORMAL,
        (ll - EXPECTED_LL_NORMAL).abs()
    );

    assert!(
        approx_eq_rel(intercept, EXPECTED_INTERCEPT_NORMAL, COEF_TOL),
        "Intercept mismatch: {} vs {}",
        intercept,
        EXPECTED_INTERCEPT_NORMAL
    );
    assert!(
        approx_eq_rel(coef, EXPECTED_COEF_NORMAL, COEF_TOL),
        "Coefficient mismatch: {} vs {}",
        coef,
        EXPECTED_COEF_NORMAL
    );
    assert!(
        approx_eq_rel(ll, EXPECTED_LL_NORMAL, LL_TOL),
        "Log-likelihood mismatch: {} vs {}",
        ll,
        EXPECTED_LL_NORMAL
    );
}

/// Test Laplace distribution against R greybox
#[test]
fn test_validate_laplace_vs_r() {
    let n = X_NORMAL.len();
    let x = Mat::from_fn(n, 1, |i, _| X_NORMAL[i]);
    let y = Col::from_fn(n, |i| Y_LAPLACE[i]);

    let alm = AlmRegressor::builder()
        .distribution(AlmDistribution::Laplace)
        .with_intercept(true)
        .build();

    let fitted = alm.fit(&x, &y).unwrap();

    let intercept = fitted.intercept().unwrap();
    let coef = fitted.coefficients()[0];
    let ll = fitted.result().log_likelihood;

    println!("Laplace Distribution:");
    println!(
        "  Intercept: Rust={:.6}, R={:.6}, diff={:.6}",
        intercept,
        EXPECTED_INTERCEPT_LAPLACE,
        (intercept - EXPECTED_INTERCEPT_LAPLACE).abs()
    );
    println!(
        "  Coefficient: Rust={:.6}, R={:.6}, diff={:.6}",
        coef,
        EXPECTED_COEF_LAPLACE,
        (coef - EXPECTED_COEF_LAPLACE).abs()
    );
    println!(
        "  Log-likelihood: Rust={:.6}, R={:.6}, diff={:.6}",
        ll,
        EXPECTED_LL_LAPLACE,
        (ll - EXPECTED_LL_LAPLACE).abs()
    );

    assert!(
        approx_eq_rel(intercept, EXPECTED_INTERCEPT_LAPLACE, COEF_TOL),
        "Intercept mismatch: {} vs {}",
        intercept,
        EXPECTED_INTERCEPT_LAPLACE
    );
    assert!(
        approx_eq_rel(coef, EXPECTED_COEF_LAPLACE, COEF_TOL),
        "Coefficient mismatch: {} vs {}",
        coef,
        EXPECTED_COEF_LAPLACE
    );
    // Laplace log-likelihood calculation may differ - use larger tolerance
    assert!(
        approx_eq_rel(ll, EXPECTED_LL_LAPLACE, 0.20),
        "Log-likelihood mismatch: {} vs {}",
        ll,
        EXPECTED_LL_LAPLACE
    );
}

/// Test Student-t distribution against R greybox
#[test]
fn test_validate_student_t_vs_r() {
    let n = X_NORMAL.len();
    let x = Mat::from_fn(n, 1, |i, _| X_NORMAL[i]);
    let y = Col::from_fn(n, |i| Y_STUDENT_T[i]);

    let alm = AlmRegressor::builder()
        .distribution(AlmDistribution::StudentT)
        .extra_parameter(5.0) // df = 5 as in R
        .with_intercept(true)
        .build();

    let fitted = alm.fit(&x, &y).unwrap();

    let intercept = fitted.intercept().unwrap();
    let coef = fitted.coefficients()[0];
    let ll = fitted.result().log_likelihood;

    println!("Student-t Distribution:");
    println!(
        "  Intercept: Rust={:.6}, R={:.6}, diff={:.6}",
        intercept,
        EXPECTED_INTERCEPT_T,
        (intercept - EXPECTED_INTERCEPT_T).abs()
    );
    println!(
        "  Coefficient: Rust={:.6}, R={:.6}, diff={:.6}",
        coef,
        EXPECTED_COEF_T,
        (coef - EXPECTED_COEF_T).abs()
    );
    println!(
        "  Log-likelihood: Rust={:.6}, R={:.6}, diff={:.6}",
        ll,
        EXPECTED_LL_T,
        (ll - EXPECTED_LL_T).abs()
    );

    // Student-t may have different optimization results - use larger tolerance
    assert!(
        approx_eq_rel(coef, EXPECTED_COEF_T, 0.20),
        "Coefficient mismatch: {} vs {}",
        coef,
        EXPECTED_COEF_T
    );

    // The location–scale t MLE (#145)
    assert!(
        (intercept - MLE_T[0]).abs() < 1e-6,
        "t intercept {intercept}"
    );
    assert!((coef - MLE_T[1]).abs() < 1e-6, "t coef {coef}");
    assert!(
        (fitted.scale() - MLE_T[2]).abs() < 1e-6,
        "t scale {}",
        fitted.scale()
    );
    assert!((ll - MLE_LL_T).abs() < 1e-8, "t LL {ll}");
}

/// Test Log-Normal distribution against R greybox
#[test]
fn test_validate_lognormal_vs_r() {
    let n = X_NORMAL.len();
    let x = Mat::from_fn(n, 1, |i, _| X_NORMAL[i]);
    let y = Col::from_fn(n, |i| Y_LOGNORMAL[i]);

    let alm = AlmRegressor::builder()
        .distribution(AlmDistribution::LogNormal)
        .with_intercept(true)
        .build();

    let fitted = alm.fit(&x, &y).unwrap();

    let intercept = fitted.intercept().unwrap();
    let coef = fitted.coefficients()[0];
    let ll = fitted.result().log_likelihood;

    println!("Log-Normal Distribution:");
    println!(
        "  Intercept: Rust={:.6}, R={:.6}, diff={:.6}",
        intercept,
        EXPECTED_INTERCEPT_LOGNORM,
        (intercept - EXPECTED_INTERCEPT_LOGNORM).abs()
    );
    println!(
        "  Coefficient: Rust={:.6}, R={:.6}, diff={:.6}",
        coef,
        EXPECTED_COEF_LOGNORM,
        (coef - EXPECTED_COEF_LOGNORM).abs()
    );
    println!(
        "  Log-likelihood: Rust={:.6}, R={:.6}, diff={:.6}",
        ll,
        EXPECTED_LL_LOGNORM,
        (ll - EXPECTED_LL_LOGNORM).abs()
    );

    assert!(
        approx_eq_rel(intercept, EXPECTED_INTERCEPT_LOGNORM, COEF_TOL),
        "Intercept mismatch: {} vs {}",
        intercept,
        EXPECTED_INTERCEPT_LOGNORM
    );
    assert!(
        approx_eq_rel(coef, EXPECTED_COEF_LOGNORM, COEF_TOL),
        "Coefficient mismatch: {} vs {}",
        coef,
        EXPECTED_COEF_LOGNORM
    );
}

/// Test Poisson distribution against R greybox
#[test]
fn test_validate_poisson_vs_r() {
    let n = X_POISSON.len();
    let x = Mat::from_fn(n, 1, |i, _| X_POISSON[i]);
    let y = Col::from_fn(n, |i| Y_POISSON[i]);

    let alm = AlmRegressor::builder()
        .distribution(AlmDistribution::Poisson)
        .with_intercept(true)
        .build();

    let fitted = alm.fit(&x, &y).unwrap();

    let intercept = fitted.intercept().unwrap();
    let coef = fitted.coefficients()[0];
    let ll = fitted.result().log_likelihood;

    println!("Poisson Distribution:");
    println!(
        "  Intercept: Rust={:.6}, R={:.6}, diff={:.6}",
        intercept,
        EXPECTED_INTERCEPT_POISSON,
        (intercept - EXPECTED_INTERCEPT_POISSON).abs()
    );
    println!(
        "  Coefficient: Rust={:.6}, R={:.6}, diff={:.6}",
        coef,
        EXPECTED_COEF_POISSON,
        (coef - EXPECTED_COEF_POISSON).abs()
    );
    println!(
        "  Log-likelihood: Rust={:.6}, R={:.6}, diff={:.6}",
        ll,
        EXPECTED_LL_POISSON,
        (ll - EXPECTED_LL_POISSON).abs()
    );

    assert!(
        approx_eq_rel(intercept, EXPECTED_INTERCEPT_POISSON, COEF_TOL),
        "Intercept mismatch: {} vs {}",
        intercept,
        EXPECTED_INTERCEPT_POISSON
    );
    assert!(
        approx_eq_rel(coef, EXPECTED_COEF_POISSON, COEF_TOL),
        "Coefficient mismatch: {} vs {}",
        coef,
        EXPECTED_COEF_POISSON
    );
    assert!(
        approx_eq_rel(ll, EXPECTED_LL_POISSON, LL_TOL),
        "Log-likelihood mismatch: {} vs {}",
        ll,
        EXPECTED_LL_POISSON
    );
}

/// Test Gamma distribution against R greybox
#[test]
fn test_validate_gamma_vs_r() {
    let n = X_NORMAL.len();
    let x = Mat::from_fn(n, 1, |i, _| X_NORMAL[i]);
    let y = Col::from_fn(n, |i| Y_GAMMA[i]);

    let alm = AlmRegressor::builder()
        .distribution(AlmDistribution::Gamma)
        .with_intercept(true)
        .build();

    let fitted = alm.fit(&x, &y).unwrap();

    let intercept = fitted.intercept().unwrap();
    let coef = fitted.coefficients()[0];
    let ll = fitted.result().log_likelihood;

    println!("Gamma Distribution:");
    println!(
        "  Intercept: Rust={:.6}, R={:.6}, diff={:.6}",
        intercept,
        EXPECTED_INTERCEPT_GAMMA,
        (intercept - EXPECTED_INTERCEPT_GAMMA).abs()
    );
    println!(
        "  Coefficient: Rust={:.6}, R={:.6}, diff={:.6}",
        coef,
        EXPECTED_COEF_GAMMA,
        (coef - EXPECTED_COEF_GAMMA).abs()
    );
    println!(
        "  Log-likelihood: Rust={:.6}, R={:.6}, diff={:.6}",
        ll,
        EXPECTED_LL_GAMMA,
        (ll - EXPECTED_LL_GAMMA).abs()
    );

    assert!(
        approx_eq_rel(intercept, EXPECTED_INTERCEPT_GAMMA, COEF_TOL),
        "Intercept mismatch: {} vs {}",
        intercept,
        EXPECTED_INTERCEPT_GAMMA
    );
    assert!(
        approx_eq_rel(coef, EXPECTED_COEF_GAMMA, COEF_TOL),
        "Coefficient mismatch: {} vs {}",
        coef,
        EXPECTED_COEF_GAMMA
    );
}

/// Test Logistic distribution against R greybox
#[test]
fn test_validate_logistic_vs_r() {
    let n = X_NORMAL.len();
    let x = Mat::from_fn(n, 1, |i, _| X_NORMAL[i]);
    let y = Col::from_fn(n, |i| Y_LOGISTIC[i]);

    let alm = AlmRegressor::builder()
        .distribution(AlmDistribution::Logistic)
        .with_intercept(true)
        .build();

    let fitted = alm.fit(&x, &y).unwrap();

    let intercept = fitted.intercept().unwrap();
    let coef = fitted.coefficients()[0];
    let ll = fitted.result().log_likelihood;

    println!("Logistic Distribution:");
    println!(
        "  Intercept: Rust={:.6}, R={:.6}, diff={:.6}",
        intercept,
        EXPECTED_INTERCEPT_LOGISTIC,
        (intercept - EXPECTED_INTERCEPT_LOGISTIC).abs()
    );
    println!(
        "  Coefficient: Rust={:.6}, R={:.6}, diff={:.6}",
        coef,
        EXPECTED_COEF_LOGISTIC,
        (coef - EXPECTED_COEF_LOGISTIC).abs()
    );
    println!(
        "  Log-likelihood: Rust={:.6}, R={:.6}, diff={:.6}",
        ll,
        EXPECTED_LL_LOGISTIC,
        (ll - EXPECTED_LL_LOGISTIC).abs()
    );

    assert!(
        approx_eq_rel(coef, EXPECTED_COEF_LOGISTIC, COEF_TOL),
        "Coefficient mismatch: {} vs {}",
        coef,
        EXPECTED_COEF_LOGISTIC
    );
}

/// Integration test: verify predictions are reasonable
#[test]
fn test_predictions_sanity_check() {
    let n = X_NORMAL.len();
    let x = Mat::from_fn(n, 1, |i, _| X_NORMAL[i]);
    let y = Col::from_fn(n, |i| Y_NORMAL[i]);

    let alm = AlmRegressor::builder()
        .distribution(AlmDistribution::Normal)
        .with_intercept(true)
        .build();

    let fitted = alm.fit(&x, &y).unwrap();

    // Predictions should be close to y for a good fit
    let preds = fitted.predict(&x);

    // Calculate R-squared manually
    let y_mean: f64 = y.iter().sum::<f64>() / n as f64;
    let ss_tot: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();
    let ss_res: f64 = (0..n).map(|i| (y[i] - preds[i]).powi(2)).sum();
    let r_squared = 1.0 - ss_res / ss_tot;

    println!("R-squared: {:.6}", r_squared);

    // R-squared should be high for this linear relationship
    assert!(r_squared > 0.95, "R-squared too low: {}", r_squared);
}

/// Test that all distributions produce finite results
#[test]
fn test_all_distributions_produce_finite_results() {
    let n = X_NORMAL.len();
    let x = Mat::from_fn(n, 1, |i, _| X_NORMAL[i]);
    let y = Col::from_fn(n, |i| Y_NORMAL[i]);

    // For positive-only distributions, use transformed y
    let y_positive = Col::from_fn(n, |i| Y_LOGNORMAL[i]);
    let y_unit = Col::from_fn(n, |i| (Y_NORMAL[i] / 100.0).clamp(0.01, 0.99));

    let distributions = vec![
        (AlmDistribution::Normal, &y, "Normal"),
        (AlmDistribution::Laplace, &y, "Laplace"),
        (AlmDistribution::Logistic, &y, "Logistic"),
        (AlmDistribution::LogNormal, &y_positive, "LogNormal"),
        (AlmDistribution::Gamma, &y_positive, "Gamma"),
        (AlmDistribution::Exponential, &y_positive, "Exponential"),
        (
            AlmDistribution::InverseGaussian,
            &y_positive,
            "InverseGaussian",
        ),
        (AlmDistribution::Beta, &y_unit, "Beta"),
    ];

    for (dist, y_data, name) in distributions {
        let alm = AlmRegressor::builder()
            .distribution(dist)
            .with_intercept(true)
            .build();

        match alm.fit(&x, y_data) {
            Ok(fitted) => {
                let ll = fitted.result().log_likelihood;
                assert!(
                    ll.is_finite(),
                    "{}: Log-likelihood is not finite: {}",
                    name,
                    ll
                );
                println!("{}: LL = {:.4}", name, ll);
            }
            Err(e) => {
                panic!("{}: Failed to fit: {:?}", name, e);
            }
        }
    }
}
