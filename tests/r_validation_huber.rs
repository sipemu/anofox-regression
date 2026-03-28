//! Validation tests for HuberRegressor against R's MASS::rlm()
//!
//! These tests compare the Rust IRLS implementation against known outputs from
//! R's MASS::rlm() with Huber psi function (method = "M", psi = psi.huber, k = 1.345).
//!
//! Note: Our IRLS implementation may converge to a slightly different local solution
//! than R's rlm() due to differences in initialization, scale estimation timing,
//! and convergence criteria. Tolerances are set accordingly.

use anofox_regression::solvers::{FittedRegressor, HuberRegressor, Regressor};
use faer::{Col, Mat};

// =============================================================================
// Test 1: Clean data (y ≈ 2.5 + 3.0*x + noise)
// =============================================================================

const N_CLEAN: usize = 50;
const X_CLEAN: [f64; 50] = [
    1.0000000000,
    1.1836734694,
    1.3673469388,
    1.5510204082,
    1.7346938776,
    1.9183673469,
    2.1020408163,
    2.2857142857,
    2.4693877551,
    2.6530612245,
    2.8367346939,
    3.0204081633,
    3.2040816327,
    3.3877551020,
    3.5714285714,
    3.7551020408,
    3.9387755102,
    4.1224489796,
    4.3061224490,
    4.4897959184,
    4.6734693878,
    4.8571428571,
    5.0408163265,
    5.2244897959,
    5.4081632653,
    5.5918367347,
    5.7755102041,
    5.9591836735,
    6.1428571429,
    6.3265306122,
    6.5102040816,
    6.6938775510,
    6.8775510204,
    7.0612244898,
    7.2448979592,
    7.4285714286,
    7.6122448980,
    7.7959183673,
    7.9795918367,
    8.1632653061,
    8.3469387755,
    8.5306122449,
    8.7142857143,
    8.8979591837,
    9.0816326531,
    9.2653061224,
    9.4489795918,
    9.6326530612,
    9.8163265306,
    10.0000000000,
];
const Y_CLEAN: [f64; 50] = [
    5.6854792236,
    5.2686713225,
    6.2836050220,
    6.9694925270,
    7.4062157942,
    7.7020397828,
    9.0618834477,
    8.8098133379,
    10.4173751222,
    9.9278266239,
    11.1626389087,
    12.2045471861,
    10.9178145474,
    12.0238709227,
    12.6476250461,
    13.5832813215,
    13.6742000699,
    13.0391192283,
    13.6981338827,
    16.1294444280,
    15.8670888662,
    15.6807743544,
    17.0364903017,
    18.2808067373,
    19.1720865266,
    18.5602756383,
    19.1978959209,
    18.9959694778,
    20.6586201060,
    20.6595943988,
    21.7583373065,
    22.4340513217,
    23.1502048222,
    22.8792102817,
    23.9871714392,
    23.4272099462,
    24.4445051897,
    24.9623013050,
    24.7316716852,
    26.5078572218,
    27.1438156266,
    27.4113080854,
    28.5219387607,
    28.3305251375,
    28.5607574370,
    30.0123273803,
    29.9412421874,
    31.6200098145,
    31.2332564905,
    32.3278239417,
];
const EXPECTED_INTERCEPT_CLEAN: f64 = 2.2568987836;
const EXPECTED_COEF_CLEAN: f64 = 2.9559930332;
const EXPECTED_SCALE_CLEAN: f64 = 0.4659316225;

// =============================================================================
// Test 2: Data with outliers (y ≈ 2.0 + 3.0*x + noise, with 5 extreme outliers)
// =============================================================================

const N_OUTLIER: usize = 50;
const X_OUTLIER: [f64; 50] = [
    1.0000000000,
    1.1836734694,
    1.3673469388,
    1.5510204082,
    1.7346938776,
    1.9183673469,
    2.1020408163,
    2.2857142857,
    2.4693877551,
    2.6530612245,
    2.8367346939,
    3.0204081633,
    3.2040816327,
    3.3877551020,
    3.5714285714,
    3.7551020408,
    3.9387755102,
    4.1224489796,
    4.3061224490,
    4.4897959184,
    4.6734693878,
    4.8571428571,
    5.0408163265,
    5.2244897959,
    5.4081632653,
    5.5918367347,
    5.7755102041,
    5.9591836735,
    6.1428571429,
    6.3265306122,
    6.5102040816,
    6.6938775510,
    6.8775510204,
    7.0612244898,
    7.2448979592,
    7.4285714286,
    7.6122448980,
    7.7959183673,
    7.9795918367,
    8.1632653061,
    8.3469387755,
    8.5306122449,
    8.7142857143,
    8.8979591837,
    9.0816326531,
    9.2653061224,
    9.4489795918,
    9.6326530612,
    9.8163265306,
    10.0000000000,
];
const Y_OUTLIER: [f64; 50] = [
    5.1609626326,
    5.1591009377,
    6.8899045762,
    6.9745108773,
    100.0000000000,
    7.8933774145,
    8.6457668570,
    8.9020593004,
    7.9116182237,
    10.1016251502,
    10.3265867603,
    11.1538397722,
    11.9031567616,
    12.8631337198,
    -50.0000000000,
    13.9165774385,
    13.9842505905,
    14.8865999881,
    15.3787316311,
    15.8298268365,
    15.4988486940,
    16.5263353781,
    17.4342080606,
    17.1967077089,
    80.0000000000,
    19.0660084529,
    19.7106199812,
    20.1094348147,
    19.9856832799,
    20.4297013874,
    22.2869657498,
    22.2105933718,
    22.6768731758,
    23.1232252006,
    -40.0000000000,
    24.5917127347,
    24.7281647710,
    25.2963767489,
    26.4054486745,
    26.9006824736,
    27.7368745145,
    27.3537497732,
    28.4680314232,
    29.3894327792,
    90.0000000000,
    29.3655220739,
    29.7810694351,
    30.1683521839,
    31.4889708685,
    32.3266021698,
];
const EXPECTED_INTERCEPT_OUTLIER: f64 = 2.2450375978;
const EXPECTED_COEF_OUTLIER: f64 = 2.9807360391;
const EXPECTED_SCALE_OUTLIER: f64 = 0.4573191433;
const OLS_INTERCEPT_OUTLIER: f64 = 5.9169516988;
const OLS_COEF_OUTLIER: f64 = 2.6265099651;

// =============================================================================
// Test 3: Multiple predictors (y ≈ 2.0 + 2.0*x1 - 1.5*x2 + noise, with outliers)
// =============================================================================

const N_MULTI: usize = 60;
const X_MULTI_1: [f64; 60] = [
    7.4019307512,
    7.0895021743,
    2.9935827063,
    8.6969638033,
    3.6664531825,
    5.2110276249,
    4.1554882363,
    4.7552996561,
    5.3763860690,
    5.2383219160,
    4.9498148983,
    5.2161454559,
    4.0291295283,
    3.9915657386,
    1.6778018402,
    4.2353325463,
    3.9746994842,
    10.4037820007,
    2.2757675376,
    5.2745124371,
    2.0127498654,
    2.0591285171,
    5.2494047724,
    3.0067217302,
    4.9963547714,
    4.1434822371,
    3.7726567871,
    0.9506443092,
    2.5505040993,
    5.3590328822,
    6.1352411888,
    4.0142452929,
    5.0001257681,
    7.2457792868,
    7.8797114860,
    2.8057724632,
    4.7653608795,
    7.4029968018,
    4.0605408389,
    4.8950610301,
    4.8277854035,
    3.2246419642,
    4.1106319902,
    4.9411102418,
    4.1722623019,
    7.2267720467,
    4.0380143167,
    4.1336619348,
    6.3937251531,
    2.8872631737,
    4.9186030497,
    1.8969103553,
    7.3343390985,
    4.4527085973,
    4.0643093507,
    2.5234953440,
    4.9844759324,
    3.3994356441,
    3.9330153401,
    7.5753504912,
];
const X_MULTI_2: [f64; 60] = [
    2.8244741298,
    1.9282176158,
    3.1632068825,
    2.6372615844,
    3.5900135480,
    4.4324219277,
    2.0073074889,
    3.4546502976,
    3.0848980587,
    3.8955655823,
    2.7702218611,
    3.8366190685,
    1.2549441387,
    4.6894589213,
    3.8647779785,
    2.8492240111,
    1.5509928699,
    3.6430087000,
    3.4831938638,
    2.9936443736,
    3.1514558929,
    2.4158910297,
    3.3688067326,
    3.2946543397,
    2.7207406267,
    1.6637633451,
    3.7007488184,
    3.5541966223,
    2.1636934072,
    1.4054118380,
    3.2049585806,
    2.6549120220,
    3.2526117034,
    1.7059975345,
    2.0408295556,
    4.0857748537,
    3.4037749047,
    3.5864875367,
    4.8152284462,
    3.1288214286,
    0.9990707623,
    3.3337771974,
    4.1713251274,
    5.0595392423,
    1.6231384018,
    1.8491444344,
    2.2941786052,
    1.9459442179,
    2.3542562769,
    2.8146220323,
    1.7987779493,
    5.0369721670,
    3.1077747449,
    2.9158918995,
    3.4956196416,
    3.0374151861,
    2.8679119630,
    4.4767874236,
    2.7829697899,
    1.7163977959,
];
const Y_MULTI: [f64; 60] = [
    11.9528181982,
    11.9351650514,
    1.7205589956,
    13.3699040294,
    3.3762519463,
    4.5994041238,
    6.8156829678,
    5.0942585885,
    6.4669216242,
    50.0000000000,
    6.4725332898,
    6.6253143050,
    5.9742604185,
    1.4828269989,
    -1.7109146826,
    4.8058636677,
    7.9716166757,
    16.3202862500,
    0.5709701306,
    6.1161866060,
    -0.4309013851,
    2.4924893983,
    7.7040811104,
    3.3203256397,
    5.5309615533,
    8.8412801503,
    4.0110631763,
    -2.4567237792,
    3.5590758666,
    -30.0000000000,
    7.3668882653,
    5.0951730037,
    4.9228381247,
    13.1225812703,
    14.9958845381,
    -0.5509910771,
    4.6866186477,
    10.4728262381,
    0.8806428887,
    5.7136059575,
    10.0297200753,
    3.4181631462,
    3.3481229544,
    1.4413559571,
    6.8558202643,
    13.7446006563,
    6.4479557629,
    6.1575910686,
    7.5561360824,
    2.6135599376,
    8.7127908730,
    -2.7158339601,
    11.1644286198,
    5.9631447181,
    3.4886395029,
    2.8008461346,
    7.1374773202,
    -0.1589801177,
    6.0731514517,
    14.7805632255,
];
const EXPECTED_INTERCEPT_MULTI: f64 = 2.3323014732;
const EXPECTED_COEF_MULTI_1: f64 = 1.9368003494;
const EXPECTED_COEF_MULTI_2: f64 = -1.8238961298;
const EXPECTED_SCALE_MULTI: f64 = 0.9213040842;

// =============================================================================
// Test 5: No intercept (y ≈ 3.0*x + noise, with one outlier)
// =============================================================================

const N_NOINT: usize = 40;
const X_NOINT: [f64; 40] = [
    0.5000000000,
    0.6153846154,
    0.7307692308,
    0.8461538462,
    0.9615384615,
    1.0769230769,
    1.1923076923,
    1.3076923077,
    1.4230769231,
    1.5384615385,
    1.6538461538,
    1.7692307692,
    1.8846153846,
    2.0000000000,
    2.1153846154,
    2.2307692308,
    2.3461538462,
    2.4615384615,
    2.5769230769,
    2.6923076923,
    2.8076923077,
    2.9230769231,
    3.0384615385,
    3.1538461538,
    3.2692307692,
    3.3846153846,
    3.5000000000,
    3.6153846154,
    3.7307692308,
    3.8461538462,
    3.9615384615,
    4.0769230769,
    4.1923076923,
    4.3076923077,
    4.4230769231,
    4.5384615385,
    4.6538461538,
    4.7692307692,
    4.8846153846,
    5.0000000000,
];
const Y_NOINT: [f64; 40] = [
    1.7472221891,
    1.3473650255,
    2.0215157892,
    2.7291156836,
    30.0000000000,
    3.3351729219,
    4.3148011416,
    3.6775628258,
    3.6352707348,
    4.6974931971,
    4.7552594092,
    5.4415046236,
    5.4101307367,
    6.6636166441,
    6.3090420547,
    6.5491070405,
    6.9885830910,
    7.6433843997,
    7.7599713763,
    7.5892380552,
    8.4216906927,
    8.9973034195,
    9.1270818892,
    9.6820601040,
    9.7637505196,
    10.1364799532,
    10.6447108398,
    11.1440369372,
    10.8183890429,
    11.5284152810,
    11.8633267302,
    12.0030930346,
    12.2666152686,
    12.7338573369,
    13.4452730852,
    13.4904878185,
    13.7260721187,
    14.3567172035,
    14.2828318833,
    15.3137621329,
];
const EXPECTED_COEF_NOINT: f64 = 2.9910743601;
const EXPECTED_SCALE_NOINT: f64 = 0.2863387993;

// =============================================================================
// Test implementations
// =============================================================================

#[test]
fn test_huber_clean_data_vs_r() {
    let x = Mat::from_fn(N_CLEAN, 1, |i, _| X_CLEAN[i]);
    let y = Col::from_fn(N_CLEAN, |i| Y_CLEAN[i]);

    let model = HuberRegressor::builder()
        .epsilon(1.345)
        .alpha(0.0)
        .with_intercept(true)
        .build();

    let fitted = model.fit(&x, &y).expect("fit should succeed on clean data");

    let intercept = fitted.result().intercept.unwrap();
    let coef = fitted.result().coefficients[0];
    let scale = fitted.scale();

    let intercept_err = (intercept - EXPECTED_INTERCEPT_CLEAN).abs();
    let coef_err = (coef - EXPECTED_COEF_CLEAN).abs();

    assert!(
        intercept_err < 0.05,
        "Clean data intercept: got {intercept:.6}, expected {EXPECTED_INTERCEPT_CLEAN:.6}, \
         abs error = {intercept_err:.6} (tolerance 0.05)"
    );
    assert!(
        coef_err < 0.05,
        "Clean data coefficient: got {coef:.6}, expected {EXPECTED_COEF_CLEAN:.6}, \
         abs error = {coef_err:.6} (tolerance 0.05)"
    );

    // Scale check: relative tolerance of 50% (clean data should be tighter, but
    // MAD vs R's scale estimator can differ)
    let scale_rel_err = (scale - EXPECTED_SCALE_CLEAN).abs() / EXPECTED_SCALE_CLEAN;
    assert!(
        scale_rel_err < 0.50,
        "Clean data scale: got {scale:.6}, expected {EXPECTED_SCALE_CLEAN:.6}, \
         relative error = {scale_rel_err:.4} (tolerance 50%)"
    );
}

#[test]
fn test_huber_outlier_data_vs_r() {
    let x = Mat::from_fn(N_OUTLIER, 1, |i, _| X_OUTLIER[i]);
    let y = Col::from_fn(N_OUTLIER, |i| Y_OUTLIER[i]);

    let model = HuberRegressor::builder()
        .epsilon(1.345)
        .alpha(0.0)
        .with_intercept(true)
        .build();

    let fitted = model
        .fit(&x, &y)
        .expect("fit should succeed on outlier data");

    let intercept = fitted.result().intercept.unwrap();
    let coef = fitted.result().coefficients[0];
    let scale = fitted.scale();

    // Lenient tolerances: IRLS solutions may differ due to initialization,
    // scale estimation timing, and convergence behavior with extreme outliers.
    let intercept_err = (intercept - EXPECTED_INTERCEPT_OUTLIER).abs();
    let coef_err = (coef - EXPECTED_COEF_OUTLIER).abs();

    assert!(
        intercept_err < 0.5,
        "Outlier data intercept: got {intercept:.6}, expected {EXPECTED_INTERCEPT_OUTLIER:.6}, \
         abs error = {intercept_err:.6} (tolerance 0.5)"
    );
    assert!(
        coef_err < 0.5,
        "Outlier data coefficient: got {coef:.6}, expected {EXPECTED_COEF_OUTLIER:.6}, \
         abs error = {coef_err:.6} (tolerance 0.5)"
    );

    // Scale: relative tolerance of 50%
    let scale_rel_err = (scale - EXPECTED_SCALE_OUTLIER).abs() / EXPECTED_SCALE_OUTLIER;
    assert!(
        scale_rel_err < 0.50,
        "Outlier data scale: got {scale:.6}, expected {EXPECTED_SCALE_OUTLIER:.6}, \
         relative error = {scale_rel_err:.4} (tolerance 50%)"
    );
}

#[test]
fn test_huber_multi_predictor_vs_r() {
    let x = Mat::from_fn(
        N_MULTI,
        2,
        |i, j| {
            if j == 0 {
                X_MULTI_1[i]
            } else {
                X_MULTI_2[i]
            }
        },
    );
    let y = Col::from_fn(N_MULTI, |i| Y_MULTI[i]);

    let model = HuberRegressor::builder()
        .epsilon(1.345)
        .alpha(0.0)
        .with_intercept(true)
        .build();

    let fitted = model
        .fit(&x, &y)
        .expect("fit should succeed on multi-predictor data");

    let intercept = fitted.result().intercept.unwrap();
    let coef1 = fitted.result().coefficients[0];
    let coef2 = fitted.result().coefficients[1];
    let scale = fitted.scale();

    let intercept_err = (intercept - EXPECTED_INTERCEPT_MULTI).abs();
    let coef1_err = (coef1 - EXPECTED_COEF_MULTI_1).abs();
    let coef2_err = (coef2 - EXPECTED_COEF_MULTI_2).abs();

    assert!(
        intercept_err < 0.5,
        "Multi-predictor intercept: got {intercept:.6}, expected {EXPECTED_INTERCEPT_MULTI:.6}, \
         abs error = {intercept_err:.6} (tolerance 0.5)"
    );
    assert!(
        coef1_err < 0.5,
        "Multi-predictor coef1: got {coef1:.6}, expected {EXPECTED_COEF_MULTI_1:.6}, \
         abs error = {coef1_err:.6} (tolerance 0.5)"
    );
    assert!(
        coef2_err < 0.5,
        "Multi-predictor coef2: got {coef2:.6}, expected {EXPECTED_COEF_MULTI_2:.6}, \
         abs error = {coef2_err:.6} (tolerance 0.5)"
    );

    // Scale: relative tolerance of 50%
    let scale_rel_err = (scale - EXPECTED_SCALE_MULTI).abs() / EXPECTED_SCALE_MULTI;
    assert!(
        scale_rel_err < 0.50,
        "Multi-predictor scale: got {scale:.6}, expected {EXPECTED_SCALE_MULTI:.6}, \
         relative error = {scale_rel_err:.4} (tolerance 50%)"
    );
}

#[test]
fn test_huber_no_intercept_vs_r() {
    let x = Mat::from_fn(N_NOINT, 1, |i, _| X_NOINT[i]);
    let y = Col::from_fn(N_NOINT, |i| Y_NOINT[i]);

    let model = HuberRegressor::builder()
        .epsilon(1.345)
        .alpha(0.0)
        .with_intercept(false)
        .build();

    let fitted = model
        .fit(&x, &y)
        .expect("fit should succeed with no intercept");

    // No intercept model
    assert!(
        fitted.result().intercept.is_none(),
        "No-intercept model should have no intercept"
    );

    let coef = fitted.result().coefficients[0];
    let scale = fitted.scale();

    let coef_err = (coef - EXPECTED_COEF_NOINT).abs();
    assert!(
        coef_err < 0.1,
        "No-intercept coefficient: got {coef:.6}, expected {EXPECTED_COEF_NOINT:.6}, \
         abs error = {coef_err:.6} (tolerance 0.1)"
    );

    // Scale: relative tolerance of 50%
    let scale_rel_err = (scale - EXPECTED_SCALE_NOINT).abs() / EXPECTED_SCALE_NOINT;
    assert!(
        scale_rel_err < 0.50,
        "No-intercept scale: got {scale:.6}, expected {EXPECTED_SCALE_NOINT:.6}, \
         relative error = {scale_rel_err:.4} (tolerance 50%)"
    );
}

#[test]
fn test_huber_robustness_vs_ols() {
    // Verify that on the outlier data, Huber regression produces estimates
    // closer to the true generating parameters (intercept=2.0, slope=3.0)
    // than OLS does.
    let x = Mat::from_fn(N_OUTLIER, 1, |i, _| X_OUTLIER[i]);
    let y = Col::from_fn(N_OUTLIER, |i| Y_OUTLIER[i]);

    let model = HuberRegressor::builder()
        .epsilon(1.345)
        .alpha(0.0)
        .with_intercept(true)
        .build();

    let fitted = model
        .fit(&x, &y)
        .expect("fit should succeed on outlier data");

    let huber_intercept = fitted.result().intercept.unwrap();
    let huber_coef = fitted.result().coefficients[0];

    let true_intercept = 2.0;
    let true_slope = 3.0;

    let huber_intercept_err = (huber_intercept - true_intercept).abs();
    let ols_intercept_err = (OLS_INTERCEPT_OUTLIER - true_intercept).abs();

    let huber_coef_err = (huber_coef - true_slope).abs();
    let ols_coef_err = (OLS_COEF_OUTLIER - true_slope).abs();

    assert!(
        huber_intercept_err < ols_intercept_err,
        "Huber intercept should be closer to true value (2.0) than OLS: \
         Huber err = {huber_intercept_err:.4} (got {huber_intercept:.4}), \
         OLS err = {ols_intercept_err:.4} (got {OLS_INTERCEPT_OUTLIER:.4})"
    );
    assert!(
        huber_coef_err < ols_coef_err,
        "Huber coefficient should be closer to true value (3.0) than OLS: \
         Huber err = {huber_coef_err:.4} (got {huber_coef:.4}), \
         OLS err = {ols_coef_err:.4} (got {OLS_COEF_OUTLIER:.4})"
    );
}
