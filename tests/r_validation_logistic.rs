//! R Validation Tests for LogisticRegression.
//!
//! These tests compare anofox-regression's LogisticRegression against R's
//! glm(family = binomial(link = "logit")) output. Since LogisticRegression
//! wraps BinomialRegressor (which is already R-validated), coefficients
//! should match very closely.

use anofox_regression::solvers::LogisticRegression;
use faer::{Col, Mat};

// ============================================================================
// R-generated validation data
// ============================================================================

// Test 1: Simple binary classification
const N_SIMPLE: usize = 40;
const X_SIMPLE: [f64; 40] = [
    -3.0000000000,
    -2.8461538462,
    -2.6923076923,
    -2.5384615385,
    -2.3846153846,
    -2.2307692308,
    -2.0769230769,
    -1.9230769231,
    -1.7692307692,
    -1.6153846154,
    -1.4615384615,
    -1.3076923077,
    -1.1538461538,
    -1.0000000000,
    -0.8461538462,
    -0.6923076923,
    -0.5384615385,
    -0.3846153846,
    -0.2307692308,
    -0.0769230769,
    0.0769230769,
    0.2307692308,
    0.3846153846,
    0.5384615385,
    0.6923076923,
    0.8461538462,
    1.0000000000,
    1.1538461538,
    1.3076923077,
    1.4615384615,
    1.6153846154,
    1.7692307692,
    1.9230769231,
    2.0769230769,
    2.2307692308,
    2.3846153846,
    2.5384615385,
    2.6923076923,
    2.8461538462,
    3.0000000000,
];
const Y_SIMPLE: [f64; 40] = [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0,
    1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0,
];
const EXPECTED_INTERCEPT_SIMPLE: f64 = 0.2265172505;
const EXPECTED_COEF_SIMPLE: f64 = 1.4372007440;
const EXPECTED_PROBS_SIMPLE: [f64; 40] = [
    0.0165434211,
    0.0205530813,
    0.0255093625,
    0.0316222408,
    0.0391411283,
    0.0483585199,
    0.0596118609,
    0.0732822693,
    0.0897883004,
    0.1095725520,
    0.1330787696,
    0.1607175118,
    0.1928197323,
    0.2295801370,
    0.2709958956,
    0.3168106277,
    0.3664771088,
    0.4191527829,
    0.4737380752,
    0.5289583926,
    0.5834788962,
    0.6360310823,
    0.6855263756,
    0.7311361496,
    0.7723278325,
    0.8088585286,
    0.8407364684,
    0.8681644453,
    0.8914785578,
    0.9110919726,
    0.9274490793,
    0.9409917456,
    0.9521369392,
    0.9612637378,
    0.9687073846,
    0.9747582041,
    0.9796635840,
    0.9836316818,
    0.9868359174,
    0.9894196440,
];
const EXPECTED_CLASSES_SIMPLE: [f64; 40] = [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0,
];
const EXPECTED_ACCURACY_SIMPLE: f64 = 0.8000000000;
const EXPECTED_LOGODDS_SIMPLE: [f64; 40] = [
    -4.0850849815,
    -3.8639771748,
    -3.6428693680,
    -3.4217615612,
    -3.2006537544,
    -2.9795459477,
    -2.7584381409,
    -2.5373303341,
    -2.3162225274,
    -2.0951147206,
    -1.8740069138,
    -1.6528991070,
    -1.4317913003,
    -1.2106834935,
    -0.9895756867,
    -0.7684678800,
    -0.5473600732,
    -0.3262522664,
    -0.1051444596,
    0.1159633471,
    0.3370711539,
    0.5581789607,
    0.7792867674,
    1.0003945742,
    1.2215023810,
    1.4426101877,
    1.6637179945,
    1.8848258013,
    2.1059336081,
    2.3270414148,
    2.5481492216,
    2.7692570284,
    2.9903648351,
    3.2114726419,
    3.4325804487,
    3.6536882555,
    3.8747960622,
    4.0959038690,
    4.3170116758,
    4.5381194825,
];

// Test 2: Multiple predictors
const N_MULTI: usize = 50;
const X_MULTI_1: [f64; 50] = [
    -0.3066385941,
    -1.7813084340,
    -0.1719173558,
    1.2146746992,
    1.8951934613,
    -0.4304691316,
    -0.2572693828,
    -1.7631630852,
    0.4600973548,
    -0.6399948760,
    0.4554501232,
    0.7048373372,
    1.0351035220,
    -0.6089263754,
    0.5049551233,
    -1.7170086791,
    -0.7844590084,
    -0.8509075942,
    -2.4142076499,
    0.0361226069,
    0.2059986002,
    -0.3610572985,
    0.7581632357,
    -0.7267048271,
    -1.3682810444,
    0.4328180259,
    -0.8113931762,
    1.4441012617,
    -0.4314462026,
    0.6556478834,
    0.3219252652,
    -0.7838389409,
    1.5757275198,
    0.6428993057,
    0.0897606466,
    0.2765507473,
    0.6792888161,
    0.0898328866,
    -2.9930900832,
    0.2848829535,
    -0.3672346427,
    0.1852305649,
    0.5818237274,
    1.3997368273,
    -0.7272920595,
    1.3025426320,
    0.3358481198,
    1.0385060987,
    0.9207285683,
    0.7208781629,
];
const X_MULTI_2: [f64; 50] = [
    -1.0431189386,
    -0.0901863866,
    0.6235181620,
    -0.9535233578,
    -0.5428288146,
    0.5809964977,
    0.7681787378,
    0.4637675885,
    -0.8857762974,
    -1.0997808986,
    1.5127070098,
    0.2579214375,
    0.0884402292,
    -0.1208965375,
    -1.1943288952,
    0.6119968980,
    -0.2171398457,
    -0.1827567063,
    0.9333463286,
    0.8217731105,
    1.3921163759,
    -0.4761739231,
    0.6503485607,
    1.3911104564,
    -1.1107888794,
    -0.8607925869,
    -1.1317386809,
    -1.4592139995,
    0.0799825532,
    0.6532043396,
    1.2009653756,
    1.0447510872,
    -1.0032086468,
    1.8484819017,
    -0.6667734088,
    0.1055138125,
    -0.4222558819,
    -0.1223501720,
    0.1881930345,
    0.1191609580,
    -0.0250925509,
    0.1080727279,
    -0.4854352358,
    -0.5042171307,
    -1.6610990799,
    -0.3823337269,
    -0.5126502579,
    2.7018910003,
    -1.3621162312,
    0.1372562186,
];
const Y_MULTI: [f64; 50] = [
    0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0,
    0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
];
const EXPECTED_INTERCEPT_MULTI: f64 = -0.4108141179;
const EXPECTED_COEF_MULTI_1: f64 = 1.5589989796;
const EXPECTED_COEF_MULTI_2: f64 = 1.4005130486;
const EXPECTED_ACCURACY_MULTI: f64 = 0.8200000000;

// Test 3: Well-separated data
const N_SEP: usize = 30;
const X_SEP: [f64; 30] = [
    -1.4433069883,
    -2.2404964208,
    -2.2165845163,
    -1.6515687117,
    -2.5281842066,
    -2.0203492376,
    -2.7757724112,
    -1.4164152254,
    -2.1368228507,
    -2.2339226623,
    -2.6191261640,
    -2.0038810169,
    -2.4001410890,
    -2.2667461650,
    -1.3561623772,
    1.9122370649,
    1.4641088079,
    2.0816034412,
    1.8186307922,
    2.2950067740,
    2.7162109639,
    1.5036537444,
    2.2273251488,
    2.0424490293,
    2.4477827911,
    1.8851109305,
    2.4183095342,
    1.1274720693,
    2.8447294607,
    2.4323889893,
];
const Y_SEP: [f64; 30] = [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
];
const EXPECTED_INTERCEPT_SEP: f64 = 1.8981313024;
const EXPECTED_COEF_SEP: f64 = 18.7226405374;
const EXPECTED_ACCURACY_SEP: f64 = 1.0000000000;

// Test 4: Regularization check
const UNREG_INTERCEPT: f64 = 0.2265172505;
const UNREG_COEF: f64 = 1.4372007440;

// ============================================================================
// Tests
// ============================================================================

/// Test 1: Simple logistic regression coefficients vs R's glm().
///
/// R code:
/// ```r
/// x <- seq(-3, 3, length.out = 40)
/// y <- c(0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1,0,1,1,
///        0,1,0,0,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1)
/// fit <- glm(y ~ x, family = binomial(link = "logit"))
/// coef(fit)
/// ```
#[test]
fn test_logistic_simple_coefficients_vs_r() {
    let x = Mat::from_fn(N_SIMPLE, 1, |i, _| X_SIMPLE[i]);
    let y = Col::from_fn(N_SIMPLE, |i| Y_SIMPLE[i]);

    let model = LogisticRegression::builder().with_intercept(true).build();
    let fitted = model.fit(&x, &y).unwrap();

    let intercept = fitted.intercept().unwrap();
    let coef = fitted.coefficients()[0];

    assert!(
        (intercept - EXPECTED_INTERCEPT_SIMPLE).abs() < 0.01,
        "intercept: expected {}, got {}",
        EXPECTED_INTERCEPT_SIMPLE,
        intercept
    );
    assert!(
        (coef - EXPECTED_COEF_SIMPLE).abs() < 0.01,
        "coefficient: expected {}, got {}",
        EXPECTED_COEF_SIMPLE,
        coef
    );
}

/// Test 2: Predicted probabilities vs R's fitted(glm(...)).
#[test]
fn test_logistic_simple_probabilities_vs_r() {
    let x = Mat::from_fn(N_SIMPLE, 1, |i, _| X_SIMPLE[i]);
    let y = Col::from_fn(N_SIMPLE, |i| Y_SIMPLE[i]);

    let model = LogisticRegression::builder().with_intercept(true).build();
    let fitted = model.fit(&x, &y).unwrap();

    let probs = fitted.predict_proba(&x);

    for i in 0..N_SIMPLE {
        assert!(
            (probs[i] - EXPECTED_PROBS_SIMPLE[i]).abs() < 0.01,
            "probability at index {}: expected {}, got {}",
            i,
            EXPECTED_PROBS_SIMPLE[i],
            probs[i]
        );
    }
}

/// Test 3: Predicted class labels vs R (threshold = 0.5).
#[test]
fn test_logistic_simple_classes_vs_r() {
    let x = Mat::from_fn(N_SIMPLE, 1, |i, _| X_SIMPLE[i]);
    let y = Col::from_fn(N_SIMPLE, |i| Y_SIMPLE[i]);

    let model = LogisticRegression::builder().with_intercept(true).build();
    let fitted = model.fit(&x, &y).unwrap();

    let classes = fitted.predict(&x);

    for i in 0..N_SIMPLE {
        assert_eq!(
            classes[i], EXPECTED_CLASSES_SIMPLE[i],
            "class at index {}: expected {}, got {}",
            i, EXPECTED_CLASSES_SIMPLE[i], classes[i]
        );
    }
}

/// Test 4: Classification accuracy vs R.
#[test]
fn test_logistic_simple_accuracy_vs_r() {
    let x = Mat::from_fn(N_SIMPLE, 1, |i, _| X_SIMPLE[i]);
    let y = Col::from_fn(N_SIMPLE, |i| Y_SIMPLE[i]);

    let model = LogisticRegression::builder().with_intercept(true).build();
    let fitted = model.fit(&x, &y).unwrap();

    let accuracy = fitted.score(&x, &y);

    assert!(
        (accuracy - EXPECTED_ACCURACY_SIMPLE).abs() < 1e-6,
        "accuracy: expected {}, got {}",
        EXPECTED_ACCURACY_SIMPLE,
        accuracy
    );
}

/// Test 5: Decision function (log-odds / linear predictor) vs R.
///
/// R code:
/// ```r
/// predict(fit, type = "link")  # returns log-odds
/// ```
#[test]
fn test_logistic_simple_logodds_vs_r() {
    let x = Mat::from_fn(N_SIMPLE, 1, |i, _| X_SIMPLE[i]);
    let y = Col::from_fn(N_SIMPLE, |i| Y_SIMPLE[i]);

    let model = LogisticRegression::builder().with_intercept(true).build();
    let fitted = model.fit(&x, &y).unwrap();

    let logodds = fitted.decision_function(&x);

    for i in 0..N_SIMPLE {
        assert!(
            (logodds[i] - EXPECTED_LOGODDS_SIMPLE[i]).abs() < 0.01,
            "log-odds at index {}: expected {}, got {}",
            i,
            EXPECTED_LOGODDS_SIMPLE[i],
            logodds[i]
        );
    }
}

/// Test 6: Multiple predictors vs R's glm() with two covariates.
///
/// R code:
/// ```r
/// fit <- glm(y ~ x1 + x2, family = binomial(link = "logit"))
/// coef(fit)
/// mean(ifelse(fitted(fit) >= 0.5, 1, 0) == y)
/// ```
#[test]
fn test_logistic_multi_predictor_vs_r() {
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

    let model = LogisticRegression::builder().with_intercept(true).build();
    let fitted = model.fit(&x, &y).unwrap();

    let intercept = fitted.intercept().unwrap();
    let coef1 = fitted.coefficients()[0];
    let coef2 = fitted.coefficients()[1];
    let accuracy = fitted.score(&x, &y);

    assert!(
        (intercept - EXPECTED_INTERCEPT_MULTI).abs() < 0.05,
        "intercept: expected {}, got {}",
        EXPECTED_INTERCEPT_MULTI,
        intercept
    );
    assert!(
        (coef1 - EXPECTED_COEF_MULTI_1).abs() < 0.05,
        "coef1: expected {}, got {}",
        EXPECTED_COEF_MULTI_1,
        coef1
    );
    assert!(
        (coef2 - EXPECTED_COEF_MULTI_2).abs() < 0.05,
        "coef2: expected {}, got {}",
        EXPECTED_COEF_MULTI_2,
        coef2
    );
    assert!(
        (accuracy - EXPECTED_ACCURACY_MULTI).abs() < 0.05,
        "accuracy: expected {}, got {}",
        EXPECTED_ACCURACY_MULTI,
        accuracy
    );
}

/// Test 7: Well-separated data (quasi-separation).
///
/// With well-separated classes the coefficients grow large, so we use a
/// wider tolerance (2.0) for coefficients. The key check is that accuracy
/// is perfect (1.0).
///
/// R code:
/// ```r
/// fit <- glm(y ~ x, family = binomial(link = "logit"))
/// coef(fit)
/// mean(ifelse(fitted(fit) >= 0.5, 1, 0) == y)  # 1.0
/// ```
#[test]
fn test_logistic_separated_data_vs_r() {
    let x = Mat::from_fn(N_SEP, 1, |i, _| X_SEP[i]);
    let y = Col::from_fn(N_SEP, |i| Y_SEP[i]);

    let model = LogisticRegression::builder().with_intercept(true).build();
    let fitted = model.fit(&x, &y).unwrap();

    let intercept = fitted.intercept().unwrap();
    let coef = fitted.coefficients()[0];
    let accuracy = fitted.score(&x, &y);

    assert!(
        (intercept - EXPECTED_INTERCEPT_SEP).abs() < 2.0,
        "intercept: expected {}, got {} (tolerance 2.0 for quasi-separation)",
        EXPECTED_INTERCEPT_SEP,
        intercept
    );
    assert!(
        (coef - EXPECTED_COEF_SEP).abs() < 2.0,
        "coefficient: expected {}, got {} (tolerance 2.0 for quasi-separation)",
        EXPECTED_COEF_SEP,
        coef
    );
    assert!(
        (accuracy - EXPECTED_ACCURACY_SEP).abs() < 1e-6,
        "accuracy: expected {}, got {}",
        EXPECTED_ACCURACY_SEP,
        accuracy
    );
}

/// Test 8: L2 regularization shrinks coefficients toward zero.
///
/// Fit the same simple dataset with and without L2 penalty and verify that
/// the regularized coefficients are smaller in absolute value than the
/// unregularized baseline from R.
#[test]
fn test_logistic_l2_shrinkage() {
    let x = Mat::from_fn(N_SIMPLE, 1, |i, _| X_SIMPLE[i]);
    let y = Col::from_fn(N_SIMPLE, |i| Y_SIMPLE[i]);

    // Fit unregularized model and verify it matches R baseline
    let model_unreg = LogisticRegression::builder().with_intercept(true).build();
    let fitted_unreg = model_unreg.fit(&x, &y).unwrap();

    let unreg_intercept = fitted_unreg.intercept().unwrap();
    let unreg_coef = fitted_unreg.coefficients()[0];

    assert!(
        (unreg_intercept - UNREG_INTERCEPT).abs() < 0.01,
        "unregularized intercept: expected {}, got {}",
        UNREG_INTERCEPT,
        unreg_intercept
    );
    assert!(
        (unreg_coef - UNREG_COEF).abs() < 0.01,
        "unregularized coefficient: expected {}, got {}",
        UNREG_COEF,
        unreg_coef
    );

    // Fit with L2 regularization
    let model_l2 = LogisticRegression::builder()
        .with_intercept(true)
        .l2(1.0)
        .build();
    let fitted_l2 = model_l2.fit(&x, &y).unwrap();

    let l2_coef = fitted_l2.coefficients()[0];

    // L2 penalty should shrink the coefficient toward zero
    assert!(
        l2_coef.abs() < unreg_coef.abs(),
        "L2 coefficient ({}) should be smaller in magnitude than unregularized ({})",
        l2_coef.abs(),
        unreg_coef.abs()
    );
}
