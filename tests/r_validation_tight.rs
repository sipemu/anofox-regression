//! #209: full-precision R validation for functionality whose existing tests
//! used loose tolerances (HC standard errors, OLS intervals at other levels,
//! influence diagnostics, ridge, quantile regression, logistic, Huber).
//!
//! Reference values: `tests/r_scripts/generate_tight_validation.R`
//! (sandwich, lmtest, MASS, quantreg; `glm` / `rlm` with tight control).

#![allow(clippy::excessive_precision)]

use anofox_regression::core::IntervalType;
use anofox_regression::diagnostics::{
    compute_leverage, cooks_distance, dffits, externally_studentized_residuals,
    studentized_residuals,
};
use anofox_regression::inference::{compute_hc_inference, compute_hc_standard_errors, HcType};
use anofox_regression::solvers::{
    BinomialRegressor, FittedRegressor, HuberRegressor, OlsRegressor, QuantileRegressor, Regressor,
    RidgeRegressor,
};
use faer::{Col, Mat};

const X1: [f64; N] = [
    9.5990000000000002,
    5.0830000000000002,
    4.5650000000000004,
    4.5140000000000002,
    1.597,
    5.9480000000000004,
    1.1950000000000001,
    2.6859999999999999,
    3.5630000000000002,
    9.9190000000000005,
    8.8580000000000005,
    5.7240000000000002,
    5.9880000000000004,
    3.883,
    8.8019999999999996,
    1.548,
    5.7889999999999997,
    4.5220000000000002,
    7.8200000000000003,
    7.1159999999999997,
    9.9290000000000003,
    2.137,
    3.6920000000000002,
    4.9820000000000002,
    0.68500000000000005,
    5.8920000000000003,
    3.9729999999999999,
    0.24399999999999999,
    0.27100000000000002,
    6.8499999999999996,
];
const X2: [f64; N] = [
    -0.38,
    -0.092999999999999999,
    -1.3660000000000001,
    0.83399999999999996,
    0.51400000000000001,
    -1.71,
    -0.75600000000000001,
    1.9790000000000001,
    -1.034,
    -1.1040000000000001,
    1.8069999999999999,
    -0.55600000000000005,
    0.93100000000000005,
    -1.9199999999999999,
    0.16600000000000001,
    -1.4379999999999999,
    0.122,
    0.76000000000000001,
    -0.28899999999999998,
    0.183,
    2.1200000000000001,
    -2.5249999999999999,
    0.41199999999999998,
    -1.532,
    1.651,
    0.13500000000000001,
    -0.35199999999999998,
    0.161,
    -0.42299999999999999,
    0.59399999999999997,
];
const Y: [f64; N] = [
    4.3480999999999996,
    4.3903999999999996,
    0.52159999999999995,
    5.5515999999999996,
    2.6833,
    0.38169999999999998,
    -0.42099999999999999,
    6.2385999999999999,
    1.0885,
    5.2077,
    8.5360999999999994,
    2.8652000000000002,
    5.7309000000000001,
    -0.76049999999999995,
    6.0095999999999998,
    -0.5988,
    5.4116,
    3.7616000000000001,
    3.3868999999999998,
    5.5758000000000001,
    9.9916,
    -2.4552999999999998,
    3.3315999999999999,
    0.34329999999999999,
    4.5559000000000003,
    4.6707999999999998,
    2.8929,
    1.6092,
    0.21829999999999999,
    5.1524999999999999,
];
const HC0_SE: [f64; 3] = [
    0.16758278175740504,
    0.039153726660598319,
    0.064698561772393443,
];
const HC0_T: [f64; 3] = [6.171833116962639, 12.998413076162434, 28.899486732345295];
const HC0_P: [f64; 3] = [
    1.3469307800485511e-06,
    3.9017059718818882e-13,
    7.6120245406038854e-22,
];
const HC0_CI_LO: [f64; 3] = [0.6904414966365755, 0.42859950140946756, 1.7370047441261252];
const HC0_CI_HI: [f64; 3] = [1.378144427929574, 0.58927312380175423, 2.0025057109600883];
const HC1_SE: [f64; 3] = [
    0.17664776232677173,
    0.041271651710383078,
    0.068198272179287894,
];
const HC1_T: [f64; 3] = [5.8551149964174956, 12.331377386516694, 27.416460385208403];
const HC1_P: [f64; 3] = [
    3.1018508796900762e-06,
    1.3272191824428896e-12,
    3.0169072115273265e-21,
];
const HC1_CI_LO: [f64; 3] = [0.67184169287304807, 0.42425387816070115, 1.7298239315144155];
const HC1_CI_HI: [f64; 3] = [1.3967442316931016, 0.59361874705052065, 2.0096865235717982];
const HC2_SE: [f64; 3] = [
    0.17937275203618724,
    0.042393269436875325,
    0.069604218699304402,
];
const HC2_T: [f64; 3] = [5.7661654322748701, 12.005120609143633, 26.862671005914077];
const HC2_P: [f64; 3] = [
    3.9263432815906723e-06,
    2.4575982462860771e-12,
    5.1365292547549261e-21,
];
const HC2_CI_LO: [f64; 3] = [0.6662504758301746, 0.42195250868165912, 1.7269391675401065];
const HC2_CI_HI: [f64; 3] = [1.4023354487359749, 0.59592011652956267, 2.0125712875461073];
const HC3_SE: [f64; 3] = [
    0.19235595594942664,
    0.045963671108475782,
    0.075023684713908856,
];
const HC3_T: [f64; 3] = [5.3769739396840217, 11.072577545089999, 24.922199365082207];
const HC3_P: [f64; 3] = [
    1.1078809277110882e-05,
    1.5274700310232846e-11,
    3.6051862063933567e-20,
];
const HC3_CI_LO: [f64; 3] = [0.63961114183930379, 0.41462664957577711, 1.7158193417883132];
const HC3_CI_HI: [f64; 3] = [1.4289747827268457, 0.60324597563544469, 2.0236911132979003];
const HC0_NOINT_SE: [f64; 2] = [0.024447681777895587, 0.10141717185680563];
const HC3_NOINT_SE: [f64; 2] = [0.026709661700463239, 0.1133578561836007];
const CI90_LO: [f64; 3] = [0.65813522847956363, 0.44285099901058589, 1.707987476874776];
const CI90_HI: [f64; 3] = [1.4104506960865859, 0.5750216262006359, 2.0315229782114375];
const F_P: f64 = 9.4938139243646089e-20;
const NEW_X1: [f64; 3] = [2.0, 5.0, 9.5];
const NEW_X2: [f64; 3] = [-1.0, 0.0, 1.5];
const CONF90_LO: [f64; 3] = [-0.10508916140087382, 3.393854420578045, 8.2655331787557298];
const CONF90_HI: [f64; 3] = [0.46990988130325279, 3.7640946300442137, 9.0821083679463488];
const PRED90_LO: [f64; 3] = [-0.86748696222056587, 2.5523792469749464, 7.584634300690742];
const PRED90_HI: [f64; 3] = [1.2323076821229448, 4.6055698036473123, 9.7630072460113375];
const HAT: [f64; N] = [
    0.13394311871725534,
    0.033454392110183924,
    0.073200078944111138,
    0.057896733427132238,
    0.097439459143634208,
    0.1100286758054501,
    0.094881255267750361,
    0.18226368295261122,
    0.058913599317957761,
    0.18406856815705508,
    0.16707288734130868,
    0.042702941870078608,
    0.061830805418308779,
    0.11594131331323394,
    0.096292870169102818,
    0.11165668272781104,
    0.037232160227139668,
    0.054320148082002487,
    0.072328084564190764,
    0.054008181814295224,
    0.22862111399374163,
    0.19306321807017485,
    0.04876005498786349,
    0.086067495628033092,
    0.21509825052816378,
    0.038078825931908526,
    0.037872628889151612,
    0.13284515610995259,
    0.12298870502680913,
    0.057128911463587312,
];
const COOKS: [f64; N] = [
    0.12554876080509331,
    0.030207197455391188,
    0.0064231466351369901,
    0.02699838279569108,
    0.001767473458140075,
    0.030668898783374834,
    0.04640074090353178,
    0.0048562247597360823,
    0.0019144122758488547,
    0.37102058386378117,
    0.033855038125359872,
    8.0432141570087648e-05,
    0.00055960314903468085,
    0.0046126114914550728,
    0.0038385154226655803,
    0.0096227203078853869,
    0.055130761052919926,
    0.057047173219890356,
    0.094172274631441663,
    0.01910642665489581,
    0.0013032734116674412,
    0.0058265211327042203,
    0.0063335064341273339,
    0.012810262608445602,
    0.0024502826541485836,
    0.0057986292475740086,
    0.0094985489023254738,
    0.0037547924960783549,
    0.0040298343905905776,
    0.013963197567654765,
];
const RSTANDARD: [f64; N] = [
    -1.560557760001082,
    1.6180817104855785,
    -0.49393723361258735,
    1.1480260079658207,
    -0.22161942142313951,
    -0.86267017217717057,
    -1.1523533100804639,
    0.25566223954658829,
    0.3028902774602299,
    2.2212459267911284,
    -0.71157845710732703,
    -0.073547837528802326,
    -0.15960218570871201,
    -0.3248295315828133,
    0.32874492774710495,
    0.47924503040988331,
    2.0680416976137836,
    -1.7261132550136216,
    -1.9035569745331098,
    1.001992047604654,
    -0.11485610921030252,
    0.27029329706660438,
    -0.60882949186228352,
    -0.63881796711780026,
    0.16377889235993925,
    0.66290495585672327,
    0.85083004518577898,
    0.27116214671334188,
    -0.29361227338597162,
    -0.83147894022410118,
];
const RSTUDENT: [f64; N] = [
    -1.605503096320952,
    1.6709141530318123,
    -0.4869088118985942,
    1.1551108767363527,
    -0.21767471104599492,
    -0.85845755455323947,
    -1.1596896555631797,
    0.25118731340403283,
    0.29773453991239457,
    2.41113127721803,
    -0.70491786645002419,
    -0.072180221160034502,
    -0.15669263664122207,
    -0.3193820891342658,
    0.3232472022947534,
    0.472299490121661,
    2.2121327521211289,
    -1.7958275975097162,
    -2.0075347933028089,
    1.0020689027514984,
    -0.11273662108036822,
    0.26560022351835766,
    -0.60159229193363162,
    -0.6316682048997353,
    0.16079722766614671,
    0.65587239224215299,
    0.84634831882720873,
    0.26645631507483758,
    -0.28858478836490503,
    -0.82658725718321224,
];
const DFFITS: [f64; N] = [
    -0.63139064443114701,
    0.3108633574115679,
    -0.1368391648555633,
    0.28635280989187251,
    -0.071521602667869846,
    -0.30184484500334124,
    -0.37547343395139304,
    0.1185881048826289,
    0.074494163744171893,
    1.1452065991781619,
    -0.31570975966285519,
    -0.015244870652923164,
    -0.040226334074027127,
    -0.1156615790613813,
    0.10551591911609486,
    0.16744393340655497,
    0.43502029236172013,
    -0.43040077608836114,
    -0.5605564556653585,
    0.23943281594029661,
    -0.061374689604448127,
    0.1299147466267877,
    -0.13620382904185926,
    -0.19384363202180541,
    0.084176205993894651,
    0.13049425216491795,
    0.1679173520451443,
    0.1042918934155076,
    -0.10806957816297624,
    -0.20346537858332175,
];
const RIDGE_COEF: [f64; 3] = [1.0115644419621028, 0.51122950801771805, 1.7591414517421242];
const RQ25: [f64; 3] = [0.95125218004412349, 0.4611241312984149, 1.9370283701795883];
const RQ50: [f64; 3] = [1.0988343881270384, 0.49214805718249477, 1.8897299774094205];
const RQ80: [f64; 3] = [1.161057040768503, 0.60172069956867758, 1.8715721027126675];
const YB: [f64; N] = [
    1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0,
    1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0,
];
const LOGIT_COEF: [f64; 3] = [-6.889845310186784, 1.3191393176303734, 4.8822849329041107];
const LOGIT_SE: [f64; 3] = [3.6435750801666762, 0.63246568134337089, 2.3458407559985388];
const HUBER_COEF: [f64; 3] = [1.0653707715699314, 0.49848970634729883, 1.8910392325987604];
const HUBER_SCALE: f64 = 0.49056880317696233;

const N: usize = 30;

fn x() -> Mat<f64> {
    Mat::from_fn(N, 2, |i, j| if j == 0 { X1[i] } else { X2[i] })
}

fn y() -> Col<f64> {
    Col::from_fn(N, |i| Y[i])
}

fn close(a: f64, b: f64, tol: f64, what: &str) {
    assert!(
        (a - b).abs() <= tol * (1.0 + b.abs()),
        "{what}: {a} vs R {b}"
    );
}

/// Intercept first, then the coefficients.
fn with_int(int: f64, c: &Col<f64>) -> Vec<f64> {
    std::iter::once(int).chain(c.iter().copied()).collect()
}

#[test]
fn hc0_to_hc3_match_sandwich_and_coeftest() {
    let fit = OlsRegressor::builder().build().fit(&x(), &y()).unwrap();
    let r = fit.result();
    let refs = [
        (HcType::HC0, HC0_SE, HC0_T, HC0_P, HC0_CI_LO, HC0_CI_HI),
        (HcType::HC1, HC1_SE, HC1_T, HC1_P, HC1_CI_LO, HC1_CI_HI),
        (HcType::HC2, HC2_SE, HC2_T, HC2_P, HC2_CI_LO, HC2_CI_HI),
        (HcType::HC3, HC3_SE, HC3_T, HC3_P, HC3_CI_LO, HC3_CI_HI),
    ];
    for (t, se, tv, p, lo, hi) in refs {
        let hc = compute_hc_inference(
            &x(),
            &r.coefficients,
            r.intercept,
            &r.residuals,
            &r.aliased,
            true,
            t,
            0.95,
        )
        .unwrap();
        let i = hc.intercept.as_ref().unwrap();
        let got = [
            (with_int(i.std_error, &hc.std_errors), se, "se"),
            (with_int(i.t_statistic, &hc.t_statistics), tv, "t"),
            (with_int(i.p_value, &hc.p_values), p, "p"),
            (
                with_int(i.conf_interval.0, &hc.conf_interval_lower),
                lo,
                "ci lower",
            ),
            (
                with_int(i.conf_interval.1, &hc.conf_interval_upper),
                hi,
                "ci upper",
            ),
        ];
        for (g, w, what) in got {
            for k in 0..3 {
                close(g[k], w[k], 1e-9, &format!("{t:?} {what}[{k}]"));
            }
        }
    }
    let fit0 = OlsRegressor::builder()
        .with_intercept(false)
        .build()
        .fit(&x(), &y())
        .unwrap();
    let r0 = fit0.result();
    for (t, se) in [(HcType::HC0, HC0_NOINT_SE), (HcType::HC3, HC3_NOINT_SE)] {
        let hc = compute_hc_standard_errors(&x(), &r0.residuals, &r0.aliased, false, t).unwrap();
        for (k, &want) in se.iter().enumerate() {
            close(
                hc.std_errors[k],
                want,
                1e-9,
                &format!("{t:?} no intercept se[{k}]"),
            );
        }
    }
}

#[test]
fn ols_intervals_at_90_percent_and_f_pvalue_match_r() {
    let fit = OlsRegressor::builder()
        .confidence_level(0.9)
        .compute_inference(true)
        .build()
        .fit(&x(), &y())
        .unwrap();
    let r = fit.result();
    let (ilo, ihi) = r.intercept_conf_interval.unwrap();
    let lo = with_int(ilo, r.conf_interval_lower.as_ref().unwrap());
    let hi = with_int(ihi, r.conf_interval_upper.as_ref().unwrap());
    for k in 0..3 {
        close(lo[k], CI90_LO[k], 1e-10, "confint lower");
        close(hi[k], CI90_HI[k], 1e-10, "confint upper");
    }
    close(r.f_pvalue / F_P, 1.0, 1e-8, "F p-value (relative)");
    let nd = Mat::from_fn(3, 2, |i, j| if j == 0 { NEW_X1[i] } else { NEW_X2[i] });
    let c = fit.predict_with_interval(&nd, Some(IntervalType::Confidence), 0.9);
    let p = fit.predict_with_interval(&nd, Some(IntervalType::Prediction), 0.9);
    for i in 0..3 {
        close(c.lower[i], CONF90_LO[i], 1e-10, "confidence lower");
        close(c.upper[i], CONF90_HI[i], 1e-10, "confidence upper");
        close(p.lower[i], PRED90_LO[i], 1e-10, "prediction lower");
        close(p.upper[i], PRED90_HI[i], 1e-10, "prediction upper");
    }
}

#[test]
fn influence_diagnostics_match_r() {
    let fit = OlsRegressor::builder().build().fit(&x(), &y()).unwrap();
    let r = fit.result();
    let lev = compute_leverage(&x(), true);
    let cooks = cooks_distance(&r.residuals, &lev, r.mse, 3);
    let rstd = studentized_residuals(&r.residuals, &lev, r.mse);
    let rstu = externally_studentized_residuals(&r.residuals, &lev, r.mse, 3);
    let dff = dffits(&r.residuals, &lev, r.mse, 3);
    for i in 0..N {
        close(lev[i], HAT[i], 1e-10, "hatvalues");
        close(cooks[i], COOKS[i], 1e-9, "cooks.distance");
        close(rstd[i], RSTANDARD[i], 1e-9, "rstandard");
        close(rstu[i], RSTUDENT[i], 1e-9, "rstudent");
        close(dff[i], DFFITS[i], 1e-9, "dffits");
    }
}

#[test]
fn ridge_quantile_logistic_and_huber_match_r() {
    let fit = RidgeRegressor::builder()
        .lambda(2.5)
        .build()
        .fit(&x(), &y())
        .unwrap();
    let got = with_int(fit.result().intercept.unwrap(), &fit.result().coefficients);
    for k in 0..3 {
        close(got[k], RIDGE_COEF[k], 1e-10, "ridge");
    }

    for (tau, want) in [(0.25, RQ25), (0.5, RQ50), (0.8, RQ80)] {
        let fit = QuantileRegressor::builder()
            .tau(tau)
            .build()
            .fit(&x(), &y())
            .unwrap();
        let got = with_int(fit.result().intercept.unwrap(), &fit.result().coefficients);
        for k in 0..3 {
            close(got[k], want[k], 1e-8, &format!("rq tau = {tau}"));
        }
    }

    let yb = Col::from_fn(N, |i| YB[i]);
    let fit = BinomialRegressor::logistic()
        .compute_inference(true)
        .tolerance(1e-15)
        .build()
        .fit(&x(), &yb)
        .unwrap();
    let r = fit.result();
    let coef = with_int(r.intercept.unwrap(), &r.coefficients);
    let se = with_int(
        r.intercept_std_error.unwrap(),
        r.std_errors.as_ref().unwrap(),
    );
    for k in 0..3 {
        close(coef[k], LOGIT_COEF[k], 1e-8, "logistic coef");
        close(se[k], LOGIT_SE[k], 1e-8, "logistic se");
    }

    let fit = HuberRegressor::builder()
        .epsilon(1.345)
        .alpha(0.0)
        .tolerance(1e-14)
        .max_iterations(500)
        .build()
        .fit(&x(), &y())
        .unwrap();
    let got = with_int(fit.result().intercept.unwrap(), &fit.result().coefficients);
    for k in 0..3 {
        close(got[k], HUBER_COEF[k], 1e-8, "huber coef");
    }
    close(fit.scale(), HUBER_SCALE, 1e-8, "huber scale");
}
