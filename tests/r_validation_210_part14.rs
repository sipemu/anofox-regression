//! #210, part 14: `greybox::alm` — confidence intervals of identity- and
//! log-location families and the Normal prediction interval
//! (`predict.alm(interval = , level = 0.9)`), the log-S distribution, the
//! scale of several distributions, fits without intercept, and the MAE /
//! HAM / ROLE losses, against greybox at `xtol_rel = ftol_rel = 1e-14`.
//!
//! Reference values: `tests/r_scripts/generate_validation_210_part14.R`.

#![allow(clippy::excessive_precision)]

use anofox_regression::core::IntervalType;
use anofox_regression::solvers::{
    AlmDistribution, AlmLoss, AlmRegressor, FittedAlm, FittedRegressor, Regressor,
};
use faer::{Col, Mat};

const X: [f64; 60] = [
    0.067149478942155838,
    2.7249907832592726,
    3.6747017158195376,
    3.1121145915240049,
    3.9651849307119846,
    3.1358459647744894,
    1.8307895977050066,
    0.91579532809555531,
    3.4194256626069546,
    0.90440075192600489,
    0.71320820599794388,
    2.1033437112346292,
    1.0894856285303831,
    2.4701539734378457,
    2.2812725463882089,
    1.7542161801829934,
    1.4424112048000097,
    3.2190624428912997,
    1.7893634922802448,
    2.1852579796686769,
    1.0128514776006341,
    3.6088280640542507,
    1.381116695702076,
    0.93835554551333189,
    2.2849490959197283,
    0.66498100571334362,
    0.3988965256139636,
    0.55342443007975817,
    0.81584388017654419,
    3.7344112740829587,
    0.011388438753783703,
    1.9499280089512467,
    1.0496164057403803,
    1.4587852004915476,
    2.9852478029206395,
    1.8969794558361173,
    3.4966152030974627,
    0.60185468010604382,
    3.1583034377545118,
    2.1290525970980525,
    3.4128103721886873,
    2.970354669727385,
    1.3088302910327911,
    0.92539634928107262,
    2.2805768670514226,
    1.5655123302713037,
    0.67141911387443542,
    1.0381442494690418,
    0.14705639611929655,
    0.86140672862529755,
    1.4939105454832315,
    0.24173099640756845,
    3.7483529262244701,
    0.5653711399063468,
    3.6184307243674994,
    0.77473080717027187,
    2.8293709941208363,
    1.8164043491706252,
    1.0950931524857879,
    2.3958171224221587,
];
const YN: [f64; 60] = [
    1.1120489004139251,
    3.2405505230703815,
    4.9241983277637074,
    2.9963096613789011,
    3.8571063052400412,
    3.4934831028749276,
    2.7918060160050415,
    1.9840415895024186,
    4.7902874871092997,
    2.3330611067420897,
    0.98316695663495901,
    3.9321306888887468,
    0.69633030424701903,
    3.2030664371796935,
    2.6176054591227924,
    1.8884339731518172,
    2.1819905565812294,
    2.5458336051766794,
    1.8573009944860306,
    3.1550795434672554,
    1.8726910660456493,
    3.3348625346261294,
    2.4176512822997034,
    2.4367156257721065,
    3.6154249419328592,
    2.5148740486214365,
    1.0237731331117375,
    1.3852904663895196,
    3.2415636412860662,
    3.5830434106443905,
    1.9462103398584736,
    3.4538915183921342,
    1.8887478928525352,
    2.2680068721560631,
    4.2691671259815056,
    2.2892486821907609,
    3.5946843357707552,
    2.0604809815268115,
    4.2590287760285195,
    2.0169098423287353,
    2.9317753427731321,
    3.5090843980120163,
    1.1839074510433973,
    2.3995657789929288,
    2.7907968637085556,
    2.2265107520173104,
    1.0816234101352682,
    2.3634551832029618,
    1.819853884491287,
    2.3292098445450353,
    2.3402151955077772,
    2.0694683145029389,
    4.6464080578342584,
    1.9129045105451874,
    4.224534667642323,
    1.5261775462429488,
    2.9317551867248173,
    3.3745939347861542,
    1.4953259510906596,
    4.047384217138795,
];
const YPOS: [f64; 60] = [
    1.337303964953158,
    2.2377540823041171,
    3.1393649455091741,
    2.7549264247723015,
    2.7479205203815114,
    6.8873277989173642,
    1.9961995774605685,
    2.1409877020023282,
    2.9235955854260265,
    1.2215300310626669,
    2.1251737912477329,
    1.0193373888284796,
    1.7469319718020182,
    3.631538407554125,
    1.8247706984410526,
    3.0227521538157096,
    1.3658457437911373,
    10.001234222501886,
    1.8901236602134401,
    1.9614310165612232,
    1.0363596698484259,
    2.8882384884032244,
    1.0125613993517886,
    1.3215515531317654,
    2.446886615065945,
    1.3982856580560306,
    1.3051068128093712,
    1.5987947113933783,
    1.7110661973040104,
    5.6315367958931599,
    0.83948954252071395,
    1.2463431789648893,
    0.61675777201710325,
    1.8493744756249328,
    2.5121261671150634,
    1.6687646290882758,
    2.5743472858671494,
    1.9089236360888795,
    1.1444728957740908,
    1.4584725601152055,
    3.4130039921209185,
    5.0100750866245676,
    1.3970281519508478,
    1.3404886932273368,
    1.7781257069781773,
    0.88818061499939116,
    0.82860734016852744,
    1.1422397917354026,
    2.0962802001847085,
    1.2011229023175303,
    1.4613018011989465,
    2.2925946084860307,
    2.141020816959621,
    1.075792467405547,
    3.9299892221585759,
    1.07039038765816,
    3.2886603381499979,
    1.8674881182619429,
    1.9431050441653632,
    2.7171129113852537,
];
const YP: [f64; 60] = [
    0.0, 8.0, 12.0, 4.0, 3.0, 6.0, 3.0, 1.0, 8.0, 1.0, 3.0, 2.0, 3.0, 4.0, 5.0, 3.0, 2.0, 4.0, 1.0,
    4.0, 3.0, 3.0, 1.0, 2.0, 4.0, 2.0, 2.0, 2.0, 1.0, 6.0, 0.0, 5.0, 3.0, 3.0, 7.0, 4.0, 6.0, 2.0,
    4.0, 2.0, 2.0, 8.0, 1.0, 0.0, 2.0, 4.0, 0.0, 2.0, 2.0, 0.0, 2.0, 2.0, 7.0, 2.0, 5.0, 1.0, 0.0,
    2.0, 1.0, 3.0,
];
const YO: [f64; 60] = [
    1.1120489004139251,
    3.2405505230703815,
    4.9241983277637074,
    2.9963096613789011,
    3.8571063052400412,
    3.4934831028749276,
    10.791806016005042,
    1.9840415895024186,
    4.7902874871092997,
    2.3330611067420897,
    0.98316695663495901,
    3.9321306888887468,
    0.69633030424701903,
    3.2030664371796935,
    2.6176054591227924,
    1.8884339731518172,
    2.1819905565812294,
    2.5458336051766794,
    1.8573009944860306,
    3.1550795434672554,
    1.8726910660456493,
    3.3348625346261294,
    2.4176512822997034,
    2.4367156257721065,
    3.6154249419328592,
    2.5148740486214365,
    1.0237731331117375,
    1.3852904663895196,
    3.2415636412860662,
    -3.4169565893556095,
    1.9462103398584736,
    3.4538915183921342,
    1.8887478928525352,
    2.2680068721560631,
    4.2691671259815056,
    2.2892486821907609,
    3.5946843357707552,
    2.0604809815268115,
    4.2590287760285195,
    2.0169098423287353,
    2.9317753427731321,
    3.5090843980120163,
    1.1839074510433973,
    2.3995657789929288,
    2.7907968637085556,
    2.2265107520173104,
    1.0816234101352682,
    2.3634551832029618,
    1.819853884491287,
    2.3292098445450353,
    11.340215195507778,
    2.0694683145029389,
    4.6464080578342584,
    1.9129045105451874,
    4.224534667642323,
    1.5261775462429488,
    2.9317551867248173,
    3.3745939347861542,
    1.4953259510906596,
    4.047384217138795,
];
const X_NEW: [f64; 3] = [0.5, 2.0, 3.5];
const CI_NORMAL: [f64; 9] = [
    1.6861633170134669,
    2.7660483699886549,
    3.8459334229638431,
    1.4763455807636683,
    2.630428615237093,
    3.6063612106723881,
    1.8959810532632655,
    2.9016681247402167,
    4.0855056352552976,
];
const CI_LAPLACE: [f64; 9] = [
    1.7013369137591732,
    2.7212065565212633,
    3.7410761992833539,
    1.6948062107276864,
    2.7113297008591308,
    3.7249745276584463,
    1.7078676167906599,
    2.7310834121833958,
    3.7571778709082615,
];
const CI_LOGISTIC: [f64; 9] = [
    1.7102090723055972,
    2.7567302714651776,
    3.8032514706247578,
    1.5013980020073399,
    2.6181200686263129,
    3.55272314968086,
    1.9190201426038545,
    2.8953404743040423,
    4.0537797915686555,
];
const CI_LOGNORMAL: [f64; 9] = [
    1.243222964168174,
    1.9858630409320108,
    3.1721196687984157,
    1.0858665160689038,
    1.8195353485885528,
    2.7179557273867276,
    1.4233824468872602,
    2.1673951101850788,
    3.7021733252633062,
];
const CI_LOGLAPLACE: [f64; 9] = [
    1.1751105973312164,
    1.8694679765226667,
    2.9741119884213583,
    1.1692141239369513,
    1.855842259265204,
    2.9395331978028931,
    1.1810368072791009,
    1.8831937347021708,
    3.0090975418420021,
];
const PI_NORMAL: [f64; 9] = [
    1.6861633170134669,
    2.7660483699886549,
    3.8459334229638431,
    0.62463640790962405,
    1.7166634957954723,
    2.7781269137081681,
    2.7476902261173097,
    3.8154332441818375,
    4.9137399322195181,
];
const LOGS: [f64; 4] = [
    -0.11092069850232597,
    0.34150832370657225,
    0.23895588163024284,
    -69.656947798081802,
];
const SCALES: [f64; 7] = [
    0.36795158219267704,
    0.49974617771764895,
    0.3347286738334021,
    0.32257068171758596,
    0.15816858969366043,
    0.19438996783954543,
    0.17328372306925374,
];
const NOINT_NORMAL: [f64; 3] = [1.2449588226562367, 0.84219584192182928, -79.984030979413532];
const NOINT_POISSON: [f64; 2] = [0.50868340293052738, -109.19184387364774];
const NOINT_GAMMA: [f64; 3] = [0.38974164578642145, 0.2031047997003099, -71.910834112759062];
const LOSS_MAE: [f64; 2] = [1.565337406150511, 0.61475918664476226];
const LOSS_HAM: [f64; 2] = [1.7389819622814862, 0.53418047098481469];
const LOSS_ROLE05: [f64; 2] = [1.3118216486301217, 0.73118537807303907];
const LOSS_ROLE20: [f64; 2] = [1.3742160583370482, 0.66584464484213712];

fn xm(v: &[f64]) -> Mat<f64> {
    Mat::from_fn(v.len(), 1, |i, _| v[i])
}

fn yc(v: &[f64]) -> Col<f64> {
    Col::from_fn(v.len(), |i| v[i])
}

fn fit(dist: AlmDistribution, y: &[f64], intercept: bool, loss: AlmLoss) -> FittedAlm {
    AlmRegressor::builder()
        .distribution(dist)
        .with_intercept(intercept)
        .loss(loss)
        .compute_inference(true)
        .tolerance(1e-12)
        .max_iterations(10_000)
        .build()
        .fit(&xm(&X), &yc(y))
        .unwrap()
}

fn close(got: &[f64], want: &[f64], tol: f64, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (k, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= tol * (1.0 + w.abs()),
            "{what}[{k}]: {g} vs greybox {w}"
        );
    }
}

fn interval(f: &FittedAlm, kind: IntervalType) -> Vec<f64> {
    let p = f.predict_with_interval(&xm(&X_NEW), Some(kind), 0.9);
    p.fit
        .iter()
        .chain(p.lower.iter())
        .chain(p.upper.iter())
        .copied()
        .collect()
}

#[test]
fn intervals_match_predict_alm() {
    use AlmDistribution as D;
    let lik = AlmLoss::Likelihood;
    // (the S distribution is left out: greybox stops at a local optimum)
    for (dist, y, want, tol) in [
        (D::Normal, &YN[..], &CI_NORMAL, 1e-7),
        (D::Laplace, &YN, &CI_LAPLACE, 1e-5),
        (D::Logistic, &YN, &CI_LOGISTIC, 1e-5),
        (D::LogNormal, &YPOS, &CI_LOGNORMAL, 1e-7),
        (D::LogLaplace, &YPOS, &CI_LOGLAPLACE, 1e-5),
    ] {
        let f = fit(dist, y, true, lik);
        close(
            &interval(&f, IntervalType::Confidence),
            want,
            tol,
            &format!("{dist:?} confidence"),
        );
    }
    let f = fit(D::Normal, &YN, true, lik);
    close(
        &interval(&f, IntervalType::Prediction),
        &PI_NORMAL,
        1e-7,
        "Normal prediction",
    );
    // unsupported combinations, and fits without inference, stay NaN
    let g = fit(D::Gamma, &YPOS, true, lik);
    let p = g.predict_with_interval(&xm(&X_NEW), Some(IntervalType::Confidence), 0.9);
    assert!(p.lower.iter().all(|v| v.is_nan()));
    let f = AlmRegressor::builder()
        .compute_inference(false)
        .build()
        .fit(&xm(&X), &yc(&YN))
        .unwrap();
    assert!(f.vcov().is_none());
    let p = f.predict_with_interval(&xm(&X_NEW), Some(IntervalType::Confidence), 0.9);
    assert!(p.upper.iter().all(|v| v.is_nan()));
    let p = f.predict_with_interval(&xm(&X_NEW), None, 0.9);
    assert_eq!(p.fit.nrows(), 3);
}

/// greybox's `$scale` is the variance for the normal and log-normal and the
/// scale parameter otherwise; the crate's `scale()` is always the latter (σ).
#[test]
fn log_s_and_scales_match_greybox() {
    use AlmDistribution as D;
    let lik = AlmLoss::Likelihood;
    // log-S: the likelihood has kinks and greybox's optimiser stops short; the
    // crate's optimum is at least as good under greybox's own density
    let f = AlmRegressor::builder()
        .distribution(D::LogS)
        .build()
        .fit(&xm(&X), &yc(&YPOS))
        .unwrap();
    let ll = f.result().log_likelihood;
    assert!(
        ll >= LOGS[3] - 1e-9,
        "log-S logLik {ll} vs greybox {}",
        LOGS[3]
    );
    close(&[f.scale()], &LOGS[2..3], 1e-2, "log-S scale");
    let cases = [
        (D::Normal, &YN[..], SCALES[0]),
        (D::Laplace, &YN, SCALES[1]),
        (D::Logistic, &YN, SCALES[2]),
        (D::LogNormal, &YPOS, SCALES[4]),
        (D::Gamma, &YPOS, SCALES[5]),
        (D::InverseGaussian, &YPOS, SCALES[6]),
    ];
    for (d, y, want) in cases {
        let sc = fit(d, y, true, lik).scale();
        let got = if matches!(d, D::Normal | D::LogNormal) {
            sc * sc
        } else {
            sc
        };
        close(&[got], &[want], 1e-5, &format!("{d:?} scale"));
    }
}

#[test]
fn fits_without_intercept_match_greybox() {
    use AlmDistribution as D;
    let lik = AlmLoss::Likelihood;
    for (dist, y, want) in [
        (D::Normal, &YN[..], &NOINT_NORMAL[..]),
        (D::Gamma, &YPOS, &NOINT_GAMMA),
    ] {
        let f = fit(dist, y, false, lik);
        assert!(f.result().intercept.is_none());
        let sc = if dist == D::Normal {
            f.scale() * f.scale()
        } else {
            f.scale()
        };
        let got = [f.result().coefficients[0], sc, f.result().log_likelihood];
        close(&got, want, 1e-6, &format!("{dist:?} without intercept"));
    }
    let f = fit(D::Poisson, &YP, false, lik);
    let got = [f.result().coefficients[0], f.result().log_likelihood];
    close(&got, &NOINT_POISSON, 1e-6, "Poisson without intercept");
}

/// The losses used to be ignored: every loss gave the likelihood fit.
#[test]
fn robust_losses_match_greybox() {
    for (loss, want, tol) in [
        (AlmLoss::MAE, &LOSS_MAE, 1e-7),
        (AlmLoss::role_with_trim(0.05), &LOSS_ROLE05, 1e-4),
        // greybox stops short here: its trimmed objective (σ profiled) is
        // 40.12711 at its estimate and 40.12657 at the crate's
        (AlmLoss::role_with_trim(0.2), &LOSS_ROLE20, 2e-3),
    ] {
        let f = fit(AlmDistribution::Normal, &YO, true, loss);
        let got = [f.result().intercept.unwrap(), f.result().coefficients[0]];
        close(&got, want, tol, &format!("{loss:?}"));
    }
    // HAM: greybox stops short; the crate's mean √|e| is at least as small
    let ham = |b0: f64, b1: f64| {
        (0..X.len())
            .map(|i| (YO[i] - b0 - b1 * X[i]).abs().sqrt())
            .sum::<f64>()
            / X.len() as f64
    };
    let f = fit(AlmDistribution::Normal, &YO, true, AlmLoss::HAM);
    let (b0, b1) = (f.result().intercept.unwrap(), f.result().coefficients[0]);
    assert!(ham(b0, b1) <= ham(LOSS_HAM[0], LOSS_HAM[1]) + 1e-12);
    close(&[b0, b1], &LOSS_HAM, 5e-3, "HAM");
    // MSE with a Normal is least squares
    let f = fit(AlmDistribution::Normal, &YO, true, AlmLoss::MSE);
    let g = fit(AlmDistribution::Normal, &YO, true, AlmLoss::Likelihood);
    let (a, b) = (f.result().coefficients[0], g.result().coefficients[0]);
    close(&[a], &[b], 1e-6, "MSE = OLS");
}
