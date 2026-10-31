//! #210, part 5: GLM options without an R comparison so far — Gamma(log)
//! without intercept and with an offset, Tweedie's default link `1 − p` and
//! `p = 0`, the inverse-Gaussian log-likelihood, the ridge-penalised IRLS
//! (`lambda`) of Poisson / binomial / negative binomial / Gamma vs
//! `glmnet(alpha = 0)`, and binomial `predict_with_se` for probit and cloglog.
//!
//! Reference values: `tests/r_scripts/generate_validation_210_part5.R`.

#![allow(clippy::excessive_precision)]

use anofox_regression::core::{IntervalType, PredictionType};
use anofox_regression::solvers::{
    BinomialRegressor, FittedRegressor, GammaRegressor, NegativeBinomialRegressor,
    PoissonRegressor, Regressor, TweedieRegressor,
};
use faer::{Col, Mat};

const X1: [f64; 40] = [
    0.64193447637371714,
    1.02349066385068,
    1.7463715191930533,
    1.7126430497039109,
    0.54401508821174505,
    1.0566771749872714,
    1.8093783285003155,
    0.4848712885286659,
    1.7878167354501784,
    0.81219813283532871,
    0.82893138336949046,
    1.4595572489313782,
    0.86299284198321402,
    1.7611378698144109,
    1.3756726313382388,
    0.9739629493560642,
    1.5311241078656166,
    1.5723142829257994,
    0.29796316642314197,
    0.94087387365289032,
    1.7923739904537797,
    0.55718171447515497,
    0.80214449050836278,
    0.89260011953301732,
    0.56886592321097851,
    1.486144236754626,
    1.7658656613435597,
    1.3193456129636616,
    1.6345004393719136,
    1.1516654551960528,
    0.83622398613952109,
    0.46661321390420202,
    1.2323543142061681,
    0.69527378464117651,
    1.1016896709799766,
    1.5282414833549409,
    0.21146288332529367,
    0.73748741983436061,
    1.2312981981784106,
    0.79967465880326927,
];
const X2: [f64; 40] = [
    -1.1606199917234024,
    -0.91352650746984099,
    -1.5196205891915704,
    -1.6852660439369369,
    -1.1569667759614601,
    0.28421174306237151,
    -1.0202135000211172,
    -0.23676758388510569,
    0.56157941331112671,
    -1.5895049827706567,
    -0.77914953385606545,
    -1.1106369536624203,
    -0.26983963657746263,
    -1.2299483984002535,
    -0.45107908919568473,
    0.43142686456920926,
    -0.42405213907453854,
    1.6505372020217215,
    -0.64706282760857647,
    -0.85041100334621667,
    2.2458637312006098,
    1.0264009490190751,
    -1.1296105116839001,
    0.013259786046862057,
    -0.037990882722014646,
    -0.59019879184859814,
    0.37435140384728643,
    0.21658634449394731,
    -0.8236631950920078,
    1.1386460526560771,
    -0.092841834754162259,
    2.8263416971609168,
    -0.84488191283175651,
    -1.1688886754154477,
    -0.25480075900777821,
    1.9890782045030917,
    0.65405213853627753,
    -0.27735742636906774,
    -1.3294754887098295,
    -0.43758225141480678,
];
const EXPO: [f64; 40] = [
    0.82129749306477606,
    1.2663586526177824,
    1.1290193588938564,
    2.1084593679988757,
    2.0044745500199497,
    2.8815435415599495,
    2.2057288717478514,
    0.5858311407500878,
    0.96296778426039964,
    0.60828530241269618,
    2.1033602977404371,
    1.1319748610258102,
    2.5687900460325181,
    1.558828323148191,
    2.0043587499530986,
    0.52499538997653872,
    2.6586968272458762,
    2.5439750829245895,
    1.2428337068995461,
    1.9791362022515386,
    0.50720774871297181,
    1.9542270125821233,
    1.7952209217473865,
    1.8869217305909842,
    2.9454430589685217,
    1.1595211924286559,
    1.1997494021197781,
    2.1415957943536341,
    1.1736853806069121,
    2.9451032187789679,
    0.96423875377513468,
    1.1740227923728526,
    2.7633622569264844,
    2.8514836485264823,
    1.9530520990956575,
    2.3539153196616098,
    2.0109494980424643,
    2.6879663848085329,
    1.0711040197638795,
    0.85069609235506505,
];
const Y_GAMMA: [f64; 40] = [
    5.2354105210562718,
    1.0570280730032739,
    2.3055342502693716,
    5.9615635098807056,
    4.1738693908801832,
    3.3427709097094795,
    6.1235123485224445,
    1.3438382010081202,
    1.4700653933584567,
    5.7901630908925013,
    2.9433626235052648,
    9.0229682673163207,
    3.2709364578562523,
    2.6735618275878945,
    5.6463966058961761,
    4.5318719092347859,
    3.8327863960625779,
    1.6901885158793488,
    3.4891361292811185,
    3.1711848342779847,
    1.6660602083313776,
    1.1985851227852873,
    4.2474894220086332,
    1.5561930427701283,
    2.9723645943199744,
    0.56642754126362493,
    2.8256356210561733,
    2.5241167623504257,
    4.9007699435471155,
    1.7570085211827069,
    1.0281056134992521,
    0.37605001136714195,
    1.035474281477514,
    3.7590819913002145,
    3.5721782363447021,
    1.1723103020565646,
    1.1945179186510333,
    0.89194898744460749,
    3.7445865598699735,
    1.697093636071503,
];
const Y_GAMMA_OFFSET: [f64; 40] = [
    2.0932420025152352,
    2.0825164502365503,
    1.7098408632783004,
    5.707917859028103,
    5.3048482636042342,
    4.4513000970451548,
    2.3495679112942183,
    1.3259553454089537,
    0.6194258908521737,
    1.7182643059273024,
    4.3594776212686339,
    4.0895972311047579,
    2.9751401491345701,
    5.2807450974621171,
    10.75912702048527,
    1.242041431062715,
    2.6298308055750135,
    9.3997194284821592,
    0.90115124859994,
    4.282176528892621,
    0.50485814416219588,
    5.9868527349605767,
    2.033793568275486,
    1.580975704919398,
    6.1823348074163631,
    1.5005643976370469,
    4.5950677725708733,
    6.9717774306907954,
    1.9479877246464907,
    7.4773670799366627,
    1.3865587131969328,
    1.7250851979163511,
    7.2456743631808349,
    2.958732382548487,
    5.2293725378119627,
    5.6928079167022423,
    0.87619592512547517,
    2.9612913065112441,
    2.0731089966807175,
    1.9026579572757485,
];
const GAMMA_NOINT: [f64; 10] = [
    0.77107809256260551,
    -0.44186134544410788,
    0.075759625240605075,
    0.08349110762390087,
    0.61771074936536663,
    -0.61088025645647892,
    0.92444543575984439,
    -0.27284243443173684,
    12.719211038329075,
    -70.049418655866063,
];
const GAMMA_OFFSET: [f64; 10] = [
    0.40525720282935257,
    0.28808381149344303,
    0.19783514070679414,
    0.16561146835107066,
    0.0047608985658152414,
    -0.047179078513356232,
    0.80575350709288984,
    0.6233467015002423,
    10.253066732176626,
    -71.584425084293557,
];
const Y_CPG: [f64; 40] = [
    1.1849476498167038,
    0.0,
    7.5099430923025601,
    10.894321351940507,
    2.0673947130074817,
    1.9318797966539329,
    1.7948536203153149,
    3.878957756012881,
    1.8138002815352454,
    2.2547007372912824,
    6.3245366398315754,
    4.4902086654867484,
    0.29473621041904363,
    0.044861943606442967,
    4.2432850777932938,
    1.9325235143997879,
    6.5262071152056196,
    10.86374800186341,
    0.95011726793877449,
    0.0,
    0.78552980415263696,
    1.1343112895662726,
    0.0,
    0.88721235775563179,
    2.1999150748752214,
    9.9585652335686632,
    0.91768456308597612,
    1.3757772682325866,
    2.8578454998336551,
    1.1701010151757512,
    2.1290399304959906,
    3.0194834144598421,
    2.2176878536407219,
    0.5443831829571717,
    2.0723521908327434,
    4.5686220202597312,
    0.74623405834896384,
    0.82254613815413002,
    1.7917434382036985,
    1.9264897441029118,
];
const TWEEDIE_DEFAULT_LINK: [f64; 7] = [
    0.93587078939869173,
    -0.26727260992560586,
    0.0075941314620542218,
    0.13770989829428981,
    0.095677258022412143,
    0.036300480490983077,
    58.580209397771306,
];
const Y_NORMAL: [f64; 40] = [
    3.4118535870403304,
    4.3154422901974359,
    7.0462809283131858,
    6.0025172759893923,
    3.0169018374369738,
    2.9199900266116972,
    6.0464938704353237,
    2.9143072700188464,
    2.5145807732531571,
    5.179763192222369,
    1.5696321728074298,
    3.4761475498625716,
    3.4558954461305147,
    6.682631594565847,
    4.0840629251734955,
    2.8037018855406792,
    4.7888579625920631,
    3.4364187728615496,
    1.8122150595925572,
    3.5663296450593434,
    2.5667685358694992,
    1.4700119825334175,
    3.7676124163617377,
    4.2126390108587719,
    0.40532080858296493,
    4.7288706999186374,
    2.9726932627122902,
    3.2479314951409517,
    6.0507496416936792,
    1.09620816720773,
    2.866207528879765,
    0.050154035520372564,
    3.2792920747569987,
    1.8672239750636992,
    4.7125081789667167,
    1.174490308441182,
    1.2717616692486184,
    4.7608065517217559,
    4.7449194920488003,
    2.2503632009966701,
];
const TWEEDIE_NORMAL: [f64; 8] = [
    1.0701299311623849,
    1.9393656046973946,
    -0.96704049785466073,
    0.39301431809632076,
    0.32849018289858467,
    0.14340779337798043,
    34.334162714435436,
    -53.702769365925903,
];
const Y_INVGAUSS: [f64; 40] = [
    0.49280595011288847,
    1.4023079147480615,
    1.0324594917241823,
    1.3706548900167184,
    0.42998792243845702,
    1.7285728422235265,
    1.2410169068529984,
    1.0316047688381906,
    0.53094186512202823,
    0.84833032414576959,
    2.725648008353569,
    3.3715448765416132,
    0.87647370382908851,
    1.512711565510261,
    1.119568157710112,
    0.85449197663508591,
    0.52529476772528549,
    1.7652600656283375,
    1.8934195548286097,
    0.98857693915840172,
    1.2107429451202623,
    2.7277343241945178,
    1.9122940309470775,
    1.7712534976686112,
    0.75772624190401572,
    1.1998251683089705,
    0.99581504623163952,
    0.94290794845582115,
    1.7305751543833838,
    1.8384387470074237,
    0.76675042282623429,
    0.81414332684232826,
    0.63554667650625085,
    3.1324045441692716,
    1.3657792085179215,
    1.3537558933572269,
    0.36305616300134003,
    2.2002623322649124,
    0.98923768747191931,
    1.6057679992942493,
];
const INVGAUSS_LOGLIK: [f64; 7] = [
    0.33404277038761204,
    -0.038001448026606488,
    -0.037103204239098846,
    9.4248822007357393,
    -37.77943256727648,
    83.55886513455296,
    90.31438295100871,
];
const LAMBDA: f64 = 3.0;
const Y_POISSON: [f64; 40] = [
    1.0, 4.0, 10.0, 7.0, 5.0, 7.0, 8.0, 3.0, 1.0, 2.0, 5.0, 9.0, 3.0, 2.0, 7.0, 1.0, 5.0, 1.0, 3.0,
    1.0, 5.0, 3.0, 6.0, 3.0, 2.0, 9.0, 2.0, 3.0, 10.0, 1.0, 1.0, 3.0, 6.0, 3.0, 6.0, 1.0, 0.0, 3.0,
    4.0, 4.0,
];
const Y_BINOMIAL: [f64; 40] = [
    1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0,
    0.0, 1.0,
];
const Y_NEGBIN: [f64; 40] = [
    3.0, 2.0, 6.0, 4.0, 5.0, 1.0, 1.0, 2.0, 6.0, 4.0, 0.0, 1.0, 2.0, 6.0, 0.0, 5.0, 6.0, 4.0, 0.0,
    3.0, 4.0, 0.0, 9.0, 0.0, 1.0, 8.0, 2.0, 5.0, 1.0, 2.0, 7.0, 1.0, 0.0, 2.0, 6.0, 5.0, 4.0, 1.0,
    9.0, 2.0,
];
const RIDGE_POISSON: [f64; 3] = [
    0.6609397420392753,
    0.53665047553133594,
    -0.28229760887014071,
];
const RIDGE_BINOMIAL: [f64; 3] = [
    0.34659263608871804,
    -0.0050411913113705591,
    0.52597478493066829,
];
const RIDGE_NEGBIN: [f64; 3] = [
    0.71554025563163703,
    0.37950485565403563,
    -0.095242674624216442,
];
const RIDGE_GAMMA: [f64; 3] = [
    0.68864440025869611,
    0.22062063974399834,
    -0.37586051827629574,
];
const PREDICT_PROBIT: [f64; 18] = [
    0.48622946345114981,
    0.16963113729810653,
    0.73542631559349003,
    0.41710020023799438,
    0.20941096571280787,
    0.49359464851636842,
    0.68659775813214641,
    0.56734988330245317,
    0.76896005000162138,
    0.14784706612439458,
    0.082349531690664349,
    0.15025745128316534,
    0.42080312676906168,
    0.43061083129972644,
    0.46952472546693169,
    0.87946133113255809,
    0.69640250276038529,
    0.93910661954577823,
];
const PREDICT_CLOGLOG: [f64; 18] = [
    0.12894144341395591,
    -0.21082984704300389,
    0.43879605786439263,
    0.43859146063317139,
    0.23005688780802394,
    0.47469309461065268,
    0.67942002663447199,
    0.55510272406367733,
    0.78793002196844042,
    0.15995400461359696,
    0.082895842121752397,
    0.15612009686195077,
    0.4247529872415825,
    0.42578355417676572,
    0.50852535945241428,
    0.90371952574337133,
    0.69347009274530103,
    0.96615019401214275,
];
const NEW_X1: [f64; 3] = [0.40000000000000002, 1.1000000000000001, 1.8];
const NEW_X2: [f64; 3] = [0.5, -0.20000000000000001, 1.0];

const N: usize = 40;

fn design(cols: &[&[f64]]) -> Mat<f64> {
    Mat::from_fn(cols[0].len(), cols.len(), |i, j| cols[j][i])
}

fn col(v: &[f64]) -> Col<f64> {
    Col::from_fn(v.len(), |i| v[i])
}

fn assert_close(got: &[f64], want: &[f64], tol: f64, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (k, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= tol * (1.0 + w.abs()),
            "{what}[{k}]: {g} vs R {w}"
        );
    }
}

/// Estimates, SEs, lower and upper limits (intercept first), deviance,
/// log-likelihood.
fn table(fit: &impl FittedRegressor, deviance: f64) -> Vec<f64> {
    let r = fit.result();
    let mut est = vec![];
    let mut se = vec![];
    let mut lo = vec![];
    let mut hi = vec![];
    if let Some(b0) = r.intercept {
        est.push(b0);
        se.push(r.intercept_std_error.unwrap());
        let (l, h) = r.intercept_conf_interval.unwrap();
        lo.push(l);
        hi.push(h);
    }
    for j in 0..r.coefficients.nrows() {
        est.push(r.coefficients[j]);
        se.push(r.std_errors.as_ref().unwrap()[j]);
        lo.push(r.conf_interval_lower.as_ref().unwrap()[j]);
        hi.push(r.conf_interval_upper.as_ref().unwrap()[j]);
    }
    [est, se, lo, hi, vec![deviance, r.log_likelihood]].concat()
}

#[test]
fn gamma_without_intercept_matches_glm() {
    let f = GammaRegressor::builder()
        .with_intercept(false)
        .compute_inference(true)
        .tolerance(1e-13)
        .build()
        .fit(&design(&[&X1, &X2]), &col(&Y_GAMMA))
        .unwrap();
    assert_close(
        &table(&f, f.inner().deviance),
        &GAMMA_NOINT,
        1e-7,
        "gamma no intercept",
    );
}

#[test]
fn gamma_with_offset_matches_glm() {
    let offset = Col::from_fn(N, |i| EXPO[i].ln());
    let f = GammaRegressor::builder()
        .offset(offset)
        .compute_inference(true)
        .tolerance(1e-13)
        .build()
        .fit(&design(&[&X1]), &col(&Y_GAMMA_OFFSET))
        .unwrap();
    assert_close(
        &table(&f, f.inner().deviance),
        &GAMMA_OFFSET,
        1e-7,
        "gamma offset",
    );
}

#[test]
fn tweedie_default_link_is_one_minus_p() {
    let f = TweedieRegressor::builder()
        .var_power(1.5)
        .compute_inference(true)
        .tolerance(1e-13)
        .build()
        .fit(&design(&[&X1, &X2]), &col(&Y_CPG))
        .unwrap();
    let r = f.result();
    let se = r.std_errors.as_ref().unwrap();
    let got = [
        r.intercept.unwrap(),
        r.coefficients[0],
        r.coefficients[1],
        r.intercept_std_error.unwrap(),
        se[0],
        se[1],
        f.deviance,
    ];
    assert_close(
        &got,
        &TWEEDIE_DEFAULT_LINK,
        1e-7,
        "tweedie p = 1.5, link 1 - p",
    );
}

#[test]
fn tweedie_power_zero_is_lm() {
    let f = TweedieRegressor::builder()
        .var_power(0.0)
        .compute_inference(true)
        .tolerance(1e-13)
        .build()
        .fit(&design(&[&X1, &X2]), &col(&Y_NORMAL))
        .unwrap();
    let r = f.result();
    let se = r.std_errors.as_ref().unwrap();
    let got = [
        r.intercept.unwrap(),
        r.coefficients[0],
        r.coefficients[1],
        r.intercept_std_error.unwrap(),
        se[0],
        se[1],
        f.deviance,
        r.log_likelihood,
    ];
    assert_close(&got, &TWEEDIE_NORMAL, 1e-9, "tweedie p = 0");
}

#[test]
fn inverse_gaussian_log_likelihood_matches_glm() {
    let f = TweedieRegressor::inverse_gaussian()
        .tolerance(1e-13)
        .build()
        .fit(&design(&[&X1, &X2]), &col(&Y_INVGAUSS))
        .unwrap();
    let r = f.result();
    let got = [
        r.intercept.unwrap(),
        r.coefficients[0],
        r.coefficients[1],
        f.deviance,
        r.log_likelihood,
        r.aic,
        r.bic,
    ];
    assert_close(&got, &INVGAUSS_LOGLIK, 1e-7, "inverse gaussian");
}

fn coefs(fit: &impl FittedRegressor) -> Vec<f64> {
    let r = fit.result();
    [
        vec![r.intercept.unwrap()],
        r.coefficients.iter().copied().collect(),
    ]
    .concat()
}

#[test]
fn ridge_irls_matches_glmnet_alpha_zero() {
    let x = design(&[&X1, &X2]);
    let p = PoissonRegressor::log()
        .lambda(LAMBDA)
        .tolerance(1e-13)
        .build()
        .fit(&x, &col(&Y_POISSON))
        .unwrap();
    assert_close(&coefs(&p), &RIDGE_POISSON, 1e-8, "poisson ridge");
    let b = BinomialRegressor::logistic()
        .lambda(LAMBDA)
        .tolerance(1e-13)
        .build()
        .fit(&x, &col(&Y_BINOMIAL))
        .unwrap();
    assert_close(&coefs(&b), &RIDGE_BINOMIAL, 1e-8, "binomial ridge");
    let nb = NegativeBinomialRegressor::with_theta(2.0)
        .lambda(LAMBDA)
        .tolerance(1e-13)
        .build()
        .fit(&x, &col(&Y_NEGBIN))
        .unwrap();
    assert_close(&coefs(&nb), &RIDGE_NEGBIN, 1e-8, "negative binomial ridge");
    // Gamma(log) IRLS converges linearly; the deviance criterion gives
    // ~√tol coefficient accuracy (#206), hence the tighter tolerance.
    let g = GammaRegressor::builder()
        .lambda(LAMBDA)
        .tolerance(1e-15)
        .max_iterations(1000)
        .build()
        .fit(&x, &col(&Y_GAMMA))
        .unwrap();
    assert_close(&coefs(&g), &RIDGE_GAMMA, 1e-8, "gamma ridge");
}

fn check_predict(fit: &impl BinomialPredict, want: &[f64; 18], what: &str) {
    let new = design(&[&NEW_X1, &NEW_X2]);
    let link = fit.predict_se(&new, PredictionType::Link);
    let resp = fit.predict_se(&new, PredictionType::Response);
    let got = [
        link.fit.iter().copied().collect::<Vec<_>>(),
        link.se.iter().copied().collect(),
        resp.fit.iter().copied().collect(),
        resp.se.iter().copied().collect(),
        resp.lower.iter().copied().collect(),
        resp.upper.iter().copied().collect(),
    ]
    .concat();
    assert_close(&got, want, 1e-7, what);
}

trait BinomialPredict {
    fn predict_se(
        &self,
        x: &Mat<f64>,
        t: PredictionType,
    ) -> anofox_regression::core::PredictionResult;
}

impl BinomialPredict for anofox_regression::solvers::FittedBinomial {
    fn predict_se(
        &self,
        x: &Mat<f64>,
        t: PredictionType,
    ) -> anofox_regression::core::PredictionResult {
        self.predict_with_se(x, t, Some(IntervalType::Confidence), 0.9)
    }
}

#[test]
fn binomial_predict_with_se_probit_cloglog_matches_predict_glm() {
    let x = design(&[&X1, &X2]);
    let y = col(&Y_BINOMIAL);
    let probit = BinomialRegressor::probit()
        .compute_inference(true)
        .tolerance(1e-13)
        .build()
        .fit(&x, &y)
        .unwrap();
    check_predict(&probit, &PREDICT_PROBIT, "probit");
    let cloglog = BinomialRegressor::cloglog()
        .compute_inference(true)
        .tolerance(1e-13)
        .build()
        .fit(&x, &y)
        .unwrap();
    check_predict(&cloglog, &PREDICT_CLOGLOG, "cloglog");
}

/// Step-halving and convergence used the unpenalised deviance, which rises
/// on the way to the shrunken solution: Poisson needed ~2600 iterations
/// (beyond the default limit) and Gamma failed to converge.
#[test]
fn ridge_irls_converges_with_default_options() {
    let x = design(&[&X1, &X2]);
    let p = PoissonRegressor::log()
        .lambda(LAMBDA)
        .build()
        .fit(&x, &col(&Y_POISSON))
        .unwrap();
    assert!(p.iterations < 20, "poisson iterations {}", p.iterations);
    assert_close(&coefs(&p), &RIDGE_POISSON, 1e-6, "poisson ridge, defaults");
    let t = TweedieRegressor::gamma()
        .lambda(LAMBDA)
        .build()
        .fit(&x, &col(&Y_GAMMA))
        .unwrap();
    assert!(t.iterations < 25, "gamma iterations {}", t.iterations);
    assert_close(&coefs(&t), &RIDGE_GAMMA, 1e-3, "gamma ridge, defaults");
    let b = BinomialRegressor::logistic()
        .lambda(LAMBDA)
        .build()
        .fit(&x, &col(&Y_BINOMIAL))
        .unwrap();
    assert_close(
        &coefs(&b),
        &RIDGE_BINOMIAL,
        1e-6,
        "binomial ridge, defaults",
    );
}
