//! Issue #307: multinomial (softmax) logistic regression and L1 / elastic-net
//! penalties, against scikit-learn 1.9.1 `LogisticRegression` (lbfgs with
//! `tol=1e-12` for L2, saga with `tol=1e-14` for L1 / elastic-net).
//!
//! sklearn's lbfgs solutions have gradients of ~1e-7, so they are compared
//! at 1e-7; `binary_l2_satisfies_kkt` checks our optimum directly.
//!
//! Data: anofox-ml's `tests/sklearn_logistic.rs` (RandomState(42), 60 × 3,
//! binary `YB`, 3-class `Y3`, weights `W`).

use anofox_regression::solvers::{IterativeFit, MultinomialLogisticRegressor};
use faer::{Col, Mat};

const X: [f64; 180] = [
    0.497, -0.138, 0.648, 1.523, -0.234, -0.234, 1.579, 0.767, -0.469, 0.543, -0.463, -0.466,
    0.242, -1.913, -1.725, -0.562, -1.013, 0.314, -0.908, -1.412, 1.466, -0.226, 0.068, -1.425,
    -0.544, 0.111, -1.151, 0.376, -0.601, -0.292, -0.602, 1.852, -0.013, -1.058, 0.823, -1.221,
    0.209, -1.96, -1.328, 0.197, 0.738, 0.171, -0.116, -0.301, -1.479, -0.72, -0.461, 1.057, 0.344,
    -1.763, 0.324, -0.385, -0.677, 0.612, 1.031, 0.931, -0.839, -0.309, 0.331, 0.976, -0.479,
    -0.186, -1.106, -1.196, 0.813, 1.356, -0.072, 1.004, 0.362, -0.645, 0.361, 1.538, -0.036,
    1.565, -2.62, 0.822, 0.087, -0.299, 0.092, -1.988, -0.22, 0.357, 1.478, -0.518, -0.808, -0.502,
    0.915, 0.329, -0.53, 0.513, 0.097, 0.969, -0.702, -0.328, -0.392, -1.464, 0.296, 0.261, 0.005,
    -0.235, -1.415, -0.421, -0.343, -0.802, -0.161, 0.404, 1.886, 0.175, 0.258, -0.074, -1.919,
    -0.027, 0.06, 2.463, -0.192, 0.302, -0.035, -1.169, 1.143, 0.752, 0.791, -0.909, 1.403, -1.402,
    0.587, 2.19, -0.991, -0.566, 0.1, -0.503, -1.551, 0.069, -1.062, 0.474, -0.919, 1.55, -0.783,
    -0.322, 0.814, -1.231, 0.227, 1.307, -1.607, 0.185, 0.26, 0.782, -1.237, -1.32, 0.522, 0.297,
    0.25, 0.346, -0.68, 0.232, 0.293, -0.714, 1.866, 0.474, -1.191, 0.657, -0.975, 0.787, 1.159,
    -0.821, 0.963, 0.413, 0.822, 1.897, -0.245, -0.754, -0.89, -0.816, -0.077, 0.341, 0.277, 0.827,
    0.013, 1.454, -0.265, 2.72,
];
const YB: [f64; 60] = [
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0,
    1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 0.0, 1.0,
];
const Y3: [f64; 60] = [
    2.0, 0.0, 2.0, 1.0, 0.0, 0.0, 2.0, 1.0, 1.0, 2.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 2.0,
    1.0, 2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 2.0, 2.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0,
    0.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 2.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    1.0, 0.0, 2.0,
];
const W: [f64; 60] = [
    0.4, 2.0, 0.27, 1.84, 2.83, 1.81, 1.29, 2.0, 1.48, 1.73, 2.84, 1.28, 2.89, 2.73, 0.75, 0.39,
    0.48, 0.25, 0.46, 2.11, 0.4, 1.09, 2.57, 0.27, 2.48, 0.99, 0.53, 2.15, 1.96, 2.66, 2.26, 2.45,
    0.99, 0.7, 2.3, 2.46, 2.97, 1.36, 1.24, 2.37, 1.15, 2.81, 2.6, 1.4, 2.3, 2.31, 0.49, 2.73,
    1.61, 2.51, 1.1, 2.71, 1.29, 0.23, 2.74, 0.46, 1.09, 2.86, 2.86, 1.81,
];

fn x() -> Mat<f64> {
    Mat::from_fn(60, 3, |i, j| X[i * 3 + j])
}
fn col(v: &[f64]) -> Col<f64> {
    Col::from_fn(v.len(), |i| v[i])
}
fn close(what: &str, got: &[f64], want: &[f64], tol: f64) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!((g - w).abs() < tol, "{what}[{i}]: {g} vs sklearn {w}");
    }
}
fn flat(m: &Mat<f64>) -> Vec<f64> {
    (0..m.nrows())
        .flat_map(|i| (0..m.ncols()).map(move |j| (i, j)))
        .map(|(i, j)| m[(i, j)])
        .collect()
}

fn check(
    what: &str,
    b: anofox_regression::solvers::MultinomialLogisticRegressorBuilder,
    y: &[f64],
    coef: &[f64],
    intercept: &[f64],
    tol: f64,
) -> anofox_regression::solvers::FittedMultinomialLogistic {
    let f = b.build().fit(&x(), &col(y)).unwrap();
    assert!(f.converged(), "{what}: not converged");
    close(what, &flat(f.coefficients()), coef, tol);
    let b0: Vec<f64> = f.intercepts().iter().copied().collect();
    close(what, &b0, intercept, tol);
    f
}

#[test]
fn multinomial_l2_matches_sklearn() {
    let f = check(
        "multinomial C=1",
        MultinomialLogisticRegressor::builder(),
        &Y3,
        &[
            0.5453637893962024,
            -0.35904004469501577,
            -0.34214973475422816,
            -0.4684985998162452,
            0.4253202613010895,
            -0.2564204079806024,
            -0.07686518957995704,
            -0.0662802166060746,
            0.5985701427348307,
        ],
        &[0.3925209572267445, -0.5389281686445537, 0.1464072114178093],
        1e-7,
    );
    assert_eq!(f.classes(), &[0.0, 1.0, 2.0]);
    let x2 = Mat::from_fn(2, 3, |i, j| X[i * 3 + j]);
    close(
        "proba",
        &flat(&f.predict_proba(&x2)),
        &[
            0.44648086694812456,
            0.10082322102621634,
            0.45269591202565906,
            0.7717764494031475,
            0.05295976191565734,
            0.175263788681195,
        ],
        1e-7,
    );
    let pred: Vec<f64> = f.predict(&x2).iter().copied().collect();
    assert_eq!(pred, vec![2.0, 0.0]);
    let eta = f.decision_function(&x2);
    assert_eq!((eta.nrows(), eta.ncols()), (2, 3));
    assert!(f.score(&x(), &col(&Y3)) > 0.5);
    assert!(f.log_likelihood() < 0.0 && f.objective() > -f.log_likelihood());
}

#[test]
fn multinomial_weighted_matches_sklearn() {
    check(
        "multinomial C=0.3 weighted",
        MultinomialLogisticRegressor::builder()
            .c(0.3)
            .weights(col(&W)),
        &Y3,
        &[
            0.5536813825643878,
            -0.22842894726525964,
            -0.5398282447108845,
            -0.26938792953731727,
            0.4101264030468333,
            -0.21741198744197124,
            -0.28429345302707076,
            -0.1816974557815736,
            0.7572402321528559,
        ],
        &[
            0.3743166196480657,
            -0.3669552108047707,
            -0.00736140884329499,
        ],
        1e-7,
    );
}

#[test]
fn multinomial_without_intercept_matches_sklearn() {
    check(
        "multinomial no intercept",
        MultinomialLogisticRegressor::builder().with_intercept(false),
        &Y3,
        &[
            0.5800883440054361,
            -0.37616974571171874,
            -0.4894896555970282,
            -0.371885373123477,
            0.3540737138323686,
            -0.14013909044153794,
            -0.20820297088195877,
            0.0220960318793505,
            0.6296287460385661,
        ],
        &[0.0, 0.0, 0.0],
        1e-7,
    );
}

#[test]
fn binary_l2_matches_sklearn() {
    check(
        "binary C=1",
        MultinomialLogisticRegressor::builder(),
        &YB,
        &[1.1165985805533376, -0.9543015443975524, 0.26609388872584333],
        &[0.337744297594217],
        1e-7,
    );
}

#[test]
fn binary_l1_and_elastic_net_match_sklearn_saga() {
    check(
        "binary L1 C=0.5",
        MultinomialLogisticRegressor::builder().c(0.5).l1_ratio(1.0),
        &YB,
        &[0.8906884616279431, -0.8214379009127455, 0.10453903054580903],
        &[0.31326346483484513],
        1e-7,
    );
    check(
        "binary EN C=0.5",
        MultinomialLogisticRegressor::builder().c(0.5).l1_ratio(0.5),
        &YB,
        &[0.9231724614396523, -0.8374083313221147, 0.1702674208384755],
        &[0.3185754398762322],
        1e-7,
    );
    check(
        "binary L1 C=0.2 weighted",
        MultinomialLogisticRegressor::builder()
            .c(0.2)
            .l1_ratio(1.0)
            .weights(col(&W)),
        &YB,
        &[0.5595585742976685, -0.6339198842606191, 0.11911144102915061],
        &[0.309717636148254],
        1e-7,
    );
}

#[test]
fn l1_gives_exact_zeros() {
    let f = check(
        "binary L1 C=0.05",
        MultinomialLogisticRegressor::builder()
            .c(0.05)
            .l1_ratio(1.0),
        &YB,
        &[0.0, 0.0, 0.0],
        // sklearn's saga stops at intercept 0.247; with all slopes at 0 the
        // optimum is logit(ȳ) = logit(35/60).
        &[0.336472236621213],
        1e-7,
    );
    assert!(f.coefficients().col_iter().all(|c| c[0] == 0.0));
}

#[test]
fn multinomial_l1_and_elastic_net_match_sklearn_saga() {
    let f = check(
        "multinomial L1 C=0.2",
        MultinomialLogisticRegressor::builder().c(0.2).l1_ratio(1.0),
        &Y3,
        &[
            0.18265592134767228,
            -0.08053482965695696,
            0.0,
            0.0,
            0.02189457840470761,
            0.0,
            0.0,
            0.0,
            0.5313717093182582,
        ],
        &[0.4052442083325136, -0.508002090607541, 0.10275788227504525],
        1e-7,
    );
    assert_eq!(
        flat(f.coefficients()).iter().filter(|c| **c == 0.0).count(),
        5
    );
    check(
        "multinomial EN C=0.5 weighted",
        MultinomialLogisticRegressor::builder()
            .c(0.5)
            .l1_ratio(0.3)
            .weights(col(&W)),
        &Y3,
        &[
            0.7175999152084797,
            -0.08578369004335387,
            -0.42602379368963955,
            -0.1386504137437084,
            0.5515660457517523,
            -0.08132115896024068,
            -0.15037807289337807,
            -0.03721092713700795,
            0.9359163812212686,
        ],
        &[
            0.3704763668349445,
            -0.3574965567305725,
            -0.01297981010437426,
        ],
        1e-7,
    );
}

#[test]
fn arbitrary_labels_are_sorted_classes() {
    let y: Vec<f64> = Y3.iter().map(|v| [7.0, -1.0, 3.5][*v as usize]).collect();
    let f = MultinomialLogisticRegressor::builder()
        .build()
        .fit(&x(), &col(&y))
        .unwrap();
    assert_eq!(f.classes(), &[-1.0, 3.5, 7.0]);
    let p = f.predict_proba(&x());
    for i in 0..60 {
        let s: f64 = (0..3).map(|k| p[(i, k)]).sum();
        assert!((s - 1.0).abs() < 1e-12);
    }
}

/// The gradient of the penalized objective vanishes at the fit.
#[test]
fn binary_l2_satisfies_kkt() {
    let f = MultinomialLogisticRegressor::builder()
        .build()
        .fit(&x(), &col(&YB))
        .unwrap();
    let xm = x();
    let b0 = f.intercepts()[0];
    let b = f.coefficients();
    let mut g = [0.0; 4];
    for i in 0..60 {
        let eta = b0 + (0..3).map(|j| xm[(i, j)] * b[(0, j)]).sum::<f64>();
        let r = 1.0 / (1.0 + (-eta).exp()) - YB[i];
        g[0] += r;
        for j in 0..3 {
            g[j + 1] += r * xm[(i, j)];
        }
    }
    for j in 0..3 {
        g[j + 1] += b[(0, j)];
    }
    assert!(g.iter().all(|v| v.abs() < 1e-12), "{g:?}");
}
