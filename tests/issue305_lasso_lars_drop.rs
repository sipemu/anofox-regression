//! Issue #305: LassoLars must drop a variable whose coefficient crosses zero
//! and must not re-admit it straight away with its old sign.
//!
//! References from scikit-learn 1.8 `lars_path(method="lasso")` on the
//! centred data and `LassoLars(alpha)` (sklearn's alpha is the max
//! correlation / n, so it is multiplied by n = 16 here).

use anofox_regression::solvers::{FittedRegressor, LarsMethod, LarsRegressor, Regressor};
use faer::{Col, Mat};

/// 16×4 design (sklearn `RandomState(19)`, rounded) whose Lasso path drops
/// feature 2 at the 4th knot.
fn drop_data() -> (Mat<f64>, Col<f64>) {
    let x = [
        0.22, -0.34, -0.15, -0.4, -0.6, -0.24, -0.27, 0.58, 0.25, -1.46, -0.84, 0.27, -0.0, -1.02,
        -0.49, -1.19, 1.56, -0.93, 0.39, -0.15, 0.52, 1.34, 1.17, 1.26, 0.91, -0.84, 0.18, -1.02,
        0.18, -0.42, 0.18, 1.92, -1.03, -0.48, -0.85, 0.06, 0.55, 0.85, 0.52, 0.37, 0.94, -0.35,
        0.36, -0.05, 0.42, -0.7, -0.12, 0.47, 0.04, 0.72, 0.31, -0.05, -0.27, 0.31, -0.25, 1.15,
        -0.22, -1.22, -0.77, 1.55, -0.21, -0.24, -0.24, 0.56,
    ];
    let y = [
        0.41, -0.19, -0.73, -1.47, 0.58, 1.9, -0.61, -0.29, -1.87, 2.17, 0.7, -0.11, 0.4, 0.32,
        -1.39, -1.02,
    ];
    (
        Mat::from_fn(16, 4, |i, j| x[i * 4 + j]),
        Col::from_fn(16, |i| y[i]),
    )
}

fn close(a: &[f64], b: &[f64], tol: f64) {
    assert_eq!(a.len(), b.len());
    for (i, (u, v)) in a.iter().zip(b).enumerate() {
        assert!((u - v).abs() < tol, "[{i}] {u} vs {v}");
    }
}

#[test]
fn lasso_path_drops_variable_like_sklearn() {
    let (x, y) = drop_data();
    let f = LarsRegressor::builder()
        .method(LarsMethod::Lasso)
        .build()
        .fit(&x, &y)
        .unwrap();
    let sk_alphas = [
        9.637599999999999,
        6.153681623321029,
        5.0325063239429095,
        3.9815274576016355,
        1.4225017411130627,
        0.1862839737651476,
        0.0,
    ];
    let alphas: Vec<f64> = f.alphas().iter().copied().collect();
    close(&alphas, &sk_alphas, 1e-10);

    let path = f.coefs_path();
    let sk_path = [
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.385_449_960_494_765_04, 0.0, 0.0],
        [0.0, 0.386_944_471_149_059, 0.246_255_628_758_316_1, 0.0],
        [0.321_889_903_606_056, 0.635_178_422_227_341_7, 0.0, 0.0],
        [0.764_262_143_924_932_4, 0.931_230_554_679_062_5, 0.0, 0.0],
        [
            1.024_591_507_055_361_7,
            1.035_482_611_678_038_7,
            0.0,
            0.137_579_926_606_402_2,
        ],
        [
            1.744_031_637_664_198,
            1.636_733_340_401_942_2,
            -1.165_447_879_452_730_5,
            0.207_420_765_652_578,
        ],
    ];
    assert_eq!(path.len(), sk_path.len());
    for (row, sk) in path.iter().zip(&sk_path) {
        close(row, sk, 1e-10);
    }
    // Feature 2 left the path at knot 3 and re-entered with the opposite
    // sign at the last step.
    assert_eq!(f.active(), &[1, 0, 3, 2]);
}

#[test]
fn lasso_alpha_matches_sklearn_after_drop() {
    let (x, y) = drop_data();
    for (alpha, coef, intercept) in [
        (
            0.01,
            [
                1.1261018256099606,
                1.1203168465327138,
                -0.16444034808276728,
                0.147434208454286,
            ],
            -0.01099930098672497,
        ),
        (
            0.003,
            [
                1.5586526940479257,
                1.4818083922411727,
                -0.8651456200417398,
                0.18942479849309032,
            ],
            -0.037802532434377215,
        ),
    ] {
        let f = LarsRegressor::builder()
            .method(LarsMethod::Lasso)
            .alpha(alpha * 16.0)
            .build()
            .fit(&x, &y)
            .unwrap();
        let b: Vec<f64> = f.result().coefficients.iter().copied().collect();
        close(&b, &coef, 1e-9);
        assert!((f.result().intercept.unwrap() - intercept).abs() < 1e-9);
        assert_eq!(f.active(), &[1, 0, 3, 2]);
    }
}

#[test]
fn plain_lar_active_set_in_entry_order() {
    let (x, y) = drop_data();
    let f = LarsRegressor::builder()
        .n_nonzero_coefs(2)
        .build()
        .fit(&x, &y)
        .unwrap();
    assert_eq!(f.active(), &[1, 2]);
    let b: Vec<f64> = f.result().coefficients.iter().copied().collect();
    close(
        &b,
        &[0.0, 0.386944471149059, 0.24625562875831633, 0.0],
        1e-10,
    );
}
