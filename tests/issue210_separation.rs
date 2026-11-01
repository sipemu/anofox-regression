//! #210: exact (linear-program) separation detection, which sees separation
//! by a combination of predictors that the per-column heuristic
//! `check_binary_separation` misses.
use anofox_regression::diagnostics::{check_binary_separation, detect_separation};
use faer::{Col, Mat};

/// Deterministic pseudo-random values in (−1, 1).
fn noise(seed: u64, n: usize) -> Vec<f64> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        })
        .collect()
}

#[test]
fn separation_by_a_combination_of_predictors() {
    let n = 60;
    let (a, b) = (noise(1, n), noise(2, n));
    let x = Mat::from_fn(n, 2, |i, j| if j == 0 { a[i] } else { b[i] });
    let y = Col::from_fn(n, |i| f64::from(u8::from(a[i] + b[i] > 0.0)));
    // neither column alone separates
    assert!(!check_binary_separation(&x, &y).has_separation);
    let sep = detect_separation(&x, &y, true);
    assert!(sep.separated);
    assert_eq!(sep.infinite[1], f64::INFINITY);
    assert_eq!(sep.infinite[2], f64::INFINITY);
    // the direction weakly separates every row
    for i in 0..n {
        let s = if y[i] > 0.5 { 1.0 } else { -1.0 };
        let eta = sep.direction[0] + sep.direction[1] * a[i] + sep.direction[2] * b[i];
        assert!(s * eta >= -1e-9, "row {i}: {}", s * eta);
    }
    assert!(sep.direction.iter().all(|d| d.abs() <= 1.0 + 1e-12));
}

#[test]
fn overlapping_classes_are_not_separated() {
    let n = 80;
    let (a, b, e) = (noise(3, n), noise(4, n), noise(5, n));
    let x = Mat::from_fn(n, 2, |i, j| if j == 0 { a[i] } else { b[i] });
    let y = Col::from_fn(n, |i| f64::from(u8::from(a[i] - b[i] + 1.5 * e[i] > 0.0)));
    let sep = detect_separation(&x, &y, true);
    assert!(!sep.separated);
    assert!(sep.infinite.iter().all(|&v| v == 0.0));
    assert!(sep.direction.iter().all(|&v| v == 0.0));
}

#[test]
fn quasi_separation_by_a_binary_predictor() {
    // group 1: all y = 1; group 0: both classes, so only quasi-separation
    let n = 40;
    let e = noise(6, n);
    let g: Vec<f64> = (0..n).map(|i| f64::from(u8::from(i % 3 == 0))).collect();
    let x = Mat::from_fn(n, 2, |i, j| if j == 0 { g[i] } else { e[i] });
    let y = Col::from_fn(n, |i| {
        if g[i] == 1.0 {
            1.0
        } else {
            f64::from(u8::from(e[i] + 0.3 * noise(7, n)[i] > 0.0))
        }
    });
    let sep = detect_separation(&x, &y, true);
    assert!(sep.separated);
    assert_eq!(sep.infinite[1], f64::INFINITY, "{:?}", sep.direction);
}

#[test]
fn without_intercept() {
    // y = 1 exactly where x > 0: separated through the origin
    let n = 30;
    let a = noise(8, n);
    let x = Mat::from_fn(n, 1, |i, _| a[i]);
    let y = Col::from_fn(n, |i| f64::from(u8::from(a[i] > 0.0)));
    let sep = detect_separation(&x, &y, false);
    assert!(sep.separated);
    assert_eq!(sep.infinite, vec![f64::INFINITY]);
    // shifted classes are not separable through the origin alone
    let y = Col::from_fn(n, |i| {
        f64::from(u8::from(a[i] > 0.4) | u8::from(a[i] < -0.6))
    });
    assert!(!detect_separation(&x, &y, false).separated);
}
