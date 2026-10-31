//! #210: RANSAC breaks ties in the inlier count as sklearn's
//! `RANSACRegressor` — the subset model with the higher R² on its inliers —
//! instead of keeping the first subset seen, which depended on the sampling
//! order.
use anofox_regression::solvers::{RansacRegressor, Regressor};
use faer::{Col, Mat};

#[test]
fn equal_inlier_counts_prefer_the_better_fitting_set() {
    // Two consensus sets of 8 points: y = x exactly (R² = 1 on its inliers)
    // and y ≈ 40 − x with noise (R² < 1). No point is within the threshold
    // of the other line.
    let noise = [0.04, -0.03, 0.05, -0.05, 0.02, -0.04, 0.03, -0.02];
    let n = 16;
    let x = Mat::from_fn(n, 1, |i, _| if i < 8 { i as f64 } else { (i + 2) as f64 });
    let y = Col::from_fn(n, |i| {
        if i < 8 {
            i as f64
        } else {
            40.0 - (i + 2) as f64 + noise[i - 8]
        }
    });
    for seed in 0..25 {
        let f = RansacRegressor::builder()
            .min_samples(2)
            .residual_threshold(0.5)
            .max_trials(400)
            .stop_probability(1.0)
            .random_state(seed)
            .build()
            .fit(&x, &y)
            .unwrap();
        let mask = f.inlier_mask();
        assert!(
            (0..8).all(|i| mask[i]) && (8..16).all(|i| !mask[i]),
            "seed {seed}: kept the noisy set {mask:?}"
        );
    }
}
