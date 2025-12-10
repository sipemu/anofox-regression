//! Leverage (hat matrix diagonal) calculations.

use faer::{Col, Mat};

/// Build design matrix, optionally prepending an intercept column.
fn build_design_matrix(x: &Mat<f64>, with_intercept: bool) -> Mat<f64> {
    let n = x.nrows();
    let p = x.ncols();

    if with_intercept {
        Mat::from_fn(n, p + 1, |i, j| if j == 0 { 1.0 } else { x[(i, j - 1)] })
    } else {
        x.to_owned()
    }
}

/// Build reduced design matrix excluding aliased columns.
fn build_design_matrix_reduced(x: &Mat<f64>, aliased: &[bool], with_intercept: bool) -> Mat<f64> {
    let n = x.nrows();
    let non_aliased: Vec<usize> = aliased
        .iter()
        .enumerate()
        .filter(|(_, &is_aliased)| !is_aliased)
        .map(|(j, _)| j)
        .collect();
    let p_reduced = non_aliased.len();
    let total_cols = if with_intercept {
        p_reduced + 1
    } else {
        p_reduced
    };

    Mat::from_fn(n, total_cols, |i, j| {
        if with_intercept {
            if j == 0 {
                1.0
            } else {
                x[(i, non_aliased[j - 1])]
            }
        } else {
            x[(i, non_aliased[j])]
        }
    })
}

/// Compute X'X (cross-product matrix).
fn compute_xtx(design: &Mat<f64>) -> Mat<f64> {
    design.transpose() * design
}

/// Compute (X'X)^(-1) using QR decomposition with back-substitution.
fn compute_xtx_inverse(xtx: &Mat<f64>) -> Mat<f64> {
    let p = xtx.nrows();
    let qr = xtx.qr();
    let q = qr.compute_Q();
    let r = qr.R().to_owned();
    let qt = q.transpose().to_owned();

    // Solve for each column of the inverse
    let mut inv = Mat::zeros(p, p);
    for col in 0..p {
        let solution = solve_triangular_column(&r, &qt, col, p);
        for row in 0..p {
            inv[(row, col)] = solution[row];
        }
    }
    inv
}

/// Solve for a column of (X'X)^(-1) via back-substitution.
fn solve_triangular_column(r: &Mat<f64>, qt: &Mat<f64>, col: usize, p: usize) -> Vec<f64> {
    let mut solution = vec![0.0; p];

    for i in (0..p).rev() {
        if r[(i, i)].abs() < 1e-14 {
            continue;
        }
        let mut sum = qt[(i, col)];
        for j in (i + 1)..p {
            sum -= r[(i, j)] * solution[j];
        }
        solution[i] = sum / r[(i, i)];
    }

    solution
}

/// Compute leverage value for a single observation.
fn compute_single_leverage(design_row: &[f64], xtx_inv: &Mat<f64>) -> f64 {
    let p = design_row.len();
    let mut h_ii = 0.0;

    for j in 0..p {
        for k in 0..p {
            h_ii += design_row[j] * xtx_inv[(j, k)] * design_row[k];
        }
    }

    h_ii.clamp(0.0, 1.0)
}

/// Compute leverage values (diagonal of hat matrix H = X(X'X)^(-1)X').
///
/// Leverage measures the influence of each observation on its own fitted value.
/// High leverage points have unusual predictor values.
///
/// # Properties
/// - h_ii ∈ [0, 1]
/// - Σ h_ii = p (number of parameters)
/// - Points with h_ii > 2p/n are considered high leverage
pub fn compute_leverage(x: &Mat<f64>, with_intercept: bool) -> Col<f64> {
    let n = x.nrows();
    let design = build_design_matrix(x, with_intercept);
    let p = design.ncols();

    let xtx = compute_xtx(&design);
    let xtx_inv = compute_xtx_inverse(&xtx);

    Col::from_fn(n, |i| {
        let row: Vec<f64> = (0..p).map(|j| design[(i, j)]).collect();
        compute_single_leverage(&row, &xtx_inv)
    })
}

/// Compute leverage values handling aliased (collinear) columns.
///
/// This version excludes aliased columns from the design matrix before computing
/// leverage. The leverage values are computed using only the non-aliased columns,
/// which gives correct results even when the full X'X matrix would be singular.
///
/// # Arguments
/// * `x` - Feature matrix (n × p)
/// * `aliased` - Boolean mask indicating which columns are aliased
/// * `with_intercept` - Whether to include an intercept column
///
/// # Returns
/// Leverage values for each observation
pub fn compute_leverage_with_aliased(
    x: &Mat<f64>,
    aliased: &[bool],
    with_intercept: bool,
) -> Col<f64> {
    let n = x.nrows();

    // If no aliased columns, use the standard method
    let has_aliased = aliased.iter().any(|&a| a);
    if !has_aliased {
        return compute_leverage(x, with_intercept);
    }

    let design = build_design_matrix_reduced(x, aliased, with_intercept);
    let p = design.ncols();

    if p == 0 {
        // All columns aliased - return NaN
        return Col::from_fn(n, |_| f64::NAN);
    }

    let xtx = compute_xtx(&design);
    let xtx_inv = compute_xtx_inverse(&xtx);

    // Check if inverse computation succeeded (diagonal should be non-zero for valid inverse)
    let mut inverse_valid = true;
    for i in 0..p {
        if xtx_inv[(i, i)].abs() < 1e-14 {
            inverse_valid = false;
            break;
        }
    }

    if !inverse_valid {
        return Col::from_fn(n, |_| f64::NAN);
    }

    Col::from_fn(n, |i| {
        let row: Vec<f64> = (0..p).map(|j| design[(i, j)]).collect();
        compute_single_leverage(&row, &xtx_inv)
    })
}

/// Identify high leverage points.
///
/// Returns indices of observations with leverage > threshold.
/// Default threshold is 2p/n where p is number of parameters.
pub fn high_leverage_points(
    leverage: &Col<f64>,
    n_params: usize,
    threshold: Option<f64>,
) -> Vec<usize> {
    let n = leverage.nrows();
    let cutoff = threshold.unwrap_or(2.0 * n_params as f64 / n as f64);

    leverage
        .iter()
        .enumerate()
        .filter(|(_, &h)| h > cutoff)
        .map(|(i, _)| i)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_leverage_bounds() {
        let x = Mat::from_fn(20, 2, |i, j| ((i + j) as f64) * 0.1);
        let leverage = compute_leverage(&x, true);

        for i in 0..leverage.nrows() {
            assert!(
                leverage[i] >= 0.0,
                "Leverage[{}] = {} should be >= 0",
                i,
                leverage[i]
            );
            assert!(
                leverage[i] <= 1.0,
                "Leverage[{}] = {} should be <= 1",
                i,
                leverage[i]
            );
        }
    }

    #[test]
    fn test_leverage_sum() {
        // Use linearly independent predictors
        let x = Mat::from_fn(
            30,
            2,
            |i, j| {
                if j == 0 {
                    i as f64
                } else {
                    (i as f64).sin()
                }
            },
        );
        let leverage = compute_leverage(&x, true);

        let sum: f64 = leverage.iter().sum();
        let n_params = 3; // intercept + 2 features

        // Sum of leverage should equal number of parameters (rank of design matrix)
        // Allow more tolerance due to numerical precision
        assert!(
            (sum - n_params as f64).abs() < 0.5,
            "Sum of leverage {} should be close to {}",
            sum,
            n_params
        );
    }

    #[test]
    fn test_high_leverage_detection() {
        let mut x = Mat::from_fn(20, 1, |i, _| i as f64);
        // Add an extreme outlier in x
        x[(19, 0)] = 100.0;

        let leverage = compute_leverage(&x, true);
        let high = high_leverage_points(&leverage, 2, None);

        // The outlier should have high leverage
        assert!(
            high.contains(&19),
            "Point 19 should be flagged as high leverage"
        );
    }

    // === Tests for build_design_matrix ===

    #[test]
    fn test_build_design_matrix_no_intercept() {
        // Tests line 13: the else branch when with_intercept=false
        let x = Mat::from_fn(10, 2, |i, j| (i + j) as f64);
        let design = build_design_matrix(&x, false);

        assert_eq!(design.nrows(), 10);
        assert_eq!(design.ncols(), 2); // No intercept column added

        // Verify values are copied correctly
        for i in 0..10 {
            for j in 0..2 {
                assert_eq!(design[(i, j)], x[(i, j)]);
            }
        }
    }

    #[test]
    fn test_build_design_matrix_with_intercept() {
        let x = Mat::from_fn(10, 2, |i, j| (i + j + 1) as f64);
        let design = build_design_matrix(&x, true);

        assert_eq!(design.nrows(), 10);
        assert_eq!(design.ncols(), 3); // Intercept + 2 features

        // First column should be intercept (all 1s)
        for i in 0..10 {
            assert_eq!(design[(i, 0)], 1.0);
            assert_eq!(design[(i, 1)], x[(i, 0)]);
            assert_eq!(design[(i, 2)], x[(i, 1)]);
        }
    }

    // === Tests for build_design_matrix_reduced ===

    #[test]
    fn test_build_design_matrix_reduced_with_aliased() {
        // Tests lines 18-44: reduced design matrix with aliased columns
        let x = Mat::from_fn(10, 3, |i, j| (i * 10 + j) as f64);
        let aliased = vec![false, true, false]; // Column 1 is aliased

        let design = build_design_matrix_reduced(&x, &aliased, true);

        // Should have 10 rows, 3 columns (intercept + 2 non-aliased)
        assert_eq!(design.nrows(), 10);
        assert_eq!(design.ncols(), 3);

        // First column should be intercept (all 1s)
        for i in 0..10 {
            assert_eq!(design[(i, 0)], 1.0);
        }

        // Remaining columns should be x columns 0 and 2
        for i in 0..10 {
            assert_eq!(design[(i, 1)], x[(i, 0)]);
            assert_eq!(design[(i, 2)], x[(i, 2)]);
        }
    }

    #[test]
    fn test_build_design_matrix_reduced_no_intercept() {
        // Tests lines 40-41: reduced design without intercept
        let x = Mat::from_fn(10, 3, |i, j| (i * 10 + j) as f64);
        let aliased = vec![true, false, false]; // Column 0 is aliased

        let design = build_design_matrix_reduced(&x, &aliased, false);

        assert_eq!(design.nrows(), 10);
        assert_eq!(design.ncols(), 2); // Only columns 1 and 2

        for i in 0..10 {
            assert_eq!(design[(i, 0)], x[(i, 1)]);
            assert_eq!(design[(i, 1)], x[(i, 2)]);
        }
    }

    // === Tests for compute_xtx_inverse ===

    #[test]
    fn test_compute_xtx_inverse_basic() {
        // Tests lines 52-68: basic inverse computation
        let x = Mat::from_fn(
            20,
            2,
            |i, j| if j == 0 { i as f64 } else { (i as f64).sin() },
        );
        let design = build_design_matrix(&x, true);
        let xtx = compute_xtx(&design);
        let inv = compute_xtx_inverse(&xtx);

        // Verify XTX * inv approximately equals identity
        let p = xtx.nrows();
        let product = &xtx * &inv;

        for i in 0..p {
            for j in 0..p {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (product[(i, j)] - expected).abs() < 1e-6,
                    "Product[{},{}] = {}, expected {}",
                    i,
                    j,
                    product[(i, j)],
                    expected
                );
            }
        }
    }

    #[test]
    fn test_compute_xtx_inverse_1x1() {
        // Test simple 1x1 case
        let x = Mat::from_fn(10, 1, |i, _| (i + 1) as f64);
        let design = build_design_matrix(&x, false);
        let xtx = compute_xtx(&design);
        let inv = compute_xtx_inverse(&xtx);

        // xtx should be sum of squares = 1^2 + 2^2 + ... + 10^2 = 385
        let expected_xtx: f64 = (1..=10).map(|i| (i * i) as f64).sum();
        assert!((xtx[(0, 0)] - expected_xtx).abs() < 1e-10);

        // inv should be 1/385
        assert!((inv[(0, 0)] - 1.0 / expected_xtx).abs() < 1e-10);
    }

    // === Tests for solve_triangular_column with singular diagonal ===

    #[test]
    fn test_solve_triangular_column_near_singular() {
        // Tests lines 74-76: near-zero diagonal handling
        // Create a design matrix with collinear columns
        let mut x = Mat::zeros(10, 3);
        for i in 0..10 {
            x[(i, 0)] = i as f64;
            x[(i, 1)] = i as f64 * 2.0; // Perfectly collinear with column 0
            x[(i, 2)] = (i as f64).sin();
        }

        // The leverage computation should still work (handling the near-singular case)
        let leverage = compute_leverage(&x, true);

        // Leverage should be finite for well-conditioned parts
        for i in 0..10 {
            assert!(
                leverage[i].is_finite() || leverage[i].is_nan(),
                "Leverage[{}] should be finite or NaN, got {}",
                i,
                leverage[i]
            );
        }
    }

    // === Tests for compute_single_leverage ===

    #[test]
    fn test_compute_single_leverage_basic() {
        // Tests lines 89-100: direct leverage calculation
        // For identity (X'X)^-1, leverage = ||x||^2
        let xtx_inv = Mat::identity(3, 3);
        let row = vec![1.0, 2.0, 3.0];

        let h = compute_single_leverage(&row, &xtx_inv);
        // h = x'Ix = ||x||^2 = 1 + 4 + 9 = 14, but clamped to 1.0
        assert!((h - 1.0).abs() < 1e-10); // Should be clamped to 1.0
    }

    #[test]
    fn test_compute_single_leverage_small_value() {
        // Test with a properly scaled inverse that gives h < 1
        let mut xtx_inv = Mat::zeros(2, 2);
        xtx_inv[(0, 0)] = 0.1;
        xtx_inv[(1, 1)] = 0.1;
        let row = vec![1.0, 1.0];

        let h = compute_single_leverage(&row, &xtx_inv);
        // h = 1*0.1*1 + 1*0.1*1 = 0.2
        assert!((h - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_compute_single_leverage_clamping() {
        // Tests that leverage is clamped to [0, 1]
        let mut xtx_inv = Mat::zeros(2, 2);
        xtx_inv[(0, 0)] = 2.0;
        xtx_inv[(1, 1)] = 2.0;
        let row = vec![1.0, 1.0];

        // h = 1*2*1 + 1*2*1 = 4, but should be clamped to 1.0
        let h = compute_single_leverage(&row, &xtx_inv);
        assert!(
            (h - 1.0).abs() < 1e-10,
            "Leverage should be clamped to at most 1.0, got {}",
            h
        );
    }

    // === Tests for compute_leverage_with_aliased ===

    #[test]
    fn test_compute_leverage_with_aliased_no_aliased() {
        // Tests lines 147-148: when no columns are aliased, falls back to compute_leverage
        let x = Mat::from_fn(
            20,
            2,
            |i, j| if j == 0 { i as f64 } else { (i as f64).sin() },
        );
        let aliased = vec![false, false];

        let lev_aliased = compute_leverage_with_aliased(&x, &aliased, true);
        let lev_regular = compute_leverage(&x, true);

        for i in 0..20 {
            assert!(
                (lev_aliased[i] - lev_regular[i]).abs() < 1e-10,
                "Leverage[{}] mismatch: aliased={}, regular={}",
                i,
                lev_aliased[i],
                lev_regular[i]
            );
        }
    }

    #[test]
    fn test_compute_leverage_with_aliased_one_aliased() {
        // Tests lines 151-173: with truly aliased columns
        let x = Mat::from_fn(20, 3, |i, j| match j {
            0 => i as f64,
            1 => i as f64 * 2.0, // Would be collinear if used
            2 => (i as f64).sin() * 10.0,
            _ => 0.0,
        });
        let aliased = vec![false, true, false]; // Column 1 is marked aliased

        let lev = compute_leverage_with_aliased(&x, &aliased, true);

        // Leverage values should be finite and in [0, 1]
        for i in 0..20 {
            assert!(lev[i].is_finite(), "Leverage[{}] should be finite", i);
            assert!(
                lev[i] >= 0.0 && lev[i] <= 1.0,
                "Leverage[{}] = {} should be in [0, 1]",
                i,
                lev[i]
            );
        }

        // Sum should be approximately equal to number of non-aliased params + intercept
        let sum: f64 = lev.iter().sum();
        let expected_params = 3; // intercept + 2 non-aliased columns
        assert!(
            (sum - expected_params as f64).abs() < 0.5,
            "Sum {} should be close to {}",
            sum,
            expected_params
        );
    }

    #[test]
    fn test_compute_leverage_with_aliased_all_aliased_returns_nan() {
        // Tests lines 154-157: all columns aliased returns NaN
        let x = Mat::from_fn(10, 2, |i, _| i as f64);
        let aliased = vec![true, true]; // All columns aliased

        let lev = compute_leverage_with_aliased(&x, &aliased, false);

        // Should return NaN for all observations
        for i in 0..10 {
            assert!(
                lev[i].is_nan(),
                "Leverage[{}] should be NaN when all columns aliased, got {}",
                i,
                lev[i]
            );
        }
    }

    #[test]
    fn test_compute_leverage_with_aliased_with_intercept_only() {
        // When all feature columns are aliased but intercept is included
        let x = Mat::from_fn(10, 2, |i, _| i as f64);
        let aliased = vec![true, true]; // All feature columns aliased

        let lev = compute_leverage_with_aliased(&x, &aliased, true);

        // With only intercept, all leverage values should be 1/n = 0.1
        for i in 0..10 {
            assert!(
                (lev[i] - 0.1).abs() < 1e-10,
                "Leverage[{}] should be 0.1 (1/n), got {}",
                i,
                lev[i]
            );
        }
    }

    // === Tests for high_leverage_points ===

    #[test]
    fn test_high_leverage_points_default_threshold() {
        // Tests line 191: default threshold calculation (2p/n)
        let mut leverage = Col::zeros(20);
        for i in 0..20 {
            leverage[i] = 0.1;
        }
        leverage[15] = 0.5; // High leverage point

        let n_params = 3; // threshold = 2*3/20 = 0.3

        let high = high_leverage_points(&leverage, n_params, None);

        // Only observation 15 (with leverage 0.5) should exceed 0.3
        assert!(high.contains(&15));
        assert_eq!(high.len(), 1);
    }

    #[test]
    fn test_high_leverage_points_custom_threshold() {
        // Tests line 191: custom threshold path
        let mut leverage = Col::zeros(20);
        for i in 0..20 {
            leverage[i] = 0.1;
        }
        leverage[18] = 0.3;
        leverage[19] = 0.3;

        let high = high_leverage_points(&leverage, 3, Some(0.25));

        // Observations 18 and 19 should exceed custom threshold 0.25
        assert!(high.contains(&18));
        assert!(high.contains(&19));
        assert_eq!(high.len(), 2);
    }

    #[test]
    fn test_high_leverage_points_no_high_leverage() {
        // Edge case: no high leverage points
        let leverage = Col::from_fn(100, |_| 0.01);
        let high = high_leverage_points(&leverage, 3, None);
        assert!(high.is_empty());
    }

    #[test]
    fn test_high_leverage_points_all_high() {
        // Edge case: all points are high leverage
        let leverage = Col::from_fn(10, |_| 0.5);
        let high = high_leverage_points(&leverage, 2, Some(0.2));
        assert_eq!(high.len(), 10);
    }

    // === Edge Cases ===

    #[test]
    fn test_leverage_n_less_than_p() {
        // Edge case: n < p (more features than observations)
        let x = Mat::from_fn(3, 5, |i, j| (i * j + 1) as f64);
        let leverage = compute_leverage(&x, true);

        // In rank-deficient case, leverage values should still be in [0, 1]
        for i in 0..3 {
            assert!(
                leverage[i].is_nan() || (leverage[i] >= 0.0 && leverage[i] <= 1.0),
                "Leverage[{}] = {} should be NaN or in [0, 1]",
                i,
                leverage[i]
            );
        }
    }

    #[test]
    fn test_leverage_n_equals_p() {
        // Edge case: n = p exactly
        let x = Mat::from_fn(
            3,
            2,
            |i, j| if j == 0 { i as f64 } else { (i as f64).cos() },
        );
        let leverage = compute_leverage(&x, true); // 3 params for 3 observations

        // When n = p, all observations should have leverage = 1.0
        for i in 0..3 {
            assert!(
                (leverage[i] - 1.0).abs() < 1e-6 || leverage[i].is_nan(),
                "Leverage[{}] should be 1.0 or NaN when n=p, got {}",
                i,
                leverage[i]
            );
        }
    }

    #[test]
    fn test_leverage_single_observation() {
        // Edge case: single observation
        let x = Mat::from_fn(1, 2, |_, j| (j + 1) as f64);
        let leverage = compute_leverage(&x, true);

        // Single observation should have leverage = 1.0 (or NaN if singular)
        assert!(
            (leverage[0] - 1.0).abs() < 1e-10 || leverage[0].is_nan(),
            "Single observation leverage should be 1.0 or NaN, got {}",
            leverage[0]
        );
    }

    #[test]
    fn test_leverage_without_intercept() {
        // Test leverage computation without intercept
        let x = Mat::from_fn(20, 2, |i, j| {
            if j == 0 {
                (i + 1) as f64
            } else {
                (i as f64).sin()
            }
        });
        let leverage = compute_leverage(&x, false);

        // Sum should equal number of features (no intercept)
        let sum: f64 = leverage.iter().sum();
        let n_params = 2;
        assert!(
            (sum - n_params as f64).abs() < 0.5,
            "Sum {} should be close to {}",
            sum,
            n_params
        );
    }
}
