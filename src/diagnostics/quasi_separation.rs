//! Quasi-separation detection for GLMs.
//!
//! Quasi-separation occurs when a predictor (or combination of predictors) perfectly
//! or nearly perfectly separates the response categories. This causes coefficient
//! estimates to diverge to infinity.
//!
//! # Background
//!
//! For binary responses (logistic regression):
//! - **Complete separation**: A linear combination of predictors perfectly predicts
//!   the response (all y=1 on one side of a hyperplane, all y=0 on the other).
//! - **Quasi-separation**: One or more observations are on the boundary.
//!
//! For count data (Poisson, Negative Binomial):
//! - A similar issue occurs when some predictor values have all-zero responses,
//!   causing the coefficient to drift toward negative infinity.
//!
//! # References
//!
//! - Albert, A. and Anderson, J.A. (1984). On the Existence of Maximum Likelihood
//!   Estimates in Logistic Regression Models.
//! - Heinze, G. and Schemper, M. (2002). A solution to the problem of separation
//!   in logistic regression.
//! - R package `detectseparation`: <https://cran.r-project.org/web/packages/detectseparation/>

use faer::{Col, Mat};

/// Result of quasi-separation detection.
#[derive(Debug, Clone, Default)]
pub struct SeparationCheck {
    /// True if separation was detected.
    pub has_separation: bool,
    /// Indices of predictors involved in separation (0-based, excludes intercept).
    pub separated_predictors: Vec<usize>,
    /// Description of the separation type for each predictor.
    pub separation_types: Vec<SeparationType>,
    /// Warning message for the user.
    pub warning_message: Option<String>,
}

/// Type of separation detected for a predictor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeparationType {
    /// No separation detected.
    None,
    /// Complete separation: predictor perfectly divides the classes.
    Complete,
    /// Quasi-separation: predictor nearly perfectly divides the classes.
    Quasi,
    /// All responses in one class for this predictor level (for binary predictors).
    MonotonicResponse,
}

/// Check for quasi-separation in binary response data.
///
/// This function examines each predictor to detect patterns that indicate
/// separation or quasi-separation.
///
/// # Arguments
///
/// * `x` - Design matrix (n_samples x n_features), excluding intercept
/// * `y` - Binary response vector (values in {0, 1})
///
/// # Returns
///
/// A `SeparationCheck` struct with detection results.
///
/// # Example
///
/// ```rust,ignore
/// use anofox_regression::diagnostics::check_binary_separation;
/// use faer::{Mat, Col};
///
/// let x = Mat::from_fn(10, 2, |i, j| (i + j) as f64);
/// let y = Col::from_fn(10, |i| if i < 5 { 0.0 } else { 1.0 });
///
/// let result = check_binary_separation(&x, &y);
/// if result.has_separation {
///     eprintln!("Warning: {}", result.warning_message.unwrap());
/// }
/// ```
pub fn check_binary_separation(x: &Mat<f64>, y: &Col<f64>) -> SeparationCheck {
    let n_samples = x.nrows();
    let n_features = x.ncols();

    if n_samples == 0 || n_features == 0 {
        return SeparationCheck::default();
    }

    let mut result = SeparationCheck::default();
    let mut warnings = Vec::new();

    // Convert y to Vec for easier manipulation
    let y_vec: Vec<f64> = (0..n_samples).map(|i| y[i]).collect();

    // Check each predictor
    for j in 0..n_features {
        let x_col: Vec<f64> = (0..n_samples).map(|i| x[(i, j)]).collect();
        let sep_type = check_predictor_separation(&x_col, &y_vec);

        result.separation_types.push(sep_type);

        if sep_type != SeparationType::None {
            result.has_separation = true;
            result.separated_predictors.push(j);

            let msg = match sep_type {
                SeparationType::Complete => {
                    format!("Feature {} shows complete separation", j)
                }
                SeparationType::Quasi => {
                    format!("Feature {} shows quasi-separation", j)
                }
                SeparationType::MonotonicResponse => {
                    format!(
                        "Feature {} has all responses in one class for some values",
                        j
                    )
                }
                SeparationType::None => unreachable!(),
            };
            warnings.push(msg);
        }
    }

    if !warnings.is_empty() {
        let combined = warnings.join("; ");
        result.warning_message = Some(format!(
            "Quasi-separation detected: {}. Coefficients may be unstable. \
             Consider using regularization (Ridge/Lasso) or removing problematic features.",
            combined
        ));
    }

    result
}

/// Check a single predictor for separation patterns.
fn check_predictor_separation(x: &[f64], y: &[f64]) -> SeparationType {
    let n = x.len();
    if n == 0 {
        return SeparationType::None;
    }

    // Count positive and negative class observations above/below threshold
    // We'll check if there's a threshold that separates the classes

    // Collect (x, y) pairs and sort by x
    let mut pairs: Vec<(f64, f64)> = x.iter().zip(y.iter()).map(|(&xi, &yi)| (xi, yi)).collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Check for monotonic separation
    // Count 0s and 1s in y for each unique x value
    let mut x_prev = pairs[0].0;
    let mut count_0 = 0;
    let mut count_1 = 0;
    let mut all_same_class_for_x = true;
    let mut found_only_zeros = false;
    let mut found_only_ones = false;
    let mut groups_with_multiple_obs = 0; // Track groups with repeated x values

    for &(xi, yi) in &pairs {
        if (xi - x_prev).abs() > 1e-10 {
            // New x value - check previous group
            if count_0 + count_1 > 1 {
                groups_with_multiple_obs += 1;
            }
            if count_0 > 0 && count_1 == 0 {
                found_only_zeros = true;
            }
            if count_1 > 0 && count_0 == 0 {
                found_only_ones = true;
            }
            if count_0 > 0 && count_1 > 0 {
                all_same_class_for_x = false;
            }
            count_0 = 0;
            count_1 = 0;
            x_prev = xi;
        }

        if yi < 0.5 {
            count_0 += 1;
        } else {
            count_1 += 1;
        }
    }

    // Check last group
    if count_0 + count_1 > 1 {
        groups_with_multiple_obs += 1;
    }
    if count_0 > 0 && count_1 == 0 {
        found_only_zeros = true;
    }
    if count_1 > 0 && count_0 == 0 {
        found_only_ones = true;
    }
    if count_0 > 0 && count_1 > 0 {
        all_same_class_for_x = false;
    }

    // Check for monotonic ordering
    let total_1s: usize = pairs.iter().filter(|(_, yi)| *yi >= 0.5).count();
    let total_0s = n - total_1s;

    if total_1s == 0 || total_0s == 0 {
        // All same class - no separation issue
        return SeparationType::None;
    }

    // Check if all 1s come after all 0s (or vice versa)
    let first_1_idx = pairs.iter().position(|(_, yi)| *yi >= 0.5);
    let last_0_idx = pairs.iter().rposition(|(_, yi)| *yi < 0.5);

    if let (Some(first_1), Some(last_0)) = (first_1_idx, last_0_idx) {
        if first_1 > last_0 {
            // All 1s come after all 0s - complete separation
            return SeparationType::Complete;
        }
        // Check for quasi-separation (almost complete)
        let overlap_count = pairs[..=last_0].iter().filter(|(_, yi)| *yi >= 0.5).count();
        if overlap_count <= 2 && (overlap_count as f64) / (total_1s as f64) < 0.1 {
            return SeparationType::Quasi;
        }
    }

    // Check for monotonic response (all same class for some x values)
    // Only applies when there are repeated x values (categorical-like predictors)
    if found_only_zeros && found_only_ones && all_same_class_for_x && groups_with_multiple_obs > 0 {
        return SeparationType::MonotonicResponse;
    }

    SeparationType::None
}

/// Check for sparsity issues in count data.
///
/// For Poisson/Negative Binomial regression, extreme sparsity (all-zero segments)
/// can cause coefficient divergence similar to separation in logistic regression.
///
/// # Arguments
///
/// * `x` - Design matrix (n_samples x n_features)
/// * `y` - Count response vector (non-negative values)
///
/// # Returns
///
/// A `SeparationCheck` struct indicating problematic predictors.
pub fn check_count_sparsity(x: &Mat<f64>, y: &Col<f64>) -> SeparationCheck {
    let n_samples = x.nrows();
    let n_features = x.ncols();

    if n_samples == 0 || n_features == 0 {
        return SeparationCheck::default();
    }

    let mut result = SeparationCheck::default();
    let mut warnings = Vec::new();

    // Convert y to Vec
    let y_vec: Vec<f64> = (0..n_samples).map(|i| y[i]).collect();
    let total_zeros = y_vec.iter().filter(|&&yi| yi < 1e-10).count();

    // If data is highly sparse (>90% zeros), check for segment sparsity
    if total_zeros as f64 / n_samples as f64 > 0.9 {
        for j in 0..n_features {
            let x_col: Vec<f64> = (0..n_samples).map(|i| x[(i, j)]).collect();

            // Check if this is a binary indicator (changepoint-style)
            let unique_vals: std::collections::HashSet<i64> =
                x_col.iter().map(|&v| (v * 1000.0) as i64).collect();

            if unique_vals.len() <= 3 {
                // Binary or near-binary predictor
                // Check if non-zero x values have all-zero y values
                let non_zero_x_indices: Vec<usize> = x_col
                    .iter()
                    .enumerate()
                    .filter(|(_, &xi)| xi.abs() > 1e-10)
                    .map(|(i, _)| i)
                    .collect();

                if !non_zero_x_indices.is_empty() {
                    let non_zero_y_count = non_zero_x_indices
                        .iter()
                        .filter(|&&i| y_vec[i] > 1e-10)
                        .count();

                    if non_zero_y_count == 0 {
                        result.has_separation = true;
                        result.separated_predictors.push(j);
                        result
                            .separation_types
                            .push(SeparationType::MonotonicResponse);
                        warnings.push(format!(
                            "Feature {} (binary indicator) has all-zero responses when active",
                            j
                        ));
                    }
                }
            }
        }
    }

    if !warnings.is_empty() {
        let combined = warnings.join("; ");
        result.warning_message = Some(format!(
            "Sparse data separation detected: {}. Coefficients may diverge to -infinity. \
             Consider using regularization or removing sparse indicator features.",
            combined
        ));
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complete_separation() {
        // Create data with complete separation: all 0s for x < 5, all 1s for x >= 5
        let x = Mat::from_fn(10, 1, |i, _| i as f64);
        let y = Col::from_fn(10, |i| if i < 5 { 0.0 } else { 1.0 });

        let result = check_binary_separation(&x, &y);
        assert!(result.has_separation);
        assert!(result.separated_predictors.contains(&0));
        assert_eq!(result.separation_types[0], SeparationType::Complete);
    }

    #[test]
    fn test_no_separation() {
        // Create data without separation: truly mixed classes with overlap
        // x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        // y = [0, 1, 0, 1, 1, 0, 1, 0, 1, 0]  - mixed pattern with no clear separation
        let x = Mat::from_fn(10, 1, |i, _| i as f64);
        let y_vals = [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let y = Col::from_fn(10, |i| y_vals[i]);

        let result = check_binary_separation(&x, &y);
        assert!(!result.has_separation);
    }

    #[test]
    fn test_count_sparsity() {
        // Create sparse data with all-zero segment
        let x = Mat::from_fn(20, 1, |i, _| if i < 10 { 0.0 } else { 1.0 });
        let y = Col::from_fn(20, |i| if i < 10 { (i as f64) * 0.01 } else { 0.0 });

        // First 10 have small non-zero values, last 10 (where x=1) have zeros
        // Need high enough sparsity
        let result = check_count_sparsity(&x, &y);
        // This test verifies the function runs without error
        assert!(!result.has_separation || result.separated_predictors.is_empty());
    }

    #[test]
    fn test_quasi_separation() {
        // Quasi-separation: almost complete separation with 1-2 overlap points
        // x sorted: 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19
        // Most 0s before most 1s, but with 1 or 2 early 1s (<=10% overlap)
        let n = 30;
        let x = Mat::from_fn(n, 1, |i, _| i as f64);
        // y: 28 0s followed by 2 1s, except one 1 at position 1
        // Positions 0-27 are 0s except position 1 which is 1
        // Positions 28-29 are 1s
        // This gives: total_1s = 3, overlap_count = 1, overlap_count/total_1s = 0.33 > 0.1
        // Need fewer overlaps relative to total 1s

        // Let's try: 18 0s, then 12 1s (total), with 1 early 1 at position 0
        // total_1s = 12, overlap_count = 1, 1/12 = 0.083 < 0.1
        let y = Col::from_fn(n, |i| {
            if i == 0 {
                1.0 // one early 1 (overlap)
            } else if i < 18 {
                0.0 // remaining 0s
            } else {
                1.0 // 12 1s at the end
            }
        });

        let result = check_binary_separation(&x, &y);
        assert!(result.has_separation);
        assert_eq!(result.separation_types[0], SeparationType::Quasi);
        assert!(result.warning_message.is_some());
        let msg = result.warning_message.unwrap();
        assert!(msg.contains("quasi-separation"));
    }

    #[test]
    fn test_monotonic_response() {
        // Monotonic response: categorical predictor where each level has all same class
        // x has repeated values (like a categorical), and each unique x value
        // has all responses in one class
        // x = [0, 0, 0, 0, 1, 1, 1, 1]
        // y = [0, 0, 0, 0, 1, 1, 1, 1]
        // But this is also complete separation...

        // For MonotonicResponse without complete separation, we need:
        // - found_only_zeros = true (some x level has all 0s)
        // - found_only_ones = true (some x level has all 1s)
        // - all_same_class_for_x = true (no x level has mixed)
        // - groups_with_multiple_obs > 0 (at least one repeated x value)
        // - Not complete separation (first_1 not > last_0)

        // x = [0, 0, 1, 1, 2, 2]
        // y = [0, 0, 1, 1, 0, 0]
        // Sorted: x=[0,0,1,1,2,2], y=[0,0,1,1,0,0]
        // Groups: x=0 has all 0s, x=1 has all 1s, x=2 has all 0s
        // found_only_zeros = true, found_only_ones = true, all_same_class_for_x = true
        // first_1 = 2, last_0 = 5, so first_1 (2) not > last_0 (5) - not complete
        // groups_with_multiple_obs = 3 (all groups have 2 obs)
        let x = Mat::from_fn(6, 1, |i, _| (i / 2) as f64);
        let y_vals = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0];
        let y = Col::from_fn(6, |i| y_vals[i]);

        let result = check_binary_separation(&x, &y);
        assert!(result.has_separation);
        assert_eq!(
            result.separation_types[0],
            SeparationType::MonotonicResponse
        );
        assert!(result.warning_message.is_some());
        let msg = result.warning_message.unwrap();
        assert!(msg.contains("all responses in one class"));
    }

    #[test]
    fn test_count_sparsity_with_separation() {
        // Create highly sparse data (>90% zeros) with binary indicator
        // where x=1 always has y=0 (separation)
        let n = 100;
        // x is binary: 0 for first 50, 1 for last 50
        let x = Mat::from_fn(n, 1, |i, _| if i < 50 { 0.0 } else { 1.0 });
        // y: first 5 observations have small counts, rest are all zeros
        // This gives >95% zeros
        let y = Col::from_fn(n, |i| if i < 5 { 1.0 } else { 0.0 });

        let result = check_count_sparsity(&x, &y);
        // When x=1 (indices 50-99), all y values are 0
        // This should trigger the separation detection
        assert!(result.has_separation);
        assert!(result.separated_predictors.contains(&0));
        assert_eq!(
            result.separation_types[0],
            SeparationType::MonotonicResponse
        );
        assert!(result.warning_message.is_some());
        let msg = result.warning_message.unwrap();
        assert!(msg.contains("all-zero responses when active"));
    }

    #[test]
    fn test_empty_matrix_binary() {
        let x = Mat::<f64>::zeros(0, 0);
        let y = Col::<f64>::zeros(0);
        let result = check_binary_separation(&x, &y);
        assert!(!result.has_separation);
        assert!(result.separated_predictors.is_empty());
    }

    #[test]
    fn test_empty_matrix_count() {
        let x = Mat::<f64>::zeros(0, 0);
        let y = Col::<f64>::zeros(0);
        let result = check_count_sparsity(&x, &y);
        assert!(!result.has_separation);
        assert!(result.separated_predictors.is_empty());
    }

    #[test]
    fn test_all_same_class() {
        // All y values are the same class - no separation possible
        let x = Mat::from_fn(10, 1, |i, _| i as f64);
        let y = Col::from_fn(10, |_| 1.0); // all 1s

        let result = check_binary_separation(&x, &y);
        assert!(!result.has_separation);
    }

    #[test]
    fn test_multiple_predictors() {
        // Test with multiple predictors, only one showing separation
        // Feature 0: no separation (mixed)
        // Feature 1: complete separation
        let x = Mat::from_fn(10, 2, |i, j| {
            if j == 0 {
                (i % 3) as f64 // cycles through 0,1,2
            } else {
                i as f64 // monotonic
            }
        });
        let y = Col::from_fn(10, |i| if i < 5 { 0.0 } else { 1.0 });

        let result = check_binary_separation(&x, &y);
        assert!(result.has_separation);
        // Feature 1 should show complete separation
        assert!(result.separated_predictors.contains(&1));
        assert_eq!(result.separation_types[1], SeparationType::Complete);
    }

    #[test]
    fn test_count_sparsity_not_sparse_enough() {
        // Data that's not sparse enough (< 90% zeros)
        let n = 10;
        let x = Mat::from_fn(n, 1, |i, _| if i < 5 { 0.0 } else { 1.0 });
        let y = Col::from_fn(n, |i| if i < 3 { 1.0 } else { 0.0 }); // 70% zeros

        let result = check_count_sparsity(&x, &y);
        // Should not detect separation because data is not sparse enough
        assert!(!result.has_separation);
    }

    #[test]
    fn test_count_sparsity_non_binary_predictor() {
        // Highly sparse data but predictor has >3 unique values (not binary-like)
        let n = 100;
        let x = Mat::from_fn(n, 1, |i, _| i as f64); // 100 unique values
        let y = Col::from_fn(n, |i| if i < 5 { 1.0 } else { 0.0 }); // 95% zeros

        let result = check_count_sparsity(&x, &y);
        // Should not flag this since x is not binary-like
        assert!(!result.has_separation);
    }

    #[test]
    fn test_separation_type_clone() {
        let sep = SeparationType::Complete;
        let cloned = sep.clone();
        assert_eq!(sep, cloned);
    }

    #[test]
    fn test_separation_check_default() {
        let check = SeparationCheck::default();
        assert!(!check.has_separation);
        assert!(check.separated_predictors.is_empty());
        assert!(check.separation_types.is_empty());
        assert!(check.warning_message.is_none());
    }

    #[test]
    fn test_separation_check_clone() {
        let mut check = SeparationCheck::default();
        check.has_separation = true;
        check.separated_predictors.push(0);
        check.separation_types.push(SeparationType::Complete);
        check.warning_message = Some("test".to_string());

        let cloned = check.clone();
        assert!(cloned.has_separation);
        assert_eq!(cloned.separated_predictors, vec![0]);
        assert_eq!(cloned.separation_types, vec![SeparationType::Complete]);
        assert_eq!(cloned.warning_message, Some("test".to_string()));
    }
}
