//! Condition number diagnostics for regression matrices.
//!
//! The condition number is a measure of how sensitive a matrix is to numerical
//! errors. A high condition number indicates that small changes in the input
//! can lead to large changes in the output, making the regression numerically
//! unstable.
//!
//! # Interpretation
//!
//! - κ < 30: Well-conditioned, stable
//! - 30 ≤ κ < 100: Moderate collinearity
//! - 100 ≤ κ < 1000: High collinearity, potential instability
//! - κ ≥ 1000: Severe collinearity, numerical instability likely
//!
//! # References
//!
//! - Belsley, D.A., Kuh, E. and Welsch, R.E. (1980). Regression Diagnostics.
//! - R's `kappa()` function in base package

use faer::Mat;

/// Condition number severity classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConditionSeverity {
    /// κ < 30: Well-conditioned
    WellConditioned,
    /// 30 ≤ κ < 100: Moderate collinearity
    Moderate,
    /// 100 ≤ κ < 1000: High collinearity
    High,
    /// κ ≥ 1000: Severe collinearity
    Severe,
}

impl ConditionSeverity {
    /// Get a human-readable description of the severity.
    pub fn description(&self) -> &'static str {
        match self {
            Self::WellConditioned => "Well-conditioned: numerically stable",
            Self::Moderate => "Moderate collinearity: some instability possible",
            Self::High => "High collinearity: numerical instability likely",
            Self::Severe => "Severe collinearity: coefficients may be unreliable",
        }
    }
}

/// Result of condition number analysis.
///
/// `condition_number` is the plain κ of the design (R's `kappa(X, exact =
/// TRUE)`), which depends on the units of the columns. The collinearity
/// diagnostics of Belsley, Kuh & Welsch — `singular_values`,
/// `condition_indices`, `scaled_condition_number` and the `severity`
/// classification — use the design with every column (including the
/// intercept) scaled to unit length, as `olsrr::ols_eigen_cindex` and
/// `perturb::colldiag`, so they do not depend on units.
#[derive(Debug, Clone)]
pub struct ConditionDiagnostic {
    /// Condition number of the (unscaled) design matrix X, `kappa(X, exact = TRUE)`.
    pub condition_number: f64,
    /// Condition number of X'X (equals κ² of X).
    pub condition_number_xtx: f64,
    /// Singular values of the unit-column-length design (sorted descending).
    pub singular_values: Vec<f64>,
    /// BKW condition indices `σ_max / σ_k` of the unit-column-length design.
    pub condition_indices: Vec<f64>,
    /// Largest BKW condition index (κ of the unit-column-length design).
    pub scaled_condition_number: f64,
    /// Severity classification of `scaled_condition_number` (BKW cut-offs).
    pub severity: ConditionSeverity,
    /// Warning message if condition number is problematic.
    pub warning: Option<String>,
}

/// Compute the condition number of a design matrix.
///
/// The condition number κ(X) = σ_max / σ_min where σ are the singular values
/// of the unscaled design, R's `kappa(X, exact = TRUE)`. It depends on the
/// units of the columns; for the scale-free Belsley–Kuh–Welsch condition
/// indices use [`condition_diagnostic`].
///
/// # Arguments
///
/// * `x` - Design matrix (n_samples x n_features)
/// * `with_intercept` - If true, prepend an intercept column
///
/// # Returns
///
/// The condition number (f64::INFINITY if matrix is rank deficient)
///
/// # Example
///
/// ```rust,ignore
/// use anofox_regression::diagnostics::condition_number;
/// use faer::Mat;
///
/// let x = Mat::from_fn(100, 3, |i, j| (i + j) as f64);
/// let cond = condition_number(&x, true);
/// println!("Condition number: {:.2}", cond);
/// ```
pub fn condition_number(x: &Mat<f64>, with_intercept: bool) -> f64 {
    let x_design = if with_intercept {
        let n = x.nrows();
        let p = x.ncols();
        let mut x_aug = Mat::zeros(n, p + 1);
        for i in 0..n {
            x_aug[(i, 0)] = 1.0;
            for j in 0..p {
                x_aug[(i, j + 1)] = x[(i, j)];
            }
        }
        x_aug
    } else {
        x.clone()
    };

    let svd = match x_design.thin_svd() {
        Ok(svd) => svd,
        Err(_) => return f64::INFINITY,
    };
    let s = svd.S();
    let s_col = s.column_vector();

    let n_params = s_col.nrows();
    if n_params == 0 {
        return f64::INFINITY;
    }

    let mut s_max = f64::NEG_INFINITY;
    let mut s_min = f64::INFINITY;

    for i in 0..n_params {
        let si = s_col[i];
        if si > s_max {
            s_max = si;
        }
        if si > 0.0 && si < s_min {
            s_min = si;
        }
    }

    if s_min <= 0.0 {
        f64::INFINITY
    } else {
        s_max / s_min
    }
}

/// Compute comprehensive condition number diagnostics.
///
/// This function provides detailed analysis including singular values,
/// condition indices, and severity classification.
///
/// # Arguments
///
/// * `x` - Design matrix (n_samples x n_features)
/// * `with_intercept` - If true, prepend an intercept column
///
/// # Returns
///
/// A `ConditionDiagnostic` struct with detailed analysis.
///
/// # Example
///
/// ```rust,ignore
/// use anofox_regression::diagnostics::condition_diagnostic;
/// use faer::Mat;
///
/// let x = Mat::from_fn(100, 3, |i, j| (i + j) as f64);
/// let diag = condition_diagnostic(&x, true);
///
/// println!("Condition number: {:.2}", diag.condition_number);
/// println!("Severity: {}", diag.severity.description());
/// if let Some(warning) = &diag.warning {
///     eprintln!("Warning: {}", warning);
/// }
/// ```
pub fn condition_diagnostic(x: &Mat<f64>, with_intercept: bool) -> ConditionDiagnostic {
    let x_design = design(x, with_intercept);
    let condition_number = condition_number(x, with_intercept);
    let Some((singular_values, _)) = svd_desc(&scale_unit_length(&x_design)) else {
        return ConditionDiagnostic {
            condition_number,
            condition_number_xtx: condition_number * condition_number,
            singular_values: vec![],
            condition_indices: vec![],
            scaled_condition_number: f64::INFINITY,
            severity: ConditionSeverity::Severe,
            warning: Some("SVD computation failed or empty design matrix".to_string()),
        };
    };

    let s_max = singular_values[0];
    let condition_indices: Vec<f64> = singular_values
        .iter()
        .map(|&si| if si > 0.0 { s_max / si } else { f64::INFINITY })
        .collect();
    let scaled_condition_number = *condition_indices.last().unwrap_or(&f64::INFINITY);

    // Classify severity (Belsley's cut-offs apply to the scaled design)
    let severity = classify_condition_number(scaled_condition_number);
    let k = scaled_condition_number;

    // Generate warning if needed
    let warning = match severity {
        ConditionSeverity::WellConditioned => None,
        ConditionSeverity::Moderate => Some(format!(
            "Moderate collinearity detected (condition index {k:.1}). Results should be reliable but verify."
        )),
        ConditionSeverity::High => Some(format!(
            "High collinearity detected (condition index {k:.1}). Coefficients may be unstable. \
             Consider removing collinear features or using regularization."
        )),
        ConditionSeverity::Severe => Some(format!(
            "Severe collinearity detected (condition index {k:.1}). Coefficients are likely unreliable. \
             Use regularization (Ridge/Lasso) or remove highly correlated features."
        )),
    };

    ConditionDiagnostic {
        condition_number,
        condition_number_xtx: condition_number * condition_number,
        singular_values,
        condition_indices,
        scaled_condition_number,
        severity,
        warning,
    }
}

/// The design `[1, X]` (with intercept) or `X`.
fn design(x: &Mat<f64>, with_intercept: bool) -> Mat<f64> {
    if with_intercept {
        Mat::from_fn(x.nrows(), x.ncols() + 1, |i, j| {
            if j == 0 {
                1.0
            } else {
                x[(i, j - 1)]
            }
        })
    } else {
        x.clone()
    }
}

/// Scale every column to unit Euclidean length (BKW); zero columns stay zero.
fn scale_unit_length(x: &Mat<f64>) -> Mat<f64> {
    let norms: Vec<f64> = (0..x.ncols())
        .map(|j| {
            (0..x.nrows())
                .map(|i| x[(i, j)].powi(2))
                .sum::<f64>()
                .sqrt()
        })
        .collect();
    Mat::from_fn(x.nrows(), x.ncols(), |i, j| {
        if norms[j] > 0.0 {
            x[(i, j)] / norms[j]
        } else {
            0.0
        }
    })
}

/// Singular values (descending) and the matching right singular vectors
/// (columns of V in the same order).
fn svd_desc(x: &Mat<f64>) -> Option<(Vec<f64>, Mat<f64>)> {
    if x.ncols() == 0 || x.nrows() == 0 {
        return None;
    }
    let svd = x.thin_svd().ok()?;
    let s = svd.S().column_vector();
    let v = svd.V();
    let mut order: Vec<usize> = (0..s.nrows()).collect();
    order.sort_by(|&a, &b| s[b].total_cmp(&s[a]));
    let values = order.iter().map(|&k| s[k]).collect();
    let vecs = Mat::from_fn(v.nrows(), order.len(), |j, c| v[(j, order[c])]);
    Some((values, vecs))
}

/// Classify the condition number into severity levels.
///
/// # Arguments
///
/// * `cond` - The condition number
///
/// # Returns
///
/// A `ConditionSeverity` classification.
pub fn classify_condition_number(cond: f64) -> ConditionSeverity {
    if cond < 30.0 {
        ConditionSeverity::WellConditioned
    } else if cond < 100.0 {
        ConditionSeverity::Moderate
    } else if cond < 1000.0 {
        ConditionSeverity::High
    } else {
        ConditionSeverity::Severe
    }
}

/// Variance-decomposition proportions of Belsley, Kuh & Welsch (1980).
///
/// The design (with the intercept column if `with_intercept`) is scaled to
/// unit column length; with its SVD `X = U D Vᵀ`, `φ_jk = v_jk² / d_k²` and
/// the proportion of the variance of coefficient `j` associated with
/// component `k` is `π_jk = φ_jk / Σ_k φ_jk`.
///
/// # Returns
///
/// A `p × p` matrix with `π_jk` in row `j` (coefficient, intercept first) and
/// column `k` (component, in the order of decreasing singular value, i.e.
/// increasing condition index) — the transpose of the table printed by
/// `olsrr::ols_eigen_cindex` / `perturb::colldiag`. Proportions above 0.5 for
/// two or more coefficients on a component with a high condition index point
/// to the variables involved in a near dependency.
pub fn variance_decomposition_proportions(x: &Mat<f64>, with_intercept: bool) -> Mat<f64> {
    let Some((d, v)) = svd_desc(&scale_unit_length(&design(x, with_intercept))) else {
        return Mat::zeros(0, 0);
    };
    let p = d.len();
    let phi = Mat::from_fn(v.nrows(), p, |j, k| {
        if d[k] > 0.0 {
            v[(j, k)].powi(2) / d[k].powi(2)
        } else {
            0.0
        }
    });
    Mat::from_fn(phi.nrows(), p, |j, k| {
        let row_sum: f64 = (0..p).map(|c| phi[(j, c)]).sum();
        if row_sum > 0.0 {
            phi[(j, k)] / row_sum
        } else {
            0.0
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_condition_diagnostic_empty_design() {
        let d = condition_diagnostic(&Mat::<f64>::zeros(5, 0), false);
        assert!(d.condition_indices.is_empty());
        assert!(d.scaled_condition_number.is_infinite());
        assert_eq!(d.severity, ConditionSeverity::Severe);
        assert!(d.warning.is_some());
    }

    #[test]
    fn test_condition_diagnostic_zero_column() {
        // A zero column stays zero under unit-length scaling: singular design.
        let x = Mat::from_fn(6, 2, |i, j| if j == 0 { i as f64 } else { 0.0 });
        let d = condition_diagnostic(&x, true);
        assert!(d.scaled_condition_number.is_infinite());
        assert_eq!(d.severity, ConditionSeverity::Severe);
    }

    #[test]
    fn test_condition_number_orthogonal() {
        // Orthogonal columns should have condition number close to 1
        let n = 100;
        let x = Mat::from_fn(
            n,
            2,
            |i, j| if j == 0 { i as f64 } else { 100.0 - i as f64 },
        );
        let cond = condition_number(&x, false);
        // Not exactly orthogonal, but should be well-conditioned
        assert!(cond < 10.0, "Expected low condition number, got {}", cond);
    }

    #[test]
    fn test_condition_number_collinear() {
        // Nearly collinear columns should have high condition number
        let n = 100;
        let x = Mat::from_fn(n, 2, |i, j| {
            if j == 0 {
                i as f64
            } else {
                i as f64 + 0.001 // Almost identical to first column
            }
        });
        let cond = condition_number(&x, false);
        assert!(
            cond > 100.0,
            "Expected high condition number for collinear data, got {}",
            cond
        );
    }

    #[test]
    fn test_condition_diagnostic() {
        let n = 100;
        let x = Mat::from_fn(n, 2, |i, j| (i + j) as f64);
        let diag = condition_diagnostic(&x, true);

        assert!(diag.condition_number > 0.0);
        assert_eq!(diag.singular_values.len(), 3); // 2 features + intercept
        assert_eq!(diag.condition_indices.len(), 3);

        // Condition indices should be >= 1.0 and first should be 1.0
        assert!((diag.condition_indices[0] - 1.0).abs() < 1e-10);
        for ci in &diag.condition_indices {
            assert!(*ci >= 1.0);
        }
    }

    #[test]
    fn test_classify_condition_number() {
        assert_eq!(
            classify_condition_number(10.0),
            ConditionSeverity::WellConditioned
        );
        assert_eq!(classify_condition_number(50.0), ConditionSeverity::Moderate);
        assert_eq!(classify_condition_number(500.0), ConditionSeverity::High);
        assert_eq!(classify_condition_number(5000.0), ConditionSeverity::Severe);
    }

    #[test]
    fn test_severity_descriptions() {
        assert_eq!(
            ConditionSeverity::WellConditioned.description(),
            "Well-conditioned: numerically stable"
        );
        assert_eq!(
            ConditionSeverity::Moderate.description(),
            "Moderate collinearity: some instability possible"
        );
        assert_eq!(
            ConditionSeverity::High.description(),
            "High collinearity: numerical instability likely"
        );
        assert_eq!(
            ConditionSeverity::Severe.description(),
            "Severe collinearity: coefficients may be unreliable"
        );
    }

    #[test]
    fn test_condition_number_empty_matrix() {
        let x = Mat::<f64>::zeros(0, 0);
        let cond = condition_number(&x, false);
        assert!(cond.is_infinite());
    }

    #[test]
    fn test_condition_number_with_intercept() {
        let n = 50;
        let x = Mat::from_fn(n, 2, |i, j| (i + j) as f64 / 10.0);
        let cond_no_int = condition_number(&x, false);
        let cond_with_int = condition_number(&x, true);

        // Both should be finite
        assert!(cond_no_int.is_finite());
        assert!(cond_with_int.is_finite());
    }

    #[test]
    fn test_condition_diagnostic_warnings() {
        // Create data with moderate collinearity
        let n = 100;
        let x = Mat::from_fn(n, 2, |i, j| {
            if j == 0 {
                i as f64
            } else {
                i as f64 * 1.5 + (i % 5) as f64 // Correlated but not perfect
            }
        });

        let diag = condition_diagnostic(&x, true);
        // Check structure
        assert!(diag.condition_number > 0.0);
        assert_eq!(diag.condition_number_xtx, diag.condition_number.powi(2));
    }

    #[test]
    fn test_condition_diagnostic_severe_collinearity() {
        // Create nearly collinear data
        let n = 100;
        let x = Mat::from_fn(n, 2, |i, j| {
            if j == 0 {
                i as f64
            } else {
                i as f64 + 0.0001 // Almost identical
            }
        });

        let diag = condition_diagnostic(&x, false);
        assert_eq!(diag.severity, ConditionSeverity::Severe);
        assert!(diag.warning.is_some());
        let warning = diag.warning.unwrap();
        assert!(warning.contains("Severe collinearity"));
    }

    #[test]
    fn test_condition_diagnostic_moderate_collinearity() {
        // Create data with moderate condition number (30-100)
        let n = 100;
        let x = Mat::from_fn(n, 2, |i, j| {
            if j == 0 {
                (i as f64).sin()
            } else {
                (i as f64 * 0.1).cos() * 10.0
            }
        });

        let diag = condition_diagnostic(&x, true);
        // Just verify the diagnostic runs and produces valid output
        assert!(diag.condition_number > 0.0);
        assert!(!diag.singular_values.is_empty());
    }

    #[test]
    fn test_condition_diagnostic_high_collinearity() {
        // Create data with high condition number (100-1000)
        let n = 100;
        let x = Mat::from_fn(n, 2, |i, j| {
            if j == 0 {
                i as f64
            } else {
                i as f64 + 0.1 * ((i % 10) as f64)
            }
        });

        let diag = condition_diagnostic(&x, false);
        // Verify warning is generated for non-well-conditioned cases
        if diag.severity != ConditionSeverity::WellConditioned {
            assert!(diag.warning.is_some());
        }
    }

    #[test]
    fn test_variance_decomposition_proportions() {
        // Basic test for variance decomposition
        let n = 50;
        let x = Mat::from_fn(n, 2, |i, j| (i + j) as f64);

        let vdp = variance_decomposition_proportions(&x, true);

        // Should return a 3x3 matrix (2 features + intercept)
        assert_eq!(vdp.nrows(), 3);
        assert_eq!(vdp.ncols(), 3);

        // Each row should sum approximately to 1 (proportions)
        for j in 0..vdp.nrows() {
            let row_sum: f64 = (0..vdp.ncols()).map(|k| vdp[(j, k)]).sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-6,
                "Row {} sum = {}, expected 1.0",
                j,
                row_sum
            );
        }
    }

    #[test]
    fn test_variance_decomposition_proportions_empty() {
        let x = Mat::<f64>::zeros(0, 0);
        let vdp = variance_decomposition_proportions(&x, false);
        assert_eq!(vdp.nrows(), 0);
        assert_eq!(vdp.ncols(), 0);
    }

    #[test]
    fn test_variance_decomposition_proportions_no_intercept() {
        let n = 50;
        let x = Mat::from_fn(n, 2, |i, j| (i * (j + 1)) as f64);

        let vdp = variance_decomposition_proportions(&x, false);

        // Should return a 2x2 matrix (2 features, no intercept)
        assert_eq!(vdp.nrows(), 2);
        assert_eq!(vdp.ncols(), 2);
    }

    #[test]
    fn test_condition_number_rank_deficient() {
        // Create a rank-deficient matrix (duplicate columns)
        let n = 50;
        let x = Mat::from_fn(n, 3, |i, j| {
            if j == 2 {
                // Third column is same as first
                i as f64
            } else {
                i as f64 + j as f64
            }
        });

        let diag = condition_diagnostic(&x, false);
        // Should detect severe collinearity or infinity
        assert!(
            diag.condition_number > 1000.0 || diag.condition_number.is_infinite(),
            "Expected high condition number for rank-deficient matrix"
        );
    }

    #[test]
    fn test_condition_indices_ordering() {
        let n = 100;
        let x = Mat::from_fn(n, 2, |i, j| (i + j) as f64);
        let diag = condition_diagnostic(&x, true);

        // First condition index should be 1.0 (max/max)
        assert!((diag.condition_indices[0] - 1.0).abs() < 1e-10);

        // Condition indices should be non-decreasing (sorted by singular values descending)
        for i in 1..diag.condition_indices.len() {
            assert!(
                diag.condition_indices[i] >= diag.condition_indices[i - 1],
                "Condition indices should be non-decreasing"
            );
        }
    }

    #[test]
    fn test_singular_values_sorted() {
        let n = 100;
        let x = Mat::from_fn(n, 3, |i, j| (i * j + 1) as f64);
        let diag = condition_diagnostic(&x, false);

        // Singular values should be sorted descending
        for i in 1..diag.singular_values.len() {
            assert!(
                diag.singular_values[i] <= diag.singular_values[i - 1],
                "Singular values should be sorted descending"
            );
        }
    }

    #[test]
    fn test_condition_diagnostic_boundary_values() {
        // Test boundary values for classification
        // κ = 30 should be Moderate
        // κ = 100 should be High
        // κ = 1000 should be Severe

        assert_eq!(
            classify_condition_number(29.9),
            ConditionSeverity::WellConditioned
        );
        assert_eq!(classify_condition_number(30.0), ConditionSeverity::Moderate);
        assert_eq!(classify_condition_number(99.9), ConditionSeverity::Moderate);
        assert_eq!(classify_condition_number(100.0), ConditionSeverity::High);
        assert_eq!(classify_condition_number(999.9), ConditionSeverity::High);
        assert_eq!(classify_condition_number(1000.0), ConditionSeverity::Severe);
    }
}
