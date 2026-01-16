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
#[derive(Debug, Clone)]
pub struct ConditionDiagnostic {
    /// Condition number of the design matrix X.
    pub condition_number: f64,
    /// Condition number of X'X (equals κ² of X).
    pub condition_number_xtx: f64,
    /// Singular values of X (sorted descending).
    pub singular_values: Vec<f64>,
    /// Condition indices: max(σ) / σ_j for each singular value.
    pub condition_indices: Vec<f64>,
    /// Severity classification.
    pub severity: ConditionSeverity,
    /// Warning message if condition number is problematic.
    pub warning: Option<String>,
}

/// Compute the condition number of a design matrix.
///
/// The condition number κ(X) = σ_max / σ_min where σ are the singular values.
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

    let svd = match x_design.svd() {
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

    let svd = match x_design.svd() {
        Ok(svd) => svd,
        Err(_) => {
            return ConditionDiagnostic {
                condition_number: f64::INFINITY,
                condition_number_xtx: f64::INFINITY,
                singular_values: vec![],
                condition_indices: vec![],
                severity: ConditionSeverity::Severe,
                warning: Some("SVD computation failed".to_string()),
            };
        }
    };
    let s = svd.S();
    let s_col = s.column_vector();

    let n_params = s_col.nrows();
    if n_params == 0 {
        return ConditionDiagnostic {
            condition_number: f64::INFINITY,
            condition_number_xtx: f64::INFINITY,
            singular_values: vec![],
            condition_indices: vec![],
            severity: ConditionSeverity::Severe,
            warning: Some("Empty design matrix".to_string()),
        };
    }

    // Extract and sort singular values (descending)
    let mut singular_values: Vec<f64> = (0..n_params).map(|i| s_col[i]).collect();
    singular_values.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    let s_max = singular_values[0];
    let s_min = *singular_values.iter().rfind(|&&v| v > 0.0).unwrap_or(&0.0);

    let condition_number = if s_min > 0.0 {
        s_max / s_min
    } else {
        f64::INFINITY
    };

    let condition_number_xtx = condition_number * condition_number;

    // Compute condition indices: max(σ) / σ_j
    let condition_indices: Vec<f64> = singular_values
        .iter()
        .map(|&si| if si > 0.0 { s_max / si } else { f64::INFINITY })
        .collect();

    // Classify severity
    let severity = classify_condition_number(condition_number);

    // Generate warning if needed
    let warning = match severity {
        ConditionSeverity::WellConditioned => None,
        ConditionSeverity::Moderate => Some(format!(
            "Moderate collinearity detected (κ = {:.1}). Results should be reliable but verify.",
            condition_number
        )),
        ConditionSeverity::High => Some(format!(
            "High collinearity detected (κ = {:.1}). Coefficients may be unstable. \
             Consider removing collinear features or using regularization.",
            condition_number
        )),
        ConditionSeverity::Severe => Some(format!(
            "Severe collinearity detected (κ = {:.1}). Coefficients are likely unreliable. \
             Use regularization (Ridge/Lasso) or remove highly correlated features.",
            condition_number
        )),
    };

    ConditionDiagnostic {
        condition_number,
        condition_number_xtx,
        singular_values,
        condition_indices,
        severity,
        warning,
    }
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

/// Find features contributing most to collinearity.
///
/// Uses the variance-decomposition proportions approach from
/// Belsley, Kuh, and Welsch (1980).
///
/// # Arguments
///
/// * `x` - Design matrix (n_samples x n_features)
/// * `with_intercept` - If true, prepend an intercept column
///
/// # Returns
///
/// A matrix of variance-decomposition proportions (n_features x n_features).
/// High proportions (>0.5) on rows with high condition indices indicate
/// which features are involved in collinearity.
pub fn variance_decomposition_proportions(x: &Mat<f64>, with_intercept: bool) -> Mat<f64> {
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

    let svd = match x_design.svd() {
        Ok(svd) => svd,
        Err(_) => return Mat::zeros(0, 0),
    };
    let v = svd.V().to_owned();
    let s = svd.S();
    let s_col = s.column_vector();

    let n_params = s_col.nrows();
    if n_params == 0 {
        return Mat::zeros(0, 0);
    }

    // Compute φ_jk = V_jk² / σ_k² for each variable j and component k
    // Then normalize: π_jk = φ_jk / Σ_k' φ_jk'

    let mut phi = Mat::zeros(n_params, n_params);
    for j in 0..n_params {
        for k in 0..n_params {
            let s_k = s_col[k];
            if s_k > 0.0 {
                phi[(j, k)] = v[(j, k)] * v[(j, k)] / (s_k * s_k);
            }
        }
    }

    // Normalize rows
    let mut proportions = Mat::zeros(n_params, n_params);
    for j in 0..n_params {
        let row_sum: f64 = (0..n_params).map(|k| phi[(j, k)]).sum();
        if row_sum > 0.0 {
            for k in 0..n_params {
                proportions[(j, k)] = phi[(j, k)] / row_sum;
            }
        }
    }

    proportions
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
