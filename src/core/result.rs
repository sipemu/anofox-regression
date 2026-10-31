//! Regression result structures.

use super::na_action::NaInfo;
use faer::Col;

/// Complete result from a regression fit.
///
/// Contains coefficients, fit statistics, and optionally inference statistics
/// (standard errors, t-statistics, p-values, confidence intervals).
#[derive(Debug, Clone)]
pub struct RegressionResult {
    // ========== Core Results ==========
    /// Estimated coefficients (excluding intercept).
    /// Aliased (collinear) coefficients are set to NaN.
    pub coefficients: Col<f64>,

    /// Intercept term (if model was fit with intercept).
    pub intercept: Option<f64>,

    /// Residuals (y - fitted_values).
    pub residuals: Col<f64>,

    /// Fitted values (predictions on training data).
    pub fitted_values: Col<f64>,

    // ========== Rank Information ==========
    /// Numerical rank of the design matrix.
    pub rank: usize,

    /// Number of parameters (including intercept if present).
    pub n_parameters: usize,

    /// Number of observations.
    pub n_observations: usize,

    /// Indicates which coefficients are aliased (perfectly collinear).
    pub aliased: Vec<bool>,

    /// Column permutation from QR decomposition (if used).
    pub column_permutation: Option<Vec<usize>>,

    /// Tolerance used for rank determination.
    pub rank_tolerance: f64,

    // ========== Fit Statistics ==========
    /// Coefficient of determination (R²).
    pub r_squared: f64,

    /// Adjusted R².
    pub adj_r_squared: f64,

    /// Root mean squared error.
    pub rmse: f64,

    /// Mean squared error.
    pub mse: f64,

    /// F-statistic for overall model significance.
    pub f_statistic: f64,

    /// P-value for F-statistic.
    pub f_pvalue: f64,

    // ========== Information Criteria ==========
    /// Akaike Information Criterion.
    pub aic: f64,

    /// Corrected AIC (for small samples).
    pub aicc: f64,

    /// Bayesian Information Criterion.
    pub bic: f64,

    /// Log-likelihood.
    pub log_likelihood: f64,

    // ========== Inference Statistics (Optional) ==========
    /// Standard errors of coefficients.
    pub std_errors: Option<Col<f64>>,

    /// Standard error of intercept.
    pub intercept_std_error: Option<f64>,

    /// t-statistics for coefficients.
    pub t_statistics: Option<Col<f64>>,

    /// t-statistic for intercept.
    pub intercept_t_statistic: Option<f64>,

    /// P-values for coefficient significance tests.
    pub p_values: Option<Col<f64>>,

    /// P-value for intercept.
    pub intercept_p_value: Option<f64>,

    /// Lower bounds of confidence intervals.
    pub conf_interval_lower: Option<Col<f64>>,

    /// Upper bounds of confidence intervals.
    pub conf_interval_upper: Option<Col<f64>>,

    /// Intercept confidence interval (lower, upper).
    pub intercept_conf_interval: Option<(f64, f64)>,

    /// Confidence level used for intervals.
    pub confidence_level: f64,

    // ========== NA Handling Information ==========
    /// Information about NA handling (if NA values were present).
    ///
    /// Used by `NaAction::Exclude` to expand residuals/fitted values
    /// back to original length.
    pub na_info: Option<NaInfo>,
}

impl RegressionResult {
    /// Create a new empty result (used internally by solvers).
    pub(crate) fn empty(n_features: usize, n_observations: usize) -> Self {
        Self {
            coefficients: Col::zeros(n_features),
            intercept: None,
            residuals: Col::zeros(n_observations),
            fitted_values: Col::zeros(n_observations),
            rank: 0,
            n_parameters: 0,
            n_observations,
            aliased: vec![false; n_features],
            column_permutation: None,
            rank_tolerance: 1e-10,
            r_squared: 0.0,
            adj_r_squared: 0.0,
            rmse: 0.0,
            mse: 0.0,
            f_statistic: 0.0,
            f_pvalue: 1.0,
            aic: 0.0,
            aicc: 0.0,
            bic: 0.0,
            log_likelihood: 0.0,
            std_errors: None,
            intercept_std_error: None,
            t_statistics: None,
            intercept_t_statistic: None,
            p_values: None,
            intercept_p_value: None,
            conf_interval_lower: None,
            conf_interval_upper: None,
            intercept_conf_interval: None,
            confidence_level: 0.95,
            na_info: None,
        }
    }

    /// Residual degrees of freedom (n - p).
    pub fn residual_df(&self) -> usize {
        self.n_observations.saturating_sub(self.n_parameters)
    }

    /// Model degrees of freedom (p - 1 if intercept, else p).
    pub fn model_df(&self) -> usize {
        if self.intercept.is_some() {
            self.n_parameters.saturating_sub(1)
        } else {
            self.n_parameters
        }
    }

    /// Count of non-aliased (active) coefficients.
    pub fn n_active_coefficients(&self) -> usize {
        self.aliased.iter().filter(|&&a| !a).count()
    }

    /// Check if the model is valid (has been successfully fit).
    pub fn is_valid(&self) -> bool {
        self.rank > 0 && self.n_observations > self.n_parameters
    }

    /// Check if any coefficients are aliased.
    pub fn has_aliased(&self) -> bool {
        self.aliased.iter().any(|&a| a)
    }

    /// Get coefficient value, returning None for aliased coefficients.
    pub fn get_coefficient(&self, index: usize) -> Option<f64> {
        if index < self.coefficients.nrows() && !self.aliased[index] {
            Some(self.coefficients[index])
        } else {
            None
        }
    }

    /// Total sum of squares (TSS).
    pub fn tss(&self) -> f64 {
        let y_mean = self.fitted_values.iter().sum::<f64>() / self.n_observations as f64
            + self.residuals.iter().sum::<f64>() / self.n_observations as f64;

        self.residuals
            .iter()
            .zip(self.fitted_values.iter())
            .map(|(&r, &f)| {
                let y = f + r;
                (y - y_mean).powi(2)
            })
            .sum()
    }

    /// Residual sum of squares (RSS).
    pub fn rss(&self) -> f64 {
        self.residuals.iter().map(|&r| r.powi(2)).sum()
    }

    /// Explained sum of squares (ESS = TSS - RSS).
    pub fn ess(&self) -> f64 {
        self.tss() - self.rss()
    }

    // ========== NA-Aware Methods ==========

    /// Get residuals expanded to original length (for `NaAction::Exclude`).
    ///
    /// If NA values were removed with `NaAction::Exclude`, this returns
    /// residuals padded with NaN at the positions where rows were removed.
    /// Otherwise, returns the residuals unchanged.
    ///
    /// # Example
    ///
    /// With `na_action = NaAction::Exclude` and rows 2,3 removed:
    /// - `residuals = [r0, r1, r4]` (length 3)
    /// - `residuals_expanded() = [r0, r1, NaN, NaN, r4]` (length 5)
    pub fn residuals_expanded(&self) -> Col<f64> {
        match &self.na_info {
            Some(info) if info.needs_expansion() => info.expand(&self.residuals),
            _ => self.residuals.clone(),
        }
    }

    /// Get fitted values expanded to original length (for `NaAction::Exclude`).
    ///
    /// If NA values were removed with `NaAction::Exclude`, this returns
    /// fitted values padded with NaN at the positions where rows were removed.
    /// Otherwise, returns the fitted values unchanged.
    pub fn fitted_expanded(&self) -> Col<f64> {
        match &self.na_info {
            Some(info) if info.needs_expansion() => info.expand(&self.fitted_values),
            _ => self.fitted_values.clone(),
        }
    }

    /// The coefficient standard errors (one per coefficient, not per row, so
    /// there is nothing to expand).
    #[deprecated(
        since = "0.8.0",
        note = "standard errors are per coefficient, not per row; use `std_errors`"
    )]
    pub fn std_errors_expanded(&self) -> Option<Col<f64>> {
        self.std_errors.clone()
    }

    /// Check if NA values were removed during fitting.
    pub fn had_na_removed(&self) -> bool {
        self.na_info.as_ref().is_some_and(|info| info.has_removed())
    }

    /// Get the number of rows that were removed due to NA values.
    pub fn n_na_removed(&self) -> usize {
        self.na_info.as_ref().map_or(0, |info| info.n_removed)
    }

    /// Get the original number of observations before NA removal.
    pub fn n_original_observations(&self) -> usize {
        self.na_info
            .as_ref()
            .map_or(self.n_observations, |info| info.n_original)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_result() {
        let result = RegressionResult::empty(3, 10);
        assert_eq!(result.coefficients.nrows(), 3);
        assert_eq!(result.n_observations, 10);
        assert_eq!(result.residual_df(), 10);
    }

    #[test]
    fn test_degrees_of_freedom() {
        let mut result = RegressionResult::empty(3, 100);
        result.n_parameters = 4; // 3 coefficients + 1 intercept
        result.intercept = Some(1.0);

        assert_eq!(result.residual_df(), 96); // 100 - 4
        assert_eq!(result.model_df(), 3); // 4 - 1
    }

    #[test]
    fn test_aliased_detection() {
        let mut result = RegressionResult::empty(3, 10);
        assert!(!result.has_aliased());

        result.aliased[1] = true;
        assert!(result.has_aliased());
        assert_eq!(result.n_active_coefficients(), 2);
    }

    #[test]
    fn test_is_valid() {
        let mut result = RegressionResult::empty(3, 10);

        // Empty result is not valid (rank = 0)
        assert!(!result.is_valid());

        // Valid when rank > 0 and n > p
        result.rank = 3;
        result.n_parameters = 4;
        result.n_observations = 10;
        assert!(result.is_valid());

        // Not valid when n <= p
        result.n_observations = 4;
        assert!(!result.is_valid());

        // Not valid when n < p
        result.n_observations = 3;
        assert!(!result.is_valid());
    }

    #[test]
    fn test_get_coefficient() {
        let mut result = RegressionResult::empty(3, 10);
        result.coefficients[0] = 1.0;
        result.coefficients[1] = 2.0;
        result.coefficients[2] = 3.0;

        // Normal access
        assert_eq!(result.get_coefficient(0), Some(1.0));
        assert_eq!(result.get_coefficient(1), Some(2.0));
        assert_eq!(result.get_coefficient(2), Some(3.0));

        // Out of bounds
        assert_eq!(result.get_coefficient(5), None);

        // Aliased coefficient
        result.aliased[1] = true;
        assert_eq!(result.get_coefficient(1), None);
    }

    #[test]
    fn test_model_df_no_intercept() {
        let mut result = RegressionResult::empty(3, 100);
        result.n_parameters = 3;
        result.intercept = None;

        assert_eq!(result.model_df(), 3);
    }

    #[test]
    fn test_tss_rss_ess() {
        let mut result = RegressionResult::empty(2, 5);

        // Create simple fitted values and residuals
        // y = [1, 2, 3, 4, 5], fitted = [1.1, 2.0, 2.9, 4.0, 5.1], residuals = [-0.1, 0, 0.1, 0, -0.1]
        result.fitted_values = Col::from_fn(5, |i| match i {
            0 => 1.1,
            1 => 2.0,
            2 => 2.9,
            3 => 4.0,
            _ => 5.1,
        });
        result.residuals = Col::from_fn(5, |i| match i {
            0 => -0.1,
            1 => 0.0,
            2 => 0.1,
            3 => 0.0,
            _ => -0.1,
        });
        result.n_observations = 5;

        let rss = result.rss();
        let tss = result.tss();
        let ess = result.ess();

        // RSS = 0.01 + 0 + 0.01 + 0 + 0.01 = 0.03
        assert!((rss - 0.03).abs() < 1e-10);

        // TSS should be greater than RSS for a decent fit
        assert!(tss > rss);

        // ESS = TSS - RSS
        assert!((ess - (tss - rss)).abs() < 1e-10);
    }

    #[test]
    fn test_residuals_expanded_with_na() {
        use super::super::na_action::{NaAction, NaInfo};

        let mut result = RegressionResult::empty(2, 3);
        // Original 5 rows, rows 2 and 3 were removed
        result.residuals = Col::from_fn(3, |i| (i + 1) as f64); // [1.0, 2.0, 3.0]

        let na_info = NaInfo {
            n_original: 5,
            n_clean: 3,
            na_mask: vec![false, false, true, true, false],
            kept_indices: vec![0, 1, 4],
            n_removed: 2,
            action: NaAction::Exclude,
        };
        result.na_info = Some(na_info);

        let expanded = result.residuals_expanded();
        assert_eq!(expanded.nrows(), 5);
        assert!((expanded[0] - 1.0).abs() < 1e-10);
        assert!((expanded[1] - 2.0).abs() < 1e-10);
        assert!(expanded[2].is_nan());
        assert!(expanded[3].is_nan());
        assert!((expanded[4] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_residuals_expanded_no_na() {
        let mut result = RegressionResult::empty(2, 5);
        result.residuals = Col::from_fn(5, |i| i as f64);
        result.na_info = None;

        let expanded = result.residuals_expanded();
        assert_eq!(expanded.nrows(), 5);
        for i in 0..5 {
            assert!((expanded[i] - i as f64).abs() < 1e-10);
        }
    }

    #[test]
    fn test_fitted_expanded_with_na() {
        use super::super::na_action::{NaAction, NaInfo};

        let mut result = RegressionResult::empty(2, 3);
        result.fitted_values = Col::from_fn(3, |i| (i * 10) as f64); // [0.0, 10.0, 20.0]

        let na_info = NaInfo {
            n_original: 5,
            n_clean: 3,
            na_mask: vec![false, false, true, true, false],
            kept_indices: vec![0, 1, 4],
            n_removed: 2,
            action: NaAction::Exclude,
        };
        result.na_info = Some(na_info);

        let expanded = result.fitted_expanded();
        assert_eq!(expanded.nrows(), 5);
        assert!((expanded[0] - 0.0).abs() < 1e-10);
        assert!((expanded[1] - 10.0).abs() < 1e-10);
        assert!(expanded[2].is_nan());
        assert!(expanded[3].is_nan());
        assert!((expanded[4] - 20.0).abs() < 1e-10);
    }

    #[test]
    #[allow(deprecated)]
    fn test_std_errors_expanded_no_std_errors() {
        let result = RegressionResult::empty(2, 5);
        assert!(result.std_errors_expanded().is_none());
    }

    #[test]
    fn test_had_na_removed() {
        use super::super::na_action::{NaAction, NaInfo};

        let mut result = RegressionResult::empty(2, 3);
        assert!(!result.had_na_removed());

        let na_info = NaInfo {
            n_original: 5,
            n_clean: 3,
            na_mask: vec![false, false, true, true, false],
            kept_indices: vec![0, 1, 4],
            n_removed: 2,
            action: NaAction::Exclude,
        };
        result.na_info = Some(na_info);
        assert!(result.had_na_removed());
    }

    #[test]
    fn test_n_na_removed() {
        use super::super::na_action::{NaAction, NaInfo};

        let mut result = RegressionResult::empty(2, 3);
        assert_eq!(result.n_na_removed(), 0);

        let na_info = NaInfo {
            n_original: 5,
            n_clean: 3,
            na_mask: vec![false, false, true, true, false],
            kept_indices: vec![0, 1, 4],
            n_removed: 2,
            action: NaAction::Exclude,
        };
        result.na_info = Some(na_info);
        assert_eq!(result.n_na_removed(), 2);
    }

    #[test]
    fn test_n_original_observations() {
        use super::super::na_action::{NaAction, NaInfo};

        let mut result = RegressionResult::empty(2, 3);
        result.n_observations = 3;
        assert_eq!(result.n_original_observations(), 3);

        let na_info = NaInfo {
            n_original: 5,
            n_clean: 3,
            na_mask: vec![false, false, true, true, false],
            kept_indices: vec![0, 1, 4],
            n_removed: 2,
            action: NaAction::Exclude,
        };
        result.na_info = Some(na_info);
        assert_eq!(result.n_original_observations(), 5);
    }

    #[test]
    fn test_had_na_removed_with_no_removal() {
        use super::super::na_action::{NaAction, NaInfo};

        let mut result = RegressionResult::empty(2, 5);
        // NaInfo with n_removed = 0
        let na_info = NaInfo::no_na(5, NaAction::Omit);
        result.na_info = Some(na_info);
        assert!(!result.had_na_removed());
    }
}
