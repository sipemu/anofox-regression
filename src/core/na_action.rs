//! NA (missing value) handling for regression analysis.
//!
//! This module provides R-compatible missing value handling with four policies:
//! - `Omit`: Remove rows with NA, output is shorter than input
//! - `Exclude`: Remove rows with NA, pad output with NA at original positions
//! - `Fail`: Return error if any NA present
//! - `Pass`: Pass through NA values (solver must handle them)
//!
//! # Example
//!
//! ```
//! use anofox_regression::core::{NaAction, NaHandler};
//! use faer::{Mat, Col};
//!
//! let x = Mat::from_fn(5, 2, |i, j| if i == 2 { f64::NAN } else { (i + j) as f64 });
//! let y = Col::from_fn(5, |i| if i == 3 { f64::NAN } else { i as f64 });
//!
//! // Remove rows with NA
//! let result = NaHandler::process(&x, &y, NaAction::Omit).unwrap();
//! assert_eq!(result.x_clean.nrows(), 3); // Rows 0, 1, 4 kept
//! ```

use faer::{Col, Mat};
use thiserror::Error;

/// Action to take when missing values (NA/NaN) are encountered.
///
/// Mirrors R's `na.action` parameter in `lm()` and other regression functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NaAction {
    /// Remove rows containing NA values. Output vectors (residuals, fitted)
    /// will be shorter than input.
    ///
    /// Equivalent to R's `na.omit`.
    #[default]
    Omit,

    /// Remove rows containing NA values, but pad output vectors with NA
    /// at the original positions so they match the input length.
    ///
    /// Equivalent to R's `na.exclude`.
    Exclude,

    /// Return an error if any NA values are present.
    ///
    /// Equivalent to R's `na.fail`.
    Fail,

    /// Pass NA values through without modification.
    /// The solver must handle NA values appropriately.
    ///
    /// Equivalent to R's `na.pass`.
    Pass,
}

/// Error when NA values are encountered with `NaAction::Fail`.
#[derive(Debug, Error)]
pub enum NaError {
    /// NA values found in input data when using `NaAction::Fail`.
    #[error("NA values found in data (na.fail): {n_na} rows contain missing values")]
    NaValuesPresent { n_na: usize },

    /// All observations were removed due to NA values.
    #[error("all observations contain NA values")]
    AllNa,

    /// Insufficient observations remaining after NA removal.
    #[error("insufficient observations after NA removal: {remaining} remaining, {needed} needed")]
    InsufficientAfterNa { remaining: usize, needed: usize },
}

/// Information about NA handling applied to data.
///
/// Used by `NaAction::Exclude` to reconstruct original-length output.
#[derive(Debug, Clone)]
pub struct NaInfo {
    /// Original number of observations before NA removal.
    pub n_original: usize,

    /// Number of observations after NA removal.
    pub n_clean: usize,

    /// Mask indicating which rows had NA values (true = had NA).
    pub na_mask: Vec<bool>,

    /// Indices of rows that were kept (no NA).
    pub kept_indices: Vec<usize>,

    /// Number of rows removed due to NA.
    pub n_removed: usize,

    /// The NA action that was applied.
    pub action: NaAction,
}

impl NaInfo {
    /// Check if any rows were removed.
    pub fn has_removed(&self) -> bool {
        self.n_removed > 0
    }

    /// Check if this info requires expansion (i.e., was created with `Exclude`).
    pub fn needs_expansion(&self) -> bool {
        self.action == NaAction::Exclude && self.n_removed > 0
    }

    /// Expand a vector to original length, inserting NaN at removed positions.
    ///
    /// Used with `NaAction::Exclude` to pad residuals/fitted values.
    pub fn expand(&self, clean_values: &Col<f64>) -> Col<f64> {
        if !self.needs_expansion() {
            return clean_values.clone();
        }

        let mut expanded = Col::zeros(self.n_original);
        let mut clean_idx = 0;

        for (orig_idx, &had_na) in self.na_mask.iter().enumerate() {
            if had_na {
                expanded[orig_idx] = f64::NAN;
            } else {
                expanded[orig_idx] = clean_values[clean_idx];
                clean_idx += 1;
            }
        }

        expanded
    }

    /// Create NaInfo for data with no NA values.
    pub fn no_na(n_observations: usize, action: NaAction) -> Self {
        Self {
            n_original: n_observations,
            n_clean: n_observations,
            na_mask: vec![false; n_observations],
            kept_indices: (0..n_observations).collect(),
            n_removed: 0,
            action,
        }
    }
}

/// Result of NA preprocessing.
#[derive(Debug, Clone)]
pub struct NaResult {
    /// Cleaned feature matrix (rows with NA removed, unless `Pass`).
    pub x_clean: Mat<f64>,

    /// Cleaned response vector (rows with NA removed, unless `Pass`).
    pub y_clean: Col<f64>,

    /// Information about the NA handling applied.
    pub na_info: NaInfo,
}

/// Handler for missing value processing.
pub struct NaHandler;

impl NaHandler {
    /// Process input data according to the specified NA action.
    ///
    /// # Arguments
    ///
    /// * `x` - Feature matrix (n_samples x n_features)
    /// * `y` - Response vector (n_samples)
    /// * `action` - How to handle NA values
    ///
    /// # Returns
    ///
    /// `NaResult` containing cleaned data and NA information.
    ///
    /// # Errors
    ///
    /// - `NaError::NaValuesPresent` if `action` is `Fail` and NA values exist
    /// - `NaError::AllNa` if all rows contain NA values
    pub fn process(x: &Mat<f64>, y: &Col<f64>, action: NaAction) -> Result<NaResult, NaError> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        // Find rows with NA values
        let na_mask = Self::find_na_rows(x, y);
        let n_na = na_mask.iter().filter(|&&v| v).count();

        // Handle based on action
        match action {
            NaAction::Fail => {
                if n_na > 0 {
                    return Err(NaError::NaValuesPresent { n_na });
                }
                Ok(NaResult {
                    x_clean: x.clone(),
                    y_clean: y.clone(),
                    na_info: NaInfo::no_na(n_samples, action),
                })
            }

            NaAction::Pass => {
                // Pass through without modification
                Ok(NaResult {
                    x_clean: x.clone(),
                    y_clean: y.clone(),
                    na_info: NaInfo::no_na(n_samples, action),
                })
            }

            NaAction::Omit | NaAction::Exclude => {
                if n_na == n_samples {
                    return Err(NaError::AllNa);
                }

                if n_na == 0 {
                    // No NA values, return as-is
                    return Ok(NaResult {
                        x_clean: x.clone(),
                        y_clean: y.clone(),
                        na_info: NaInfo::no_na(n_samples, action),
                    });
                }

                // Build list of kept indices
                let kept_indices: Vec<usize> = na_mask
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &had_na)| if !had_na { Some(i) } else { None })
                    .collect();

                let n_clean = kept_indices.len();

                // Create cleaned matrices
                let x_clean = Mat::from_fn(n_clean, n_features, |i, j| x[(kept_indices[i], j)]);
                let y_clean = Col::from_fn(n_clean, |i| y[kept_indices[i]]);

                let na_info = NaInfo {
                    n_original: n_samples,
                    n_clean,
                    na_mask,
                    kept_indices,
                    n_removed: n_na,
                    action,
                };

                Ok(NaResult {
                    x_clean,
                    y_clean,
                    na_info,
                })
            }
        }
    }

    /// Process input data with optional weights according to the specified NA action.
    ///
    /// Similar to `process`, but also handles NA values in weights.
    pub fn process_with_weights(
        x: &Mat<f64>,
        y: &Col<f64>,
        weights: &Col<f64>,
        action: NaAction,
    ) -> Result<(NaResult, Col<f64>), NaError> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        // Find rows with NA values (including weights)
        let na_mask = Self::find_na_rows_with_weights(x, y, weights);
        let n_na = na_mask.iter().filter(|&&v| v).count();

        match action {
            NaAction::Fail => {
                if n_na > 0 {
                    return Err(NaError::NaValuesPresent { n_na });
                }
                Ok((
                    NaResult {
                        x_clean: x.clone(),
                        y_clean: y.clone(),
                        na_info: NaInfo::no_na(n_samples, action),
                    },
                    weights.clone(),
                ))
            }

            NaAction::Pass => Ok((
                NaResult {
                    x_clean: x.clone(),
                    y_clean: y.clone(),
                    na_info: NaInfo::no_na(n_samples, action),
                },
                weights.clone(),
            )),

            NaAction::Omit | NaAction::Exclude => {
                if n_na == n_samples {
                    return Err(NaError::AllNa);
                }

                if n_na == 0 {
                    return Ok((
                        NaResult {
                            x_clean: x.clone(),
                            y_clean: y.clone(),
                            na_info: NaInfo::no_na(n_samples, action),
                        },
                        weights.clone(),
                    ));
                }

                let kept_indices: Vec<usize> = na_mask
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &had_na)| if !had_na { Some(i) } else { None })
                    .collect();

                let n_clean = kept_indices.len();

                let x_clean = Mat::from_fn(n_clean, n_features, |i, j| x[(kept_indices[i], j)]);
                let y_clean = Col::from_fn(n_clean, |i| y[kept_indices[i]]);
                let weights_clean = Col::from_fn(n_clean, |i| weights[kept_indices[i]]);

                let na_info = NaInfo {
                    n_original: n_samples,
                    n_clean,
                    na_mask,
                    kept_indices,
                    n_removed: n_na,
                    action,
                };

                Ok((
                    NaResult {
                        x_clean,
                        y_clean,
                        na_info,
                    },
                    weights_clean,
                ))
            }
        }
    }

    /// Find rows containing NA values in X or y.
    fn find_na_rows(x: &Mat<f64>, y: &Col<f64>) -> Vec<bool> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        (0..n_samples)
            .map(|i| {
                // Check y
                if y[i].is_nan() {
                    return true;
                }
                // Check all features in row
                for j in 0..n_features {
                    if x[(i, j)].is_nan() {
                        return true;
                    }
                }
                false
            })
            .collect()
    }

    /// Find rows containing NA values in X, y, or weights.
    fn find_na_rows_with_weights(x: &Mat<f64>, y: &Col<f64>, weights: &Col<f64>) -> Vec<bool> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        (0..n_samples)
            .map(|i| {
                // Check y
                if y[i].is_nan() {
                    return true;
                }
                // Check weights
                if weights[i].is_nan() {
                    return true;
                }
                // Check all features in row
                for j in 0..n_features {
                    if x[(i, j)].is_nan() {
                        return true;
                    }
                }
                false
            })
            .collect()
    }

    /// Check if a matrix contains any NA values.
    pub fn has_na_matrix(x: &Mat<f64>) -> bool {
        let n_rows = x.nrows();
        let n_cols = x.ncols();
        for i in 0..n_rows {
            for j in 0..n_cols {
                if x[(i, j)].is_nan() {
                    return true;
                }
            }
        }
        false
    }

    /// Check if a vector contains any NA values.
    pub fn has_na_vector(v: &Col<f64>) -> bool {
        v.iter().any(|&x| x.is_nan())
    }

    /// Count NA values in a matrix.
    pub fn count_na_matrix(x: &Mat<f64>) -> usize {
        let n_rows = x.nrows();
        let n_cols = x.ncols();
        let mut count = 0;
        for i in 0..n_rows {
            for j in 0..n_cols {
                if x[(i, j)].is_nan() {
                    count += 1;
                }
            }
        }
        count
    }

    /// Count NA values in a vector.
    pub fn count_na_vector(v: &Col<f64>) -> usize {
        v.iter().filter(|&&x| x.is_nan()).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_data_with_na() -> (Mat<f64>, Col<f64>) {
        // 5 rows, 2 features
        // Row 2: NA in x[2,0]
        // Row 3: NA in y[3]
        let x = Mat::from_fn(5, 2, |i, j| {
            if i == 2 && j == 0 {
                f64::NAN
            } else {
                (i * 2 + j) as f64
            }
        });
        let y = Col::from_fn(5, |i| if i == 3 { f64::NAN } else { (i * 10) as f64 });
        (x, y)
    }

    fn create_clean_data() -> (Mat<f64>, Col<f64>) {
        let x = Mat::from_fn(5, 2, |i, j| (i * 2 + j) as f64);
        let y = Col::from_fn(5, |i| (i * 10) as f64);
        (x, y)
    }

    #[test]
    fn test_na_omit() {
        let (x, y) = create_test_data_with_na();
        let result = NaHandler::process(&x, &y, NaAction::Omit).unwrap();

        // Should keep rows 0, 1, 4 (remove rows 2 and 3)
        assert_eq!(result.x_clean.nrows(), 3);
        assert_eq!(result.y_clean.nrows(), 3);
        assert_eq!(result.na_info.n_removed, 2);
        assert_eq!(result.na_info.kept_indices, vec![0, 1, 4]);
        assert!(!result.na_info.needs_expansion());
    }

    #[test]
    fn test_na_exclude() {
        let (x, y) = create_test_data_with_na();
        let result = NaHandler::process(&x, &y, NaAction::Exclude).unwrap();

        assert_eq!(result.x_clean.nrows(), 3);
        assert_eq!(result.y_clean.nrows(), 3);
        assert_eq!(result.na_info.n_removed, 2);
        assert!(result.na_info.needs_expansion());

        // Test expansion
        let clean_resid = Col::from_fn(3, |i| (i + 1) as f64);
        let expanded = result.na_info.expand(&clean_resid);
        assert_eq!(expanded.nrows(), 5);
        assert!((expanded[0] - 1.0).abs() < 1e-10);
        assert!((expanded[1] - 2.0).abs() < 1e-10);
        assert!(expanded[2].is_nan()); // Was NA row
        assert!(expanded[3].is_nan()); // Was NA row
        assert!((expanded[4] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_na_fail() {
        let (x, y) = create_test_data_with_na();
        let result = NaHandler::process(&x, &y, NaAction::Fail);

        assert!(matches!(result, Err(NaError::NaValuesPresent { n_na: 2 })));
    }

    #[test]
    fn test_na_fail_no_na() {
        let (x, y) = create_clean_data();
        let result = NaHandler::process(&x, &y, NaAction::Fail).unwrap();

        assert_eq!(result.x_clean.nrows(), 5);
        assert_eq!(result.na_info.n_removed, 0);
    }

    #[test]
    fn test_na_pass() {
        let (x, y) = create_test_data_with_na();
        let result = NaHandler::process(&x, &y, NaAction::Pass).unwrap();

        // Should keep all data unchanged
        assert_eq!(result.x_clean.nrows(), 5);
        assert!(result.x_clean[(2, 0)].is_nan());
        assert!(result.y_clean[3].is_nan());
    }

    #[test]
    fn test_clean_data() {
        let (x, y) = create_clean_data();
        let result = NaHandler::process(&x, &y, NaAction::Omit).unwrap();

        assert_eq!(result.x_clean.nrows(), 5);
        assert_eq!(result.na_info.n_removed, 0);
        assert!(!result.na_info.needs_expansion());
    }

    #[test]
    fn test_all_na() {
        let x = Mat::from_fn(3, 2, |_, _| f64::NAN);
        let y = Col::from_fn(3, |_| f64::NAN);
        let result = NaHandler::process(&x, &y, NaAction::Omit);

        assert!(matches!(result, Err(NaError::AllNa)));
    }

    #[test]
    fn test_has_na_helpers() {
        let (x_na, y_na) = create_test_data_with_na();
        let (x_clean, y_clean) = create_clean_data();

        assert!(NaHandler::has_na_matrix(&x_na));
        assert!(NaHandler::has_na_vector(&y_na));
        assert!(!NaHandler::has_na_matrix(&x_clean));
        assert!(!NaHandler::has_na_vector(&y_clean));
    }

    #[test]
    fn test_count_na() {
        let (x, y) = create_test_data_with_na();
        assert_eq!(NaHandler::count_na_matrix(&x), 1);
        assert_eq!(NaHandler::count_na_vector(&y), 1);
    }

    #[test]
    fn test_process_with_weights() {
        let (x, y) = create_clean_data();
        let mut weights = Col::from_fn(5, |i| (i + 1) as f64);
        weights[2] = f64::NAN; // NA in weights

        let (result, clean_weights) =
            NaHandler::process_with_weights(&x, &y, &weights, NaAction::Omit).unwrap();

        // Row 2 should be removed due to NA in weights
        assert_eq!(result.x_clean.nrows(), 4);
        assert_eq!(clean_weights.nrows(), 4);
        assert_eq!(result.na_info.n_removed, 1);
    }

    #[test]
    fn test_na_info_no_na() {
        let info = NaInfo::no_na(10, NaAction::Omit);
        assert_eq!(info.n_original, 10);
        assert_eq!(info.n_clean, 10);
        assert_eq!(info.n_removed, 0);
        assert!(!info.has_removed());
        assert!(!info.needs_expansion());
    }

    #[test]
    fn test_expand_no_expansion_needed() {
        let info = NaInfo::no_na(5, NaAction::Omit);
        let values = Col::from_fn(5, |i| i as f64);
        let expanded = info.expand(&values);
        assert_eq!(expanded.nrows(), 5);
        for i in 0..5 {
            assert!((expanded[i] - i as f64).abs() < 1e-10);
        }
    }

    #[test]
    fn test_na_action_default() {
        assert_eq!(NaAction::default(), NaAction::Omit);
    }
}

/// The complete rows of `x` and `y`, and which rows were dropped.
pub(crate) type CompleteCases = (Mat<f64>, Col<f64>, NaInfo);

/// Applies `action` before a fit of `x` and `y`: `Ok(None)` when the data
/// are fitted as given (no NaN, `Pass`, or mismatched lengths left for the
/// estimator to report), `Ok(Some((x, y)))` with the complete cases for
/// `Omit` / `Exclude` (store the [`NaInfo`] on the fit's result, so that
/// `Exclude` pads residuals and fitted values back to the input rows), and an
/// error for `Fail`. With row-aligned extras
/// (weights, an offset) rows cannot be dropped here, so NaN rows are an
/// error.
pub(crate) fn apply_na_action(
    x: &Mat<f64>,
    y: &Col<f64>,
    action: NaAction,
    has_row_extras: bool,
) -> Result<Option<CompleteCases>, crate::solvers::RegressionError> {
    use crate::solvers::RegressionError;
    if action == NaAction::Pass
        || x.nrows() != y.nrows()
        || (!NaHandler::has_na_matrix(x) && !NaHandler::has_na_vector(y))
    {
        return Ok(None);
    }
    if has_row_extras && action != NaAction::Fail {
        return Err(RegressionError::NumericalError(
            "x or y contain NaN and the fit has row-aligned weights or an offset: \
             drop the incomplete rows from all inputs first"
                .to_string(),
        ));
    }
    let r = NaHandler::process(x, y, action)
        .map_err(|e| RegressionError::NumericalError(e.to_string()))?;
    Ok(Some((r.x_clean, r.y_clean, r.na_info)))
}
