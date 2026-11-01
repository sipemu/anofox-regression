//! Input checks shared by the estimators.

use crate::solvers::RegressionError;
use faer::{Col, Mat};

/// `x` and `y` have the same number of rows, at least one, and only finite
/// values. Estimators with an [`NaAction`](crate::core::NaAction) call this
/// after removing incomplete rows, so a NaN left over (`NaAction::Pass`) or
/// an infinite value is an error, as R's `lm.fit` rejects
/// `NA/NaN/Inf in 'x'`.
pub(crate) fn validate_xy(x: &Mat<f64>, y: &Col<f64>) -> Result<(), RegressionError> {
    if x.nrows() != y.nrows() {
        return Err(RegressionError::DimensionMismatch {
            x_rows: x.nrows(),
            y_len: y.nrows(),
        });
    }
    if x.nrows() == 0 {
        return Err(RegressionError::InsufficientObservations { needed: 1, got: 0 });
    }
    validate_finite(x, y)
}

/// Every value of `x` and `y` is finite.
pub(crate) fn validate_finite(x: &Mat<f64>, y: &Col<f64>) -> Result<(), RegressionError> {
    for j in 0..x.ncols() {
        if let Some(i) = (0..x.nrows()).find(|&i| !x[(i, j)].is_finite()) {
            return Err(RegressionError::NumericalError(format!(
                "X contains a non-finite value at row {i}, column {j}"
            )));
        }
    }
    match y.iter().position(|v| !v.is_finite()) {
        Some(i) => Err(RegressionError::NumericalError(format!(
            "y contains a non-finite value at row {i}"
        ))),
        None => Ok(()),
    }
}

/// Every value of a row-aligned extra (weights, an offset) is finite.
pub(crate) fn validate_finite_extra(name: &str, v: &Col<f64>) -> Result<(), RegressionError> {
    match v.iter().position(|v| !v.is_finite()) {
        Some(i) => Err(RegressionError::NumericalError(format!(
            "{name} contains a non-finite value at row {i}"
        ))),
        None => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_mismatch_empty_and_non_finite() {
        let x = Mat::from_fn(3, 2, |i, j| (i + j) as f64);
        let y = Col::from_fn(3, |i| i as f64);
        assert!(validate_xy(&x, &y).is_ok());
        assert!(validate_xy(&x, &Col::zeros(2)).is_err());
        assert!(validate_xy(&Mat::zeros(0, 2), &Col::zeros(0)).is_err());
        let mut xn = x.clone();
        xn[(1, 1)] = f64::INFINITY;
        assert!(validate_xy(&xn, &y).is_err());
        let mut yn = y.clone();
        yn[2] = f64::NAN;
        assert!(validate_xy(&x, &yn).is_err());
        assert!(validate_finite_extra("weights", &Col::from_fn(2, |_| f64::NAN)).is_err());
        assert!(validate_finite_extra("weights", &Col::from_fn(2, |_| 1.0)).is_ok());
    }
}
