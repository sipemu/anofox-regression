//! Variance factors of linear estimators and leverage of new rows.
//!
//! For a linear predictor `ŷ₀ = x₀'β̂` (with `x₀` augmented by a leading 1 when
//! the model has an intercept) the sampling variance is `σ² · x₀' M x₀`, where
//! the *variance factor* `M` depends on the estimator:
//!
//! | Estimator | `M` |
//! |---|---|
//! | OLS | `(X'X)⁻¹` |
//! | WLS | `(X'WX)⁻¹` |
//! | Ridge | `A X'X A`, `A = (X'X + λP)⁻¹` |
//! | Weighted ridge | `A X'WX A`, `A = (X'WX + λP)⁻¹` |
//!
//! `P = I` except that an intercept is not penalised (`P₀₀ = 0`); this is the
//! augmented form of the centred ridge estimator used by
//! [`RidgeRegressor`](crate::solvers::RidgeRegressor). For ridge, `λ` is the
//! effective (raw) penalty; with glmnet scaling pass `λ · n`.
//!
//! `x₀' M x₀` is the leverage of a new row: the confidence-interval standard
//! error of the mean is `s·sqrt(x₀'Mx₀)`, the prediction-interval standard
//! error `s·sqrt(1 + x₀'Mx₀)`. These functions let callers that keep only the
//! coefficients (e.g. a serialized model or a database aggregate state)
//! compute leverage-aware intervals without refitting.

use crate::core::{IntervalType, PredictionResult};
use faer::{Col, Mat};

/// Compute the variance factor `M` of a (weighted, optionally ridge-penalised)
/// linear fit.
///
/// * `x` – feature matrix (`n × p`) **without** an intercept column.
/// * `weights` – optional non-negative observation weights (`None` = OLS).
/// * `with_intercept` – whether the model has an (unpenalised) intercept.
/// * `excluded` – columns that are not part of the fitted model (aliased,
///   constant, or inactive in a sparse fit); length `p`. Their rows and
///   columns of `M` are zero.
/// * `ridge_lambda` – effective ridge penalty, `0` for OLS / WLS.
///
/// Returns the full `dim × dim` matrix, `dim = p + 1` with an intercept (index
/// 0 is the intercept) and `p` without. Errors if the inputs are inconsistent
/// or the (penalised) Gram matrix of the retained columns is numerically
/// singular.
pub fn compute_variance_factor(
    x: &Mat<f64>,
    weights: Option<&Col<f64>>,
    with_intercept: bool,
    excluded: &[bool],
    ridge_lambda: f64,
) -> Result<Mat<f64>, &'static str> {
    let n = x.nrows();
    let p = x.ncols();
    if excluded.len() != p {
        return Err("excluded must have one entry per column of x");
    }
    if let Some(w) = weights {
        if w.nrows() != n {
            return Err("weights must have one entry per row of x");
        }
        if w.iter().any(|&wi| !(wi.is_finite() && wi >= 0.0)) {
            return Err("weights must be finite and non-negative");
        }
    }
    if !(ridge_lambda.is_finite() && ridge_lambda >= 0.0) {
        return Err("ridge_lambda must be finite and non-negative");
    }

    let off = usize::from(with_intercept);
    let dim = p + off;
    let retained: Vec<usize> = (0..dim)
        .filter(|&j| j < off || !excluded[j - off])
        .collect();
    let q = retained.len();
    if q == 0 {
        return Err("all columns are excluded");
    }
    let value = |i: usize, j: usize| if j < off { 1.0 } else { x[(i, j - off)] };

    // G = X_r' W X_r over the retained columns.
    let mut g = Mat::<f64>::zeros(q, q);
    for i in 0..n {
        let w = weights.map_or(1.0, |w| w[i]);
        if w == 0.0 {
            continue;
        }
        for a in 0..q {
            let va = w * value(i, retained[a]);
            for b in a..q {
                g[(a, b)] += va * value(i, retained[b]);
            }
        }
    }
    for a in 0..q {
        for b in (a + 1)..q {
            g[(b, a)] = g[(a, b)];
        }
    }

    let m_reduced = if ridge_lambda > 0.0 {
        let mut h = g.clone();
        for (a, &j) in retained.iter().enumerate() {
            if j >= off {
                h[(a, a)] += ridge_lambda;
            }
        }
        let a_inv = super::prediction::compute_matrix_inverse(&h)?;
        &(&a_inv * &g) * &a_inv
    } else {
        super::prediction::compute_matrix_inverse(&g)?
    };

    let mut m = Mat::zeros(dim, dim);
    for (a, &ja) in retained.iter().enumerate() {
        for (b, &jb) in retained.iter().enumerate() {
            m[(ja, jb)] = m_reduced[(a, b)];
        }
    }
    Ok(m)
}

/// Expand a reduced factor (retained columns only, intercept first) to the
/// full dimension with zero rows/columns for `aliased` columns.
pub(crate) fn expand_reduced_factor(
    reduced: &Mat<f64>,
    aliased: &[bool],
    with_intercept: bool,
) -> Mat<f64> {
    let off = usize::from(with_intercept);
    let dim = aliased.len() + off;
    let retained: Vec<usize> = (0..dim).filter(|&j| j < off || !aliased[j - off]).collect();
    let mut m = Mat::zeros(dim, dim);
    for (a, &ja) in retained.iter().enumerate() {
        for (b, &jb) in retained.iter().enumerate() {
            m[(ja, jb)] = reduced[(a, b)];
        }
    }
    m
}

/// Ridge sandwich variance factor `A X'WX A`, `A = (X'WX + λP)⁻¹`.
///
/// Convenience wrapper around [`compute_variance_factor`]; `lambda` is the
/// effective penalty (for glmnet scaling pass `λ · n`) and `aliased` marks
/// columns to exclude.
pub fn compute_ridge_variance_factor(
    x: &Mat<f64>,
    weights: Option<&Col<f64>>,
    lambda: f64,
    with_intercept: bool,
    aliased: &[bool],
) -> Result<Mat<f64>, &'static str> {
    compute_variance_factor(x, weights, with_intercept, aliased, lambda)
}

/// Leverage `x₀' M x₀` of each row of `x_new`.
///
/// `m` is a full variance factor as returned by [`compute_variance_factor`]
/// (`x_new.ncols() + 1` square with an intercept, `x_new.ncols()` without).
/// Excluded columns have zero rows/columns in `m`, so their values in `x_new`
/// do not matter (they may be NaN). Rows whose retained values are non-finite
/// give NaN. Panics if the dimensions do not match.
pub fn leverage_new(m: &Mat<f64>, x_new: &Mat<f64>, with_intercept: bool) -> Col<f64> {
    let off = usize::from(with_intercept);
    let dim = x_new.ncols() + off;
    assert!(
        m.nrows() == dim && m.ncols() == dim,
        "variance factor is {}x{}, expected {dim}x{dim}",
        m.nrows(),
        m.ncols()
    );
    let active: Vec<usize> = (0..dim).filter(|&j| m[(j, j)] != 0.0).collect();
    Col::from_fn(x_new.nrows(), |i| {
        let v = |j: usize| if j < off { 1.0 } else { x_new[(i, j - off)] };
        let mut h = 0.0;
        for &a in &active {
            let va = v(a);
            for &b in &active {
                h += va * m[(a, b)] * v(b);
            }
        }
        h
    })
}

/// Leverage-aware confidence / prediction intervals from a variance factor.
///
/// `predictions` are the point predictions for `x_new`, `sigma2` the residual
/// variance estimate (e.g. `RegressionResult::mse`) and `df` the residual
/// degrees of freedom. The standard error is `sqrt(σ²·h)` (confidence) or
/// `sqrt(σ²·(1 + h))` (prediction) with `h = x₀'Mx₀`; bounds use the
/// `t_{df}` quantile. Returns NaN intervals when `df ≤ 0` or `σ²` is not
/// finite.
#[allow(clippy::too_many_arguments)]
pub fn intervals_from_variance_factor(
    x_new: &Mat<f64>,
    m: &Mat<f64>,
    predictions: &Col<f64>,
    sigma2: f64,
    df: f64,
    confidence_level: f64,
    interval_type: IntervalType,
    with_intercept: bool,
) -> PredictionResult {
    let n = x_new.nrows();
    if !(df > 0.0 && sigma2.is_finite() && sigma2 >= 0.0) {
        let nan = Col::from_fn(n, |_| f64::NAN);
        return PredictionResult::with_intervals(
            predictions.clone(),
            nan.clone(),
            nan.clone(),
            nan,
        );
    }
    let h = leverage_new(m, x_new, with_intercept);
    let t_crit = {
        use statrs::distribution::{ContinuousCDF, StudentsT};
        StudentsT::new(0.0, 1.0, df)
            .map(|d| d.inverse_cdf(1.0 - (1.0 - confidence_level) / 2.0))
            .unwrap_or(f64::NAN)
    };
    let se = Col::from_fn(n, |i| {
        let var = match interval_type {
            IntervalType::Confidence => sigma2 * h[i],
            IntervalType::Prediction => sigma2 * (1.0 + h[i]),
        };
        if var >= 0.0 {
            var.sqrt()
        } else {
            f64::NAN
        }
    });
    let lower = Col::from_fn(n, |i| predictions[i] - t_crit * se[i]);
    let upper = Col::from_fn(n, |i| predictions[i] + t_crit * se[i]);
    PredictionResult::with_intervals(predictions.clone(), lower, upper, se)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ols_factor_is_inverse_gram() {
        let x = Mat::from_fn(6, 1, |i, _| i as f64);
        let m = compute_variance_factor(&x, None, true, &[false], 0.0).unwrap();
        // [n, Σx; Σx, Σx²] = [6, 15; 15, 55], det = 105
        assert!((m[(0, 0)] - 55.0 / 105.0).abs() < 1e-12);
        assert!((m[(0, 1)] + 15.0 / 105.0).abs() < 1e-12);
        assert!((m[(1, 1)] - 6.0 / 105.0).abs() < 1e-12);
        let h = leverage_new(&m, &Mat::from_fn(1, 1, |_, _| 2.5), true);
        assert!((h[0] - 1.0 / 6.0).abs() < 1e-12); // at the centroid: 1/n
    }

    #[test]
    fn excluded_columns_are_zero() {
        let x = Mat::from_fn(6, 2, |i, j| if j == 0 { i as f64 } else { 3.0 });
        let m = compute_variance_factor(&x, None, true, &[false, true], 0.0).unwrap();
        assert_eq!(m.nrows(), 3);
        assert_eq!(m[(2, 2)], 0.0);
        assert_eq!(m[(0, 2)], 0.0);
        let x_new = Mat::from_fn(1, 2, |_, j| if j == 0 { 2.5 } else { f64::NAN });
        assert!((leverage_new(&m, &x_new, true)[0] - 1.0 / 6.0).abs() < 1e-12);
    }
}
