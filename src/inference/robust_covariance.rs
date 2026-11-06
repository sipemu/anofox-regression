//! Heteroskedasticity-Consistent (HC) standard errors.
//!
//! Implements HC0–HC3 robust covariance estimators using the sandwich formula:
//! `V_HC = (X'X)^-1 X' Ω X (X'X)^-1`
//!
//! where Ω is a diagonal matrix with observation-specific weights that depend
//! on the HC variant.
//!
//! # References
//!
//! - White, H. (1980). "A Heteroskedasticity-Consistent Covariance Matrix Estimator
//!   and a Direct Test for Heteroskedasticity." *Econometrica*, 48(4), 817–838.
//! - MacKinnon, J.G. & White, H. (1985). "Some Heteroskedasticity-Consistent
//!   Covariance Matrix Estimators with Improved Finite Sample Properties."
//!   *Journal of Econometrics*, 29(3), 305–325.

use crate::inference::coefficient::CoefficientInference;
use crate::solvers::RegressionError;
use faer::{Col, Mat};
use statrs::distribution::{ContinuousCDF, StudentsT};

/// Type of heteroskedasticity-consistent standard errors.
///
/// All variants use the sandwich estimator `(X'X)^-1 X' Ω X (X'X)^-1`
/// with different diagonal weights in Ω.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum HcType {
    /// HC0: `ω_i = e_i²`
    ///
    /// The original White (1980) estimator. Consistent but biased in finite samples.
    HC0,
    /// HC1: `ω_i = n/(n-p) * e_i²`
    ///
    /// Degrees-of-freedom corrected version. Default in many software packages.
    /// This is the default variant.
    #[default]
    HC1,
    /// HC2: `ω_i = e_i² / (1 - h_ii)`
    ///
    /// Uses leverage values to reduce bias. Recommended by MacKinnon & White (1985).
    HC2,
    /// HC3: `ω_i = e_i² / (1 - h_ii)²`
    ///
    /// Jackknife-like estimator. Most conservative, best finite-sample properties.
    HC3,
}

/// Result of HC standard error computation.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct HcResult {
    /// HC standard errors for each coefficient.
    pub std_errors: Col<f64>,
    /// HC standard error for the intercept (if the model has one).
    pub intercept_std_error: Option<f64>,
}

/// Full HC inference for coefficients: SE, t-statistics, p-values, confidence intervals.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct HcInference {
    /// Which HC variant was used.
    pub hc_type: HcType,
    /// HC standard errors for each coefficient.
    pub std_errors: Col<f64>,
    /// t-statistics using HC standard errors.
    pub t_statistics: Col<f64>,
    /// Two-sided p-values from t-distribution.
    pub p_values: Col<f64>,
    /// Lower bounds of confidence intervals.
    pub conf_interval_lower: Col<f64>,
    /// Upper bounds of confidence intervals.
    pub conf_interval_upper: Col<f64>,
    /// Confidence level used (e.g., 0.95).
    pub confidence_level: f64,
    /// Intercept inference (if the model has an intercept).
    pub intercept: Option<HcInterceptInference>,
}

/// HC inference for the intercept term.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct HcInterceptInference {
    /// HC standard error for the intercept.
    pub std_error: f64,
    /// t-statistic using HC standard error.
    pub t_statistic: f64,
    /// Two-sided p-value from t-distribution.
    pub p_value: f64,
    /// Confidence interval (lower, upper).
    pub conf_interval: (f64, f64),
}

/// Compute HC standard errors for OLS coefficients.
///
/// Uses the sandwich estimator: `V_HC = (X'X)^-1 X' Ω X (X'X)^-1`; the
/// standard errors are the square roots of the diagonal of
/// [`compute_hc_vcov`] (`NaN` for aliased columns).
///
/// # Arguments
/// * `x` - Feature matrix (n × p), WITHOUT the intercept column
/// * `residuals` - OLS residuals (n × 1)
/// * `aliased` - Boolean mask for aliased (collinear) columns
/// * `with_intercept` - Whether the model includes an intercept
/// * `hc_type` - Which HC variant to compute
///
/// # Returns
/// `HcResult` with standard errors for coefficients and optionally the intercept.
pub fn compute_hc_standard_errors(
    x: &Mat<f64>,
    residuals: &Col<f64>,
    aliased: &[bool],
    with_intercept: bool,
    hc_type: HcType,
) -> Result<HcResult, RegressionError> {
    let vcov = compute_hc_vcov(x, residuals, None, aliased, with_intercept, hc_type)?;
    let off = usize::from(with_intercept);
    let se = |k: usize| {
        let var = vcov[(k, k)];
        if var >= 0.0 {
            var.sqrt()
        } else {
            f64::NAN
        }
    };
    let std_errors = Col::from_fn(
        x.ncols(),
        |j| {
            if aliased[j] {
                f64::NAN
            } else {
                se(j + off)
            }
        },
    );
    Ok(HcResult {
        std_errors,
        intercept_std_error: with_intercept.then(|| vcov[(0, 0)].max(0.0).sqrt()),
    })
}

/// Heteroskedasticity-consistent covariance matrix of an OLS (`weights =
/// None`) or WLS fit, as `sandwich::vcovHC(lm(...), type)`.
///
/// `V = (XᵀWX)⁻¹ (Σᵢ ωᵢ xᵢxᵢᵀ) (XᵀWX)⁻¹` with `ωᵢ = (wᵢeᵢ)²` (HC0),
/// times `n / (n − k)` (HC1), divided by `1 − hᵢᵢ` (HC2) or `(1 − hᵢᵢ)²`
/// (HC3), where `hᵢᵢ = wᵢ xᵢᵀ(XᵀWX)⁻¹xᵢ` is the (weighted) leverage, `n` the
/// number of rows and `k` the number of estimated coefficients. A row with
/// leverage 1 keeps its HC0 weight.
///
/// `x` is the design without intercept column, `residuals = y − ŷ` (one per
/// row of `x`), `weights` the case weights of a WLS fit, and `aliased` marks
/// columns that are not part of the fit. The result is `[intercept,] p`
/// square (intercept first) with zero rows and columns for aliased columns,
/// the layout of `FittedOls::vcov`.
pub fn compute_hc_vcov(
    x: &Mat<f64>,
    residuals: &Col<f64>,
    weights: Option<&Col<f64>>,
    aliased: &[bool],
    with_intercept: bool,
    hc_type: HcType,
) -> Result<Mat<f64>, RegressionError> {
    let n = x.nrows();
    if residuals.nrows() != n {
        // E.g. a fit from sufficient statistics, which keeps no residuals.
        return Err(RegressionError::invalid_input(
            "HC standard errors need one residual per row of x",
        ));
    }
    if aliased.len() != x.ncols() {
        return Err(RegressionError::invalid_parameter(
            "aliased",
            "needs one entry per column of x",
        ));
    }
    if weights.is_some_and(|w| w.nrows() != n) {
        return Err(RegressionError::invalid_parameter(
            "weights",
            "needs one entry per row of x",
        ));
    }
    let design = reduced_design(x, aliased, with_intercept);
    let k = design.ncols();
    if n <= k {
        return Err(RegressionError::invalid_input(
            "not enough observations for HC standard errors",
        ));
    }
    let w: Vec<f64> = (0..n).map(|i| weights.map_or(1.0, |w| w[i])).collect();
    let xw = Mat::from_fn(n, k, |i, j| w[i] * design[(i, j)]);
    let bread = super::prediction::compute_matrix_inverse(&(design.transpose() * &xw))?;
    let leverage = |i: usize| -> f64 {
        let mut h = 0.0;
        for a in 0..k {
            for b in 0..k {
                h += design[(i, a)] * bread[(a, b)] * design[(i, b)];
            }
        }
        w[i] * h
    };
    let hc1 = n as f64 / (n - k) as f64;
    let omega: Vec<f64> = (0..n)
        .map(|i| {
            let e_sq = (w[i] * residuals[i]).powi(2);
            match hc_type {
                HcType::HC0 => e_sq,
                HcType::HC1 => hc1 * e_sq,
                HcType::HC2 | HcType::HC3 => {
                    let h = leverage(i);
                    // HC0 weight when h_ii >= 1 (or NaN), for numerical safety
                    if h.is_nan() || h >= 1.0 {
                        e_sq
                    } else if hc_type == HcType::HC2 {
                        e_sq / (1.0 - h)
                    } else {
                        e_sq / ((1.0 - h) * (1.0 - h))
                    }
                }
            }
        })
        .collect();
    let xo = Mat::from_fn(n, k, |i, j| omega[i] * design[(i, j)]);
    let meat = design.transpose() * &xo;
    let reduced = &bread * &meat * &bread;
    Ok(crate::inference::expand_reduced_factor(
        &reduced,
        aliased,
        with_intercept,
    ))
}

/// Design with a leading intercept column (if any) and the non-aliased
/// columns of `x`.
fn reduced_design(x: &Mat<f64>, aliased: &[bool], with_intercept: bool) -> Mat<f64> {
    let kept: Vec<usize> = (0..x.ncols()).filter(|&j| !aliased[j]).collect();
    let off = usize::from(with_intercept);
    Mat::from_fn(x.nrows(), kept.len() + off, |i, j| {
        if j < off {
            1.0
        } else {
            x[(i, kept[j - off])]
        }
    })
}

/// Compute full HC inference (SE, t-stats, p-values, confidence intervals).
///
/// # Arguments
/// * `x` - Feature matrix (n × p), WITHOUT the intercept column
/// * `coefficients` - OLS coefficient estimates
/// * `intercept` - Intercept estimate (if model has one)
/// * `residuals` - OLS residuals
/// * `aliased` - Boolean mask for aliased columns
/// * `with_intercept` - Whether the model includes an intercept
/// * `hc_type` - Which HC variant to compute
/// * `confidence_level` - Confidence level for intervals (e.g., 0.95)
///
/// # Returns
/// `HcInference` with full inference statistics.
#[allow(clippy::too_many_arguments)]
pub fn compute_hc_inference(
    x: &Mat<f64>,
    coefficients: &Col<f64>,
    intercept: Option<f64>,
    residuals: &Col<f64>,
    aliased: &[bool],
    with_intercept: bool,
    hc_type: HcType,
    confidence_level: f64,
) -> Result<HcInference, RegressionError> {
    let hc_result = compute_hc_standard_errors(x, residuals, aliased, with_intercept, hc_type)?;

    let n = x.nrows();
    let p = aliased.iter().filter(|&&a| !a).count() + if with_intercept { 1 } else { 0 };
    let df = (n - p) as f64;

    // Compute t-statistics and p-values using existing CoefficientInference utilities
    let t_stats = CoefficientInference::t_statistics(coefficients, &hc_result.std_errors);
    let p_values = CoefficientInference::p_values(&t_stats, df);
    let (ci_lower, ci_upper) = CoefficientInference::confidence_intervals(
        coefficients,
        &hc_result.std_errors,
        df,
        confidence_level,
    );

    // Intercept inference
    let intercept_inference = match (intercept, hc_result.intercept_std_error) {
        (Some(int_val), Some(int_se)) => {
            let t_int = if int_se > 0.0 {
                int_val / int_se
            } else {
                f64::NAN
            };

            let t_dist = StudentsT::new(0.0, 1.0, df).ok();
            let p_int = if t_int.is_finite() {
                t_dist.map_or(f64::NAN, |d| 2.0 * d.sf(t_int.abs()))
            } else {
                f64::NAN
            };

            let t_crit = t_dist.map_or(f64::NAN, |d| {
                d.inverse_cdf(1.0 - (1.0 - confidence_level) / 2.0)
            });
            let ci = (int_val - t_crit * int_se, int_val + t_crit * int_se);

            Some(HcInterceptInference {
                std_error: int_se,
                t_statistic: t_int,
                p_value: p_int,
                conf_interval: ci,
            })
        }
        _ => None,
    };

    Ok(HcInference {
        hc_type,
        std_errors: hc_result.std_errors,
        t_statistics: t_stats,
        p_values,
        conf_interval_lower: ci_lower,
        conf_interval_upper: ci_upper,
        confidence_level,
        intercept: intercept_inference,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hc_type_default() {
        assert_eq!(HcType::default(), HcType::HC1);
    }

    #[test]
    fn test_hc0_simple() {
        // y = 2 + 3*x with heteroskedastic noise
        let x_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let y_data = [5.1, 7.8, 11.3, 13.6, 17.5, 19.8, 23.7, 25.2, 29.9, 31.4];

        let n = x_data.len();
        let x = Mat::from_fn(n, 1, |i, _| x_data[i]);
        let _y = Col::from_fn(n, |i| y_data[i]);

        // Fit OLS manually: compute coefficients
        let x_mean: f64 = x_data.iter().sum::<f64>() / n as f64;
        let y_mean: f64 = y_data.iter().sum::<f64>() / n as f64;

        let mut ss_xy = 0.0;
        let mut ss_xx = 0.0;
        for i in 0..n {
            ss_xy += (x_data[i] - x_mean) * (y_data[i] - y_mean);
            ss_xx += (x_data[i] - x_mean) * (x_data[i] - x_mean);
        }
        let beta1 = ss_xy / ss_xx;
        let beta0 = y_mean - beta1 * x_mean;

        let residuals = Col::from_fn(n, |i| y_data[i] - beta0 - beta1 * x_data[i]);
        let aliased = vec![false];

        let result =
            compute_hc_standard_errors(&x, &residuals, &aliased, true, HcType::HC0).unwrap();

        // HC0 SE should be positive and finite
        assert!(result.std_errors[0] > 0.0);
        assert!(result.std_errors[0].is_finite());
        assert!(result.intercept_std_error.unwrap() > 0.0);
    }

    #[test]
    fn test_hc_variants_ordering() {
        // HC3 >= HC2 >= HC0 (generally, for non-extreme leverage)
        let x_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let y_data = [5.1, 7.8, 11.3, 13.6, 17.5, 19.8, 23.7, 25.2, 29.9, 31.4];

        let n = x_data.len();
        let x = Mat::from_fn(n, 1, |i, _| x_data[i]);

        let x_mean: f64 = x_data.iter().sum::<f64>() / n as f64;
        let y_mean: f64 = y_data.iter().sum::<f64>() / n as f64;
        let mut ss_xy = 0.0;
        let mut ss_xx = 0.0;
        for i in 0..n {
            ss_xy += (x_data[i] - x_mean) * (y_data[i] - y_mean);
            ss_xx += (x_data[i] - x_mean) * (x_data[i] - x_mean);
        }
        let beta1 = ss_xy / ss_xx;
        let beta0 = y_mean - beta1 * x_mean;
        let residuals = Col::from_fn(n, |i| y_data[i] - beta0 - beta1 * x_data[i]);
        let aliased = vec![false];

        let hc0 = compute_hc_standard_errors(&x, &residuals, &aliased, true, HcType::HC0).unwrap();
        let hc2 = compute_hc_standard_errors(&x, &residuals, &aliased, true, HcType::HC2).unwrap();
        let hc3 = compute_hc_standard_errors(&x, &residuals, &aliased, true, HcType::HC3).unwrap();

        // HC3 >= HC2 >= HC0 for the slope SE
        assert!(
            hc3.std_errors[0] >= hc2.std_errors[0] - 1e-10,
            "HC3 ({}) should be >= HC2 ({})",
            hc3.std_errors[0],
            hc2.std_errors[0]
        );
        assert!(
            hc2.std_errors[0] >= hc0.std_errors[0] - 1e-10,
            "HC2 ({}) should be >= HC0 ({})",
            hc2.std_errors[0],
            hc0.std_errors[0]
        );
    }

    #[test]
    fn test_hc_no_intercept() {
        let x = Mat::from_fn(10, 1, |i, _| (i + 1) as f64);
        let residuals = Col::from_fn(10, |i| (-1.0_f64).powi(i as i32) * 0.5);
        let aliased = vec![false];

        let result =
            compute_hc_standard_errors(&x, &residuals, &aliased, false, HcType::HC1).unwrap();

        assert!(result.std_errors[0] > 0.0);
        assert!(result.std_errors[0].is_finite());
        assert!(result.intercept_std_error.is_none());
    }

    #[test]
    fn test_hc_inference() {
        let x_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let y_data = [5.1, 7.8, 11.3, 13.6, 17.5, 19.8, 23.7, 25.2, 29.9, 31.4];

        let n = x_data.len();
        let x = Mat::from_fn(n, 1, |i, _| x_data[i]);

        let x_mean: f64 = x_data.iter().sum::<f64>() / n as f64;
        let y_mean: f64 = y_data.iter().sum::<f64>() / n as f64;
        let mut ss_xy = 0.0;
        let mut ss_xx = 0.0;
        for i in 0..n {
            ss_xy += (x_data[i] - x_mean) * (y_data[i] - y_mean);
            ss_xx += (x_data[i] - x_mean) * (x_data[i] - x_mean);
        }
        let beta1 = ss_xy / ss_xx;
        let beta0 = y_mean - beta1 * x_mean;

        let coefficients = Col::from_fn(1, |_| beta1);
        let residuals = Col::from_fn(n, |i| y_data[i] - beta0 - beta1 * x_data[i]);
        let aliased = vec![false];

        let inf = compute_hc_inference(
            &x,
            &coefficients,
            Some(beta0),
            &residuals,
            &aliased,
            true,
            HcType::HC1,
            0.95,
        )
        .unwrap();

        // Check inference fields are populated and sensible
        assert_eq!(inf.hc_type, HcType::HC1);
        assert!(inf.std_errors[0] > 0.0);
        assert!(inf.t_statistics[0].is_finite());
        assert!(inf.p_values[0] >= 0.0 && inf.p_values[0] <= 1.0);
        assert!(inf.conf_interval_lower[0] < inf.conf_interval_upper[0]);
        assert_eq!(inf.confidence_level, 0.95);

        let int_inf = inf.intercept.unwrap();
        assert!(int_inf.std_error > 0.0);
        assert!(int_inf.t_statistic.is_finite());
        assert!(int_inf.p_value >= 0.0 && int_inf.p_value <= 1.0);
        assert!(int_inf.conf_interval.0 < int_inf.conf_interval.1);
    }
}
