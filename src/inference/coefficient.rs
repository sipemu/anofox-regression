//! Coefficient inference calculations.

use faer::{Col, Mat};
use statrs::distribution::{ContinuousCDF, StudentsT};

/// Computes inference statistics for regression coefficients.
pub struct CoefficientInference;

impl CoefficientInference {
    /// Compute standard errors for OLS coefficients.
    ///
    /// SE(β_j) = sqrt(σ² * (X'X)^(-1)_{jj})
    pub fn standard_errors(
        x: &Mat<f64>,
        mse: f64,
        aliased: &[bool],
    ) -> Result<Col<f64>, &'static str> {
        let n_features = x.ncols();
        let mut se = Col::zeros(n_features);

        // Compute X'X inverse for non-aliased columns
        let xtx_inv = Self::compute_xtx_inverse(x, aliased)?;

        for j in 0..n_features {
            if aliased[j] {
                se[j] = f64::NAN;
            } else {
                let var = mse * xtx_inv[(j, j)];
                se[j] = if var >= 0.0 { var.sqrt() } else { f64::NAN };
            }
        }

        Ok(se)
    }

    /// Compute t-statistics for coefficients.
    ///
    /// t_j = β_j / SE(β_j)
    pub fn t_statistics(coefficients: &Col<f64>, std_errors: &Col<f64>) -> Col<f64> {
        let n = coefficients.nrows();
        let mut t_stats = Col::zeros(n);

        for j in 0..n {
            if std_errors[j].is_nan() || std_errors[j] == 0.0 {
                t_stats[j] = f64::NAN;
            } else {
                t_stats[j] = coefficients[j] / std_errors[j];
            }
        }

        t_stats
    }

    /// Compute p-values from t-statistics.
    ///
    /// p_j = 2 * P(|T| > |t_j|) where T ~ t(df)
    pub fn p_values(t_statistics: &Col<f64>, df: f64) -> Col<f64> {
        let n = t_statistics.nrows();
        let mut p_vals = Col::zeros(n);

        if df <= 0.0 {
            for j in 0..n {
                p_vals[j] = f64::NAN;
            }
            return p_vals;
        }

        let t_dist = StudentsT::new(0.0, 1.0, df).expect("valid t-distribution parameters");

        for j in 0..n {
            if t_statistics[j].is_nan() {
                p_vals[j] = f64::NAN;
            } else {
                // Two-tailed test
                let abs_t = t_statistics[j].abs();
                p_vals[j] = 2.0 * (1.0 - t_dist.cdf(abs_t));
            }
        }

        p_vals
    }

    /// Compute confidence intervals for coefficients.
    ///
    /// CI_j = β_j ± t_{α/2, df} * SE(β_j)
    pub fn confidence_intervals(
        coefficients: &Col<f64>,
        std_errors: &Col<f64>,
        df: f64,
        confidence_level: f64,
    ) -> (Col<f64>, Col<f64>) {
        let n = coefficients.nrows();
        let mut lower = Col::zeros(n);
        let mut upper = Col::zeros(n);

        if df <= 0.0 {
            for j in 0..n {
                lower[j] = f64::NAN;
                upper[j] = f64::NAN;
            }
            return (lower, upper);
        }

        let t_dist = StudentsT::new(0.0, 1.0, df).expect("valid t-distribution parameters");
        let alpha = 1.0 - confidence_level;
        let t_crit = t_dist.inverse_cdf(1.0 - alpha / 2.0);

        for j in 0..n {
            if std_errors[j].is_nan() {
                lower[j] = f64::NAN;
                upper[j] = f64::NAN;
            } else {
                let margin = t_crit * std_errors[j];
                lower[j] = coefficients[j] - margin;
                upper[j] = coefficients[j] + margin;
            }
        }

        (lower, upper)
    }

    /// Compute standard errors for both intercept and coefficients using the augmented design matrix.
    ///
    /// This is the proper way to compute SE for models with intercept, matching R's `lm()`.
    /// Uses the augmented design matrix [1 | X] to compute (X_aug'X_aug)^-1.
    ///
    /// Returns (coefficient_SE, intercept_SE).
    pub fn standard_errors_with_intercept(
        x: &Mat<f64>,
        mse: f64,
        aliased: &[bool],
    ) -> Result<(Col<f64>, f64), &'static str> {
        // (X_aug'X_aug)^-1 over the intercept and the non-aliased columns only:
        // aliased columns are not part of the estimated model (R's lm reports
        // the standard errors of the reduced fit).
        let inv = super::prediction::compute_xtx_inverse_augmented_reduced(x, aliased)?;
        Ok(Self::expand_augmented_se(&inv, mse, aliased))
    }

    /// Compute standard errors for WLS with intercept using the weighted augmented design matrix.
    ///
    /// This is the proper way to compute SE for WLS with intercept, matching R's `lm()` with weights.
    /// Uses the weighted augmented design matrix [1 | X] to compute (X_aug'WX_aug)^-1.
    ///
    /// Returns (coefficient_SE, intercept_SE).
    pub fn standard_errors_wls_with_intercept(
        x: &Mat<f64>,
        weights: &Col<f64>,
        mse: f64,
        aliased: &[bool],
    ) -> Result<(Col<f64>, f64), &'static str> {
        let inv = super::prediction::compute_xtwx_inverse_augmented_reduced(x, weights, aliased)?;
        Ok(Self::expand_augmented_se(&inv, mse, aliased))
    }

    /// Map the diagonal of a reduced augmented inverse (intercept first, then
    /// the non-aliased columns in order) to `(coefficient_SE, intercept_SE)`.
    fn expand_augmented_se(inv: &Mat<f64>, mse: f64, aliased: &[bool]) -> (Col<f64>, f64) {
        let sd = |v: f64| {
            let var = mse * v;
            if var >= 0.0 {
                var.sqrt()
            } else {
                f64::NAN
            }
        };
        let se_intercept = sd(inv[(0, 0)]);
        let mut se_coef = Col::zeros(aliased.len());
        let mut k = 1;
        for (j, &a) in aliased.iter().enumerate() {
            if a {
                se_coef[j] = f64::NAN;
            } else {
                se_coef[j] = sd(inv[(k, k)]);
                k += 1;
            }
        }
        (se_coef, se_intercept)
    }

    /// Compute (X'X)^(-1) for non-aliased columns.
    fn compute_xtx_inverse(x: &Mat<f64>, aliased: &[bool]) -> Result<Mat<f64>, &'static str> {
        let n_features = x.ncols();
        let inv_active = super::prediction::compute_xtx_inverse_reduced(x, aliased)?;
        let active: Vec<usize> = (0..n_features).filter(|&j| !aliased[j]).collect();
        let mut xtx_inv = Mat::zeros(n_features, n_features);
        for (ai, &i) in active.iter().enumerate() {
            for (aj, &j) in active.iter().enumerate() {
                xtx_inv[(i, j)] = inv_active[(ai, aj)];
            }
        }
        Ok(xtx_inv)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_t_statistics() {
        let coefficients = Col::from_fn(3, |i| (i + 1) as f64);
        let std_errors = Col::from_fn(3, |_| 0.5);

        let t_stats = CoefficientInference::t_statistics(&coefficients, &std_errors);

        assert!((t_stats[0] - 2.0).abs() < 1e-10);
        assert!((t_stats[1] - 4.0).abs() < 1e-10);
        assert!((t_stats[2] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_p_values_bounds() {
        let t_stats = Col::from_fn(3, |i| (i + 1) as f64);
        let p_vals = CoefficientInference::p_values(&t_stats, 10.0);

        for p in p_vals.iter() {
            assert!(*p >= 0.0 && *p <= 1.0);
        }
    }
}
