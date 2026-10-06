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
        let n_samples = x.nrows();
        let n_features = x.ncols();

        // Build augmented design matrix [1 | X]
        let mut x_aug = Mat::zeros(n_samples, n_features + 1);
        for i in 0..n_samples {
            x_aug[(i, 0)] = 1.0;
            for j in 0..n_features {
                x_aug[(i, j + 1)] = x[(i, j)];
            }
        }

        // Compute X_aug'X_aug
        let xtx_aug = x_aug.transpose() * &x_aug;

        // Compute inverse using QR decomposition
        let qr = xtx_aug.qr();
        let q = qr.compute_thin_Q();
        let r = qr.R();

        // Check if R is singular
        let aug_size = n_features + 1;
        for i in 0..aug_size {
            if r[(i, i)].abs() < 1e-10 {
                return Err("Augmented matrix is singular");
            }
        }

        // Solve R * X = Q' for each column of identity to get inverse
        let mut xtx_aug_inv = Mat::zeros(aug_size, aug_size);
        let qt = q.transpose();

        for col in 0..aug_size {
            for i in (0..aug_size).rev() {
                let mut sum = qt[(i, col)];
                for j in (i + 1)..aug_size {
                    sum -= r[(i, j)] * xtx_aug_inv[(j, col)];
                }
                xtx_aug_inv[(i, col)] = sum / r[(i, i)];
            }
        }

        // Intercept SE from (0,0) element
        let se_intercept = (mse * xtx_aug_inv[(0, 0)]).sqrt();

        // Coefficient SEs from diagonal (1..n_features+1), respecting aliased
        let mut se_coef = Col::zeros(n_features);
        for j in 0..n_features {
            if aliased[j] {
                se_coef[j] = f64::NAN;
            } else {
                let var = mse * xtx_aug_inv[(j + 1, j + 1)];
                se_coef[j] = if var >= 0.0 { var.sqrt() } else { f64::NAN };
            }
        }

        Ok((se_coef, se_intercept))
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
        let n_samples = x.nrows();
        let n_features = x.ncols();

        // Build X'WX for augmented matrix [1 | X]
        let aug_size = n_features + 1;
        let mut xtwx_aug: Mat<f64> = Mat::zeros(aug_size, aug_size);

        for i in 0..n_samples {
            let w = weights[i];

            // (0,0): sum of weights
            xtwx_aug[(0, 0)] += w;

            // (0,j+1) and (j+1,0): weighted sum of x_j
            for j in 0..n_features {
                xtwx_aug[(0, j + 1)] += w * x[(i, j)];
                xtwx_aug[(j + 1, 0)] += w * x[(i, j)];
            }

            // (j+1, k+1): weighted x_j * x_k
            for j in 0..n_features {
                for k in 0..n_features {
                    xtwx_aug[(j + 1, k + 1)] += w * x[(i, j)] * x[(i, k)];
                }
            }
        }

        // Compute inverse using QR decomposition
        let qr: faer::linalg::solvers::Qr<f64> = xtwx_aug.qr();
        let q = qr.compute_thin_Q();
        let r = qr.R();

        // Check if R is singular
        for i in 0..aug_size {
            if r[(i, i)].abs() < 1e-10 {
                return Err("Weighted augmented matrix is singular");
            }
        }

        // Solve R * X = Q' for each column of identity to get inverse
        let mut xtwx_aug_inv: Mat<f64> = Mat::zeros(aug_size, aug_size);
        let qt = q.transpose();

        for col in 0..aug_size {
            for i in (0..aug_size).rev() {
                let mut sum = qt[(i, col)];
                for j in (i + 1)..aug_size {
                    sum -= r[(i, j)] * xtwx_aug_inv[(j, col)];
                }
                xtwx_aug_inv[(i, col)] = sum / r[(i, i)];
            }
        }

        // Intercept SE from (0,0) element
        let se_intercept = (mse * xtwx_aug_inv[(0, 0)]).sqrt();

        // Coefficient SEs from diagonal (1..n_features+1), respecting aliased
        let mut se_coef = Col::zeros(n_features);
        for j in 0..n_features {
            if aliased[j] {
                se_coef[j] = f64::NAN;
            } else {
                let var = mse * xtwx_aug_inv[(j + 1, j + 1)];
                se_coef[j] = if var >= 0.0 { var.sqrt() } else { f64::NAN };
            }
        }

        Ok((se_coef, se_intercept))
    }

    /// Compute (X'X)^(-1) for non-aliased columns.
    fn compute_xtx_inverse(x: &Mat<f64>, aliased: &[bool]) -> Result<Mat<f64>, &'static str> {
        let n_features = x.ncols();
        let n_active: usize = aliased.iter().filter(|&&a| !a).count();

        if n_active == 0 {
            return Err("All features are aliased");
        }

        // Extract non-aliased columns
        let mut x_active = Mat::zeros(x.nrows(), n_active);
        let mut col_idx = 0;
        for j in 0..n_features {
            if !aliased[j] {
                for i in 0..x.nrows() {
                    x_active[(i, col_idx)] = x[(i, j)];
                }
                col_idx += 1;
            }
        }

        // Compute X'X
        let xtx = x_active.transpose() * &x_active;

        // Compute inverse using QR decomposition (more numerically stable)
        let qr = xtx.qr();
        let q = qr.compute_thin_Q();
        let r = qr.R();

        // Check if R is singular
        for i in 0..n_active {
            if r[(i, i)].abs() < 1e-10 {
                return Err("Matrix is singular");
            }
        }

        // Solve R * X = Q' for each column of identity to get inverse
        let mut xtx_inv_active = Mat::zeros(n_active, n_active);
        let qt = q.transpose();

        for col in 0..n_active {
            // Back-substitution for R * x = qt_col
            for i in (0..n_active).rev() {
                let mut sum = qt[(i, col)];
                for j in (i + 1)..n_active {
                    sum -= r[(i, j)] * xtx_inv_active[(j, col)];
                }
                xtx_inv_active[(i, col)] = sum / r[(i, i)];
            }
        }

        // Map back to full size
        let mut xtx_inv = Mat::zeros(n_features, n_features);
        let mut ai = 0;
        for i in 0..n_features {
            if aliased[i] {
                continue;
            }
            let mut aj = 0;
            for j in 0..n_features {
                if aliased[j] {
                    continue;
                }
                xtx_inv[(i, j)] = xtx_inv_active[(ai, aj)];
                aj += 1;
            }
            ai += 1;
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
