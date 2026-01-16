//! Negative Binomial regression solver.
//!
//! Implements GLM with Negative Binomial family for overdispersed count data
//! using Iteratively Reweighted Least Squares (IRLS).
//!
//! # Theta Estimation
//!
//! - Fixed: User specifies theta
//! - Alternating: Iterate between estimating β (IRLS) and θ (ML)
//!
//! # Example
//!
//! ```rust,ignore
//! use anofox_regression::solvers::{NegativeBinomialRegressor, Regressor, FittedRegressor};
//! use faer::{Mat, Col};
//!
//! let x = Mat::from_fn(100, 2, |i, j| (i + j) as f64);
//! let y = Col::from_fn(100, |i| (i % 10) as f64);  // overdispersed count data
//!
//! // Negative binomial with estimated theta
//! let fitted = NegativeBinomialRegressor::builder()
//!     .estimate_theta(true)
//!     .build()
//!     .fit(&x, &y)?;
//!
//! println!("Theta: {}", fitted.theta);
//! ```

use crate::core::{
    estimate_theta_ml, GlmFamily, IntervalType, NegativeBinomialFamily, PredictionResult,
    PredictionType, RegressionOptions, RegressionOptionsBuilder, RegressionResult,
};
use crate::diagnostics::{deviance_residuals, pearson_residuals, working_residuals};
use crate::solvers::traits::{FittedRegressor, RegressionError, Regressor};
use crate::utils::detect_constant_columns;
use faer::{Col, Mat};
use statrs::distribution::{ContinuousCDF, FisherSnedecor, Normal};

/// Negative Binomial GLM regression estimator.
///
/// Fits a generalized linear model with Negative Binomial family using IRLS
/// (Iteratively Reweighted Least Squares). Supports both fixed and estimated theta.
///
/// # Model
///
/// The Negative Binomial GLM models:
/// - `E\[Y\] = μ = exp(Xβ)` (log link)
/// - `Var\[Y\] = μ + μ²/θ` (quadratic variance function)
///
/// As θ → ∞, this approaches Poisson (Var\[Y\] = μ).
#[derive(Debug, Clone)]
pub struct NegativeBinomialRegressor {
    options: RegressionOptions,
    family: NegativeBinomialFamily,
    offset: Option<Col<f64>>,
    estimate_theta: bool,
    theta_max_iter: usize,
    theta_tol: f64,
}

impl NegativeBinomialRegressor {
    /// Create a new Negative Binomial regressor with the given options and family.
    pub fn new(options: RegressionOptions, family: NegativeBinomialFamily) -> Self {
        Self {
            options,
            family,
            offset: None,
            estimate_theta: true,
            theta_max_iter: 25,
            theta_tol: 1e-6,
        }
    }

    /// Create a builder for Negative Binomial regression.
    pub fn builder() -> NegativeBinomialRegressorBuilder {
        NegativeBinomialRegressorBuilder::default()
    }

    /// Create a builder with a fixed theta value.
    pub fn with_theta(theta: f64) -> NegativeBinomialRegressorBuilder {
        NegativeBinomialRegressorBuilder::default()
            .theta(theta)
            .estimate_theta(false)
    }

    /// Fit the GLM using IRLS with optional theta estimation.
    fn fit_irls(
        &self,
        x: &Mat<f64>,
        y: &Col<f64>,
    ) -> Result<FittedNegativeBinomial, RegressionError> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        let n_params = if self.options.with_intercept {
            n_features + 1
        } else {
            n_features
        };

        // Build design matrix
        let x_design = if self.options.with_intercept {
            let mut x_aug = Mat::zeros(n_samples, n_features + 1);
            for i in 0..n_samples {
                x_aug[(i, 0)] = 1.0;
                for j in 0..n_features {
                    x_aug[(i, j + 1)] = x[(i, j)];
                }
            }
            x_aug
        } else {
            x.clone()
        };

        // Detect constant/collinear columns
        let aliased = detect_constant_columns(x, self.options.rank_tolerance);

        let y_vec: Vec<f64> = (0..n_samples).map(|i| y[i]).collect();
        let mut family = self.family;
        let mut theta = family.theta;

        let mut beta = Col::zeros(n_params);
        let mut mu: Vec<f64> = family.initialize_mu(&y_vec);

        let max_outer_iter = if self.estimate_theta { 25 } else { 1 };
        let mut total_iterations = 0;
        let mut converged = false;

        for _outer in 0..max_outer_iter {
            // Inner IRLS loop for given theta
            let mut eta: Vec<f64> = mu
                .iter()
                .enumerate()
                .map(|(i, &m)| {
                    let base_eta = family.link(m);
                    if let Some(ref offset) = self.offset {
                        base_eta - offset[i]
                    } else {
                        base_eta
                    }
                })
                .collect();

            let max_iter = self.options.max_iterations;
            let tol = self.options.tolerance;
            let mut inner_converged = false;

            for _iter in 0..max_iter {
                total_iterations += 1;

                // Compute working weights and response
                let (weights, z) = self.compute_irls_quantities(&family, &y_vec, &mu, &eta);

                // Solve weighted least squares
                let beta_new = self.solve_weighted_ls(&x_design, &z, &weights)?;

                // Store old beta for step-halving
                let beta_old = beta.clone();

                // Check convergence based on coefficient change
                let max_change: f64 = beta_new
                    .iter()
                    .zip(beta_old.iter())
                    .map(|(&b_new, &b_old)| {
                        let diff: f64 = b_new - b_old;
                        diff.abs()
                    })
                    .fold(0.0_f64, f64::max);

                // Update coefficients
                beta = beta_new;

                // Update η and μ with validity checks
                self.update_eta_mu_with_family(
                    &family, &x_design, &beta, &mut eta, &mut mu, n_samples, n_params,
                )?;

                // Check convergence based on coefficient change
                if max_change < tol {
                    inner_converged = true;
                    break;
                }
            }

            if !inner_converged && !self.estimate_theta {
                return Err(RegressionError::ConvergenceFailed {
                    iterations: total_iterations,
                });
            }

            // Update theta if estimating
            if self.estimate_theta {
                let old_theta = theta;
                theta = estimate_theta_ml(&y_vec, &mu, self.theta_max_iter, self.theta_tol);
                theta = theta.clamp(0.01, 1e8);
                family = NegativeBinomialFamily::new(theta);

                // Check outer convergence
                if (theta - old_theta).abs() < self.theta_tol * old_theta.max(1.0)
                    && inner_converged
                {
                    converged = true;
                    break;
                }
            } else {
                converged = inner_converged;
                break;
            }
        }

        if !converged {
            return Err(RegressionError::ConvergenceFailed {
                iterations: total_iterations,
            });
        }

        self.build_result(
            x,
            y,
            &x_design,
            &beta,
            &mu,
            n_params,
            total_iterations,
            family,
            aliased,
        )
    }

    fn compute_irls_quantities(
        &self,
        family: &NegativeBinomialFamily,
        y: &[f64],
        mu: &[f64],
        eta: &[f64],
    ) -> (Vec<f64>, Vec<f64>) {
        let n = y.len();
        let mut weights = vec![0.0; n];
        let mut z = vec![0.0; n];

        for i in 0..n {
            weights[i] = family.irls_weight(mu[i]);
            let eta_no_offset = if let Some(ref offset) = self.offset {
                eta[i] - offset[i]
            } else {
                eta[i]
            };
            z[i] = eta_no_offset + (y[i] - mu[i]) * family.link_derivative(mu[i]);
        }

        (weights, z)
    }

    /// Update η and μ from current coefficients with validity checks.
    #[allow(clippy::too_many_arguments)]
    fn update_eta_mu_with_family(
        &self,
        family: &NegativeBinomialFamily,
        x_design: &Mat<f64>,
        beta: &Col<f64>,
        eta: &mut [f64],
        mu: &mut [f64],
        n_samples: usize,
        n_params: usize,
    ) -> Result<(), RegressionError> {
        for i in 0..n_samples {
            let mut eta_i = 0.0;
            for j in 0..n_params {
                eta_i += x_design[(i, j)] * beta[j];
            }
            // Add offset if present
            if let Some(ref offset) = self.offset {
                eta_i += offset[i];
            }
            eta[i] = eta_i;

            // Check eta validity (following R's valideta)
            if !family.valid_eta(eta_i) {
                return Err(RegressionError::NumericalError(
                    "Invalid linear predictor (eta) during IRLS: non-finite value".to_string(),
                ));
            }

            let mu_new = family.link_inverse(eta_i);

            // Check mu validity and clamp if needed (following R's validmu)
            if !family.valid_mu(mu_new) {
                mu[i] = family.clamp_mu(mu_new);
            } else {
                mu[i] = mu_new;
            }
        }
        Ok(())
    }

    fn solve_weighted_ls(
        &self,
        x: &Mat<f64>,
        z: &[f64],
        weights: &[f64],
    ) -> Result<Col<f64>, RegressionError> {
        let n_samples = x.nrows();
        let n_params = x.ncols();

        // Check if we need penalized IRLS
        let lambda = self.options.lambda;

        if lambda > 0.0 {
            // Penalized IRLS: solve (X'WX + λI)β = X'Wz
            // Note: Do not penalize intercept (first column if with_intercept)
            let mut xtwx: Mat<f64> = Mat::zeros(n_params, n_params);
            let mut xtwz: Col<f64> = Col::zeros(n_params);

            // Compute X'WX and X'Wz
            for i in 0..n_samples {
                let w = weights[i];
                for j in 0..n_params {
                    xtwz[j] += w * x[(i, j)] * z[i];
                    for k in 0..n_params {
                        xtwx[(j, k)] += w * x[(i, j)] * x[(i, k)];
                    }
                }
            }

            // Add λI to the diagonal (skip intercept at index 0 if with_intercept)
            let start_idx = if self.options.with_intercept { 1 } else { 0 };
            for j in start_idx..n_params {
                xtwx[(j, j)] += lambda;
            }

            // Solve using QR decomposition
            let qr = xtwx.qr();
            let q = qr.compute_Q();
            let r: Mat<f64> = qr.R().to_owned();

            // Solve R β = Q' (X'Wz)
            let qty = q.transpose() * &xtwz;

            let mut beta: Col<f64> = Col::zeros(n_params);
            for i in (0..n_params).rev() {
                let mut sum: f64 = qty[i];
                for j in (i + 1)..n_params {
                    sum -= r[(i, j)] * beta[j];
                }
                if r[(i, i)].abs() > self.options.rank_tolerance {
                    beta[i] = sum / r[(i, i)];
                } else {
                    beta[i] = 0.0;
                }
            }

            Ok(beta)
        } else {
            // Standard IRLS: solve using QR on weighted design matrix
            let mut x_weighted = Mat::zeros(n_samples, n_params);
            let mut z_weighted = Col::zeros(n_samples);

            for i in 0..n_samples {
                let sqrt_w = weights[i].sqrt();
                for j in 0..n_params {
                    x_weighted[(i, j)] = sqrt_w * x[(i, j)];
                }
                z_weighted[i] = sqrt_w * z[i];
            }

            let qr = x_weighted.col_piv_qr();
            let q = qr.compute_Q();
            let r = qr.R();
            let perm = qr.P();

            let qtz = q.transpose() * z_weighted;

            let mut beta_perm = Col::zeros(n_params);
            for i in (0..n_params).rev() {
                let mut sum = qtz[i];
                for j in (i + 1)..n_params {
                    sum -= r[(i, j)] * beta_perm[j];
                }
                if r[(i, i)].abs() > self.options.rank_tolerance {
                    beta_perm[i] = sum / r[(i, i)];
                } else {
                    beta_perm[i] = 0.0;
                }
            }

            let mut beta = Col::zeros(n_params);
            for i in 0..n_params {
                beta[perm.inverse().arrays().0[i]] = beta_perm[i];
            }

            Ok(beta)
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn build_result(
        &self,
        x: &Mat<f64>,
        y: &Col<f64>,
        x_design: &Mat<f64>,
        beta: &Col<f64>,
        mu: &[f64],
        n_params: usize,
        iterations: usize,
        family: NegativeBinomialFamily,
        aliased: Vec<bool>,
    ) -> Result<FittedNegativeBinomial, RegressionError> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        let (intercept, coefficients) = if self.options.with_intercept {
            let int = Some(beta[0]);
            let coefs = Col::from_fn(n_features, |j| beta[j + 1]);
            (int, coefs)
        } else {
            (None, beta.clone())
        };

        let fitted_values = Col::from_fn(n_samples, |i| mu[i]);
        let residuals = Col::from_fn(n_samples, |i| y[i] - mu[i]);

        let y_vec: Vec<f64> = (0..n_samples).map(|i| y[i]).collect();
        let deviance = family.deviance(&y_vec, mu);
        let null_deviance = family.null_deviance(&y_vec);

        // Estimate dispersion (typically 1 for NB, but can estimate)
        let df_resid = (n_samples.saturating_sub(n_params)) as f64;
        let dispersion = if df_resid > 0.0 {
            let pearson_chi2: f64 = (0..n_samples)
                .map(|i| {
                    let v = family.variance(mu[i]);
                    (y[i] - mu[i]).powi(2) / v
                })
                .sum();
            (pearson_chi2 / df_resid).max(1.0)
        } else {
            1.0
        };

        let r_squared = if null_deviance > 0.0 {
            1.0 - deviance / null_deviance
        } else {
            f64::NAN
        };

        let df_total = (n_samples - 1) as f64;
        let adj_r_squared = if df_resid > 0.0 && df_total > 0.0 {
            1.0 - (1.0 - r_squared) * df_total / df_resid
        } else {
            f64::NAN
        };

        let rss: f64 = residuals.iter().map(|&r| r.powi(2)).sum();
        let mse = if df_resid > 0.0 {
            rss / df_resid
        } else {
            f64::NAN
        };
        let rmse = mse.sqrt();

        let df_model = (n_params - if intercept.is_some() { 1 } else { 0 }) as f64;
        let f_statistic = if df_model > 0.0 && df_resid > 0.0 {
            ((null_deviance - deviance) / df_model) / dispersion
        } else {
            f64::NAN
        };

        let f_pvalue = if f_statistic.is_finite() && df_model > 0.0 && df_resid > 0.0 {
            FisherSnedecor::new(df_model, df_resid)
                .map(|d| 1.0 - d.cdf(f_statistic))
                .unwrap_or(f64::NAN)
        } else {
            f64::NAN
        };

        let n = n_samples as f64;
        let k = (n_params + 1) as f64; // +1 for theta
        let log_likelihood = -deviance / 2.0;

        let aic = 2.0 * k - 2.0 * log_likelihood;
        let aicc = if (n - k - 1.0) > 0.0 {
            aic + 2.0 * k * (k + 1.0) / (n - k - 1.0)
        } else {
            f64::NAN
        };
        let bic = k * n.ln() - 2.0 * log_likelihood;

        let rank = n_params;

        let mut result = RegressionResult::empty(n_features, n_samples);
        result.coefficients = coefficients;
        result.intercept = intercept;
        result.residuals = residuals;
        result.fitted_values = fitted_values;
        result.rank = rank;
        result.n_parameters = n_params;
        result.n_observations = n_samples;
        result.aliased = aliased.clone();
        result.r_squared = r_squared;
        result.adj_r_squared = adj_r_squared;
        result.mse = mse;
        result.rmse = rmse;
        result.f_statistic = f_statistic;
        result.f_pvalue = f_pvalue;
        result.aic = aic;
        result.aicc = aicc;
        result.bic = bic;
        result.log_likelihood = log_likelihood;
        result.confidence_level = self.options.confidence_level;

        // Compute standard errors and (X'WX)⁻¹
        let mut xtwx_inverse = None;
        if self.options.compute_inference {
            if let Ok((se, xtwx_inv)) =
                self.compute_standard_errors_and_covariance(x_design, mu, dispersion, &family)
            {
                result.std_errors = Some(if self.options.with_intercept {
                    Col::from_fn(n_features, |j| se[j + 1])
                } else {
                    se.clone()
                });

                if self.options.with_intercept {
                    result.intercept_std_error = Some(se[0]);
                }

                // Compute z-statistics and p-values
                let t_stats = Col::from_fn(n_params, |j| beta[j] / se[j]);
                let p_vals = Col::from_fn(n_params, |j| {
                    let z = t_stats[j].abs();
                    2.0 * Normal::new(0.0, 1.0)
                        .map(|d| 1.0 - d.cdf(z))
                        .unwrap_or(f64::NAN)
                });

                result.t_statistics = Some(if self.options.with_intercept {
                    Col::from_fn(n_features, |j| t_stats[j + 1])
                } else {
                    t_stats.clone()
                });

                result.p_values = Some(if self.options.with_intercept {
                    Col::from_fn(n_features, |j| p_vals[j + 1])
                } else {
                    p_vals.clone()
                });

                if self.options.with_intercept {
                    result.intercept_t_statistic = Some(t_stats[0]);
                    result.intercept_p_value = Some(p_vals[0]);
                }

                xtwx_inverse = Some(xtwx_inv);
            }
        }

        Ok(FittedNegativeBinomial {
            result,
            options: self.options.clone(),
            family,
            deviance,
            null_deviance,
            dispersion,
            theta: family.theta,
            iterations,
            y_values: y.clone(),
            xtwx_inverse,
            offset: self.offset.clone(),
            aliased,
        })
    }

    fn compute_standard_errors_and_covariance(
        &self,
        x: &Mat<f64>,
        mu: &[f64],
        dispersion: f64,
        family: &NegativeBinomialFamily,
    ) -> Result<(Col<f64>, Mat<f64>), RegressionError> {
        let n_samples = x.nrows();
        let n_params = x.ncols();

        let mut xtwx: Mat<f64> = Mat::zeros(n_params, n_params);
        for i in 0..n_samples {
            let w = family.irls_weight(mu[i]);
            for j in 0..n_params {
                for k in 0..n_params {
                    xtwx[(j, k)] += w * x[(i, j)] * x[(i, k)];
                }
            }
        }

        let qr = xtwx.qr();
        let q = qr.compute_Q();
        let r = qr.R().to_owned();

        let mut xtwx_inv: Mat<f64> = Mat::zeros(n_params, n_params);
        for col in 0..n_params {
            let mut e = Col::zeros(n_params);
            e[col] = 1.0;
            let qte = q.transpose() * e;

            let mut sol = Col::zeros(n_params);
            for i in (0..n_params).rev() {
                let mut sum = qte[i];
                for j in (i + 1)..n_params {
                    sum -= r[(i, j)] * sol[j];
                }
                if r[(i, i)].abs() > 1e-14 {
                    sol[i] = sum / r[(i, i)];
                }
            }

            for i in 0..n_params {
                xtwx_inv[(i, col)] = sol[i];
            }
        }

        let se = Col::from_fn(n_params, |j| (dispersion * xtwx_inv[(j, j)]).sqrt());

        Ok((se, xtwx_inv))
    }
}

impl Regressor for NegativeBinomialRegressor {
    type Fitted = FittedNegativeBinomial;

    fn fit(&self, x: &Mat<f64>, y: &Col<f64>) -> Result<Self::Fitted, RegressionError> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        if x.nrows() != y.nrows() {
            return Err(RegressionError::DimensionMismatch {
                x_rows: x.nrows(),
                y_len: y.nrows(),
            });
        }

        if n_samples < 2 {
            return Err(RegressionError::InsufficientObservations {
                needed: 2,
                got: n_samples,
            });
        }

        let n_params = if self.options.with_intercept {
            n_features + 1
        } else {
            n_features
        };

        if n_samples < n_params {
            return Err(RegressionError::InsufficientObservations {
                needed: n_params,
                got: n_samples,
            });
        }

        // Check offset length if provided
        if let Some(ref offset) = self.offset {
            if offset.nrows() != n_samples {
                return Err(RegressionError::DimensionMismatch {
                    x_rows: n_samples,
                    y_len: offset.nrows(),
                });
            }
        }

        // Validate y values are non-negative
        for i in 0..n_samples {
            if y[i] < 0.0 {
                return Err(RegressionError::NumericalError(format!(
                    "y values must be non-negative for Negative Binomial, got y[{}] = {}",
                    i, y[i]
                )));
            }
        }

        self.fit_irls(x, y)
    }
}

/// Fitted Negative Binomial GLM model for overdispersed count data.
///
/// Contains the estimated coefficients, dispersion parameter (theta), and
/// model diagnostics from fitting a negative binomial regression using IRLS.
///
/// # Overdispersion
///
/// The negative binomial model is used when count data exhibits overdispersion
/// (variance greater than mean). The `theta` parameter controls the degree of
/// overdispersion: as theta increases, the model approaches Poisson.
///
/// # Available Methods
///
/// - [`predict`](FittedRegressor::predict) - Predict counts for new data
/// - [`predict_count`](Self::predict_count) - Alias for predict
/// - [`predict_linear`](Self::predict_linear) - Predict on link scale (log)
/// - [`predict_with_se`](Self::predict_with_se) - Predictions with standard errors
/// - [`overdispersion_ratio`](Self::overdispersion_ratio) - Get Var/Mean ratio
/// - [`pearson_residuals`](Self::pearson_residuals) - Pearson residuals
/// - [`deviance_residuals`](Self::deviance_residuals) - Deviance residuals
///
/// # Example
///
/// ```rust,ignore
/// use anofox_regression::prelude::*;
///
/// // Overdispersed count data
/// let x = Mat::from_fn(100, 2, |i, j| (i + j) as f64 / 10.0);
/// let y = Col::from_fn(100, |i| ((i % 10) as f64 * 2.0).round());
///
/// let fitted = NegativeBinomialRegressor::builder()
///     .with_intercept(true)
///     .estimate_theta(true)
///     .compute_inference(true)
///     .build()
///     .fit(&x, &y)?;
///
/// // Access estimated theta (dispersion parameter)
/// println!("Theta: {}", fitted.theta);
/// println!("Overdispersion ratio: {}", fitted.overdispersion_ratio());
///
/// // Make predictions
/// let counts = fitted.predict_count(&x_new);
/// ```
#[derive(Debug, Clone)]
pub struct FittedNegativeBinomial {
    result: RegressionResult,
    options: RegressionOptions,
    family: NegativeBinomialFamily,
    /// Total deviance.
    pub deviance: f64,
    /// Null deviance (intercept-only model).
    pub null_deviance: f64,
    /// Dispersion parameter.
    pub dispersion: f64,
    /// Estimated theta (size/dispersion parameter).
    pub theta: f64,
    /// Number of iterations.
    pub iterations: usize,
    /// Original y values.
    y_values: Col<f64>,
    /// (X'WX)⁻¹ matrix for prediction SE.
    xtwx_inverse: Option<Mat<f64>>,
    /// Offset used in fitting (stored for potential residual calculations).
    #[allow(dead_code)]
    offset: Option<Col<f64>>,
    /// Aliased (collinear or constant) columns.
    #[allow(dead_code)]
    aliased: Vec<bool>,
}

impl FittedNegativeBinomial {
    /// Get the Negative Binomial family used for this model.
    pub fn family(&self) -> &NegativeBinomialFamily {
        &self.family
    }

    /// Compute predicted counts (response scale).
    pub fn predict_count(&self, x: &Mat<f64>) -> Col<f64> {
        self.predict(x)
    }

    /// Compute predicted linear predictor (link scale).
    pub fn predict_linear(&self, x: &Mat<f64>) -> Col<f64> {
        let mu = self.predict(x);
        Col::from_fn(mu.nrows(), |i| self.family.link(mu[i]))
    }

    /// Compute Pearson residuals: (y - μ) / sqrt(V(μ)).
    pub fn pearson_residuals(&self) -> Col<f64> {
        let mu = &self.result.fitted_values;
        pearson_residuals(&self.y_values, mu, &self.family)
    }

    /// Compute deviance residuals: sign(y - μ) * sqrt(d_i).
    pub fn deviance_residuals(&self) -> Col<f64> {
        let mu = &self.result.fitted_values;
        deviance_residuals(&self.y_values, mu, &self.family)
    }

    /// Compute working residuals: (y - μ) * (dη/dμ).
    pub fn working_residuals(&self) -> Col<f64> {
        let mu = &self.result.fitted_values;
        working_residuals(&self.y_values, mu, &self.family)
    }

    /// Get the overdispersion ratio at the mean of fitted values.
    pub fn overdispersion_ratio(&self) -> f64 {
        let mu_bar: f64 =
            self.result.fitted_values.iter().sum::<f64>() / self.result.n_observations as f64;
        self.family.overdispersion_ratio(mu_bar)
    }

    /// Predict with a new offset (for rate modeling).
    pub fn predict_with_offset(&self, x: &Mat<f64>, offset: &Col<f64>) -> Col<f64> {
        let n_samples = x.nrows();
        let n_features = x.ncols();
        let intercept = self.result.intercept.unwrap_or(0.0);

        Col::from_fn(n_samples, |i| {
            let mut eta = intercept;
            for j in 0..n_features {
                eta += x[(i, j)] * self.result.coefficients[j];
            }
            eta += offset[i];
            self.family.link_inverse(eta)
        })
    }

    /// Compute predictions with standard errors and optional confidence intervals.
    pub fn predict_with_se(
        &self,
        x: &Mat<f64>,
        pred_type: PredictionType,
        interval: Option<IntervalType>,
        level: f64,
    ) -> PredictionResult {
        let n_new = x.nrows();
        let n_features = x.ncols();

        let xtwx_inv = match &self.xtwx_inverse {
            Some(inv) => inv,
            None => {
                let predictions = match pred_type {
                    PredictionType::Response => self.predict(x),
                    PredictionType::Link => self.predict_linear(x),
                };
                return PredictionResult::point_only(predictions);
            }
        };

        let x_design = if self.options.with_intercept {
            let mut x_aug = Mat::zeros(n_new, n_features + 1);
            for i in 0..n_new {
                x_aug[(i, 0)] = 1.0;
                for j in 0..n_features {
                    x_aug[(i, j + 1)] = x[(i, j)];
                }
            }
            x_aug
        } else {
            x.clone()
        };

        let n_params = xtwx_inv.nrows();
        let mut eta = Col::zeros(n_new);
        let mut se_eta = Col::zeros(n_new);

        for i in 0..n_new {
            let mut eta_i = 0.0;
            for j in 0..n_params {
                eta_i += x_design[(i, j)]
                    * if self.options.with_intercept && j == 0 {
                        self.result.intercept.unwrap_or(0.0)
                    } else {
                        let coef_idx = if self.options.with_intercept {
                            j - 1
                        } else {
                            j
                        };
                        if coef_idx < self.result.coefficients.nrows() {
                            self.result.coefficients[coef_idx]
                        } else {
                            0.0
                        }
                    };
            }
            eta[i] = eta_i;

            let mut var_eta = 0.0;
            for j in 0..n_params {
                for k in 0..n_params {
                    var_eta += x_design[(i, j)] * xtwx_inv[(j, k)] * x_design[(i, k)];
                }
            }
            se_eta[i] = (var_eta * self.dispersion).sqrt();
        }

        let (fit, se) = match pred_type {
            PredictionType::Link => (eta.clone(), se_eta.clone()),
            PredictionType::Response => {
                let mu = Col::from_fn(n_new, |i| self.family.link_inverse(eta[i]));
                let se_mu = Col::from_fn(n_new, |i| {
                    // For log link: dμ/dη = μ
                    let dmu_deta = mu[i];
                    se_eta[i] * dmu_deta.abs()
                });
                (mu, se_mu)
            }
        };

        match interval {
            None => PredictionResult::with_intervals(
                fit.clone(),
                Col::zeros(n_new),
                Col::zeros(n_new),
                se,
            ),
            Some(_) => {
                let alpha = 1.0 - level;
                let z = Normal::new(0.0, 1.0)
                    .map(|d| d.inverse_cdf(1.0 - alpha / 2.0))
                    .unwrap_or(1.96);

                let (lower, upper) = match pred_type {
                    PredictionType::Link => {
                        let lower = Col::from_fn(n_new, |i| eta[i] - z * se_eta[i]);
                        let upper = Col::from_fn(n_new, |i| eta[i] + z * se_eta[i]);
                        (lower, upper)
                    }
                    PredictionType::Response => {
                        let lower = Col::from_fn(n_new, |i| {
                            self.family.link_inverse(eta[i] - z * se_eta[i])
                        });
                        let upper = Col::from_fn(n_new, |i| {
                            self.family.link_inverse(eta[i] + z * se_eta[i])
                        });
                        (lower, upper)
                    }
                };

                PredictionResult::with_intervals(fit, lower, upper, se)
            }
        }
    }
}

impl FittedRegressor for FittedNegativeBinomial {
    fn predict(&self, x: &Mat<f64>) -> Col<f64> {
        let n_samples = x.nrows();
        let n_features = x.ncols();
        let intercept = self.result.intercept.unwrap_or(0.0);

        Col::from_fn(n_samples, |i| {
            let mut eta = intercept;
            for j in 0..n_features {
                eta += x[(i, j)] * self.result.coefficients[j];
            }
            self.family.link_inverse(eta)
        })
    }

    fn result(&self) -> &RegressionResult {
        &self.result
    }

    fn predict_with_interval(
        &self,
        x: &Mat<f64>,
        interval: Option<IntervalType>,
        level: f64,
    ) -> PredictionResult {
        self.predict_with_se(x, PredictionType::Response, interval, level)
    }
}

/// Builder for configuring a Negative Binomial regression model.
///
/// Provides a fluent API for setting regression options, including whether
/// to estimate or fix the dispersion parameter theta.
///
/// # Example
///
/// ```rust,ignore
/// use anofox_regression::prelude::*;
///
/// // Estimate theta automatically (default)
/// let model = NegativeBinomialRegressor::builder()
///     .with_intercept(true)
///     .estimate_theta(true)
///     .build();
///
/// // Use fixed theta value
/// let model = NegativeBinomialRegressor::with_theta(2.0)
///     .with_intercept(true)
///     .build();
///
/// // Full configuration
/// let model = NegativeBinomialRegressor::builder()
///     .with_intercept(true)
///     .theta(1.0)                  // Initial/fixed theta
///     .estimate_theta(true)        // Estimate theta during fit
///     .theta_max_iter(25)          // Max iterations for theta estimation
///     .theta_tolerance(1e-6)       // Convergence tolerance for theta
///     .compute_inference(true)
///     .max_iterations(100)
///     .tolerance(1e-8)
///     .build();
/// ```
#[derive(Debug, Clone)]
pub struct NegativeBinomialRegressorBuilder {
    options_builder: RegressionOptionsBuilder,
    theta: f64,
    offset: Option<Col<f64>>,
    estimate_theta: bool,
    theta_max_iter: usize,
    theta_tol: f64,
}

impl Default for NegativeBinomialRegressorBuilder {
    fn default() -> Self {
        Self {
            options_builder: RegressionOptionsBuilder::default(),
            theta: 1.0,
            offset: None,
            estimate_theta: true,
            theta_max_iter: 25,
            theta_tol: 1e-6,
        }
    }
}

impl NegativeBinomialRegressorBuilder {
    /// Set whether to include an intercept term.
    pub fn with_intercept(mut self, include: bool) -> Self {
        self.options_builder = self.options_builder.with_intercept(include);
        self
    }

    /// Set whether to compute inference statistics.
    pub fn compute_inference(mut self, compute: bool) -> Self {
        self.options_builder = self.options_builder.compute_inference(compute);
        self
    }

    /// Set the confidence level for confidence intervals.
    pub fn confidence_level(mut self, level: f64) -> Self {
        self.options_builder = self.options_builder.confidence_level(level);
        self
    }

    /// Set the maximum iterations for IRLS.
    pub fn max_iterations(mut self, max_iter: usize) -> Self {
        self.options_builder = self.options_builder.max_iterations(max_iter);
        self
    }

    /// Set the convergence tolerance.
    pub fn tolerance(mut self, tol: f64) -> Self {
        self.options_builder = self.options_builder.tolerance(tol);
        self
    }

    /// Set the L2 regularization parameter (Ridge penalty).
    ///
    /// When lambda > 0, the IRLS algorithm solves a penalized weighted least
    /// squares problem at each iteration: (X'WX + λI)β = X'Wz
    ///
    /// This helps stabilize estimation when there is overdispersion or
    /// sparse data with extreme values.
    ///
    /// Note: The intercept is not penalized.
    pub fn lambda(mut self, lambda: f64) -> Self {
        self.options_builder = self.options_builder.lambda(lambda);
        self
    }

    /// Set the initial theta value.
    ///
    /// If `estimate_theta` is true, this is the starting value.
    /// If false, this is the fixed value used.
    pub fn theta(mut self, theta: f64) -> Self {
        self.theta = theta.max(0.01);
        self
    }

    /// Set whether to estimate theta during fitting.
    ///
    /// Default is true (alternating estimation).
    pub fn estimate_theta(mut self, estimate: bool) -> Self {
        self.estimate_theta = estimate;
        self
    }

    /// Set maximum iterations for theta estimation.
    pub fn theta_max_iter(mut self, max_iter: usize) -> Self {
        self.theta_max_iter = max_iter;
        self
    }

    /// Set tolerance for theta convergence.
    pub fn theta_tolerance(mut self, tol: f64) -> Self {
        self.theta_tol = tol;
        self
    }

    /// Set the offset term for rate modeling.
    ///
    /// The offset enters the linear predictor: η = Xβ + offset.
    /// For rate modeling with exposure, use offset = log(exposure).
    pub fn offset(mut self, offset: Col<f64>) -> Self {
        self.offset = Some(offset);
        self
    }

    /// Build the regressor.
    pub fn build(self) -> NegativeBinomialRegressor {
        NegativeBinomialRegressor {
            options: self.options_builder.build_unchecked(),
            family: NegativeBinomialFamily::new(self.theta),
            offset: self.offset,
            estimate_theta: self.estimate_theta,
            theta_max_iter: self.theta_max_iter,
            theta_tol: self.theta_tol,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_overdispersed_data(n: usize) -> (Mat<f64>, Col<f64>) {
        // Generate overdispersed count data
        let x = Mat::from_fn(n, 1, |i, _| (i as f64) / (n as f64) * 5.0);
        let y = Col::from_fn(n, |i| {
            let xi = (i as f64) / (n as f64) * 5.0;
            let mu = (0.5 + 0.3 * xi).exp();
            // Add overdispersion via deterministic pattern
            let extra = if i % 3 == 0 { 3.0 } else { 0.0 };
            (mu + extra + 0.5 * ((i % 5) as f64 - 2.0)).max(0.0).round()
        });
        (x, y)
    }

    #[test]
    fn test_negative_binomial_fixed_theta() {
        let (x, y) = create_overdispersed_data(100);

        let fitted = NegativeBinomialRegressor::with_theta(2.0)
            .with_intercept(true)
            .max_iterations(100)
            .build()
            .fit(&x, &y)
            .expect("model should fit");

        // Theta should remain fixed
        assert!((fitted.theta - 2.0).abs() < 1e-10);

        // Coefficient should be positive
        assert!(
            fitted.result.coefficients[0] > 0.0,
            "Coefficient should be positive"
        );

        // Deviance should be less than null deviance
        assert!(
            fitted.deviance <= fitted.null_deviance,
            "Deviance {} should be <= null deviance {}",
            fitted.deviance,
            fitted.null_deviance
        );
    }

    #[test]
    fn test_negative_binomial_estimate_theta() {
        let (x, y) = create_overdispersed_data(100);

        let fitted = NegativeBinomialRegressor::builder()
            .with_intercept(true)
            .estimate_theta(true)
            .max_iterations(100)
            .build()
            .fit(&x, &y)
            .expect("model should fit");

        // Theta should have been estimated (not default 1.0)
        assert!(fitted.theta > 0.0);

        // Model should converge
        assert!(fitted.iterations < 500);
    }

    #[test]
    fn test_predict_count() {
        let (x, y) = create_overdispersed_data(100);

        let fitted = NegativeBinomialRegressor::builder()
            .with_intercept(true)
            .build()
            .fit(&x, &y)
            .expect("model should fit");

        let counts = fitted.predict_count(&x);

        // Counts should be positive
        for i in 0..x.nrows() {
            assert!(counts[i] > 0.0);
        }

        // Counts should generally increase with x
        assert!(counts[x.nrows() - 1] > counts[0]);
    }

    #[test]
    fn test_residual_types() {
        let (x, y) = create_overdispersed_data(100);

        let fitted = NegativeBinomialRegressor::builder()
            .with_intercept(true)
            .build()
            .fit(&x, &y)
            .expect("model should fit");

        let pearson = fitted.pearson_residuals();
        let deviance = fitted.deviance_residuals();
        let working = fitted.working_residuals();

        assert_eq!(pearson.nrows(), 100);
        assert_eq!(deviance.nrows(), 100);
        assert_eq!(working.nrows(), 100);
    }

    #[test]
    fn test_standard_errors() {
        let (x, y) = create_overdispersed_data(100);

        let fitted = NegativeBinomialRegressor::builder()
            .with_intercept(true)
            .compute_inference(true)
            .build()
            .fit(&x, &y)
            .expect("model should fit");

        assert!(fitted.result.std_errors.is_some());
        let se = fitted.result.std_errors.as_ref().expect("std errors exist");
        assert!(se[0] > 0.0);
    }

    #[test]
    fn test_overdispersion_ratio() {
        let (x, y) = create_overdispersed_data(100);

        let fitted = NegativeBinomialRegressor::builder()
            .with_intercept(true)
            .build()
            .fit(&x, &y)
            .expect("model should fit");

        let ratio = fitted.overdispersion_ratio();

        // Overdispersion ratio should be > 1 for overdispersed data
        assert!(ratio > 1.0, "Overdispersion ratio should be > 1");
    }

    #[test]
    fn test_invalid_y_values() {
        let x = Mat::from_fn(10, 1, |i, _| i as f64);
        let y = Col::from_fn(10, |i| if i < 5 { -1.0 } else { 1.0 }); // Invalid: negative

        let result = NegativeBinomialRegressor::builder().build().fit(&x, &y);

        assert!(matches!(result, Err(RegressionError::NumericalError(_))));
    }

    #[test]
    fn test_predict_with_se() {
        let (x, y) = create_overdispersed_data(100);

        let fitted = NegativeBinomialRegressor::builder()
            .with_intercept(true)
            .compute_inference(true)
            .build()
            .fit(&x, &y)
            .expect("model should fit");

        let x_new = Mat::from_fn(5, 1, |i, _| i as f64 + 1.0);
        let pred = fitted.predict_with_se(
            &x_new,
            PredictionType::Response,
            Some(IntervalType::Confidence),
            0.95,
        );

        // Check that SE is computed
        for i in 0..5 {
            assert!(pred.se[i] > 0.0);
            // CI should contain the prediction
            assert!(pred.lower[i] <= pred.fit[i]);
            assert!(pred.upper[i] >= pred.fit[i]);
        }
    }

    #[test]
    fn test_offset() {
        let n = 100;
        let x = Mat::from_fn(n, 1, |i, _| (i as f64) / 10.0);
        // Different exposures
        let exposure = Col::from_fn(n, |i| 1.0 + (i % 3) as f64);
        let offset = Col::from_fn(n, |i| exposure[i].ln());

        // Generate y based on exposure
        let y = Col::from_fn(n, |i| {
            let xi = (i as f64) / 10.0;
            let rate = (0.5 + 0.2 * xi).exp();
            (exposure[i] * rate).round().max(0.0)
        });

        let fitted = NegativeBinomialRegressor::builder()
            .with_intercept(true)
            .offset(offset)
            .build()
            .fit(&x, &y)
            .expect("model should fit");

        // Model should converge
        assert!(fitted.iterations < 500);

        // Coefficient should be positive
        assert!(fitted.result.coefficients[0] > 0.0);
    }

    #[test]
    fn test_predict_with_offset() {
        let (x, y) = create_overdispersed_data(50);

        let fitted = NegativeBinomialRegressor::builder()
            .with_intercept(true)
            .build()
            .fit(&x, &y)
            .expect("model should fit");

        let x_new = Mat::from_fn(5, 1, |i, _| i as f64 + 1.0);
        let offset_new = Col::from_fn(5, |_| 0.5_f64.ln()); // Half exposure

        let pred_no_offset = fitted.predict(&x_new);
        let pred_with_offset = fitted.predict_with_offset(&x_new, &offset_new);

        // Predictions with offset should be different (smaller due to negative offset)
        for i in 0..5 {
            assert!(
                pred_with_offset[i] < pred_no_offset[i],
                "Offset should reduce predictions"
            );
        }
    }

    #[test]
    fn test_approaches_poisson_with_high_theta() {
        let (x, y) = create_overdispersed_data(100);

        // High theta should give Poisson-like variance
        let fitted = NegativeBinomialRegressor::with_theta(1e6)
            .with_intercept(true)
            .build()
            .fit(&x, &y)
            .expect("model should fit");

        // Overdispersion ratio should be close to 1
        let ratio = fitted.overdispersion_ratio();
        assert!(
            (ratio - 1.0).abs() < 0.01,
            "High theta should give ratio near 1"
        );
    }

    #[test]
    fn test_new_constructor() {
        let options = RegressionOptionsBuilder::default()
            .with_intercept(true)
            .build()
            .expect("valid options");
        let family = NegativeBinomialFamily::new(1.0);
        let regressor = NegativeBinomialRegressor::new(options, family);

        let (x, y) = create_overdispersed_data(100);
        let fitted = regressor.fit(&x, &y).expect("model should fit");
        assert!(fitted.result.coefficients[0] > 0.0);
    }

    #[test]
    fn test_no_intercept() {
        let (x, y) = create_overdispersed_data(100);

        let fitted = NegativeBinomialRegressor::builder()
            .with_intercept(false)
            .build()
            .fit(&x, &y)
            .expect("model should fit");

        assert!(fitted.result.intercept.is_none());
    }

    #[test]
    fn test_convergence_failure() {
        let (x, y) = create_overdispersed_data(20);

        let result = NegativeBinomialRegressor::builder()
            .with_intercept(true)
            .max_iterations(1)
            .tolerance(1e-20)
            .build()
            .fit(&x, &y);

        assert!(matches!(
            result,
            Err(RegressionError::ConvergenceFailed { .. })
        ));
    }

    #[test]
    fn test_predict_linear() {
        let (x, y) = create_overdispersed_data(100);

        let fitted = NegativeBinomialRegressor::builder()
            .with_intercept(true)
            .build()
            .fit(&x, &y)
            .expect("model should fit");

        let pred_linear = fitted.predict_linear(&x);
        let pred_response = fitted.predict(&x);

        // Linear predictions should be log of response
        for i in 0..10 {
            assert!((pred_linear[i] - pred_response[i].ln()).abs() < 0.01);
        }
    }

    #[test]
    fn test_predict_with_se_link_scale() {
        let (x, y) = create_overdispersed_data(100);

        let fitted = NegativeBinomialRegressor::builder()
            .with_intercept(true)
            .compute_inference(true)
            .build()
            .fit(&x, &y)
            .expect("model should fit");

        let x_new = Mat::from_fn(5, 1, |i, _| i as f64 + 1.0);
        let pred = fitted.predict_with_se(&x_new, PredictionType::Link, None, 0.95);

        // Link scale predictions should be finite
        for i in 0..5 {
            assert!(pred.fit[i].is_finite());
            assert!(pred.se[i] > 0.0);
        }
    }

    #[test]
    fn test_predict_with_se_prediction_interval() {
        let (x, y) = create_overdispersed_data(100);

        let fitted = NegativeBinomialRegressor::builder()
            .with_intercept(true)
            .compute_inference(true)
            .build()
            .fit(&x, &y)
            .expect("model should fit");

        let x_new = Mat::from_fn(5, 1, |i, _| i as f64 + 1.0);
        let pred = fitted.predict_with_se(
            &x_new,
            PredictionType::Response,
            Some(IntervalType::Prediction),
            0.95,
        );

        // Prediction intervals should be wider
        for i in 0..5 {
            assert!(pred.lower[i] <= pred.fit[i]);
            assert!(pred.upper[i] >= pred.fit[i]);
        }
    }

    #[test]
    fn test_tolerance_builder() {
        let (x, y) = create_overdispersed_data(100);

        let fitted = NegativeBinomialRegressor::builder()
            .with_intercept(true)
            .tolerance(1e-4)
            .build()
            .fit(&x, &y)
            .expect("model should fit");

        assert!(fitted.result.coefficients[0] > 0.0);
    }

    #[test]
    fn test_confidence_level_builder() {
        let (x, y) = create_overdispersed_data(100);

        let fitted = NegativeBinomialRegressor::builder()
            .with_intercept(true)
            .compute_inference(true)
            .confidence_level(0.99)
            .build()
            .fit(&x, &y)
            .expect("model should fit");

        assert!(fitted.result.std_errors.is_some());
    }

    #[test]
    fn test_theta_estimation_settings() {
        let (x, y) = create_overdispersed_data(100);

        let fitted = NegativeBinomialRegressor::builder()
            .with_intercept(true)
            .estimate_theta(true)
            .theta_max_iter(10)
            .theta_tolerance(1e-4)
            .build()
            .fit(&x, &y)
            .expect("model should fit");

        assert!(fitted.theta > 0.0);
    }
}
