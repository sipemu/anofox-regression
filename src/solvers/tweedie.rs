//! Tweedie Generalized Linear Model regression solver.
//!
//! Implements GLM with Tweedie family using Iteratively Reweighted Least Squares (IRLS).
//!
//! # Reference
//!
//! - R package `statmod`: <https://cran.r-project.org/web/packages/statmod/index.html>
//! - Dunn, P.K. and Smyth, G.K. (2018). "Generalized linear models with examples in R".
//!   Springer, New York, NY.

use crate::core::{
    GlmFamily, IntervalType, PredictionResult, PredictionType, RegressionOptions,
    RegressionOptionsBuilder, RegressionResult, TweedieFamily,
};
use crate::diagnostics::{deviance_residuals, pearson_residuals, working_residuals};
use crate::solvers::traits::{FittedRegressor, RegressionError, Regressor};
use crate::utils::detect_constant_columns;
use faer::{Col, Mat};
use statrs::distribution::{ContinuousCDF, FisherSnedecor, Normal};

/// Tweedie GLM regression estimator.
///
/// Fits a generalized linear model with Tweedie family using IRLS
/// (Iteratively Reweighted Least Squares).
///
/// # Model
///
/// The Tweedie GLM models:
/// - `E[Y] = μ = g^(-1)(Xβ + offset)` where g is the link function
/// - `Var[Y] = φ * V(μ)` where `V(μ) = μ^var_power`
///
/// # Example
///
/// ```rust,ignore
/// use anofox_regression::solvers::{TweedieRegressor, Regressor, FittedRegressor};
/// use faer::{Mat, Col};
///
/// let x = Mat::from_fn(100, 2, |i, j| (i + j) as f64);
/// let y = Col::from_fn(100, |i| (i as f64 + 0.1).max(0.0));
///
/// // Gamma regression with log link
/// let fitted = TweedieRegressor::gamma()
///     .build()
///     .fit(&x, &y)?;
///
/// // Compound Poisson-Gamma for zero-inflated data
/// let fitted = TweedieRegressor::builder()
///     .var_power(1.5)
///     .link_power(0.0)  // log link
///     .build()
///     .fit(&x, &y)?;
/// ```
#[derive(Debug, Clone)]
pub struct TweedieRegressor {
    options: RegressionOptions,
    family: TweedieFamily,
    offset: Option<Col<f64>>,
}

impl TweedieRegressor {
    /// Create a new Tweedie regressor with the given options and family.
    pub fn new(options: RegressionOptions, family: TweedieFamily) -> Self {
        Self {
            options,
            family,
            offset: None,
        }
    }

    /// Create a builder for configuring the regressor.
    pub fn builder() -> TweedieRegressorBuilder {
        TweedieRegressorBuilder::default()
    }

    /// Create a builder for Gaussian (Normal) regression.
    pub fn gaussian() -> TweedieRegressorBuilder {
        TweedieRegressorBuilder::default()
            .var_power(0.0)
            .link_power(1.0)
    }

    /// Create a builder for Poisson regression with log link.
    pub fn poisson() -> TweedieRegressorBuilder {
        TweedieRegressorBuilder::default()
            .var_power(1.0)
            .link_power(0.0)
    }

    /// Create a builder for Gamma regression with log link.
    pub fn gamma() -> TweedieRegressorBuilder {
        TweedieRegressorBuilder::default()
            .var_power(2.0)
            .link_power(0.0)
    }

    /// Create a builder for Inverse-Gaussian regression with log link.
    pub fn inverse_gaussian() -> TweedieRegressorBuilder {
        TweedieRegressorBuilder::default()
            .var_power(3.0)
            .link_power(0.0)
    }

    /// Fit the GLM using IRLS (Iteratively Reweighted Least Squares).
    ///
    /// IRLS Algorithm:
    /// 1. Initialize μ and compute η = g(μ)
    /// 2. Compute working weights W = 1 / (V(μ) * (dη/dμ)²)
    /// 3. Compute working response z = η + (y - μ) * (dη/dμ)
    /// 4. Solve weighted least squares: β = (X'WX)^(-1) X'Wz
    /// 5. Update η = Xβ, μ = g^(-1)(η)
    /// 6. Check convergence, repeat until done
    fn fit_irls(&self, x: &Mat<f64>, y: &Col<f64>) -> Result<FittedTweedie, RegressionError> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        // Determine number of parameters
        let n_params = if self.options.with_intercept {
            n_features + 1
        } else {
            n_features
        };

        // Build design matrix (with intercept if needed)
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

        // Initialize μ
        let y_vec: Vec<f64> = (0..n_samples).map(|i| y[i]).collect();
        let mut mu: Vec<f64> = self.family.initialize_mu(&y_vec);

        // Initialize η = g(μ) - offset (so that η + offset = g(μ))
        let mut eta: Vec<f64> = mu
            .iter()
            .enumerate()
            .map(|(i, &m)| {
                let base_eta = self.family.link(m);
                if let Some(ref offset) = self.offset {
                    base_eta - offset[i]
                } else {
                    base_eta
                }
            })
            .collect();

        // Initialize β (will be updated in IRLS)
        let mut beta = Col::zeros(n_params);

        let max_iter = self.options.max_iterations;
        let tol = self.options.tolerance;
        let mut converged = false;
        let mut iterations = 0;

        // Compute initial deviance for deviance-based convergence (following R's glm.fit)
        let mut dev = self.family.deviance(&y_vec, &mu);
        let mut dev_old: f64;

        // Maximum step halvings per iteration (following R's glm.control default)
        const MAX_HALVINGS: usize = 10;

        for iter in 0..max_iter {
            iterations = iter + 1;
            dev_old = dev;

            // Compute working weights and response
            let (weights, z) = self.compute_irls_quantities(&y_vec, &mu, &eta);

            // Solve weighted least squares: min_β Σ wᵢ (zᵢ - xᵢ'β)²
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

            // Try full step first
            beta = beta_new;

            // Update η and μ with validity checks
            self.update_eta_mu(&x_design, &beta, &mut eta, &mut mu, n_samples, n_params)?;

            // Update deviance after μ update
            dev = self.family.deviance(&y_vec, &mu);

            // Step-halving: if deviance increased, halve the step (following R's glm.fit)
            if dev.is_finite() && dev_old.is_finite() {
                let mut n_halvings = 0;
                while dev > dev_old + 1e-7 * dev_old.abs() && n_halvings < MAX_HALVINGS {
                    n_halvings += 1;

                    // Halve the step: β = (β_new + β_old) / 2
                    for j in 0..n_params {
                        beta[j] = (beta[j] + beta_old[j]) / 2.0;
                    }

                    // Recompute η and μ with halved coefficients
                    self.update_eta_mu(&x_design, &beta, &mut eta, &mut mu, n_samples, n_params)?;

                    // Recompute deviance
                    dev = self.family.deviance(&y_vec, &mu);
                }
            }

            // Check convergence using both coefficient change and deviance criteria
            // Following R's glm.fit: |dev - dev_old| / (0.1 + |dev|) < epsilon
            let dev_converged = (dev - dev_old).abs() / (0.1 + dev.abs()) < tol;
            let coef_converged = max_change < tol;

            // Converge if either criterion is met
            if dev_converged || coef_converged {
                converged = true;
                break;
            }
        }

        if !converged {
            return Err(RegressionError::ConvergenceFailed {
                iterations: max_iter,
            });
        }

        // Build result
        self.build_result(
            x,
            y,
            &x_design,
            &beta,
            &mu,
            &eta,
            n_params,
            iterations,
            self.offset.clone(),
            aliased,
        )
    }

    /// Compute IRLS working weights and working response.
    fn compute_irls_quantities(&self, y: &[f64], mu: &[f64], eta: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = y.len();
        let mut weights = vec![0.0; n];
        let mut z = vec![0.0; n];

        for i in 0..n {
            // Weight = 1 / (V(μ) * (dη/dμ)²)
            weights[i] = self.family.irls_weight(mu[i]);

            // Working response z = η + (y - μ) * (dη/dμ)
            z[i] = self.family.working_response(y[i], mu[i], eta[i]);
        }

        (weights, z)
    }

    /// Update η and μ from current coefficients with validity checks.
    fn update_eta_mu(
        &self,
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
            let eta_with_offset = if let Some(ref offset) = self.offset {
                eta_i + offset[i]
            } else {
                eta_i
            };
            eta[i] = eta_i; // Store η without offset for working response

            // Check eta validity (following R's valideta)
            if !self.family.valid_eta(eta_with_offset) {
                return Err(RegressionError::NumericalError(
                    "Invalid linear predictor (eta) during IRLS: non-finite value".to_string(),
                ));
            }

            let mu_new = self.family.link_inverse(eta_with_offset);

            // Check mu validity and clamp if needed (following R's validmu)
            if !self.family.valid_mu(mu_new) {
                mu[i] = self.family.clamp_mu(mu_new);
            } else {
                mu[i] = mu_new;
            }
        }
        Ok(())
    }

    /// Solve weighted least squares: min_β Σ wᵢ (zᵢ - xᵢ'β)²
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
            // Standard IRLS: Transform X_w = sqrt(W) * X, z_w = sqrt(W) * z
            let mut x_weighted = Mat::zeros(n_samples, n_params);
            let mut z_weighted = Col::zeros(n_samples);

            for i in 0..n_samples {
                let sqrt_w = weights[i].sqrt();
                for j in 0..n_params {
                    x_weighted[(i, j)] = sqrt_w * x[(i, j)];
                }
                z_weighted[i] = sqrt_w * z[i];
            }

            // Solve via QR decomposition
            let qr = x_weighted.col_piv_qr();
            let q = qr.compute_Q();
            let r = qr.R();
            let perm = qr.P();

            // Compute Q'z
            let qtz = q.transpose() * z_weighted;

            // Back substitution
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

            // Unpermute
            let mut beta = Col::zeros(n_params);
            for i in 0..n_params {
                beta[perm.inverse().arrays().0[i]] = beta_perm[i];
            }

            Ok(beta)
        }
    }

    /// Build the regression result from fitted values.
    #[allow(clippy::too_many_arguments)]
    fn build_result(
        &self,
        x: &Mat<f64>,
        y: &Col<f64>,
        x_design: &Mat<f64>,
        beta: &Col<f64>,
        mu: &[f64],
        _eta: &[f64],
        n_params: usize,
        iterations: usize,
        offset: Option<Col<f64>>,
        aliased: Vec<bool>,
    ) -> Result<FittedTweedie, RegressionError> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        // Extract intercept and coefficients
        let (intercept, coefficients) = if self.options.with_intercept {
            let int = Some(beta[0]);
            let coefs = Col::from_fn(n_features, |j| beta[j + 1]);
            (int, coefs)
        } else {
            (None, beta.clone())
        };

        // Compute fitted values and residuals
        let fitted_values = Col::from_fn(n_samples, |i| mu[i]);
        let residuals = Col::from_fn(n_samples, |i| y[i] - mu[i]);

        // Compute deviance
        let y_vec: Vec<f64> = (0..n_samples).map(|i| y[i]).collect();
        let deviance = self.family.deviance(&y_vec, mu);
        let null_deviance = self.family.null_deviance(&y_vec);

        // Estimate dispersion parameter φ
        // φ = D / (n - p) where D is deviance
        let df_resid = (n_samples.saturating_sub(n_params)) as f64;
        let dispersion = if df_resid > 0.0 {
            deviance / df_resid
        } else {
            1.0
        };

        // Compute R² (using deviance-based definition for GLM)
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

        // MSE approximation for GLM
        let rss: f64 = residuals.iter().map(|&r| r.powi(2)).sum();
        let mse = if df_resid > 0.0 {
            rss / df_resid
        } else {
            f64::NAN
        };
        let rmse = mse.sqrt();

        // F-statistic (approximate)
        let df_model = (n_params - if intercept.is_some() { 1 } else { 0 }) as f64;
        let f_statistic = if df_model > 0.0 && df_resid > 0.0 && dispersion > 0.0 {
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

        // Information criteria
        // For GLM: log-likelihood ≈ -D / (2φ) + constant
        let n = n_samples as f64;
        let k = n_params as f64;
        let log_likelihood = -deviance / (2.0 * dispersion)
            - n * (2.0 * std::f64::consts::PI * dispersion).ln() / 2.0;

        let aic = 2.0 * k - 2.0 * log_likelihood;
        let aicc = if (n - k - 1.0) > 0.0 {
            aic + 2.0 * k * (k + 1.0) / (n - k - 1.0)
        } else {
            f64::NAN
        };
        let bic = k * n.ln() - 2.0 * log_likelihood;

        // Determine rank
        let rank = n_params; // Full rank assumed for converged model

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

        // GLM-specific: compute standard errors and (X'WX)⁻¹
        let mut xtwx_inverse = None;
        if self.options.compute_inference {
            // Standard errors from Fisher information: SE = sqrt(diag(φ * (X'WX)^(-1)))
            if let Ok((se, xtwx_inv)) =
                self.compute_standard_errors_and_covariance(x_design, mu, dispersion)
            {
                result.std_errors = Some(if self.options.with_intercept {
                    Col::from_fn(n_features, |j| se[j + 1])
                } else {
                    se.clone()
                });

                if self.options.with_intercept {
                    result.intercept_std_error = Some(se[0]);
                }

                xtwx_inverse = Some(xtwx_inv);
            }
        }

        Ok(FittedTweedie {
            result,
            options: self.options.clone(),
            family: self.family,
            deviance,
            null_deviance,
            dispersion,
            iterations,
            y_values: y.clone(),
            xtwx_inverse,
            offset,
            aliased,
        })
    }

    /// Compute standard errors and (X'WX)⁻¹ covariance matrix.
    fn compute_standard_errors_and_covariance(
        &self,
        x: &Mat<f64>,
        mu: &[f64],
        dispersion: f64,
    ) -> Result<(Col<f64>, Mat<f64>), RegressionError> {
        let n_samples = x.nrows();
        let n_params = x.ncols();

        // Compute X'WX
        let mut xtwx: Mat<f64> = Mat::zeros(n_params, n_params);
        for i in 0..n_samples {
            let w = self.family.irls_weight(mu[i]);
            for j in 0..n_params {
                for k in 0..n_params {
                    xtwx[(j, k)] += w * x[(i, j)] * x[(i, k)];
                }
            }
        }

        // Invert via QR
        let qr = xtwx.qr();
        let q = qr.compute_Q();
        let r = qr.R().to_owned();

        // Compute inverse column by column
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

        // SE = sqrt(φ * diag((X'WX)^(-1)))
        let se = Col::from_fn(n_params, |j| (dispersion * xtwx_inv[(j, j)]).sqrt());

        Ok((se, xtwx_inv))
    }
}

impl Regressor for TweedieRegressor {
    type Fitted = FittedTweedie;

    fn fit(&self, x: &Mat<f64>, y: &Col<f64>) -> Result<Self::Fitted, RegressionError> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        // Validate dimensions
        if x.nrows() != y.nrows() {
            return Err(RegressionError::DimensionMismatch {
                x_rows: x.nrows(),
                y_len: y.nrows(),
            });
        }

        // Need at least 2 observations
        if n_samples < 2 {
            return Err(RegressionError::InsufficientObservations {
                needed: 2,
                got: n_samples,
            });
        }

        // Minimum observations
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

        // Validate family
        if !self.family.is_valid() {
            return Err(RegressionError::NumericalError(
                "var_power in (0, 1) is not allowed".to_string(),
            ));
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

        // Check for valid y values based on family
        for i in 0..n_samples {
            if self.family.var_power > 0.0 && y[i] < 0.0 {
                return Err(RegressionError::NumericalError(format!(
                    "y values must be non-negative for var_power > 0, got y[{}] = {}",
                    i, y[i]
                )));
            }
        }

        self.fit_irls(x, y)
    }
}

/// Fitted Tweedie GLM model for flexible variance modeling.
///
/// Contains the estimated coefficients and model diagnostics from fitting
/// a Tweedie regression using IRLS. The Tweedie family unifies several
/// common distributions through the power parameter.
///
/// # Power Parameter
///
/// The Tweedie power parameter controls the variance function `Var[Y] = phi * mu^p`:
/// - p = 0: Normal (Gaussian)
/// - p = 1: Poisson
/// - 1 < p < 2: Compound Poisson-Gamma (insurance claims)
/// - p = 2: Gamma
/// - p = 3: Inverse Gaussian
///
/// # Available Methods
///
/// - [`predict`](FittedRegressor::predict) - Predict response values
/// - [`predict_mu`](Self::predict_mu) - Alias for predict
/// - [`predict_eta`](Self::predict_eta) - Predict on link scale
/// - [`predict_with_se`](Self::predict_with_se) - Predictions with standard errors
/// - [`pearson_residuals`](Self::pearson_residuals) - Pearson residuals
/// - [`deviance_residuals`](Self::deviance_residuals) - Deviance residuals
///
/// # Example
///
/// ```rust,ignore
/// use anofox_regression::prelude::*;
///
/// let x = Mat::from_fn(100, 2, |i, j| (i + j) as f64 / 10.0);
/// let y = Col::from_fn(100, |i| (i as f64 + 1.0).max(0.1));
///
/// // Gamma regression (power = 2)
/// let fitted = TweedieRegressor::gamma()
///     .with_intercept(true)
///     .build()
///     .fit(&x, &y)?;
///
/// // Access model results
/// let coefs = fitted.coefficients();
/// let deviance = fitted.deviance;
/// let dispersion = fitted.dispersion;
/// ```
#[derive(Debug, Clone)]
pub struct FittedTweedie {
    result: RegressionResult,
    options: RegressionOptions,
    family: TweedieFamily,
    /// Total deviance.
    pub deviance: f64,
    /// Null deviance (intercept-only model).
    pub null_deviance: f64,
    /// Estimated dispersion parameter.
    pub dispersion: f64,
    /// Number of IRLS iterations.
    pub iterations: usize,
    /// Original y values (for residual calculation).
    y_values: Col<f64>,
    /// (X'WX)⁻¹ matrix (for prediction standard errors).
    xtwx_inverse: Option<Mat<f64>>,
    /// Offset used in fitting (stored for potential residual calculations).
    #[allow(dead_code)]
    offset: Option<Col<f64>>,
    /// Aliased (collinear or constant) columns.
    #[allow(dead_code)]
    aliased: Vec<bool>,
}

impl FittedTweedie {
    /// Get the Tweedie family used for this model.
    pub fn family(&self) -> &TweedieFamily {
        &self.family
    }

    /// Compute predicted μ values on the response scale.
    pub fn predict_mu(&self, x: &Mat<f64>) -> Col<f64> {
        self.predict(x)
    }

    /// Compute predicted η values on the linear predictor scale.
    pub fn predict_eta(&self, x: &Mat<f64>) -> Col<f64> {
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

    /// Predict with a new offset (for rate modeling).
    ///
    /// The offset enters the linear predictor: η = Xβ + offset.
    /// For rate modeling with exposure, use offset = log(exposure).
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
    ///
    /// # Arguments
    ///
    /// * `x` - New data matrix for prediction
    /// * `pred_type` - Whether to predict on response or link scale
    /// * `interval` - Type of interval to compute (None for no intervals)
    /// * `level` - Confidence level (default 0.95)
    ///
    /// # Details
    ///
    /// Standard error on link scale: `SE(η) = sqrt(x' · (X'WX)⁻¹ · x · φ)`
    /// Standard error on response scale: `SE(μ) = SE(η) · |dμ/dη|`
    ///
    /// Confidence intervals are computed on the link scale and transformed.
    pub fn predict_with_se(
        &self,
        x: &Mat<f64>,
        pred_type: PredictionType,
        interval: Option<IntervalType>,
        level: f64,
    ) -> PredictionResult {
        let n_new = x.nrows();
        let n_features = x.ncols();

        // Check if we have (X'WX)⁻¹
        let xtwx_inv = match &self.xtwx_inverse {
            Some(inv) => inv,
            None => {
                // Fallback to point predictions only
                let predictions = match pred_type {
                    PredictionType::Response => self.predict(x),
                    PredictionType::Link => self.predict_eta(x),
                };
                return PredictionResult::point_only(predictions);
            }
        };

        // Build design matrix for new data
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

        // Compute predictions and standard errors on link scale
        let n_params = xtwx_inv.nrows();
        let mut eta = Col::zeros(n_new);
        let mut se_eta = Col::zeros(n_new);

        for i in 0..n_new {
            // Compute η = x'β
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

            // Compute SE(η) = sqrt(x' · (X'WX)⁻¹ · x · φ)
            let mut var_eta = 0.0;
            for j in 0..n_params {
                for k in 0..n_params {
                    var_eta += x_design[(i, j)] * xtwx_inv[(j, k)] * x_design[(i, k)];
                }
            }
            se_eta[i] = (var_eta * self.dispersion).sqrt();
        }

        // Transform to desired scale
        let (fit, se) = match pred_type {
            PredictionType::Link => (eta.clone(), se_eta.clone()),
            PredictionType::Response => {
                let mu = Col::from_fn(n_new, |i| self.family.link_inverse(eta[i]));
                // SE(μ) = SE(η) * |dμ/dη| (delta method)
                let se_mu = Col::from_fn(n_new, |i| {
                    let dmu_deta = self.family.link_inverse_derivative(eta[i]);
                    se_eta[i] * dmu_deta.abs()
                });
                (mu, se_mu)
            }
        };

        // Compute intervals if requested
        match interval {
            None => PredictionResult::with_intervals(
                fit.clone(),
                Col::zeros(n_new),
                Col::zeros(n_new),
                se,
            ),
            Some(_interval_type) => {
                // Get critical value
                let alpha = 1.0 - level;
                let z = Normal::new(0.0, 1.0)
                    .map(|d| d.inverse_cdf(1.0 - alpha / 2.0))
                    .unwrap_or(1.96);

                // Compute CI on link scale, then transform
                let (lower, upper) = match pred_type {
                    PredictionType::Link => {
                        let lower = Col::from_fn(n_new, |i| eta[i] - z * se_eta[i]);
                        let upper = Col::from_fn(n_new, |i| eta[i] + z * se_eta[i]);
                        (lower, upper)
                    }
                    PredictionType::Response => {
                        // CI on link scale, then transform
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

impl FittedRegressor for FittedTweedie {
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
        _level: f64,
    ) -> PredictionResult {
        let predictions = self.predict(x);

        // GLM prediction intervals are complex due to non-constant variance
        // Return NaN for now (full implementation would need simulation or delta method)
        match interval {
            None => PredictionResult::point_only(predictions),
            Some(_) => {
                let n = x.nrows();
                let nan_vec = Col::from_fn(n, |_| f64::NAN);
                PredictionResult::with_intervals(
                    predictions,
                    nan_vec.clone(),
                    nan_vec.clone(),
                    nan_vec,
                )
            }
        }
    }
}

/// Builder for configuring a Tweedie regression model.
///
/// Provides a fluent API for setting the power parameter and other options.
///
/// # Example
///
/// ```rust,ignore
/// use anofox_regression::prelude::*;
///
/// // Gamma regression (most common for positive continuous)
/// let model = TweedieRegressor::gamma()
///     .with_intercept(true)
///     .build();
///
/// // Poisson regression
/// let model = TweedieRegressor::poisson()
///     .with_intercept(true)
///     .build();
///
/// // Custom power (e.g., compound Poisson-Gamma for insurance)
/// let model = TweedieRegressor::builder()
///     .power(1.5)
///     .with_intercept(true)
///     .compute_inference(true)
///     .build();
/// ```
#[derive(Debug, Clone, Default)]
pub struct TweedieRegressorBuilder {
    options_builder: RegressionOptionsBuilder,
    var_power: f64,
    link_power: Option<f64>,
    offset: Option<Col<f64>>,
}

impl TweedieRegressorBuilder {
    /// Create a new builder with default options.
    pub fn new() -> Self {
        Self {
            options_builder: RegressionOptionsBuilder::default(),
            var_power: 1.5, // Default: compound Poisson-Gamma
            link_power: None,
            offset: None,
        }
    }

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
    /// This helps stabilize estimation when there is collinearity or
    /// extreme values in the data.
    ///
    /// Note: The intercept is not penalized.
    pub fn lambda(mut self, lambda: f64) -> Self {
        self.options_builder = self.options_builder.lambda(lambda);
        self
    }

    /// Set the variance power for the Tweedie family.
    ///
    /// Common values:
    /// - 0: Normal (Gaussian)
    /// - 1: Poisson
    /// - 1.5: Compound Poisson-Gamma
    /// - 2: Gamma
    /// - 3: Inverse-Gaussian
    pub fn var_power(mut self, p: f64) -> Self {
        self.var_power = p;
        self
    }

    /// Set the link power for the link function.
    ///
    /// - 0: Log link (most common)
    /// - 1: Identity link
    /// - -1: Inverse link
    /// - None: Use canonical link (1 - var_power)
    pub fn link_power(mut self, q: f64) -> Self {
        self.link_power = Some(q);
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
    pub fn build(self) -> TweedieRegressor {
        let link_power = self.link_power.unwrap_or(1.0 - self.var_power);
        let family = TweedieFamily::new(self.var_power, link_power);

        TweedieRegressor {
            options: self.options_builder.build_unchecked(),
            family,
            offset: self.offset,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaussian_regression() {
        // Simple linear regression
        let x = Mat::from_fn(20, 1, |i, _| i as f64);
        let y = Col::from_fn(20, |i| 2.0 + 3.0 * i as f64);

        let fitted = TweedieRegressor::gaussian()
            .with_intercept(true)
            .build()
            .fit(&x, &y)
            .expect("model should fit");

        // Should be close to true values
        assert!((fitted.result.intercept.expect("intercept exists") - 2.0).abs() < 0.5);
        assert!((fitted.result.coefficients[0] - 3.0).abs() < 0.1);
    }

    #[test]
    fn test_poisson_regression() {
        // Poisson-like data: y = exp(0.1 + 0.2*x)
        let x = Mat::from_fn(50, 1, |i, _| i as f64 / 10.0);
        let y = Col::from_fn(50, |i| {
            let eta = 0.1 + 0.2 * (i as f64 / 10.0);
            eta.exp().max(0.1)
        });

        let fitted = TweedieRegressor::poisson()
            .with_intercept(true)
            .build()
            .fit(&x, &y)
            .expect("model should fit");

        assert!(fitted.result.r_squared > 0.5);
        assert!(fitted.deviance < fitted.null_deviance);
    }

    #[test]
    fn test_gamma_regression() {
        // Gamma-like data
        let x = Mat::from_fn(30, 1, |i, _| (i + 1) as f64);
        let y = Col::from_fn(30, |i| {
            let eta = 1.0 + 0.05 * (i + 1) as f64;
            eta.exp()
        });

        let fitted = TweedieRegressor::gamma()
            .with_intercept(true)
            .build()
            .fit(&x, &y)
            .expect("model should fit");

        assert!(fitted.result.r_squared > 0.5);
    }

    #[test]
    fn test_compound_poisson_gamma() {
        // Data with zeros
        let x = Mat::from_fn(30, 1, |i, _| i as f64);
        let y = Col::from_fn(30, |i| {
            if i % 5 == 0 {
                0.0
            } else {
                (1.0 + 0.1 * i as f64).exp()
            }
        });

        let fitted = TweedieRegressor::builder()
            .var_power(1.5)
            .link_power(0.0)
            .with_intercept(true)
            .build()
            .fit(&x, &y)
            .expect("model should fit");

        assert!(fitted.iterations > 0);
        assert!(fitted.dispersion > 0.0);
    }

    #[test]
    fn test_deviance_decrease() {
        // Generate data from known log-linear model: y = exp(1 + 0.1*x)
        // Use non-collinear predictor
        let x = Mat::from_fn(30, 1, |i, _| i as f64);
        let y = Col::from_fn(30, |i| (1.0 + 0.1 * i as f64).exp());

        let fitted = TweedieRegressor::gamma()
            .with_intercept(true)
            .build()
            .fit(&x, &y)
            .expect("model should fit");

        // Deviance should be less than null deviance for a good model
        assert!(
            fitted.deviance <= fitted.null_deviance * 1.01,
            "Deviance {} should be <= null deviance {}",
            fitted.deviance,
            fitted.null_deviance
        );
    }

    #[test]
    fn test_predict() {
        let x = Mat::from_fn(20, 1, |i, _| i as f64);
        let y = Col::from_fn(20, |i| (1.0 + 0.1 * i as f64).exp());

        let fitted = TweedieRegressor::gamma()
            .with_intercept(true)
            .build()
            .fit(&x, &y)
            .expect("model should fit");

        let x_new = Mat::from_fn(5, 1, |i, _| (i + 20) as f64);
        let pred = fitted.predict(&x_new);

        // Predictions should be positive
        for i in 0..5 {
            assert!(pred[i] > 0.0);
        }
    }

    #[test]
    fn test_negative_y_error() {
        let x = Mat::from_fn(10, 1, |i, _| i as f64);
        let y = Col::from_fn(10, |i| if i < 5 { i as f64 } else { -1.0 });

        let result = TweedieRegressor::gamma().build().fit(&x, &y);

        assert!(matches!(result, Err(RegressionError::NumericalError(_))));
    }

    #[test]
    fn test_standard_errors() {
        let x = Mat::from_fn(30, 1, |i, _| i as f64);
        let y = Col::from_fn(30, |i| (1.0 + 0.1 * i as f64).exp());

        let fitted = TweedieRegressor::gamma()
            .with_intercept(true)
            .compute_inference(true)
            .build()
            .fit(&x, &y)
            .expect("model should fit");

        assert!(fitted.result.std_errors.is_some());
        let se = fitted.result.std_errors.as_ref().expect("std errors exist");
        assert!(se[0] > 0.0);
    }

    // ==================== Additional tests for coverage ====================

    #[test]
    fn test_tweedie_new_constructor() {
        let options = RegressionOptionsBuilder::default()
            .build()
            .expect("valid options");
        let family = TweedieFamily::new(2.0, 0.0); // Gamma with log link
        let regressor = TweedieRegressor::new(options, family);

        let x = Mat::from_fn(20, 1, |i, _| (i + 1) as f64);
        let y = Col::from_fn(20, |i| (i + 1) as f64);

        let fitted = regressor.fit(&x, &y).expect("should fit");
        assert!(fitted.result.coefficients.nrows() > 0);
    }

    #[test]
    fn test_inverse_gaussian_regression() {
        let x = Mat::from_fn(30, 1, |i, _| (i + 1) as f64);
        let y = Col::from_fn(30, |i| ((i + 1) as f64).powf(1.5));

        let fitted = TweedieRegressor::inverse_gaussian()
            .with_intercept(true)
            .build()
            .fit(&x, &y)
            .expect("model should fit");

        assert!(fitted.result.r_squared > 0.0);
    }

    #[test]
    fn test_fitted_family_getter() {
        let x = Mat::from_fn(20, 1, |i, _| i as f64);
        let y = Col::from_fn(20, |i| (1.0 + 0.1 * i as f64).exp());

        let fitted = TweedieRegressor::gamma()
            .with_intercept(true)
            .build()
            .fit(&x, &y)
            .expect("model should fit");

        let family = fitted.family();
        assert!((family.var_power - 2.0).abs() < 1e-10);
        assert!((family.link_power - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_predict_eta() {
        let x = Mat::from_fn(20, 1, |i, _| i as f64);
        let y = Col::from_fn(20, |i| (1.0 + 0.1 * i as f64).exp());

        let fitted = TweedieRegressor::gamma()
            .with_intercept(true)
            .build()
            .fit(&x, &y)
            .expect("model should fit");

        let x_new = Mat::from_fn(5, 1, |i, _| (i + 20) as f64);
        let eta = fitted.predict_eta(&x_new);
        let mu = fitted.predict(&x_new);

        // For log link: eta = log(mu)
        for i in 0..5 {
            assert!((eta[i] - mu[i].ln()).abs() < 1e-10);
        }
    }

    #[test]
    fn test_residual_methods() {
        let x = Mat::from_fn(30, 1, |i, _| i as f64);
        let y = Col::from_fn(30, |i| (1.0 + 0.1 * i as f64).exp() + (i % 3) as f64 * 0.1);

        let fitted = TweedieRegressor::gamma()
            .with_intercept(true)
            .build()
            .fit(&x, &y)
            .expect("model should fit");

        let pearson = fitted.pearson_residuals();
        let deviance = fitted.deviance_residuals();
        let working = fitted.working_residuals();

        // All residuals should be finite
        for i in 0..30 {
            assert!(pearson[i].is_finite());
            assert!(deviance[i].is_finite());
            assert!(working[i].is_finite());
        }
    }

    #[test]
    fn test_predict_with_offset() {
        let x = Mat::from_fn(30, 1, |i, _| i as f64);
        let y = Col::from_fn(30, |i| (1.0 + 0.1 * i as f64).exp());

        let fitted = TweedieRegressor::gamma()
            .with_intercept(true)
            .build()
            .fit(&x, &y)
            .expect("model should fit");

        let x_new = Mat::from_fn(5, 1, |i, _| (i + 20) as f64);
        let offset = Col::from_fn(5, |i| (i as f64) * 0.1);

        let pred_no_offset = fitted.predict(&x_new);
        let pred_with_offset = fitted.predict_with_offset(&x_new, &offset);

        // With log link, offset shifts the linear predictor
        // mu_with_offset = exp(eta + offset) = exp(eta) * exp(offset)
        for i in 0..5 {
            let expected = pred_no_offset[i] * offset[i].exp();
            assert!(
                (pred_with_offset[i] - expected).abs() / expected < 0.01,
                "Expected {}, got {}",
                expected,
                pred_with_offset[i]
            );
        }
    }

    #[test]
    fn test_predict_with_se_link_scale() {
        let x = Mat::from_fn(30, 1, |i, _| i as f64);
        let y = Col::from_fn(30, |i| (1.0 + 0.1 * i as f64).exp());

        let fitted = TweedieRegressor::gamma()
            .with_intercept(true)
            .compute_inference(true)
            .build()
            .fit(&x, &y)
            .expect("model should fit");

        let x_new = Mat::from_fn(5, 1, |i, _| (i + 20) as f64);
        let result = fitted.predict_with_se(&x_new, PredictionType::Link, None, 0.95);

        assert_eq!(result.fit.nrows(), 5);
        assert_eq!(result.se.nrows(), 5);
        for i in 0..5 {
            assert!(result.se[i] > 0.0);
        }
    }

    #[test]
    fn test_predict_with_se_response_scale() {
        let x = Mat::from_fn(30, 1, |i, _| i as f64);
        let y = Col::from_fn(30, |i| (1.0 + 0.1 * i as f64).exp());

        let fitted = TweedieRegressor::gamma()
            .with_intercept(true)
            .compute_inference(true)
            .build()
            .fit(&x, &y)
            .expect("model should fit");

        let x_new = Mat::from_fn(5, 1, |i, _| (i + 20) as f64);
        let result = fitted.predict_with_se(&x_new, PredictionType::Response, None, 0.95);

        assert_eq!(result.fit.nrows(), 5);
        for i in 0..5 {
            assert!(result.fit[i] > 0.0); // Response predictions should be positive
        }
    }

    #[test]
    fn test_predict_with_se_confidence_interval() {
        let x = Mat::from_fn(30, 1, |i, _| i as f64);
        let y = Col::from_fn(30, |i| (1.0 + 0.1 * i as f64).exp());

        let fitted = TweedieRegressor::gamma()
            .with_intercept(true)
            .compute_inference(true)
            .build()
            .fit(&x, &y)
            .expect("model should fit");

        let x_new = Mat::from_fn(5, 1, |i, _| (i + 20) as f64);
        let result = fitted.predict_with_se(
            &x_new,
            PredictionType::Response,
            Some(IntervalType::Confidence),
            0.95,
        );

        assert_eq!(result.fit.nrows(), 5);
        assert_eq!(result.lower.nrows(), 5);
        assert_eq!(result.upper.nrows(), 5);
        for i in 0..5 {
            assert!(result.lower[i] < result.fit[i]);
            assert!(result.upper[i] > result.fit[i]);
        }
    }

    #[test]
    fn test_predict_with_se_no_inference() {
        // When compute_inference=false, xtwx_inverse is None
        let x = Mat::from_fn(30, 1, |i, _| i as f64);
        let y = Col::from_fn(30, |i| (1.0 + 0.1 * i as f64).exp());

        let fitted = TweedieRegressor::gamma()
            .with_intercept(true)
            .compute_inference(false)
            .build()
            .fit(&x, &y)
            .expect("model should fit");

        let x_new = Mat::from_fn(5, 1, |i, _| (i + 20) as f64);
        let result = fitted.predict_with_se(&x_new, PredictionType::Response, None, 0.95);

        // Should return point predictions only
        assert_eq!(result.fit.nrows(), 5);
    }

    #[test]
    fn test_dimension_mismatch_error() {
        let x = Mat::from_fn(10, 2, |i, j| (i + j) as f64);
        let y = Col::from_fn(5, |i| i as f64); // Wrong size

        let result = TweedieRegressor::gamma().build().fit(&x, &y);

        assert!(matches!(
            result,
            Err(RegressionError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_insufficient_observations_error() {
        let x = Mat::from_fn(1, 2, |_, _| 1.0);
        let y = Col::from_fn(1, |_| 1.0);

        let result = TweedieRegressor::gamma().build().fit(&x, &y);

        assert!(matches!(
            result,
            Err(RegressionError::InsufficientObservations { .. })
        ));
    }

    #[test]
    fn test_insufficient_observations_for_params() {
        let x = Mat::from_fn(3, 5, |i, j| (i + j) as f64); // 3 obs, 5 features + intercept = 6 params
        let y = Col::from_fn(3, |i| (i + 1) as f64);

        let result = TweedieRegressor::gamma()
            .with_intercept(true)
            .build()
            .fit(&x, &y);

        assert!(matches!(
            result,
            Err(RegressionError::InsufficientObservations { .. })
        ));
    }

    #[test]
    fn test_without_intercept() {
        let x = Mat::from_fn(30, 1, |i, _| (i + 1) as f64);
        let y = Col::from_fn(30, |i| (0.1 * (i + 1) as f64).exp());

        let fitted = TweedieRegressor::gamma()
            .with_intercept(false)
            .build()
            .fit(&x, &y)
            .expect("model should fit");

        assert!(fitted.result.intercept.is_none());
    }

    #[test]
    fn test_predict_without_intercept() {
        let x = Mat::from_fn(30, 1, |i, _| (i + 1) as f64);
        let y = Col::from_fn(30, |i| (0.1 * (i + 1) as f64).exp());

        let fitted = TweedieRegressor::gamma()
            .with_intercept(false)
            .build()
            .fit(&x, &y)
            .expect("model should fit");

        let x_new = Mat::from_fn(5, 1, |i, _| (i + 30) as f64);
        let pred = fitted.predict(&x_new);

        for i in 0..5 {
            assert!(pred[i] > 0.0);
        }
    }
}
