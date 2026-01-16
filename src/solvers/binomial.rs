//! Binomial (Logistic/Probit) regression solver.
//!
//! Implements GLM with binomial family using Iteratively Reweighted Least Squares (IRLS).
//!
//! # Supported Link Functions
//!
//! - Logit (logistic regression) - canonical link
//! - Probit (probit regression)
//! - Complementary log-log (cloglog)
//!
//! # Example
//!
//! ```rust,ignore
//! use anofox_regression::solvers::{BinomialRegressor, Regressor, FittedRegressor};
//! use faer::{Mat, Col};
//!
//! let x = Mat::from_fn(100, 2, |i, j| (i + j) as f64);
//! let y = Col::from_fn(100, |i| if i % 2 == 0 { 0.0 } else { 1.0 });
//!
//! // Logistic regression
//! let fitted = BinomialRegressor::logistic()
//!     .build()
//!     .fit(&x, &y)?;
//!
//! let probs = fitted.predict_probability(&x);
//! ```

use crate::core::{
    BinomialFamily, BinomialLink, GlmFamily, IntervalType, PredictionResult, PredictionType,
    RegressionOptions, RegressionOptionsBuilder, RegressionResult,
};
use crate::diagnostics::{deviance_residuals, pearson_residuals, working_residuals};
use crate::solvers::traits::{FittedRegressor, RegressionError, Regressor};
use crate::utils::detect_constant_columns;
use faer::{Col, Mat};
use statrs::distribution::{ContinuousCDF, FisherSnedecor, Normal};

/// Binomial GLM regression estimator.
///
/// Fits a generalized linear model with binomial family using IRLS
/// (Iteratively Reweighted Least Squares).
///
/// # Model
///
/// The binomial GLM models:
/// - `E[Y] = μ = g^(-1)(Xβ + offset)` where g is the link function (logit, probit, or cloglog)
/// - `Var[Y] = μ(1-μ)` (binomial variance)
#[derive(Debug, Clone)]
pub struct BinomialRegressor {
    options: RegressionOptions,
    family: BinomialFamily,
    offset: Option<Col<f64>>,
}

impl BinomialRegressor {
    /// Create a new binomial regressor with the given options and family.
    pub fn new(options: RegressionOptions, family: BinomialFamily) -> Self {
        Self {
            options,
            family,
            offset: None,
        }
    }

    /// Create a builder for logistic regression (logit link).
    pub fn logistic() -> BinomialRegressorBuilder {
        BinomialRegressorBuilder::default().link(BinomialLink::Logit)
    }

    /// Create a builder for probit regression.
    pub fn probit() -> BinomialRegressorBuilder {
        BinomialRegressorBuilder::default().link(BinomialLink::Probit)
    }

    /// Create a builder for complementary log-log regression.
    pub fn cloglog() -> BinomialRegressorBuilder {
        BinomialRegressorBuilder::default().link(BinomialLink::Cloglog)
    }

    /// Create a general builder.
    pub fn builder() -> BinomialRegressorBuilder {
        BinomialRegressorBuilder::default()
    }

    /// Fit the GLM using IRLS (Iteratively Reweighted Least Squares).
    fn fit_irls(&self, x: &Mat<f64>, y: &Col<f64>) -> Result<FittedBinomial, RegressionError> {
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

        // Detect constant columns in original feature matrix
        let aliased = detect_constant_columns(x, self.options.rank_tolerance);

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

    fn compute_irls_quantities(&self, y: &[f64], mu: &[f64], eta: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = y.len();
        let mut weights = vec![0.0; n];
        let mut z = vec![0.0; n];

        for i in 0..n {
            weights[i] = self.family.irls_weight(mu[i]);
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
        _eta: &[f64],
        n_params: usize,
        iterations: usize,
        offset: Option<Col<f64>>,
        aliased: Vec<bool>,
    ) -> Result<FittedBinomial, RegressionError> {
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
        let deviance = self.family.deviance(&y_vec, mu);
        let null_deviance = self.family.null_deviance(&y_vec);

        // For binomial, dispersion is typically fixed at 1
        let dispersion = 1.0;

        let r_squared = if null_deviance > 0.0 {
            1.0 - deviance / null_deviance
        } else {
            f64::NAN
        };

        let df_resid = (n_samples.saturating_sub(n_params)) as f64;
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
        let k = n_params as f64;
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

        Ok(FittedBinomial {
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

    fn compute_standard_errors_and_covariance(
        &self,
        x: &Mat<f64>,
        mu: &[f64],
        dispersion: f64,
    ) -> Result<(Col<f64>, Mat<f64>), RegressionError> {
        let n_samples = x.nrows();
        let n_params = x.ncols();

        let mut xtwx: Mat<f64> = Mat::zeros(n_params, n_params);
        for i in 0..n_samples {
            let w = self.family.irls_weight(mu[i]);
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

impl Regressor for BinomialRegressor {
    type Fitted = FittedBinomial;

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

        // Validate y values are in [0, 1]
        for i in 0..n_samples {
            if !(0.0..=1.0).contains(&y[i]) {
                return Err(RegressionError::NumericalError(format!(
                    "y values must be in [0, 1] for binomial, got y[{}] = {}",
                    i, y[i]
                )));
            }
        }

        self.fit_irls(x, y)
    }
}

/// Fitted Binomial GLM model (logistic/probit regression).
///
/// Contains the estimated coefficients and model diagnostics from fitting
/// a binomial regression to binary outcome data using IRLS (Iteratively
/// Reweighted Least Squares).
///
/// # Available Methods
///
/// - [`predict`](FittedRegressor::predict) - Predict probabilities for new data
/// - [`predict_probability`](Self::predict_probability) - Alias for predict
/// - [`predict_linear`](Self::predict_linear) - Predict on link scale (log-odds for logit)
/// - [`predict_with_se`](Self::predict_with_se) - Predictions with standard errors
/// - [`pearson_residuals`](Self::pearson_residuals) - Pearson residuals
/// - [`deviance_residuals`](Self::deviance_residuals) - Deviance residuals
/// - [`working_residuals`](Self::working_residuals) - Working residuals
///
/// # Example
///
/// ```rust,ignore
/// use anofox_regression::prelude::*;
///
/// // Binary classification data
/// let x = Mat::from_fn(100, 2, |i, j| (i + j) as f64 / 10.0);
/// let y = Col::from_fn(100, |i| if i % 2 == 0 { 0.0 } else { 1.0 });
///
/// let fitted = BinomialRegressor::logistic()
///     .with_intercept(true)
///     .compute_inference(true)
///     .build()
///     .fit(&x, &y)?;
///
/// // Access model results
/// let coefs = fitted.coefficients();
/// let deviance = fitted.deviance;
///
/// // Predict probabilities
/// let probs = fitted.predict_probability(&x_new);
///
/// // Get log-odds predictions
/// let log_odds = fitted.predict_linear(&x_new);
///
/// // Predictions with confidence intervals
/// let pred = fitted.predict_with_se(
///     &x_new,
///     PredictionType::Response,
///     Some(IntervalType::Confidence),
///     0.95
/// );
/// ```
#[derive(Debug, Clone)]
pub struct FittedBinomial {
    result: RegressionResult,
    options: RegressionOptions,
    family: BinomialFamily,
    /// Total deviance.
    pub deviance: f64,
    /// Null deviance (intercept-only model).
    pub null_deviance: f64,
    /// Dispersion parameter (typically 1 for binomial).
    pub dispersion: f64,
    /// Number of IRLS iterations.
    pub iterations: usize,
    /// Original y values.
    y_values: Col<f64>,
    /// (X'WX)⁻¹ matrix for prediction SE.
    xtwx_inverse: Option<Mat<f64>>,
    /// Offset used in fitting (stored for potential residual calculations).
    #[allow(dead_code)]
    offset: Option<Col<f64>>,
    /// Which columns are aliased (constant or collinear).
    #[allow(dead_code)]
    aliased: Vec<bool>,
}

impl FittedBinomial {
    /// Get the binomial family used for this model.
    pub fn family(&self) -> &BinomialFamily {
        &self.family
    }

    /// Compute predicted probabilities (response scale).
    pub fn predict_probability(&self, x: &Mat<f64>) -> Col<f64> {
        self.predict(x)
    }

    /// Compute predicted log-odds (logit) or equivalent on linear predictor scale.
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

    /// Predict with a new offset.
    ///
    /// The offset enters the linear predictor: η = Xβ + offset.
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
                    let dmu_deta = self.family.link.link_inverse_derivative(eta[i]);
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

impl FittedRegressor for FittedBinomial {
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

/// Builder for configuring a binomial (logistic/probit) regression model.
///
/// Provides a fluent API for setting regression options before fitting.
///
/// # Example
///
/// ```rust,ignore
/// use anofox_regression::prelude::*;
///
/// // Logistic regression (most common)
/// let model = BinomialRegressor::logistic()
///     .with_intercept(true)
///     .build();
///
/// // Probit regression
/// let model = BinomialRegressor::probit()
///     .with_intercept(true)
///     .build();
///
/// // Full configuration
/// let model = BinomialRegressor::builder()
///     .link(BinomialLink::Logit)
///     .with_intercept(true)
///     .compute_inference(true)
///     .confidence_level(0.95)
///     .max_iterations(100)
///     .tolerance(1e-8)
///     .build();
/// ```
#[derive(Debug, Clone, Default)]
pub struct BinomialRegressorBuilder {
    options_builder: RegressionOptionsBuilder,
    link: BinomialLink,
    offset: Option<Col<f64>>,
}

impl BinomialRegressorBuilder {
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
    /// This helps stabilize estimation when there is quasi-separation or
    /// perfect prediction in the data.
    ///
    /// Note: The intercept is not penalized.
    pub fn lambda(mut self, lambda: f64) -> Self {
        self.options_builder = self.options_builder.lambda(lambda);
        self
    }

    /// Set the link function.
    pub fn link(mut self, link: BinomialLink) -> Self {
        self.link = link;
        self
    }

    /// Set the offset term.
    ///
    /// The offset enters the linear predictor: η = Xβ + offset.
    pub fn offset(mut self, offset: Col<f64>) -> Self {
        self.offset = Some(offset);
        self
    }

    /// Build the regressor.
    pub fn build(self) -> BinomialRegressor {
        BinomialRegressor {
            options: self.options_builder.build_unchecked(),
            family: BinomialFamily::new(self.link),
            offset: self.offset,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper function to create test data with overlap (not perfectly separable)
    fn create_test_data(n: usize) -> (Mat<f64>, Col<f64>) {
        // Data with overlapping classes - probabilistic relationship
        let x = Mat::from_fn(n, 1, |i, _| (i as f64) / (n as f64) * 4.0 - 2.0);
        // y is probabilistic: P(y=1) increases with x but not perfectly
        let y = Col::from_fn(n, |i| {
            let xi = (i as f64) / (n as f64) * 4.0 - 2.0;
            // Use logistic probability with some "noise" from deterministic pattern
            let prob = 1.0 / (1.0 + (-xi).exp());
            if prob > 0.5 + 0.1 * ((i % 5) as f64 - 2.0) / 2.0 {
                1.0
            } else {
                0.0
            }
        });
        (x, y)
    }

    #[test]
    fn test_logistic_regression() {
        let (x, y) = create_test_data(100);

        let fitted = BinomialRegressor::logistic()
            .with_intercept(true)
            .max_iterations(100)
            .build()
            .fit(&x, &y)
            .expect("model should fit");

        // Coefficient should be positive (higher x -> higher probability of 1)
        assert!(fitted.result.coefficients[0] > 0.0);

        // Deviance should be less than null deviance
        assert!(fitted.deviance <= fitted.null_deviance);
    }

    #[test]
    fn test_probit_regression() {
        let (x, y) = create_test_data(100);

        let fitted = BinomialRegressor::probit()
            .with_intercept(true)
            .max_iterations(100)
            .build()
            .fit(&x, &y)
            .expect("model should fit");

        assert!(fitted.result.coefficients[0] > 0.0);
    }

    #[test]
    fn test_cloglog_regression() {
        let (x, y) = create_test_data(100);

        let fitted = BinomialRegressor::cloglog()
            .with_intercept(true)
            .max_iterations(100)
            .build()
            .fit(&x, &y)
            .expect("model should fit");

        assert!(fitted.result.coefficients[0] > 0.0);
    }

    #[test]
    fn test_predict_probability() {
        let (x, y) = create_test_data(100);

        let fitted = BinomialRegressor::logistic()
            .with_intercept(true)
            .build()
            .fit(&x, &y)
            .expect("model should fit");

        let probs = fitted.predict_probability(&x);

        // Probabilities should be in (0, 1)
        for i in 0..x.nrows() {
            assert!(probs[i] > 0.0 && probs[i] < 1.0);
        }

        // Probabilities should generally increase with x (check first vs last)
        assert!(probs[x.nrows() - 1] > probs[0]);
    }

    #[test]
    fn test_residual_types() {
        let (x, y) = create_test_data(100);

        let fitted = BinomialRegressor::logistic()
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
        let (x, y) = create_test_data(100);

        let fitted = BinomialRegressor::logistic()
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
    fn test_invalid_y_values() {
        let x = Mat::from_fn(10, 1, |i, _| i as f64);
        let y = Col::from_fn(10, |i| if i < 5 { 0.5 } else { 2.0 }); // Invalid: > 1

        let result = BinomialRegressor::logistic().build().fit(&x, &y);

        assert!(matches!(result, Err(RegressionError::NumericalError(_))));
    }

    #[test]
    fn test_predict_with_se() {
        let (x, y) = create_test_data(100);

        let fitted = BinomialRegressor::logistic()
            .with_intercept(true)
            .compute_inference(true)
            .build()
            .fit(&x, &y)
            .expect("model should fit");

        let x_new = Mat::from_fn(5, 1, |i, _| (i as f64 - 2.0) / 2.0);
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
    fn test_new_constructor() {
        let options = RegressionOptionsBuilder::default()
            .with_intercept(true)
            .build()
            .expect("valid options");
        let family = BinomialFamily::logistic();
        let regressor = BinomialRegressor::new(options, family);

        let (x, y) = create_test_data(100);
        let fitted = regressor.fit(&x, &y).expect("model should fit");
        assert!(fitted.result.coefficients[0] > 0.0);
    }

    #[test]
    fn test_builder_method() {
        let (x, y) = create_test_data(100);

        let fitted = BinomialRegressor::builder()
            .link(BinomialLink::Logit)
            .with_intercept(true)
            .build()
            .fit(&x, &y)
            .expect("model should fit");

        assert!(fitted.result.coefficients[0] > 0.0);
    }

    #[test]
    fn test_no_intercept() {
        let (x, y) = create_test_data(100);

        let fitted = BinomialRegressor::logistic()
            .with_intercept(false)
            .build()
            .fit(&x, &y)
            .expect("model should fit");

        // Without intercept, should have no intercept term
        assert!(fitted.result.intercept.is_none());
    }

    #[test]
    fn test_with_offset() {
        let (x, y) = create_test_data(100);
        let offset = Col::from_fn(100, |i| 0.1 * (i as f64 - 50.0) / 50.0);

        let fitted = BinomialRegressor::logistic()
            .with_intercept(true)
            .offset(offset)
            .build()
            .fit(&x, &y)
            .expect("model should fit");

        // Model should still fit with offset
        assert!(fitted.deviance < fitted.null_deviance);
    }

    #[test]
    fn test_convergence_failure() {
        // Create difficult data that won't converge in 1 iteration
        let x = Mat::from_fn(10, 1, |i, _| (i as f64 - 5.0).signum());
        let y = Col::from_fn(10, |i| if i < 5 { 0.0 } else { 1.0 });

        let result = BinomialRegressor::logistic()
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
    fn test_confidence_level_builder() {
        let (x, y) = create_test_data(100);

        let fitted = BinomialRegressor::logistic()
            .with_intercept(true)
            .compute_inference(true)
            .confidence_level(0.99)
            .build()
            .fit(&x, &y)
            .expect("model should fit");

        assert!(fitted.result.std_errors.is_some());
    }

    #[test]
    fn test_tolerance_builder() {
        let (x, y) = create_test_data(100);

        let fitted = BinomialRegressor::logistic()
            .with_intercept(true)
            .tolerance(1e-4)
            .build()
            .fit(&x, &y)
            .expect("model should fit");

        assert!(fitted.result.coefficients[0] > 0.0);
    }

    #[test]
    fn test_predict_link() {
        let (x, y) = create_test_data(100);

        let fitted = BinomialRegressor::logistic()
            .with_intercept(true)
            .build()
            .fit(&x, &y)
            .expect("model should fit");

        let pred_link = fitted.predict_linear(&x);
        let pred_response = fitted.predict(&x);

        // Link predictions should have larger range than probabilities
        let link_range = pred_link.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
            - pred_link.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let resp_range = pred_response
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b))
            - pred_response.iter().fold(f64::INFINITY, |a, &b| a.min(b));

        assert!(link_range > resp_range);
    }

    #[test]
    fn test_predict_with_se_link_scale() {
        let (x, y) = create_test_data(100);

        let fitted = BinomialRegressor::logistic()
            .with_intercept(true)
            .compute_inference(true)
            .build()
            .fit(&x, &y)
            .expect("model should fit");

        let x_new = Mat::from_fn(5, 1, |i, _| (i as f64 - 2.0) / 2.0);
        let pred = fitted.predict_with_se(&x_new, PredictionType::Link, None, 0.95);

        // Link scale predictions should be unbounded
        for i in 0..5 {
            assert!(pred.fit[i].is_finite());
            assert!(pred.se[i] > 0.0);
        }
    }

    #[test]
    fn test_predict_with_se_prediction_interval() {
        let (x, y) = create_test_data(100);

        let fitted = BinomialRegressor::logistic()
            .with_intercept(true)
            .compute_inference(true)
            .build()
            .fit(&x, &y)
            .expect("model should fit");

        let x_new = Mat::from_fn(5, 1, |i, _| (i as f64 - 2.0) / 2.0);
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
    fn test_no_inference_prediction_se() {
        let (x, y) = create_test_data(100);

        let fitted = BinomialRegressor::logistic()
            .with_intercept(true)
            .compute_inference(false)
            .build()
            .fit(&x, &y)
            .expect("model should fit");

        let x_new = Mat::from_fn(5, 1, |i, _| (i as f64 - 2.0) / 2.0);
        let pred = fitted.predict_with_se(&x_new, PredictionType::Response, None, 0.95);

        // Without inference (xtwx_inverse is None), returns point_only with zeros
        for i in 0..5 {
            assert!((pred.se[i] - 0.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_penalized_irls() {
        // Test the penalized IRLS path with lambda > 0
        let (x, y) = create_test_data(100);

        // Fit with regularization
        let fitted_penalized = BinomialRegressor::logistic()
            .with_intercept(true)
            .lambda(0.1)
            .max_iterations(100)
            .build()
            .fit(&x, &y)
            .expect("penalized model should fit");

        // Fit without regularization
        let fitted_unpenalized = BinomialRegressor::logistic()
            .with_intercept(true)
            .lambda(0.0)
            .max_iterations(100)
            .build()
            .fit(&x, &y)
            .expect("unpenalized model should fit");

        // Both should converge
        assert!(fitted_penalized.iterations < 100);
        assert!(fitted_unpenalized.iterations < 100);

        // Penalized coefficients should be shrunk toward zero
        let coef_penalized = fitted_penalized.result.coefficients[0].abs();
        let coef_unpenalized = fitted_unpenalized.result.coefficients[0].abs();
        assert!(
            coef_penalized <= coef_unpenalized + 0.1,
            "Penalized coefficient {} should not be much larger than unpenalized {}",
            coef_penalized,
            coef_unpenalized
        );
    }

    #[test]
    fn test_penalized_irls_no_intercept() {
        // Test penalized IRLS without intercept
        let (x, y) = create_test_data(100);

        let fitted = BinomialRegressor::logistic()
            .with_intercept(false)
            .lambda(0.5)
            .max_iterations(100)
            .build()
            .fit(&x, &y)
            .expect("penalized model without intercept should fit");

        assert!(fitted.result.intercept.is_none());
        assert!(fitted.iterations < 100);
    }

    #[test]
    fn test_penalized_irls_high_lambda() {
        // Test penalized IRLS with high lambda - coefficients should be heavily shrunk
        let (x, y) = create_test_data(100);

        let fitted = BinomialRegressor::logistic()
            .with_intercept(true)
            .lambda(100.0)
            .max_iterations(100)
            .build()
            .fit(&x, &y)
            .expect("heavily penalized model should fit");

        // Coefficient should be small due to heavy penalization
        assert!(
            fitted.result.coefficients[0].abs() < 1.0,
            "Heavily penalized coefficient should be small: {}",
            fitted.result.coefficients[0]
        );
    }

    #[test]
    fn test_penalized_irls_multivariate() {
        // Test penalized IRLS with multiple features
        let n = 100;
        let x = Mat::from_fn(n, 3, |i, j| (i as f64 + j as f64 * 0.5) / 10.0 - 2.0);
        let y = Col::from_fn(n, |i| {
            let xi = (i as f64) / (n as f64);
            if xi > 0.5 {
                1.0
            } else {
                0.0
            }
        });

        let fitted = BinomialRegressor::logistic()
            .with_intercept(true)
            .lambda(0.1)
            .max_iterations(100)
            .build()
            .fit(&x, &y)
            .expect("multivariate penalized model should fit");

        assert_eq!(fitted.result.coefficients.nrows(), 3);
        assert!(fitted.iterations < 100);
    }

    #[test]
    fn test_penalized_irls_probit() {
        // Test penalized IRLS with probit link
        let (x, y) = create_test_data(100);

        let fitted = BinomialRegressor::probit()
            .with_intercept(true)
            .lambda(0.5)
            .max_iterations(100)
            .build()
            .fit(&x, &y)
            .expect("penalized probit model should fit");

        assert!(fitted.iterations < 100);
        assert!(fitted.result.coefficients[0] > 0.0);
    }
}
