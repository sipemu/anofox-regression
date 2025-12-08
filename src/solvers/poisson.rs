//! Poisson regression solver.
//!
//! Implements GLM with Poisson family using Iteratively Reweighted Least Squares (IRLS).
//!
//! # Supported Link Functions
//!
//! - Log (canonical) - most common choice
//! - Identity - linear mean
//! - Square root - variance stabilizing
//!
//! # Example
//!
//! ```rust,ignore
//! use regress_rs::solvers::{PoissonRegressor, Regressor, FittedRegressor};
//! use faer::{Mat, Col};
//!
//! let x = Mat::from_fn(100, 2, |i, j| (i + j) as f64);
//! let y = Col::from_fn(100, |i| (i % 10) as f64);  // count data
//!
//! // Poisson regression with log link
//! let fitted = PoissonRegressor::log()
//!     .build()
//!     .fit(&x, &y)?;
//!
//! let counts = fitted.predict(&x);
//! ```

use crate::core::{
    GlmFamily, IntervalType, PoissonFamily, PoissonLink, PredictionResult, PredictionType,
    RegressionOptions, RegressionOptionsBuilder, RegressionResult,
};
use crate::diagnostics::{deviance_residuals, pearson_residuals, working_residuals};
use crate::solvers::traits::{FittedRegressor, RegressionError, Regressor};
use faer::{Col, Mat};
use statrs::distribution::{ContinuousCDF, FisherSnedecor, Normal};

/// Poisson GLM regression estimator.
///
/// Fits a generalized linear model with Poisson family using IRLS
/// (Iteratively Reweighted Least Squares).
///
/// # Model
///
/// The Poisson GLM models:
/// - `E[Y] = μ = g^(-1)(Xβ)` where g is the link function (log, identity, or sqrt)
/// - `Var[Y] = μ` (Poisson variance)
#[derive(Debug, Clone)]
pub struct PoissonRegressor {
    options: RegressionOptions,
    family: PoissonFamily,
    offset: Option<Col<f64>>,
}

impl PoissonRegressor {
    /// Create a new Poisson regressor with the given options and family.
    pub fn new(options: RegressionOptions, family: PoissonFamily) -> Self {
        Self {
            options,
            family,
            offset: None,
        }
    }

    /// Create a builder for Poisson regression with log link (canonical).
    pub fn log() -> PoissonRegressorBuilder {
        PoissonRegressorBuilder::default().link(PoissonLink::Log)
    }

    /// Create a builder for Poisson regression with identity link.
    pub fn identity() -> PoissonRegressorBuilder {
        PoissonRegressorBuilder::default().link(PoissonLink::Identity)
    }

    /// Create a builder for Poisson regression with square root link.
    pub fn sqrt() -> PoissonRegressorBuilder {
        PoissonRegressorBuilder::default().link(PoissonLink::Sqrt)
    }

    /// Create a general builder.
    pub fn builder() -> PoissonRegressorBuilder {
        PoissonRegressorBuilder::default()
    }

    /// Fit the GLM using IRLS (Iteratively Reweighted Least Squares).
    fn fit_irls(&self, x: &Mat<f64>, y: &Col<f64>) -> Result<FittedPoisson, RegressionError> {
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

        // Initialize η = g(μ) + offset
        let mut eta: Vec<f64> = mu
            .iter()
            .enumerate()
            .map(|(i, &m)| {
                let base_eta = self.family.link(m);
                if let Some(ref offset) = self.offset {
                    base_eta - offset[i] // offset is added to Xβ, so subtract when computing initial η
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

        for iter in 0..max_iter {
            iterations = iter + 1;

            // Compute working weights and response
            let (weights, z) = self.compute_irls_quantities(&y_vec, &mu, &eta);

            // Solve weighted least squares
            let beta_new = self.solve_weighted_ls(&x_design, &z, &weights)?;

            // Check convergence
            let max_change: f64 = beta_new
                .iter()
                .zip(beta.iter())
                .map(|(&b_new, &b_old)| {
                    let diff: f64 = b_new - b_old;
                    diff.abs()
                })
                .fold(0.0_f64, f64::max);

            beta = beta_new;

            // Update η and μ
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
                mu[i] = self.family.link_inverse(eta_i);
            }

            if max_change < tol {
                converged = true;
                break;
            }
        }

        if !converged {
            return Err(RegressionError::ConvergenceFailed {
                iterations: max_iter,
            });
        }

        self.build_result(x, y, &x_design, &beta, &mu, &eta, n_params, iterations)
    }

    fn compute_irls_quantities(&self, y: &[f64], mu: &[f64], eta: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = y.len();
        let mut weights = vec![0.0; n];
        let mut z = vec![0.0; n];

        for i in 0..n {
            weights[i] = self.family.irls_weight(mu[i]);
            // Working response needs adjustment for offset
            let eta_no_offset = if let Some(ref offset) = self.offset {
                eta[i] - offset[i]
            } else {
                eta[i]
            };
            z[i] = eta_no_offset + (y[i] - mu[i]) * self.family.link_derivative(mu[i]);
        }

        (weights, z)
    }

    fn solve_weighted_ls(
        &self,
        x: &Mat<f64>,
        z: &[f64],
        weights: &[f64],
    ) -> Result<Col<f64>, RegressionError> {
        let n_samples = x.nrows();
        let n_params = x.ncols();

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
        let q = qr.compute_q();
        let r = qr.compute_r();
        let perm = qr.col_permutation();

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
            use faer::Index;
            beta[perm.inverse().arrays().0[i].to_signed().unsigned_abs()] = beta_perm[i];
        }

        Ok(beta)
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
    ) -> Result<FittedPoisson, RegressionError> {
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

        // Estimate dispersion (for Poisson, typically 1, but can estimate for overdispersion)
        let df_resid = (n_samples.saturating_sub(n_params)) as f64;
        let dispersion = if df_resid > 0.0 {
            // Pearson chi-squared / df for quasi-Poisson estimate
            let pearson_chi2: f64 = (0..n_samples)
                .map(|i| {
                    let v = self.family.variance(mu[i]);
                    (y[i] - mu[i]).powi(2) / v
                })
                .sum();
            (pearson_chi2 / df_resid).max(1.0) // Use 1.0 for standard Poisson
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
        result.aliased = vec![false; n_features];
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

        Ok(FittedPoisson {
            result,
            options: self.options.clone(),
            family: self.family,
            deviance,
            null_deviance,
            dispersion,
            iterations,
            y_values: y.clone(),
            xtwx_inverse,
            offset: self.offset.clone(),
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
        let q = qr.compute_q();
        let r: Mat<f64> = qr.compute_r();

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

impl Regressor for PoissonRegressor {
    type Fitted = FittedPoisson;

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
                    "y values must be non-negative for Poisson, got y[{}] = {}",
                    i, y[i]
                )));
            }
        }

        self.fit_irls(x, y)
    }
}

/// Fitted Poisson GLM model.
#[derive(Debug, Clone)]
pub struct FittedPoisson {
    result: RegressionResult,
    options: RegressionOptions,
    family: PoissonFamily,
    /// Total deviance.
    pub deviance: f64,
    /// Null deviance (intercept-only model).
    pub null_deviance: f64,
    /// Dispersion parameter (1 for standard Poisson).
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
}

impl FittedPoisson {
    /// Get the Poisson family used for this model.
    pub fn family(&self) -> &PoissonFamily {
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

impl FittedRegressor for FittedPoisson {
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

/// Builder for `PoissonRegressor`.
#[derive(Debug, Clone, Default)]
pub struct PoissonRegressorBuilder {
    options_builder: RegressionOptionsBuilder,
    link: PoissonLink,
    offset: Option<Col<f64>>,
}

impl PoissonRegressorBuilder {
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

    /// Set the link function.
    pub fn link(mut self, link: PoissonLink) -> Self {
        self.link = link;
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
    pub fn build(self) -> PoissonRegressor {
        PoissonRegressor {
            options: self.options_builder.build_unchecked(),
            family: PoissonFamily::new(self.link),
            offset: self.offset,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_poisson_data(n: usize) -> (Mat<f64>, Col<f64>) {
        // Generate count data: y ~ Poisson(exp(0.5 + 0.3*x))
        let x = Mat::from_fn(n, 1, |i, _| (i as f64) / (n as f64) * 5.0);
        let y = Col::from_fn(n, |i| {
            let xi = (i as f64) / (n as f64) * 5.0;
            let mu = (0.5 + 0.3 * xi).exp();
            // Use deterministic "counts" based on mu
            (mu + 0.5 * ((i % 5) as f64 - 2.0)).max(0.0).round()
        });
        (x, y)
    }

    #[test]
    fn test_poisson_log_regression() {
        let (x, y) = create_poisson_data(100);

        let fitted = PoissonRegressor::log()
            .with_intercept(true)
            .max_iterations(100)
            .build()
            .fit(&x, &y)
            .unwrap();

        // Coefficient should be positive (higher x -> higher count)
        assert!(
            fitted.result.coefficients[0] > 0.0,
            "Coefficient should be positive, got {}",
            fitted.result.coefficients[0]
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
    fn test_poisson_identity_regression() {
        let (x, y) = create_poisson_data(100);

        let fitted = PoissonRegressor::identity()
            .with_intercept(true)
            .max_iterations(100)
            .build()
            .fit(&x, &y)
            .unwrap();

        assert!(fitted.result.coefficients[0] > 0.0);
    }

    #[test]
    fn test_poisson_sqrt_regression() {
        let (x, y) = create_poisson_data(100);

        let fitted = PoissonRegressor::sqrt()
            .with_intercept(true)
            .max_iterations(100)
            .build()
            .fit(&x, &y)
            .unwrap();

        assert!(fitted.result.coefficients[0] > 0.0);
    }

    #[test]
    fn test_predict_count() {
        let (x, y) = create_poisson_data(100);

        let fitted = PoissonRegressor::log()
            .with_intercept(true)
            .build()
            .fit(&x, &y)
            .unwrap();

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
        let (x, y) = create_poisson_data(100);

        let fitted = PoissonRegressor::log()
            .with_intercept(true)
            .build()
            .fit(&x, &y)
            .unwrap();

        let pearson = fitted.pearson_residuals();
        let deviance = fitted.deviance_residuals();
        let working = fitted.working_residuals();

        assert_eq!(pearson.nrows(), 100);
        assert_eq!(deviance.nrows(), 100);
        assert_eq!(working.nrows(), 100);
    }

    #[test]
    fn test_standard_errors() {
        let (x, y) = create_poisson_data(100);

        let fitted = PoissonRegressor::log()
            .with_intercept(true)
            .compute_inference(true)
            .build()
            .fit(&x, &y)
            .unwrap();

        assert!(fitted.result.std_errors.is_some());
        let se = fitted.result.std_errors.as_ref().unwrap();
        assert!(se[0] > 0.0);
    }

    #[test]
    fn test_invalid_y_values() {
        let x = Mat::from_fn(10, 1, |i, _| i as f64);
        let y = Col::from_fn(10, |i| if i < 5 { -1.0 } else { 1.0 }); // Invalid: negative

        let result = PoissonRegressor::log().build().fit(&x, &y);

        assert!(matches!(result, Err(RegressionError::NumericalError(_))));
    }

    #[test]
    fn test_predict_with_se() {
        let (x, y) = create_poisson_data(100);

        let fitted = PoissonRegressor::log()
            .with_intercept(true)
            .compute_inference(true)
            .build()
            .fit(&x, &y)
            .unwrap();

        let x_new = Mat::from_fn(5, 1, |i, _| (i as f64 + 1.0));
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
        let exposure = Col::from_fn(n, |i| (1.0 + (i % 3) as f64));
        let offset = Col::from_fn(n, |i| exposure[i].ln());

        // Generate y = Poisson(exposure * exp(0.5 + 0.2*x))
        let y = Col::from_fn(n, |i| {
            let xi = (i as f64) / 10.0;
            let rate = (0.5 + 0.2 * xi).exp();
            (exposure[i] * rate).round().max(0.0)
        });

        let fitted = PoissonRegressor::log()
            .with_intercept(true)
            .offset(offset)
            .build()
            .fit(&x, &y)
            .unwrap();

        // Model should converge
        assert!(fitted.iterations < 100);

        // Coefficient should be positive
        assert!(fitted.result.coefficients[0] > 0.0);
    }

    #[test]
    fn test_predict_with_offset() {
        let (x, y) = create_poisson_data(50);

        let fitted = PoissonRegressor::log()
            .with_intercept(true)
            .build()
            .fit(&x, &y)
            .unwrap();

        let x_new = Mat::from_fn(5, 1, |i, _| (i as f64 + 1.0));
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
}
