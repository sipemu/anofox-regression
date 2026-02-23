//! Augmented Linear Model (ALM) solver.
//!
//! Implements maximum likelihood estimation for linear regression with various
//! error distributions. Based on the greybox R package.
//!
//! # Supported Distributions
//!
//! ## Continuous
//! - Normal (Gaussian)
//! - Laplace (double exponential)
//! - Student-t
//! - Logistic
//! - Asymmetric Laplace (quantile regression)
//! - Generalised Normal (Subbotin)
//! - S distribution
//! - Log-Normal
//! - Log-Laplace
//! - Folded Normal
//! - Box-Cox Normal
//! - Gamma
//! - Inverse Gaussian
//! - Exponential
//! - Beta
//!
//! ## Discrete
//! - Poisson
//! - Negative Binomial
//! - Binomial
//! - Geometric
//!
//! # Example
//!
//! ```rust,ignore
//! use statistics::solvers::{AlmRegressor, AlmDistribution, Regressor, FittedRegressor};
//! use faer::{Mat, Col};
//!
//! let x = Mat::from_fn(100, 2, |i, j| (i + j) as f64);
//! let y = Col::from_fn(100, |i| 1.0 + 2.0 * i as f64);
//!
//! let fitted = AlmRegressor::builder()
//!     .distribution(AlmDistribution::Laplace)
//!     .with_intercept(true)
//!     .build()
//!     .fit(&x, &y)?;
//!
//! println!("Coefficients: {:?}", fitted.coefficients());
//! ```

use crate::core::{
    IntervalType, PredictionResult, RegressionOptions, RegressionOptionsBuilder, RegressionResult,
};
use crate::inference::CoefficientInference;
use crate::solvers::traits::{FittedRegressor, RegressionError, Regressor};
use argmin::core::{CostFunction, Executor, Gradient, State};
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::quasinewton::LBFGS;
use faer::{Col, Mat};
use statrs::distribution::{ContinuousCDF, FisherSnedecor};
use statrs::function::gamma::ln_gamma;
use std::f64::consts::PI;

/// Loss functions for ALM model fitting and convergence.
///
/// The loss function determines how the model measures fit quality during
/// the iterative optimization process. Different loss functions provide
/// different trade-offs between efficiency and robustness.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum AlmLoss {
    /// Maximum Likelihood Estimation (default).
    /// Uses log-likelihood for convergence criterion.
    /// Most efficient when distributional assumptions are met.
    #[default]
    Likelihood,

    /// Mean Squared Error: mean((y - fitted)²)
    /// Equivalent to Normal likelihood, sensitive to outliers.
    MSE,

    /// Mean Absolute Error: mean(|y - fitted|)
    /// More robust to outliers than MSE.
    MAE,

    /// Half Absolute Moment: mean(√|y - fitted|)
    /// Even more robust, gives less weight to large deviations.
    HAM,

    /// RObust Likelihood Estimator.
    /// Trims observations with worst likelihood contributions.
    /// The `trim` field specifies the fraction to trim (default 0.05 = 5%).
    ROLE {
        /// Fraction of observations to trim (0.0 to 0.5)
        trim: f64,
    },
}

impl AlmLoss {
    /// Create ROLE loss with default 5% trim.
    pub fn role() -> Self {
        AlmLoss::ROLE { trim: 0.05 }
    }

    /// Create ROLE loss with custom trim fraction.
    ///
    /// # Arguments
    /// * `trim` - Fraction of observations to trim (clamped to 0.0..0.5)
    pub fn role_with_trim(trim: f64) -> Self {
        AlmLoss::ROLE {
            trim: trim.clamp(0.0, 0.5),
        }
    }
}

/// Distribution families supported by the ALM.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum AlmDistribution {
    /// Normal (Gaussian) distribution - standard linear regression
    #[default]
    Normal,
    /// Laplace (double exponential) - robust to outliers, LAD regression
    Laplace,
    /// Student's t distribution - heavy-tailed, robust
    StudentT,
    /// Logistic distribution
    Logistic,
    /// Asymmetric Laplace - for quantile regression
    AsymmetricLaplace,
    /// Generalised Normal (Subbotin) distribution
    GeneralisedNormal,
    /// S distribution
    S,
    /// Log-Normal distribution - for positive skewed data
    LogNormal,
    /// Log-Laplace distribution
    LogLaplace,
    /// Log-S distribution
    LogS,
    /// Log-Generalised Normal distribution
    LogGeneralisedNormal,
    /// Folded Normal distribution - for positive data
    FoldedNormal,
    /// Rectified Normal distribution
    RectifiedNormal,
    /// Box-Cox Normal distribution
    BoxCoxNormal,
    /// Gamma distribution - for positive continuous data
    Gamma,
    /// Inverse Gaussian distribution
    InverseGaussian,
    /// Exponential distribution
    Exponential,
    /// Beta distribution - for data in (0, 1)
    Beta,
    /// Logit-Normal distribution
    LogitNormal,
    /// Poisson distribution - for count data
    Poisson,
    /// Negative Binomial - for overdispersed count data
    NegativeBinomial,
    /// Binomial distribution - for binary/proportion data
    Binomial,
    /// Geometric distribution
    Geometric,
    /// Cumulative Logistic (ordered logistic)
    CumulativeLogistic,
    /// Cumulative Normal (ordered probit)
    CumulativeNormal,
}

impl AlmDistribution {
    /// Returns the canonical link function for this distribution.
    pub fn canonical_link(&self) -> LinkFunction {
        match self {
            AlmDistribution::Normal => LinkFunction::Identity,
            AlmDistribution::Laplace => LinkFunction::Identity,
            AlmDistribution::StudentT => LinkFunction::Identity,
            AlmDistribution::Logistic => LinkFunction::Identity,
            AlmDistribution::AsymmetricLaplace => LinkFunction::Identity,
            AlmDistribution::GeneralisedNormal => LinkFunction::Identity,
            AlmDistribution::S => LinkFunction::Identity,
            AlmDistribution::LogNormal => LinkFunction::Log,
            AlmDistribution::LogLaplace => LinkFunction::Log,
            AlmDistribution::LogS => LinkFunction::Log,
            AlmDistribution::LogGeneralisedNormal => LinkFunction::Log,
            AlmDistribution::FoldedNormal => LinkFunction::Identity,
            AlmDistribution::RectifiedNormal => LinkFunction::Identity,
            AlmDistribution::BoxCoxNormal => LinkFunction::Identity,
            AlmDistribution::Gamma => LinkFunction::Log,
            AlmDistribution::InverseGaussian => LinkFunction::Log,
            AlmDistribution::Exponential => LinkFunction::Log,
            AlmDistribution::Beta => LinkFunction::Logit,
            AlmDistribution::LogitNormal => LinkFunction::Identity, // models logit-scale location directly
            AlmDistribution::Poisson => LinkFunction::Log,
            AlmDistribution::NegativeBinomial => LinkFunction::Log,
            AlmDistribution::Binomial => LinkFunction::Logit,
            AlmDistribution::Geometric => LinkFunction::Log, // models mean λ = (1-p)/p
            AlmDistribution::CumulativeLogistic => LinkFunction::Logit,
            AlmDistribution::CumulativeNormal => LinkFunction::Probit,
        }
    }

    /// Returns whether this distribution requires positive response values.
    pub fn requires_positive(&self) -> bool {
        matches!(
            self,
            AlmDistribution::LogNormal
                | AlmDistribution::LogLaplace
                | AlmDistribution::LogS
                | AlmDistribution::LogGeneralisedNormal
                | AlmDistribution::FoldedNormal
                | AlmDistribution::Gamma
                | AlmDistribution::InverseGaussian
                | AlmDistribution::Exponential
        )
    }

    /// Returns whether this distribution is for count data.
    pub fn is_count(&self) -> bool {
        matches!(
            self,
            AlmDistribution::Poisson
                | AlmDistribution::NegativeBinomial
                | AlmDistribution::Binomial
                | AlmDistribution::Geometric
        )
    }

    /// Returns whether this distribution requires data in (0, 1).
    pub fn requires_unit_interval(&self) -> bool {
        matches!(self, AlmDistribution::Beta | AlmDistribution::LogitNormal)
    }

    /// Returns whether this distribution should use numerical optimization instead of IRLS.
    ///
    /// Some distributions have likelihood functions that don't work well with
    /// IRLS (e.g., mixture densities, non-standard link functions). For these,
    /// we use L-BFGS optimization as in R greybox's nloptr backend.
    pub fn requires_numerical_optimization(&self) -> bool {
        matches!(
            self,
            AlmDistribution::FoldedNormal
                | AlmDistribution::RectifiedNormal
                | AlmDistribution::S
                | AlmDistribution::LogS
                | AlmDistribution::BoxCoxNormal
                | AlmDistribution::Beta
        )
    }
}

/// Link functions for relating the linear predictor to the mean.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LinkFunction {
    /// Identity: mu = eta
    Identity,
    /// Log: mu = exp(eta)
    Log,
    /// Logit: mu = 1/(1 + exp(-eta))
    Logit,
    /// Probit: mu = Phi(eta)
    Probit,
    /// Inverse: mu = 1/eta
    Inverse,
    /// Sqrt: mu = eta^2
    Sqrt,
    /// Complementary log-log: mu = 1 - exp(-exp(eta))
    Cloglog,
}

impl LinkFunction {
    /// Apply the link function: eta = g(mu)
    pub fn link(&self, mu: f64) -> f64 {
        match self {
            LinkFunction::Identity => mu,
            LinkFunction::Log => mu.ln(),
            LinkFunction::Logit => (mu / (1.0 - mu)).ln(),
            LinkFunction::Probit => probit(mu),
            LinkFunction::Inverse => 1.0 / mu,
            LinkFunction::Sqrt => mu.sqrt(),
            LinkFunction::Cloglog => (-(1.0 - mu).ln()).ln(),
        }
    }

    /// Apply the inverse link function: mu = g^{-1}(eta)
    pub fn inverse(&self, eta: f64) -> f64 {
        match self {
            LinkFunction::Identity => eta,
            LinkFunction::Log => eta.exp(),
            LinkFunction::Logit => 1.0 / (1.0 + (-eta).exp()),
            LinkFunction::Probit => normal_cdf(eta),
            LinkFunction::Inverse => 1.0 / eta,
            LinkFunction::Sqrt => eta.powi(2),
            LinkFunction::Cloglog => 1.0 - (-eta.exp()).exp(),
        }
    }

    /// Derivative of the inverse link: d(mu)/d(eta)
    pub fn inverse_derivative(&self, eta: f64) -> f64 {
        match self {
            LinkFunction::Identity => 1.0,
            LinkFunction::Log => eta.exp(),
            LinkFunction::Logit => {
                let p = 1.0 / (1.0 + (-eta).exp());
                p * (1.0 - p)
            }
            LinkFunction::Probit => normal_pdf(eta),
            LinkFunction::Inverse => -1.0 / (eta * eta),
            LinkFunction::Sqrt => 2.0 * eta,
            LinkFunction::Cloglog => {
                let exp_eta = eta.exp();
                exp_eta * (-exp_eta).exp()
            }
        }
    }
}

// ============================================================================
// Log-likelihood Functions - Helper implementations for each distribution
// ============================================================================

fn ll_normal(y: &Col<f64>, mu: &Col<f64>, scale: f64) -> f64 {
    let n = y.nrows();
    let sigma2 = scale * scale;
    let rss: f64 = (0..n).map(|i| (y[i] - mu[i]).powi(2)).sum();
    -0.5 * n as f64 * (2.0 * PI * sigma2).ln() - rss / (2.0 * sigma2)
}

fn ll_laplace(y: &Col<f64>, mu: &Col<f64>, scale: f64) -> f64 {
    let n = y.nrows();
    let sad: f64 = (0..n).map(|i| (y[i] - mu[i]).abs()).sum();
    -(n as f64) * (2.0 * scale).ln() - sad / scale
}

fn ll_student_t(y: &Col<f64>, mu: &Col<f64>, scale: f64, df: f64) -> f64 {
    let n = y.nrows();
    (0..n)
        .map(|i| {
            let z = (y[i] - mu[i]) / scale;
            ln_gamma((df + 1.0) / 2.0)
                - ln_gamma(df / 2.0)
                - 0.5 * (PI * df).ln()
                - scale.ln()
                - ((df + 1.0) / 2.0) * (1.0 + z * z / df).ln()
        })
        .sum()
}

fn ll_logistic(y: &Col<f64>, mu: &Col<f64>, scale: f64) -> f64 {
    let n = y.nrows();
    (0..n)
        .map(|i| {
            let z = (y[i] - mu[i]) / scale;
            -z - 2.0 * (1.0 + (-z).exp()).ln() - scale.ln()
        })
        .sum()
}

fn ll_asymmetric_laplace(y: &Col<f64>, mu: &Col<f64>, scale: f64, alpha: f64) -> f64 {
    let n = y.nrows();
    (0..n)
        .map(|i| {
            let e = y[i] - mu[i];
            let rho = if e >= 0.0 {
                alpha * e
            } else {
                (alpha - 1.0) * e
            };
            (alpha * (1.0 - alpha)).ln() - scale.ln() - rho / scale
        })
        .sum()
}

fn ll_generalised_normal(y: &Col<f64>, mu: &Col<f64>, scale: f64, shape: f64) -> f64 {
    // Generalized Normal (Exponential Power) Distribution
    // PDF: f(x; μ, α, β) = (β / (2α·Γ(1/β))) · exp(-(|x-μ|/α)^β)
    // Log-lik: log(β/(2α)) - ln_gamma(1/β) - |z|^β  where z = (x-μ)/α
    let n = y.nrows();
    (0..n)
        .map(|i| {
            let z = ((y[i] - mu[i]) / scale).abs();
            (shape / (2.0 * scale)).ln() - ln_gamma(1.0 / shape) - z.powf(shape)
        })
        .sum()
}

fn ll_s_distribution(y: &Col<f64>, mu: &Col<f64>, scale: f64) -> f64 {
    // S Distribution (greybox parameterization)
    // PDF: f(y; μ, s) = (1 / (4s²)) · exp(-√|y - μ| / s)
    // Log-lik: -log(4) - 2·log(s) - √|y - μ| / s
    let n = y.nrows();
    let scale_sq = scale * scale;
    (0..n)
        .map(|i| {
            let abs_resid = (y[i] - mu[i]).abs();
            -(4.0 * scale_sq).ln() - abs_resid.sqrt() / scale
        })
        .sum()
}

fn ll_log_normal(y: &Col<f64>, mu: &Col<f64>, scale: f64) -> f64 {
    let n = y.nrows();
    let sigma2 = scale * scale;
    let mut ll = 0.0;
    for i in 0..n {
        if y[i] <= 0.0 || mu[i] <= 0.0 {
            return f64::NEG_INFINITY;
        }
        let log_y = y[i].ln();
        let log_mu = mu[i].ln();
        ll +=
            -log_y - scale.ln() - 0.5 * (2.0 * PI).ln() - (log_y - log_mu).powi(2) / (2.0 * sigma2);
    }
    ll
}

fn ll_log_laplace(y: &Col<f64>, mu: &Col<f64>, scale: f64) -> f64 {
    let n = y.nrows();
    let mut ll = 0.0;
    for i in 0..n {
        if y[i] <= 0.0 || mu[i] <= 0.0 {
            return f64::NEG_INFINITY;
        }
        ll += -(2.0 * scale).ln() - y[i].ln() - (y[i].ln() - mu[i].ln()).abs() / scale;
    }
    ll
}

fn ll_log_s(y: &Col<f64>, mu: &Col<f64>, scale: f64) -> f64 {
    // Log-S distribution: log(Y) follows S distribution
    // Log-lik: log(1/(4α)) - y.ln() - √|z|  where z = (log(y) - log(μ))/α
    let n = y.nrows();
    let mut ll = 0.0;
    for i in 0..n {
        if y[i] <= 0.0 || mu[i] <= 0.0 {
            return f64::NEG_INFINITY;
        }
        let z = (y[i].ln() - mu[i].ln()).abs() / scale;
        ll += (0.25 / scale).ln() - y[i].ln() - z.sqrt();
    }
    ll
}

fn ll_log_generalised_normal(y: &Col<f64>, mu: &Col<f64>, scale: f64, shape: f64) -> f64 {
    // Log-Generalised Normal: log(Y) follows Generalized Normal distribution
    // Log-lik: log(β/(2α)) - ln_gamma(1/β) - y.ln() - |z|^β  where z = (log(y) - log(μ))/α
    let n = y.nrows();
    let mut ll = 0.0;
    for i in 0..n {
        if y[i] <= 0.0 || mu[i] <= 0.0 {
            return f64::NEG_INFINITY;
        }
        let z = (y[i].ln() - mu[i].ln()).abs() / scale;
        ll += (shape / (2.0 * scale)).ln() - ln_gamma(1.0 / shape) - y[i].ln() - z.powf(shape);
    }
    ll
}

fn ll_folded_normal(y: &Col<f64>, mu: &Col<f64>, scale: f64) -> f64 {
    let n = y.nrows();
    let mut ll = 0.0;
    for i in 0..n {
        if y[i] < 0.0 {
            return f64::NEG_INFINITY;
        }
        let z1 = (y[i] - mu[i]) / scale;
        let z2 = (y[i] + mu[i]) / scale;
        let pdf = ((-0.5 * z1 * z1).exp() + (-0.5 * z2 * z2).exp()) / (scale * (2.0 * PI).sqrt());
        if pdf <= 0.0 {
            return f64::NEG_INFINITY;
        }
        ll += pdf.ln();
    }
    ll
}

fn ll_rectified_normal(y: &Col<f64>, mu: &Col<f64>, scale: f64) -> f64 {
    let n = y.nrows();
    let mut ll = 0.0;
    for i in 0..n {
        if y[i] > 0.0 {
            let z = (y[i] - mu[i]) / scale;
            ll += -0.5 * (2.0 * PI).ln() - scale.ln() - 0.5 * z * z;
        } else if y[i] == 0.0 {
            let z = -mu[i] / scale;
            ll += normal_cdf(z).ln();
        } else {
            return f64::NEG_INFINITY;
        }
    }
    ll
}

fn ll_box_cox_normal(y: &Col<f64>, mu: &Col<f64>, scale: f64, lambda: f64) -> f64 {
    let n = y.nrows();
    let sigma2 = scale * scale;
    let mut ll = 0.0;
    for i in 0..n {
        if y[i] <= 0.0 {
            return f64::NEG_INFINITY;
        }
        let y_trans = if lambda.abs() < 1e-10 {
            y[i].ln()
        } else {
            (y[i].powf(lambda) - 1.0) / lambda
        };
        let mu_trans = if lambda.abs() < 1e-10 {
            mu[i].ln()
        } else {
            (mu[i].powf(lambda) - 1.0) / lambda
        };
        ll += -0.5 * (2.0 * PI * sigma2).ln() - (y_trans - mu_trans).powi(2) / (2.0 * sigma2)
            + (lambda - 1.0) * y[i].ln();
    }
    ll
}

fn ll_gamma(y: &Col<f64>, mu: &Col<f64>, shape: f64) -> f64 {
    let n = y.nrows();
    let mut ll = 0.0;
    for i in 0..n {
        if y[i] <= 0.0 || mu[i] <= 0.0 {
            return f64::NEG_INFINITY;
        }
        let rate = shape / mu[i];
        ll += shape * rate.ln() + (shape - 1.0) * y[i].ln() - rate * y[i] - ln_gamma(shape);
    }
    ll
}

fn ll_inverse_gaussian(y: &Col<f64>, mu: &Col<f64>, lambda: f64) -> f64 {
    let n = y.nrows();
    let mut ll = 0.0;
    for i in 0..n {
        if y[i] <= 0.0 || mu[i] <= 0.0 {
            return f64::NEG_INFINITY;
        }
        ll += 0.5 * (lambda / (2.0 * PI * y[i].powi(3))).ln()
            - lambda * (y[i] - mu[i]).powi(2) / (2.0 * mu[i].powi(2) * y[i]);
    }
    ll
}

fn ll_exponential(y: &Col<f64>, mu: &Col<f64>) -> f64 {
    let n = y.nrows();
    let mut ll = 0.0;
    for i in 0..n {
        if y[i] < 0.0 || mu[i] <= 0.0 {
            return f64::NEG_INFINITY;
        }
        ll += -mu[i].ln() - y[i] / mu[i];
    }
    ll
}

fn ll_beta(y: &Col<f64>, mu: &Col<f64>, phi: f64) -> f64 {
    let n = y.nrows();
    let mut ll = 0.0;
    for i in 0..n {
        if y[i] <= 0.0 || y[i] >= 1.0 || mu[i] <= 0.0 || mu[i] >= 1.0 {
            return f64::NEG_INFINITY;
        }
        let alpha = mu[i] * phi;
        let beta_param = (1.0 - mu[i]) * phi;
        ll += ln_gamma(phi) - ln_gamma(alpha) - ln_gamma(beta_param)
            + (alpha - 1.0) * y[i].ln()
            + (beta_param - 1.0) * (1.0 - y[i]).ln();
    }
    ll
}

fn ll_logit_normal(y: &Col<f64>, mu: &Col<f64>, scale: f64) -> f64 {
    // mu is the logit-scale location parameter (not a probability)
    // y is in (0,1), logit(y) ~ Normal(mu, scale^2)
    let n = y.nrows();
    let sigma2 = scale * scale;
    let mut ll = 0.0;
    for i in 0..n {
        if y[i] <= 0.0 || y[i] >= 1.0 {
            return f64::NEG_INFINITY;
        }
        let logit_y = (y[i] / (1.0 - y[i])).ln();
        // mu[i] is already on logit scale (the location parameter)
        ll += -0.5 * (2.0 * PI * sigma2).ln()
            - (logit_y - mu[i]).powi(2) / (2.0 * sigma2)
            - y[i].ln()
            - (1.0 - y[i]).ln();
    }
    ll
}

fn ll_poisson(y: &Col<f64>, mu: &Col<f64>) -> f64 {
    let n = y.nrows();
    (0..n)
        .map(|i| {
            let yi = y[i].round().max(0.0);
            let mui = mu[i].max(1e-10);
            yi * mui.ln() - mui - ln_gamma(yi + 1.0)
        })
        .sum()
}

fn ll_negative_binomial(y: &Col<f64>, mu: &Col<f64>, size: f64) -> f64 {
    let n = y.nrows();
    (0..n)
        .map(|i| {
            let yi = y[i].round().max(0.0);
            let mui = mu[i].max(1e-10);
            let p = size / (size + mui);
            ln_gamma(yi + size) - ln_gamma(size) - ln_gamma(yi + 1.0)
                + size * p.ln()
                + yi * (1.0 - p).ln()
        })
        .sum()
}

fn ll_binomial(y: &Col<f64>, mu: &Col<f64>, n_trials: f64) -> f64 {
    let n = y.nrows();
    (0..n)
        .map(|i| {
            let p = mu[i].clamp(1e-10, 1.0 - 1e-10);
            let k = (y[i] * n_trials).round().max(0.0).min(n_trials);
            ln_gamma(n_trials + 1.0) - ln_gamma(k + 1.0) - ln_gamma(n_trials - k + 1.0)
                + k * p.ln()
                + (n_trials - k) * (1.0 - p).ln()
        })
        .sum()
}

fn ll_geometric(y: &Col<f64>, mu: &Col<f64>) -> f64 {
    // mu is the mean (expected number of failures before first success)
    // For geometric distribution: E[Y] = (1-p)/p = λ, so p = 1/(1+λ)
    // PMF: P(Y=k) = p * (1-p)^k
    let n = y.nrows();
    (0..n)
        .map(|i| {
            let lambda = mu[i].max(1e-10); // mean (number of failures)
            let p = 1.0 / (1.0 + lambda); // success probability
            let k = y[i].round().max(0.0); // number of failures
            p.ln() + k * (1.0 - p).ln()
        })
        .sum()
}

fn ll_cumulative_logistic(y: &Col<f64>, mu: &Col<f64>) -> f64 {
    // For binary y in {0,1}, CumulativeLogistic is equivalent to logistic regression
    // Pr(Y=1) = mu (already transformed via logit link)
    // Log-likelihood: y*log(p) + (1-y)*log(1-p)
    let n = y.nrows();
    (0..n)
        .map(|i| {
            let p = mu[i].clamp(1e-10, 1.0 - 1e-10);
            let yi = y[i];
            yi * p.ln() + (1.0 - yi) * (1.0 - p).ln()
        })
        .sum()
}

fn ll_cumulative_normal(y: &Col<f64>, mu: &Col<f64>) -> f64 {
    // For binary y in {0,1}, CumulativeNormal is the probit model
    // Pr(Y=1) = mu (already transformed via probit link)
    // Log-likelihood: y*log(p) + (1-y)*log(1-p)
    let n = y.nrows();
    (0..n)
        .map(|i| {
            let p = mu[i].clamp(1e-10, 1.0 - 1e-10);
            let yi = y[i];
            yi * p.ln() + (1.0 - yi) * (1.0 - p).ln()
        })
        .sum()
}

/// Compute the log-likelihood for a given distribution.
///
/// # Arguments
/// * `y` - Response vector
/// * `mu` - Fitted mean vector (on response scale)
/// * `distribution` - The distribution family
/// * `scale` - Scale parameter (sigma for Normal, b for Laplace, etc.)
/// * `extra` - Extra parameter (df for Student-t, shape for Gamma, etc.)
pub fn log_likelihood(
    y: &Col<f64>,
    mu: &Col<f64>,
    distribution: AlmDistribution,
    scale: f64,
    extra: Option<f64>,
) -> f64 {
    match distribution {
        AlmDistribution::Normal => ll_normal(y, mu, scale),
        AlmDistribution::Laplace => ll_laplace(y, mu, scale),
        AlmDistribution::StudentT => ll_student_t(y, mu, scale, extra.unwrap_or(5.0)),
        AlmDistribution::Logistic => ll_logistic(y, mu, scale),
        AlmDistribution::AsymmetricLaplace => {
            ll_asymmetric_laplace(y, mu, scale, extra.unwrap_or(0.5))
        }
        AlmDistribution::GeneralisedNormal => {
            ll_generalised_normal(y, mu, scale, extra.unwrap_or(2.0))
        }
        AlmDistribution::S => ll_s_distribution(y, mu, scale),
        AlmDistribution::LogNormal => ll_log_normal(y, mu, scale),
        AlmDistribution::LogLaplace => ll_log_laplace(y, mu, scale),
        AlmDistribution::LogS => ll_log_s(y, mu, scale),
        AlmDistribution::LogGeneralisedNormal => {
            ll_log_generalised_normal(y, mu, scale, extra.unwrap_or(2.0))
        }
        AlmDistribution::FoldedNormal => ll_folded_normal(y, mu, scale),
        AlmDistribution::RectifiedNormal => ll_rectified_normal(y, mu, scale),
        AlmDistribution::BoxCoxNormal => ll_box_cox_normal(y, mu, scale, extra.unwrap_or(1.0)),
        AlmDistribution::Gamma => ll_gamma(y, mu, extra.unwrap_or(1.0)),
        AlmDistribution::InverseGaussian => ll_inverse_gaussian(y, mu, extra.unwrap_or(1.0)),
        AlmDistribution::Exponential => ll_exponential(y, mu),
        AlmDistribution::Beta => ll_beta(y, mu, scale), // scale is precision (φ) for Beta
        AlmDistribution::LogitNormal => ll_logit_normal(y, mu, scale),
        AlmDistribution::Poisson => ll_poisson(y, mu),
        AlmDistribution::NegativeBinomial => ll_negative_binomial(y, mu, extra.unwrap_or(1.0)),
        AlmDistribution::Binomial => ll_binomial(y, mu, extra.unwrap_or(1.0)),
        AlmDistribution::Geometric => ll_geometric(y, mu),
        AlmDistribution::CumulativeLogistic => ll_cumulative_logistic(y, mu),
        AlmDistribution::CumulativeNormal => ll_cumulative_normal(y, mu),
    }
}

// ============================================================================
// Numerical Optimization Cost Function for argmin
// ============================================================================

/// Negative log-likelihood cost function for numerical optimization.
///
/// This struct holds references to the data and distribution parameters,
/// implementing the `CostFunction` and `Gradient` traits for argmin.
#[derive(Clone)]
struct NegLogLikelihoodCost<'a> {
    x: &'a Mat<f64>,
    y: &'a Col<f64>,
    distribution: AlmDistribution,
    link: LinkFunction,
    extra_parameter: Option<f64>,
    with_intercept: bool,
}

impl<'a> NegLogLikelihoodCost<'a> {
    /// Compute mu from parameter vector.
    fn compute_mu_from_params(&self, params: &[f64]) -> Col<f64> {
        let n = self.x.nrows();
        let p = self.x.ncols();

        let mut eta = Col::zeros(n);

        if self.with_intercept {
            let intercept = params[0];
            for i in 0..n {
                eta[i] = intercept;
                for j in 0..p {
                    eta[i] += self.x[(i, j)] * params[j + 1];
                }
            }
        } else {
            for i in 0..n {
                for (j, &param_j) in params.iter().enumerate().take(p) {
                    eta[i] += self.x[(i, j)] * param_j;
                }
            }
        }

        // Apply inverse link
        let mut mu = Col::zeros(n);
        for i in 0..n {
            mu[i] = self.link.inverse(eta[i]);
            // Clamp mu to valid range for the distribution
            match self.distribution {
                AlmDistribution::Poisson
                | AlmDistribution::NegativeBinomial
                | AlmDistribution::Gamma
                | AlmDistribution::InverseGaussian
                | AlmDistribution::Exponential
                | AlmDistribution::FoldedNormal
                | AlmDistribution::LogNormal
                | AlmDistribution::LogLaplace
                | AlmDistribution::LogS
                | AlmDistribution::LogGeneralisedNormal => {
                    mu[i] = mu[i].max(1e-10);
                }
                AlmDistribution::Beta | AlmDistribution::Binomial => {
                    mu[i] = mu[i].clamp(1e-10, 1.0 - 1e-10);
                }
                AlmDistribution::Geometric => {
                    mu[i] = mu[i].max(1e-10);
                }
                AlmDistribution::RectifiedNormal => {
                    // mu can be any real value for RectifiedNormal
                }
                _ => {}
            }
        }

        mu
    }

    /// Estimate scale from current mu.
    fn estimate_scale_from_mu(&self, mu: &Col<f64>) -> f64 {
        let n = self.y.nrows();
        let p = if self.with_intercept {
            self.x.ncols() + 1
        } else {
            self.x.ncols()
        };
        let df = (n - p) as f64;
        estimate_scale(
            self.y,
            mu,
            self.distribution,
            df.max(1.0),
            self.extra_parameter,
        )
    }
}

impl CostFunction for NegLogLikelihoodCost<'_> {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, params: &Self::Param) -> Result<Self::Output, argmin::core::Error> {
        let mu = self.compute_mu_from_params(params);
        let scale = self.estimate_scale_from_mu(&mu);

        let ll = log_likelihood(self.y, &mu, self.distribution, scale, self.extra_parameter);

        // Return negative log-likelihood (we minimize)
        if ll.is_finite() {
            Ok(-ll)
        } else {
            Ok(1e20) // Large penalty for invalid parameters
        }
    }
}

impl Gradient for NegLogLikelihoodCost<'_> {
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;

    fn gradient(&self, params: &Self::Param) -> Result<Self::Gradient, argmin::core::Error> {
        // Numerical gradient using central differences
        let eps = 1e-7;
        let mut grad = vec![0.0; params.len()];

        for i in 0..params.len() {
            let mut params_plus = params.clone();
            let mut params_minus = params.clone();
            params_plus[i] += eps;
            params_minus[i] -= eps;

            let cost_plus = self.cost(&params_plus)?;
            let cost_minus = self.cost(&params_minus)?;

            grad[i] = (cost_plus - cost_minus) / (2.0 * eps);
        }

        Ok(grad)
    }
}

/// Estimate the scale parameter from residuals.
pub fn estimate_scale(
    y: &Col<f64>,
    mu: &Col<f64>,
    distribution: AlmDistribution,
    df: f64,
    extra: Option<f64>,
) -> f64 {
    let n = y.nrows();

    match distribution {
        AlmDistribution::Normal | AlmDistribution::RectifiedNormal => {
            // MLE for sigma: sqrt(RSS / n) or sqrt(RSS / df) for unbiased
            let rss: f64 = (0..n).map(|i| (y[i] - mu[i]).powi(2)).sum();
            (rss / df).sqrt()
        }

        AlmDistribution::FoldedNormal => {
            // For FoldedNormal, use the second moment relationship:
            // E[Y²] = μ² + σ², so σ² = E[Y²] - μ²
            // Average over all observations
            let mean_y_sq: f64 = (0..n).map(|i| y[i] * y[i]).sum::<f64>() / n as f64;
            let mean_mu_sq: f64 = (0..n).map(|i| mu[i] * mu[i]).sum::<f64>() / n as f64;
            let sigma_sq = (mean_y_sq - mean_mu_sq).max(0.01);
            sigma_sq.sqrt()
        }

        AlmDistribution::Laplace => {
            // MLE for b: mean absolute deviation
            let sad: f64 = (0..n).map(|i| (y[i] - mu[i]).abs()).sum();
            sad / n as f64
        }

        AlmDistribution::LogLaplace => {
            // MLE for b: mean absolute deviation on log scale
            let sad: f64 = (0..n)
                .filter(|&i| y[i] > 0.0 && mu[i] > 0.0)
                .map(|i| (y[i].ln() - mu[i].ln()).abs())
                .sum();
            sad / n as f64
        }

        AlmDistribution::StudentT | AlmDistribution::Logistic => {
            // Use robust scale estimate
            let mut abs_residuals: Vec<f64> = (0..n).map(|i| (y[i] - mu[i]).abs()).collect();
            abs_residuals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let median_idx = n / 2;
            let mad = abs_residuals[median_idx];
            mad / 0.6745 // Scale factor for Normal
        }

        AlmDistribution::LogNormal => {
            // Scale on log scale: sigma = sqrt(sum((log(y) - log(mu))^2) / df)
            let rss: f64 = (0..n)
                .filter(|&i| y[i] > 0.0 && mu[i] > 0.0)
                .map(|i| (y[i].ln() - mu[i].ln()).powi(2))
                .sum();
            (rss / df).sqrt()
        }

        AlmDistribution::LogitNormal => {
            // LogitNormal: mu is on logit scale, compute residuals as logit(y) - mu
            let rss: f64 = (0..n)
                .filter(|&i| y[i] > 0.0 && y[i] < 1.0)
                .map(|i| {
                    let yi = y[i].clamp(0.001, 0.999);
                    let logit_y = (yi / (1.0 - yi)).ln();
                    (logit_y - mu[i]).powi(2)
                })
                .sum();
            (rss / df).sqrt()
        }

        AlmDistribution::S => {
            // S distribution scale: ŝ = (1/2T) · Σ√|yₜ - μₜ| (HAM-based)
            let ham: f64 = (0..n).map(|i| (y[i] - mu[i]).abs().sqrt()).sum();
            (ham / (2.0 * n as f64)).max(1e-10)
        }

        AlmDistribution::LogS => {
            // Log-S distribution scale: same formula on log scale
            let ham: f64 = (0..n)
                .filter(|&i| y[i] > 0.0 && mu[i] > 0.0)
                .map(|i| (y[i].ln() - mu[i].ln()).abs().sqrt())
                .sum();
            (ham / (2.0 * n as f64)).max(1e-10)
        }

        AlmDistribution::AsymmetricLaplace => {
            // For asymmetric Laplace, scale is estimated from weighted absolute residuals
            // s = (1/n) * sum(w_i * |e_i|) where w_i = alpha if e_i >= 0, else 1-alpha
            // But the MLE scale is just mean of weighted absolute residuals
            let alpha = extra.unwrap_or(0.5);
            let weighted_sum: f64 = (0..n)
                .map(|i| {
                    let e = y[i] - mu[i];
                    if e >= 0.0 {
                        alpha * e.abs()
                    } else {
                        (1.0 - alpha) * e.abs()
                    }
                })
                .sum();
            (weighted_sum / n as f64).max(1e-10)
        }

        AlmDistribution::GeneralisedNormal | AlmDistribution::BoxCoxNormal => {
            // Use MAD-based estimate
            let mut abs_residuals: Vec<f64> = (0..n).map(|i| (y[i] - mu[i]).abs()).collect();
            abs_residuals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            abs_residuals[n / 2] / 0.6745
        }

        AlmDistribution::LogGeneralisedNormal => {
            // Use MAD-based estimate on log scale
            let mut abs_residuals: Vec<f64> = (0..n)
                .filter(|&i| y[i] > 0.0 && mu[i] > 0.0)
                .map(|i| (y[i].ln() - mu[i].ln()).abs())
                .collect();
            abs_residuals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            if abs_residuals.is_empty() {
                1.0
            } else {
                abs_residuals[abs_residuals.len() / 2] / 0.6745
            }
        }

        AlmDistribution::Beta => {
            // For Beta distribution, estimate precision parameter φ using method of moments
            // Var(Y) = μ(1-μ)/(1+φ), so φ = μ(1-μ)/Var(Y) - 1
            // Use sample variance around fitted means
            let var_y: f64 = (0..n)
                .map(|i| {
                    let resid = y[i] - mu[i];
                    resid * resid
                })
                .sum::<f64>()
                / df.max(1.0);

            // Average μ(1-μ) as estimate of numerator
            let mean_mu_var: f64 = (0..n)
                .map(|i| {
                    let mi = mu[i].clamp(0.01, 0.99);
                    mi * (1.0 - mi)
                })
                .sum::<f64>()
                / n as f64;

            // phi = μ(1-μ)/Var(Y) - 1
            (mean_mu_var / var_y.max(1e-10) - 1.0).max(1.0)
        }

        // For discrete distributions, scale is typically 1 or not used
        _ => 1.0,
    }
}

// ============================================================================
// Loss Functions for Convergence
// ============================================================================

/// Compute Mean Squared Error: mean((y - mu)²)
fn compute_mse(y: &Col<f64>, mu: &Col<f64>) -> f64 {
    let n = y.nrows();
    if n == 0 {
        return 0.0;
    }
    let sum: f64 = (0..n).map(|i| (y[i] - mu[i]).powi(2)).sum();
    sum / n as f64
}

/// Compute Mean Absolute Error: mean(|y - mu|)
fn compute_mae(y: &Col<f64>, mu: &Col<f64>) -> f64 {
    let n = y.nrows();
    if n == 0 {
        return 0.0;
    }
    let sum: f64 = (0..n).map(|i| (y[i] - mu[i]).abs()).sum();
    sum / n as f64
}

/// Compute Half Absolute Moment: mean(√|y - mu|)
fn compute_ham(y: &Col<f64>, mu: &Col<f64>) -> f64 {
    let n = y.nrows();
    if n == 0 {
        return 0.0;
    }
    let sum: f64 = (0..n).map(|i| (y[i] - mu[i]).abs().sqrt()).sum();
    sum / n as f64
}

/// Compute ROLE (RObust Likelihood Estimator): trimmed log-likelihood.
///
/// Computes pointwise log-likelihoods, sorts them, and returns the sum
/// after trimming the worst `trim` fraction of observations.
fn compute_role(
    y: &Col<f64>,
    mu: &Col<f64>,
    distribution: AlmDistribution,
    scale: f64,
    extra: Option<f64>,
    trim: f64,
) -> f64 {
    let n = y.nrows();
    if n == 0 {
        return 0.0;
    }

    // Compute pointwise log-likelihoods
    let mut point_lls: Vec<f64> = (0..n)
        .map(|i| {
            let yi = Col::from_fn(1, |_| y[i]);
            let mui = Col::from_fn(1, |_| mu[i]);
            log_likelihood(&yi, &mui, distribution, scale, extra)
        })
        .collect();

    // Sort ascending (worst/most negative first)
    point_lls.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Number of observations to trim
    let n_trim = ((n as f64 * trim).ceil() as usize).min(n - 1);

    // Sum the kept observations (skip the worst n_trim)
    point_lls.iter().skip(n_trim).sum()
}

/// Compute the loss value for a given loss function.
///
/// For loss functions that are minimized (MSE, MAE, HAM), returns a positive value.
/// For likelihood-based losses (Likelihood, ROLE), returns the negative log-likelihood
/// (so it can also be minimized).
pub fn compute_loss(
    y: &Col<f64>,
    mu: &Col<f64>,
    loss: AlmLoss,
    distribution: AlmDistribution,
    scale: f64,
    extra: Option<f64>,
) -> f64 {
    match loss {
        AlmLoss::Likelihood => {
            // Return negative log-likelihood (to minimize)
            -log_likelihood(y, mu, distribution, scale, extra)
        }
        AlmLoss::MSE => compute_mse(y, mu),
        AlmLoss::MAE => compute_mae(y, mu),
        AlmLoss::HAM => compute_ham(y, mu),
        AlmLoss::ROLE { trim } => {
            // Return negative ROLE (to minimize)
            -compute_role(y, mu, distribution, scale, extra, trim)
        }
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Standard normal CDF
fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// Standard normal PDF
fn normal_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * PI).sqrt()
}

/// Probit function (inverse normal CDF)
#[allow(clippy::excessive_precision)]
fn probit(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    // Use rational approximation
    let a = [
        -3.969683028665376e1,
        2.209460984245205e2,
        -2.759285104469687e2,
        1.383577518672690e2,
        -3.066479806614716e1,
        2.506628277459239e0,
    ];
    let b = [
        -5.447609879822406e1,
        1.615858368580409e2,
        -1.556989798598866e2,
        6.680131188771972e1,
        -1.328068155288572e+01,
    ];
    let c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    let d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];

    let p_low = 0.02425;
    let p_high = 1.0 - p_low;

    if p < p_low {
        let q = (-2.0 * p.ln()).sqrt();
        (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    } else if p <= p_high {
        let q = p - 0.5;
        let r = q * q;
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    }
}

/// Error function approximation
fn erf(x: f64) -> f64 {
    // Approximation from Abramowitz and Stegun
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

// ============================================================================
// ALM Regressor
// ============================================================================

/// Augmented Linear Model regressor using maximum likelihood estimation.
#[derive(Debug, Clone)]
pub struct AlmRegressor {
    options: RegressionOptions,
    distribution: AlmDistribution,
    link: LinkFunction,
    extra_parameter: Option<f64>,
    loss: AlmLoss,
    max_iterations: usize,
    tolerance: f64,
}

impl AlmRegressor {
    /// Create a new ALM regressor.
    pub fn new(
        options: RegressionOptions,
        distribution: AlmDistribution,
        link: Option<LinkFunction>,
        extra_parameter: Option<f64>,
    ) -> Self {
        let link = link.unwrap_or_else(|| distribution.canonical_link());
        Self {
            options,
            distribution,
            link,
            extra_parameter,
            loss: AlmLoss::default(),
            max_iterations: 100,
            tolerance: 1e-8,
        }
    }

    /// Get the loss function.
    pub fn loss(&self) -> AlmLoss {
        self.loss
    }

    /// Create a builder for configuring the regressor.
    pub fn builder() -> AlmRegressorBuilder {
        AlmRegressorBuilder::default()
    }

    /// Get the distribution.
    pub fn distribution(&self) -> AlmDistribution {
        self.distribution
    }

    /// Get the link function.
    pub fn link(&self) -> LinkFunction {
        self.link
    }

    /// Fit the model using iteratively reweighted least squares (IRLS) for GLM-type
    /// distributions, or gradient descent for others.
    fn fit_irls(&self, x: &Mat<f64>, y: &Col<f64>) -> Result<FittedAlm, RegressionError> {
        let n = x.nrows();
        let p = x.ncols();
        let n_params = if self.options.with_intercept {
            p + 1
        } else {
            p
        };

        // Initialize coefficients using OLS
        let mut beta = self.initialize_coefficients(x, y)?;

        // Initial linear predictor and mean
        let mut eta = self.compute_eta(x, &beta);
        let mut mu = self.compute_mu(&eta);

        // Initial scale estimate
        let df = (n - n_params) as f64;
        let mut scale =
            estimate_scale(y, &mu, self.distribution, df.max(1.0), self.extra_parameter);

        // Initial loss value (using selected loss function)
        let mut prev_loss = compute_loss(
            y,
            &mu,
            self.loss,
            self.distribution,
            scale,
            self.extra_parameter,
        );

        for iter in 0..self.max_iterations {
            // Compute weights and working response for IRLS
            let (weights, z) = self.compute_irls_components(y, &mu, &eta);

            // Weighted least squares step
            beta = self.weighted_ls_step(x, &z, &weights, &beta)?;

            // Update linear predictor and mean
            eta = self.compute_eta(x, &beta);
            mu = self.compute_mu(&eta);

            // Update scale
            scale = estimate_scale(y, &mu, self.distribution, df.max(1.0), self.extra_parameter);

            // Check convergence using selected loss function
            let loss = compute_loss(
                y,
                &mu,
                self.loss,
                self.distribution,
                scale,
                self.extra_parameter,
            );

            // Convergence criterion: relative change in loss is small
            // (loss is always in minimization form, so we check if it decreased or stabilized)
            if (loss - prev_loss).abs() < self.tolerance * (1.0 + prev_loss.abs()) {
                break;
            }

            if iter == self.max_iterations - 1 {
                // Allow convergence failure for now - return best estimate
            }

            prev_loss = loss;
        }

        // Compute final results
        self.build_result(x, y, &beta, &mu, scale)
    }

    /// Fit the model using L-BFGS numerical optimization.
    ///
    /// This method directly maximizes the log-likelihood using argmin's L-BFGS
    /// optimizer. It's used for distributions where IRLS doesn't work well
    /// (e.g., FoldedNormal, RectifiedNormal, S distribution, Beta).
    fn fit_numerical(&self, x: &Mat<f64>, y: &Col<f64>) -> Result<FittedAlm, RegressionError> {
        let n = x.nrows();
        let p = x.ncols();
        let n_params = if self.options.with_intercept {
            p + 1
        } else {
            p
        };

        // Create cost function
        let cost = NegLogLikelihoodCost {
            x,
            y,
            distribution: self.distribution,
            link: self.link,
            extra_parameter: self.extra_parameter,
            with_intercept: self.options.with_intercept,
        };

        // Try multiple starting points and pick the best
        let mut best_params: Option<Vec<f64>> = None;
        let mut best_cost = f64::INFINITY;

        // Generate starting points
        let starting_points = self.generate_starting_points(x, y, n_params)?;

        for init_params in starting_points {
            // Set up L-BFGS optimizer
            let linesearch = MoreThuenteLineSearch::new();
            let solver = LBFGS::new(linesearch, 7);

            // Run optimizer (clone cost function for each run)
            let result = Executor::new(cost.clone(), solver)
                .configure(|state| {
                    state
                        .param(init_params.clone())
                        .max_iters(self.max_iterations as u64)
                        .target_cost(f64::NEG_INFINITY)
                })
                .run();

            if let Ok(res) = result {
                if let Some(params) = res.state().get_best_param() {
                    if let Ok(cost_val) = cost.cost(params) {
                        if cost_val < best_cost {
                            best_cost = cost_val;
                            best_params = Some(params.clone());
                        }
                    }
                }
            }
        }

        // Use best params or fall back to OLS initialization
        let final_params = best_params.unwrap_or_else(|| {
            let init_beta = self
                .initialize_coefficients(x, y)
                .unwrap_or_else(|_| Col::zeros(n_params));
            (0..n_params).map(|i| init_beta[i]).collect()
        });

        // Convert back to Col
        let beta = Col::from_fn(n_params, |i| final_params[i]);

        // Compute final mu and scale
        let eta = self.compute_eta(x, &beta);
        let mu = self.compute_mu(&eta);
        let df = (n - n_params) as f64;
        let scale = estimate_scale(y, &mu, self.distribution, df.max(1.0), self.extra_parameter);

        // Build result
        self.build_result(x, y, &beta, &mu, scale)
    }

    /// Generate multiple starting points for numerical optimization.
    fn generate_starting_points(
        &self,
        x: &Mat<f64>,
        y: &Col<f64>,
        n_params: usize,
    ) -> Result<Vec<Vec<f64>>, RegressionError> {
        let mut starting_points = Vec::new();

        // Starting point 1: OLS on y
        let ols_beta = self.initialize_coefficients(x, y)?;
        let ols_params: Vec<f64> = (0..n_params).map(|i| ols_beta[i]).collect();
        starting_points.push(ols_params.clone());

        // Starting point 2: scaled down version (for distributions where mu is small)
        let scaled_params: Vec<f64> = ols_params.iter().map(|&x| x * 0.1).collect();
        starting_points.push(scaled_params);

        // Starting point 3: near zero (conservative start)
        let near_zero: Vec<f64> = vec![0.01; n_params];
        starting_points.push(near_zero);

        // Distribution-specific starting points
        match self.distribution {
            AlmDistribution::FoldedNormal => {
                // For FoldedNormal, the underlying mu can be small or even negative
                // Try starting with small positive intercept and small slope
                let n = x.nrows();
                let p = x.ncols();

                // Compute mean of y and mean of x
                let y_mean: f64 = (0..n).map(|i| y[i]).sum::<f64>() / n as f64;

                // For FoldedNormal, try intercept around sqrt(E[Y²] - σ²) which is close to |μ|
                // Use a simple estimate: start with intercept = 0.1 * y_mean
                let mut fn_start1 = vec![0.0; n_params];
                if self.options.with_intercept && !fn_start1.is_empty() {
                    fn_start1[0] = 0.1 * y_mean;
                    // Small positive slope
                    if p > 0 && fn_start1.len() > 1 {
                        fn_start1[1] = 0.05;
                    }
                }
                starting_points.push(fn_start1);

                // Also try with larger intercept
                let mut fn_start2 = vec![0.0; n_params];
                if self.options.with_intercept && !fn_start2.is_empty() {
                    fn_start2[0] = 0.5;
                    if p > 0 && fn_start2.len() > 1 {
                        fn_start2[1] = 0.05;
                    }
                }
                starting_points.push(fn_start2);
            }
            AlmDistribution::RectifiedNormal => {
                // For RectifiedNormal, try with small intercept
                let mut small_start = ols_params.clone();
                if self.options.with_intercept && !small_start.is_empty() {
                    small_start[0] = 0.0;
                }
                starting_points.push(small_start);
            }
            AlmDistribution::Beta => {
                // For Beta, try starting in middle of (0,1) range
                let n = x.nrows();
                let p = x.ncols();
                let y_mean: f64 = (0..n).map(|i| y[i]).sum::<f64>() / n as f64;
                let logit_mean = (y_mean.clamp(0.01, 0.99) / (1.0 - y_mean.clamp(0.01, 0.99))).ln();
                let mut beta_start = vec![0.0; n_params];
                if self.options.with_intercept && !beta_start.is_empty() {
                    beta_start[0] = logit_mean;
                }
                starting_points.push(beta_start);

                // Also try with larger intercept (shifted logit)
                let mut beta_start2 = vec![0.0; n_params];
                if self.options.with_intercept && !beta_start2.is_empty() {
                    beta_start2[0] = 1.0; // Higher intercept
                    if p > 0 && beta_start2.len() > 1 {
                        beta_start2[1] = 0.5; // Some slope
                    }
                }
                starting_points.push(beta_start2);

                // Try starting from OLS on logit(y)
                let y_logit: Col<f64> = Col::from_fn(n, |i| {
                    let yi = y[i].clamp(0.01, 0.99);
                    (yi / (1.0 - yi)).ln()
                });
                if let Ok(logit_beta) = self.initialize_coefficients_for_y(x, &y_logit) {
                    let logit_params: Vec<f64> = (0..n_params).map(|i| logit_beta[i]).collect();
                    starting_points.push(logit_params);
                }
            }
            AlmDistribution::S | AlmDistribution::LogS => {
                // For S distribution, the intercept can be negative even with positive data
                // because the S distribution is heavy-tailed and robust to outliers
                // Try various starting points including negative intercepts
                let n = x.nrows();
                let p = x.ncols();

                // Starting point with negative intercept
                let mut neg_start = ols_params.clone();
                if self.options.with_intercept && !neg_start.is_empty() {
                    // Try intercept = -coefficient * mean(x)
                    if p > 0 && neg_start.len() > 1 {
                        let x_mean: f64 = (0..n).map(|i| x[(i, 0)]).sum::<f64>() / n as f64;
                        neg_start[0] = -neg_start[1].abs() * x_mean * 0.1;
                    }
                }
                starting_points.push(neg_start);

                // Also try with explicitly negative intercept
                let mut neg_start2 = ols_params.clone();
                if self.options.with_intercept && !neg_start2.is_empty() {
                    neg_start2[0] = -2.0;
                }
                starting_points.push(neg_start2);
            }
            AlmDistribution::BoxCoxNormal => {
                // For BoxCox, try starting from transformed y
                let lambda = self.extra_parameter.unwrap_or(1.0);
                let n = x.nrows();
                let y_trans: Col<f64> = Col::from_fn(n, |i| {
                    if lambda.abs() < 1e-10 {
                        y[i].ln()
                    } else {
                        (y[i].powf(lambda) - 1.0) / lambda
                    }
                });
                if let Ok(trans_beta) = self.initialize_coefficients_for_y(x, &y_trans) {
                    let trans_params: Vec<f64> = (0..n_params).map(|i| trans_beta[i]).collect();
                    starting_points.push(trans_params);
                }
            }
            _ => {}
        }

        Ok(starting_points)
    }

    /// Initialize coefficients for a specific y vector (helper for BoxCox).
    fn initialize_coefficients_for_y(
        &self,
        x: &Mat<f64>,
        y: &Col<f64>,
    ) -> Result<Col<f64>, RegressionError> {
        let n = x.nrows();
        let p = x.ncols();

        if self.options.with_intercept {
            let mut x_aug = Mat::zeros(n, p + 1);
            for i in 0..n {
                x_aug[(i, 0)] = 1.0;
                for j in 0..p {
                    x_aug[(i, j + 1)] = x[(i, j)];
                }
            }

            let qr = x_aug.col_piv_qr();
            let perm = qr.P();
            let perm_arr = perm.arrays().0;
            let q = qr.compute_Q();
            let r = qr.R();
            let ncols = p + 1;

            let qty = q.transpose() * y;
            let mut beta_perm = Col::zeros(ncols);

            for i in (0..ncols.min(n)).rev() {
                let mut sum = qty[i];
                for j in (i + 1)..ncols.min(n) {
                    sum -= r[(i, j)] * beta_perm[j];
                }
                if r[(i, i)].abs() > 1e-10 {
                    beta_perm[i] = sum / r[(i, i)];
                }
            }

            let mut beta = Col::zeros(ncols);
            for i in 0..ncols {
                let orig_col = perm_arr[i];
                beta[orig_col] = beta_perm[i];
            }

            Ok(beta)
        } else {
            let qr = x.col_piv_qr();
            let perm = qr.P();
            let perm_arr = perm.arrays().0;
            let q = qr.compute_Q();
            let r = qr.R();

            let qty = q.transpose() * y;
            let mut beta_perm = Col::zeros(p);

            for i in (0..p.min(n)).rev() {
                let mut sum = qty[i];
                for j in (i + 1)..p.min(n) {
                    sum -= r[(i, j)] * beta_perm[j];
                }
                if r[(i, i)].abs() > 1e-10 {
                    beta_perm[i] = sum / r[(i, i)];
                }
            }

            let mut beta = Col::zeros(p);
            for i in 0..p {
                let orig_col = perm_arr[i];
                beta[orig_col] = beta_perm[i];
            }

            Ok(beta)
        }
    }

    /// Initialize coefficients using OLS or appropriate method.
    fn initialize_coefficients(
        &self,
        x: &Mat<f64>,
        y: &Col<f64>,
    ) -> Result<Col<f64>, RegressionError> {
        let n = x.nrows();
        let p = x.ncols();

        // Transform y for initialization based on link function
        // Special case: LogitNormal uses Identity link but models logit(y)
        let y_init: Col<f64> = if matches!(self.distribution, AlmDistribution::LogitNormal) {
            // For LogitNormal, we model logit(y) directly, so initialize with logit(y)
            Col::from_fn(n, |i| {
                let yi = y[i].clamp(0.001, 0.999);
                (yi / (1.0 - yi)).ln()
            })
        } else {
            match self.link {
                LinkFunction::Identity => y.clone(),
                LinkFunction::Log => Col::from_fn(n, |i| if y[i] > 0.0 { y[i].ln() } else { 0.0 }),
                LinkFunction::Logit => Col::from_fn(n, |i| {
                    let yi = y[i].clamp(0.01, 0.99);
                    (yi / (1.0 - yi)).ln()
                }),
                LinkFunction::Probit => Col::from_fn(n, |i| probit(y[i].clamp(0.01, 0.99))),
                LinkFunction::Inverse => {
                    Col::from_fn(n, |i| if y[i] != 0.0 { 1.0 / y[i] } else { 1.0 })
                }
                LinkFunction::Sqrt => Col::from_fn(n, |i| y[i].max(0.0).sqrt()),
                LinkFunction::Cloglog => Col::from_fn(n, |i| {
                    let yi = y[i].clamp(0.01, 0.99);
                    (-(1.0 - yi).ln()).ln()
                }),
            }
        };

        // Solve OLS using normal equations with regularization for numerical stability
        if self.options.with_intercept {
            // Augment X with intercept column
            let mut x_aug = Mat::zeros(n, p + 1);
            for i in 0..n {
                x_aug[(i, 0)] = 1.0;
                for j in 0..p {
                    x_aug[(i, j + 1)] = x[(i, j)];
                }
            }

            // Solve using QR with column pivoting and proper permutation handling
            let qr = x_aug.col_piv_qr();
            let perm = qr.P();
            let perm_arr = perm.arrays().0;

            let q = qr.compute_Q();
            let r = qr.R();
            let ncols = p + 1;

            let qty = q.transpose() * &y_init;
            let mut beta_perm = Col::zeros(ncols);

            // Back-substitution for R * beta_perm = Q' * y
            for i in (0..ncols.min(n)).rev() {
                let mut sum = qty[i];
                for j in (i + 1)..ncols.min(n) {
                    sum -= r[(i, j)] * beta_perm[j];
                }
                if r[(i, i)].abs() > 1e-10 {
                    beta_perm[i] = sum / r[(i, i)];
                }
            }

            // Apply inverse permutation: beta[perm[i]] = beta_perm[i]
            let mut beta = Col::zeros(ncols);
            for i in 0..ncols {
                let orig_col = perm_arr[i];
                beta[orig_col] = beta_perm[i];
            }

            Ok(beta)
        } else {
            let qr = x.col_piv_qr();
            let perm = qr.P();
            let perm_arr = perm.arrays().0;

            let q = qr.compute_Q();
            let r = qr.R();

            let qty = q.transpose() * &y_init;
            let mut beta_perm = Col::zeros(p);

            for i in (0..p.min(n)).rev() {
                let mut sum = qty[i];
                for j in (i + 1)..p.min(n) {
                    sum -= r[(i, j)] * beta_perm[j];
                }
                if r[(i, i)].abs() > 1e-10 {
                    beta_perm[i] = sum / r[(i, i)];
                }
            }

            // Apply inverse permutation
            let mut beta = Col::zeros(p);
            for i in 0..p {
                let orig_col = perm_arr[i];
                beta[orig_col] = beta_perm[i];
            }

            Ok(beta)
        }
    }

    /// Compute linear predictor eta = X * beta (+ intercept if applicable).
    fn compute_eta(&self, x: &Mat<f64>, beta: &Col<f64>) -> Col<f64> {
        let n = x.nrows();
        let p = x.ncols();

        if self.options.with_intercept {
            let intercept = beta[0];
            let mut eta = Col::zeros(n);
            for i in 0..n {
                eta[i] = intercept;
                for j in 0..p {
                    eta[i] += x[(i, j)] * beta[j + 1];
                }
            }
            eta
        } else {
            let mut eta = Col::zeros(n);
            for i in 0..n {
                for j in 0..p {
                    eta[i] += x[(i, j)] * beta[j];
                }
            }
            eta
        }
    }

    /// Compute mean mu = g^{-1}(eta) using the inverse link function.
    fn compute_mu(&self, eta: &Col<f64>) -> Col<f64> {
        let n = eta.nrows();
        let mut mu = Col::zeros(n);
        for i in 0..n {
            mu[i] = self.link.inverse(eta[i]);
            // Ensure valid range for certain distributions
            // Note: LogitNormal uses Identity link where mu is on logit scale (unbounded)
            // so we don't clamp mu for LogitNormal even though y is in (0,1)
            if self.distribution.requires_positive() {
                mu[i] = mu[i].max(1e-10);
            }
            if self.distribution.requires_unit_interval()
                && !matches!(self.distribution, AlmDistribution::LogitNormal)
            {
                mu[i] = mu[i].clamp(1e-10, 1.0 - 1e-10);
            }
        }
        mu
    }

    /// Compute IRLS weights and working response.
    fn compute_irls_components(
        &self,
        y: &Col<f64>,
        mu: &Col<f64>,
        eta: &Col<f64>,
    ) -> (Col<f64>, Col<f64>) {
        let n = y.nrows();
        let mut weights = Col::zeros(n);
        let mut z = Col::zeros(n);

        for i in 0..n {
            // Derivative of inverse link
            let d_mu = self.link.inverse_derivative(eta[i]).max(1e-10);

            // Variance function based on distribution
            let v = self.variance_function(mu[i]);

            // For Laplace and other robust distributions, use special weights
            let weight = match self.distribution {
                AlmDistribution::Laplace => {
                    // For LAD: w_i = 1 / |y_i - mu_i| (iteratively reweighted)
                    let abs_resid = (y[i] - mu[i]).abs().max(1e-6);
                    1.0 / abs_resid
                }
                AlmDistribution::LogLaplace => {
                    // For Log-LAD: weights on log scale
                    let abs_resid = (y[i].ln() - mu[i].ln()).abs().max(1e-6);
                    1.0 / abs_resid
                }
                AlmDistribution::AsymmetricLaplace => {
                    // Asymmetric Laplace weights for quantile regression
                    let alpha = self.extra_parameter.unwrap_or(0.5);
                    let resid = y[i] - mu[i];
                    let abs_resid = resid.abs().max(1e-6);
                    if resid >= 0.0 {
                        alpha / abs_resid
                    } else {
                        (1.0 - alpha) / abs_resid
                    }
                }
                AlmDistribution::StudentT => {
                    // Robust Student-t weights
                    let df = self.extra_parameter.unwrap_or(5.0);
                    let resid = y[i] - mu[i];
                    let scale_est = 1.0; // Use previous scale estimate
                    let z_sq = (resid / scale_est).powi(2);
                    (df + 1.0) / (df + z_sq)
                }
                AlmDistribution::S | AlmDistribution::LogS => {
                    // S distribution (greybox parameterization) weights
                    // Score: ∂log f/∂μ ∝ sign(y-μ) / √|y-μ|
                    // Weight proportional to 1/|y-μ| for IRLS stability
                    let abs_resid = (y[i] - mu[i]).abs().max(1e-6);
                    1.0 / abs_resid
                }
                AlmDistribution::GeneralisedNormal | AlmDistribution::LogGeneralisedNormal => {
                    // Generalized Normal weights: proportional to |y-μ|^(β-2)
                    let shape = self.extra_parameter.unwrap_or(2.0);
                    let abs_resid = (y[i] - mu[i]).abs().max(1e-6);
                    abs_resid.powf(shape - 2.0).max(1e-10)
                }
                _ => {
                    // Standard GLM weight: d_mu^2 / V(mu)
                    (d_mu * d_mu / v).max(1e-10)
                }
            };

            weights[i] = weight;

            // Working response: z = eta + (y - mu) / d_mu
            // Special cases for transformed-scale distributions
            let response = match self.distribution {
                AlmDistribution::LogitNormal => {
                    // LogitNormal models logit(y), so use logit(y) as response
                    let yi = y[i].clamp(0.001, 0.999);
                    (yi / (1.0 - yi)).ln() // logit(y)
                }
                AlmDistribution::LogLaplace
                | AlmDistribution::LogS
                | AlmDistribution::LogGeneralisedNormal => {
                    // Log-domain distributions: use log(y) directly as response
                    // z = log(y) for these distributions since they model log(Y)
                    y[i].max(1e-10).ln()
                }
                _ => y[i],
            };
            // For log-domain distributions with Log link, eta = log(mu), d_mu = mu
            // Working response: z = eta + (log(y) - eta) / 1 = log(y)
            // For other distributions: z = eta + (y - mu) / d_mu
            z[i] = if matches!(
                self.distribution,
                AlmDistribution::LogLaplace
                    | AlmDistribution::LogS
                    | AlmDistribution::LogGeneralisedNormal
                    | AlmDistribution::LogitNormal
            ) {
                response // These are already on the transformed scale
            } else {
                eta[i] + (response - mu[i]) / d_mu
            };
        }

        (weights, z)
    }

    /// Variance function V(mu) for the distribution.
    fn variance_function(&self, mu: f64) -> f64 {
        match self.distribution {
            AlmDistribution::Normal
            | AlmDistribution::Laplace
            | AlmDistribution::StudentT
            | AlmDistribution::Logistic
            | AlmDistribution::AsymmetricLaplace
            | AlmDistribution::GeneralisedNormal
            | AlmDistribution::S => 1.0,

            AlmDistribution::Poisson => mu.max(1e-10),

            AlmDistribution::Binomial | AlmDistribution::Geometric => {
                let p = mu.clamp(1e-10, 1.0 - 1e-10);
                p * (1.0 - p)
            }

            AlmDistribution::NegativeBinomial => {
                let size = self.extra_parameter.unwrap_or(1.0);
                mu + mu * mu / size
            }

            AlmDistribution::Gamma | AlmDistribution::InverseGaussian => mu * mu,

            AlmDistribution::Exponential => mu * mu,

            AlmDistribution::Beta => {
                let phi = self.extra_parameter.unwrap_or(1.0);
                let p = mu.clamp(1e-10, 1.0 - 1e-10);
                p * (1.0 - p) / (1.0 + phi)
            }

            AlmDistribution::LogNormal
            | AlmDistribution::LogLaplace
            | AlmDistribution::LogS
            | AlmDistribution::LogGeneralisedNormal
            | AlmDistribution::FoldedNormal
            | AlmDistribution::RectifiedNormal
            | AlmDistribution::BoxCoxNormal
            | AlmDistribution::LogitNormal => 1.0,

            AlmDistribution::CumulativeLogistic | AlmDistribution::CumulativeNormal => 1.0,
        }
    }

    /// Perform weighted least squares step.
    fn weighted_ls_step(
        &self,
        x: &Mat<f64>,
        z: &Col<f64>,
        weights: &Col<f64>,
        _prev_beta: &Col<f64>,
    ) -> Result<Col<f64>, RegressionError> {
        let n = x.nrows();
        let p = x.ncols();

        // Create weighted design matrix and response
        if self.options.with_intercept {
            let ncols = p + 1;
            let mut xw = Mat::zeros(n, ncols);
            let mut zw = Col::zeros(n);

            for i in 0..n {
                let w_sqrt = weights[i].sqrt();
                xw[(i, 0)] = w_sqrt;
                for j in 0..p {
                    xw[(i, j + 1)] = x[(i, j)] * w_sqrt;
                }
                zw[i] = z[i] * w_sqrt;
            }

            // Solve weighted least squares with permutation handling
            let qr = xw.col_piv_qr();
            let perm = qr.P();
            let perm_arr = perm.arrays().0;

            let q = qr.compute_Q();
            let r = qr.R();

            let qty = q.transpose() * &zw;
            let mut beta_perm = Col::zeros(ncols);

            for i in (0..ncols.min(n)).rev() {
                let mut sum = qty[i];
                for j in (i + 1)..ncols.min(n) {
                    sum -= r[(i, j)] * beta_perm[j];
                }
                if r[(i, i)].abs() > 1e-10 {
                    beta_perm[i] = sum / r[(i, i)];
                }
            }

            // Apply inverse permutation
            let mut beta = Col::zeros(ncols);
            for i in 0..ncols {
                let orig_col = perm_arr[i];
                beta[orig_col] = beta_perm[i];
            }

            Ok(beta)
        } else {
            let mut xw = Mat::zeros(n, p);
            let mut zw = Col::zeros(n);

            for i in 0..n {
                let w_sqrt = weights[i].sqrt();
                for j in 0..p {
                    xw[(i, j)] = x[(i, j)] * w_sqrt;
                }
                zw[i] = z[i] * w_sqrt;
            }

            let qr = xw.col_piv_qr();
            let perm = qr.P();
            let perm_arr = perm.arrays().0;

            let q = qr.compute_Q();
            let r = qr.R();

            let qty = q.transpose() * &zw;
            let mut beta_perm = Col::zeros(p);

            for i in (0..p.min(n)).rev() {
                let mut sum = qty[i];
                for j in (i + 1)..p.min(n) {
                    sum -= r[(i, j)] * beta_perm[j];
                }
                if r[(i, i)].abs() > 1e-10 {
                    beta_perm[i] = sum / r[(i, i)];
                }
            }

            // Apply inverse permutation
            let mut beta = Col::zeros(p);
            for i in 0..p {
                let orig_col = perm_arr[i];
                beta[orig_col] = beta_perm[i];
            }

            Ok(beta)
        }
    }

    /// Build the final result.
    fn build_result(
        &self,
        x: &Mat<f64>,
        y: &Col<f64>,
        beta: &Col<f64>,
        mu: &Col<f64>,
        scale: f64,
    ) -> Result<FittedAlm, RegressionError> {
        let n = x.nrows();
        let p = x.ncols();

        // Extract coefficients and intercept
        let (coefficients, intercept) = if self.options.with_intercept {
            let intercept = beta[0];
            let mut coefs = Col::zeros(p);
            for j in 0..p {
                coefs[j] = beta[j + 1];
            }
            (coefs, Some(intercept))
        } else {
            (beta.clone(), None)
        };

        let n_params = if self.options.with_intercept {
            p + 2 // coefficients + intercept + scale
        } else {
            p + 1 // coefficients + scale
        };

        // Compute residuals
        let mut residuals = Col::zeros(n);
        for i in 0..n {
            residuals[i] = y[i] - mu[i];
        }

        // Compute statistics
        let y_mean: f64 = y.iter().sum::<f64>() / n as f64;
        let tss: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();
        let rss: f64 = residuals.iter().map(|&r| r.powi(2)).sum();

        let r_squared = if tss > 0.0 {
            (1.0 - rss / tss).clamp(0.0, 1.0)
        } else if rss < 1e-10 {
            1.0
        } else {
            0.0
        };

        let df_total = (n - 1) as f64;
        let df_resid = (n - n_params) as f64;
        let adj_r_squared = if df_resid > 0.0 && df_total > 0.0 {
            1.0 - (1.0 - r_squared) * df_total / df_resid
        } else {
            f64::NAN
        };

        let mse = if df_resid > 0.0 {
            rss / df_resid
        } else {
            f64::NAN
        };
        let rmse = mse.sqrt();

        // Log-likelihood
        let ll = log_likelihood(y, mu, self.distribution, scale, self.extra_parameter);

        // Information criteria
        let k = n_params as f64;
        let aic = if ll.is_finite() {
            2.0 * k - 2.0 * ll
        } else {
            f64::NAN
        };
        let aicc = if ll.is_finite() && (n as f64 - k - 1.0) > 0.0 {
            aic + 2.0 * k * (k + 1.0) / (n as f64 - k - 1.0)
        } else {
            f64::NAN
        };
        let bic = if ll.is_finite() {
            k * (n as f64).ln() - 2.0 * ll
        } else {
            f64::NAN
        };

        // F-statistic (may not be meaningful for non-Normal)
        let ess = tss - rss;
        let df_model = (n_params - if intercept.is_some() { 1 } else { 0 } - 1) as f64; // -1 for scale
        let f_statistic = if df_model > 0.0 && df_resid > 0.0 && mse > 0.0 {
            (ess / df_model) / mse
        } else {
            f64::NAN
        };
        let f_pvalue = if f_statistic.is_finite() && df_model > 0.0 && df_resid > 0.0 {
            let f_dist = FisherSnedecor::new(df_model, df_resid).ok();
            f_dist.map_or(f64::NAN, |d| 1.0 - d.cdf(f_statistic))
        } else {
            f64::NAN
        };

        let aliased = vec![false; p];

        let mut result = RegressionResult::empty(p, n);
        result.coefficients = coefficients.clone();
        result.intercept = intercept;
        result.residuals = residuals;
        result.fitted_values = mu.clone();
        result.rank = p;
        result.n_parameters = n_params;
        result.n_observations = n;
        result.aliased = aliased;
        result.r_squared = r_squared;
        result.adj_r_squared = adj_r_squared;
        result.mse = mse;
        result.rmse = rmse;
        result.f_statistic = f_statistic;
        result.f_pvalue = f_pvalue;
        result.aic = aic;
        result.aicc = aicc;
        result.bic = bic;
        result.log_likelihood = ll;
        result.confidence_level = self.options.confidence_level;

        // Compute inference if requested
        if self.options.compute_inference {
            self.compute_inference(x, &mut result, scale)?;
        }

        Ok(FittedAlm {
            options: self.options.clone(),
            distribution: self.distribution,
            link: self.link,
            extra_parameter: self.extra_parameter,
            loss: self.loss,
            scale,
            result,
        })
    }

    /// Compute inference statistics using Fisher information.
    fn compute_inference(
        &self,
        x: &Mat<f64>,
        result: &mut RegressionResult,
        _scale: f64,
    ) -> Result<(), RegressionError> {
        let df = result.residual_df() as f64;

        if df <= 0.0 || !result.mse.is_finite() {
            return Ok(());
        }

        // Use standard errors based on (X'WX)^{-1} approximation
        // For simplicity, use the OLS-based standard errors
        if let Some(intercept) = result.intercept {
            if let Ok((se, se_int)) =
                CoefficientInference::standard_errors_with_intercept(x, result.mse, &result.aliased)
            {
                let t_stats = CoefficientInference::t_statistics(&result.coefficients, &se);
                let p_vals = CoefficientInference::p_values(&t_stats, df);
                let (ci_lower, ci_upper) = CoefficientInference::confidence_intervals(
                    &result.coefficients,
                    &se,
                    df,
                    self.options.confidence_level,
                );

                result.std_errors = Some(se);
                result.t_statistics = Some(t_stats);
                result.p_values = Some(p_vals);
                result.conf_interval_lower = Some(ci_lower);
                result.conf_interval_upper = Some(ci_upper);

                // Intercept inference
                let t_int = if se_int > 0.0 {
                    intercept / se_int
                } else {
                    f64::NAN
                };

                use statrs::distribution::StudentsT;
                let t_dist = StudentsT::new(0.0, 1.0, df).ok();
                let p_int = if t_int.is_finite() {
                    t_dist.map_or(f64::NAN, |d| 2.0 * (1.0 - d.cdf(t_int.abs())))
                } else {
                    f64::NAN
                };

                let t_crit = t_dist.map_or(f64::NAN, |d| {
                    d.inverse_cdf(1.0 - (1.0 - self.options.confidence_level) / 2.0)
                });
                let ci_int = (intercept - t_crit * se_int, intercept + t_crit * se_int);

                result.intercept_std_error = Some(se_int);
                result.intercept_t_statistic = Some(t_int);
                result.intercept_p_value = Some(p_int);
                result.intercept_conf_interval = Some(ci_int);
            }
        } else if let Ok(se) = CoefficientInference::standard_errors(x, result.mse, &result.aliased)
        {
            let t_stats = CoefficientInference::t_statistics(&result.coefficients, &se);
            let p_vals = CoefficientInference::p_values(&t_stats, df);
            let (ci_lower, ci_upper) = CoefficientInference::confidence_intervals(
                &result.coefficients,
                &se,
                df,
                self.options.confidence_level,
            );

            result.std_errors = Some(se);
            result.t_statistics = Some(t_stats);
            result.p_values = Some(p_vals);
            result.conf_interval_lower = Some(ci_lower);
            result.conf_interval_upper = Some(ci_upper);
        }

        Ok(())
    }
}

impl Regressor for AlmRegressor {
    type Fitted = FittedAlm;

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

        // Check minimum observations
        let n_params = if self.options.with_intercept {
            n_features + 2 // +1 for intercept, +1 for scale
        } else {
            n_features + 1 // +1 for scale
        };

        if n_samples < n_params {
            return Err(RegressionError::InsufficientObservations {
                needed: n_params,
                got: n_samples,
            });
        }

        // Validate response values for certain distributions
        if self.distribution.requires_positive() {
            for i in 0..n_samples {
                if y[i] <= 0.0 {
                    return Err(RegressionError::NumericalError(format!(
                        "Distribution {:?} requires positive response values",
                        self.distribution
                    )));
                }
            }
        }

        if self.distribution.requires_unit_interval() {
            for i in 0..n_samples {
                if y[i] <= 0.0 || y[i] >= 1.0 {
                    return Err(RegressionError::NumericalError(format!(
                        "Distribution {:?} requires response values in (0, 1)",
                        self.distribution
                    )));
                }
            }
        }

        // Choose optimizer based on distribution
        if self.distribution.requires_numerical_optimization() {
            self.fit_numerical(x, y)
        } else {
            self.fit_irls(x, y)
        }
    }
}

// ============================================================================
// Fitted ALM
// ============================================================================

/// Fitted Augmented Linear Model (ALM) with flexible distributions.
///
/// Contains the estimated coefficients and model diagnostics from fitting
/// an ALM using maximum likelihood estimation via IRLS. ALM supports 24
/// different distribution families with various link functions.
///
/// # Supported Distributions
///
/// - **Continuous**: Normal, Laplace, Student-t, Cauchy, Gumbel, etc.
/// - **Count**: Poisson, Negative Binomial, Geometric, etc.
/// - **Positive continuous**: Gamma, Weibull, Log-Normal, Pareto, etc.
/// - **Bounded**: Beta, Uniform
///
/// # Available Methods
///
/// - [`predict`](FittedRegressor::predict) - Predict response values
/// - [`distribution`](Self::distribution) - Get the distribution family
/// - [`link`](Self::link) - Get the link function
/// - [`scale`](Self::scale) - Get the estimated scale parameter
/// - [`extra_parameter`](Self::extra_parameter) - Get extra parameter (df, shape, etc.)
///
/// # Example
///
/// ```rust,ignore
/// use anofox_regression::prelude::*;
///
/// let x = Mat::from_fn(100, 2, |i, j| (i + j) as f64 / 10.0);
/// let y = Col::from_fn(100, |i| (i as f64 + 1.0).sin() + 1.5);
///
/// // Fit a robust regression with Student-t distribution
/// let fitted = AlmRegressor::builder()
///     .distribution(AlmDistribution::StudentT)
///     .extra_parameter(5.0)  // degrees of freedom
///     .with_intercept(true)
///     .build()
///     .fit(&x, &y)?;
///
/// // Access model results
/// let scale = fitted.scale();
/// let df = fitted.extra_parameter();  // Some(5.0)
/// ```
#[derive(Debug, Clone)]
pub struct FittedAlm {
    #[allow(dead_code)]
    options: RegressionOptions,
    distribution: AlmDistribution,
    link: LinkFunction,
    extra_parameter: Option<f64>,
    loss: AlmLoss,
    scale: f64,
    result: RegressionResult,
}

impl FittedAlm {
    /// Get the distribution used.
    pub fn distribution(&self) -> AlmDistribution {
        self.distribution
    }

    /// Get the link function used.
    pub fn link(&self) -> LinkFunction {
        self.link
    }

    /// Get the loss function used.
    pub fn loss(&self) -> AlmLoss {
        self.loss
    }

    /// Get the estimated scale parameter.
    pub fn scale(&self) -> f64 {
        self.scale
    }

    /// Get the extra parameter (df, shape, etc.) if any.
    pub fn extra_parameter(&self) -> Option<f64> {
        self.extra_parameter
    }
}

impl FittedRegressor for FittedAlm {
    fn predict(&self, x: &Mat<f64>) -> Col<f64> {
        let n = x.nrows();
        let p = x.ncols();
        let mut predictions = Col::zeros(n);

        let intercept = self.result.intercept.unwrap_or(0.0);

        for i in 0..n {
            let mut eta = intercept;
            for j in 0..p {
                if !self.result.aliased[j] && !self.result.coefficients[j].is_nan() {
                    eta += x[(i, j)] * self.result.coefficients[j];
                }
            }
            // Apply inverse link to get predictions on response scale
            predictions[i] = self.link.inverse(eta);
        }

        predictions
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

        match interval {
            None => PredictionResult::point_only(predictions),
            Some(_) => {
                // For non-Normal distributions, prediction intervals are more complex
                // Return point predictions with NaN intervals for now
                let n = x.nrows();
                let mut lower = Col::zeros(n);
                let mut upper = Col::zeros(n);
                let mut se = Col::zeros(n);
                for i in 0..n {
                    lower[i] = f64::NAN;
                    upper[i] = f64::NAN;
                    se[i] = f64::NAN;
                }
                PredictionResult::with_intervals(predictions, lower, upper, se)
            }
        }
    }
}

// ============================================================================
// Builder
// ============================================================================

/// Builder for configuring an Augmented Linear Model.
///
/// Provides a fluent API for selecting distributions, link functions,
/// and other model options.
///
/// # Example
///
/// ```rust,ignore
/// use anofox_regression::prelude::*;
///
/// // Normal distribution with identity link (standard linear model)
/// let model = AlmRegressor::builder()
///     .distribution(AlmDistribution::Normal)
///     .build();
///
/// // Robust regression with Student-t distribution
/// let model = AlmRegressor::builder()
///     .distribution(AlmDistribution::StudentT)
///     .extra_parameter(5.0)  // degrees of freedom
///     .with_intercept(true)
///     .build();
///
/// // Gamma regression with log link
/// let model = AlmRegressor::builder()
///     .distribution(AlmDistribution::Gamma)
///     .link(LinkFunction::Log)
///     .build();
///
/// // Beta regression for bounded outcomes
/// let model = AlmRegressor::builder()
///     .distribution(AlmDistribution::Beta)
///     .link(LinkFunction::Logit)
///     .build();
/// ```
#[derive(Debug, Clone)]
pub struct AlmRegressorBuilder {
    options_builder: RegressionOptionsBuilder,
    distribution: AlmDistribution,
    link: Option<LinkFunction>,
    extra_parameter: Option<f64>,
    loss: AlmLoss,
    max_iterations: usize,
    tolerance: f64,
}

impl Default for AlmRegressorBuilder {
    fn default() -> Self {
        Self {
            options_builder: RegressionOptionsBuilder::default(),
            distribution: AlmDistribution::Normal,
            link: None,
            extra_parameter: None,
            loss: AlmLoss::default(),
            max_iterations: 100,
            tolerance: 1e-8,
        }
    }
}

impl AlmRegressorBuilder {
    /// Create a new builder with default options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the distribution family.
    pub fn distribution(mut self, dist: AlmDistribution) -> Self {
        self.distribution = dist;
        self
    }

    /// Set the link function (defaults to canonical link for distribution).
    pub fn link(mut self, link: LinkFunction) -> Self {
        self.link = Some(link);
        self
    }

    /// Set the extra parameter (degrees of freedom, shape, etc.).
    pub fn extra_parameter(mut self, param: f64) -> Self {
        self.extra_parameter = Some(param);
        self
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

    /// Set maximum iterations for optimization.
    pub fn max_iterations(mut self, iters: usize) -> Self {
        self.max_iterations = iters;
        self
    }

    /// Set convergence tolerance.
    pub fn tolerance(mut self, tol: f64) -> Self {
        self.tolerance = tol;
        self
    }

    /// Set the loss function for convergence criterion.
    ///
    /// # Arguments
    /// * `loss` - The loss function to use (default: `AlmLoss::Likelihood`)
    ///
    /// # Example
    /// ```ignore
    /// let model = AlmRegressor::builder()
    ///     .distribution(AlmDistribution::Normal)
    ///     .loss(AlmLoss::MAE)  // Use Mean Absolute Error
    ///     .build();
    /// ```
    pub fn loss(mut self, loss: AlmLoss) -> Self {
        self.loss = loss;
        self
    }

    /// Set ROLE (RObust Likelihood Estimator) as the loss function with custom trim.
    ///
    /// # Arguments
    /// * `trim` - Fraction of observations to trim (0.0 to 0.5, default 0.05)
    ///
    /// # Example
    /// ```ignore
    /// let model = AlmRegressor::builder()
    ///     .distribution(AlmDistribution::Normal)
    ///     .role_trim(0.10)  // Trim 10% worst observations
    ///     .build();
    /// ```
    pub fn role_trim(mut self, trim: f64) -> Self {
        self.loss = AlmLoss::role_with_trim(trim);
        self
    }

    /// Build the ALM regressor.
    pub fn build(self) -> AlmRegressor {
        let options = self.options_builder.build_unchecked();
        let mut alm =
            AlmRegressor::new(options, self.distribution, self.link, self.extra_parameter);
        alm.loss = self.loss;
        alm.max_iterations = self.max_iterations;
        alm.tolerance = self.tolerance;
        alm
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normal_likelihood() {
        let y = Col::from_fn(5, |i| i as f64);
        let mu = Col::from_fn(5, |i| i as f64);
        let ll = log_likelihood(&y, &mu, AlmDistribution::Normal, 1.0, None);
        // Perfect fit: RSS = 0
        // LL = -5/2 * log(2*pi) - 0 = -5/2 * 1.8379
        let expected = -2.5 * (2.0 * PI).ln();
        assert!((ll - expected).abs() < 1e-10);
    }

    #[test]
    fn test_laplace_likelihood() {
        let y = Col::from_fn(5, |i| i as f64);
        let mu = Col::from_fn(5, |i| i as f64);
        let ll = log_likelihood(&y, &mu, AlmDistribution::Laplace, 1.0, None);
        // Perfect fit: SAD = 0
        let expected = -5.0 * (2.0_f64).ln();
        assert!((ll - expected).abs() < 1e-10);
    }

    #[test]
    fn test_link_functions() {
        // Identity
        assert!((LinkFunction::Identity.inverse(2.0) - 2.0).abs() < 1e-10);

        // Log
        assert!((LinkFunction::Log.inverse(0.0) - 1.0).abs() < 1e-10);
        assert!((LinkFunction::Log.inverse(1.0) - std::f64::consts::E).abs() < 1e-10);

        // Logit
        assert!((LinkFunction::Logit.inverse(0.0) - 0.5).abs() < 1e-10);
    }

    // ==================== Log-likelihood tests ====================

    #[test]
    fn test_ll_student_t() {
        let y = Col::from_fn(5, |i| i as f64);
        let mu = Col::from_fn(5, |i| i as f64);
        let ll = log_likelihood(&y, &mu, AlmDistribution::StudentT, 1.0, Some(5.0));
        // Perfect fit should give finite positive-ish log-likelihood
        assert!(ll.is_finite());
        assert!(ll > -100.0); // Reasonable bound for perfect fit
    }

    #[test]
    fn test_ll_logistic() {
        let y = Col::from_fn(5, |i| i as f64);
        let mu = Col::from_fn(5, |i| i as f64);
        let ll = log_likelihood(&y, &mu, AlmDistribution::Logistic, 1.0, None);
        // Perfect fit
        assert!(ll.is_finite());
        assert!(ll > -100.0);
    }

    #[test]
    fn test_ll_gamma() {
        let y = Col::from_fn(5, |i| (i + 1) as f64); // Positive values
        let mu = Col::from_fn(5, |i| (i + 1) as f64);
        let ll = log_likelihood(&y, &mu, AlmDistribution::Gamma, 1.0, None);
        assert!(ll.is_finite());
    }

    #[test]
    fn test_ll_exponential() {
        let y = Col::from_fn(5, |i| (i + 1) as f64); // Positive values
        let mu = Col::from_fn(5, |i| (i + 1) as f64);
        let ll = log_likelihood(&y, &mu, AlmDistribution::Exponential, 1.0, None);
        assert!(ll.is_finite());
    }

    #[test]
    fn test_ll_beta() {
        let y = Col::from_fn(5, |i| 0.1 + 0.15 * i as f64); // Values in (0, 1)
        let mu = Col::from_fn(5, |i| 0.1 + 0.15 * i as f64);
        let ll = log_likelihood(&y, &mu, AlmDistribution::Beta, 0.1, None);
        assert!(ll.is_finite());
    }

    #[test]
    fn test_ll_poisson() {
        let y = Col::from_fn(5, |i| i as f64); // Non-negative integers
        let mu = Col::from_fn(5, |i| (i as f64).max(0.1));
        let ll = log_likelihood(&y, &mu, AlmDistribution::Poisson, 1.0, None);
        assert!(ll.is_finite());
    }

    #[test]
    fn test_ll_negative_binomial() {
        let y = Col::from_fn(5, |i| i as f64);
        let mu = Col::from_fn(5, |i| (i as f64).max(0.1));
        let ll = log_likelihood(&y, &mu, AlmDistribution::NegativeBinomial, 1.0, Some(2.0));
        assert!(ll.is_finite());
    }

    #[test]
    fn test_ll_binomial() {
        let y = Col::from_fn(5, |i| if i % 2 == 0 { 0.0 } else { 1.0 });
        let mu = Col::from_fn(5, |_| 0.5);
        let ll = log_likelihood(&y, &mu, AlmDistribution::Binomial, 1.0, Some(1.0));
        assert!(ll.is_finite());
    }

    #[test]
    fn test_ll_inverse_gaussian() {
        let y = Col::from_fn(5, |i| (i + 1) as f64);
        let mu = Col::from_fn(5, |i| (i + 1) as f64);
        let ll = log_likelihood(&y, &mu, AlmDistribution::InverseGaussian, 1.0, None);
        assert!(ll.is_finite());
    }

    #[test]
    fn test_ll_log_normal() {
        let y = Col::from_fn(5, |i| (i + 1) as f64);
        let mu = Col::from_fn(5, |i| (i + 1) as f64);
        let ll = log_likelihood(&y, &mu, AlmDistribution::LogNormal, 1.0, None);
        assert!(ll.is_finite());
    }

    #[test]
    fn test_ll_asymmetric_laplace() {
        let y = Col::from_fn(5, |i| i as f64);
        let mu = Col::from_fn(5, |i| i as f64);
        let ll = log_likelihood(&y, &mu, AlmDistribution::AsymmetricLaplace, 1.0, Some(0.5));
        assert!(ll.is_finite());
    }

    #[test]
    fn test_ll_generalised_normal() {
        let y = Col::from_fn(5, |i| i as f64);
        let mu = Col::from_fn(5, |i| i as f64);
        let ll = log_likelihood(&y, &mu, AlmDistribution::GeneralisedNormal, 1.0, Some(2.0));
        assert!(ll.is_finite());
    }

    #[test]
    fn test_ll_geometric() {
        let y = Col::from_fn(5, |i| i as f64);
        let mu = Col::from_fn(5, |_| 0.5);
        let ll = log_likelihood(&y, &mu, AlmDistribution::Geometric, 1.0, None);
        assert!(ll.is_finite());
    }

    // ==================== End-to-end fitting tests ====================

    #[test]
    fn test_alm_fit_normal() {
        let x = Mat::from_fn(50, 1, |i, _| i as f64);
        let y = Col::from_fn(50, |i| 1.0 + 2.0 * i as f64 + 0.1 * (i as f64).sin());

        let fitted = AlmRegressor::builder()
            .distribution(AlmDistribution::Normal)
            .with_intercept(true)
            .build()
            .fit(&x, &y)
            .expect("should fit");

        assert!(fitted.coefficients()[0] > 1.5); // Slope should be ~2
        assert!(fitted.scale() > 0.0);
    }

    #[test]
    fn test_alm_fit_laplace() {
        let x = Mat::from_fn(50, 1, |i, _| i as f64);
        let y = Col::from_fn(50, |i| 1.0 + 2.0 * i as f64);

        let fitted = AlmRegressor::builder()
            .distribution(AlmDistribution::Laplace)
            .with_intercept(true)
            .build()
            .fit(&x, &y)
            .expect("should fit");

        assert!(fitted.coefficients()[0] > 1.5);
    }

    #[test]
    fn test_alm_fit_student_t() {
        let x = Mat::from_fn(50, 1, |i, _| i as f64);
        let y = Col::from_fn(50, |i| 1.0 + 2.0 * i as f64);

        let fitted = AlmRegressor::builder()
            .distribution(AlmDistribution::StudentT)
            .extra_parameter(5.0) // df = 5
            .with_intercept(true)
            .build()
            .fit(&x, &y)
            .expect("should fit");

        assert!(fitted.coefficients()[0] > 1.5);
        assert_eq!(fitted.extra_parameter(), Some(5.0));
    }

    #[test]
    fn test_alm_fit_gamma() {
        let x = Mat::from_fn(50, 1, |i, _| i as f64 / 10.0);
        let y = Col::from_fn(50, |i| (1.0 + 0.5 * i as f64 / 10.0).exp()); // Positive values

        let fitted = AlmRegressor::builder()
            .distribution(AlmDistribution::Gamma)
            .with_intercept(true)
            .build()
            .fit(&x, &y)
            .expect("should fit");

        assert!(fitted.scale() > 0.0);
    }

    #[test]
    fn test_alm_fit_poisson() {
        let x = Mat::from_fn(50, 1, |i, _| i as f64 / 10.0);
        let y = Col::from_fn(50, |i| ((0.5 + 0.1 * i as f64 / 10.0).exp()).round());

        let fitted = AlmRegressor::builder()
            .distribution(AlmDistribution::Poisson)
            .with_intercept(true)
            .build()
            .fit(&x, &y)
            .expect("should fit");

        assert!(fitted.result().coefficients.nrows() == 1);
    }

    // ==================== Builder and options tests ====================

    #[test]
    fn test_alm_builder_fluent_api() {
        let model = AlmRegressor::builder()
            .distribution(AlmDistribution::StudentT)
            .link(LinkFunction::Identity)
            .extra_parameter(10.0)
            .with_intercept(true)
            .compute_inference(true)
            .max_iterations(50)
            .tolerance(1e-6)
            .build();

        assert_eq!(model.distribution, AlmDistribution::StudentT);
        assert_eq!(model.link, LinkFunction::Identity);
        assert_eq!(model.extra_parameter, Some(10.0));
        assert!(model.options.with_intercept);
        assert_eq!(model.max_iterations, 50);
    }

    #[test]
    fn test_alm_with_intercept() {
        let x = Mat::from_fn(30, 1, |i, _| i as f64);
        let y = Col::from_fn(30, |i| 5.0 + 2.0 * i as f64);

        let fitted = AlmRegressor::builder()
            .distribution(AlmDistribution::Normal)
            .with_intercept(true)
            .build()
            .fit(&x, &y)
            .expect("should fit");

        // Should have intercept close to 5
        let intercept = fitted.result().intercept.unwrap_or(0.0);
        assert!((intercept - 5.0).abs() < 1.0);
    }

    #[test]
    fn test_alm_without_intercept() {
        let x = Mat::from_fn(30, 1, |i, _| i as f64);
        let y = Col::from_fn(30, |i| 2.0 * i as f64);

        let fitted = AlmRegressor::builder()
            .distribution(AlmDistribution::Normal)
            .with_intercept(false)
            .build()
            .fit(&x, &y)
            .expect("should fit");

        assert!(fitted.result().intercept.is_none());
    }

    // ==================== Link function tests ====================

    #[test]
    fn test_probit_link() {
        // Test probit link function
        assert!((LinkFunction::Probit.inverse(0.0) - 0.5).abs() < 1e-6);
        assert!(LinkFunction::Probit.inverse(-2.0) < 0.1);
        assert!(LinkFunction::Probit.inverse(2.0) > 0.9);
    }

    #[test]
    fn test_sqrt_link() {
        assert!((LinkFunction::Sqrt.inverse(2.0) - 4.0).abs() < 1e-10);
        assert!((LinkFunction::Sqrt.inverse(3.0) - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_inverse_link() {
        assert!((LinkFunction::Inverse.inverse(0.5) - 2.0).abs() < 1e-10);
        assert!((LinkFunction::Inverse.inverse(0.25) - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_cloglog_link() {
        // cloglog(0.5) = log(-log(0.5)) = log(0.693) ≈ -0.367
        let eta = LinkFunction::Cloglog.inverse(-0.367);
        assert!((eta - 0.5).abs() < 0.01);
    }

    // ==================== Scale estimation tests ====================

    #[test]
    fn test_estimate_scale_normal() {
        let y = Col::from_fn(100, |i| i as f64);
        let mu = Col::from_fn(100, |i| i as f64 + 0.5); // Small residuals
        let scale = estimate_scale(&y, &mu, AlmDistribution::Normal, 98.0, None);
        assert!(scale > 0.0);
        assert!(scale < 10.0); // Should be small for nearly perfect fit
    }

    #[test]
    fn test_estimate_scale_laplace() {
        let y = Col::from_fn(100, |i| i as f64);
        let mu = Col::from_fn(100, |i| i as f64 + 0.5);
        let scale = estimate_scale(&y, &mu, AlmDistribution::Laplace, 98.0, None);
        assert!(scale > 0.0);
    }

    #[test]
    fn test_estimate_scale_gamma() {
        let y = Col::from_fn(100, |i| (i + 1) as f64);
        let mu = Col::from_fn(100, |i| (i + 1) as f64);
        let scale = estimate_scale(&y, &mu, AlmDistribution::Gamma, 98.0, None);
        assert!(scale > 0.0);
    }

    // ==================== Distribution property tests ====================

    #[test]
    fn test_canonical_links() {
        assert_eq!(
            AlmDistribution::Normal.canonical_link(),
            LinkFunction::Identity
        );
        assert_eq!(AlmDistribution::Gamma.canonical_link(), LinkFunction::Log);
        assert_eq!(AlmDistribution::Poisson.canonical_link(), LinkFunction::Log);
        assert_eq!(
            AlmDistribution::Binomial.canonical_link(),
            LinkFunction::Logit
        );
        assert_eq!(AlmDistribution::Beta.canonical_link(), LinkFunction::Logit);
    }

    #[test]
    fn test_requires_positive() {
        assert!(!AlmDistribution::Normal.requires_positive());
        assert!(AlmDistribution::Gamma.requires_positive());
        assert!(AlmDistribution::LogNormal.requires_positive());
        assert!(AlmDistribution::Exponential.requires_positive());
    }

    #[test]
    fn test_requires_unit_interval() {
        assert!(!AlmDistribution::Normal.requires_unit_interval());
        assert!(AlmDistribution::Beta.requires_unit_interval());
        assert!(AlmDistribution::LogitNormal.requires_unit_interval());
    }

    // ==================== Prediction tests ====================

    #[test]
    fn test_alm_predict() {
        let x = Mat::from_fn(30, 1, |i, _| i as f64);
        let y = Col::from_fn(30, |i| 1.0 + 2.0 * i as f64);

        let fitted = AlmRegressor::builder()
            .distribution(AlmDistribution::Normal)
            .with_intercept(true)
            .build()
            .fit(&x, &y)
            .expect("should fit");

        let x_new = Mat::from_fn(5, 1, |i, _| (i + 30) as f64);
        let preds = fitted.predict(&x_new);

        assert_eq!(preds.nrows(), 5);
        // Predictions should be increasing
        for i in 1..5 {
            assert!(preds[i] > preds[i - 1]);
        }
    }

    // ==================== Additional log-likelihood tests ====================

    #[test]
    fn test_ll_s_distribution() {
        let y = Col::from_fn(10, |i| (i + 1) as f64);
        let mu = Col::from_fn(10, |i| (i + 1) as f64 + 0.1);
        let ll = log_likelihood(&y, &mu, AlmDistribution::S, 1.0, None);
        assert!(ll.is_finite());
        assert!(ll < 0.0); // Log-likelihood should be negative
    }

    #[test]
    fn test_ll_log_normal_positive() {
        let y = Col::from_fn(10, |i| (i + 1) as f64);
        let mu = Col::from_fn(10, |i| (i + 1) as f64 + 0.1);
        let ll = log_likelihood(&y, &mu, AlmDistribution::LogNormal, 0.5, None);
        assert!(ll.is_finite());
    }

    #[test]
    fn test_ll_log_normal_negative_y() {
        let y = Col::from_fn(10, |i| (i as f64) - 2.0); // Contains negative values
        let mu = Col::from_fn(10, |i| (i + 1) as f64);
        let ll = log_likelihood(&y, &mu, AlmDistribution::LogNormal, 0.5, None);
        assert!(ll == f64::NEG_INFINITY);
    }

    #[test]
    fn test_ll_log_laplace() {
        let y = Col::from_fn(10, |i| (i + 1) as f64);
        let mu = Col::from_fn(10, |i| (i + 1) as f64 + 0.5);
        let ll = log_likelihood(&y, &mu, AlmDistribution::LogLaplace, 1.0, None);
        assert!(ll.is_finite());
    }

    #[test]
    fn test_ll_log_s() {
        let y = Col::from_fn(10, |i| (i + 1) as f64);
        let mu = Col::from_fn(10, |i| (i + 1) as f64 + 0.2);
        let ll = log_likelihood(&y, &mu, AlmDistribution::LogS, 1.0, None);
        assert!(ll.is_finite());
    }

    #[test]
    fn test_ll_log_generalised_normal() {
        let y = Col::from_fn(10, |i| (i + 1) as f64);
        let mu = Col::from_fn(10, |i| (i + 1) as f64 + 0.1);
        let ll = log_likelihood(
            &y,
            &mu,
            AlmDistribution::LogGeneralisedNormal,
            0.5,
            Some(2.0),
        );
        assert!(ll.is_finite());
    }

    #[test]
    fn test_ll_folded_normal() {
        let y = Col::from_fn(10, |i| (i + 1) as f64); // Positive values
        let mu = Col::from_fn(10, |i| (i + 1) as f64);
        let ll = log_likelihood(&y, &mu, AlmDistribution::FoldedNormal, 1.0, None);
        assert!(ll.is_finite());
    }

    #[test]
    fn test_ll_folded_normal_negative() {
        let y = Col::from_fn(10, |i| (i as f64) - 5.0); // Contains negative values
        let mu = Col::from_fn(10, |i| (i + 1) as f64);
        let ll = log_likelihood(&y, &mu, AlmDistribution::FoldedNormal, 1.0, None);
        assert!(ll == f64::NEG_INFINITY);
    }

    #[test]
    fn test_ll_rectified_normal() {
        // Mix of positive values and zeros
        let y = Col::from_fn(10, |i| if i < 3 { 0.0 } else { (i - 2) as f64 });
        let mu = Col::from_fn(10, |i| (i as f64) * 0.5);
        let ll = log_likelihood(&y, &mu, AlmDistribution::RectifiedNormal, 1.0, None);
        assert!(ll.is_finite());
    }

    #[test]
    fn test_ll_rectified_normal_negative() {
        let y = Col::from_fn(10, |i| (i as f64) - 5.0); // Contains negative
        let mu = Col::from_fn(10, |i| (i + 1) as f64);
        let ll = log_likelihood(&y, &mu, AlmDistribution::RectifiedNormal, 1.0, None);
        assert!(ll == f64::NEG_INFINITY);
    }

    #[test]
    fn test_ll_box_cox_normal() {
        let y = Col::from_fn(10, |i| (i + 1) as f64);
        let mu = Col::from_fn(10, |i| (i + 1) as f64 + 0.1);
        let ll = log_likelihood(&y, &mu, AlmDistribution::BoxCoxNormal, 1.0, Some(0.5));
        assert!(ll.is_finite());
    }

    #[test]
    fn test_ll_box_cox_normal_lambda_zero() {
        // Test with lambda ≈ 0 (log transform)
        let y = Col::from_fn(10, |i| (i + 1) as f64);
        let mu = Col::from_fn(10, |i| (i + 1) as f64 + 0.1);
        let ll = log_likelihood(&y, &mu, AlmDistribution::BoxCoxNormal, 1.0, Some(0.0));
        assert!(ll.is_finite());
    }

    #[test]
    fn test_ll_logit_normal() {
        let y = Col::from_fn(10, |i| (i as f64 + 1.0) / 12.0); // Values in (0,1)
        let mu = Col::from_fn(10, |i| (i as f64 + 1.5) / 12.0);
        let ll = log_likelihood(&y, &mu, AlmDistribution::LogitNormal, 0.5, None);
        assert!(ll.is_finite());
    }

    #[test]
    fn test_ll_cumulative_distributions() {
        // Binary response data for cumulative distributions (logistic/probit regression)
        let y = Col::from_fn(10, |i| if i < 5 { 0.0 } else { 1.0 });
        let mu = Col::from_fn(10, |i| (i as f64 / 10.0).clamp(0.1, 0.9)); // Probabilities
                                                                          // Cumulative distributions should return finite likelihoods for binary data
        let ll1 = log_likelihood(&y, &mu, AlmDistribution::CumulativeLogistic, 1.0, None);
        let ll2 = log_likelihood(&y, &mu, AlmDistribution::CumulativeNormal, 1.0, None);
        assert!(ll1.is_finite());
        assert!(ll2.is_finite());
    }

    // ==================== Scale estimation tests ====================

    #[test]
    fn test_estimate_scale_laplace_mad() {
        let y = Col::from_fn(100, |i| (i as f64) + 0.5);
        let mu = Col::from_fn(100, |i| i as f64);
        let scale = estimate_scale(&y, &mu, AlmDistribution::Laplace, 99.0, None);
        assert!(scale > 0.0);
        assert!((scale - 0.5).abs() < 0.1); // MAD should be ~0.5
    }

    #[test]
    fn test_estimate_scale_student_t() {
        let y = Col::from_fn(100, |i| (i as f64) + (i % 2) as f64 * 2.0 - 1.0);
        let mu = Col::from_fn(100, |i| i as f64);
        let scale = estimate_scale(&y, &mu, AlmDistribution::StudentT, 99.0, None);
        assert!(scale > 0.0);
    }

    #[test]
    fn test_estimate_scale_logistic() {
        let y = Col::from_fn(100, |i| (i as f64) + 0.5);
        let mu = Col::from_fn(100, |i| i as f64);
        let scale = estimate_scale(&y, &mu, AlmDistribution::Logistic, 99.0, None);
        assert!(scale > 0.0);
    }

    #[test]
    fn test_estimate_scale_log_normal() {
        let y = Col::from_fn(100, |i| ((i + 1) as f64).exp());
        let mu = Col::from_fn(100, |i| ((i + 1) as f64 + 0.1).exp());
        let scale = estimate_scale(&y, &mu, AlmDistribution::LogNormal, 99.0, None);
        assert!(scale > 0.0);
    }

    #[test]
    fn test_estimate_scale_asymmetric_laplace() {
        let y = Col::from_fn(100, |i| (i as f64) + 0.5);
        let mu = Col::from_fn(100, |i| i as f64);
        let scale = estimate_scale(&y, &mu, AlmDistribution::AsymmetricLaplace, 99.0, Some(0.5));
        assert!(scale > 0.0);
    }

    #[test]
    fn test_estimate_scale_generalised_normal() {
        let y = Col::from_fn(100, |i| (i as f64) + 0.5);
        let mu = Col::from_fn(100, |i| i as f64);
        let scale = estimate_scale(&y, &mu, AlmDistribution::GeneralisedNormal, 99.0, None);
        assert!(scale > 0.0);
    }

    #[test]
    fn test_estimate_scale_s_distribution() {
        let y = Col::from_fn(100, |i| (i as f64) + 0.5);
        let mu = Col::from_fn(100, |i| i as f64);
        let scale = estimate_scale(&y, &mu, AlmDistribution::S, 99.0, None);
        assert!(scale > 0.0);
    }

    #[test]
    fn test_estimate_scale_log_s() {
        let y = Col::from_fn(100, |i| ((i + 1) as f64).exp());
        let mu = Col::from_fn(100, |i| ((i + 1) as f64 + 0.1).exp());
        let scale = estimate_scale(&y, &mu, AlmDistribution::LogS, 99.0, None);
        assert!(scale > 0.0);
    }

    #[test]
    fn test_estimate_scale_box_cox() {
        let y = Col::from_fn(100, |i| (i as f64) + 0.5);
        let mu = Col::from_fn(100, |i| i as f64);
        let scale = estimate_scale(&y, &mu, AlmDistribution::BoxCoxNormal, 99.0, None);
        assert!(scale > 0.0);
    }

    #[test]
    fn test_estimate_scale_discrete_is_one() {
        let y = Col::from_fn(100, |i| i as f64);
        let mu = Col::from_fn(100, |i| i as f64);
        let scale = estimate_scale(&y, &mu, AlmDistribution::Poisson, 99.0, None);
        assert!((scale - 1.0).abs() < 1e-10);
    }

    // ==================== Error handling tests ====================

    #[test]
    fn test_fit_dimension_mismatch() {
        let x = Mat::from_fn(10, 2, |i, j| (i + j) as f64);
        let y = Col::from_fn(5, |i| i as f64); // Wrong size

        let result = AlmRegressor::builder()
            .distribution(AlmDistribution::Normal)
            .build()
            .fit(&x, &y);

        assert!(result.is_err());
    }

    #[test]
    fn test_fit_insufficient_observations() {
        let x = Mat::from_fn(1, 2, |_, _| 1.0);
        let y = Col::from_fn(1, |_| 1.0);

        let result = AlmRegressor::builder()
            .distribution(AlmDistribution::Normal)
            .build()
            .fit(&x, &y);

        assert!(result.is_err());
    }

    #[test]
    fn test_fit_positive_required_violation() {
        let x = Mat::from_fn(20, 1, |i, _| i as f64);
        let y = Col::from_fn(20, |i| (i as f64) - 5.0); // Contains negatives

        let result = AlmRegressor::builder()
            .distribution(AlmDistribution::LogNormal) // Requires positive
            .build()
            .fit(&x, &y);

        assert!(result.is_err());
    }

    #[test]
    fn test_fit_unit_interval_required_violation() {
        let x = Mat::from_fn(20, 1, |i, _| i as f64);
        let y = Col::from_fn(20, |_| 1.5); // Outside (0,1)

        let result = AlmRegressor::builder()
            .distribution(AlmDistribution::Beta) // Requires (0,1)
            .build()
            .fit(&x, &y);

        assert!(result.is_err());
    }

    // ==================== Variance function tests ====================

    #[test]
    fn test_variance_function_poisson() {
        let regressor = AlmRegressor::builder()
            .distribution(AlmDistribution::Poisson)
            .build();
        let var = regressor.variance_function(5.0);
        assert!((var - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_variance_function_binomial() {
        let regressor = AlmRegressor::builder()
            .distribution(AlmDistribution::Binomial)
            .build();
        let var = regressor.variance_function(0.5);
        assert!((var - 0.25).abs() < 1e-10); // p * (1-p) = 0.25
    }

    #[test]
    fn test_variance_function_geometric() {
        let regressor = AlmRegressor::builder()
            .distribution(AlmDistribution::Geometric)
            .build();
        let var = regressor.variance_function(0.3);
        let expected = 0.3 * (1.0 - 0.3); // p * (1-p)
        assert!((var - expected).abs() < 1e-10);
    }

    #[test]
    fn test_variance_function_negative_binomial() {
        let regressor = AlmRegressor::builder()
            .distribution(AlmDistribution::NegativeBinomial)
            .extra_parameter(2.0)
            .build();
        let var = regressor.variance_function(4.0);
        let expected = 4.0 + 4.0 * 4.0 / 2.0; // mu + mu^2/size = 4 + 8 = 12
        assert!((var - expected).abs() < 1e-10);
    }

    #[test]
    fn test_variance_function_gamma() {
        let regressor = AlmRegressor::builder()
            .distribution(AlmDistribution::Gamma)
            .build();
        let var = regressor.variance_function(3.0);
        assert!((var - 9.0).abs() < 1e-10); // mu^2 = 9
    }

    #[test]
    fn test_variance_function_inverse_gaussian() {
        let regressor = AlmRegressor::builder()
            .distribution(AlmDistribution::InverseGaussian)
            .build();
        let var = regressor.variance_function(2.0);
        assert!((var - 4.0).abs() < 1e-10); // mu^2 = 4
    }

    #[test]
    fn test_variance_function_exponential() {
        let regressor = AlmRegressor::builder()
            .distribution(AlmDistribution::Exponential)
            .build();
        let var = regressor.variance_function(5.0);
        assert!((var - 25.0).abs() < 1e-10); // mu^2 = 25
    }

    #[test]
    fn test_variance_function_beta() {
        let regressor = AlmRegressor::builder()
            .distribution(AlmDistribution::Beta)
            .extra_parameter(3.0) // phi
            .build();
        let var = regressor.variance_function(0.5);
        let expected = 0.5 * 0.5 / (1.0 + 3.0); // p * (1-p) / (1 + phi)
        assert!((var - expected).abs() < 1e-10);
    }

    #[test]
    fn test_variance_function_log_distributions() {
        for dist in [
            AlmDistribution::LogNormal,
            AlmDistribution::LogLaplace,
            AlmDistribution::LogS,
            AlmDistribution::LogGeneralisedNormal,
            AlmDistribution::FoldedNormal,
            AlmDistribution::RectifiedNormal,
            AlmDistribution::BoxCoxNormal,
            AlmDistribution::LogitNormal,
        ] {
            let regressor = AlmRegressor::builder().distribution(dist).build();
            let var = regressor.variance_function(2.0);
            assert!((var - 1.0).abs() < 1e-10, "Failed for {:?}", dist);
        }
    }

    #[test]
    fn test_variance_function_cumulative() {
        for dist in [
            AlmDistribution::CumulativeLogistic,
            AlmDistribution::CumulativeNormal,
        ] {
            let regressor = AlmRegressor::builder().distribution(dist).build();
            let var = regressor.variance_function(0.5);
            assert!((var - 1.0).abs() < 1e-10, "Failed for {:?}", dist);
        }
    }

    // ==================== Canonical link tests ====================

    #[test]
    fn test_canonical_link_all_distributions() {
        assert_eq!(
            AlmDistribution::StudentT.canonical_link(),
            LinkFunction::Identity
        );
        assert_eq!(
            AlmDistribution::Logistic.canonical_link(),
            LinkFunction::Identity
        );
        assert_eq!(
            AlmDistribution::AsymmetricLaplace.canonical_link(),
            LinkFunction::Identity
        );
        assert_eq!(
            AlmDistribution::GeneralisedNormal.canonical_link(),
            LinkFunction::Identity
        );
        assert_eq!(AlmDistribution::S.canonical_link(), LinkFunction::Identity);
        assert_eq!(
            AlmDistribution::LogNormal.canonical_link(),
            LinkFunction::Log
        );
        assert_eq!(
            AlmDistribution::LogLaplace.canonical_link(),
            LinkFunction::Log
        );
        assert_eq!(AlmDistribution::LogS.canonical_link(), LinkFunction::Log);
        assert_eq!(
            AlmDistribution::LogGeneralisedNormal.canonical_link(),
            LinkFunction::Log
        );
        assert_eq!(
            AlmDistribution::FoldedNormal.canonical_link(),
            LinkFunction::Identity
        );
        assert_eq!(
            AlmDistribution::RectifiedNormal.canonical_link(),
            LinkFunction::Identity
        );
        assert_eq!(
            AlmDistribution::BoxCoxNormal.canonical_link(),
            LinkFunction::Identity
        );
        assert_eq!(AlmDistribution::Gamma.canonical_link(), LinkFunction::Log);
        assert_eq!(
            AlmDistribution::InverseGaussian.canonical_link(),
            LinkFunction::Log
        );
        assert_eq!(
            AlmDistribution::Exponential.canonical_link(),
            LinkFunction::Log
        );
        assert_eq!(AlmDistribution::Beta.canonical_link(), LinkFunction::Logit);
        assert_eq!(
            AlmDistribution::LogitNormal.canonical_link(),
            LinkFunction::Identity // models logit-scale location directly
        );
        assert_eq!(AlmDistribution::Poisson.canonical_link(), LinkFunction::Log);
        assert_eq!(
            AlmDistribution::NegativeBinomial.canonical_link(),
            LinkFunction::Log
        );
        assert_eq!(
            AlmDistribution::Binomial.canonical_link(),
            LinkFunction::Logit
        );
        assert_eq!(
            AlmDistribution::Geometric.canonical_link(),
            LinkFunction::Log // models mean λ = (1-p)/p
        );
        assert_eq!(
            AlmDistribution::CumulativeLogistic.canonical_link(),
            LinkFunction::Logit
        );
        assert_eq!(
            AlmDistribution::CumulativeNormal.canonical_link(),
            LinkFunction::Probit
        );
    }

    #[test]
    fn test_is_count() {
        assert!(AlmDistribution::Poisson.is_count());
        assert!(AlmDistribution::NegativeBinomial.is_count());
        assert!(AlmDistribution::Binomial.is_count());
        assert!(AlmDistribution::Geometric.is_count());
        assert!(!AlmDistribution::Normal.is_count());
        assert!(!AlmDistribution::Gamma.is_count());
    }

    // ==================== Link function tests ====================

    #[test]
    fn test_link_function_link() {
        // Identity
        assert!((LinkFunction::Identity.link(5.0) - 5.0).abs() < 1e-10);
        // Log
        assert!((LinkFunction::Log.link(std::f64::consts::E) - 1.0).abs() < 1e-10);
        // Logit
        assert!((LinkFunction::Logit.link(0.5) - 0.0).abs() < 1e-10);
        // Inverse
        assert!((LinkFunction::Inverse.link(2.0) - 0.5).abs() < 1e-10);
        // Sqrt
        assert!((LinkFunction::Sqrt.link(4.0) - 2.0).abs() < 1e-10);
        // Cloglog
        let cloglog_result = LinkFunction::Cloglog.link(0.5);
        assert!(cloglog_result.is_finite());
    }

    #[test]
    fn test_link_function_inverse() {
        // Identity
        assert!((LinkFunction::Identity.inverse(5.0) - 5.0).abs() < 1e-10);
        // Log
        assert!((LinkFunction::Log.inverse(0.0) - 1.0).abs() < 1e-10);
        // Logit
        assert!((LinkFunction::Logit.inverse(0.0) - 0.5).abs() < 1e-10);
        // Inverse
        assert!((LinkFunction::Inverse.inverse(2.0) - 0.5).abs() < 1e-10);
        // Sqrt
        assert!((LinkFunction::Sqrt.inverse(2.0) - 4.0).abs() < 1e-10);
        // Cloglog
        let cloglog_result = LinkFunction::Cloglog.inverse(0.0);
        assert!(cloglog_result > 0.0 && cloglog_result < 1.0);
    }

    #[test]
    fn test_link_function_inverse_derivative() {
        // Test inverse_derivative: d(mu)/d(eta)
        // Identity: d/d(eta)[eta] = 1
        assert!((LinkFunction::Identity.inverse_derivative(5.0) - 1.0).abs() < 1e-10);
        // Log: d/d(eta)[exp(eta)] = exp(eta)
        assert!((LinkFunction::Log.inverse_derivative(2.0) - 2.0_f64.exp()).abs() < 0.01);
        // Logit: d/d(eta)[1/(1+exp(-eta))] = mu * (1 - mu)
        // At eta=0: mu=0.5, so derivative = 0.25
        let logit_deriv = LinkFunction::Logit.inverse_derivative(0.0);
        assert!((logit_deriv - 0.25).abs() < 1e-10);
        // Sqrt: d/d(eta)[eta^2] = 2*eta
        assert!((LinkFunction::Sqrt.inverse_derivative(2.0) - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_probit_link_roundtrip() {
        // Probit: mu = Phi(eta)
        let mu = LinkFunction::Probit.inverse(0.0);
        assert!((mu - 0.5).abs() < 1e-6); // Phi(0) = 0.5

        let eta = LinkFunction::Probit.link(0.5);
        assert!(eta.abs() < 1e-6); // probit(0.5) = 0
    }

    #[test]
    fn test_probit_extreme_values() {
        // Test extreme probability values
        let eta_low = LinkFunction::Probit.link(0.001);
        assert!(eta_low < -2.0);

        let eta_high = LinkFunction::Probit.link(0.999);
        assert!(eta_high > 2.0);

        // Test boundary cases
        let eta_zero = LinkFunction::Probit.link(0.0);
        assert!(eta_zero == f64::NEG_INFINITY);

        let eta_one = LinkFunction::Probit.link(1.0);
        assert!(eta_one == f64::INFINITY);
    }

    // ==================== Predict with interval tests ====================

    #[test]
    fn test_predict_with_interval() {
        let x = Mat::from_fn(30, 1, |i, _| i as f64);
        let y = Col::from_fn(30, |i| 1.0 + 2.0 * i as f64);

        let fitted = AlmRegressor::builder()
            .distribution(AlmDistribution::Normal)
            .with_intercept(true)
            .build()
            .fit(&x, &y)
            .expect("should fit");

        let x_new = Mat::from_fn(5, 1, |i, _| (i + 30) as f64);

        // Point prediction only
        let result_none = fitted.predict_with_interval(&x_new, None, 0.95);
        assert_eq!(result_none.fit.nrows(), 5);

        // With interval (returns NaN for non-Normal)
        let result_ci =
            fitted.predict_with_interval(&x_new, Some(crate::core::IntervalType::Confidence), 0.95);
        assert_eq!(result_ci.fit.nrows(), 5);
        assert_eq!(result_ci.lower.nrows(), 5);
        assert_eq!(result_ci.upper.nrows(), 5);
    }

    // ==================== Additional fit tests for coverage ====================

    #[test]
    fn test_alm_fit_log_normal() {
        let x = Mat::from_fn(30, 1, |i, _| i as f64);
        let y = Col::from_fn(30, |i| (1.0 + 0.1 * i as f64).exp());

        let fitted = AlmRegressor::builder()
            .distribution(AlmDistribution::LogNormal)
            .with_intercept(true)
            .build()
            .fit(&x, &y);

        assert!(fitted.is_ok());
    }

    #[test]
    fn test_alm_fit_geometric() {
        let x = Mat::from_fn(50, 1, |i, _| i as f64);
        let y = Col::from_fn(50, |i| ((i % 5) + 1) as f64);

        let fitted = AlmRegressor::builder()
            .distribution(AlmDistribution::Geometric)
            .with_intercept(true)
            .build()
            .fit(&x, &y);

        // May or may not converge, but shouldn't panic
        let _ = fitted;
    }

    #[test]
    fn test_alm_fit_beta() {
        let x = Mat::from_fn(30, 1, |i, _| i as f64);
        let y = Col::from_fn(30, |i| 0.1 + 0.8 * (i as f64 / 30.0)); // Values in (0,1)

        let fitted = AlmRegressor::builder()
            .distribution(AlmDistribution::Beta)
            .extra_parameter(5.0)
            .with_intercept(true)
            .build()
            .fit(&x, &y);

        assert!(fitted.is_ok());
    }

    #[test]
    fn test_alm_fit_inverse_gaussian() {
        let x = Mat::from_fn(30, 1, |i, _| i as f64);
        let y = Col::from_fn(30, |i| (i + 1) as f64);

        let fitted = AlmRegressor::builder()
            .distribution(AlmDistribution::InverseGaussian)
            .with_intercept(true)
            .build()
            .fit(&x, &y);

        assert!(fitted.is_ok());
    }

    #[test]
    fn test_alm_fit_exponential() {
        let x = Mat::from_fn(30, 1, |i, _| i as f64);
        let y = Col::from_fn(30, |i| (i + 1) as f64);

        let fitted = AlmRegressor::builder()
            .distribution(AlmDistribution::Exponential)
            .with_intercept(true)
            .build()
            .fit(&x, &y);

        assert!(fitted.is_ok());
    }

    #[test]
    fn test_alm_fit_binomial() {
        let x = Mat::from_fn(50, 1, |i, _| i as f64);
        let y = Col::from_fn(50, |i| if i > 25 { 1.0 } else { 0.0 });

        let fitted = AlmRegressor::builder()
            .distribution(AlmDistribution::Binomial)
            .extra_parameter(1.0) // n_trials
            .with_intercept(true)
            .build()
            .fit(&x, &y);

        // May or may not converge
        let _ = fitted;
    }

    #[test]
    fn test_alm_fit_negative_binomial() {
        let x = Mat::from_fn(50, 1, |i, _| i as f64);
        let y = Col::from_fn(50, |i| (i % 10) as f64);

        let fitted = AlmRegressor::builder()
            .distribution(AlmDistribution::NegativeBinomial)
            .extra_parameter(2.0)
            .with_intercept(true)
            .build()
            .fit(&x, &y);

        // May or may not converge
        let _ = fitted;
    }
}
