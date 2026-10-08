//! Public value types of the penalized GLM engine: priors, covariance policy and
//! the fit / inference results.

use crate::core::{BinomialLink, PoissonLink};
use faer::Mat;

/// Family of an explicit prior placed on a single coefficient.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PriorKind {
    /// No prior: the coefficient is estimated by unpenalized maximum likelihood.
    #[default]
    Flat,
    /// Gaussian prior `beta ~ N(loc, scale^2)`. Contributes a quadratic penalty
    /// `(beta - loc)^2 / scale^2`, i.e. a precision block `1 / scale^2` on the diagonal.
    Normal,
    /// Laplace (double-exponential) prior `beta ~ Laplace(loc, scale)`. Contributes an
    /// L1 penalty `|beta - loc| / scale`. Not a precision block: the MAP objective is
    /// non-differentiable at `loc`, so it is fitted by proximal coordinate descent and
    /// curvature-based standard errors exist only for the active set.
    Laplace,
}

/// An explicit prior on one coefficient.
///
/// `scale` is the prior standard deviation for [`PriorKind::Normal`] and the Laplace
/// scale parameter `b` for [`PriorKind::Laplace`]. A flat prior ignores both fields.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PriorSpec {
    pub kind: PriorKind,
    pub loc: f64,
    pub scale: f64,
}

impl Default for PriorSpec {
    fn default() -> Self {
        Self {
            kind: PriorKind::Flat,
            loc: 0.0,
            scale: f64::INFINITY,
        }
    }
}

impl PriorSpec {
    /// A flat (improper uniform) prior — no penalty contribution.
    pub fn flat() -> Self {
        Self::default()
    }

    /// A Gaussian prior with the given mean and standard deviation.
    pub fn normal(loc: f64, scale: f64) -> Self {
        Self {
            kind: PriorKind::Normal,
            loc,
            scale,
        }
    }

    /// A Laplace prior with the given location and scale.
    pub fn laplace(loc: f64, scale: f64) -> Self {
        Self {
            kind: PriorKind::Laplace,
            loc,
            scale,
        }
    }

    /// Gaussian precision `1 / scale^2`; zero for flat and for Laplace priors
    /// (the latter contributes no quadratic term).
    pub fn precision(&self) -> f64 {
        match self.kind {
            PriorKind::Normal if self.scale.is_finite() && self.scale > 0.0 => {
                1.0 / (self.scale * self.scale)
            }
            _ => 0.0,
        }
    }

    /// Returns true when this prior contributes nothing to the objective.
    pub fn is_flat(&self) -> bool {
        match self.kind {
            PriorKind::Flat => true,
            PriorKind::Normal | PriorKind::Laplace => !self.scale.is_finite() || self.scale <= 0.0,
        }
    }
}

/// How the coefficient covariance matrix is computed for a penalized/MAP fit.
///
/// For an unpenalized fit all three coincide.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VcovType {
    /// `(X'WX + P)^-1` — the curvature of the log posterior at the mode (Laplace
    /// approximation). This is the correct observed information for a MAP estimate
    /// and is the default.
    #[default]
    Laplace,
    /// `(X'WX + P)^-1 X'WX (X'WX + P)^-1` — the sandwich form, which targets the
    /// frequentist sampling variance of the penalized estimator rather than the
    /// posterior curvature.
    Sandwich,
    /// `(X'WX)^-1` — ignores the penalty entirely. This reproduces the (incorrect)
    /// standard errors that penalized GLM fits reported before the Laplace work;
    /// retained so the old numbers stay reachable.
    Naive,
}

/// Prior / covariance options shared by every GLM family.
///
/// Kept as a separate struct rather than duplicated across the six family option
/// structs. Kept flat so a C ABI front end can mirror the per-family structs.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GlmPriorOptions {
    /// Per-coefficient priors, positionally aligned with the feature columns.
    /// When `fit_intercept` is set, an extra leading entry applies to the intercept.
    /// An empty vector means "no explicit priors" and is the default.
    pub priors: Vec<PriorSpec>,
    /// How to compute the coefficient covariance matrix.
    pub vcov: VcovType,
}

impl GlmPriorOptions {
    /// True when no explicit prior was supplied (or every supplied prior is flat).
    pub fn is_empty(&self) -> bool {
        self.priors.iter().all(PriorSpec::is_flat)
    }

    /// True when any prior is a Laplace/L1 prior, which requires the proximal path.
    pub fn has_l1(&self) -> bool {
        self.priors
            .iter()
            .any(|p| p.kind == PriorKind::Laplace && !p.is_flat())
    }
}

/// Options for Poisson regression (GLM)
#[derive(Debug, Clone)]
pub struct PoissonOptions {
    /// Whether to fit an intercept term
    pub fit_intercept: bool,
    /// Link function (Log, Identity, or Sqrt)
    pub link: PoissonLink,
    /// Maximum iterations for IRLS
    pub max_iterations: u32,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Whether to compute inference statistics
    pub compute_inference: bool,
    /// Confidence level for confidence intervals
    pub confidence_level: f64,
    /// L2 regularization parameter for penalized IRLS (0.0 = no regularization)
    pub lambda: f64,
    /// Explicit per-coefficient priors and covariance type (see [`GlmPriorOptions`]).
    pub prior_opts: GlmPriorOptions,
    /// 1-based index into `x` of a column used as an offset (added to the linear
    /// predictor with coefficient fixed at 1 and dropped from the design).
    /// `None` = no offset. The value is used as-is; take logs upstream if needed.
    pub offset_column: Option<usize>,
    /// Scale the covariance by the Pearson dispersion `sum(pearson^2) / df_resid`
    /// (R's `quasipoisson`). Default `false`: dispersion fixed at 1, matching R's
    /// `glm(family = poisson)` / `summary.glm`.
    pub estimate_dispersion: bool,
}

impl Default for PoissonOptions {
    fn default() -> Self {
        Self {
            fit_intercept: true,
            link: PoissonLink::Log,
            max_iterations: 100,
            tolerance: 1e-8,
            compute_inference: false,
            confidence_level: 0.95,
            lambda: 0.0,
            prior_opts: GlmPriorOptions::default(),
            offset_column: None,
            estimate_dispersion: false,
        }
    }
}

/// Options for Binomial regression (GLM)
#[derive(Debug, Clone)]
pub struct BinomialOptions {
    /// Whether to fit an intercept term
    pub fit_intercept: bool,
    /// Link function (Logit, Probit, etc.)
    pub link: BinomialLink,
    /// Maximum iterations for IRLS
    pub max_iterations: u32,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Whether to compute inference statistics
    pub compute_inference: bool,
    /// Confidence level for confidence intervals
    pub confidence_level: f64,
    /// L2 regularization parameter for penalized IRLS (0.0 = no regularization)
    pub lambda: f64,
    /// Explicit per-coefficient priors and covariance type (see [`GlmPriorOptions`]).
    pub prior_opts: GlmPriorOptions,
    /// 1-based index into `x` of a column used as an offset (added to the linear
    /// predictor with coefficient fixed at 1 and dropped from the design).
    /// `None` = no offset. The value is used as-is; take logs upstream if needed.
    pub offset_column: Option<usize>,
}

impl Default for BinomialOptions {
    fn default() -> Self {
        Self {
            fit_intercept: true,
            link: BinomialLink::Logit,
            max_iterations: 100,
            tolerance: 1e-8,
            compute_inference: false,
            confidence_level: 0.95,
            lambda: 0.0,
            prior_opts: GlmPriorOptions::default(),
            offset_column: None,
        }
    }
}

/// Options for Negative Binomial regression (GLM)
#[derive(Debug, Clone)]
pub struct NegBinomialOptions {
    /// Whether to fit an intercept term
    pub fit_intercept: bool,
    /// Dispersion parameter (alpha). If None, it will be estimated.
    pub alpha: Option<f64>,
    /// Maximum iterations for IRLS
    pub max_iterations: u32,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Whether to compute inference statistics
    pub compute_inference: bool,
    /// Confidence level for confidence intervals
    pub confidence_level: f64,
    /// L2 regularization parameter for penalized IRLS (0.0 = no regularization)
    pub lambda: f64,
    /// Explicit per-coefficient priors and covariance type (see [`GlmPriorOptions`]).
    pub prior_opts: GlmPriorOptions,
    /// 1-based index into `x` of a column used as an offset (added to the linear
    /// predictor with coefficient fixed at 1 and dropped from the design).
    /// `None` = no offset. The value is used as-is; take logs upstream if needed.
    pub offset_column: Option<usize>,
}

impl Default for NegBinomialOptions {
    fn default() -> Self {
        Self {
            fit_intercept: true,
            alpha: None,
            max_iterations: 100,
            tolerance: 1e-8,
            compute_inference: false,
            confidence_level: 0.95,
            lambda: 0.0,
            prior_opts: GlmPriorOptions::default(),
            offset_column: None,
        }
    }
}

/// Options for Tweedie regression (GLM)
#[derive(Debug, Clone)]
pub struct TweedieOptions {
    /// Whether to fit an intercept term
    pub fit_intercept: bool,
    /// Tweedie power parameter (typically 1 < p < 2 for zero-inflated continuous)
    /// p=1 is Poisson, p=2 is Gamma, 1<p<2 is compound Poisson-Gamma
    pub power: f64,
    /// Maximum iterations for IRLS
    pub max_iterations: u32,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Whether to compute inference statistics
    pub compute_inference: bool,
    /// Confidence level for confidence intervals
    pub confidence_level: f64,
    /// L2 regularization parameter for penalized IRLS (0.0 = no regularization)
    pub lambda: f64,
    /// Explicit per-coefficient priors and covariance type (see [`GlmPriorOptions`]).
    pub prior_opts: GlmPriorOptions,
    /// 1-based index into `x` of a column used as an offset (added to the linear
    /// predictor with coefficient fixed at 1 and dropped from the design).
    /// `None` = no offset. The value is used as-is; take logs upstream if needed.
    pub offset_column: Option<usize>,
}

impl Default for TweedieOptions {
    fn default() -> Self {
        Self {
            fit_intercept: true,
            power: 1.5,
            max_iterations: 100,
            tolerance: 1e-8,
            compute_inference: false,
            confidence_level: 0.95,
            lambda: 0.0,
            prior_opts: GlmPriorOptions::default(),
            offset_column: None,
        }
    }
}

/// Options for Gamma regression (GLM with var_power = 2.0 fixed).
#[derive(Debug, Clone)]
pub struct GammaOptions {
    /// Whether to fit an intercept term.
    pub fit_intercept: bool,
    /// Maximum iterations for IRLS.
    pub max_iterations: u32,
    /// Convergence tolerance.
    pub tolerance: f64,
    /// Whether to compute inference statistics.
    pub compute_inference: bool,
    /// Confidence level for confidence intervals.
    pub confidence_level: f64,
    /// L2 regularization parameter for penalized IRLS (0.0 = no regularization).
    pub lambda: f64,
    /// Explicit per-coefficient priors and covariance type (see [`GlmPriorOptions`]).
    pub prior_opts: GlmPriorOptions,
    /// 1-based index into `x` of a column used as an offset (added to the linear
    /// predictor with coefficient fixed at 1 and dropped from the design).
    /// `None` = no offset. The value is used as-is; take logs upstream if needed.
    pub offset_column: Option<usize>,
}

impl Default for GammaOptions {
    fn default() -> Self {
        Self {
            fit_intercept: true,
            max_iterations: 100,
            tolerance: 1e-8,
            compute_inference: false,
            confidence_level: 0.95,
            lambda: 0.0,
            prior_opts: GlmPriorOptions::default(),
            offset_column: None,
        }
    }
}

/// Options for binary Logistic regression (a thin wrapper around the
/// binomial GLM with logit link; classifier-oriented API).
#[derive(Debug, Clone)]
pub struct LogisticOptions {
    /// Whether to fit an intercept term.
    pub fit_intercept: bool,
    /// L2 (ridge) penalty strength. 0.0 = unpenalised.
    pub lambda: f64,
    /// Classification threshold on the predicted probability.
    pub threshold: f64,
    /// Maximum IRLS iterations.
    pub max_iterations: u32,
    /// Convergence tolerance.
    pub tolerance: f64,
    /// Whether to compute inference statistics on the coefficient vector.
    pub compute_inference: bool,
    /// Confidence level for inference intervals.
    pub confidence_level: f64,
    /// Explicit per-coefficient priors and covariance type (see [`GlmPriorOptions`]).
    pub prior_opts: GlmPriorOptions,
    /// 1-based index into `x` of a column used as an offset (added to the linear
    /// predictor with coefficient fixed at 1 and dropped from the design).
    /// `None` = no offset. The value is used as-is; take logs upstream if needed.
    pub offset_column: Option<usize>,
}

impl Default for LogisticOptions {
    fn default() -> Self {
        Self {
            fit_intercept: true,
            lambda: 0.0,
            threshold: 0.5,
            max_iterations: 100,
            tolerance: 1e-8,
            compute_inference: false,
            confidence_level: 0.95,
            prior_opts: GlmPriorOptions::default(),
            offset_column: None,
        }
    }
}

/// Result from GLM fitting - uses deviance-based metrics instead of R²
#[derive(Debug, Clone)]
pub struct GlmFitResult {
    /// Regression coefficients (excluding intercept)
    pub coefficients: Vec<f64>,
    /// Intercept term (if fitted with intercept)
    pub intercept: Option<f64>,
    /// Null deviance (deviance of intercept-only model)
    pub null_deviance: f64,
    /// Residual deviance (deviance of fitted model)
    pub residual_deviance: f64,
    /// Pseudo R-squared (1 - residual_deviance/null_deviance)
    pub pseudo_r_squared: f64,
    /// AIC (Akaike Information Criterion)
    pub aic: f64,
    /// Number of observations used
    pub n_observations: usize,
    /// Number of features (excluding intercept)
    pub n_features: usize,
    /// Number of iterations to convergence
    pub iterations: u32,
    /// Whether the algorithm converged
    pub converged: bool,
    /// Dispersion parameter (for NegBinomial, Tweedie)
    pub dispersion: Option<f64>,
}

/// GLM inference results
#[derive(Debug, Clone)]
pub struct GlmInferenceResult {
    /// Standard errors of coefficients
    pub std_errors: Vec<f64>,
    /// z-statistics for coefficients (Wald test)
    pub z_values: Vec<f64>,
    /// p-values for coefficients
    pub p_values: Vec<f64>,
    /// Lower bound of confidence intervals
    pub ci_lower: Vec<f64>,
    /// Upper bound of confidence intervals
    pub ci_upper: Vec<f64>,
    /// Confidence level used (e.g., 0.95)
    pub confidence_level: f64,
    /// The full covariance of the fitted parameters at the mode, `None` when
    /// inference was not requested.
    ///
    /// **In fitted order, not the expanded feature order the vectors above use.**
    /// The vectors strip the intercept and put `NaN` where a column was dropped for
    /// rank deficiency; a matrix cannot follow that convention, because removing the
    /// intercept row would discard the intercept/slope covariance that is most of the
    /// reason to want the matrix. [`Self::matrix_parameters`] says what each row is.
    pub vcov: Option<Mat<f64>>,
    /// The penalised observed information at the mode — the inverse of [`Self::vcov`]
    /// — in the same order, `None` when inference was not requested.
    ///
    /// Reported alongside the covariance because they answer different questions:
    /// `vcov` is what a report reads, the information is what a sampler factorises to
    /// draw from the Laplace approximation.
    pub information: Option<Mat<f64>>,
    /// What each row and column of [`Self::vcov`] and [`Self::information`] refers to:
    /// `None` for the intercept, `Some(j)` for feature `j` of the *original* design.
    ///
    /// Dropped columns are absent rather than present-and-`NaN`, so a caller cannot
    /// index a row that was never fitted.
    pub matrix_parameters: Vec<Option<usize>>,
}
