//! Regression options and configuration.

use super::na_action::NaAction;
use thiserror::Error;

/// Solver type for linear algebra operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SolverType {
    /// QR decomposition with column pivoting (default, handles rank deficiency).
    #[default]
    Qr,
    /// SVD decomposition (most robust but slower).
    Svd,
    /// Cholesky decomposition (fastest but requires positive definite X'X).
    Cholesky,
}

/// Lambda scaling convention for regularized regression.
///
/// Different software packages use different conventions for the regularization
/// parameter lambda. This enum allows matching the behavior of various implementations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LambdaScaling {
    /// Use lambda as-is (default).
    ///
    /// Minimizes: ||y - Xβ||² + λ||β||² (for Ridge)
    #[default]
    Raw,

    /// Scale lambda to match R's glmnet package.
    ///
    /// glmnet minimizes: (1/2n) * ||y - Xβ||² + λ * [(1-α)/2 ||β||² + α||β||₁]
    ///
    /// To match, we multiply lambda by n:
    /// ||y - Xβ||² + n*λ * ||β||² (for Ridge)
    Glmnet,
}

/// Configuration options for regression models.
#[derive(Debug, Clone)]
pub struct RegressionOptions {
    /// Whether to include an intercept term (default: true).
    pub with_intercept: bool,
    /// Whether to compute standard errors and inference statistics (default: true).
    pub compute_inference: bool,
    /// Confidence level for confidence intervals (default: 0.95).
    pub confidence_level: f64,
    /// L2 regularization parameter (Ridge/Elastic Net).
    pub lambda: f64,
    /// L1/L2 mixing parameter for Elastic Net (0=Ridge, 1=Lasso).
    pub alpha: f64,
    /// Lambda scaling convention (default: Raw).
    pub lambda_scaling: LambdaScaling,
    /// Forgetting factor for RLS (0 < forgetting_factor <= 1).
    pub forgetting_factor: f64,
    /// Solver type for linear algebra operations.
    pub solver: SolverType,
    /// Maximum iterations for iterative solvers (Elastic Net).
    pub max_iterations: usize,
    /// Convergence tolerance for iterative solvers.
    pub tolerance: f64,
    /// Rank tolerance for QR decomposition.
    pub rank_tolerance: f64,
    /// How to handle NA (missing) values (default: Omit).
    ///
    /// - `Omit`: Remove rows with NA, output is shorter
    /// - `Exclude`: Remove rows with NA, pad output with NA at original positions
    /// - `Fail`: Return error if any NA present
    /// - `Pass`: Pass through NA values (solver must handle them)
    pub na_action: NaAction,
}

impl Default for RegressionOptions {
    fn default() -> Self {
        Self {
            with_intercept: true,
            compute_inference: true,
            confidence_level: 0.95,
            lambda: 0.0,
            alpha: 0.0,
            lambda_scaling: LambdaScaling::Raw,
            forgetting_factor: 1.0,
            solver: SolverType::Qr,
            max_iterations: 1000,
            tolerance: 1e-6,
            rank_tolerance: 1e-10,
            na_action: NaAction::Omit,
        }
    }
}

/// Errors that can occur when validating regression options.
#[derive(Debug, Error)]
pub enum OptionsError {
    #[error("lambda must be non-negative, got {0}")]
    InvalidLambda(f64),
    #[error("alpha must be in [0, 1], got {0}")]
    InvalidAlpha(f64),
    #[error("confidence_level must be in (0, 1), got {0}")]
    InvalidConfidenceLevel(f64),
    #[error("forgetting_factor must be in (0, 1], got {0}")]
    InvalidForgettingFactor(f64),
    #[error("tolerance must be positive, got {0}")]
    InvalidTolerance(f64),
    #[error("max_iterations must be at least 1, got {0}")]
    InvalidMaxIterations(usize),
    #[error("alpha > 0 requires lambda > 0 for elastic net")]
    ElasticNetRequiresLambda,
}

impl RegressionOptions {
    /// Create a new builder for regression options.
    pub fn builder() -> RegressionOptionsBuilder {
        RegressionOptionsBuilder::default()
    }

    /// Create default options for OLS regression.
    pub fn ols() -> Self {
        Self::default()
    }

    /// Create options for Ridge regression with given lambda.
    pub fn ridge(lambda: f64) -> Self {
        Self {
            lambda,
            ..Default::default()
        }
    }

    /// Create options for Lasso regression with given lambda.
    pub fn lasso(lambda: f64) -> Self {
        Self {
            lambda,
            alpha: 1.0,
            ..Default::default()
        }
    }

    /// Create options for Elastic Net with given lambda and alpha.
    pub fn elastic_net(lambda: f64, alpha: f64) -> Self {
        Self {
            lambda,
            alpha,
            ..Default::default()
        }
    }

    /// Create options for Recursive Least Squares.
    pub fn rls(forgetting_factor: f64) -> Self {
        Self {
            forgetting_factor,
            ..Default::default()
        }
    }

    /// Validate the options and return an error if invalid.
    pub fn validate(&self) -> Result<(), OptionsError> {
        if self.lambda < 0.0 {
            return Err(OptionsError::InvalidLambda(self.lambda));
        }
        if !(0.0..=1.0).contains(&self.alpha) {
            return Err(OptionsError::InvalidAlpha(self.alpha));
        }
        if self.confidence_level <= 0.0 || self.confidence_level >= 1.0 {
            return Err(OptionsError::InvalidConfidenceLevel(self.confidence_level));
        }
        if self.forgetting_factor <= 0.0 || self.forgetting_factor > 1.0 {
            return Err(OptionsError::InvalidForgettingFactor(
                self.forgetting_factor,
            ));
        }
        if self.tolerance <= 0.0 {
            return Err(OptionsError::InvalidTolerance(self.tolerance));
        }
        if self.max_iterations < 1 {
            return Err(OptionsError::InvalidMaxIterations(self.max_iterations));
        }
        if self.alpha > 0.0 && self.lambda == 0.0 {
            return Err(OptionsError::ElasticNetRequiresLambda);
        }
        Ok(())
    }
}

/// Builder for `RegressionOptions`.
#[derive(Debug, Clone, Default)]
pub struct RegressionOptionsBuilder {
    options: RegressionOptions,
}

impl RegressionOptionsBuilder {
    /// Create a new builder with default options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set whether to include an intercept term.
    pub fn with_intercept(mut self, include: bool) -> Self {
        self.options.with_intercept = include;
        self
    }

    /// Set whether to compute inference statistics.
    pub fn compute_inference(mut self, compute: bool) -> Self {
        self.options.compute_inference = compute;
        self
    }

    /// Set the confidence level for confidence intervals.
    pub fn confidence_level(mut self, level: f64) -> Self {
        self.options.confidence_level = level;
        self
    }

    /// Set the L2 regularization parameter (lambda).
    pub fn lambda(mut self, lambda: f64) -> Self {
        self.options.lambda = lambda;
        self
    }

    /// Set the L1/L2 mixing parameter (alpha).
    pub fn alpha(mut self, alpha: f64) -> Self {
        self.options.alpha = alpha;
        self
    }

    /// Set the lambda scaling convention.
    ///
    /// Use `LambdaScaling::Glmnet` to match R's glmnet package behavior.
    pub fn lambda_scaling(mut self, scaling: LambdaScaling) -> Self {
        self.options.lambda_scaling = scaling;
        self
    }

    /// Set the forgetting factor for RLS.
    pub fn forgetting_factor(mut self, factor: f64) -> Self {
        self.options.forgetting_factor = factor;
        self
    }

    /// Set the solver type.
    pub fn solver(mut self, solver: SolverType) -> Self {
        self.options.solver = solver;
        self
    }

    /// Set the maximum iterations for iterative solvers.
    pub fn max_iterations(mut self, max_iter: usize) -> Self {
        self.options.max_iterations = max_iter;
        self
    }

    /// Set the convergence tolerance.
    pub fn tolerance(mut self, tol: f64) -> Self {
        self.options.tolerance = tol;
        self
    }

    /// Set the rank tolerance for QR decomposition.
    pub fn rank_tolerance(mut self, tol: f64) -> Self {
        self.options.rank_tolerance = tol;
        self
    }

    /// Set the NA (missing value) handling action.
    ///
    /// - `NaAction::Omit`: Remove rows with NA, output is shorter (default)
    /// - `NaAction::Exclude`: Remove rows with NA, pad output with NA
    /// - `NaAction::Fail`: Return error if any NA present
    /// - `NaAction::Pass`: Pass through NA values
    pub fn na_action(mut self, action: NaAction) -> Self {
        self.options.na_action = action;
        self
    }

    /// Build and validate the options.
    pub fn build(self) -> Result<RegressionOptions, OptionsError> {
        self.options.validate()?;
        Ok(self.options)
    }

    /// Build the options without validation.
    pub fn build_unchecked(self) -> RegressionOptions {
        self.options
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_options() {
        let opts = RegressionOptions::default();
        assert!(opts.with_intercept);
        assert!(opts.compute_inference);
        assert!((opts.confidence_level - 0.95).abs() < 1e-10);
        assert!((opts.lambda - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_builder() {
        let opts = RegressionOptions::builder()
            .with_intercept(false)
            .lambda(0.5)
            .build()
            .unwrap();

        assert!(!opts.with_intercept);
        assert!((opts.lambda - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_validation_invalid_lambda() {
        let result = RegressionOptions::builder().lambda(-1.0).build();
        assert!(matches!(result, Err(OptionsError::InvalidLambda(_))));
    }

    #[test]
    fn test_validation_invalid_alpha() {
        let result = RegressionOptions::builder().alpha(1.5).build();
        assert!(matches!(result, Err(OptionsError::InvalidAlpha(_))));
    }

    #[test]
    fn test_validation_elastic_net_requires_lambda() {
        let result = RegressionOptions::builder().alpha(0.5).lambda(0.0).build();
        assert!(matches!(
            result,
            Err(OptionsError::ElasticNetRequiresLambda)
        ));
    }

    #[test]
    fn test_factory_methods() {
        let ols = RegressionOptions::ols();
        assert!((ols.lambda - 0.0).abs() < 1e-10);

        let ridge = RegressionOptions::ridge(0.5);
        assert!((ridge.lambda - 0.5).abs() < 1e-10);

        let lasso = RegressionOptions::lasso(0.5);
        assert!((lasso.alpha - 1.0).abs() < 1e-10);

        let elastic = RegressionOptions::elastic_net(0.5, 0.5);
        assert!((elastic.lambda - 0.5).abs() < 1e-10);
        assert!((elastic.alpha - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_rls_factory() {
        let rls = RegressionOptions::rls(0.99);
        assert!((rls.forgetting_factor - 0.99).abs() < 1e-10);
    }

    #[test]
    fn test_builder_new() {
        let builder = RegressionOptionsBuilder::new();
        let opts = builder.build_unchecked();
        assert!(opts.with_intercept); // default value
    }

    #[test]
    fn test_builder_solver() {
        let opts = RegressionOptions::builder()
            .solver(SolverType::Svd)
            .build_unchecked();
        assert_eq!(opts.solver, SolverType::Svd);
    }

    #[test]
    fn test_builder_rank_tolerance() {
        let opts = RegressionOptions::builder()
            .rank_tolerance(1e-8)
            .build_unchecked();
        assert!((opts.rank_tolerance - 1e-8).abs() < 1e-14);
    }

    #[test]
    fn test_validation_invalid_confidence_level_zero() {
        let result = RegressionOptions::builder().confidence_level(0.0).build();
        assert!(matches!(
            result,
            Err(OptionsError::InvalidConfidenceLevel(_))
        ));
    }

    #[test]
    fn test_validation_invalid_confidence_level_one() {
        let result = RegressionOptions::builder().confidence_level(1.0).build();
        assert!(matches!(
            result,
            Err(OptionsError::InvalidConfidenceLevel(_))
        ));
    }

    #[test]
    fn test_validation_invalid_forgetting_factor_zero() {
        let result = RegressionOptions::builder().forgetting_factor(0.0).build();
        assert!(matches!(
            result,
            Err(OptionsError::InvalidForgettingFactor(_))
        ));
    }

    #[test]
    fn test_validation_invalid_forgetting_factor_over_one() {
        let result = RegressionOptions::builder().forgetting_factor(1.5).build();
        assert!(matches!(
            result,
            Err(OptionsError::InvalidForgettingFactor(_))
        ));
    }

    #[test]
    fn test_validation_invalid_tolerance() {
        let result = RegressionOptions::builder().tolerance(0.0).build();
        assert!(matches!(result, Err(OptionsError::InvalidTolerance(_))));
    }

    #[test]
    fn test_validation_invalid_max_iterations() {
        let result = RegressionOptions::builder().max_iterations(0).build();
        assert!(matches!(result, Err(OptionsError::InvalidMaxIterations(_))));
    }

    #[test]
    fn test_solver_type_default() {
        let solver = SolverType::default();
        assert_eq!(solver, SolverType::Qr);
    }

    #[test]
    fn test_lambda_scaling_default() {
        let scaling = LambdaScaling::default();
        assert_eq!(scaling, LambdaScaling::Raw);
    }
}
