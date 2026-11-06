//! Regression solvers implementing various estimation methods.

pub mod aft;
pub mod aid;
pub mod alm;
mod bayesian;
mod binomial;
mod bls;
pub mod continuation_ratio;
pub mod cv;
pub mod eb_shrink;
mod elastic_net;
pub(crate) mod fit_stats;
mod gamma;
mod glm_alias;
mod glm_weights;
mod glmm;
mod glmm_satterthwaite;
pub mod gls;
mod huber;
mod huber_joint;
mod isotonic;
mod lars;
pub mod lm_dynamic;
mod logistic;
pub mod lowess;
pub mod mediation;
mod moments;
mod multinomial_logistic;
mod negative_binomial;
mod ols;
pub mod ordinal;
mod passive_aggressive;
pub mod path;
pub mod penalized_glm;
pub mod pentrace;
mod pls;
mod poisson;
mod pspline;
mod quantile;
pub mod quantile_inference;
mod ransac;
mod ridge;
mod rls;
pub mod suff_stats;
mod theil_sen;
mod traits;
mod tweedie;
mod wls;

pub use aid::{
    AidClassifier, AidClassifierBuilder, AnomalyType, DemandClassification, DemandDistribution,
    DemandType, DistributionParameters,
};
pub use alm::{
    AlmDistribution, AlmLoss, AlmRegressor, AlmRegressorBuilder, FittedAlm, LinkFunction,
};
pub use bayesian::{
    ArdRegressor, ArdRegressorBuilder, BayesianRidgeRegressor, BayesianRidgeRegressorBuilder,
    CredibleIntervals, FittedArd, FittedBayesianRidge,
};
pub use binomial::{BinomialRegressor, FittedBinomial};
pub use bls::{BlsRegressor, FittedBls};
pub use continuation_ratio::{
    cr_setup, ContinuationRatioData, ContinuationRatioRegressor, ContinuationRatioRegressorBuilder,
    FittedContinuationRatio,
};
pub use elastic_net::{ElasticNetRegressor, FittedElasticNet};
pub use gamma::{FittedGamma, GammaRegressor, GammaRegressorBuilder};
pub use glmm::{
    FactorSummary, FittedGlmm, GlmmDdf, GlmmInference, GlmmRegressor, GlmmRegressorBuilder,
};
pub use gls::{
    FittedGls, GlsCorrelation, GlsError, GlsMethod, GlsRegressor, GlsRegressorBuilder,
    GlsResidualType, GlsVariance, VarCovIntervals, VariogramBin,
};
pub use huber::{FittedHuber, HuberRegressor, HuberRegressorBuilder, HuberScale};
pub use isotonic::{FittedIsotonic, IsotonicRegressor, IsotonicRegressorBuilder, OutOfBounds};
pub use lars::{FittedLars, LarsMethod, LarsRegressor, LarsRegressorBuilder};
pub use lm_dynamic::{
    CandidateModel, FittedLmDynamic, InformationCriterion, LmDynamicRegressor,
    LmDynamicRegressorBuilder,
};
pub use logistic::{FittedLogistic, LogisticRegressor, LogisticRegressorBuilder, Penalty};
pub use moments::MomentAccumulator;
pub use multinomial_logistic::{
    FittedMultinomialLogistic, MultinomialLogisticRegressor, MultinomialLogisticRegressorBuilder,
};
pub use negative_binomial::{FittedNegativeBinomial, NegativeBinomialRegressor};
pub use ols::{FittedOls, OlsRegressor, OlsRegressorBuilder};
pub use ordinal::{
    xmean_ordinaly, FittedOrdinal, OrdinalInformation, OrdinalLink, OrdinalQuantileMethod,
    OrdinalRegressor, OrdinalRegressorBuilder, XmeanOrdinaly,
};
pub use passive_aggressive::{
    FittedPassiveAggressive, PaLoss, PaState, PassiveAggressiveRegressor,
    PassiveAggressiveRegressorBuilder,
};
pub use pentrace::{
    base_penalty_matrix, effective_df_by_term, penalized_ols, penalty_grid, penalty_matrix,
    penalty_multipliers, pentrace, pentrace_ols, PenalizedOls, PenaltySpec, Pentrace,
    PentraceCriterion, PentraceRow,
};
pub use pls::{FittedPls, PlsRegressor};
pub use poisson::{FittedPoisson, PoissonRegressor};
pub use pspline::{FittedPSpline, PSplineRegressor};
pub use quantile::{FittedQuantile, QuantileRegressor, QuantileRegressorBuilder};
pub use ransac::{FittedRansac, RansacRegressor, RansacRegressorBuilder};
pub use ridge::{
    gram_condition_number, ConditioningLambda, FittedRidge, RidgeLoo, RidgeRegressor,
    RidgeRegressorBuilder, DEFAULT_CONDITION_THRESHOLD, DEFAULT_LAMBDA_SMALL,
};
pub use rls::{FittedRls, RlsRegressor};
pub use suff_stats::SufficientStats;
pub use theil_sen::{
    FittedTheilSen, TheilSenCiMethod, TheilSenRegressor, TheilSenRegressorBuilder,
};
pub use traits::{
    DegenerateData, FittedRegressor, GroupedRegressor, IterativeFit, RegressionError, Regressor,
    SurvivalRegressor,
};
pub use tweedie::{FittedTweedie, TweedieRegressor};
pub use wls::{FittedWls, WlsRegressor, WlsRegressorBuilder};

/// Former name of [`LogisticRegressor`].
#[deprecated(since = "0.9.0", note = "renamed to `LogisticRegressor`")]
pub type LogisticRegression = LogisticRegressor;
/// Former name of [`LogisticRegressorBuilder`].
#[deprecated(since = "0.9.0", note = "renamed to `LogisticRegressorBuilder`")]
pub type LogisticRegressionBuilder = LogisticRegressorBuilder;
/// Former name of [`BayesianRidgeRegressor`].
#[deprecated(since = "0.9.0", note = "renamed to `BayesianRidgeRegressor`")]
pub type BayesianRidge = BayesianRidgeRegressor;
/// Former name of [`ArdRegressor`].
#[deprecated(since = "0.9.0", note = "renamed to `ArdRegressor`")]
pub type ArdRegression = ArdRegressor;
