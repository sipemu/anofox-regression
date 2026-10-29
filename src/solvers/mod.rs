//! Regression solvers implementing various estimation methods.

pub mod aft;
pub mod aid;
pub mod alm;
mod bayesian;
mod binomial;
mod bls;
pub mod cv;
pub mod eb_shrink;
mod elastic_net;
pub(crate) mod fit_stats;
mod gamma;
mod glm_alias;
mod glmm;
mod glmm_satterthwaite;
mod huber;
mod isotonic;
mod lars;
pub mod lm_dynamic;
mod logistic;
pub mod lowess;
pub mod mediation;
mod moments;
mod negative_binomial;
mod ols;
pub mod ordinal;
mod passive_aggressive;
pub mod path;
pub mod penalized_glm;
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
    ArdRegression, ArdRegressionBuilder, BayesianRidge, BayesianRidgeBuilder, CredibleIntervals,
    FittedArd, FittedBayesianRidge,
};
pub use binomial::{BinomialRegressor, FittedBinomial};
pub use bls::{BlsRegressor, FittedBls};
pub use elastic_net::{ElasticNetRegressor, FittedElasticNet};
pub use gamma::{FittedGamma, GammaRegressor, GammaRegressorBuilder};
pub use glmm::{
    FactorSummary, FittedGlmm, GlmmDdf, GlmmInference, GlmmRegressor, GlmmRegressorBuilder,
};
pub use huber::{FittedHuber, HuberRegressor, HuberRegressorBuilder};
pub use isotonic::{FittedIsotonic, IsotonicRegressor, IsotonicRegressorBuilder, OutOfBounds};
pub use lars::{FittedLars, LarsMethod, LarsRegressor, LarsRegressorBuilder};
pub use lm_dynamic::{
    FittedLmDynamic, InformationCriterion, LmDynamicRegressor, LmDynamicRegressorBuilder, ModelSpec,
};
pub use logistic::{FittedLogistic, LogisticRegression, LogisticRegressionBuilder, Penalty};
pub use moments::MomentAccumulator;
pub use negative_binomial::{FittedNegativeBinomial, NegativeBinomialRegressor};
pub use ols::{FittedOls, OlsRegressor, OlsRegressorBuilder};
pub use ordinal::{
    FittedOrdinal, OrdinalLink, OrdinalQuantileMethod, OrdinalRegressor, OrdinalRegressorBuilder,
};
pub use passive_aggressive::{
    FittedPassiveAggressive, PaLoss, PaState, PassiveAggressiveRegressor,
    PassiveAggressiveRegressorBuilder,
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
pub use traits::{FittedRegressor, RegressionError, Regressor};
pub use tweedie::{FittedTweedie, TweedieRegressor};
pub use wls::{FittedWls, WlsRegressor, WlsRegressorBuilder};
