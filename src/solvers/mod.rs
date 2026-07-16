//! Regression solvers implementing various estimation methods.

pub mod aid;
pub mod alm;
mod bayesian;
mod binomial;
mod bls;
mod elastic_net;
mod gamma;
mod huber;
mod isotonic;
mod lars;
pub mod lm_dynamic;
mod logistic;
pub mod lowess;
mod moments;
mod negative_binomial;
mod ols;
mod passive_aggressive;
mod pls;
mod poisson;
mod pspline;
mod quantile;
mod ransac;
mod ridge;
mod rls;
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
    ArdRegression, ArdRegressionBuilder, BayesianRidge, BayesianRidgeBuilder, FittedArd,
    FittedBayesianRidge,
};
pub use binomial::{BinomialRegressor, FittedBinomial};
pub use bls::{BlsRegressor, FittedBls};
pub use elastic_net::{ElasticNetRegressor, FittedElasticNet};
pub use gamma::{FittedGamma, GammaRegressor, GammaRegressorBuilder};
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
pub use passive_aggressive::{
    FittedPassiveAggressive, PaLoss, PaState, PassiveAggressiveRegressor,
    PassiveAggressiveRegressorBuilder,
};
pub use pls::{FittedPls, PlsRegressor};
pub use poisson::{FittedPoisson, PoissonRegressor};
pub use pspline::{FittedPSpline, PSplineRegressor};
pub use quantile::{FittedQuantile, QuantileRegressor, QuantileRegressorBuilder};
pub use ransac::{FittedRansac, RansacRegressor, RansacRegressorBuilder};
pub use ridge::{FittedRidge, RidgeRegressor, RidgeRegressorBuilder};
pub use rls::{FittedRls, RlsRegressor};
pub use theil_sen::{FittedTheilSen, TheilSenRegressor, TheilSenRegressorBuilder};
pub use traits::{FittedRegressor, RegressionError, Regressor};
pub use tweedie::{FittedTweedie, TweedieRegressor};
pub use wls::{FittedWls, WlsRegressor, WlsRegressorBuilder};
