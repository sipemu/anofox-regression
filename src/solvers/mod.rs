//! Regression solvers implementing various estimation methods.

pub mod aid;
pub mod alm;
mod binomial;
mod bls;
mod elastic_net;
mod huber;
mod isotonic;
pub mod lm_dynamic;
mod logistic;
pub mod lowess;
mod negative_binomial;
mod ols;
mod pls;
mod poisson;
mod quantile;
mod ridge;
mod rls;
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
pub use binomial::{BinomialRegressor, FittedBinomial};
pub use bls::{BlsRegressor, FittedBls};
pub use elastic_net::{ElasticNetRegressor, FittedElasticNet};
pub use huber::{FittedHuber, HuberRegressor, HuberRegressorBuilder};
pub use isotonic::{FittedIsotonic, IsotonicRegressor, IsotonicRegressorBuilder, OutOfBounds};
pub use lm_dynamic::{
    FittedLmDynamic, InformationCriterion, LmDynamicRegressor, LmDynamicRegressorBuilder, ModelSpec,
};
pub use logistic::{FittedLogistic, LogisticRegression, LogisticRegressionBuilder, Penalty};
pub use negative_binomial::{FittedNegativeBinomial, NegativeBinomialRegressor};
pub use ols::{FittedOls, OlsRegressor, OlsRegressorBuilder};
pub use pls::{FittedPls, PlsRegressor};
pub use poisson::{FittedPoisson, PoissonRegressor};
pub use quantile::{FittedQuantile, QuantileRegressor, QuantileRegressorBuilder};
pub use ridge::{FittedRidge, RidgeRegressor, RidgeRegressorBuilder};
pub use rls::{FittedRls, RlsRegressor};
pub use traits::{FittedRegressor, RegressionError, Regressor};
pub use tweedie::{FittedTweedie, TweedieRegressor};
pub use wls::{FittedWls, WlsRegressor, WlsRegressorBuilder};
