//! Regression solvers implementing various estimation methods.

pub mod aid;
pub mod alm;
mod binomial;
mod bls;
mod elastic_net;
pub mod lm_dynamic;
pub mod lowess;
mod negative_binomial;
mod ols;
mod poisson;
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
pub use lm_dynamic::{
    FittedLmDynamic, InformationCriterion, LmDynamicRegressor, LmDynamicRegressorBuilder, ModelSpec,
};
pub use negative_binomial::{FittedNegativeBinomial, NegativeBinomialRegressor};
pub use ols::{FittedOls, OlsRegressor};
pub use poisson::{FittedPoisson, PoissonRegressor};
pub use ridge::{FittedRidge, RidgeRegressor};
pub use rls::{FittedRls, RlsRegressor};
pub use traits::{FittedRegressor, RegressionError, Regressor};
pub use tweedie::{FittedTweedie, TweedieRegressor};
pub use wls::{FittedWls, WlsRegressor};
