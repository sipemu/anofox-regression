//! Regression solvers implementing various estimation methods.

mod elastic_net;
mod ols;
mod ridge;
mod rls;
mod traits;
mod wls;

pub use elastic_net::{ElasticNetRegressor, FittedElasticNet};
pub use ols::{FittedOls, OlsRegressor};
pub use ridge::{FittedRidge, RidgeRegressor};
pub use rls::{FittedRls, RlsRegressor};
pub use traits::{FittedRegressor, RegressionError, Regressor};
pub use wls::{FittedWls, WlsRegressor};
