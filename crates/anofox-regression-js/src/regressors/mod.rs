//! Regression model wrappers for WebAssembly.

mod binomial;
mod elastic_net;
mod isotonic;
mod negative_binomial;
mod ols;
mod poisson;
mod quantile;
mod ridge;
mod tweedie;
mod wls;

pub use binomial::*;
pub use elastic_net::*;
pub use isotonic::*;
pub use negative_binomial::*;
pub use ols::*;
pub use poisson::*;
pub use quantile::*;
pub use ridge::*;
pub use tweedie::*;
pub use wls::*;
