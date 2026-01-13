//! Regression model wrappers for WebAssembly.

mod isotonic;
mod ols;
mod poisson;
mod quantile;
mod ridge;

pub use isotonic::*;
pub use ols::*;
pub use poisson::*;
pub use quantile::*;
pub use ridge::*;
