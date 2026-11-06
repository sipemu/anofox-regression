//! Crate-private numerical kernels shared by several solvers.
//!
//! Each routine here used to exist as a private copy in two or more modules
//! (#284); they are collected so there is one implementation of each. Nothing
//! in this module is part of the public API.

pub(crate) mod linalg;
pub(crate) mod optim;
pub(crate) mod quantile;
pub(crate) mod t_dist;

pub(crate) use t_dist::student_t;
