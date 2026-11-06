//! Student's t reference distribution that is the standard normal at
//! `df = ∞`: statrs' `StudentsT` accepts an infinite `df` but its
//! `inverse_cdf` does not terminate there.

use statrs::distribution::{Continuous, ContinuousCDF, Normal, StudentsT, StudentsTError};
use statrs::statistics::{Max, Min};

/// `t(df)` with location 0 and scale 1, or the standard normal for an
/// infinite `df`.
#[derive(Debug, Clone, Copy)]
pub(crate) enum StudentT {
    T(StudentsT),
    Normal(Normal),
}

/// `StudentsT::new(0.0, 1.0, df)`, with `df = +∞` mapped to the standard
/// normal. Errors as statrs for a NaN or non-positive `df`.
pub(crate) fn student_t(df: f64) -> Result<StudentT, StudentsTError> {
    if df == f64::INFINITY {
        return Ok(StudentT::Normal(
            Normal::new(0.0, 1.0).expect("standard normal"),
        ));
    }
    StudentsT::new(0.0, 1.0, df).map(StudentT::T)
}

impl Min<f64> for StudentT {
    fn min(&self) -> f64 {
        f64::NEG_INFINITY
    }
}

impl Max<f64> for StudentT {
    fn max(&self) -> f64 {
        f64::INFINITY
    }
}

impl ContinuousCDF<f64, f64> for StudentT {
    fn cdf(&self, x: f64) -> f64 {
        match self {
            Self::T(d) => d.cdf(x),
            Self::Normal(d) => d.cdf(x),
        }
    }

    fn sf(&self, x: f64) -> f64 {
        match self {
            Self::T(d) => d.sf(x),
            Self::Normal(d) => d.sf(x),
        }
    }

    fn inverse_cdf(&self, p: f64) -> f64 {
        match self {
            Self::T(d) => d.inverse_cdf(p),
            Self::Normal(d) => d.inverse_cdf(p),
        }
    }
}

impl Continuous<f64, f64> for StudentT {
    fn pdf(&self, x: f64) -> f64 {
        match self {
            Self::T(d) => d.pdf(x),
            Self::Normal(d) => d.pdf(x),
        }
    }

    fn ln_pdf(&self, x: f64) -> f64 {
        match self {
            Self::T(d) => d.ln_pdf(x),
            Self::Normal(d) => d.ln_pdf(x),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn infinite_df_is_the_normal() {
        let t = student_t(f64::INFINITY).unwrap();
        assert!((t.inverse_cdf(0.975) - 1.959963984540054).abs() < 1e-9);
        assert!((t.sf(1.96) - 0.024997895148220435).abs() < 1e-10);
        assert!((t.pdf(0.0) - 0.3989422804014327).abs() < 1e-12);
        let t = student_t(10.0).unwrap();
        assert!((t.inverse_cdf(0.975) - 2.2281388519649385).abs() < 1e-9);
        assert!(student_t(f64::NAN).is_err() && student_t(0.0).is_err());
    }
}
