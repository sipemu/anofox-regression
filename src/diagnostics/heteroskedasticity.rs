//! Breusch-Pagan and White tests for heteroskedasticity (#80).
//!
//! Both regress the squared residuals of a linear fit on auxiliary regressors
//! (with an intercept) and compare a score statistic to `χ²(df)`, `df` being the
//! rank of the auxiliary design minus one, as `lmtest::bptest`:
//!
//! * studentized (Koenker 1981, the `bptest` default):
//!   `n · R²` of `e²` on `Z`;
//! * non-studentized (Breusch & Pagan 1979): `ESS / 2` of `e² / σ̂² - 1` on `Z`,
//!   with `σ̂² = Σ e² / n`.
//!
//! The White test is the studentized statistic with the regressors, their squares
//! and their pairwise cross-products as `Z` (`bptest(fit, ~ x1 * x2 + I(x1^2) + I(x2^2))`).

use super::partial_residuals::Residualizer;
use crate::solvers::RegressionError;
use faer::{Col, Mat};
use statrs::distribution::{ChiSquared, ContinuousCDF};

/// Result of a heteroskedasticity test.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HeteroskedasticityTest {
    /// Test statistic (asymptotically `χ²(df)` under homoskedasticity).
    pub statistic: f64,
    /// Degrees of freedom: rank of the auxiliary regressors (without the intercept).
    pub df: f64,
    /// Upper-tail p-value.
    pub p_value: f64,
    /// `"breusch_pagan_koenker"`, `"breusch_pagan"` or `"white"`.
    pub method: &'static str,
}

/// Breusch-Pagan test of the residuals `residuals` of a linear fit against the
/// auxiliary regressors `z` (one column per regressor, no intercept column; use
/// the model's design for the usual test). `studentize = true` is Koenker's
/// robust variant (the `lmtest::bptest` default).
///
/// # Errors
///
/// [`RegressionError::DimensionMismatch`] when the lengths differ,
/// [`RegressionError::InsufficientObservations`] when `n <= df + 1`, and
/// [`RegressionError::NumericalError`] for non-finite input or all-zero residuals.
pub fn breusch_pagan(
    residuals: &Col<f64>,
    z: &Mat<f64>,
    studentize: bool,
) -> Result<HeteroskedasticityTest, RegressionError> {
    let method = if studentize {
        "breusch_pagan_koenker"
    } else {
        "breusch_pagan"
    };
    score_test(residuals, z, studentize, method)
}

/// White's test: the studentized Breusch-Pagan test with `x`, the squares and the
/// pairwise cross-products of its columns as auxiliary regressors (duplicates,
/// e.g. the square of a 0/1 column, are absorbed by the rank).
///
/// # Errors
///
/// As [`breusch_pagan`].
pub fn white_test(
    residuals: &Col<f64>,
    x: &Mat<f64>,
) -> Result<HeteroskedasticityTest, RegressionError> {
    let n = x.nrows();
    let p = x.ncols();
    let mut cols: Vec<Vec<f64>> = (0..p)
        .map(|j| (0..n).map(|i| x[(i, j)]).collect())
        .collect();
    for a in 0..p {
        for b in a..p {
            cols.push((0..n).map(|i| x[(i, a)] * x[(i, b)]).collect());
        }
    }
    let z = Mat::from_fn(n, cols.len(), |i, j| cols[j][i]);
    score_test(residuals, &z, true, "white")
}

fn score_test(
    residuals: &Col<f64>,
    z: &Mat<f64>,
    studentize: bool,
    method: &'static str,
) -> Result<HeteroskedasticityTest, RegressionError> {
    let n = residuals.nrows();
    if z.nrows() != n {
        return Err(RegressionError::DimensionMismatch {
            x_rows: z.nrows(),
            y_len: n,
        });
    }
    if (0..n).any(|i| !residuals[i].is_finite())
        || z.col_iter().any(|c| c.iter().any(|v| !v.is_finite()))
    {
        return Err(RegressionError::NumericalError(
            "residuals and regressors must be finite".to_string(),
        ));
    }
    let aux = Mat::from_fn(
        n,
        z.ncols() + 1,
        |i, j| if j == 0 { 1.0 } else { z[(i, j - 1)] },
    );
    let proj = Residualizer::new(&aux, None);
    let df = proj.rank().saturating_sub(1);
    if n <= df + 1 {
        return Err(RegressionError::InsufficientObservations {
            needed: df + 2,
            got: n,
        });
    }
    let e2: Vec<f64> = (0..n).map(|i| residuals[i] * residuals[i]).collect();
    let sigma2 = e2.iter().sum::<f64>() / n as f64;
    if sigma2 <= 0.0 {
        return Err(RegressionError::NumericalError(
            "all residuals are zero".to_string(),
        ));
    }
    // Explained sum of squares of the auxiliary regression of v (centered).
    let ess_tss = |v: &[f64]| {
        let r = proj.residuals(v);
        let mean = v.iter().sum::<f64>() / n as f64;
        let tss: f64 = v.iter().map(|x| (x - mean).powi(2)).sum();
        let rss: f64 = r.iter().map(|x| x * x).sum();
        (tss - rss, tss)
    };
    let statistic = if studentize {
        let (ess, tss) = ess_tss(&e2);
        if tss > 0.0 {
            n as f64 * ess / tss
        } else {
            0.0
        }
    } else {
        let f: Vec<f64> = e2.iter().map(|v| v / sigma2 - 1.0).collect();
        0.5 * ess_tss(&f).0
    };
    let statistic = statistic.max(0.0);
    let p_value = if df == 0 {
        f64::NAN
    } else {
        ChiSquared::new(df as f64)
            .map(|d| d.sf(statistic))
            .unwrap_or(f64::NAN)
    };
    Ok(HeteroskedasticityTest {
        statistic,
        df: df as f64,
        p_value,
        method,
    })
}
