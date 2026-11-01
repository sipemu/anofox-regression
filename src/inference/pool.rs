//! Rubin's rules for combining fits to multiply imputed data (RMS §3.8):
//! pooled coefficients and covariance with the per-coefficient variance
//! inflation, fraction of missing information and degrees of freedom
//! (`Hmisc::fit.mult.impute`, and the Barnard–Rubin small-sample d.f. of
//! `mice::pool`), and the D1 pooled Wald test of several coefficients (Li,
//! Raghunathan and Rubin 1991, `mice::D1`).
//!
//! ```rust,ignore
//! use anofox_regression::inference::pool::{pool_rubin, pooled_wald_d1};
//!
//! // one fit per completed data set
//! let pooled = pool_rubin(&coefs, &vcovs, Some((n - p) as f64))?;
//! let test = pooled_wald_d1(&pooled, &[1, 2])?;
//! ```

use crate::solvers::RegressionError;
use faer::{Col, Mat};
use statrs::distribution::{ContinuousCDF, FisherSnedecor};

fn invalid(msg: &str) -> RegressionError {
    RegressionError::NumericalError(msg.to_string())
}

/// Fits combined by [`pool_rubin`].
#[derive(Debug, Clone)]
pub struct PooledFit {
    /// Number of imputations `m`.
    pub m: usize,
    /// Mean of the coefficients, `Q̄`.
    pub coefficients: Col<f64>,
    /// Total covariance `T = Ū + (1 + 1/m) B` (`fit.mult.impute`'s `var`).
    pub vcov: Mat<f64>,
    /// Mean within-imputation covariance `Ū`.
    pub within: Mat<f64>,
    /// Between-imputation covariance `B` of the coefficients.
    pub between: Mat<f64>,
    /// `diag(T) / diag(Ū)` (`variance.inflation.impute`).
    pub variance_inflation: Col<f64>,
    /// Relative increase in variance `r = (1 + 1/m) B_jj / Ū_jj`.
    pub riv: Col<f64>,
    /// `λ = (1 + 1/m) B_jj / T_jj`, the proportion of variance due to
    /// missingness; Hmisc's `missingInfo`.
    pub lambda: Col<f64>,
    /// Rubin's d.f. `(m − 1)(1 + 1/r)²` (`dfmi`).
    pub df_rubin: Col<f64>,
    /// Barnard–Rubin d.f. given the complete-data d.f. (`mice::pool`);
    /// `None` without a complete-data d.f.
    pub df_barnard_rubin: Option<Col<f64>>,
    /// Fraction of missing information `(r + 2/(ν + 3))/(1 + r)` with `ν`
    /// the Barnard–Rubin d.f. (Rubin's d.f. without a complete-data d.f.),
    /// as `mice::pool`'s `fmi`.
    pub fmi: Col<f64>,
    /// The complete-data d.f. given to [`pool_rubin`] (`mice`'s `dfcom`).
    pub df_complete: Option<f64>,
}

impl PooledFit {
    /// Standard errors `√diag(T)`.
    pub fn std_errors(&self) -> Col<f64> {
        Col::from_fn(self.vcov.nrows(), |j| self.vcov[(j, j)].sqrt())
    }
}

/// Combine `m ≥ 2` fits of the same model to imputed data sets by Rubin's
/// rules. `estimates[i]` and `vcovs[i]` are the coefficients and their
/// covariance from imputation `i`. `df_complete` is the residual d.f. the
/// model would have without missing data (e.g. `n − p`), for the
/// Barnard–Rubin d.f.
pub fn pool_rubin(
    estimates: &[Col<f64>],
    vcovs: &[Mat<f64>],
    df_complete: Option<f64>,
) -> Result<PooledFit, RegressionError> {
    let m = estimates.len();
    if m < 2 || vcovs.len() != m {
        return Err(invalid(
            "pooling needs two or more fits, each with a covariance",
        ));
    }
    let p = estimates[0].nrows();
    if estimates.iter().any(|e| e.nrows() != p)
        || vcovs.iter().any(|v| v.nrows() != p || v.ncols() != p)
    {
        return Err(invalid("every fit must have the same coefficients"));
    }
    if df_complete.is_some_and(|d| d.is_nan() || d <= 0.0) {
        return Err(invalid("the complete-data d.f. must be positive"));
    }
    let mf = m as f64;
    let qbar = Col::from_fn(p, |j| estimates.iter().map(|e| e[j]).sum::<f64>() / mf);
    let within = Mat::from_fn(p, p, |a, b| {
        vcovs.iter().map(|v| v[(a, b)]).sum::<f64>() / mf
    });
    let between = Mat::from_fn(p, p, |a, b| {
        estimates
            .iter()
            .map(|e| (e[a] - qbar[a]) * (e[b] - qbar[b]))
            .sum::<f64>()
            / (mf - 1.0)
    });
    let infl = 1.0 + 1.0 / mf;
    let vcov = Mat::from_fn(p, p, |a, b| within[(a, b)] + infl * between[(a, b)]);
    let riv = Col::from_fn(p, |j| infl * between[(j, j)] / within[(j, j)]);
    let lambda = Col::from_fn(p, |j| infl * between[(j, j)] / vcov[(j, j)]);
    let df_rubin = Col::from_fn(p, |j| (mf - 1.0) * (1.0 + 1.0 / riv[j]).powi(2));
    let df_barnard_rubin = df_complete.map(|dc| {
        Col::from_fn(p, |j| {
            let old = (mf - 1.0) / lambda[j].powi(2);
            let obs = (dc + 1.0) / (dc + 3.0) * dc * (1.0 - lambda[j]);
            old * obs / (old + obs)
        })
    });
    let df = df_barnard_rubin.as_ref().unwrap_or(&df_rubin);
    let fmi = Col::from_fn(p, |j| (riv[j] + 2.0 / (df[j] + 3.0)) / (1.0 + riv[j]));
    Ok(PooledFit {
        m,
        variance_inflation: Col::from_fn(p, |j| vcov[(j, j)] / within[(j, j)]),
        coefficients: qbar,
        vcov,
        within,
        between,
        riv,
        lambda,
        df_rubin,
        df_barnard_rubin,
        fmi,
        df_complete,
    })
}

/// The D1 pooled Wald test ([`pooled_wald_d1`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PooledWald {
    /// F statistic.
    pub f: f64,
    /// Numerator d.f. (number of coefficients tested).
    pub df1: f64,
    /// Denominator d.f.
    pub df2: f64,
    /// P-value.
    pub p_value: f64,
    /// Average relative increase in variance `r₁ = (1 + 1/m) tr(B Ū⁻¹)/k`.
    pub riv: f64,
}

/// Test that the coefficients `indices` are all zero with the D1 statistic
/// of Li, Raghunathan and Rubin (1991): `F = Q̄ᵀ T̃⁻¹ Q̄ / k` with
/// `T̃ = (1 + r₁) Ū` on `k` and `ν₂` d.f., as `mice::D1` (`mitml`'s `.D1`).
///
/// With a complete-data d.f. (`df_complete` of [`pool_rubin`]; `mice::D1`
/// takes it from the model) `ν₂` is Reiter's (2007) small-sample d.f.;
/// without one it is Li et al.'s `4 + (t − 4)(1 + (1 − 2/t)/r₁)²`,
/// `t = k(m − 1)` (for `t ≤ 4`: `t(1 + 1/k)(1 + 1/r₁)²/2`). Reiter's d.f.
/// needs `t > 4`, i.e. enough imputations.
pub fn pooled_wald_d1(
    pooled: &PooledFit,
    indices: &[usize],
) -> Result<PooledWald, RegressionError> {
    let k = indices.len();
    let p = pooled.coefficients.nrows();
    if k == 0 || indices.iter().any(|&i| i >= p) {
        return Err(invalid("indices must be non-empty and in range"));
    }
    let sub = |m: &Mat<f64>| Mat::from_fn(k, k, |a, b| m[(indices[a], indices[b])]);
    let (w, b) = (sub(&pooled.within), sub(&pooled.between));
    let w_inv = crate::inference::compute_matrix_inverse(&w)
        .map_err(|_| RegressionError::SingularMatrix)?;
    let mf = pooled.m as f64;
    let kf = k as f64;
    let bw = &b * &w_inv;
    let r1 = (1.0 + 1.0 / mf) * (0..k).map(|i| bw[(i, i)]).sum::<f64>() / kf;
    let theta = Col::from_fn(k, |a| pooled.coefficients[indices[a]]);
    let quad = (theta.transpose() * &w_inv * &theta) / (1.0 + r1);
    let f = quad / kf;
    let nu1 = kf * (mf - 1.0);
    let df2 = if let Some(dc) = pooled.df_complete {
        reiter_df(r1, nu1, dc)
    } else if nu1 > 4.0 {
        4.0 + (nu1 - 4.0) * (1.0 + (1.0 - 2.0 / nu1) / r1).powi(2)
    } else {
        nu1 * (1.0 + 1.0 / kf) * (1.0 + 1.0 / r1).powi(2) / 2.0
    };
    let p_value = FisherSnedecor::new(kf, df2).map_or(f64::NAN, |d| d.sf(f));
    Ok(PooledWald {
        f,
        df1: kf,
        df2,
        p_value,
        riv: r1,
    })
}

/// Reiter's (2007, Eq. 1–2) denominator d.f. of the D1 test, as `mitml`'s
/// `.D1` with `df.com`: `a = r t/(t − 2)`, `ν* = (ν_com + 1)/(ν_com + 3) ν_com`.
fn reiter_df(r: f64, t: f64, df_com: f64) -> f64 {
    let a = r * t / (t - 2.0);
    let vstar = (df_com + 1.0) / (df_com + 3.0) * df_com;
    let c0 = 1.0 / (t - 4.0);
    let c1 = vstar - 2.0 * (1.0 + a);
    let c2 = vstar - 4.0 * (1.0 + a);
    let a2 = a * a;
    let z = 1.0 / c2
        + c0 * (a2 * c1 / ((1.0 + a).powi(2) * c2))
        + c0 * (8.0 * a2 * c1 / ((1.0 + a) * c2 * c2) + 4.0 * a2 / ((1.0 + a) * c2))
        + c0 * (4.0 * a2 / (c2 * c1) + 16.0 * a2 * c1 / c2.powi(3))
        + c0 * (8.0 * a2 / (c2 * c2));
    4.0 + 1.0 / z
}
