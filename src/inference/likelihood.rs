//! Likelihood-ratio and score tests for nested models and
//! profile-likelihood confidence intervals (RMS Ch 9).
//!
//! * [`lr_test`] / [`model_lr_test`] – the likelihood-ratio χ² of nested
//!   fits, and of a fit against its null model, for every fit implementing
//!   [`LikelihoodFit`] (binomial, Poisson, negative binomial, Gamma /
//!   Tweedie, AFT, Cox), as `anova(fit0, fit1, test = "LRT")`.
//! * [`fit_glm`] – a compact IRLS fit of any [`GlmFamily`] with prior
//!   weights and offsets, the engine of the next two.
//! * [`glm_score_test`] – Rao's score test for adding predictors, evaluated
//!   at the reduced fit only (`anova(..., test = "Rao")`), with the
//!   per-column z statistics of `statmod::glm.scoretest`.
//! * [`profile_confint`] – profile-likelihood confidence intervals,
//!   reproducing `confint.glm` (R ≥ 4.4; `MASS::confint.glm` before):
//!   the same profiling grid, R's FMM spline and the same inversion.

use crate::core::GlmFamily;
use faer::linalg::solvers::Solve;
use faer::Mat;
use statrs::distribution::{ChiSquared, ContinuousCDF, Normal, StudentsT};
use thiserror::Error;

/// Errors of the likelihood tools.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum LikelihoodError {
    /// Inconsistent inputs.
    #[error("invalid data: {0}")]
    Data(String),
    /// The IRLS iteration failed.
    #[error("the fit failed: {0}")]
    Fit(&'static str),
    /// Profiling found a better optimum than the fit.
    #[error("profiling found a better solution; the original fit had not converged")]
    NotConverged,
}

/// A χ² test.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChiSqTest {
    /// Test statistic.
    pub statistic: f64,
    /// Degrees of freedom.
    pub df: usize,
    /// p-value.
    pub p_value: f64,
}

impl ChiSqTest {
    fn new(statistic: f64, df: usize) -> Self {
        let p_value = if df == 0 {
            f64::NAN
        } else {
            ChiSquared::new(df as f64).map_or(f64::NAN, |d| d.sf(statistic))
        };
        Self {
            statistic,
            df,
            p_value,
        }
    }
}

/// A fitted model with a likelihood, for likelihood-ratio tests.
pub trait LikelihoodFit {
    /// `−2 log L` up to a constant shared by fits of the same data and
    /// family (the deviance for GLMs), *not* divided by the dispersion.
    fn minus_two_loglik(&self) -> f64;
    /// The same for the null (intercept-only) model, if known.
    fn null_minus_two_loglik(&self) -> Option<f64>;
    /// Dispersion the deviance difference is divided by: 1 for families
    /// with a fixed dispersion, the estimate otherwise.
    fn lr_dispersion(&self) -> f64;
    /// Number of estimated mean (and shape) parameters.
    fn n_parameters(&self) -> usize;
}

/// Likelihood-ratio test of the nested fits `reduced ⊂ full`:
/// `(−2ℓ_reduced + 2ℓ_full) / φ_full` on `k_full − k_reduced` d.f.
pub fn lr_test<R: LikelihoodFit + ?Sized, F: LikelihoodFit + ?Sized>(
    reduced: &R,
    full: &F,
) -> ChiSqTest {
    let stat = (reduced.minus_two_loglik() - full.minus_two_loglik()) / full.lr_dispersion();
    ChiSqTest::new(
        stat,
        full.n_parameters().saturating_sub(reduced.n_parameters()),
    )
}

/// Model likelihood-ratio χ² against the null model, on `k − 1` d.f. (the
/// "Model L.R." rms prints).
pub fn model_lr_test<F: LikelihoodFit + ?Sized>(fit: &F) -> Option<ChiSqTest> {
    let null = fit.null_minus_two_loglik()?;
    Some(ChiSqTest::new(
        (null - fit.minus_two_loglik()) / fit.lr_dispersion(),
        fit.n_parameters().saturating_sub(1),
    ))
}

macro_rules! glm_likelihood_fit {
    ($t:ty, $fixed:expr) => {
        impl LikelihoodFit for $t {
            fn minus_two_loglik(&self) -> f64 {
                self.deviance
            }
            fn null_minus_two_loglik(&self) -> Option<f64> {
                Some(self.null_deviance)
            }
            fn lr_dispersion(&self) -> f64 {
                if $fixed {
                    1.0
                } else {
                    self.dispersion
                }
            }
            fn n_parameters(&self) -> usize {
                crate::solvers::FittedRegressor::result(self).n_parameters
            }
        }
    };
}

glm_likelihood_fit!(crate::solvers::FittedBinomial, true);
glm_likelihood_fit!(crate::solvers::FittedPoisson, true);
glm_likelihood_fit!(crate::solvers::FittedTweedie, false);

impl LikelihoodFit for crate::solvers::FittedNegativeBinomial {
    /// `−2 log L` (the deviance depends on θ, which differs between fits).
    fn minus_two_loglik(&self) -> f64 {
        -2.0 * crate::solvers::FittedRegressor::result(self).log_likelihood
    }
    fn null_minus_two_loglik(&self) -> Option<f64> {
        Some(self.minus_two_loglik() + self.null_deviance - self.deviance)
    }
    fn lr_dispersion(&self) -> f64 {
        1.0
    }
    fn n_parameters(&self) -> usize {
        crate::solvers::FittedRegressor::result(self).n_parameters
    }
}

impl LikelihoodFit for crate::solvers::FittedGamma {
    fn minus_two_loglik(&self) -> f64 {
        self.inner().minus_two_loglik()
    }
    fn null_minus_two_loglik(&self) -> Option<f64> {
        self.inner().null_minus_two_loglik()
    }
    fn lr_dispersion(&self) -> f64 {
        self.inner().lr_dispersion()
    }
    fn n_parameters(&self) -> usize {
        self.inner().n_parameters()
    }
}

impl LikelihoodFit for crate::solvers::aft::FittedAft {
    fn minus_two_loglik(&self) -> f64 {
        -2.0 * self.log_likelihood()
    }
    fn null_minus_two_loglik(&self) -> Option<f64> {
        Some(-2.0 * self.null_log_likelihood())
    }
    fn lr_dispersion(&self) -> f64 {
        1.0
    }
    /// Coefficients plus the intercept (if fitted); the scale is common to
    /// nested AFT fits of one distribution, so it cancels in the d.f.
    fn n_parameters(&self) -> usize {
        self.coefficients().nrows() + usize::from(self.intercept().is_some())
    }
}

impl LikelihoodFit for crate::survival::cox::CoxFit {
    fn minus_two_loglik(&self) -> f64 {
        -2.0 * self.loglik.1
    }
    fn null_minus_two_loglik(&self) -> Option<f64> {
        Some(-2.0 * self.loglik.0)
    }
    fn lr_dispersion(&self) -> f64 {
        1.0
    }
    fn n_parameters(&self) -> usize {
        self.df + 1
    }
}

/// Dispersion of a GLM.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Dispersion {
    /// Known (1 for binomial and Poisson).
    Fixed(f64),
    /// Pearson estimate `Σ wᵢ (yᵢ − μᵢ)² / V(μᵢ) / (n − k)` (Gamma, Gaussian, …).
    Estimated,
}

/// Data of a GLM: design without intercept column, response, optional
/// prior weights and offset.
#[derive(Debug, Clone, Copy)]
pub struct GlmData<'a> {
    /// Covariates (no intercept column).
    pub x: &'a Mat<f64>,
    /// Fit an intercept.
    pub with_intercept: bool,
    /// Response.
    pub y: &'a [f64],
    /// Prior weights.
    pub weights: Option<&'a [f64]>,
    /// Offset.
    pub offset: Option<&'a [f64]>,
}

impl GlmData<'_> {
    fn design(&self) -> Mat<f64> {
        let off = usize::from(self.with_intercept);
        Mat::from_fn(self.x.nrows(), self.x.ncols() + off, |i, j| {
            if j < off {
                1.0
            } else {
                self.x[(i, j - off)]
            }
        })
    }
    fn check(&self) -> Result<(), LikelihoodError> {
        let n = self.y.len();
        let bad = self.x.nrows() != n
            || self.weights.is_some_and(|w| w.len() != n)
            || self.offset.is_some_and(|o| o.len() != n);
        if bad {
            return Err(LikelihoodError::Data(
                "lengths of x, y, weights and offset differ".into(),
            ));
        }
        Ok(())
    }
}

/// A GLM fitted by [`fit_glm`].
#[derive(Debug, Clone)]
pub struct GlmFit {
    /// Coefficients (intercept first when fitted).
    pub coefficients: Vec<f64>,
    /// Covariance `φ (XᵀWX)⁻¹`.
    pub vcov: Mat<f64>,
    /// Deviance.
    pub deviance: f64,
    /// Null deviance (intercept-only with the offset, or μ = g⁻¹(offset)).
    pub null_deviance: f64,
    /// Dispersion used for the covariance.
    pub dispersion: f64,
    /// Residual degrees of freedom.
    pub df_residual: usize,
    /// Fitted means.
    pub mu: Vec<f64>,
    /// IRLS iterations.
    pub iterations: usize,
}

impl LikelihoodFit for GlmFit {
    fn minus_two_loglik(&self) -> f64 {
        self.deviance
    }
    fn null_minus_two_loglik(&self) -> Option<f64> {
        Some(self.null_deviance)
    }
    fn lr_dispersion(&self) -> f64 {
        self.dispersion
    }
    fn n_parameters(&self) -> usize {
        self.coefficients.len()
    }
}

struct Irls {
    beta: Vec<f64>,
    mu: Vec<f64>,
    deviance: f64,
    /// `(XᵀWX)⁻¹` at the estimate.
    xtwx_inv: Mat<f64>,
    iterations: usize,
}

/// IRLS on a full design matrix (any intercept column included).
fn irls<F: GlmFamily + ?Sized>(
    family: &F,
    x: &Mat<f64>,
    y: &[f64],
    w: &[f64],
    off: &[f64],
) -> Result<Irls, LikelihoodError> {
    let (n, k) = (x.nrows(), x.ncols());
    let dev_of = |mu: &[f64]| -> f64 {
        (0..n)
            .map(|i| w[i] * family.unit_deviance(y[i], mu[i]))
            .sum()
    };
    let mut mu = family.initialize_mu(y);
    let mut eta: Vec<f64> = mu.iter().map(|&m| family.link(m)).collect();
    let mut dev = dev_of(&mu);
    let mut beta = vec![0.0; k];
    for it in 1..=100 {
        let mut ww = vec![0.0; n];
        let mut z = vec![0.0; n];
        for i in 0..n {
            let gp = family.link_derivative(mu[i]);
            ww[i] = w[i] / (family.variance(mu[i]) * gp * gp);
            z[i] = eta[i] - off[i] + (y[i] - mu[i]) * gp;
        }
        let xw = Mat::from_fn(n, k, |i, j| x[(i, j)] * ww[i]);
        let gram = x.transpose() * &xw;
        let llt = gram
            .llt(faer::Side::Lower)
            .map_err(|_| LikelihoodError::Fit("singular weighted Gram matrix"))?;
        let rhs = faer::Col::from_fn(k, |j| (0..n).map(|i| xw[(i, j)] * z[i]).sum::<f64>());
        let b = llt.solve(&rhs);
        let mut new_beta: Vec<f64> = (0..k).map(|j| b[j]).collect();
        let mut step = 1.0;
        let (mut new_eta, mut new_mu, mut new_dev);
        loop {
            new_eta = (0..n)
                .map(|i| off[i] + (0..k).map(|j| x[(i, j)] * new_beta[j]).sum::<f64>())
                .collect::<Vec<f64>>();
            new_mu = new_eta
                .iter()
                .map(|&e| family.link_inverse(e))
                .collect::<Vec<f64>>();
            new_dev = dev_of(&new_mu);
            let valid = new_dev.is_finite() && family.all_valid_mu(&new_mu);
            if valid || step < 1e-10 {
                break;
            }
            step /= 2.0;
            new_beta = (0..k).map(|j| beta[j] + step * (b[j] - beta[j])).collect();
        }
        beta = new_beta;
        eta = new_eta;
        mu = new_mu;
        let done = (new_dev - dev).abs() / (new_dev.abs() + 0.1) < 1e-14;
        dev = new_dev;
        if done || it == 100 {
            // Information at the final means.
            let mut ww = vec![0.0; n];
            for i in 0..n {
                let gp = family.link_derivative(mu[i]);
                ww[i] = w[i] / (family.variance(mu[i]) * gp * gp);
            }
            let xw = Mat::from_fn(n, k, |i, j| x[(i, j)] * ww[i]);
            let gram = x.transpose() * &xw;
            let gram_inv = gram
                .llt(faer::Side::Lower)
                .map_err(|_| LikelihoodError::Fit("singular weighted Gram matrix"))?
                .solve(Mat::<f64>::identity(k, k));
            return Ok(Irls {
                beta,
                mu,
                deviance: dev,
                xtwx_inv: gram_inv,
                iterations: it,
            });
        }
    }
    Err(LikelihoodError::Fit("no convergence"))
}

/// Fit a GLM of `family` by IRLS (deviance criterion `1e-14`), as
/// `glm.fit`.
pub fn fit_glm<F: GlmFamily + ?Sized>(
    family: &F,
    data: &GlmData,
    dispersion: Dispersion,
) -> Result<GlmFit, LikelihoodError> {
    data.check()?;
    let n = data.y.len();
    let w: Vec<f64> = data.weights.map_or(vec![1.0; n], <[f64]>::to_vec);
    let off: Vec<f64> = data.offset.map_or(vec![0.0; n], <[f64]>::to_vec);
    let x = data.design();
    let k = x.ncols();
    let fit = irls(family, &x, data.y, &w, &off)?;
    let df_residual = w.iter().filter(|&&v| v > 0.0).count().saturating_sub(k);
    let phi = match dispersion {
        Dispersion::Fixed(v) => v,
        Dispersion::Estimated => {
            (0..n)
                .map(|i| w[i] * (data.y[i] - fit.mu[i]).powi(2) / family.variance(fit.mu[i]))
                .sum::<f64>()
                / df_residual as f64
        }
    };
    let null_deviance = if data.with_intercept {
        irls(family, &Mat::from_fn(n, 1, |_, _| 1.0), data.y, &w, &off)?.deviance
    } else {
        (0..n)
            .map(|i| w[i] * family.unit_deviance(data.y[i], family.link_inverse(off[i])))
            .sum()
    };
    Ok(GlmFit {
        coefficients: fit.beta,
        vcov: Mat::from_fn(k, k, |a, b| phi * fit.xtwx_inv[(a, b)]),
        deviance: fit.deviance,
        null_deviance,
        dispersion: phi,
        df_residual,
        mu: fit.mu,
        iterations: fit.iterations,
    })
}

/// Rao score test for adding the columns `x_add` to the model of `reduced`.
#[derive(Debug, Clone, PartialEq)]
pub struct ScoreTest {
    /// Joint score χ² `Uᵀ S⁻¹ U / φ`.
    pub test: ChiSqTest,
    /// Per-column z statistics `U_j / √(S_jj φ)` (`statmod::glm.scoretest`).
    pub z: Vec<f64>,
}

/// Rao score test for adding `x_add` to the GLM of `reduced`, evaluated at
/// the reduced fit (no fit of the larger model). With the scores
/// `U = X₂ᵀ W r` (working weights and residuals of the reduced fit) and
/// `S = X₂ᵀWX₂ − X₂ᵀWX₁ (X₁ᵀWX₁)⁻¹ X₁ᵀWX₂`, the statistic is `Uᵀ S⁻¹ U / φ`.
/// `anova(fit0, fit1, test = "Rao")` uses the dispersion of the larger
/// model; `glm.scoretest` that of the reduced one.
pub fn glm_score_test<F: GlmFamily + ?Sized>(
    family: &F,
    reduced: &GlmData,
    x_add: &Mat<f64>,
    dispersion: f64,
) -> Result<ScoreTest, LikelihoodError> {
    reduced.check()?;
    let n = reduced.y.len();
    if x_add.nrows() != n {
        return Err(LikelihoodError::Data(
            "x_add has the wrong number of rows".into(),
        ));
    }
    let fit = fit_glm(family, reduced, Dispersion::Fixed(1.0))?;
    let w: Vec<f64> = reduced.weights.map_or(vec![1.0; n], <[f64]>::to_vec);
    let x1 = reduced.design();
    let q = x_add.ncols();
    let mut ww = vec![0.0; n];
    let mut wr = vec![0.0; n];
    for i in 0..n {
        let mu = fit.mu[i];
        let gp = family.link_derivative(mu);
        let v = family.variance(mu);
        ww[i] = w[i] / (v * gp * gp);
        wr[i] = w[i] * (reduced.y[i] - mu) / (v * gp);
    }
    let u: Vec<f64> = (0..q)
        .map(|j| (0..n).map(|i| x_add[(i, j)] * wr[i]).sum())
        .collect();
    let cross = |a: &Mat<f64>, b: &Mat<f64>| {
        Mat::from_fn(a.ncols(), b.ncols(), |r, c| {
            (0..n).map(|i| a[(i, r)] * ww[i] * b[(i, c)]).sum::<f64>()
        })
    };
    let a22 = cross(x_add, x_add);
    let a21 = cross(x_add, &x1);
    let a11_inv = fit.vcov.clone(); // (X₁ᵀWX₁)⁻¹ with φ = 1
    let s = &a22 - &a21 * &a11_inv * a21.transpose();
    let uc = faer::Col::from_fn(q, |j| u[j]);
    let x = s.partial_piv_lu().solve(&uc);
    let stat = (0..q).map(|j| x[j] * u[j]).sum::<f64>() / dispersion;
    Ok(ScoreTest {
        test: ChiSqTest::new(stat, q),
        z: (0..q)
            .map(|j| u[j] / (s[(j, j)] * dispersion).sqrt())
            .collect(),
    })
}

/// Profile-likelihood confidence intervals of every coefficient (intercept
/// first), as `confint(glm_fit, level = level)` in R ≥ 4.4.
///
/// Each coefficient is profiled on R's grid (steps of `zmax/5` standard
/// errors from the estimate, at most 10 per side, `zmax = qnorm(1 − α/2)`
/// for a fixed dispersion and `qt(1 − α/2, n − k)` otherwise, `α = (1 −
/// level)/4`), the signed root deviance is interpolated with R's FMM spline
/// and inverted at `qnorm((1 ± level)/2)`. A bound the grid does not reach
/// is NaN, as R's `NA`.
pub fn profile_confint<F: GlmFamily + ?Sized>(
    family: &F,
    data: &GlmData,
    dispersion: Dispersion,
    level: f64,
) -> Result<Vec<(f64, f64)>, LikelihoodError> {
    if !(level > 0.0 && level < 1.0) {
        return Err(LikelihoodError::Data("level must be in (0, 1)".into()));
    }
    let fit = fit_glm(family, data, dispersion)?;
    let n = data.y.len();
    let w: Vec<f64> = data.weights.map_or(vec![1.0; n], <[f64]>::to_vec);
    let off: Vec<f64> = data.offset.map_or(vec![0.0; n], <[f64]>::to_vec);
    let x = data.design();
    let k = x.ncols();
    let alpha = (1.0 - level) / 4.0;
    let zmax = match dispersion {
        Dispersion::Fixed(_) => Normal::new(0.0, 1.0)
            .expect("normal")
            .inverse_cdf(1.0 - alpha / 2.0),
        Dispersion::Estimated => StudentsT::new(0.0, 1.0, (n - k) as f64)
            .map_err(|_| LikelihoodError::Data("too few residual degrees of freedom".into()))?
            .inverse_cdf(1.0 - alpha / 2.0),
    };
    let del = zmax / 5.0;
    let a = (1.0 - level) / 2.0;
    let norm = Normal::new(0.0, 1.0).expect("normal");
    let cutoff = [norm.inverse_cdf(a), norm.inverse_cdf(1.0 - a)];
    let mut out = Vec::with_capacity(k);
    for i in 0..k {
        let se = fit.vcov[(i, i)].sqrt();
        let xi = Mat::from_fn(n, k - 1, |r, c| x[(r, if c < i { c } else { c + 1 })]);
        let mut pts: Vec<(f64, f64)> = vec![(0.0, fit.coefficients[i])];
        for sgn in [-1.0, 1.0] {
            let mut z: f64 = 0.0;
            let mut step = 0;
            loop {
                step += 1;
                if step >= 10 || z.abs() >= zmax {
                    break;
                }
                let bi = fit.coefficients[i] + sgn * step as f64 * del * se;
                let o: Vec<f64> = (0..n).map(|r| off[r] + x[(r, i)] * bi).collect();
                let dev = if k == 1 {
                    (0..n)
                        .map(|r| w[r] * family.unit_deviance(data.y[r], family.link_inverse(o[r])))
                        .sum()
                } else {
                    irls(family, &xi, data.y, &w, &o)?.deviance
                };
                let mut zz = (dev - fit.deviance) / fit.dispersion;
                if zz > -0.001 {
                    zz = zz.max(0.0);
                } else {
                    return Err(LikelihoodError::NotConverged);
                }
                z = sgn * zz.sqrt();
                pts.push((z, bi));
            }
        }
        pts.sort_by(|p, q| p.0.total_cmp(&q.0));
        let zs: Vec<f64> = pts.iter().map(|p| p.0).collect();
        let bs: Vec<f64> = pts.iter().map(|p| p.1).collect();
        let (sx, sy) = r_spline(&bs, &zs);
        let lo = approx_mean(&sy, &sx, cutoff[0]);
        let hi = approx_mean(&sy, &sx, cutoff[1]);
        out.push((lo, hi));
    }
    Ok(out)
}

/// `regularize.values(x, y, ties = mean)`: sort by `x`, average ties.
fn regularize(x: &[f64], y: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut idx: Vec<usize> = (0..x.len()).collect();
    idx.sort_by(|&a, &b| x[a].total_cmp(&x[b]));
    let (mut ux, mut uy) = (Vec::new(), Vec::new());
    let mut k = 0;
    while k < idx.len() {
        let mut j = k;
        let mut s = 0.0;
        while j < idx.len() && x[idx[j]] == x[idx[k]] {
            s += y[idx[j]];
            j += 1;
        }
        ux.push(x[idx[k]]);
        uy.push(s / (j - k) as f64);
        k = j;
    }
    (ux, uy)
}

/// `approx(x, y, xout)` (linear, ties averaged, NaN outside the range).
fn approx_mean(x: &[f64], y: &[f64], xout: f64) -> f64 {
    let (ux, uy) = regularize(x, y);
    let k = ux.partition_point(|&u| u < xout);
    if k < ux.len() && ux[k] == xout {
        uy[k]
    } else if k == 0 || k == ux.len() {
        f64::NAN
    } else {
        uy[k - 1] + (uy[k] - uy[k - 1]) * (xout - ux[k - 1]) / (ux[k] - ux[k - 1])
    }
}

/// R's `spline(x, y)` (method "fmm", `n = 3·length(x)` points from
/// `min(x)` to `max(x)`); returns the evaluation grid and values.
fn r_spline(x: &[f64], y: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let nout = 3 * x.len();
    let (x, y) = regularize(x, y);
    let n = x.len();
    let (b, c, d) = fmm_spline(&x, &y);
    let (lo, hi) = (x[0], x[n - 1]);
    let by = (hi - lo) / (nout - 1) as f64;
    let grid: Vec<f64> = (0..nout)
        .map(|k| {
            if k == nout - 1 {
                hi
            } else {
                lo + k as f64 * by
            }
        })
        .collect();
    let vals = grid
        .iter()
        .map(|&u| {
            // x[i] <= u <= x[i+1]
            let i = if n == 1 {
                0
            } else {
                x.partition_point(|&v| v <= u).saturating_sub(1).min(n - 1)
            };
            let dx = u - x[i];
            y[i] + dx * (b[i] + dx * (c[i] + dx * d[i]))
        })
        .collect();
    (grid, vals)
}

/// Coefficients of R's FMM cubic spline (`fmm_spline` of `splines.c`).
fn fmm_spline(x: &[f64], y: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = x.len();
    let (mut b, mut c, mut d) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    if n < 2 {
        return (b, c, d);
    }
    if n < 3 {
        let t = (y[1] - y[0]) / (x[1] - x[0]);
        return (vec![t, t], vec![0.0; 2], vec![0.0; 2]);
    }
    // 0-based translation of the 1-based C code.
    let nm1 = n - 1;
    d[0] = x[1] - x[0];
    c[1] = (y[1] - y[0]) / d[0];
    for i in 1..nm1 {
        d[i] = x[i + 1] - x[i];
        b[i] = 2.0 * (d[i - 1] + d[i]);
        c[i + 1] = (y[i + 1] - y[i]) / d[i];
        c[i] = c[i + 1] - c[i];
    }
    b[0] = -d[0];
    b[nm1] = -d[nm1 - 1];
    c[0] = 0.0;
    c[nm1] = 0.0;
    if n > 3 {
        c[0] = c[2] / (x[3] - x[1]) - c[1] / (x[2] - x[0]);
        c[nm1] = c[nm1 - 1] / (x[nm1] - x[nm1 - 2]) - c[nm1 - 2] / (x[nm1 - 1] - x[nm1 - 3]);
        c[0] = c[0] * d[0] * d[0] / (x[3] - x[0]);
        c[nm1] = -c[nm1] * d[nm1 - 1] * d[nm1 - 1] / (x[nm1] - x[nm1 - 3]);
    }
    for i in 1..n {
        let t = d[i - 1] / b[i - 1];
        b[i] -= t * d[i - 1];
        c[i] -= t * c[i - 1];
    }
    c[nm1] /= b[nm1];
    for i in (0..nm1).rev() {
        c[i] = (c[i] - d[i] * c[i + 1]) / b[i];
    }
    b[nm1] = (y[nm1] - y[nm1 - 1]) / d[nm1 - 1] + d[nm1 - 1] * (c[nm1 - 1] + 2.0 * c[nm1]);
    for i in 0..nm1 {
        b[i] = (y[i + 1] - y[i]) / d[i] - d[i] * (c[i + 1] + 2.0 * c[i]);
        d[i] = (c[i + 1] - c[i]) / d[i];
        c[i] *= 3.0;
    }
    c[nm1] *= 3.0;
    d[nm1] = d[nm1 - 1];
    (b, c, d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fmm_spline_reproduces_a_cubic() {
        let x: Vec<f64> = (0..7).map(|i| i as f64 * 0.7 - 1.0).collect();
        let f = |v: f64| 1.0 - 2.0 * v + 0.5 * v * v - 0.3 * v * v * v;
        let y: Vec<f64> = x.iter().map(|&v| f(v)).collect();
        let (g, s) = r_spline(&x, &y);
        for (u, v) in g.iter().zip(&s) {
            assert!((v - f(*u)).abs() < 1e-10, "{u}: {v} vs {}", f(*u));
        }
    }

    #[test]
    fn approx_is_nan_outside_the_range() {
        assert!(approx_mean(&[0.0, 1.0], &[0.0, 2.0], 2.0).is_nan());
        assert_eq!(approx_mean(&[0.0, 1.0, 1.0], &[0.0, 1.0, 3.0], 1.0), 2.0);
        assert_eq!(approx_mean(&[0.0, 1.0], &[0.0, 2.0], 0.25), 0.5);
    }
}
