//! Score contributions and cluster-robust covariance of an AFT fit, shared
//! by [`FittedAft`](super::FittedAft) and the result of the column-major
//! entry point [`fit_aft`](super::fit_aft).

use super::distribution::AftDistribution;
use super::model::AftResult;
use crate::inference::{ClusterError, ClusterOptions};
use faer::linalg::solvers::Solve;
use faer::Mat;

/// Per-observation scores `∂ log Lᵢ / ∂(intercept, β, log σ)` at the
/// estimate. `row(i)` gives `(time, event, x row)` of observation `i`.
pub(crate) fn score_matrix(
    dist: AftDistribution,
    intercept: Option<f64>,
    coefficients: &[f64],
    sigma: f64,
    n: usize,
    row: impl Fn(usize) -> (f64, bool, Vec<f64>),
) -> Mat<f64> {
    let off = usize::from(intercept.is_some());
    let p = coefficients.len();
    let fit_scale = !dist.scale_is_fixed();
    let k = off + p + usize::from(fit_scale);
    let mut s = Mat::<f64>::zeros(n, k);
    for i in 0..n {
        let (t, event, x) = row(i);
        let eta = intercept.unwrap_or(0.0)
            + coefficients
                .iter()
                .zip(&x)
                .filter(|(b, _)| !b.is_nan())
                .map(|(b, v)| b * v)
                .sum::<f64>();
        let z = (t.ln() - eta) / sigma;
        let u = if event {
            dist.d_log_density(z)
        } else {
            dist.d_log_survival(z)
        };
        if off == 1 {
            s[(i, 0)] = -u / sigma;
        }
        for j in 0..p {
            s[(i, off + j)] = -u * x[j] / sigma;
        }
        if fit_scale {
            s[(i, k - 1)] = -u * z - f64::from(u8::from(event));
        }
    }
    s
}

/// The bread of the sandwich of a (possibly penalized) AFT fit: the inverse
/// of the penalized observed information, whatever covariance the fit
/// reports (`vcov_type` only changes the reported covariance).
pub(crate) fn penalized_bread(result: &AftResult) -> Result<Mat<f64>, ClusterError> {
    let info = result
        .inference
        .as_ref()
        .and_then(|i| i.information.as_ref())
        .ok_or(ClusterError::Unsupported(
            "the fit has no information matrix; fit with compute_inference(true)",
        ))?;
    let k = info.nrows();
    let llt = info
        .llt(faer::Side::Lower)
        .map_err(|_| ClusterError::Singular)?;
    Ok(llt.solve(Mat::<f64>::identity(k, k)))
}

impl AftResult {
    /// Rows [`fit_aft`](super::fit_aft) used: a finite time and event and
    /// finite features.
    fn used_rows(time: &[f64], x: &[Vec<f64>], event: &[f64]) -> Vec<usize> {
        (0..time.len())
            .filter(|&i| {
                time[i].is_finite() && event[i].is_finite() && x.iter().all(|c| c[i].is_finite())
            })
            .collect()
    }

    fn check_data(&self, time: &[f64], x: &[Vec<f64>], event: &[f64]) -> Result<(), ClusterError> {
        let n = time.len();
        if event.len() != n || x.iter().any(|c| c.len() != n) {
            return Err(ClusterError::Dimension(format!(
                "time has {n} entries, event {} and the x columns {:?}",
                event.len(),
                x.iter().map(Vec::len).collect::<Vec<_>>()
            )));
        }
        if x.len() != self.core.coefficients.len() {
            return Err(ClusterError::Dimension(format!(
                "x has {} columns, the fit {} features",
                x.len(),
                self.core.coefficients.len()
            )));
        }
        Ok(())
    }

    /// Per-observation score contributions at the estimate, one row per row
    /// that [`fit_aft`](super::fit_aft) used (rows with a non-finite time,
    /// event or feature are skipped, as in the fit), in the parameter order
    /// of [`AftInference::vcov`](super::AftInference::vcov): intercept (if
    /// fitted), the coefficients, then `log σ` when the scale is estimated.
    /// `time`, `x` (column-major) and `event` are the data of the fit. The
    /// prior is not part of the scores.
    pub fn score_contributions(
        &self,
        time: &[f64],
        x: &[Vec<f64>],
        event: &[f64],
    ) -> Result<Mat<f64>, ClusterError> {
        self.check_data(time, x, event)?;
        let dist = self
            .distribution()
            .ok_or(ClusterError::Unsupported("unknown AFT distribution"))?;
        let rows = Self::used_rows(time, x, event);
        Ok(score_matrix(
            dist,
            self.core.intercept,
            &self.core.coefficients,
            self.core.scale,
            rows.len(),
            |r| {
                let i = rows[r];
                (time[i], event[i] == 1.0, x.iter().map(|c| c[i]).collect())
            },
        ))
    }

    /// Cluster-robust covariance of all parameters (the order of
    /// [`Self::score_contributions`]): `A (Σ_g s_g s_gᵀ) A` with `A` the
    /// inverse of the penalized observed information
    /// ([`AftInference::information`](super::AftInference::information)) and
    /// `s_g` the summed likelihood scores of cluster `g`.
    ///
    /// Without priors `A` is the usual inverse information, and
    /// [`ClusterOptions::robcov`] gives `survreg(..., cluster = id)` /
    /// `rms::robcov(psm(...), cluster)`. With Gaussian priors it is the
    /// sandwich of the penalized estimating equation `Σ sᵢ − P(β − m) = 0`,
    /// whatever `vcov_type` the fit reports.
    ///
    /// `time`, `x` (column-major) and `event` are the data of the fit and
    /// `clusters` holds one code vector per clustering dimension, one entry per
    /// input row; rows the fit skipped are skipped here too. Needs a fit with
    /// `compute_inference`.
    pub fn cluster_vcov(
        &self,
        time: &[f64],
        x: &[Vec<f64>],
        event: &[f64],
        clusters: &[&[usize]],
        options: &ClusterOptions,
    ) -> Result<Mat<f64>, ClusterError> {
        let scores = self.score_contributions(time, x, event)?;
        let n = time.len();
        if let Some(c) = clusters.iter().find(|c| c.len() != n) {
            return Err(ClusterError::Dimension(format!(
                "cluster has {} entries, expected {n}",
                c.len()
            )));
        }
        let rows = Self::used_rows(time, x, event);
        let kept: Vec<Vec<usize>> = clusters
            .iter()
            .map(|c| rows.iter().map(|&i| c[i]).collect())
            .collect();
        let refs: Vec<&[usize]> = kept.iter().map(Vec::as_slice).collect();
        let bread = penalized_bread(self)?;
        crate::inference::cluster_sandwich(&bread, &scores, &refs, options)
    }
}

impl AftResult {
    /// Rebuild an AFT result from stored parts — the distribution, the
    /// intercept (`None` without one), the feature coefficients, the scale
    /// `σ` and optionally the covariance of the fitted parameters (intercept,
    /// coefficients, then `log σ` when the scale is estimated: the layout of
    /// [`AftInference::vcov`](super::AftInference::vcov)) — so a stored model
    /// predicts again: [`Self::survival_curve`], [`Self::quantiles`] and the
    /// predictions of [`FittedAft::from_result`](super::FittedAft::from_result)
    /// equal the fit's.
    ///
    /// With `vcov` the inference (standard errors, Wald z / p, 95 % Wald
    /// intervals, the information) is rebuilt from it, so
    /// [`Self::cluster_vcov`] works on the data of the fit (for a penalized
    /// fit store the Laplace covariance, the inverse of the penalized
    /// information). The fit statistics (log-likelihoods, AIC, counts) are
    /// not part of the model and are `NaN` / 0. `scale` must be positive and
    /// finite (1 for the exponential distribution), the coefficients finite
    /// (`NaN` marks a dropped column, which predictions skip).
    pub fn from_parts(
        distribution: AftDistribution,
        intercept: Option<f64>,
        coefficients: Vec<f64>,
        scale: f64,
        vcov: Option<&Mat<f64>>,
    ) -> Result<Self, crate::solvers::RegressionError> {
        use crate::solvers::RegressionError;
        let bad =
            |name: &'static str, reason: &str| RegressionError::invalid_parameter(name, reason);
        if intercept.is_some_and(|b| !b.is_finite()) || coefficients.iter().any(|b| b.is_infinite())
        {
            return Err(bad(
                "coefficients",
                "must be finite (NaN only for a dropped column)",
            ));
        }
        if !(scale.is_finite() && scale > 0.0) || (distribution.scale_is_fixed() && scale != 1.0) {
            return Err(bad(
                "scale",
                "must be positive and finite (1 for the exponential)",
            ));
        }
        let core = super::AftFitResult {
            n_features: coefficients.len(),
            coefficients,
            intercept,
            scale,
            log_likelihood: f64::NAN,
            null_log_likelihood: f64::NAN,
            aic: f64::NAN,
            bic: f64::NAN,
            n_observations: 0,
            n_events: 0,
            n_censored: 0,
            iterations: 0,
            converged: true,
        };
        let inference = vcov
            .map(|v| inference_from_vcov(&core, v, distribution))
            .transpose()?;
        Ok(Self {
            core,
            inference,
            model_info: super::model::aft_model_info(distribution),
        })
    }
}

/// Wald inference of stored parts from their covariance.
fn inference_from_vcov(
    core: &super::AftFitResult,
    vcov: &Mat<f64>,
    distribution: AftDistribution,
) -> Result<super::AftInference, crate::solvers::RegressionError> {
    use statrs::distribution::{ContinuousCDF, Normal};
    let off = usize::from(core.intercept.is_some());
    let p = core.coefficients.len();
    let fit_scale = !distribution.scale_is_fixed();
    let k = off + p + usize::from(fit_scale);
    let bad = || {
        crate::solvers::RegressionError::invalid_parameter(
            "vcov",
            format!("must be a finite, positive definite {k} x {k} matrix"),
        )
    };
    if vcov.nrows() != k
        || vcov.ncols() != k
        || (0..k).any(|r| (0..k).any(|c| !vcov[(r, c)].is_finite()))
    {
        return Err(bad());
    }
    let sym = Mat::from_fn(k, k, |r, c| 0.5 * (vcov[(r, c)] + vcov[(c, r)]));
    let llt = sym.llt(faer::Side::Lower).map_err(|_| bad())?;
    let information = llt.solve(Mat::<f64>::identity(k, k));
    let mut params: Vec<f64> = core.intercept.into_iter().collect();
    params.extend(&core.coefficients);
    let normal = Normal::new(0.0, 1.0).expect("standard normal");
    let zc = normal.inverse_cdf(0.975);
    let se: Vec<f64> = (0..off + p).map(|j| sym[(j, j)].sqrt()).collect();
    let z: Vec<f64> = (0..off + p).map(|j| params[j] / se[j]).collect();
    let pv: Vec<f64> = z.iter().map(|z| 2.0 * normal.sf(z.abs())).collect();
    let lo: Vec<f64> = (0..off + p).map(|j| params[j] - zc * se[j]).collect();
    let hi: Vec<f64> = (0..off + p).map(|j| params[j] + zc * se[j]).collect();
    let first = |v: &[f64]| (off == 1).then(|| v[0]);
    Ok(super::AftInference {
        std_errors: se[off..].to_vec(),
        z_values: z[off..].to_vec(),
        p_values: pv[off..].to_vec(),
        ci_lower: lo[off..].to_vec(),
        ci_upper: hi[off..].to_vec(),
        confidence_level: 0.95,
        intercept_std_error: first(&se),
        intercept_z_value: first(&z),
        intercept_p_value: first(&pv),
        intercept_ci_lower: first(&lo),
        intercept_ci_upper: first(&hi),
        log_scale_std_error: fit_scale.then(|| sym[(k - 1, k - 1)].sqrt()),
        vcov: Some(sym),
        information: Some(information),
    })
}
