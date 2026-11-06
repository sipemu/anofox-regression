//! Cluster-robust covariance of a [`GlmResult`] (the `fit_*` entry points),
//! with the penalized bread for fits with priors or a ridge penalty.

use super::families::GlmResult;
use crate::inference::{ClusterError, ClusterOptions, ClusterType};
use faer::linalg::solvers::Solve;
use faer::Mat;

impl GlmResult {
    /// Cluster-robust (sandwich) covariance of the fitted parameters, as
    /// `sandwich::vcovCL(glm(...), cluster, type)` for an unpenalized fit.
    ///
    /// `V = A (Σ_g s_g s_gᵀ) A` with `sᵢ = (yᵢ − μᵢ) / (V(μᵢ) g′(μᵢ)) xᵢ` the
    /// likelihood scores at the fitted means and `A` the inverse of the
    /// penalized information `XᵀWX + P` at the mode (`P = 0` without a
    /// penalty, so `A = (XᵀWX)⁻¹`). With Gaussian priors or a ridge
    /// `lambda` this is the sandwich of the penalized estimating equation
    /// `Σ sᵢ − P(β − m) = 0` (the prior has no sampling variance), whatever
    /// `vcov_type` the fit reports. The dispersion cancels; a Negative
    /// Binomial `theta` is held at its estimate, as `vcovCL(glm.nb)` does.
    ///
    /// `y` and `x` (column-major, including the offset column if the fit
    /// had one) are the data passed to the `fit_*` function; `clusters` holds
    /// one code vector per clustering dimension with one entry per input row.
    /// Rows the fit dropped ([`Self::valid_rows`]) are dropped here too.
    /// [`ClusterOptions::hc0`] is `vcovCL`'s default for GLMs; HC2 / HC3 are
    /// not available.
    ///
    /// The result is in the *fitted* order of
    /// [`GlmInferenceResult::vcov`](super::GlmInferenceResult::vcov) (see
    /// [`matrix_parameters`](super::GlmInferenceResult::matrix_parameters)):
    /// the intercept first, then the retained features. Needs a fit with
    /// `compute_inference`; a coefficient held at an L1 (Laplace) prior's
    /// kink has no sandwich and is an error.
    pub fn cluster_vcov(
        &self,
        y: &[f64],
        x: &[Vec<f64>],
        clusters: &[&[usize]],
        options: &ClusterOptions,
    ) -> Result<Mat<f64>, ClusterError> {
        if matches!(options.kind, ClusterType::HC2 | ClusterType::HC3) {
            return Err(ClusterError::Unsupported(
                "HC2 / HC3 are available for linear models only",
            ));
        }
        let inf = self.inference.as_ref().ok_or(ClusterError::Unsupported(
            "the fit has no inference; fit with compute_inference = true",
        ))?;
        let info = inf.information.as_ref().ok_or(ClusterError::Unsupported(
            "the fit has no information matrix",
        ))?;
        self.check_cluster_inputs(y, x, clusters)?;
        if self.at_l1_kink() {
            return Err(ClusterError::Unsupported(
                "a coefficient sits at the kink of its L1 prior; it has no sandwich",
            ));
        }
        let design = self.fitted_design(x, &inf.matrix_parameters);
        let scores = self.likelihood_scores(&design);
        let bread = self.penalized_bread(info)?;
        let kept: Vec<Vec<usize>> = clusters
            .iter()
            .map(|c| self.valid_rows.iter().map(|&i| c[i]).collect())
            .collect();
        let refs: Vec<&[usize]> = kept.iter().map(Vec::as_slice).collect();
        let yv: Vec<f64> = self.valid_rows.iter().map(|&i| y[i]).collect();
        let scores = Mat::from_fn(scores.nrows(), scores.ncols(), |i, j| {
            (yv[i] - self.fitted_values[i]) * scores[(i, j)]
        });
        crate::inference::cluster_sandwich(&bread, &scores, &refs, options)
    }

    fn check_cluster_inputs(
        &self,
        y: &[f64],
        x: &[Vec<f64>],
        clusters: &[&[usize]],
    ) -> Result<(), ClusterError> {
        let n = y.len();
        if x.iter().any(|c| c.len() != n) {
            return Err(ClusterError::Dimension(format!(
                "y has {n} entries, the x columns {:?}",
                x.iter().map(Vec::len).collect::<Vec<_>>()
            )));
        }
        let expected = self.core.n_features + usize::from(self.offset_column.is_some());
        if x.len() != expected {
            return Err(ClusterError::Dimension(format!(
                "x has {} columns, the fit was given {expected}",
                x.len()
            )));
        }
        if self.valid_rows.iter().any(|&i| i >= n)
            || self.fitted_values.len() != self.valid_rows.len()
        {
            return Err(ClusterError::Dimension(
                "y is not the response the model was fitted to".into(),
            ));
        }
        if clusters.is_empty() {
            return Err(ClusterError::NoClusters);
        }
        if let Some(c) = clusters.iter().find(|c| c.len() != n) {
            return Err(ClusterError::Dimension(format!(
                "cluster has {} entries, expected {n}",
                c.len()
            )));
        }
        Ok(())
    }

    /// A fitted parameter without a standard error: an L1 coefficient at its
    /// prior location.
    fn at_l1_kink(&self) -> bool {
        let Some(inf) = &self.inference else {
            return false;
        };
        inf.matrix_parameters.iter().any(|p| match p {
            None => inf.intercept_std_error.is_some_and(f64::is_nan),
            Some(j) => inf.std_errors.get(*j).is_some_and(|s| s.is_nan()),
        })
    }

    /// The design over the used rows, in fitted order.
    fn fitted_design(&self, x: &[Vec<f64>], params: &[Option<usize>]) -> Mat<f64> {
        let offset = self.offset_column.map(|c| c - 1);
        let features: Vec<usize> = (0..x.len()).filter(|&c| Some(c) != offset).collect();
        Mat::from_fn(self.valid_rows.len(), params.len(), |r, c| {
            match params[c] {
                None => 1.0,
                Some(j) => x[features[j]][self.valid_rows[r]],
            }
        })
    }

    /// `xᵢ / (V(μᵢ) g′(μᵢ))`: the scores without the residual factor.
    fn likelihood_scores(&self, design: &Mat<f64>) -> Mat<f64> {
        let family = self.family.glm_family(self.core.dispersion);
        let factor: Vec<f64> = self
            .fitted_values
            .iter()
            .map(|&mu| 1.0 / (family.variance(mu) * family.link_derivative(mu)))
            .collect();
        Mat::from_fn(design.nrows(), design.ncols(), |i, j| {
            factor[i] * design[(i, j)]
        })
    }

    /// `(XᵀWX + P)⁻¹`: the reported information is that matrix over the
    /// covariance scale (the dispersion for Pearson-scaled families).
    fn penalized_bread(&self, info: &Mat<f64>) -> Result<Mat<f64>, ClusterError> {
        let scale = self
            .dispersion_rule
            .covariance_scale(self.core.dispersion.unwrap_or(1.0));
        let k = info.nrows();
        let scaled = Mat::from_fn(k, k, |r, c| info[(r, c)] * scale);
        let llt = scaled
            .llt(faer::Side::Lower)
            .map_err(|_| ClusterError::Singular)?;
        Ok(llt.solve(Mat::<f64>::identity(k, k)))
    }
}
