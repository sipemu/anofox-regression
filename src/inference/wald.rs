//! Wald inference (statistic, p-value, confidence interval) from estimates and
//! standard errors, shared by the GLM solvers so the intercept and the slopes
//! are always reported together.

use crate::core::RegressionResult;
use faer::Col;
use statrs::distribution::{ContinuousCDF, Normal, StudentsT};

/// Reference distribution of the Wald statistic.
#[derive(Debug, Clone, Copy)]
pub(crate) enum WaldReference {
    /// Standard normal (`z` tests; fixed dispersion, as R's `summary.glm` for
    /// binomial / Poisson and `MASS::glm.nb`).
    Normal,
    /// Student t with the given degrees of freedom (estimated dispersion, as
    /// R's `summary.glm` for Gaussian / Gamma / Tweedie families).
    StudentT(f64),
}

impl WaldReference {
    fn two_sided_p(self, stat: f64) -> f64 {
        if !stat.is_finite() {
            return f64::NAN;
        }
        let tail = match self {
            WaldReference::Normal => Normal::new(0.0, 1.0)
                .map(|d| d.sf(stat.abs()))
                .unwrap_or(f64::NAN),
            WaldReference::StudentT(df) if df > 0.0 => StudentsT::new(0.0, 1.0, df)
                .map(|d| d.sf(stat.abs()))
                .unwrap_or(f64::NAN),
            WaldReference::StudentT(_) => f64::NAN,
        };
        2.0 * tail
    }

    fn quantile(self, level: f64) -> f64 {
        let p = 1.0 - (1.0 - level) / 2.0;
        match self {
            WaldReference::Normal => Normal::new(0.0, 1.0)
                .map(|d| d.inverse_cdf(p))
                .unwrap_or(f64::NAN),
            WaldReference::StudentT(df) if df > 0.0 => StudentsT::new(0.0, 1.0, df)
                .map(|d| d.inverse_cdf(p))
                .unwrap_or(f64::NAN),
            WaldReference::StudentT(_) => f64::NAN,
        }
    }
}

/// Fill statistics, p-values and confidence intervals for the slopes and the
/// intercept of `result`.
///
/// `beta` and `se` hold every fitted parameter in design order, the intercept
/// first when `with_intercept`. The interval is the Wald interval
/// `β̂ ± q · SE` (R's `confint.default`), at `result.confidence_level`.
pub(crate) fn fill_wald_inference(
    result: &mut RegressionResult,
    beta: &Col<f64>,
    se: &Col<f64>,
    with_intercept: bool,
    reference: WaldReference,
) {
    let n_params = beta.nrows();
    let off = usize::from(with_intercept);
    let n_features = n_params - off;
    let q = reference.quantile(result.confidence_level);

    let stat = Col::from_fn(n_params, |j| beta[j] / se[j]);
    let p = Col::from_fn(n_params, |j| reference.two_sided_p(stat[j]));
    let lo = Col::from_fn(n_params, |j| beta[j] - q * se[j]);
    let hi = Col::from_fn(n_params, |j| beta[j] + q * se[j]);

    result.std_errors = Some(Col::from_fn(n_features, |j| se[j + off]));
    result.t_statistics = Some(Col::from_fn(n_features, |j| stat[j + off]));
    result.p_values = Some(Col::from_fn(n_features, |j| p[j + off]));
    result.conf_interval_lower = Some(Col::from_fn(n_features, |j| lo[j + off]));
    result.conf_interval_upper = Some(Col::from_fn(n_features, |j| hi[j + off]));

    if with_intercept {
        result.intercept_std_error = Some(se[0]);
        result.intercept_t_statistic = Some(stat[0]);
        result.intercept_p_value = Some(p[0]);
        result.intercept_conf_interval = Some((lo[0], hi[0]));
    }
}
