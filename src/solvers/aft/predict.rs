//! Model-based survival curves and quantiles of an AFT fit (#80).
//!
//! For covariates `x`, `log T = η + σ W` with `η = intercept + x'β`, so
//! `S(t | x) = S_W((log t - η) / σ)` (R `psurvreg(t, η, σ, dist, lower.tail = FALSE)`)
//! and the `p`-quantile is `exp(η + σ q_W(p))` (R
//! `predict(survreg, type = "quantile", p = p)` / `qsurvreg`).

use super::distribution::AftDistribution;
use super::model::AftResult;
use super::regressor::FittedAft;

fn survival_at(dist: AftDistribution, t: f64, eta: f64, sigma: f64) -> f64 {
    if t.is_nan() || !eta.is_finite() || !(sigma.is_finite() && sigma > 0.0) {
        return f64::NAN;
    }
    if t <= 0.0 {
        return 1.0;
    }
    if t == f64::INFINITY {
        return 0.0;
    }
    dist.log_survival((t.ln() - eta) / sigma).exp()
}

fn linear_predictor(intercept: Option<f64>, coefs: &[f64], x_row: &[f64]) -> f64 {
    if x_row.len() != coefs.len() {
        return f64::NAN;
    }
    intercept.unwrap_or(0.0)
        + coefs
            .iter()
            .zip(x_row)
            .filter(|(b, _)| !b.is_nan())
            .map(|(b, x)| b * x)
            .sum::<f64>()
}

impl AftResult {
    /// The survival-time distribution of the fit (from [`Self::model_info`](
    /// crate::core::HasModelInfo::model_info)).
    pub fn distribution(&self) -> Option<AftDistribution> {
        self.model_info.family.and_then(AftDistribution::from_name)
    }

    /// Linear predictor `η = intercept + x'β` (the location of `log T`) for one
    /// covariate row (one value per feature; `NaN` when the length is wrong).
    pub fn linear_predictor(&self, x_row: &[f64]) -> f64 {
        linear_predictor(self.core.intercept, &self.core.coefficients, x_row)
    }

    /// Predicted survival `S(t | x)` at each of `times` for one covariate row, as
    /// R's `psurvreg(times, η, σ, dist, lower.tail = FALSE)`. `S = 1` for
    /// `t <= 0`; `NaN` for a `NaN` time or a row of the wrong length.
    pub fn survival_curve(&self, x_row: &[f64], times: &[f64]) -> Vec<f64> {
        let eta = self.linear_predictor(x_row);
        match self.distribution() {
            Some(d) => times
                .iter()
                .map(|&t| survival_at(d, t, eta, self.core.scale))
                .collect(),
            None => vec![f64::NAN; times.len()],
        }
    }

    /// Predicted survival-time quantiles at each of `probs` for one covariate
    /// row, as R's `predict(survreg, type = "quantile", p = probs)`: 0 at
    /// `p = 0`, `+Inf` at `p = 1`, `NaN` outside `[0, 1]`.
    pub fn quantiles(&self, x_row: &[f64], probs: &[f64]) -> Vec<f64> {
        let eta = self.linear_predictor(x_row);
        match self.distribution() {
            Some(d) => probs
                .iter()
                .map(|&p| d.quantile_time(p, eta, self.core.scale))
                .collect(),
            None => vec![f64::NAN; probs.len()],
        }
    }
}

impl FittedAft {
    /// Predicted survival curve `S(t | x)` at `times` for one covariate row; see
    /// [`AftResult::survival_curve`].
    pub fn survival_curve(&self, x_row: &[f64], times: &[f64]) -> Vec<f64> {
        self.result().survival_curve(x_row, times)
    }

    /// Predicted survival-time quantiles at `probs` for one covariate row; see
    /// [`AftResult::quantiles`].
    pub fn quantiles(&self, x_row: &[f64], probs: &[f64]) -> Vec<f64> {
        self.result().quantiles(x_row, probs)
    }
}
