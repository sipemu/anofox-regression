//! Prior (case) weights for the IRLS GLMs, as `glm(weights = )` (#308).
//!
//! A row's prior weight `wᵢ` multiplies its IRLS working weight, its unit
//! deviance, its log-likelihood contribution and its Pearson χ² term, and
//! scales its Pearson and deviance residuals by `√wᵢ`. Rows with weight 0
//! do not count towards the residual degrees of freedom (R's `n.ok`).

use crate::core::GlmFamily;
use crate::solvers::traits::RegressionError;
use faer::Col;

/// Optional prior weights (`None`: all 1).
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Default)]
pub(crate) struct PriorWeights(
    #[cfg_attr(feature = "serde", serde(with = "crate::serde_impl::opt_col"))] Option<Col<f64>>,
);

impl PriorWeights {
    pub(crate) fn new(weights: Option<Col<f64>>) -> Self {
        Self(weights)
    }

    pub(crate) fn is_some(&self) -> bool {
        self.0.is_some()
    }

    pub(crate) fn col(&self) -> Option<&Col<f64>> {
        self.0.as_ref()
    }

    /// Weight of row `i`.
    pub(crate) fn get(&self, i: usize) -> f64 {
        self.0.as_ref().map_or(1.0, |w| w[i])
    }

    pub(crate) fn to_vec(&self) -> Option<Vec<f64>> {
        self.0.as_ref().map(|w| w.iter().copied().collect())
    }

    /// Length `n`, finite and non-negative.
    pub(crate) fn validate(&self, n: usize) -> Result<(), RegressionError> {
        let Some(w) = &self.0 else {
            return Ok(());
        };
        if w.nrows() != n {
            return Err(RegressionError::DimensionMismatch {
                x_rows: n,
                y_len: w.nrows(),
            });
        }
        if w.iter().any(|v| !(v.is_finite() && *v >= 0.0)) {
            return Err(RegressionError::InvalidWeights);
        }
        Ok(())
    }

    /// Rows with a positive weight (R's `n.ok`).
    pub(crate) fn n_ok(&self, n: usize) -> usize {
        self.0
            .as_ref()
            .map_or(n, |w| w.iter().filter(|v| **v > 0.0).count())
    }

    /// `Σ wᵢ d(yᵢ, μᵢ)`.
    pub(crate) fn deviance<F: GlmFamily + ?Sized>(&self, family: &F, y: &[f64], mu: &[f64]) -> f64 {
        match &self.0 {
            None => family.deviance(y, mu),
            Some(w) => (0..y.len())
                .filter(|&i| w[i] > 0.0)
                .map(|i| w[i] * family.unit_deviance(y[i], mu[i]))
                .sum(),
        }
    }

    /// `Σ wᵢ (yᵢ − μᵢ)² / V(μᵢ)`.
    pub(crate) fn pearson_chi2<F: GlmFamily + ?Sized>(
        &self,
        family: &F,
        y: &[f64],
        mu: &[f64],
    ) -> f64 {
        (0..y.len())
            .map(|i| self.get(i) * (y[i] - mu[i]).powi(2) / family.variance(mu[i]))
            .sum()
    }

    /// `Σ wᵢ ℓᵢ` for per-row log-likelihood contributions `ℓᵢ`.
    pub(crate) fn weighted_sum(&self, terms: impl Iterator<Item = f64>) -> f64 {
        terms
            .enumerate()
            .map(|(i, t)| {
                let w = self.get(i);
                if w == 0.0 {
                    0.0
                } else {
                    w * t
                }
            })
            .sum()
    }

    /// `rᵢ √wᵢ` (Pearson and deviance residuals with prior weights).
    pub(crate) fn scale_residuals(&self, r: Col<f64>) -> Col<f64> {
        match &self.0 {
            None => r,
            Some(w) => Col::from_fn(r.nrows(), |i| r[i] * w[i].sqrt()),
        }
    }
}
