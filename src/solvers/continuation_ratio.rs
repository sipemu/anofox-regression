//! Continuation-ratio ordinal model (RMS §13.4), fitted by the expanded-data
//! trick of `rms::cr.setup`.
//!
//! For an ordered response with values `y₁ < … < y_k`, the model is
//!
//! ```text
//! logit P(Y = y_j | Y ≥ y_j, x) = θ_j + xᵀγ  (+ xᵀγ_j for cohort-specific terms),
//! ```
//!
//! `j = 1, …, k − 1`. Each observation becomes one binary record per
//! "cohort" `j` it reaches (`Y ≥ y_j`), with response `1` in the cohort
//! where it stops (`Y = y_j`); an ordinary logistic regression on the
//! expanded data with cohort indicators (cohort 1, `"all"`, is the reference)
//! is the maximum likelihood fit. Columns of `x` named in
//! [`ContinuationRatioRegressorBuilder::cohort_specific`] also get a
//! cohort × x interaction, relaxing the equal-slopes assumption.

use crate::solvers::traits::{FittedRegressor, RegressionError, Regressor};
use crate::solvers::{BinomialRegressor, FittedBinomial};
use faer::{Col, Mat};

/// The expanded data of `rms::cr.setup`.
#[derive(Debug, Clone, PartialEq)]
pub struct ContinuationRatioData {
    /// Binary response of each expanded record: 1 if the observation stops
    /// in this cohort.
    pub y: Col<f64>,
    /// Cohort of each record, `0` (`"all"`) to `k − 2`.
    pub cohort: Vec<usize>,
    /// Original observation (row) of each record.
    pub subs: Vec<usize>,
    /// Number of records of each observation.
    pub reps: Vec<usize>,
    /// The distinct response values.
    pub levels: Vec<f64>,
}

/// `rms::cr.setup(y)`: observation `i` with response level index `c`
/// (0-based) contributes records for cohorts `0, …, min(c, k − 2)`, with
/// response `1` in cohort `c` (none for the top level).
pub fn cr_setup(y: &Col<f64>) -> Result<ContinuationRatioData, RegressionError> {
    if y.iter().any(|v| !v.is_finite()) {
        return Err(RegressionError::NumericalError(
            "non-finite response".to_string(),
        ));
    }
    let mut levels: Vec<f64> = y.iter().copied().collect();
    levels.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    levels.dedup();
    let k = levels.len();
    if k < 3 {
        return Err(RegressionError::NumericalError(
            "a continuation-ratio model needs at least three response values".to_string(),
        ));
    }
    let mut ys = Vec::new();
    let mut cohort = Vec::new();
    let mut subs = Vec::new();
    let mut reps = Vec::with_capacity(y.nrows());
    for (i, v) in y.iter().enumerate() {
        let c = levels.partition_point(|l| l < v);
        let last = c.min(k - 2);
        reps.push(last + 1);
        for j in 0..=last {
            ys.push(f64::from(u8::from(c == j)));
            cohort.push(j);
            subs.push(i);
        }
    }
    Ok(ContinuationRatioData {
        y: Col::from_fn(ys.len(), |i| ys[i]),
        cohort,
        subs,
        reps,
        levels,
    })
}

/// Continuation-ratio regressor.
///
/// ```
/// use anofox_regression::solvers::ContinuationRatioRegressor;
/// use faer::{Col, Mat};
///
/// let x = Mat::from_fn(90, 1, |i, _| (i % 9) as f64 / 3.0);
/// let y = Col::from_fn(90, |i| ((i % 9) / 3 + (i % 4 == 0) as usize) as f64);
/// let fit = ContinuationRatioRegressor::builder().build().fit(&x, &y).unwrap();
/// let p = fit.category_probabilities(&x);
/// let total: f64 = (0..p.ncols()).map(|j| p[(0, j)]).sum();
/// assert!((total - 1.0).abs() < 1e-12);
/// ```
#[derive(Debug, Clone, Default)]
pub struct ContinuationRatioRegressor {
    cohort_specific: Vec<usize>,
}

/// Builder for [`ContinuationRatioRegressor`].
#[derive(Debug, Clone, Default)]
pub struct ContinuationRatioRegressorBuilder {
    inner: ContinuationRatioRegressor,
}

impl ContinuationRatioRegressorBuilder {
    /// Columns of `x` whose slope may differ by cohort (a cohort × x
    /// interaction); the others have one slope for all cohorts.
    pub fn cohort_specific(mut self, columns: &[usize]) -> Self {
        self.inner.cohort_specific = columns.to_vec();
        self
    }

    /// Build the regressor.
    pub fn build(self) -> ContinuationRatioRegressor {
        self.inner
    }
}

/// A fitted continuation-ratio model.
#[derive(Debug)]
pub struct FittedContinuationRatio {
    levels: Vec<f64>,
    n_features: usize,
    cohort_specific: Vec<usize>,
    logistic: FittedBinomial,
}

impl ContinuationRatioRegressor {
    /// Builder (equal slopes in every cohort).
    pub fn builder() -> ContinuationRatioRegressorBuilder {
        ContinuationRatioRegressorBuilder::default()
    }

    /// Fit the model to predictors `x` (no intercept column) and the ordered
    /// response `y` (at least three distinct values).
    pub fn fit(
        &self,
        x: &Mat<f64>,
        y: &Col<f64>,
    ) -> Result<FittedContinuationRatio, RegressionError> {
        if y.nrows() != x.nrows() {
            return Err(RegressionError::DimensionMismatch {
                x_rows: x.nrows(),
                y_len: y.nrows(),
            });
        }
        if let Some(&j) = self.cohort_specific.iter().find(|&&j| j >= x.ncols()) {
            return Err(RegressionError::NumericalError(format!(
                "cohort-specific column {j} is out of range"
            )));
        }
        let data = cr_setup(y)?;
        let design = expanded_design(
            x,
            &data.subs,
            &data.cohort,
            data.levels.len() - 1,
            &self.cohort_specific,
        );
        let logistic = BinomialRegressor::logistic()
            .with_intercept(true)
            .compute_inference(true)
            .tolerance(1e-12)
            .build()
            .fit(&design, &data.y)?;
        Ok(FittedContinuationRatio {
            levels: data.levels,
            n_features: x.ncols(),
            cohort_specific: self.cohort_specific.clone(),
            logistic,
        })
    }
}

/// Columns: cohort indicators `2, …, m` (cohort 1 is the intercept), the `x`
/// columns, then for each cohort-specific column its products with the
/// cohort indicators (cohort fastest), as `lrm(y ~ cohort + x + cohort:x_j)`.
fn expanded_design(
    x: &Mat<f64>,
    subs: &[usize],
    cohort: &[usize],
    m: usize,
    cohort_specific: &[usize],
) -> Mat<f64> {
    let p = x.ncols();
    let n_cols = (m - 1) + p + cohort_specific.len() * (m - 1);
    Mat::from_fn(subs.len(), n_cols, |r, c| {
        let (i, h) = (subs[r], cohort[r]);
        if c < m - 1 {
            f64::from(u8::from(h == c + 1))
        } else if c < m - 1 + p {
            x[(i, c - (m - 1))]
        } else {
            let k = c - (m - 1) - p;
            let (j, level) = (cohort_specific[k / (m - 1)], k % (m - 1) + 1);
            if h == level {
                x[(i, j)]
            } else {
                0.0
            }
        }
    })
}

impl FittedContinuationRatio {
    /// The distinct response values.
    pub fn levels(&self) -> &[f64] {
        &self.levels
    }

    /// The logistic fit on the expanded data; its coefficients are the
    /// cohort effects `θ_j − θ_1` (`j = 2, …, k − 1`), the slopes γ and the
    /// cohort-specific slope differences, with intercept `θ_1`.
    pub fn logistic(&self) -> &FittedBinomial {
        &self.logistic
    }

    /// Intercept followed by the coefficients, in the column order of the
    /// expanded design.
    pub fn coefficients(&self) -> Col<f64> {
        let r = self.logistic.result();
        let c = &r.coefficients;
        Col::from_fn(c.nrows() + 1, |j| {
            if j == 0 {
                r.intercept.unwrap_or(0.0)
            } else {
                c[j - 1]
            }
        })
    }

    /// Covariance of [`Self::coefficients`].
    pub fn vcov(&self) -> Mat<f64> {
        self.logistic.vcov().expect("fitted with inference")
    }

    /// Maximised log-likelihood (of the expanded data, which equals that of
    /// the continuation-ratio model).
    pub fn log_likelihood(&self) -> f64 {
        self.logistic.result().log_likelihood
    }

    /// `P(Y = y_j | Y ≥ y_j, x)` for `j = 1, …, k − 1`.
    pub fn conditional_probabilities(&self, x: &Mat<f64>) -> Mat<f64> {
        let m = self.levels.len() - 1;
        let n = x.nrows();
        let subs: Vec<usize> = (0..n).flat_map(|i| std::iter::repeat_n(i, m)).collect();
        let cohort: Vec<usize> = (0..n).flat_map(|_| 0..m).collect();
        let design = expanded_design(x, &subs, &cohort, m, &self.cohort_specific);
        let p = self.logistic.predict_probability(&design);
        Mat::from_fn(n, m, |i, j| p[i * m + j])
    }

    /// `P(Y = y_j | x)` for `j = 1, …, k`:
    /// `h_j ∏_{c<j} (1 − h_c)` with `h_j` the conditional probabilities.
    pub fn category_probabilities(&self, x: &Mat<f64>) -> Mat<f64> {
        let h = self.conditional_probabilities(x);
        let m = h.ncols();
        Mat::from_fn(x.nrows(), m + 1, |i, j| {
            let survive: f64 = (0..j).map(|c| 1.0 - h[(i, c)]).product();
            if j < m {
                h[(i, j)] * survive
            } else {
                survive
            }
        })
    }

    /// Number of predictors.
    pub fn n_features(&self) -> usize {
        self.n_features
    }
}
