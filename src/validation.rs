//! Resampling model validation and calibration (RMS §5.3, §10.11), as
//! `rms::validate` and `rms::calibrate`.
//!
//! [`predab_resample`] is the generic engine (`rms:::predab.resample`): it
//! refits the **whole modelling process** – any closure, so variable
//! selection or tuning can be replayed – on every training sample and
//! evaluates a vector of indexes on the training rows ("apparent") and on
//! the test rows. With
//!
//! * [`Resamples::bootstrap`] – Efron–Gong optimism bootstrap: training =
//!   bootstrap sample, test = all rows; optimism = mean(train − test);
//! * [`Resamples::crossvalidation`] – B-fold; test = held-out group;
//! * [`Resamples::point632`] – Efron's .632: test = rows left out of each
//!   bootstrap sample, weighted; optimism = `0.632 (orig − Σ wᵢ testᵢ)`;
//!
//! the bias-corrected index is `original − optimism`. Resamples are drawn
//! with a seed or given explicitly (e.g. to replay R's).
//!
//! [`validate_lrm`] / [`validate_ols`] provide `validate.lrm` /
//! `validate.ols` for binary logistic and linear models (Dxy, R², the
//! calibration intercept and slope, Emax, D, U, Q, Brier, g, gp / R², MSE,
//! g, intercept, slope), and [`calibrate_lrm`] / [`calibrate_ols`] the
//! bootstrap overfitting-corrected calibration curve with its mean and 0.9
//! quantile absolute error. [`bootcov`] gives the bootstrap covariance of
//! any fit (ordinary or cluster bootstrap) with percentile intervals for
//! coefficients and contrasts, and [`rank_intervals`] bootstrap intervals
//! for the ranks of predictor importance.

use crate::diagnostics::accuracy::{
    approx_ties_mean, binomial_deviance, gini_md, logistic_recalibration, lrm_stats, plogis,
    quantile7,
};
use crate::solvers::lowess::{lowess, LowessOptions};
use crate::solvers::{FittedRegressor, LogisticRegression, OlsRegressor, Regressor};
use faer::{Col, Mat};
use thiserror::Error;

/// Errors of the resampling validation.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum ValidationError {
    /// Invalid resampling setup.
    #[error("invalid resamples: {0}")]
    Resamples(String),
    /// Inconsistent data.
    #[error("invalid data: {0}")]
    Data(String),
    /// The model could not be fitted on the full data.
    #[error("the model could not be fitted on the full data")]
    OriginalFit,
}

/// Resampling scheme.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResampleMethod {
    /// Optimism bootstrap.
    Bootstrap,
    /// B-fold cross-validation.
    Crossvalidation,
    /// Efron's .632 bootstrap.
    Point632,
}

/// Training / test splits of a resampling validation.
#[derive(Debug, Clone, PartialEq)]
pub struct Resamples {
    /// The scheme.
    pub method: ResampleMethod,
    /// Number of rows.
    pub n: usize,
    /// `(training rows, test rows)`; training rows may repeat.
    pub splits: Vec<(Vec<usize>, Vec<usize>)>,
    /// Weight of each split's test index (.632; 1 otherwise).
    pub weights: Vec<f64>,
}

/// SplitMix64 generator for seeded resamples.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

impl Resamples {
    /// Optimism bootstrap from explicit samples (each of length `n`, rows
    /// `0..n`); the test set is all rows.
    pub fn bootstrap(n: usize, samples: Vec<Vec<usize>>) -> Result<Self, ValidationError> {
        check_samples(n, &samples)?;
        let all: Vec<usize> = (0..n).collect();
        Ok(Self {
            method: ResampleMethod::Bootstrap,
            n,
            weights: vec![1.0; samples.len()],
            splits: samples.into_iter().map(|s| (s, all.clone())).collect(),
        })
    }

    /// `b` seeded bootstrap samples.
    pub fn bootstrap_seeded(n: usize, b: usize, seed: u64) -> Result<Self, ValidationError> {
        Self::bootstrap(n, draw(n, b, seed))
    }

    /// B-fold cross-validation over the permutation `perm` of `0..n`, with
    /// rms's group boundaries (`1 + round((i−1)·n/B)` to
    /// `round(start + n/B − 1)`, R's round-half-even). As in rms, when
    /// `n/B` is not an integer neighbouring groups can share or skip a row.
    pub fn crossvalidation(perm: &[usize], b: usize) -> Result<Self, ValidationError> {
        let n = perm.len();
        let mut seen = vec![false; n];
        for &i in perm {
            if i >= n || std::mem::replace(&mut seen[i], true) {
                return Err(ValidationError::Resamples(
                    "perm must be a permutation of 0..n".into(),
                ));
            }
        }
        let per = n as f64 / b as f64;
        if b < 2 || per < 2.0 {
            return Err(ValidationError::Resamples("need 2 <= B <= n/2".into()));
        }
        let splits = (1..=b)
            .map(|i| {
                let is = 1 + ((i - 1) as f64 * per).round_ties_even() as usize;
                let ie = n.min((is as f64 + per - 1.0).round_ties_even() as usize);
                let test: Vec<usize> = perm[is - 1..ie].to_vec();
                let mut held = vec![false; n];
                for &t in &test {
                    held[t] = true;
                }
                ((0..n).filter(|&k| !held[k]).collect(), test)
            })
            .collect();
        Ok(Self {
            method: ResampleMethod::Crossvalidation,
            n,
            weights: vec![1.0; b],
            splits,
        })
    }

    /// Seeded B-fold cross-validation.
    pub fn crossvalidation_seeded(n: usize, b: usize, seed: u64) -> Result<Self, ValidationError> {
        let mut rng = Rng(seed);
        let mut perm: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            perm.swap(i, rng.below(i + 1));
        }
        Self::crossvalidation(&perm, b)
    }

    /// Efron's .632 bootstrap from explicit samples. Every row must be left
    /// out of at least one sample.
    pub fn point632(n: usize, samples: Vec<Vec<usize>>) -> Result<Self, ValidationError> {
        check_samples(n, &samples)?;
        let b = samples.len();
        let mut omitted = vec![vec![true; b]; n];
        for (i, s) in samples.iter().enumerate() {
            for &r in s {
                omitted[r][i] = false;
            }
        }
        let nomit: Vec<f64> = omitted
            .iter()
            .map(|o| o.iter().filter(|&&v| v).count() as f64)
            .collect();
        if nomit.contains(&0.0) {
            return Err(ValidationError::Resamples(
                "every observation must be omitted from at least one sample; use more samples"
                    .into(),
            ));
        }
        let weights = (0..b)
            .map(|i| {
                (0..n)
                    .filter(|&k| omitted[k][i])
                    .map(|k| 1.0 / nomit[k])
                    .sum::<f64>()
                    / n as f64
            })
            .collect();
        let splits = samples
            .into_iter()
            .enumerate()
            .map(|(i, s)| {
                let test = (0..n).filter(|&k| omitted[k][i]).collect();
                (s, test)
            })
            .collect();
        Ok(Self {
            method: ResampleMethod::Point632,
            n,
            splits,
            weights,
        })
    }

    /// `b` seeded .632 samples.
    pub fn point632_seeded(n: usize, b: usize, seed: u64) -> Result<Self, ValidationError> {
        Self::point632(n, draw(n, b, seed))
    }
}

fn draw(n: usize, b: usize, seed: u64) -> Vec<Vec<usize>> {
    let mut rng = Rng(seed);
    (0..b)
        .map(|_| (0..n).map(|_| rng.below(n)).collect())
        .collect()
}

fn check_samples(n: usize, samples: &[Vec<usize>]) -> Result<(), ValidationError> {
    if samples.is_empty() || n == 0 {
        return Err(ValidationError::Resamples("no samples".into()));
    }
    if samples
        .iter()
        .any(|s| s.len() != n || s.iter().any(|&r| r >= n))
    {
        return Err(ValidationError::Resamples(
            "samples must have n rows in 0..n".into(),
        ));
    }
    Ok(())
}

/// Result of [`predab_resample`]: one entry per index.
#[derive(Debug, Clone, PartialEq)]
pub struct ValidationTable {
    /// Index names.
    pub names: Vec<String>,
    /// Apparent index of the model fitted on all rows.
    pub index_orig: Vec<f64>,
    /// Mean index on the training samples.
    pub training: Vec<f64>,
    /// Mean index on the test samples (.632: weighted sum).
    pub test: Vec<f64>,
    /// Optimism.
    pub optimism: Vec<f64>,
    /// `index_orig − optimism`.
    pub corrected: Vec<f64>,
    /// Successful resamples with a finite index.
    pub n: Vec<usize>,
    /// Resamples whose fit failed.
    pub failed: usize,
}

impl ValidationTable {
    /// Row of the index called `name`: `(orig, training, test, optimism,
    /// corrected, n)`.
    pub fn get(&self, name: &str) -> Option<(f64, f64, f64, f64, f64, usize)> {
        let k = self.names.iter().position(|v| v == name)?;
        Some((
            self.index_orig[k],
            self.training[k],
            self.test[k],
            self.optimism[k],
            self.corrected[k],
            self.n[k],
        ))
    }
}

/// Generic resampling validation (`rms:::predab.resample`).
///
/// * `fit(rows)` fits the model (the whole modelling process) on the given
///   rows (with repetitions); `None` marks a failed fit, which is skipped.
/// * `measure(model, rows, apparent)` evaluates the indexes of a model on
///   rows: `apparent = true` on the rows it was fitted on (rms's
///   `evalfit = TRUE`), `false` on test rows.
pub fn predab_resample<M>(
    resamples: &Resamples,
    names: &[&str],
    mut fit: impl FnMut(&[usize]) -> Option<M>,
    mut measure: impl FnMut(&M, &[usize], bool) -> Vec<f64>,
) -> Result<ValidationTable, ValidationError> {
    let n = resamples.n;
    let k = names.len();
    let all: Vec<usize> = (0..n).collect();
    let orig_fit = fit(&all).ok_or(ValidationError::OriginalFit)?;
    let index_orig = measure(&orig_fit, &all, true);
    if index_orig.len() != k {
        return Err(ValidationError::Data(
            "measure returned the wrong number of indexes".into(),
        ));
    }
    let (mut train, mut test, mut num) = (vec![0.0; k], vec![0.0; k], vec![0usize; k]);
    let mut failed = 0;
    for ((tr, te), &wt) in resamples.splits.iter().zip(&resamples.weights) {
        let Some(m) = fit(tr) else {
            failed += 1;
            continue;
        };
        let a = measure(&m, tr, true);
        let b = measure(&m, te, false);
        for j in 0..k {
            if a[j].is_finite() && b[j].is_finite() {
                num[j] += 1;
                train[j] += a[j];
                test[j] += b[j] * wt;
            }
        }
    }
    let training: Vec<f64> = (0..k).map(|j| train[j] / num[j] as f64).collect();
    let (test, optimism): (Vec<f64>, Vec<f64>) = if resamples.method == ResampleMethod::Point632 {
        let o = (0..k).map(|j| 0.632 * (index_orig[j] - test[j])).collect();
        (test, o)
    } else {
        let t: Vec<f64> = (0..k).map(|j| test[j] / num[j] as f64).collect();
        let o = (0..k).map(|j| training[j] - t[j]).collect();
        (t, o)
    };
    let corrected = (0..k).map(|j| index_orig[j] - optimism[j]).collect();
    Ok(ValidationTable {
        names: names.iter().map(|s| s.to_string()).collect(),
        index_orig,
        training,
        test,
        optimism,
        corrected,
        n: num,
        failed,
    })
}

fn rows_of(x: &Mat<f64>, rows: &[usize]) -> Mat<f64> {
    Mat::from_fn(rows.len(), x.ncols(), |i, j| x[(rows[i], j)])
}

/// Linear predictor `a + xβ` of every row of `x`.
fn linear_predictor(x: &Mat<f64>, intercept: f64, beta: &Col<f64>) -> Vec<f64> {
    (0..x.nrows())
        .map(|i| intercept + (0..x.ncols()).map(|j| x[(i, j)] * beta[j]).sum::<f64>())
        .collect()
}

fn check_xy(x: &Mat<f64>, y: &[f64]) -> Result<(), ValidationError> {
    if x.nrows() != y.len() {
        return Err(ValidationError::Data(format!(
            "x has {} rows, y has {}",
            x.nrows(),
            y.len()
        )));
    }
    Ok(())
}

/// A logistic fit's linear predictor on all rows.
fn fit_logistic(x: &Mat<f64>, y: &[f64], rows: &[usize]) -> Option<Vec<f64>> {
    let xr = rows_of(x, rows);
    let yr = Col::from_fn(rows.len(), |i| y[rows[i]]);
    let f = LogisticRegression::builder()
        .tolerance(1e-12)
        .max_iterations(100)
        .build()
        .fit(&xr, &yr)
        .ok()?;
    let lp = linear_predictor(x, f.intercept()?, f.coefficients());
    lp.iter().all(|v| v.is_finite()).then_some(lp)
}

/// Index names of [`validate_lrm`].
pub const LRM_INDEXES: [&str; 11] = [
    "Dxy",
    "R2",
    "Intercept",
    "Slope",
    "Emax",
    "D",
    "U",
    "Q",
    "B",
    "g",
    "gp",
];

/// `rms::validate.lrm` for a binary logistic model of 0/1 `y` on `x`
/// (no intercept column).
pub fn validate_lrm(
    x: &Mat<f64>,
    y: &[f64],
    resamples: &Resamples,
) -> Result<ValidationTable, ValidationError> {
    check_xy(x, y)?;
    let df = x.ncols() as f64;
    predab_resample(
        resamples,
        &LRM_INDEXES,
        |rows| fit_logistic(x, y, rows),
        |lp, rows, apparent| lrm_measure(lp, y, rows, apparent, df),
    )
}

fn lrm_measure(lp_all: &[f64], y_all: &[f64], rows: &[usize], apparent: bool, df: f64) -> Vec<f64> {
    let lp: Vec<f64> = rows.iter().map(|&i| lp_all[i]).collect();
    let y: Vec<f64> = rows.iter().map(|&i| y_all[i]).collect();
    let n = y.len() as f64;
    let brier = lp
        .iter()
        .zip(&y)
        .map(|(l, v)| (v - plogis(*l)).powi(2))
        .sum::<f64>()
        / n;
    if apparent {
        let Ok(s) = lrm_stats(&y, &lp, None, df) else {
            return vec![f64::NAN; 11];
        };
        let d = (s.model_lr - 1.0) / n;
        let u = -2.0 / n;
        return vec![s.dxy, s.r2, 0.0, 1.0, 0.0, d, u, d - u, brier, s.g, s.gp];
    }
    let (a, b, recal_dev) = logistic_recalibration(&lp, &y);
    let recal: Vec<f64> = lp.iter().map(|l| a + b * l).collect();
    let Ok(s) = lrm_stats(&y, &recal, None, 1.0) else {
        return vec![f64::NAN; 11];
    };
    let emax = (0..=2000)
        .map(|k| {
            let p = k as f64 * 5e-4;
            let l = (p / (1.0 - p)).ln();
            (p - plogis(a + b * l)).abs()
        })
        .filter(|v| v.is_finite())
        .fold(0.0, f64::max);
    let d = (s.model_lr - 1.0) / n;
    let clipped: Vec<f64> = lp.iter().map(|v| v.clamp(-40.0, 40.0)).collect();
    let l01 = binomial_deviance(&y, &clipped, &|_| 1.0);
    let u = (l01 - recal_dev - 2.0) / n;
    let scaled: Vec<f64> = lp.iter().map(|v| b * v).collect();
    let probs: Vec<f64> = recal.iter().map(|&v| plogis(v)).collect();
    vec![
        s.dxy,
        s.r2,
        a,
        b,
        emax,
        d,
        u,
        d - u,
        brier,
        gini_md(&scaled),
        gini_md(&probs),
    ]
}

/// A linear fit's predictions on all rows.
fn fit_ols(x: &Mat<f64>, y: &[f64], rows: &[usize]) -> Option<Vec<f64>> {
    let xr = rows_of(x, rows);
    let yr = Col::from_fn(rows.len(), |i| y[rows[i]]);
    let f = OlsRegressor::builder().build().fit(&xr, &yr).ok()?;
    let r = f.result();
    if r.aliased.iter().any(|&a| a) {
        return None;
    }
    Some(linear_predictor(x, r.intercept?, &r.coefficients))
}

/// Index names of [`validate_ols`].
pub const OLS_INDEXES: [&str; 5] = ["R-square", "MSE", "g", "Intercept", "Slope"];

/// `rms::validate.ols` for a linear model of `y` on `x` (no intercept
/// column).
pub fn validate_ols(
    x: &Mat<f64>,
    y: &[f64],
    resamples: &Resamples,
) -> Result<ValidationTable, ValidationError> {
    check_xy(x, y)?;
    predab_resample(
        resamples,
        &OLS_INDEXES,
        |rows| fit_ols(x, y, rows),
        |pred, rows, apparent| {
            let p: Vec<f64> = rows.iter().map(|&i| pred[i]).collect();
            let yy: Vec<f64> = rows.iter().map(|&i| y[i]).collect();
            let n = yy.len() as f64;
            let ybar = yy.iter().sum::<f64>() / n;
            let sst: f64 = yy.iter().map(|v| (v - ybar).powi(2)).sum();
            let sse: f64 = yy.iter().zip(&p).map(|(a, b)| (a - b).powi(2)).sum();
            let (a, b) = if apparent {
                (0.0, 1.0)
            } else {
                let pbar = p.iter().sum::<f64>() / n;
                let sxx: f64 = p.iter().map(|v| (v - pbar).powi(2)).sum();
                let sxy: f64 = p
                    .iter()
                    .zip(&yy)
                    .map(|(u, v)| (u - pbar) * (v - ybar))
                    .sum();
                let b = sxy / sxx;
                (ybar - b * pbar, b)
            };
            let scaled: Vec<f64> = p.iter().map(|v| b * v).collect();
            vec![1.0 - sse / sst, sse / n, gini_md(&scaled), a, b]
        },
    )
}

/// Overfitting-corrected calibration curve ([`calibrate_lrm`],
/// [`calibrate_ols`]).
#[derive(Debug, Clone, PartialEq)]
pub struct Calibration {
    /// Grid of predicted values.
    pub predy: Vec<f64>,
    /// Apparent calibration (lowess of `y` on the prediction) at `predy`.
    pub calibrated_orig: Vec<f64>,
    /// Bias-corrected calibration.
    pub calibrated_corrected: Vec<f64>,
    /// Optimism at each grid point.
    pub optimism: Vec<f64>,
    /// Resamples used at each grid point.
    pub n: Vec<usize>,
    /// Mean absolute error of the corrected curve over the predictions.
    pub mean_abs_error: f64,
    /// Mean squared error.
    pub mean_squared_error: f64,
    /// 0.9 quantile of the absolute error.
    pub quantile90_abs_error: f64,
}

/// `approx(x, y, xout, ties = function(x) x[1])` for sorted `x`.
fn approx_ties_first(x: &[f64], y: &[f64], xout: &[f64]) -> Vec<f64> {
    let (mut ux, mut uy) = (Vec::new(), Vec::new());
    for (i, &v) in x.iter().enumerate() {
        if ux.last() != Some(&v) {
            ux.push(v);
            uy.push(y[i]);
        }
    }
    approx_ties_mean(&ux, &uy, xout)
}

/// `cal.error` of `calibrate.default`: lowess calibration minus the grid.
fn cal_error(pred: &[f64], y: &[f64], predy: &[f64]) -> Vec<f64> {
    let Ok(smo) = lowess(
        pred,
        y,
        &LowessOptions {
            frac: 2.0 / 3.0,
            iterations: 0,
            delta: None,
        },
    ) else {
        return vec![f64::NAN; predy.len()];
    };
    approx_ties_first(&smo.x, &smo.fitted, predy)
        .iter()
        .zip(predy)
        .map(|(c, p)| c - p)
        .collect()
}

fn calibrate_with(
    predicted: Vec<f64>,
    resamples: &Resamples,
    predy: Option<Vec<f64>>,
    fit: impl FnMut(&[usize]) -> Option<Vec<f64>>,
    y: &[f64],
    to_scale: impl Fn(f64) -> f64,
) -> Result<Calibration, ValidationError> {
    let n = predicted.len();
    let predy = match predy {
        Some(p) => p,
        None => {
            if n < 11 {
                return Err(ValidationError::Data(
                    "need n > 10 without a predy grid".into(),
                ));
            }
            let mut p = predicted.clone();
            p.sort_by(|a, b| a.total_cmp(b));
            // R's seq(from, to, length.out = 50): interior from + k·by,
            // the last point exactly `to`.
            let (lo, hi) = (p[4], p[n - 5]);
            let by = (hi - lo) / 49.0;
            (0..50)
                .map(|k| if k == 49 { hi } else { lo + k as f64 * by })
                .collect()
        }
    };
    let names: Vec<String> = (0..predy.len()).map(|k| k.to_string()).collect();
    let name_refs: Vec<&str> = names.iter().map(String::as_str).collect();
    let table = predab_resample(resamples, &name_refs, fit, |lp, rows, _| {
        let pr: Vec<f64> = rows.iter().map(|&i| to_scale(lp[i])).collect();
        let yy: Vec<f64> = rows.iter().map(|&i| y[i]).collect();
        cal_error(&pr, &yy, &predy)
    })?;
    let orig: Vec<f64> = table
        .index_orig
        .iter()
        .zip(&predy)
        .map(|(e, p)| e + p)
        .collect();
    let corrected: Vec<f64> = orig
        .iter()
        .zip(&table.optimism)
        .map(|(o, op)| o - op)
        .collect();
    // print.calibrate.default: errors of the corrected curve at the
    // predictions.
    let keep: Vec<usize> = (0..predy.len())
        .filter(|&k| (predy[k] + corrected[k]).is_finite())
        .collect();
    let gx: Vec<f64> = keep.iter().map(|&k| predy[k]).collect();
    let gy: Vec<f64> = keep.iter().map(|&k| corrected[k]).collect();
    let at = approx_ties_mean(&gx, &gy, &predicted);
    let err: Vec<f64> = predicted
        .iter()
        .zip(&at)
        .map(|(p, c)| p - c)
        .filter(|v| v.is_finite())
        .collect();
    let mut abs: Vec<f64> = err.iter().map(|v| v.abs()).collect();
    abs.sort_by(|a, b| a.total_cmp(b));
    let m = err.len() as f64;
    Ok(Calibration {
        predy,
        calibrated_orig: orig,
        calibrated_corrected: corrected,
        optimism: table.optimism,
        n: table.n,
        mean_abs_error: abs.iter().sum::<f64>() / m,
        mean_squared_error: err.iter().map(|v| v * v).sum::<f64>() / m,
        quantile90_abs_error: if abs.is_empty() {
            f64::NAN
        } else {
            quantile7(&abs, 0.9)
        },
    })
}

/// `rms::calibrate` of a binary logistic model: lowess calibration of `y`
/// on the predicted probability at the grid `predy` (default: 50 points
/// between the 5th smallest and 5th largest prediction), corrected for
/// overfitting by resampling.
pub fn calibrate_lrm(
    x: &Mat<f64>,
    y: &[f64],
    resamples: &Resamples,
    predy: Option<Vec<f64>>,
) -> Result<Calibration, ValidationError> {
    check_xy(x, y)?;
    let all: Vec<usize> = (0..y.len()).collect();
    let lp = fit_logistic(x, y, &all).ok_or(ValidationError::OriginalFit)?;
    let predicted = lp.iter().map(|&v| plogis(v)).collect();
    calibrate_with(
        predicted,
        resamples,
        predy,
        |rows| fit_logistic(x, y, rows),
        y,
        plogis,
    )
}

/// `rms::calibrate` of a linear model (lowess of `y` on the prediction).
pub fn calibrate_ols(
    x: &Mat<f64>,
    y: &[f64],
    resamples: &Resamples,
    predy: Option<Vec<f64>>,
) -> Result<Calibration, ValidationError> {
    check_xy(x, y)?;
    let all: Vec<usize> = (0..y.len()).collect();
    let predicted = fit_ols(x, y, &all).ok_or(ValidationError::OriginalFit)?;
    calibrate_with(
        predicted,
        resamples,
        predy,
        |rows| fit_ols(x, y, rows),
        y,
        |v| v,
    )
}

/// Bootstrap replicates of a fit's coefficients (`rms::bootcov`).
#[derive(Debug, Clone, PartialEq)]
pub struct BootCov {
    /// Coefficients of each successful replicate (`replicates × k`).
    pub replicates: Mat<f64>,
    /// Mean of the replicates.
    pub mean: Vec<f64>,
    /// Bootstrap covariance (divisor `B − 1`), usable as `V̂` in Wald tests,
    /// `anova` and `effects::contrast`.
    pub vcov: Mat<f64>,
    /// Replicates whose fit failed (skipped).
    pub failed: usize,
}

impl BootCov {
    /// Percentile intervals (R type-7 quantiles) of every coefficient.
    pub fn percentile_ci(&self, level: f64) -> Vec<(f64, f64)> {
        (0..self.replicates.ncols())
            .map(|j| {
                let col: Vec<f64> = (0..self.replicates.nrows())
                    .map(|b| self.replicates[(b, j)])
                    .collect();
                percentile(col, level)
            })
            .collect()
    }

    /// Percentile interval of the linear combination `cᵀβ` (a contrast or a
    /// prediction).
    pub fn linear_combination_ci(&self, c: &[f64], level: f64) -> (f64, f64) {
        let vals = (0..self.replicates.nrows())
            .map(|b| (0..c.len()).map(|j| c[j] * self.replicates[(b, j)]).sum())
            .collect();
        percentile(vals, level)
    }
}

fn percentile(mut v: Vec<f64>, level: f64) -> (f64, f64) {
    v.sort_by(|a, b| a.total_cmp(b));
    let a = (1.0 - level) / 2.0;
    (quantile7(&v, a), quantile7(&v, 1.0 - a))
}

/// Bootstrap covariance of any fit: `fit(rows)` returns the coefficient
/// vector of the model refitted on the resampled rows (`None` = failure,
/// skipped), for each resample in `samples` (from [`bootstrap_rows`],
/// [`cluster_bootstrap_rows`] or given explicitly).
pub fn bootcov(
    samples: &[Vec<usize>],
    mut fit: impl FnMut(&[usize]) -> Option<Vec<f64>>,
) -> Result<BootCov, ValidationError> {
    let reps: Vec<Vec<f64>> = samples.iter().filter_map(|s| fit(s)).collect();
    let failed = samples.len() - reps.len();
    if reps.len() < 2 {
        return Err(ValidationError::Resamples(
            "fewer than two successful replicates".into(),
        ));
    }
    let (b, k) = (reps.len(), reps[0].len());
    let replicates = Mat::from_fn(b, k, |i, j| reps[i][j]);
    let mean: Vec<f64> = (0..k)
        .map(|j| reps.iter().map(|r| r[j]).sum::<f64>() / b as f64)
        .collect();
    let vcov = Mat::from_fn(k, k, |a, c| {
        reps.iter()
            .map(|r| (r[a] - mean[a]) * (r[c] - mean[c]))
            .sum::<f64>()
            / (b - 1) as f64
    });
    Ok(BootCov {
        replicates,
        mean,
        vcov,
        failed,
    })
}

/// `b` seeded ordinary bootstrap samples of `n` rows.
pub fn bootstrap_rows(n: usize, b: usize, seed: u64) -> Vec<Vec<usize>> {
    draw(n, b, seed)
}

/// `b` seeded cluster (block) bootstrap samples: draw as many clusters as
/// there are, with replacement, and take all rows of each drawn cluster
/// (repeated measures, RMS Ch 7). `clusters` holds a cluster code per row.
pub fn cluster_bootstrap_rows(clusters: &[usize], b: usize, seed: u64) -> Vec<Vec<usize>> {
    let mut ids: Vec<usize> = clusters.to_vec();
    ids.sort_unstable();
    ids.dedup();
    let members: Vec<Vec<usize>> = ids
        .iter()
        .map(|&c| (0..clusters.len()).filter(|&i| clusters[i] == c).collect())
        .collect();
    let mut rng = Rng(seed);
    (0..b)
        .map(|_| {
            (0..ids.len())
                .flat_map(|_| members[rng.below(ids.len())].iter().copied())
                .collect()
        })
        .collect()
}

/// Bootstrap intervals for the ranks of predictor importance (RMS §5.4):
/// `original` holds one importance statistic per predictor (e.g. partial
/// χ² − d.f. from `inference::anova`) and `replicates` the same statistics
/// per bootstrap fit (`B × m`). Ranks are R's `rank` (1 = least important,
/// ties averaged); returns `(rank, lower, upper)` per predictor.
pub fn rank_intervals(original: &[f64], replicates: &Mat<f64>, level: f64) -> Vec<(f64, f64, f64)> {
    let m = original.len();
    let orig = average_ranks(original);
    let ranks: Vec<Vec<f64>> = (0..replicates.nrows())
        .map(|b| average_ranks(&(0..m).map(|j| replicates[(b, j)]).collect::<Vec<_>>()))
        .collect();
    (0..m)
        .map(|j| {
            let (lo, hi) = percentile(ranks.iter().map(|r| r[j]).collect(), level);
            (orig[j], lo, hi)
        })
        .collect()
}

fn average_ranks(v: &[f64]) -> Vec<f64> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| v[a].total_cmp(&v[b]));
    let mut r = vec![0.0; v.len()];
    let mut a = 0;
    while a < idx.len() {
        let mut b = a;
        while b < idx.len() && v[idx[b]] == v[idx[a]] {
            b += 1;
        }
        for &i in &idx[a..b] {
            r[i] = (a + b + 1) as f64 / 2.0;
        }
        a = b;
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crossvalidation_groups_cover_every_row_once() {
        let perm: Vec<usize> = (0..20).rev().collect();
        let r = Resamples::crossvalidation(&perm, 5).unwrap();
        let mut seen = [0; 20];
        for (train, test) in &r.splits {
            assert_eq!(train.len() + test.len(), 20);
            for &t in test {
                seen[t] += 1;
            }
        }
        assert!(seen.iter().all(|&c| c == 1));
        // n/B = 4.6: rms's rounded boundaries make groups 2 and 3 share a row.
        let perm: Vec<usize> = (0..23).collect();
        let r = Resamples::crossvalidation(&perm, 5).unwrap();
        assert_eq!(r.splits[1].1, vec![5, 6, 7, 8, 9]);
        assert_eq!(r.splits[2].1, vec![9, 10, 11, 12, 13]);
    }

    #[test]
    fn point632_weights_sum_to_one() {
        let r = Resamples::point632_seeded(30, 60, 7).unwrap();
        assert!((r.weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(Resamples::point632(3, vec![vec![0, 1, 2]]).is_err());
    }

    #[test]
    fn ranks_and_cluster_rows() {
        assert_eq!(
            average_ranks(&[3.0, 1.0, 3.0, 2.0]),
            vec![3.5, 1.0, 3.5, 2.0]
        );
        let rows = cluster_bootstrap_rows(&[0, 0, 1, 1, 1, 2], 5, 3);
        for r in &rows {
            // Whole clusters only: each cluster contributes all of its rows.
            let count = |c: &[usize]| r.iter().filter(|i| c.contains(i)).count();
            assert_eq!(count(&[0, 1]) % 2, 0);
            assert_eq!(count(&[2, 3, 4]) % 3, 0);
        }
        let reps = Mat::from_fn(4, 2, |b, j| (b + j) as f64);
        let r = rank_intervals(&[1.0, 2.0], &reps, 0.9);
        assert_eq!(r[1], (2.0, 2.0, 2.0));
    }

    #[test]
    fn invalid_resamples() {
        assert!(Resamples::bootstrap(3, vec![vec![0, 1]]).is_err());
        assert!(Resamples::bootstrap(3, vec![]).is_err());
        assert!(Resamples::crossvalidation(&[0, 0, 1, 2], 2).is_err());
        assert!(Resamples::crossvalidation(&[0, 1, 2], 2).is_err());
    }
}
