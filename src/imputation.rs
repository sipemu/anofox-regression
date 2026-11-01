//! Multiple imputation by additive regression, bootstrapping and predictive
//! mean matching (`Hmisc::aregImpute`, RMS §3.7).
//!
//! Every variable with missing values is imputed in turn from all the
//! others (chained equations): on a bootstrap sample of its observed rows
//! an [`areg`] model with optimal transformations
//! (splines with `nk` knots, dummies for categorical variables) predicts it,
//! and each missing value takes the observed value of a donor whose
//! predicted (transformed) value is close – drawn with tricube weights
//! (`match = "weighted"`) or the closest. After `burnin` iterations each
//! further iteration gives one completed data set; fit the model to each
//! and combine with [`crate::inference::pool::pool_rubin`].
//!
//! Random draws use R's generator ([`RRng`]), so with the same seed the
//! imputations replay `set.seed(seed); aregImpute(...)` exactly.

use crate::solvers::RegressionError;
use crate::transform::{areg, AregOptions, AregType};
use crate::utils::RRng;
use faer::Mat;

/// How a variable enters [`areg_impute`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ImputeVariable {
    /// Numeric: a restricted cubic spline (linear with two distinct values).
    #[default]
    Numeric,
    /// Numeric but always linear (`I(x)` in the formula).
    Linear,
    /// Categorical (level codes).
    Categorical,
}

/// Donor selection of [`areg_impute`] (`match =`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DonorMatch {
    /// Draw a donor with tricube weights on the distance in predicted
    /// values, scaled by `f` times the mean distance (`"weighted"`,
    /// `fweighted = f`, default 0.2).
    Weighted(f64),
    /// The donor with the closest prediction (`"closest"`; predictions are
    /// jittered by `runif(−1e-4, 1e-4)`).
    Closest,
}

/// Options of [`areg_impute`].
#[derive(Debug, Clone, PartialEq)]
pub struct AregImputeOptions {
    /// Number of imputations (`n.impute`, default 5).
    pub n_impute: usize,
    /// Spline knots (`nk`, default 3; 0 makes every variable linear).
    pub nk: usize,
    /// Model only the linear part of a spline variable being imputed
    /// (`tlinear`, default `true`).
    pub tlinear: bool,
    /// Iterations before the first kept imputation (`burnin`); `None` is
    /// Hmisc's default, 3, or 0 when only one variable has missing values.
    pub burnin: Option<usize>,
    /// Donor selection.
    pub matching: DonorMatch,
    /// Seed of R's generator (`set.seed`).
    pub seed: i32,
}

impl Default for AregImputeOptions {
    fn default() -> Self {
        Self {
            n_impute: 5,
            nk: 3,
            tlinear: true,
            burnin: None,
            matching: DonorMatch::Weighted(0.2),
            seed: 1,
        }
    }
}

/// Result of [`areg_impute`].
#[derive(Debug, Clone)]
pub struct AregImpute {
    /// Rows with missing values of each column (empty if complete).
    pub missing: Vec<Vec<usize>>,
    /// Imputed values of each column: `imputed[j][r][m]` is imputation `m`
    /// of the `r`-th missing row of column `j`.
    pub imputed: Vec<Vec<Vec<f64>>>,
    /// R² of the last imputation model of each column with missing values
    /// (NaN for complete columns).
    pub r_squared: Vec<f64>,
}

impl AregImpute {
    /// Completed data set `m` (`0..n_impute`): `x` with imputation `m`
    /// filled in.
    pub fn complete(&self, x: &Mat<f64>, m: usize) -> Mat<f64> {
        let mut out = x.to_owned();
        for (j, rows) in self.missing.iter().enumerate() {
            for (r, &i) in rows.iter().enumerate() {
                out[(i, j)] = self.imputed[j][r][m];
            }
        }
        out
    }
}

/// `whichClosePW`: for every recipient prediction, a donor index drawn with
/// tricube weights (uniforms `u`, one per recipient).
fn which_close_pw(donor: &[f64], recipient: &[f64], u: &[f64], f: f64) -> Vec<usize> {
    let lx = donor.len();
    recipient
        .iter()
        .zip(u)
        .map(|(&w, &ri)| {
            let d: Vec<f64> = donor.iter().map(|x| (x - w).abs()).collect();
            let dmean = f * d.iter().sum::<f64>() / lx as f64;
            let wt: Vec<f64> = d
                .iter()
                .map(|v| (1.0 - (v / dmean).min(1.0).powi(3)).powi(3))
                .collect();
            let sump: f64 = wt.iter().sum();
            let mut prob = 0.0;
            let mut m = 0;
            for v in &wt {
                prob += v / sump;
                if ri > prob {
                    m += 1;
                }
            }
            m.min(lx - 1)
        })
        .collect()
}

/// `whichClosest`: the first donor with the smallest distance.
fn which_closest(donor: &[f64], recipient: &[f64]) -> Vec<usize> {
    recipient
        .iter()
        .map(|&w| {
            (0..donor.len()).fold(0, |b, k| {
                if (donor[k] - w).abs() < (donor[b] - w).abs() {
                    k
                } else {
                    b
                }
            })
        })
        .collect()
}

fn distinct(v: impl Iterator<Item = f64>) -> Vec<f64> {
    let mut d: Vec<f64> = v.collect();
    d.sort_by(f64::total_cmp);
    d.dedup();
    d
}

/// `aregImpute(~ ., data)` for the columns of `x` (NaN = missing). See the
/// [module docs](self).
pub fn areg_impute(
    x: &Mat<f64>,
    kinds: &[ImputeVariable],
    options: &AregImputeOptions,
) -> Result<AregImpute, RegressionError> {
    let (n, p) = (x.nrows(), x.ncols());
    if kinds.len() != p || p < 2 {
        return Err(RegressionError::NumericalError(
            "need two or more columns and one kind per column".into(),
        ));
    }
    let mut rng = RRng::new(options.seed);
    let mut xf = x.to_owned();
    let mut missing = vec![Vec::new(); p];
    let mut vtype = vec![AregType::Spline; p];
    for j in 0..p {
        missing[j] = (0..n).filter(|&i| x[(i, j)].is_nan()).collect();
        let observed: Vec<f64> = (0..n).map(|i| x[(i, j)]).filter(|v| !v.is_nan()).collect();
        let levels = distinct(observed.iter().copied());
        if levels.len() < 2 {
            return Err(RegressionError::NumericalError(format!(
                "column {j} is constant or empty"
            )));
        }
        vtype[j] = match kinds[j] {
            ImputeVariable::Categorical => AregType::Categorical,
            _ if options.nk == 0 || levels.len() == 2 || kinds[j] == ImputeVariable::Linear => {
                AregType::Linear
            }
            _ => AregType::Spline,
        };
        // start from a random sample of the observed values
        let nm = missing[j].len();
        if nm > 0 {
            let draw = rng.sample(observed.len(), nm, nm > n - nm);
            for (r, &i) in missing[j].iter().enumerate() {
                xf[(i, j)] = observed[draw[r]];
            }
        }
    }
    let wna: Vec<usize> = (0..p).filter(|&j| !missing[j].is_empty()).collect();
    let burnin = options.burnin.unwrap_or(if wna.len() < 2 { 0 } else { 3 });
    let mut imputed: Vec<Vec<Vec<f64>>> = missing
        .iter()
        .map(|m| vec![vec![f64::NAN; options.n_impute]; m.len()])
        .collect();
    let mut r_squared = vec![f64::NAN; p];
    let n_levels: Vec<usize> = (0..p)
        .map(|j| distinct((0..n).map(|i| xf[(i, j)])).len())
        .collect();
    for iter in 0..burnin + options.n_impute {
        for &i in &wna {
            let nai = &missing[i];
            let obs: Vec<usize> = (0..n).filter(|r| !nai.contains(r)).collect();
            let npr = obs.len();
            let ytype = if options.tlinear && vtype[i] == AregType::Spline {
                AregType::Linear
            } else {
                vtype[i]
            };
            let boot: Vec<usize> = rng.sample(npr, npr, true).iter().map(|&k| obs[k]).collect();
            for c in (0..p).filter(|&c| vtype[c] == AregType::Categorical) {
                if distinct(boot.iter().map(|&r| xf[(r, c)])).len() < n_levels[c] {
                    return Err(RegressionError::NumericalError(format!(
                        "a bootstrap resample had too few unique values of column {c}"
                    )));
                }
            }
            let others: Vec<usize> = (0..p).filter(|&c| c != i).collect();
            let xo = Mat::from_fn(n, p - 1, |r, c| xf[(r, others[c])]);
            let opts = AregOptions {
                nk: options.nk,
                x_types: Some(others.iter().map(|&c| vtype[c]).collect()),
                y_type: ytype,
            };
            let rows = |idx: &[usize]| Mat::from_fn(idx.len(), p - 1, |r, c| xo[(idx[r], c)]);
            let ys = |idx: &[usize]| idx.iter().map(|&r| xf[(r, i)]).collect::<Vec<f64>>();
            // a spline target: areg draws among the inverses of a
            // non-monotone transformation (`yinv(what = "sample")`)
            let fit = |idx: &[usize], rng: &mut RRng| {
                let a = areg(&rows(idx), &ys(idx), &opts)?;
                let lp: Vec<f64> = a.linear_predictors.iter().copied().collect();
                a.sample_inverse_y(&lp, rng);
                Ok::<_, RegressionError>(a)
            };
            let f = fit(&boot, &mut rng)?;
            r_squared[i] = f.r_squared;
            let mut pti = f.predict(&xo);
            // pmmtype 1: donors' predictions from a fit to all observed rows
            let g = fit(&obs, &mut rng)?;
            let pobs = g.predict(&rows(&obs));
            for (k, &r) in obs.iter().enumerate() {
                pti[r] = pobs[k];
            }
            if ytype == AregType::Linear {
                let mean = pti.iter().sum::<f64>() / n as f64;
                let sd =
                    (pti.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1) as f64).sqrt();
                pti.iter_mut().for_each(|v| *v = (*v - mean) / sd);
            }
            let mut donor: Vec<f64> = obs.iter().map(|&r| pti[r]).collect();
            let recip: Vec<f64> = nai.iter().map(|&r| pti[r]).collect();
            let chosen = match options.matching {
                DonorMatch::Closest => {
                    donor.iter_mut().for_each(|v| *v += rng.runif(-1e-4, 1e-4));
                    which_closest(&donor, &recip)
                }
                DonorMatch::Weighted(fw) => {
                    let u: Vec<f64> = (0..recip.len()).map(|_| rng.unif_rand()).collect();
                    which_close_pw(&donor, &recip, &u, fw)
                }
            };
            for (r, &row) in nai.iter().enumerate() {
                let v = xf[(obs[chosen[r]], i)];
                xf[(row, i)] = v;
                if iter >= burnin {
                    imputed[i][r][iter - burnin] = v;
                }
            }
        }
    }
    Ok(AregImpute {
        missing,
        imputed,
        r_squared,
    })
}
