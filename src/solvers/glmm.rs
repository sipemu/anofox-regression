//! Generalized linear mixed models (GLMM): a random intercept and optional
//! random slopes over one grouping factor, or several crossed / nested
//! random-intercept factors.
//!
//! Covers **Phases A–C** of the mixed-model work tracked in issue #25 (mirroring
//! the DuckDB extension's `glmm_fit_agg`), plus the extra families, offsets and
//! crossed random slopes of issue #29:
//!
//! ```text
//! g(μ_ij) = x_ij'β + z_ij'b_j,     b_j ~ N(0, Σ)
//! ```
//!
//! where `z_ij` is the random-effects design for observation `i` in group `j` —
//! a random intercept alone (`z_ij = 1`), or an intercept plus one or more
//! random slopes (`z_ij = [1, s_ij, …]`) with an **unstructured** covariance `Σ`.
//!
//! [`fit`](GlmmRegressor::fit) handles a single grouping factor (intercept +
//! optional slopes). [`fit_crossed`](GlmmRegressor::fit_crossed) handles several
//! **crossed or nested** grouping factors, each contributing a random intercept
//! with its own variance component (`(1|a) + (1|b)`, or `(1|a/b)` via the
//! interaction id), optionally with per-factor random slopes
//! (`(1 + x | a) + (1 | b)`, see
//! [`random_slopes_per_factor`](GlmmRegressorBuilder::random_slopes_per_factor)).
//!
//! Families: Gaussian (LMM), Poisson, binomial, negative binomial (θ estimated
//! as `glmer.nb`, or fixed), Gamma and Tweedie (`1 < p < 2`), all with the log
//! link except binomial (logit). An optional per-observation
//! [`offset`](GlmmRegressorBuilder::offset) enters the linear predictor with
//! coefficient 1.
//!
//! # Method
//!
//! The library follows the profiled-deviation formulation of `lme4`
//! (Bates, Mächler, Bolker & Walker, *Fitting Linear Mixed-Effects Models Using
//! lme4*, JSS 2015). The random effects are written `b = Λ_θ u` with
//! `u ~ N(0, σ² I)`; `Λ_θ` is block-diagonal with one `q×q` lower-triangular
//! *relative covariance factor* `T(θ)` per group, so `Σ = σ² T Tᵀ`. The scalars
//! `θ` are the `q(q+1)/2` lower-triangular entries of `T` (column-major, the
//! `lme4` convention).
//!
//! * **Random intercept (`q = 1`).** `θ = σ_b/σ` is a single scale-free ratio,
//!   profiled by golden-section search.
//! * **Random slopes (`q > 1`).** `θ` is profiled by Nelder–Mead, matching
//!   `lme4`'s derivative-free optimisation over the Cholesky factor.
//!
//! For each candidate `θ`:
//!
//! * **Gaussian response (LMM).** The penalized least-squares problem is solved
//!   in closed form and the profiled REML (or ML) deviance is evaluated exactly.
//!   This matches `lme4::lmer`.
//! * **Non-Gaussian response (GLMM).** A penalized IRLS (PIRLS) inner loop finds
//!   the conditional modes `(β, u)` and the Laplace deviance is evaluated at the
//!   mode. Because the fixed effects are part of the penalized conditional mode,
//!   this matches `lme4::glmer(nAGQ = 0)`. The criterion is lme4's
//!   `log|L_θ|² + ‖u‖² + aic(y, μ)`: the family `−2 log L` (Gamma / Tweedie with
//!   dispersion `deviance / n`, as R's `Gamma()$aic`). Gamma / Tweedie report
//!   `σ = sqrt(pwrss / n)` and scale the fixed-effect standard errors by it.
//!
//! The random block `Zᵀ W Z` is block-diagonal with one dense `q×q` block per
//! group (groups share no random columns), so the random effects are eliminated
//! by a Schur complement that inverts a `q×q` block per group — a scalar when
//! `q = 1`, the small dense block once slopes are present.
//!
//! # Example
//!
//! ```rust,ignore
//! use anofox_regression::solvers::GlmmRegressor;
//! use faer::{Col, Mat};
//!
//! // Random intercept + random slope on covariate 0:  y ~ x + (x | g).
//! let fitted = GlmmRegressor::gaussian()
//!     .with_intercept(true)
//!     .random_slopes(vec![0])
//!     .build()
//!     .fit(&x, &y, &group)?;
//!
//! println!("fixed effects: {:?}", fitted.fixed_effects());
//! println!("random cov Σ: {:?}", fitted.random_cov());
//!
//! // Wald inference for the fixed effects (lme4 summary / confint(method = "Wald")).
//! let z = fitted.z_values();
//! let p = fitted.p_values();
//! let (lo, hi) = fitted.conf_int(0.95);
//! ```

use crate::core::{
    BinomialFamily, BinomialLink, GlmFamily, NegativeBinomialFamily, PoissonFamily, PoissonLink,
    TweedieFamily,
};
use crate::solvers::penalized_glm::loglik::{log_likelihood as family_log_likelihood, LogLikKind};
use crate::solvers::traits::RegressionError;
use faer::{Col, Mat};
use statrs::distribution::{ContinuousCDF, Normal};

/// A GLMM with a random intercept (and optional random slopes) over one
/// grouping factor, or several crossed / nested grouping factors.
pub struct GlmmRegressor {
    kind: ResponseKind,
    with_intercept: bool,
    /// Whether the random-effects design includes a random intercept.
    random_intercept: bool,
    /// Indices into the columns of `x` that carry a random slope.
    random_slopes: Vec<usize>,
    /// Per-factor random-slope columns for [`fit_crossed`](Self::fit_crossed).
    random_slopes_per_factor: Option<Vec<Vec<usize>>>,
    /// Use REML (`true`) rather than ML (`false`) for the Gaussian case. Ignored
    /// for non-Gaussian families, which use the Laplace ML deviance.
    reml: bool,
    /// Maximum PIRLS iterations (inner loop) for the GLMM case.
    max_iterations: usize,
    /// Convergence tolerance for the inner loop and the outer θ search.
    tolerance: f64,
    /// Upper bound on the profiled ratio θ = σ_b / σ (random-intercept case).
    theta_max: f64,
    /// Fixed negative-binomial size θ; `None` estimates it (`glmer.nb`).
    nb_theta: Option<f64>,
    /// Per-observation offset added to the linear predictor.
    offset: Option<Col<f64>>,
}

impl GlmmRegressor {
    /// Gaussian response — a linear mixed model fitted by profiled REML.
    pub fn gaussian() -> GlmmRegressorBuilder {
        GlmmRegressorBuilder::new(ResponseKind::Gaussian)
    }

    /// Poisson response with the log link.
    pub fn poisson() -> GlmmRegressorBuilder {
        GlmmRegressorBuilder::new(ResponseKind::Poisson)
    }

    /// Binomial response with the logit link (0/1 or proportion `y`).
    pub fn binomial() -> GlmmRegressorBuilder {
        GlmmRegressorBuilder::new(ResponseKind::Binomial)
    }

    /// Negative-binomial (NB2) response with the log link,
    /// `Var(y) = μ + μ²/θ`. By default the size θ is estimated by maximising
    /// the profiled Laplace log-likelihood, as `lme4::glmer.nb(nAGQ = 0)`;
    /// fix it with [`GlmmRegressorBuilder::nb_theta`] to match
    /// `glmer(family = MASS::negative.binomial(θ), nAGQ = 0)`.
    pub fn negative_binomial() -> GlmmRegressorBuilder {
        GlmmRegressorBuilder::new(ResponseKind::NegativeBinomial)
    }

    /// Gamma response with the log link (`y > 0`), matching
    /// `lme4::glmer(family = Gamma(link = "log"), nAGQ = 0)`.
    pub fn gamma() -> GlmmRegressorBuilder {
        GlmmRegressorBuilder::new(ResponseKind::Gamma)
    }

    /// Tweedie (compound Poisson–gamma) response with variance power
    /// `1 < power < 2` and the log link (`y ≥ 0`). Uses the same Laplace
    /// criterion as lme4's Gamma GLMM (dispersion `deviance / n` inside the
    /// family log-likelihood), with the Tweedie series density.
    pub fn tweedie(power: f64) -> GlmmRegressorBuilder {
        GlmmRegressorBuilder::new(ResponseKind::Tweedie(power))
    }

    /// Fit the model.
    ///
    /// * `x` — fixed-effects design of shape `(n, p)`, **without** the grouping
    ///   factor and without an intercept column (an intercept is added
    ///   automatically when `with_intercept` is set).
    /// * `y` — response of length `n`.
    /// * `group` — grouping-factor id per observation, length `n`. Ids may be
    ///   any `usize`; they are compacted to `0..J` internally.
    ///
    /// Random-slope columns named via [`GlmmRegressorBuilder::random_slopes`]
    /// index into the columns of `x`.
    pub fn fit(
        &self,
        x: &Mat<f64>,
        y: &Col<f64>,
        group: &[usize],
    ) -> Result<FittedGlmm, RegressionError> {
        let slopes: &[usize] = match &self.random_slopes_per_factor {
            None => &self.random_slopes,
            Some(per) if per.len() == 1 && self.random_slopes.is_empty() => &per[0],
            Some(_) => {
                return Err(RegressionError::NumericalError(
                    "random_slopes_per_factor must have one entry per grouping factor \
                     (and is exclusive with random_slopes)"
                        .to_string(),
                ))
            }
        };
        self.fit_single(x, y, group, slopes)
    }

    /// Single-grouping-factor fit with the given random-slope columns.
    fn fit_single(
        &self,
        x: &Mat<f64>,
        y: &Col<f64>,
        group: &[usize],
        slopes: &[usize],
    ) -> Result<FittedGlmm, RegressionError> {
        let n = x.nrows();
        if y.nrows() != n || group.len() != n {
            return Err(RegressionError::DimensionMismatch {
                x_rows: n,
                y_len: y.nrows().max(group.len()),
            });
        }
        if n < 3 {
            return Err(RegressionError::InsufficientObservations { needed: 3, got: n });
        }
        let offset = self.offset_vec(n)?;
        self.validate_response(y)?;

        let q = usize::from(self.random_intercept) + slopes.len();
        if q == 0 {
            return Err(RegressionError::NumericalError(
                "GLMM needs a random intercept or at least one random slope".to_string(),
            ));
        }
        check_slope_columns(slopes, x.ncols())?;

        // Compact arbitrary group ids into 0..J and record membership.
        let (group_idx, n_groups) = compact_groups(group);
        if n_groups < 2 {
            return Err(RegressionError::NumericalError(
                "GLMM needs at least two distinct groups".to_string(),
            ));
        }

        // Fixed design (optional leading intercept) and random design Z.
        let p = if self.with_intercept {
            x.ncols() + 1
        } else {
            x.ncols()
        };
        if n <= p + q {
            return Err(RegressionError::InsufficientObservations {
                needed: p + q + 1,
                got: n,
            });
        }
        let design = build_design(x, self.with_intercept);
        let z = build_random_design(x, self.random_intercept, slopes);

        if let ResponseKind::Gaussian = self.kind {
            let y_adj = Col::from_fn(n, |i| y[i] - offset[i]);
            return self.fit_lmm(&design, &z, &y_adj, &group_idx, n_groups, p, q);
        }
        let y_vec: Vec<f64> = y.iter().copied().collect();
        self.with_family(|family, nb_theta| {
            self.fit_glmm(
                family, nb_theta, &design, &z, &y_vec, &offset, &group_idx, n_groups, p, q,
            )
        })
        .map(|f| self.with_saturated_term(f, y))
    }

    /// Fit with **multiple crossed or nested grouping factors**, each carrying a
    /// random intercept (and, optionally, random slopes) with its own
    /// covariance:
    ///
    /// ```text
    /// g(μ_i) = x_i'β + Σ_f z_{f,i}'b_{f, level_f(i)},   b_{f,·} ~ N(0, Σ_f)
    /// ```
    ///
    /// * `groups[f]` is the level id of factor `f` for each observation
    ///   (length `n`; any `usize`, compacted internally).
    /// * Crossed factors: pass the two id vectors directly, e.g.
    ///   `(1|sku) + (1|region)` → `&[&sku, &region]`.
    /// * Nested factors `(1|a/b)`: pass `a` and the interaction id `a:b` (a
    ///   globally-unique id per `(a, b)` pair) → `&[&a, &ab]`.
    /// * Random slopes per factor, e.g. `(1 + x[0] | a) + (1 | b)`, are set
    ///   with [`GlmmRegressorBuilder::random_slopes_per_factor`]
    ///   (`vec![vec![0], vec![]]`); each factor then gets an unstructured
    ///   `q_f × q_f` covariance.
    ///
    /// With a single factor this delegates to [`fit`](Self::fit).
    pub fn fit_crossed(
        &self,
        x: &Mat<f64>,
        y: &Col<f64>,
        groups: &[&[usize]],
    ) -> Result<FittedGlmm, RegressionError> {
        if groups.is_empty() {
            return Err(RegressionError::NumericalError(
                "fit_crossed needs at least one grouping factor".to_string(),
            ));
        }
        let n_factors = groups.len();
        let slopes: Vec<Vec<usize>> = match &self.random_slopes_per_factor {
            Some(per) => {
                if per.len() != n_factors || !self.random_slopes.is_empty() {
                    return Err(RegressionError::NumericalError(format!(
                        "random_slopes_per_factor has {} entries for {n_factors} grouping \
                         factors (it must match, and is exclusive with random_slopes)",
                        per.len()
                    )));
                }
                per.clone()
            }
            None if n_factors == 1 => vec![self.random_slopes.clone()],
            None if !self.random_slopes.is_empty() => {
                return Err(RegressionError::NumericalError(
                    "random_slopes applies to a single grouping factor; use \
                     random_slopes_per_factor with multiple grouping factors"
                        .to_string(),
                ));
            }
            None => vec![Vec::new(); n_factors],
        };
        if n_factors == 1 {
            return self.fit_single(x, y, groups[0], &slopes[0]);
        }

        let n = x.nrows();
        for g in groups {
            if g.len() != n {
                return Err(RegressionError::DimensionMismatch {
                    x_rows: n,
                    y_len: g.len(),
                });
            }
        }
        if y.nrows() != n {
            return Err(RegressionError::DimensionMismatch {
                x_rows: n,
                y_len: y.nrows(),
            });
        }
        let offset = self.offset_vec(n)?;
        self.validate_response(y)?;

        // Compact each factor's ids and lay out the combined random columns.
        let mut levels = Vec::with_capacity(n_factors);
        let mut n_levels = Vec::with_capacity(n_factors);
        let mut zs = Vec::with_capacity(n_factors);
        let mut qs = Vec::with_capacity(n_factors);
        for (g, sl) in groups.iter().zip(&slopes) {
            let (idx, jf) = compact_groups(g);
            if jf < 2 {
                return Err(RegressionError::NumericalError(
                    "each grouping factor needs at least two distinct levels".to_string(),
                ));
            }
            check_slope_columns(sl, x.ncols())?;
            let qf = usize::from(self.random_intercept) + sl.len();
            if qf == 0 {
                return Err(RegressionError::NumericalError(
                    "each grouping factor needs a random intercept or a random slope".to_string(),
                ));
            }
            levels.push(idx);
            n_levels.push(jf);
            zs.push(build_random_design(x, self.random_intercept, sl));
            qs.push(qf);
        }
        let spec = MultiSpec::new(qs, n_levels, levels, zs);

        let p = if self.with_intercept {
            x.ncols() + 1
        } else {
            x.ncols()
        };
        if n <= p + spec.n_theta {
            return Err(RegressionError::InsufficientObservations {
                needed: p + spec.n_theta + 1,
                got: n,
            });
        }
        let design = build_design(x, self.with_intercept);

        if let ResponseKind::Gaussian = self.kind {
            let y_adj: Vec<f64> = (0..n).map(|i| y[i] - offset[i]).collect();
            return self.fit_lmm_multi(&design, &y_adj, &spec, p);
        }
        let y_vec: Vec<f64> = y.iter().copied().collect();
        self.with_family(|family, nb_theta| {
            self.fit_glmm_multi(family, nb_theta, &design, &y_vec, &offset, &spec, p)
        })
        .map(|f| self.with_saturated_term(f, y))
    }

    /// The offset as a dense vector (zeros when none was set).
    fn offset_vec(&self, n: usize) -> Result<Vec<f64>, RegressionError> {
        match &self.offset {
            None => Ok(vec![0.0; n]),
            Some(o) if o.nrows() == n => {
                if o.iter().all(|v| v.is_finite()) {
                    Ok(o.iter().copied().collect())
                } else {
                    Err(RegressionError::NumericalError(
                        "offset must be finite".to_string(),
                    ))
                }
            }
            Some(o) => Err(RegressionError::DimensionMismatch {
                x_rows: n,
                y_len: o.nrows(),
            }),
        }
    }

    /// Family-specific checks on the response and family parameters for the
    /// negative-binomial, Gamma and Tweedie GLMMs.
    fn validate_response(&self, y: &Col<f64>) -> Result<(), RegressionError> {
        let bad = |msg: &str| Err(RegressionError::NumericalError(msg.to_string()));
        match self.kind {
            ResponseKind::NegativeBinomial => {
                if let Some(th) = self.nb_theta {
                    if !(th.is_finite() && th > 0.0) {
                        return bad("negative-binomial theta must be finite and positive");
                    }
                }
                if y.iter().any(|&v| !(v.is_finite() && v >= 0.0)) {
                    return bad("negative-binomial GLMM needs a non-negative response");
                }
            }
            ResponseKind::Gamma => {
                if y.iter().any(|&v| !(v.is_finite() && v > 0.0)) {
                    return bad("Gamma GLMM needs a strictly positive response");
                }
            }
            ResponseKind::Tweedie(power) => {
                if !(power > 1.0 && power < 2.0) {
                    return bad("Tweedie GLMM needs a variance power in (1, 2)");
                }
                if y.iter().any(|&v| !(v.is_finite() && v >= 0.0)) {
                    return bad("Tweedie GLMM needs a non-negative response");
                }
            }
            _ => {}
        }
        Ok(())
    }

    /// Run `fit` with the GLM family of this regressor. For a negative-binomial
    /// GLMM with an unknown size θ, θ is estimated by maximising the profiled
    /// Laplace log-likelihood over `log θ` (as `lme4::glmer.nb`).
    fn with_family<F>(&self, fit: F) -> Result<FittedGlmm, RegressionError>
    where
        F: Fn(&dyn GlmFamily, f64) -> Result<FittedGlmm, RegressionError>,
    {
        match self.kind {
            ResponseKind::Gaussian => unreachable!("Gaussian is fitted as an LMM"),
            ResponseKind::Poisson => fit(&PoissonFamily::new(PoissonLink::Log), f64::NAN),
            ResponseKind::Binomial => fit(&BinomialFamily::new(BinomialLink::Logit), f64::NAN),
            ResponseKind::Gamma => fit(&TweedieFamily::new(2.0, 0.0), f64::NAN),
            ResponseKind::Tweedie(power) => fit(&TweedieFamily::new(power, 0.0), f64::NAN),
            ResponseKind::NegativeBinomial => {
                let at = |th: f64| fit(&NegativeBinomialFamily::new(th), th);
                if let Some(th) = self.nb_theta {
                    return at(th);
                }
                // −2 log L is minimised over log θ ∈ [log 1e-3, log 1e6].
                let objective = |lt: f64| -> Option<f64> {
                    at(lt.exp())
                        .ok()
                        .map(|m| m.deviance)
                        .filter(|d| d.is_finite())
                };
                let lt = golden_section_min(
                    (1e-3f64).ln(),
                    (1e6f64).ln(),
                    self.tolerance.max(1e-9),
                    &objective,
                )?;
                at(lt.exp())
            }
        }
    }

    // ----------------------------------------------------------------- LMM ---

    /// Fit the Gaussian linear mixed model by profiling the deviance over θ.
    #[allow(clippy::too_many_arguments)]
    fn fit_lmm(
        &self,
        design: &Mat<f64>,
        z: &Mat<f64>,
        y: &Col<f64>,
        group_idx: &[usize],
        n_groups: usize,
        p: usize,
        q: usize,
    ) -> Result<FittedGlmm, RegressionError> {
        let n = design.nrows();
        let y_vec: Vec<f64> = (0..n).map(|i| y[i]).collect();
        let stats = GroupStats::new(design, z, &y_vec, group_idx, n_groups, p, q, None);

        let objective = |theta: &[f64]| -> Option<f64> {
            let t = build_t(theta, q);
            solve_lmm_theta(&t, design, z, &y_vec, group_idx, &stats, p, q).map(|s| {
                if self.reml {
                    s.reml_deviance(n, p)
                } else {
                    s.ml_deviance(n)
                }
            })
        };

        let theta = optimize_theta(q, self.theta_max, self.tolerance, &objective)?;
        let t = build_t(&theta, q);
        let sol = solve_lmm_theta(&t, design, z, &y_vec, group_idx, &stats, p, q)
            .ok_or(RegressionError::SingularMatrix)?;

        let dof = if self.reml { n - p } else { n };
        let sigma2 = sol.prss / dof as f64;
        let sigma = sigma2.sqrt();
        // Σ = σ² T Tᵀ.
        let cov = scale_matrix(&mm(&t, &transpose(&t)), sigma2);
        let cov_beta = scale_matrix(&sol.m_inv, sigma2);
        let deviance = if self.reml {
            sol.reml_deviance(n, p)
        } else {
            sol.ml_deviance(n)
        };

        Ok(FittedGlmm::new(
            self.with_intercept,
            sol.beta,
            diag_sqrt(&cov_beta),
            sol.b,
            n_groups,
            q,
            t[0][0],
            sigma,
            cov,
            deviance,
            self.reml,
            true,
            1,
        ))
    }

    // ---------------------------------------------------------------- GLMM ---

    /// The Laplace deviance is relative to the saturated model; add the
    /// saturated log-likelihood `Σ log f(yᵢ | μᵢ = yᵢ)` so that
    /// `log_likelihood` matches lme4's `logLik(glmer)` (zero for 0/1 data).
    /// The negative-binomial, Gamma and Tweedie criteria already use the full
    /// family log-likelihood (`log_likelihood = −deviance / 2`).
    fn with_saturated_term(&self, mut fitted: FittedGlmm, y: &Col<f64>) -> FittedGlmm {
        let yv: Vec<f64> = y.iter().copied().collect();
        let sat = match self.kind {
            ResponseKind::Poisson => super::fit_stats::poisson_saturated_log_likelihood(&yv),
            ResponseKind::Binomial => {
                super::fit_stats::binomial_saturated_log_likelihood(&yv, None)
            }
            _ => return fitted,
        };
        fitted.log_likelihood = -0.5 * fitted.deviance + sat;
        fitted
    }

    /// The Laplace criterion `log|L_θ|² + ‖u‖² + lik(y, μ)` at the conditional
    /// mode (lme4's `ldL2 + sqrL + aic`). For Poisson / binomial `lik` is the
    /// summed unit deviance (the saturated term is added afterwards); for the
    /// other families it is `−2 log L` (+2 for the Gamma / Tweedie dispersion,
    /// estimated as `deviance / n` as in R's `Gamma()$aic`).
    fn laplace_criterion(
        &self,
        family: &dyn GlmFamily,
        nb_theta: f64,
        y: &[f64],
        mu: &[f64],
        u_sq: f64,
        logdet_l2: f64,
    ) -> f64 {
        let dev: f64 = y
            .iter()
            .zip(mu)
            .map(|(&yi, &mi)| family.unit_deviance(yi, mi))
            .sum();
        let lik = match self.kind {
            ResponseKind::NegativeBinomial => {
                -2.0 * super::fit_stats::negbin_log_likelihood(y, mu, nb_theta)
            }
            ResponseKind::Gamma => {
                let dispersion = dev / y.len() as f64;
                -2.0 * family_log_likelihood(LogLikKind::Gamma { dispersion }, y, mu) + 2.0
            }
            ResponseKind::Tweedie(power) => {
                let dispersion = dev / y.len() as f64;
                -2.0 * family_log_likelihood(LogLikKind::Tweedie { power, dispersion }, y, mu) + 2.0
            }
            _ => dev,
        };
        logdet_l2 + u_sq + lik
    }

    /// Residual scale σ and the standard-error multiplier. Families with a free
    /// dispersion (Gamma, Tweedie) use lme4's `σ = sqrt(pwrss / n)` with
    /// `pwrss = Σ (y − μ)²/V(μ) + ‖u‖²`; the others have σ = 1.
    fn glmm_scale(&self, family: &dyn GlmFamily, y: &[f64], mu: &[f64], u_sq: f64) -> f64 {
        match self.kind {
            ResponseKind::Gamma | ResponseKind::Tweedie(_) => {
                let pearson: f64 = y
                    .iter()
                    .zip(mu)
                    .map(|(&yi, &mi)| (yi - mi).powi(2) / family.variance(mi))
                    .sum();
                ((pearson + u_sq) / y.len() as f64).sqrt()
            }
            _ => 1.0,
        }
    }

    /// Fit a non-Gaussian GLMM by profiling the Laplace deviance over θ, with a
    /// PIRLS inner loop finding the conditional modes at each θ.
    #[allow(clippy::too_many_arguments)]
    fn fit_glmm(
        &self,
        family: &dyn GlmFamily,
        nb_theta: f64,
        design: &Mat<f64>,
        z: &Mat<f64>,
        y: &[f64],
        offset: &[f64],
        group_idx: &[usize],
        n_groups: usize,
        p: usize,
        q: usize,
    ) -> Result<FittedGlmm, RegressionError> {
        let beta0 = glm_warm_start(family, design, y, offset, p, self.max_iterations)?;
        let run = |t: &[Vec<f64>]| {
            pirls(
                family,
                design,
                z,
                y,
                offset,
                group_idx,
                n_groups,
                p,
                q,
                t,
                &beta0,
                self.max_iterations,
                self.tolerance,
            )
        };

        let objective = |theta: &[f64]| -> Option<f64> {
            let t = build_t(theta, q);
            run(&t).map(|s| self.laplace_criterion(family, nb_theta, y, &s.mu, s.u_sq, s.logdet_l2))
        };

        let theta = optimize_theta(q, self.theta_max, self.tolerance, &objective)?;
        let t = build_t(&theta, q);
        let sol = run(&t).ok_or(RegressionError::ConvergenceFailed {
            iterations: self.max_iterations,
        })?;
        let deviance =
            self.laplace_criterion(family, nb_theta, y, &sol.mu, sol.u_sq, sol.logdet_l2);
        let sigma = self.glmm_scale(family, y, &sol.mu, sol.u_sq);

        // Σ = T Tᵀ (lme4 reports the GLMM random-effect covariance unscaled);
        // Var(β) = σ² M⁻¹ with σ = 1 except for Gamma / Tweedie.
        let cov = mm(&t, &transpose(&t));
        let std_errors = diag_sqrt(&sol.m_inv).iter().map(|s| s * sigma).collect();

        let mut fitted = FittedGlmm::new(
            self.with_intercept,
            sol.beta,
            std_errors,
            sol.b,
            n_groups,
            q,
            t[0][0],
            sigma,
            cov,
            deviance,
            false,
            sol.converged,
            sol.iterations,
        );
        fitted.nb_theta = nb_theta.is_finite().then_some(nb_theta);
        Ok(fitted)
    }

    // -------------------------------------------------------- multi-factor ---

    /// Fit a Gaussian LMM with several crossed / nested random-effect factors
    /// by profiling the REML/ML deviance over the stacked per-factor θ.
    fn fit_lmm_multi(
        &self,
        design: &Mat<f64>,
        y: &[f64],
        spec: &MultiSpec,
        p: usize,
    ) -> Result<FittedGlmm, RegressionError> {
        let n = design.nrows();
        let m_dim = spec.m_dim;
        let solve = |theta: &[f64]| -> Option<(MultiSolution, f64)> {
            let ts = spec.ts(theta);
            let zl = spec.zl(&ts);
            let st = MultiStats::new(design, y, &zl, m_dim, p, None);
            let s = solve_multi(&st, m_dim, p)?;
            let rss = multi_rss(design, y, &zl, &s.beta, &s.u, p);
            Some((s, rss))
        };

        let objective = |theta: &[f64]| -> Option<f64> {
            solve(theta).map(|(s, rss)| {
                deviance_from_pieces(rss + s.u_sq, s.logdet_l2, s.logdet_rx2, n, p, self.reml)
            })
        };

        let theta = optimize_theta_vec(&spec.initial_theta(), self.tolerance, &objective)?;
        let (sol, rss) = solve(&theta).ok_or(RegressionError::SingularMatrix)?;
        let prss = rss + sol.u_sq;

        let dof = if self.reml { n - p } else { n };
        let sigma2 = prss / dof as f64;
        let sigma = sigma2.sqrt();
        let std_errors = diag_sqrt(&scale_matrix(&sol.m_inv, sigma2));
        let deviance = deviance_from_pieces(prss, sol.logdet_l2, sol.logdet_rx2, n, p, self.reml);

        let ts = spec.ts(&theta);
        let (factors, details) = spec.summaries(&ts, &sol.u, sigma);
        Ok(FittedGlmm::new_multi(
            self.with_intercept,
            sol.beta,
            std_errors,
            sigma,
            factors,
            details,
            deviance,
            self.reml,
            true,
            1,
        ))
    }

    /// Fit a non-Gaussian GLMM with several crossed / nested random-effect
    /// factors by profiling the Laplace deviance, PIRLS in the inner loop.
    #[allow(clippy::too_many_arguments)]
    fn fit_glmm_multi(
        &self,
        family: &dyn GlmFamily,
        nb_theta: f64,
        design: &Mat<f64>,
        y: &[f64],
        offset: &[f64],
        spec: &MultiSpec,
        p: usize,
    ) -> Result<FittedGlmm, RegressionError> {
        let beta0 = glm_warm_start(family, design, y, offset, p, self.max_iterations)?;
        let run = |theta: &[f64]| {
            pirls_multi(
                family,
                design,
                y,
                offset,
                spec,
                theta,
                &beta0,
                self.max_iterations,
                self.tolerance,
            )
        };

        let objective = |theta: &[f64]| -> Option<f64> {
            run(theta)
                .map(|s| self.laplace_criterion(family, nb_theta, y, &s.mu, s.u_sq, s.logdet_l2))
        };

        let theta = optimize_theta_vec(&spec.initial_theta(), self.tolerance, &objective)?;
        let sol = run(&theta).ok_or(RegressionError::ConvergenceFailed {
            iterations: self.max_iterations,
        })?;
        let deviance =
            self.laplace_criterion(family, nb_theta, y, &sol.mu, sol.u_sq, sol.logdet_l2);
        let sigma = self.glmm_scale(family, y, &sol.mu, sol.u_sq);
        let std_errors = diag_sqrt(&sol.m_inv).iter().map(|s| s * sigma).collect();

        let ts = spec.ts(&theta);
        // GLMM random-effect covariances are unscaled (lme4 convention).
        let (factors, details) = spec.summaries(&ts, &sol.u, 1.0);
        let mut fitted = FittedGlmm::new_multi(
            self.with_intercept,
            sol.beta,
            std_errors,
            sigma,
            factors,
            details,
            deviance,
            false,
            sol.converged,
            sol.iterations,
        );
        fitted.nb_theta = nb_theta.is_finite().then_some(nb_theta);
        Ok(fitted)
    }
}

/// Check that every random-slope column indexes into `x`.
fn check_slope_columns(slopes: &[usize], ncols: usize) -> Result<(), RegressionError> {
    for &c in slopes {
        if c >= ncols {
            return Err(RegressionError::NumericalError(format!(
                "random-slope column {c} is out of range for x with {ncols} columns"
            )));
        }
    }
    Ok(())
}

/// A fitted GLMM with a random intercept and optional random slopes.
pub struct FittedGlmm {
    with_intercept: bool,
    fixed_effects: Vec<f64>,
    std_errors: Vec<f64>,
    /// Random-effects BLUPs, `n_groups × q` (row-major per group).
    re_matrix: Vec<Vec<f64>>,
    /// Random intercept BLUP per group (component 0), for the scalar accessor.
    re_intercept: Vec<f64>,
    n_groups: usize,
    q: usize,
    /// `T[0][0]` — for a random intercept this is the ratio θ = σ_b/σ.
    theta0: f64,
    sigma: f64,
    /// Random-effects covariance Σ (`q × q`).
    cov: Vec<Vec<f64>>,
    deviance: f64,
    log_likelihood: f64,
    reml: bool,
    converged: bool,
    iterations: usize,
    /// One entry per grouping factor when the fit has **multiple** random-effect
    /// factors (crossed / nested). Empty for the single-factor path, whose
    /// random effects are described by `cov` / `re_matrix`.
    factors: Vec<FactorSummary>,
    /// Per-factor covariance and full BLUP matrices (multi-factor path).
    factor_details: Vec<FactorDetail>,
    /// Negative-binomial size θ (fixed or estimated); `None` otherwise.
    nb_theta: Option<f64>,
}

/// Per-factor summary for a crossed / nested fit: one random intercept per
/// factor, each with its own variance component.
#[derive(Debug, Clone)]
pub struct FactorSummary {
    /// Number of levels of this grouping factor.
    pub n_levels: usize,
    /// Random-intercept standard deviation σ_f for this factor.
    pub sd: f64,
    /// Random-intercept BLUP per level (0-based).
    pub blups: Vec<f64>,
}

impl FittedGlmm {
    #[allow(clippy::too_many_arguments)]
    fn new(
        with_intercept: bool,
        fixed_effects: Vec<f64>,
        std_errors: Vec<f64>,
        re_matrix: Vec<Vec<f64>>,
        n_groups: usize,
        q: usize,
        theta0: f64,
        sigma: f64,
        cov: Vec<Vec<f64>>,
        deviance: f64,
        reml: bool,
        converged: bool,
        iterations: usize,
    ) -> Self {
        let re_intercept = re_matrix.iter().map(|b| b[0]).collect();
        Self {
            with_intercept,
            fixed_effects,
            std_errors,
            re_matrix,
            re_intercept,
            n_groups,
            q,
            theta0,
            sigma,
            cov,
            deviance,
            log_likelihood: -0.5 * deviance,
            reml,
            converged,
            iterations,
            factors: Vec::new(),
            factor_details: Vec::new(),
            nb_theta: None,
        }
    }

    /// Constructor for the multiple-factor (crossed / nested) path. The scalar
    /// single-factor accessors mirror the first factor for convenience; use
    /// [`factors`](Self::factors) for the full per-factor breakdown.
    #[allow(clippy::too_many_arguments)]
    fn new_multi(
        with_intercept: bool,
        fixed_effects: Vec<f64>,
        std_errors: Vec<f64>,
        sigma: f64,
        factors: Vec<FactorSummary>,
        factor_details: Vec<FactorDetail>,
        deviance: f64,
        reml: bool,
        converged: bool,
        iterations: usize,
    ) -> Self {
        let f0 = &factors[0];
        let d0 = &factor_details[0];
        let re_matrix = d0.re.clone();
        let re_intercept = f0.blups.clone();
        let cov = d0.cov.clone();
        let q = cov.len();
        let theta0 = if sigma > 0.0 { f0.sd / sigma } else { 0.0 };
        Self {
            with_intercept,
            fixed_effects,
            std_errors,
            re_matrix,
            re_intercept,
            n_groups: f0.n_levels,
            q,
            theta0,
            sigma,
            cov,
            deviance,
            log_likelihood: -0.5 * deviance,
            reml,
            converged,
            iterations,
            factors,
            factor_details,
            nb_theta: None,
        }
    }

    /// Fixed-effect coefficients (element 0 is the intercept when present).
    pub fn fixed_effects(&self) -> &[f64] {
        &self.fixed_effects
    }

    /// Standard errors of the fixed effects.
    pub fn std_errors(&self) -> &[f64] {
        &self.std_errors
    }

    /// Wald z statistics `β̂ / SE` of the fixed effects (element 0 is the
    /// intercept when present), as in the `z value` / `t value` column of
    /// `summary()` for `lme4::glmer` / `lmer`. `NaN` where the SE is not a
    /// positive finite number.
    pub fn z_values(&self) -> Vec<f64> {
        self.fixed_effects
            .iter()
            .zip(&self.std_errors)
            .map(|(&b, &se)| {
                if se.is_finite() && se > 0.0 {
                    b / se
                } else {
                    f64::NAN
                }
            })
            .collect()
    }

    /// Two-sided asymptotic normal p-values `2·(1 − Φ(|z|))` of the fixed
    /// effects — `Pr(>|z|)` of `summary(glmer(...))`. For a Gaussian LMM this
    /// is the large-sample normal approximation (lme4 prints no p-value for
    /// `lmer`; Satterthwaite/Kenward–Roger degrees of freedom are not used).
    pub fn p_values(&self) -> Vec<f64> {
        let normal = Normal::new(0.0, 1.0).expect("standard normal");
        self.z_values()
            .into_iter()
            .map(|z| {
                if z.is_finite() {
                    2.0 * normal.sf(z.abs())
                } else {
                    f64::NAN
                }
            })
            .collect()
    }

    /// Wald confidence intervals `β̂ ∓ z_{(1+level)/2}·SE` for the fixed
    /// effects at confidence `level` (e.g. `0.95`), returned as
    /// `(lower, upper)` — `confint(m, parm = "beta_", method = "Wald", level)`
    /// in lme4. Bounds are `NaN` for an invalid `level` (outside `(0, 1)`) or
    /// a non-finite SE.
    pub fn conf_int(&self, level: f64) -> (Vec<f64>, Vec<f64>) {
        let k = self.fixed_effects.len();
        if !(level > 0.0 && level < 1.0) {
            return (vec![f64::NAN; k], vec![f64::NAN; k]);
        }
        let q = Normal::new(0.0, 1.0)
            .expect("standard normal")
            .inverse_cdf(0.5 + level / 2.0);
        self.fixed_effects
            .iter()
            .zip(&self.std_errors)
            .map(|(&b, &se)| {
                if se.is_finite() && se >= 0.0 {
                    (b - q * se, b + q * se)
                } else {
                    (f64::NAN, f64::NAN)
                }
            })
            .unzip()
    }

    /// The intercept, if the model was fit with one.
    pub fn intercept(&self) -> Option<f64> {
        if self.with_intercept {
            self.fixed_effects.first().copied()
        } else {
            None
        }
    }

    /// The non-intercept fixed-effect slopes.
    pub fn slopes(&self) -> &[f64] {
        if self.with_intercept {
            &self.fixed_effects[1..]
        } else {
            &self.fixed_effects
        }
    }

    /// The random **intercept** BLUP for each group (0-based) — component 0 of
    /// the random effects. See [`random_effects_matrix`](Self::random_effects_matrix)
    /// for the full per-group vectors when random slopes are present.
    pub fn random_effects(&self) -> &[f64] {
        &self.re_intercept
    }

    /// The full random-effects BLUPs, one `q`-vector per group (`n_groups × q`).
    /// Component 0 is the random intercept; components `1..` are the random
    /// slopes in the order given to
    /// [`random_slopes`](GlmmRegressorBuilder::random_slopes).
    pub fn random_effects_matrix(&self) -> &[Vec<f64>] {
        &self.re_matrix
    }

    /// Number of random effects per group `q` (1 for a plain random intercept).
    pub fn n_random_effects(&self) -> usize {
        self.q
    }

    /// The random-effects covariance matrix Σ (`q × q`).
    pub fn random_cov(&self) -> &[Vec<f64>] {
        &self.cov
    }

    /// Standard deviations of the random effects — `sqrt(diag Σ)`.
    pub fn random_sd(&self) -> Vec<f64> {
        (0..self.q)
            .map(|i| self.cov[i][i].max(0.0).sqrt())
            .collect()
    }

    /// Correlation matrix of the random effects (`q × q`).
    pub fn random_corr(&self) -> Vec<Vec<f64>> {
        let sd = self.random_sd();
        let mut c = vec![vec![0.0; self.q]; self.q];
        for i in 0..self.q {
            for j in 0..self.q {
                let d = sd[i] * sd[j];
                c[i][j] = if d > 0.0 { self.cov[i][j] / d } else { 0.0 };
            }
        }
        c
    }

    /// For a random intercept, the profiled ratio θ = σ_b/σ (the `T[0][0]`
    /// entry of the relative covariance factor).
    pub fn theta(&self) -> f64 {
        self.theta0
    }

    /// Residual standard deviation σ: 1.0 for the Poisson, binomial and
    /// negative-binomial families; lme4's `sigma()` = `sqrt(pwrss / n)` for
    /// Gamma / Tweedie (it scales the fixed-effect standard errors).
    pub fn sigma(&self) -> f64 {
        self.sigma
    }

    /// Random-**intercept** standard deviation σ_b = `sqrt(Σ[0][0])`.
    pub fn sd_random(&self) -> f64 {
        self.cov[0][0].max(0.0).sqrt()
    }

    /// Random-**intercept** variance σ_b² = `Σ[0][0]`.
    pub fn var_random(&self) -> f64 {
        self.cov[0][0]
    }

    /// Deviance (−2 log-likelihood); the REML criterion for a REML Gaussian fit.
    pub fn deviance(&self) -> f64 {
        self.deviance
    }

    /// Log-likelihood. For the Gaussian LMM it is `−deviance / 2` (the REML
    /// criterion for a REML fit); for Poisson / binomial GLMMs it is the
    /// Laplace-approximate log-likelihood including the saturated term,
    /// `−deviance/2 + Σ log f(yᵢ | μᵢ = yᵢ)`, as lme4's `logLik(glmer)`.
    pub fn log_likelihood(&self) -> f64 {
        self.log_likelihood
    }

    /// Number of groups (levels of the grouping factor).
    pub fn n_groups(&self) -> usize {
        self.n_groups
    }

    /// Per-factor summaries for a crossed / nested fit (from
    /// [`fit_crossed`](GlmmRegressor::fit_crossed)); empty for a single-factor
    /// fit.
    pub fn factors(&self) -> &[FactorSummary] {
        &self.factors
    }

    /// Number of grouping factors: `factors().len()` for a crossed / nested fit,
    /// otherwise 1.
    pub fn n_factors(&self) -> usize {
        if self.factors.is_empty() {
            1
        } else {
            self.factors.len()
        }
    }

    /// Random-effects covariance `Σ_f` (`q_f × q_f`) of grouping factor `f`
    /// in a crossed / nested fit — component 0 is the random intercept, then
    /// the slopes given to
    /// [`random_slopes_per_factor`](GlmmRegressorBuilder::random_slopes_per_factor).
    /// `None` for a single-factor fit (use [`random_cov`](Self::random_cov)) or
    /// an out-of-range `f`.
    pub fn factor_random_cov(&self, f: usize) -> Option<&[Vec<f64>]> {
        self.factor_details.get(f).map(|d| d.cov.as_slice())
    }

    /// Standard deviations `sqrt(diag Σ_f)` of grouping factor `f` in a crossed
    /// / nested fit; `None` for a single-factor fit or an out-of-range `f`.
    pub fn factor_random_sd(&self, f: usize) -> Option<Vec<f64>> {
        self.factor_details.get(f).map(|d| {
            (0..d.cov.len())
                .map(|i| d.cov[i][i].max(0.0).sqrt())
                .collect()
        })
    }

    /// Full BLUPs of grouping factor `f` in a crossed / nested fit, one
    /// `q_f`-vector per level (`n_levels × q_f`); `None` for a single-factor
    /// fit or an out-of-range `f`.
    pub fn factor_random_effects_matrix(&self, f: usize) -> Option<&[Vec<f64>]> {
        self.factor_details.get(f).map(|d| d.re.as_slice())
    }

    /// The negative-binomial size θ (`Var = μ + μ²/θ`) — estimated, or as
    /// fixed via [`nb_theta`](GlmmRegressorBuilder::nb_theta). `None` for the
    /// other families.
    pub fn nb_theta(&self) -> Option<f64> {
        self.nb_theta
    }

    /// Whether the fit used REML (Gaussian only).
    pub fn is_reml(&self) -> bool {
        self.reml
    }

    /// Whether the fit converged.
    pub fn converged(&self) -> bool {
        self.converged
    }

    /// PIRLS iterations at the optimum (1 for the closed-form Gaussian LMM).
    pub fn iterations(&self) -> usize {
        self.iterations
    }

    /// Population-level linear predictor `η = x'β + offset` for new data
    /// (random effects set to zero).
    pub fn predict_fixed_with_offset(&self, x: &Mat<f64>, offset: &Col<f64>) -> Col<f64> {
        let eta = self.predict_fixed(x);
        Col::from_fn(eta.nrows(), |i| eta[i] + offset[i])
    }

    /// Population-level linear predictor `η = x'β` for new data (random effects
    /// set to zero — the marginal / new-group prediction).
    pub fn predict_fixed(&self, x: &Mat<f64>) -> Col<f64> {
        let design = build_design(x, self.with_intercept);
        let p = design.ncols();
        Col::from_fn(design.nrows(), |i| {
            (0..p).map(|k| design[(i, k)] * self.fixed_effects[k]).sum()
        })
    }
}

// ============================================================ builder =======

/// Which response family a [`GlmmRegressorBuilder`] targets.
#[derive(Debug, Clone, Copy)]
enum ResponseKind {
    Gaussian,
    Poisson,
    Binomial,
    NegativeBinomial,
    Gamma,
    /// Tweedie with the given variance power.
    Tweedie(f64),
}

/// Builder for [`GlmmRegressor`].
pub struct GlmmRegressorBuilder {
    kind: ResponseKind,
    with_intercept: bool,
    random_intercept: bool,
    random_slopes: Vec<usize>,
    random_slopes_per_factor: Option<Vec<Vec<usize>>>,
    reml: bool,
    max_iterations: usize,
    tolerance: f64,
    theta_max: f64,
    nb_theta: Option<f64>,
    offset: Option<Col<f64>>,
}

impl GlmmRegressorBuilder {
    fn new(kind: ResponseKind) -> Self {
        Self {
            kind,
            with_intercept: true,
            random_intercept: true,
            random_slopes: Vec::new(),
            random_slopes_per_factor: None,
            reml: true,
            max_iterations: 100,
            tolerance: 1e-8,
            theta_max: 1000.0,
            nb_theta: None,
            offset: None,
        }
    }

    /// Whether to include a fixed-effect intercept (default `true`).
    pub fn with_intercept(mut self, include: bool) -> Self {
        self.with_intercept = include;
        self
    }

    /// Whether the random-effects design includes a random intercept
    /// (default `true`). Set `false` only together with at least one random
    /// slope.
    pub fn random_intercept(mut self, include: bool) -> Self {
        self.random_intercept = include;
        self
    }

    /// Columns of `x` (0-based) that carry a random slope alongside the random
    /// intercept — i.e. `(1 + x[c] | group)`. The random-effects covariance is
    /// unstructured over the intercept and these slopes.
    pub fn random_slopes(mut self, cols: Vec<usize>) -> Self {
        self.random_slopes = cols;
        self
    }

    /// Random-slope columns of `x` **per grouping factor** for
    /// [`fit_crossed`](GlmmRegressor::fit_crossed): entry `f` lists the
    /// columns with a random slope on factor `f`, e.g.
    /// `vec![vec![0], vec![]]` for `(1 + x[0] | a) + (1 | b)`. Must have one
    /// entry per grouping factor; exclusive with
    /// [`random_slopes`](Self::random_slopes).
    pub fn random_slopes_per_factor(mut self, cols: Vec<Vec<usize>>) -> Self {
        self.random_slopes_per_factor = Some(cols);
        self
    }

    /// Fix the negative-binomial size θ (`Var = μ + μ²/θ`) instead of
    /// estimating it. Only used by [`GlmmRegressor::negative_binomial`].
    pub fn nb_theta(mut self, theta: f64) -> Self {
        self.nb_theta = Some(theta);
        self
    }

    /// Per-observation offset added to the linear predictor with a fixed
    /// coefficient of 1 (e.g. `log(exposure)` for counts), as `offset()` in an
    /// lme4 formula. Its length must equal the number of observations.
    pub fn offset(mut self, offset: Col<f64>) -> Self {
        self.offset = Some(offset);
        self
    }

    /// Use REML (`true`, default) or ML (`false`) for the Gaussian case.
    pub fn reml(mut self, reml: bool) -> Self {
        self.reml = reml;
        self
    }

    /// Maximum PIRLS iterations for the GLMM inner loop (default `100`).
    pub fn max_iterations(mut self, max_iter: usize) -> Self {
        self.max_iterations = max_iter;
        self
    }

    /// Convergence tolerance (default `1e-8`).
    pub fn tolerance(mut self, tol: f64) -> Self {
        self.tolerance = tol;
        self
    }

    /// Upper bound of the profiled ratio θ = σ_b/σ searched in the
    /// random-intercept case (default `1000`).
    pub fn theta_max(mut self, theta_max: f64) -> Self {
        self.theta_max = theta_max;
        self
    }

    /// Build the [`GlmmRegressor`].
    pub fn build(self) -> GlmmRegressor {
        GlmmRegressor {
            kind: self.kind,
            with_intercept: self.with_intercept,
            random_intercept: self.random_intercept,
            random_slopes: self.random_slopes,
            random_slopes_per_factor: self.random_slopes_per_factor,
            reml: self.reml,
            max_iterations: self.max_iterations,
            tolerance: self.tolerance,
            theta_max: self.theta_max,
            nb_theta: self.nb_theta,
            offset: self.offset,
        }
    }
}

// ==================================================== per-group statistics ===

/// Per-group cross-products of the fixed (`X`) and random (`Z`) designs.
///
/// For the LMM these are computed once (unit weights); the GLMM recomputes them
/// each PIRLS iteration with the current working weights.
struct GroupStats {
    /// `Zⱼᵀ W Zⱼ` per group (`q×q`).
    ztz: Vec<Vec<Vec<f64>>>,
    /// `Zⱼᵀ W Xⱼ` per group (`q×p`).
    ztx: Vec<Vec<Vec<f64>>>,
    /// `Zⱼᵀ W yⱼ` (or working response) per group (`q`).
    zty: Vec<Vec<f64>>,
    /// `Xᵀ W X` (`p×p`).
    xtx: Vec<Vec<f64>>,
    /// `Xᵀ W y` (or working response) (`p`).
    xty: Vec<f64>,
}

impl GroupStats {
    #[allow(clippy::too_many_arguments)]
    fn new(
        design: &Mat<f64>,
        z: &Mat<f64>,
        y: &[f64],
        group_idx: &[usize],
        n_groups: usize,
        p: usize,
        q: usize,
        weights: Option<&[f64]>,
    ) -> Self {
        let n = design.nrows();
        let mut ztz = vec![vec![vec![0.0; q]; q]; n_groups];
        let mut ztx = vec![vec![vec![0.0; p]; q]; n_groups];
        let mut zty = vec![vec![0.0; q]; n_groups];
        let mut xtx = vec![vec![0.0; p]; p];
        let mut xty = vec![0.0; p];

        for i in 0..n {
            let w = weights.map_or(1.0, |w| w[i]);
            let g = group_idx[i];
            let yi = y[i];
            for a in 0..q {
                let wza = w * z[(i, a)];
                zty[g][a] += wza * yi;
                for b in 0..q {
                    ztz[g][a][b] += wza * z[(i, b)];
                }
                for k in 0..p {
                    ztx[g][a][k] += wza * design[(i, k)];
                }
            }
            for k in 0..p {
                let wx = w * design[(i, k)];
                xty[k] += wx * yi;
                for l in 0..p {
                    xtx[k][l] += wx * design[(i, l)];
                }
            }
        }
        Self {
            ztz,
            ztx,
            zty,
            xtx,
            xty,
        }
    }
}

// ================================================= LMM θ-solve internals =====

/// The solution of the penalized least-squares problem at a fixed `T` (LMM).
struct LmmThetaSolution {
    beta: Vec<f64>,
    /// Random-effects BLUPs bⱼ = T uⱼ, one `q`-vector per group.
    b: Vec<Vec<f64>>,
    /// Penalized residual sum of squares r²(θ) = ‖y − Xβ − Zb‖² + ‖u‖².
    prss: f64,
    /// log|L_θ|² = Σⱼ log det(Aⱼ).
    logdet_l2: f64,
    /// log det of the fixed-effects Schur complement M.
    logdet_rx2: f64,
    /// M⁻¹, used to scale the fixed-effect covariance.
    m_inv: Vec<Vec<f64>>,
}

impl LmmThetaSolution {
    fn reml_deviance(&self, n: usize, p: usize) -> f64 {
        let dof = (n - p) as f64;
        self.logdet_l2
            + self.logdet_rx2
            + dof * (1.0 + (2.0 * std::f64::consts::PI * self.prss / dof).ln())
    }

    fn ml_deviance(&self, n: usize) -> f64 {
        let nf = n as f64;
        self.logdet_l2 + nf * (1.0 + (2.0 * std::f64::consts::PI * self.prss / nf).ln())
    }
}

/// Solve the penalized least-squares problem for a fixed relative factor `T`.
/// `None` on a singular fixed-effects or random block.
#[allow(clippy::too_many_arguments)]
fn solve_lmm_theta(
    t: &[Vec<f64>],
    design: &Mat<f64>,
    z: &Mat<f64>,
    y: &[f64],
    group_idx: &[usize],
    stats: &GroupStats,
    p: usize,
    q: usize,
) -> Option<LmmThetaSolution> {
    let n = design.nrows();
    let j = stats.ztz.len();
    let tt = transpose(t);

    // Fixed-effects Schur complement M and rhs, plus per-group A⁻¹B, A⁻¹c.
    let mut m = stats.xtx.clone();
    let mut rhs = stats.xty.clone();
    let mut logdet_l2 = 0.0;
    // Cache the pieces needed to back-substitute the random effects.
    let mut aib_all: Vec<Vec<Vec<f64>>> = Vec::with_capacity(j); // q×p per group
    let mut aic_all: Vec<Vec<f64>> = Vec::with_capacity(j); // q per group

    for jj in 0..j {
        // Aⱼ = Tᵀ (ZᵀZ)ⱼ T + I ; Bⱼ = Tᵀ (ZᵀX)ⱼ ; cⱼ = Tᵀ (Zᵀy)ⱼ.
        let a = add_identity(&mm(&tt, &mm(&stats.ztz[jj], t)));
        let bmat = mm(&tt, &stats.ztx[jj]);
        let c = matvec(&tt, &stats.zty[jj]);

        let chol = cholesky(&a)?;
        logdet_l2 += 2.0 * (0..q).map(|i| chol[i][i].ln()).sum::<f64>();

        let aib = chol_solve_mat(&chol, &bmat); // q×p
        let aic = cholesky_solve(&chol, &c); // q

        // M -= Bᵀ A⁻¹ B ; rhs -= Bᵀ A⁻¹ c.
        let btaib = mm(&transpose(&bmat), &aib);
        let btaic = matvec(&transpose(&bmat), &aic);
        for k in 0..p {
            rhs[k] -= btaic[k];
            for l in 0..p {
                m[k][l] -= btaib[k][l];
            }
        }
        aib_all.push(aib);
        aic_all.push(aic);
    }

    let chol_m = cholesky(&m)?;
    let logdet_rx2 = 2.0 * (0..p).map(|i| chol_m[i][i].ln()).sum::<f64>();
    let beta = cholesky_solve(&chol_m, &rhs);
    let m_inv = cholesky_inverse(&chol_m);

    // Back-substitute uⱼ = A⁻¹c − (A⁻¹B) β, then bⱼ = T uⱼ.
    let mut b = vec![vec![0.0; q]; j];
    let mut u_sq = 0.0;
    for jj in 0..j {
        let mut u = aic_all[jj].clone();
        for (a, urow) in u.iter_mut().enumerate() {
            let dot: f64 = (0..p).map(|k| aib_all[jj][a][k] * beta[k]).sum();
            *urow -= dot;
        }
        u_sq += u.iter().map(|&v| v * v).sum::<f64>();
        b[jj] = matvec(t, &u);
    }

    // Residual sum of squares ‖y − Xβ − Zb‖².
    let mut rss = 0.0;
    for i in 0..n {
        let g = group_idx[i];
        let mut fit = 0.0;
        for k in 0..p {
            fit += design[(i, k)] * beta[k];
        }
        for a in 0..q {
            fit += z[(i, a)] * b[g][a];
        }
        let e = y[i] - fit;
        rss += e * e;
    }

    Some(LmmThetaSolution {
        beta,
        b,
        prss: rss + u_sq,
        logdet_l2,
        logdet_rx2,
        m_inv,
    })
}

// ============================================= GLMM PIRLS internals ==========

/// The converged PIRLS solution at a fixed `T` (GLMM).
struct PirlsSolution {
    beta: Vec<f64>,
    b: Vec<Vec<f64>>,
    /// Fitted means at the conditional mode.
    mu: Vec<f64>,
    /// ‖u‖² at the mode.
    u_sq: f64,
    /// log|L_θ|² at the converged weights.
    logdet_l2: f64,
    m_inv: Vec<Vec<f64>>,
    converged: bool,
    iterations: usize,
}

/// Penalized IRLS: find the conditional modes `(β, u)` for a fixed `T`; the
/// linear predictor is `η = Xβ + Z T u + offset`.
#[allow(clippy::too_many_arguments)]
fn pirls(
    family: &dyn GlmFamily,
    design: &Mat<f64>,
    z: &Mat<f64>,
    y: &[f64],
    offset: &[f64],
    group_idx: &[usize],
    n_groups: usize,
    p: usize,
    q: usize,
    t: &[Vec<f64>],
    beta0: &[f64],
    max_iter: usize,
    tol: f64,
) -> Option<PirlsSolution> {
    let n = design.nrows();
    let tt = transpose(t);

    let mut beta = beta0.to_vec();
    let mut u = vec![vec![0.0; q]; n_groups];
    let mut converged = false;
    let mut iterations = 0;
    let mut last_m_inv = vec![vec![0.0; p]; p];

    let eta_of = |beta: &[f64], u: &[Vec<f64>], i: usize| -> f64 {
        let g = group_idx[i];
        let tu = matvec(t, &u[g]);
        let mut eta = offset[i];
        for k in 0..p {
            eta += design[(i, k)] * beta[k];
        }
        for a in 0..q {
            eta += z[(i, a)] * tu[a];
        }
        eta
    };

    for iter in 0..max_iter {
        iterations = iter + 1;

        // Working response ζ (offset removed) and weights at the current mode.
        let mut zeta = vec![0.0; n];
        let mut weights = vec![0.0; n];
        for i in 0..n {
            let eta = eta_of(&beta, &u, i);
            let mu = family.clamp_mu(family.link_inverse(eta));
            weights[i] = family.irls_weight(mu).max(1e-10);
            zeta[i] = family.working_response(y[i], mu, eta) - offset[i];
        }

        // Per-group cross-products at the current working response and weights.
        let stats = GroupStats::new(design, z, &zeta, group_idx, n_groups, p, q, Some(&weights));

        // Schur complement over the fixed effects.
        let mut m = stats.xtx.clone();
        let mut rhs = stats.xty.clone();
        let mut aib_all = Vec::with_capacity(n_groups);
        let mut aic_all = Vec::with_capacity(n_groups);
        for g in 0..n_groups {
            let a = add_identity(&mm(&tt, &mm(&stats.ztz[g], t)));
            let bmat = mm(&tt, &stats.ztx[g]);
            let c = matvec(&tt, &stats.zty[g]);
            let chol = cholesky(&a)?;
            let aib = chol_solve_mat(&chol, &bmat);
            let aic = cholesky_solve(&chol, &c);
            let btaib = mm(&transpose(&bmat), &aib);
            let btaic = matvec(&transpose(&bmat), &aic);
            for k in 0..p {
                rhs[k] -= btaic[k];
                for l in 0..p {
                    m[k][l] -= btaib[k][l];
                }
            }
            aib_all.push(aib);
            aic_all.push(aic);
        }

        let chol_m = cholesky(&m)?;
        let beta_new = cholesky_solve(&chol_m, &rhs);
        last_m_inv = cholesky_inverse(&chol_m);

        let mut u_new = vec![vec![0.0; q]; n_groups];
        let mut max_change = 0.0f64;
        for g in 0..n_groups {
            let mut ug = aic_all[g].clone();
            for (a, urow) in ug.iter_mut().enumerate() {
                let dot: f64 = (0..p).map(|k| aib_all[g][a][k] * beta_new[k]).sum();
                *urow -= dot;
                max_change = max_change.max((*urow - u[g][a]).abs());
            }
            u_new[g] = ug;
        }
        for k in 0..p {
            max_change = max_change.max((beta_new[k] - beta[k]).abs());
        }
        beta = beta_new;
        u = u_new;

        if max_change < tol {
            converged = true;
            break;
        }
    }

    // Pieces of the Laplace criterion at the mode.
    let mut mu = vec![0.0; n];
    let mut weights = vec![0.0; n];
    for i in 0..n {
        let eta = eta_of(&beta, &u, i);
        mu[i] = family.clamp_mu(family.link_inverse(eta));
        weights[i] = family.irls_weight(mu[i]).max(1e-10);
    }
    let u_sq: f64 = u.iter().flatten().map(|&v| v * v).sum();
    // log|L|² = Σⱼ log det(Aⱼ) at the converged weights (only ZᵀWZ is needed).
    let zeros = vec![0.0; n];
    let stats = GroupStats::new(design, z, &zeros, group_idx, n_groups, p, q, Some(&weights));
    let mut logdet_l2 = 0.0;
    for g in 0..n_groups {
        let a = add_identity(&mm(&tt, &mm(&stats.ztz[g], t)));
        let chol = cholesky(&a)?;
        logdet_l2 += 2.0 * (0..q).map(|i| chol[i][i].ln()).sum::<f64>();
    }

    let b: Vec<Vec<f64>> = u.iter().map(|ug| matvec(t, ug)).collect();

    Some(PirlsSolution {
        beta,
        b,
        mu,
        u_sq,
        logdet_l2,
        m_inv: last_m_inv,
        converged,
        iterations,
    })
}

/// A plain GLM (IRLS, no random effect) to warm-start the PIRLS β.
fn glm_warm_start(
    family: &dyn GlmFamily,
    design: &Mat<f64>,
    y: &[f64],
    offset: &[f64],
    p: usize,
    max_iter: usize,
) -> Result<Vec<f64>, RegressionError> {
    let n = design.nrows();
    let mut mu = family.initialize_mu(y);
    let mut beta = vec![0.0; p];

    for _ in 0..max_iter {
        let mut xtwx = vec![vec![0.0; p]; p];
        let mut xtwz = vec![0.0; p];
        for i in 0..n {
            let m = family.clamp_mu(mu[i]);
            let eta = family.link(m);
            let w = family.irls_weight(m).max(1e-10);
            let zeta = family.working_response(y[i], m, eta) - offset[i];
            for k in 0..p {
                let wx = w * design[(i, k)];
                xtwz[k] += wx * zeta;
                for l in 0..p {
                    xtwx[k][l] += wx * design[(i, l)];
                }
            }
        }
        let chol = cholesky(&xtwx).ok_or(RegressionError::SingularMatrix)?;
        let beta_new = cholesky_solve(&chol, &xtwz);

        let mut max_change = 0.0f64;
        for k in 0..p {
            max_change = max_change.max((beta_new[k] - beta[k]).abs());
        }
        beta = beta_new;

        for i in 0..n {
            let mut eta = offset[i];
            for k in 0..p {
                eta += design[(i, k)] * beta[k];
            }
            mu[i] = family.clamp_mu(family.link_inverse(eta));
        }
        if max_change < 1e-8 {
            break;
        }
    }
    Ok(beta)
}

// ============================================= multi-factor engine ==========

/// Layout of several crossed / nested random-effect factors. Factor `f` has
/// `q_f` random effects per level (intercept and/or slopes) with relative
/// covariance factor `T_f` (`q_f(q_f+1)/2` parameters); its combined random
/// columns are `col_offset[f] + level·q_f + a`.
struct MultiSpec {
    q: Vec<usize>,
    n_levels: Vec<usize>,
    col_offset: Vec<usize>,
    theta_offset: Vec<usize>,
    n_theta: usize,
    m_dim: usize,
    /// Per factor: compacted level of each observation.
    levels: Vec<Vec<usize>>,
    /// Per factor: random design `Z_f` (`n × q_f`).
    z: Vec<Mat<f64>>,
}

/// Per-factor random-effect covariance and BLUPs (`n_levels × q_f`).
#[derive(Debug, Clone)]
struct FactorDetail {
    cov: Vec<Vec<f64>>,
    re: Vec<Vec<f64>>,
}

impl MultiSpec {
    fn new(q: Vec<usize>, n_levels: Vec<usize>, levels: Vec<Vec<usize>>, z: Vec<Mat<f64>>) -> Self {
        let mut col_offset = Vec::with_capacity(q.len());
        let mut theta_offset = Vec::with_capacity(q.len());
        let (mut cols, mut ths) = (0, 0);
        for (qf, jf) in q.iter().zip(&n_levels) {
            col_offset.push(cols);
            theta_offset.push(ths);
            cols += qf * jf;
            ths += qf * (qf + 1) / 2;
        }
        Self {
            q,
            n_levels,
            col_offset,
            theta_offset,
            n_theta: ths,
            m_dim: cols,
            levels,
            z,
        }
    }

    /// Stacked identity starting values (`[1.0; F]` for random intercepts).
    fn initial_theta(&self) -> Vec<f64> {
        self.q.iter().flat_map(|&qf| initial_theta(qf)).collect()
    }

    /// The per-factor relative covariance factors `T_f`.
    fn ts(&self, theta: &[f64]) -> Vec<Vec<Vec<f64>>> {
        self.q
            .iter()
            .zip(&self.theta_offset)
            .map(|(&qf, &off)| build_t(&theta[off..off + qf * (qf + 1) / 2], qf))
            .collect()
    }

    /// Non-zero entries `(column, value)` of each row of `Z Λ_θ`.
    #[allow(clippy::needless_range_loop)]
    fn zl(&self, ts: &[Vec<Vec<f64>>]) -> Vec<Vec<(usize, f64)>> {
        let n = self.levels.first().map_or(0, Vec::len);
        (0..n)
            .map(|i| {
                let mut row = Vec::new();
                for (f, t) in ts.iter().enumerate() {
                    let qf = self.q[f];
                    let base = self.col_offset[f] + self.levels[f][i] * qf;
                    for a in 0..qf {
                        let v: f64 = (0..qf).map(|r| self.z[f][(i, r)] * t[r][a]).sum();
                        row.push((base + a, v));
                    }
                }
                row
            })
            .collect()
    }

    /// Per-factor summaries from the spherical modes `u` (`b = Λ_θ u`);
    /// covariances are `scale² · T_f T_fᵀ`.
    fn summaries(
        &self,
        ts: &[Vec<Vec<f64>>],
        u: &[f64],
        scale: f64,
    ) -> (Vec<FactorSummary>, Vec<FactorDetail>) {
        let mut summaries = Vec::with_capacity(self.q.len());
        let mut details = Vec::with_capacity(self.q.len());
        for (f, t) in ts.iter().enumerate() {
            let qf = self.q[f];
            let re: Vec<Vec<f64>> = (0..self.n_levels[f])
                .map(|l| {
                    let start = self.col_offset[f] + l * qf;
                    matvec(t, &u[start..start + qf])
                })
                .collect();
            let cov = scale_matrix(&mm(t, &transpose(t)), scale * scale);
            summaries.push(FactorSummary {
                n_levels: self.n_levels[f],
                sd: cov[0][0].max(0.0).sqrt(),
                blups: re.iter().map(|b| b[0]).collect(),
            });
            details.push(FactorDetail { cov, re });
        }
        (summaries, details)
    }
}

/// Cross-products of `[ZΛ, X]` against a response with optional weights,
/// assembled into `A = ΛᵀZᵀWZΛ + I`, `B = ΛᵀZᵀWX`, `c = ΛᵀZᵀWr`.
struct MultiStats {
    a: Vec<Vec<f64>>,    // M×M
    bmat: Vec<Vec<f64>>, // M×p
    cvec: Vec<f64>,      // M
    xtx: Vec<Vec<f64>>,  // p×p
    xtr: Vec<f64>,       // p
}

impl MultiStats {
    fn new(
        design: &Mat<f64>,
        resp: &[f64],
        zl: &[Vec<(usize, f64)>],
        m_dim: usize,
        p: usize,
        weights: Option<&[f64]>,
    ) -> Self {
        let n = design.nrows();
        let mut a = vec![vec![0.0; m_dim]; m_dim];
        let mut bmat = vec![vec![0.0; p]; m_dim];
        let mut cvec = vec![0.0; m_dim];
        let mut xtx = vec![vec![0.0; p]; p];
        let mut xtr = vec![0.0; p];
        for i in 0..n {
            let w = weights.map_or(1.0, |w| w[i]);
            let ri = resp[i];
            for &(c, vc) in &zl[i] {
                let wv = w * vc;
                cvec[c] += wv * ri;
                for k in 0..p {
                    bmat[c][k] += wv * design[(i, k)];
                }
                for &(d, vd) in &zl[i] {
                    a[c][d] += wv * vd;
                }
            }
            for k in 0..p {
                let wx = w * design[(i, k)];
                xtr[k] += wx * ri;
                for l in 0..p {
                    xtx[k][l] += wx * design[(i, l)];
                }
            }
        }
        for (c, row) in a.iter_mut().enumerate() {
            row[c] += 1.0;
        }
        Self {
            a,
            bmat,
            cvec,
            xtx,
            xtr,
        }
    }
}

/// Solution of the multi-factor penalized least-squares problem at fixed θ.
struct MultiSolution {
    beta: Vec<f64>,
    /// Spherical random effects `u` (one per combined column).
    u: Vec<f64>,
    u_sq: f64,
    logdet_l2: f64,
    logdet_rx2: f64,
    m_inv: Vec<Vec<f64>>,
}

/// Solve for `(β, u)` from assembled stats (crossed / nested LMM step, or one
/// PIRLS iteration). `None` if the random block or Schur complement is
/// singular.
fn solve_multi(st: &MultiStats, m_dim: usize, p: usize) -> Option<MultiSolution> {
    let chol_a = cholesky(&st.a)?;
    let logdet_l2 = 2.0 * (0..m_dim).map(|i| chol_a[i][i].ln()).sum::<f64>();
    let aib = chol_solve_mat(&chol_a, &st.bmat); // M×p
    let aic = cholesky_solve(&chol_a, &st.cvec); // M

    let mut mfix = st.xtx.clone();
    let mut rhs = st.xtr.clone();
    let btaib = mm(&transpose(&st.bmat), &aib);
    let btaic = matvec(&transpose(&st.bmat), &aic);
    for k in 0..p {
        rhs[k] -= btaic[k];
        for l in 0..p {
            mfix[k][l] -= btaib[k][l];
        }
    }
    let chol_m = cholesky(&mfix)?;
    let logdet_rx2 = 2.0 * (0..p).map(|i| chol_m[i][i].ln()).sum::<f64>();
    let beta = cholesky_solve(&chol_m, &rhs);
    let m_inv = cholesky_inverse(&chol_m);

    let u: Vec<f64> = (0..m_dim)
        .map(|c| aic[c] - (0..p).map(|k| aib[c][k] * beta[k]).sum::<f64>())
        .collect();
    let u_sq = u.iter().map(|v| v * v).sum();

    Some(MultiSolution {
        beta,
        u,
        u_sq,
        logdet_l2,
        logdet_rx2,
        m_inv,
    })
}

/// `x_iᵀβ + (ZΛu)_i` for one observation.
fn multi_eta(design: &Mat<f64>, zl_row: &[(usize, f64)], beta: &[f64], u: &[f64], i: usize) -> f64 {
    let fixed: f64 = beta
        .iter()
        .enumerate()
        .map(|(k, b)| design[(i, k)] * b)
        .sum();
    fixed + zl_row.iter().map(|&(c, v)| v * u[c]).sum::<f64>()
}

/// Residual sum of squares ‖y − Xβ − ZΛu‖² for the multi-factor LMM.
fn multi_rss(
    design: &Mat<f64>,
    y: &[f64],
    zl: &[Vec<(usize, f64)>],
    beta: &[f64],
    u: &[f64],
    _p: usize,
) -> f64 {
    (0..design.nrows())
        .map(|i| (y[i] - multi_eta(design, &zl[i], beta, u, i)).powi(2))
        .sum()
}

/// Profiled REML / ML deviance from the pRSS and log-determinant pieces.
fn deviance_from_pieces(
    prss: f64,
    logdet_l2: f64,
    logdet_rx2: f64,
    n: usize,
    p: usize,
    reml: bool,
) -> f64 {
    if reml {
        let dof = (n - p) as f64;
        logdet_l2 + logdet_rx2 + dof * (1.0 + (2.0 * std::f64::consts::PI * prss / dof).ln())
    } else {
        let nf = n as f64;
        logdet_l2 + nf * (1.0 + (2.0 * std::f64::consts::PI * prss / nf).ln())
    }
}

/// Converged PIRLS solution for the multi-factor GLMM at fixed θ.
struct PirlsMultiSolution {
    beta: Vec<f64>,
    u: Vec<f64>,
    mu: Vec<f64>,
    u_sq: f64,
    logdet_l2: f64,
    m_inv: Vec<Vec<f64>>,
    converged: bool,
    iterations: usize,
}

/// PIRLS for the multi-factor GLMM: conditional modes `(β, u)` at fixed θ.
#[allow(clippy::too_many_arguments)]
fn pirls_multi(
    family: &dyn GlmFamily,
    design: &Mat<f64>,
    y: &[f64],
    offset: &[f64],
    spec: &MultiSpec,
    theta: &[f64],
    beta0: &[f64],
    max_iter: usize,
    tol: f64,
) -> Option<PirlsMultiSolution> {
    let n = design.nrows();
    let p = design.ncols();
    let m_dim = spec.m_dim;
    let zl = spec.zl(&spec.ts(theta));

    let mut beta = beta0.to_vec();
    let mut u = vec![0.0; m_dim];
    let mut converged = false;
    let mut iterations = 0;
    let mut last_m_inv = vec![vec![0.0; p]; p];

    for iter in 0..max_iter {
        iterations = iter + 1;
        let mut zeta = vec![0.0; n];
        let mut weights = vec![0.0; n];
        for i in 0..n {
            let eta = multi_eta(design, &zl[i], &beta, &u, i) + offset[i];
            let mu = family.clamp_mu(family.link_inverse(eta));
            weights[i] = family.irls_weight(mu).max(1e-10);
            zeta[i] = family.working_response(y[i], mu, eta) - offset[i];
        }
        let st = MultiStats::new(design, &zeta, &zl, m_dim, p, Some(&weights));
        let sol = solve_multi(&st, m_dim, p)?;
        last_m_inv = sol.m_inv;

        let max_change = sol
            .u
            .iter()
            .zip(&u)
            .chain(sol.beta.iter().zip(&beta))
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        beta = sol.beta;
        u = sol.u;
        if max_change < tol {
            converged = true;
            break;
        }
    }

    // Pieces of the Laplace criterion at the mode.
    let mut mu = vec![0.0; n];
    let mut weights = vec![0.0; n];
    for i in 0..n {
        let eta = multi_eta(design, &zl[i], &beta, &u, i) + offset[i];
        mu[i] = family.clamp_mu(family.link_inverse(eta));
        weights[i] = family.irls_weight(mu[i]).max(1e-10);
    }
    let u_sq: f64 = u.iter().map(|&v| v * v).sum();
    let st = MultiStats::new(design, &vec![0.0; n], &zl, m_dim, p, Some(&weights));
    let chol_a = cholesky(&st.a)?;
    let logdet_l2 = 2.0 * (0..m_dim).map(|i| chol_a[i][i].ln()).sum::<f64>();

    Some(PirlsMultiSolution {
        beta,
        u,
        mu,
        u_sq,
        logdet_l2,
        m_inv: last_m_inv,
        converged,
        iterations,
    })
}

// ================================================= θ optimisation ============

/// Optimise the profiled deviation over the `q(q+1)/2` entries of `T`.
/// Uses golden-section search for the scalar random-intercept case and
/// Nelder–Mead otherwise.
fn optimize_theta(
    q: usize,
    theta_max: f64,
    tol: f64,
    f: &dyn Fn(&[f64]) -> Option<f64>,
) -> Result<Vec<f64>, RegressionError> {
    if q == 1 {
        let scalar = golden_section_min(0.0, theta_max, tol, &|t| f(&[t]))?;
        Ok(vec![scalar])
    } else {
        let x0 = initial_theta(q);
        let cost = |th: &[f64]| f(th).unwrap_or(f64::INFINITY);
        let best = nelder_mead(&x0, &cost, tol.max(1e-9), 4000);
        if best.iter().all(|v| v.is_finite()) {
            Ok(best)
        } else {
            Err(RegressionError::NumericalError(
                "θ optimisation did not produce a finite value".to_string(),
            ))
        }
    }
}

/// Optimise the profiled deviance over the stacked per-factor θ (crossed /
/// nested case) by Nelder–Mead, starting from `x0`.
fn optimize_theta_vec(
    x0: &[f64],
    tol: f64,
    f: &dyn Fn(&[f64]) -> Option<f64>,
) -> Result<Vec<f64>, RegressionError> {
    let cost = |th: &[f64]| f(th).unwrap_or(f64::INFINITY);
    let best = nelder_mead(x0, &cost, tol.max(1e-9), 4000);
    if best.iter().all(|v| v.is_finite()) {
        Ok(best)
    } else {
        Err(RegressionError::NumericalError(
            "θ optimisation did not produce a finite value".to_string(),
        ))
    }
}

/// Golden-section minimisation of a unimodal objective on `[a, b]`.
fn golden_section_min(
    mut a: f64,
    mut b: f64,
    tol: f64,
    f: &dyn Fn(f64) -> Option<f64>,
) -> Result<f64, RegressionError> {
    const INV_PHI: f64 = 0.618_033_988_749_895;
    let eval = |x: f64| -> f64 { f(x).unwrap_or(f64::INFINITY) };

    let mut c = b - INV_PHI * (b - a);
    let mut d = a + INV_PHI * (b - a);
    let mut fc = eval(c);
    let mut fd = eval(d);

    for _ in 0..200 {
        if (b - a).abs() < tol * (1.0 + b.abs()) {
            break;
        }
        if fc < fd {
            b = d;
            d = c;
            fd = fc;
            c = b - INV_PHI * (b - a);
            fc = eval(c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + INV_PHI * (b - a);
            fd = eval(d);
        }
    }
    let theta = 0.5 * (a + b);
    if !theta.is_finite() {
        return Err(RegressionError::NumericalError(
            "θ optimisation did not produce a finite value".to_string(),
        ));
    }
    Ok(theta)
}

/// Nelder–Mead simplex minimisation for the multi-parameter `T` case.
#[allow(clippy::needless_range_loop)]
fn nelder_mead(x0: &[f64], f: &dyn Fn(&[f64]) -> f64, tol: f64, max_iter: usize) -> Vec<f64> {
    let n = x0.len();
    let (alpha, gamma, rho, sigma) = (1.0, 2.0, 0.5, 0.5);

    let mut simplex: Vec<Vec<f64>> = Vec::with_capacity(n + 1);
    simplex.push(x0.to_vec());
    for i in 0..n {
        let mut xi = x0.to_vec();
        let step = if x0[i].abs() > 1e-6 {
            0.5 * x0[i].abs()
        } else {
            0.5
        };
        xi[i] += step;
        simplex.push(xi);
    }
    let mut fvals: Vec<f64> = simplex.iter().map(|x| f(x)).collect();

    for _ in 0..max_iter {
        // Order vertices by cost (ascending).
        let mut order: Vec<usize> = (0..=n).collect();
        order.sort_by(|&a, &b| {
            fvals[a]
                .partial_cmp(&fvals[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        simplex = order.iter().map(|&i| simplex[i].clone()).collect();
        fvals = order.iter().map(|&i| fvals[i]).collect();

        if (fvals[n] - fvals[0]).abs() < tol * (1.0 + fvals[0].abs()) {
            break;
        }

        // Centroid of the best n vertices.
        let mut cent = vec![0.0; n];
        for vertex in simplex.iter().take(n) {
            for (k, ck) in cent.iter_mut().enumerate() {
                *ck += vertex[k];
            }
        }
        for ck in &mut cent {
            *ck /= n as f64;
        }

        // Reflection.
        let xr: Vec<f64> = (0..n)
            .map(|k| cent[k] + alpha * (cent[k] - simplex[n][k]))
            .collect();
        let fr = f(&xr);
        if fr < fvals[0] {
            // Expansion.
            let xe: Vec<f64> = (0..n)
                .map(|k| cent[k] + gamma * (xr[k] - cent[k]))
                .collect();
            let fe = f(&xe);
            if fe < fr {
                simplex[n] = xe;
                fvals[n] = fe;
            } else {
                simplex[n] = xr;
                fvals[n] = fr;
            }
        } else if fr < fvals[n - 1] {
            simplex[n] = xr;
            fvals[n] = fr;
        } else {
            // Contraction.
            let xc: Vec<f64> = (0..n)
                .map(|k| cent[k] + rho * (simplex[n][k] - cent[k]))
                .collect();
            let fc = f(&xc);
            if fc < fvals[n] {
                simplex[n] = xc;
                fvals[n] = fc;
            } else {
                // Shrink toward the best vertex.
                for i in 1..=n {
                    for k in 0..n {
                        simplex[i][k] = simplex[0][k] + sigma * (simplex[i][k] - simplex[0][k]);
                    }
                    fvals[i] = f(&simplex[i]);
                }
            }
        }
    }

    let mut best = 0;
    for i in 1..=n {
        if fvals[i] < fvals[best] {
            best = i;
        }
    }
    simplex[best].clone()
}

// ================================================= shared numeric helpers ====

/// Compact arbitrary group ids to `0..J`, preserving first-seen order.
fn compact_groups(group: &[usize]) -> (Vec<usize>, usize) {
    use std::collections::HashMap;
    let mut map: HashMap<usize, usize> = HashMap::new();
    let mut idx = Vec::with_capacity(group.len());
    for &g in group {
        let next = map.len();
        let id = *map.entry(g).or_insert(next);
        idx.push(id);
    }
    (idx, map.len())
}

/// Build the fixed-effects design, prepending an intercept column if requested.
fn build_design(x: &Mat<f64>, with_intercept: bool) -> Mat<f64> {
    let n = x.nrows();
    let p_feat = x.ncols();
    if with_intercept {
        Mat::from_fn(
            n,
            p_feat + 1,
            |i, k| if k == 0 { 1.0 } else { x[(i, k - 1)] },
        )
    } else {
        x.to_owned()
    }
}

/// Build the random-effects design `Z` (`n × q`): an optional intercept column
/// followed by the named slope columns of `x`.
fn build_random_design(x: &Mat<f64>, with_intercept: bool, slopes: &[usize]) -> Mat<f64> {
    let n = x.nrows();
    let q = usize::from(with_intercept) + slopes.len();
    Mat::from_fn(n, q, |i, a| {
        if with_intercept {
            if a == 0 {
                1.0
            } else {
                x[(i, slopes[a - 1])]
            }
        } else {
            x[(i, slopes[a])]
        }
    })
}

/// Build the lower-triangular relative covariance factor `T` from the
/// column-major parameter vector (diagonal entries taken as `|·|` for a unique
/// parameterisation).
#[allow(clippy::needless_range_loop)]
fn build_t(theta: &[f64], q: usize) -> Vec<Vec<f64>> {
    let mut t = vec![vec![0.0; q]; q];
    let mut idx = 0;
    for c in 0..q {
        for r in c..q {
            let v = theta[idx];
            idx += 1;
            t[r][c] = if r == c { v.abs() } else { v };
        }
    }
    t
}

/// Initial `T` parameters: identity factor (unit diagonal, zero off-diagonal).
fn initial_theta(q: usize) -> Vec<f64> {
    let mut th = Vec::with_capacity(q * (q + 1) / 2);
    for c in 0..q {
        for r in c..q {
            th.push(if r == c { 1.0 } else { 0.0 });
        }
    }
    th
}

/// Return `m + I` for a square matrix `m`.
fn add_identity(m: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let mut out = m.to_vec();
    for (i, row) in out.iter_mut().enumerate() {
        row[i] += 1.0;
    }
    out
}

/// Dense matrix product `a·b` (`a` is `r×k`, `b` is `k×c`).
fn mm(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let r = a.len();
    let k = b.len();
    let c = if k > 0 { b[0].len() } else { 0 };
    let mut out = vec![vec![0.0; c]; r];
    for i in 0..r {
        for s in 0..k {
            let ais = a[i][s];
            if ais == 0.0 {
                continue;
            }
            for j in 0..c {
                out[i][j] += ais * b[s][j];
            }
        }
    }
    out
}

fn transpose(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let r = a.len();
    let c = if r > 0 { a[0].len() } else { 0 };
    let mut out = vec![vec![0.0; r]; c];
    for (i, row) in a.iter().enumerate() {
        for j in 0..c {
            out[j][i] = row[j];
        }
    }
    out
}

/// Matrix–vector product `a·v` (`a` is `r×c`, `v` length `c`).
fn matvec(a: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    a.iter()
        .map(|row| row.iter().zip(v).map(|(x, y)| x * y).sum())
        .collect()
}

/// Lower-triangular Cholesky factor `L` (`M = L Lᵀ`) of a small SPD matrix.
fn cholesky(m: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let p = m.len();
    let mut l = vec![vec![0.0; p]; p];
    for i in 0..p {
        for k in 0..=i {
            let mut sum = m[i][k];
            for (li, lk) in l[i].iter().zip(&l[k]).take(k) {
                sum -= li * lk;
            }
            if i == k {
                if sum <= 0.0 {
                    return None;
                }
                l[i][k] = sum.sqrt();
            } else {
                l[i][k] = sum / l[k][k];
            }
        }
    }
    Some(l)
}

/// Solve `M x = b` given the Cholesky factor `L` of `M`.
fn cholesky_solve(l: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let p = l.len();
    let mut y = vec![0.0; p];
    for i in 0..p {
        let mut sum = b[i];
        for k in 0..i {
            sum -= l[i][k] * y[k];
        }
        y[i] = sum / l[i][i];
    }
    let mut x = vec![0.0; p];
    for i in (0..p).rev() {
        let mut sum = y[i];
        for k in (i + 1)..p {
            sum -= l[k][i] * x[k];
        }
        x[i] = sum / l[i][i];
    }
    x
}

/// Solve `M X = B` columnwise given the Cholesky factor `L` of `M`.
/// `b` is `q×c`; the result is `q×c`.
fn chol_solve_mat(l: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let q = b.len();
    let c = if q > 0 { b[0].len() } else { 0 };
    let mut out = vec![vec![0.0; c]; q];
    for col in 0..c {
        let rhs: Vec<f64> = (0..q).map(|r| b[r][col]).collect();
        let sol = cholesky_solve(l, &rhs);
        for r in 0..q {
            out[r][col] = sol[r];
        }
    }
    out
}

/// Inverse of an SPD matrix from its Cholesky factor.
fn cholesky_inverse(l: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let p = l.len();
    let mut inv = vec![vec![0.0; p]; p];
    for col in 0..p {
        let mut e = vec![0.0; p];
        e[col] = 1.0;
        let x = cholesky_solve(l, &e);
        for (row, xr) in x.iter().enumerate() {
            inv[row][col] = *xr;
        }
    }
    inv
}

fn scale_matrix(m: &[Vec<f64>], s: f64) -> Vec<Vec<f64>> {
    m.iter()
        .map(|row| row.iter().map(|&v| v * s).collect())
        .collect()
}

fn diag_sqrt(m: &[Vec<f64>]) -> Vec<f64> {
    (0..m.len()).map(|i| m[i][i].max(0.0).sqrt()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // A tiny deterministic panel: 4 groups × 4 obs, clear group offsets.
    fn panel() -> (Mat<f64>, Col<f64>, Vec<usize>) {
        let x_raw = [
            0.1, 0.4, 0.7, 1.0, // g0
            0.2, 0.5, 0.8, 1.1, // g1
            0.3, 0.6, 0.9, 1.2, // g2
            0.0, 0.3, 0.6, 0.9, // g3
        ];
        let offs = [2.0, -1.0, 0.5, -1.5];
        let group: Vec<usize> = (0..16).map(|i| i / 4).collect();
        let x = Mat::from_fn(16, 1, |i, _| x_raw[i]);
        let y = Col::from_fn(16, |i| 1.0 + 0.8 * x_raw[i] + offs[i / 4]);
        (x, y, group)
    }

    #[test]
    fn lmm_recovers_group_structure() {
        let (x, y, group) = panel();
        let fit = GlmmRegressor::gaussian()
            .with_intercept(true)
            .build()
            .fit(&x, &y, &group)
            .expect("LMM should fit");

        assert!(
            (fit.slopes()[0] - 0.8).abs() < 0.1,
            "slope = {}",
            fit.slopes()[0]
        );
        assert!(fit.sd_random() > 0.0);
        let re = fit.random_effects();
        assert!(re[0] > re[1], "g0 offset should exceed g1");
        assert!(re[2] > re[3], "g2 offset should exceed g3");
        assert_eq!(fit.n_random_effects(), 1);
    }

    #[test]
    fn poisson_glmm_fits_and_converges() {
        let group: Vec<usize> = (0..20).map(|i| i / 5).collect();
        let x = Mat::from_fn(20, 1, |i, _| (i % 5) as f64 / 5.0);
        let levels = [1.0, 5.0, 2.0, 8.0];
        let y = Col::from_fn(20, |i| levels[i / 5] + (i % 5) as f64);
        let fit = GlmmRegressor::poisson()
            .with_intercept(true)
            .build()
            .fit(&x, &y, &group)
            .expect("Poisson GLMM should fit");
        assert!(fit.converged());
        assert!(fit.sd_random() > 0.0);
        assert!(fit.deviance().is_finite());
    }

    #[test]
    fn random_slope_fits_and_reports_covariance() {
        // 6 groups × 8 obs; group-specific intercept and slope on x.
        let j = 6usize;
        let ni = 8usize;
        let n = j * ni;
        let group: Vec<usize> = (0..n).map(|i| i / ni).collect();
        let x_raw: Vec<f64> = (0..n).map(|i| ((i % ni) as f64) / 4.0 - 1.0).collect();
        let icpt = [1.0, -0.5, 0.8, -1.0, 0.3, -0.3];
        let slope = [0.5, 1.2, -0.2, 0.9, 0.1, 1.5];
        let x = Mat::from_fn(n, 1, |i, _| x_raw[i]);
        let y = Col::from_fn(n, |i| {
            0.5 + 0.7 * x_raw[i] + icpt[i / ni] + slope[i / ni] * x_raw[i]
        });

        let fit = GlmmRegressor::gaussian()
            .with_intercept(true)
            .random_slopes(vec![0])
            .build()
            .fit(&x, &y, &group)
            .expect("random-slope LMM should fit");

        assert_eq!(fit.n_random_effects(), 2);
        let cov = fit.random_cov();
        assert_eq!(cov.len(), 2);
        // Both variance components should be positive.
        assert!(cov[0][0] > 0.0 && cov[1][1] > 0.0);
        let re = fit.random_effects_matrix();
        assert_eq!(re.len(), j);
        assert_eq!(re[0].len(), 2);
        // The per-group random slopes should track the imposed ordering.
        assert!(re[1][1] > re[2][1], "group 1 slope should exceed group 2");
    }

    #[test]
    fn crossed_intercepts_run_for_lmm_and_poisson() {
        // 4 × 3 crossed design, 3 reps.
        let na = 4usize;
        let nb = 3usize;
        let rp = 3usize;
        let n = na * nb * rp;
        let a: Vec<usize> = (0..n).map(|i| i / (nb * rp)).collect();
        let b: Vec<usize> = (0..n).map(|i| (i / rp) % nb).collect();
        let x = Mat::from_fn(n, 1, |i, _| ((i % rp) as f64) - 1.0);
        let a_off = [0.6, -0.4, 0.2, -0.3];
        let b_off = [0.3, -0.2, 0.1];

        // Gaussian.
        let y = Col::from_fn(n, |i| 1.0 + 0.5 * x[(i, 0)] + a_off[a[i]] + b_off[b[i]]);
        let fit = GlmmRegressor::gaussian()
            .build()
            .fit_crossed(&x, &y, &[&a, &b])
            .expect("crossed LMM should fit");
        assert_eq!(fit.n_factors(), 2);
        assert_eq!(fit.factors()[0].blups.len(), na);
        assert_eq!(fit.factors()[1].blups.len(), nb);

        // Poisson (exercises fit_glmm_multi / pirls_multi).
        let yp = Col::from_fn(n, |i| {
            (2.0 + a_off[a[i]] + b_off[b[i]] + (i % 4) as f64)
                .round()
                .max(0.0)
        });
        let fitp = GlmmRegressor::poisson()
            .build()
            .fit_crossed(&x, &yp, &[&a, &b])
            .expect("crossed Poisson GLMM should fit");
        assert_eq!(fitp.n_factors(), 2);
        assert!(fitp.deviance().is_finite());
        assert!(fitp.factors()[0].sd >= 0.0 && fitp.factors()[1].sd >= 0.0);
    }

    #[test]
    fn rejects_single_group() {
        let x = Mat::from_fn(6, 1, |i, _| i as f64);
        let y = Col::from_fn(6, |i| i as f64);
        let group = vec![0usize; 6];
        assert!(GlmmRegressor::gaussian()
            .build()
            .fit(&x, &y, &group)
            .is_err());
    }

    #[test]
    fn rejects_bad_input() {
        let x = Mat::from_fn(10, 1, |i, _| i as f64);
        let y = Col::from_fn(10, |i| i as f64);
        let group: Vec<usize> = (0..10).map(|i| i % 2).collect();

        // Mismatched group length.
        assert!(GlmmRegressor::gaussian()
            .build()
            .fit(&x, &y, &group[..8])
            .is_err());

        // Mismatched y length.
        let y_short = Col::from_fn(8, |i| i as f64);
        assert!(GlmmRegressor::gaussian()
            .build()
            .fit(&x, &y_short, &group)
            .is_err());

        // Random-slope column out of range.
        assert!(GlmmRegressor::gaussian()
            .random_slopes(vec![5])
            .build()
            .fit(&x, &y, &group)
            .is_err());

        // fit_crossed with mismatched factor length.
        let a = group.clone();
        let b: Vec<usize> = (0..8).map(|i| i % 3).collect();
        assert!(GlmmRegressor::gaussian()
            .build()
            .fit_crossed(&x, &y, &[&a, &b])
            .is_err());

        // fit_crossed with no factors.
        assert!(GlmmRegressor::gaussian()
            .build()
            .fit_crossed(&x, &y, &[])
            .is_err());
    }

    #[test]
    fn fit_crossed_single_factor_delegates_to_fit() {
        // With one grouping factor, fit_crossed must equal fit.
        let group: Vec<usize> = (0..20).map(|i| i / 5).collect();
        let x = Mat::from_fn(20, 1, |i, _| (i % 5) as f64 / 5.0 - 0.4);
        let off = [1.0, -0.5, 0.6, -0.8];
        let y = Col::from_fn(20, |i| 0.5 + 0.7 * x[(i, 0)] + off[i / 5]);
        let f_direct = GlmmRegressor::gaussian()
            .build()
            .fit(&x, &y, &group)
            .unwrap();
        let f_crossed = GlmmRegressor::gaussian()
            .build()
            .fit_crossed(&x, &y, &[&group])
            .unwrap();
        assert_eq!(f_direct.n_factors(), 1);
        assert_eq!(f_crossed.n_factors(), 1);
        assert!((f_direct.intercept().unwrap() - f_crossed.intercept().unwrap()).abs() < 1e-12);
        assert!((f_direct.sd_random() - f_crossed.sd_random()).abs() < 1e-12);
    }
}
