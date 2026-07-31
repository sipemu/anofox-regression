//! Generalized linear mixed models (GLMM): a random intercept and optional
//! random slopes over one grouping factor, or several crossed / nested
//! random-intercept factors.
//!
//! Covers **Phases A–C** of the mixed-model work tracked in issue #25 (mirroring
//! the DuckDB extension's `glmm_fit_agg`):
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
//! interaction id). Random slopes combined with multiple factors are not yet
//! supported.
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
//!   this matches `lme4::glmer(nAGQ = 0)` for the Poisson and binomial families.
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
//! ```

use crate::core::{BinomialFamily, BinomialLink, GlmFamily, PoissonFamily, PoissonLink};
use crate::solvers::traits::RegressionError;
use faer::{Col, Mat};

/// The response family for a [`GlmmRegressor`].
enum Response {
    /// Gaussian response — solved as a linear mixed model (exact profiled REML/ML).
    Gaussian,
    /// A generalized-linear-model family solved by Laplace-approximate PIRLS.
    Glm(Box<dyn GlmFamily>),
}

/// A GLMM with a random intercept (and optional random slopes) over one
/// grouping factor.
pub struct GlmmRegressor {
    response: Response,
    with_intercept: bool,
    /// Whether the random-effects design includes a random intercept.
    random_intercept: bool,
    /// Indices into the columns of `x` that carry a random slope.
    random_slopes: Vec<usize>,
    /// Use REML (`true`) rather than ML (`false`) for the Gaussian case. Ignored
    /// for non-Gaussian families, which use the Laplace ML deviance.
    reml: bool,
    /// Maximum PIRLS iterations (inner loop) for the GLMM case.
    max_iterations: usize,
    /// Convergence tolerance for the inner loop and the outer θ search.
    tolerance: f64,
    /// Upper bound on the profiled ratio θ = σ_b / σ (random-intercept case).
    theta_max: f64,
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

        let q = usize::from(self.random_intercept) + self.random_slopes.len();
        if q == 0 {
            return Err(RegressionError::NumericalError(
                "GLMM needs a random intercept or at least one random slope".to_string(),
            ));
        }
        for &c in &self.random_slopes {
            if c >= x.ncols() {
                return Err(RegressionError::NumericalError(format!(
                    "random-slope column {c} is out of range for x with {} columns",
                    x.ncols()
                )));
            }
        }

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
        let z = build_random_design(x, self.random_intercept, &self.random_slopes);

        match &self.response {
            Response::Gaussian => self.fit_lmm(&design, &z, y, &group_idx, n_groups, p, q),
            Response::Glm(family) => {
                self.fit_glmm(family.as_ref(), &design, &z, y, &group_idx, n_groups, p, q)
            }
        }
    }

    /// Fit with **multiple crossed or nested grouping factors**, each carrying a
    /// random intercept with its own variance component:
    ///
    /// ```text
    /// g(μ_i) = x_i'β + Σ_f b_{f, level_f(i)},   b_{f,·} ~ N(0, σ_f²)
    /// ```
    ///
    /// * `groups[f]` is the level id of factor `f` for each observation
    ///   (length `n`; any `usize`, compacted internally).
    /// * Crossed factors: pass the two id vectors directly, e.g.
    ///   `(1|sku) + (1|region)` → `&[&sku, &region]`.
    /// * Nested factors `(1|a/b)`: pass `a` and the interaction id `a:b` (a
    ///   globally-unique id per `(a, b)` pair) → `&[&a, &ab]`.
    ///
    /// With a single factor this delegates to [`fit`](Self::fit) (which also
    /// supports random slopes). Random slopes combined with multiple factors are
    /// not yet supported.
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
        if groups.len() == 1 {
            return self.fit(x, y, groups[0]);
        }
        if !self.random_slopes.is_empty() {
            return Err(RegressionError::NumericalError(
                "random slopes with multiple grouping factors are not yet supported".to_string(),
            ));
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

        // Compact each factor's ids and lay out combined random columns.
        let n_factors = groups.len();
        let mut levels = Vec::with_capacity(n_factors); // per factor: level per obs
        let mut n_levels = Vec::with_capacity(n_factors);
        for g in groups {
            let (idx, jf) = compact_groups(g);
            if jf < 2 {
                return Err(RegressionError::NumericalError(
                    "each grouping factor needs at least two distinct levels".to_string(),
                ));
            }
            levels.push(idx);
            n_levels.push(jf);
        }
        let mut col_offset = vec![0usize; n_factors];
        let mut acc = 0;
        for f in 0..n_factors {
            col_offset[f] = acc;
            acc += n_levels[f];
        }
        let m_dim = acc; // total random columns

        let p = if self.with_intercept {
            x.ncols() + 1
        } else {
            x.ncols()
        };
        if n <= p + n_factors {
            return Err(RegressionError::InsufficientObservations {
                needed: p + n_factors + 1,
                got: n,
            });
        }
        let design = build_design(x, self.with_intercept);

        // Combined column index per observation per factor.
        let obs_cols: Vec<Vec<usize>> = (0..n)
            .map(|i| {
                (0..n_factors)
                    .map(|f| col_offset[f] + levels[f][i])
                    .collect()
            })
            .collect();
        // Factor owning each combined column.
        let mut col_factor = vec![0usize; m_dim];
        for f in 0..n_factors {
            for l in 0..n_levels[f] {
                col_factor[col_offset[f] + l] = f;
            }
        }

        match &self.response {
            Response::Gaussian => self.fit_lmm_multi(
                &design,
                y,
                &obs_cols,
                &col_factor,
                m_dim,
                p,
                n_factors,
                &n_levels,
                &col_offset,
            ),
            Response::Glm(family) => self.fit_glmm_multi(
                family.as_ref(),
                &design,
                y,
                &obs_cols,
                &col_factor,
                m_dim,
                p,
                n_factors,
                &n_levels,
                &col_offset,
            ),
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

    /// Fit a non-Gaussian GLMM by profiling the Laplace deviance over θ, with a
    /// PIRLS inner loop finding the conditional modes at each θ.
    #[allow(clippy::too_many_arguments)]
    fn fit_glmm(
        &self,
        family: &dyn GlmFamily,
        design: &Mat<f64>,
        z: &Mat<f64>,
        y: &Col<f64>,
        group_idx: &[usize],
        n_groups: usize,
        p: usize,
        q: usize,
    ) -> Result<FittedGlmm, RegressionError> {
        let n = design.nrows();
        let y_slice: Vec<f64> = (0..n).map(|i| y[i]).collect();
        let beta0 = glm_warm_start(family, design, &y_slice, p, self.max_iterations)?;

        let objective = |theta: &[f64]| -> Option<f64> {
            let t = build_t(theta, q);
            pirls(
                family,
                design,
                z,
                &y_slice,
                group_idx,
                n_groups,
                p,
                q,
                &t,
                &beta0,
                self.max_iterations,
                self.tolerance,
            )
            .map(|s| s.laplace_deviance)
        };

        let theta = optimize_theta(q, self.theta_max, self.tolerance, &objective)?;
        let t = build_t(&theta, q);
        let sol = pirls(
            family,
            design,
            z,
            &y_slice,
            group_idx,
            n_groups,
            p,
            q,
            &t,
            &beta0,
            self.max_iterations,
            self.tolerance,
        )
        .ok_or(RegressionError::ConvergenceFailed {
            iterations: self.max_iterations,
        })?;

        // Poisson/binomial dispersion is fixed at 1: Var(β) = M⁻¹, Σ = T Tᵀ.
        let cov = mm(&t, &transpose(&t));

        Ok(FittedGlmm::new(
            self.with_intercept,
            sol.beta,
            diag_sqrt(&sol.m_inv),
            sol.b,
            n_groups,
            q,
            t[0][0],
            1.0,
            cov,
            sol.laplace_deviance,
            false,
            sol.converged,
            sol.iterations,
        ))
    }

    // -------------------------------------------------------- multi-factor ---

    /// Fit a Gaussian LMM with several crossed / nested random-intercept
    /// factors by profiling the REML/ML deviance over `(θ_1, …, θ_F)`.
    #[allow(clippy::too_many_arguments)]
    fn fit_lmm_multi(
        &self,
        design: &Mat<f64>,
        y: &Col<f64>,
        obs_cols: &[Vec<usize>],
        col_factor: &[usize],
        m_dim: usize,
        p: usize,
        n_factors: usize,
        n_levels: &[usize],
        col_offset: &[usize],
    ) -> Result<FittedGlmm, RegressionError> {
        let n = design.nrows();
        let y_vec: Vec<f64> = (0..n).map(|i| y[i]).collect();
        let raw = MultiRawStats::new(design, &y_vec, obs_cols, m_dim, p, None);

        let objective = |theta: &[f64]| -> Option<f64> {
            solve_multi(&raw, theta, col_factor, m_dim, p).map(|s| {
                let prss = multi_prss(design, &y_vec, obs_cols, &s.beta, &s.b, p) + s.u_sq;
                deviance_from_pieces(prss, s.logdet_l2, s.logdet_rx2, n, p, self.reml)
            })
        };

        let theta = optimize_theta_vec(n_factors, self.tolerance, &objective)?;
        let sol = solve_multi(&raw, &theta, col_factor, m_dim, p)
            .ok_or(RegressionError::SingularMatrix)?;
        let prss = multi_prss(design, &y_vec, obs_cols, &sol.beta, &sol.b, p) + sol.u_sq;

        let dof = if self.reml { n - p } else { n };
        let sigma2 = prss / dof as f64;
        let sigma = sigma2.sqrt();
        let std_errors = diag_sqrt(&scale_matrix(&sol.m_inv, sigma2));
        let deviance = deviance_from_pieces(prss, sol.logdet_l2, sol.logdet_rx2, n, p, self.reml);

        let factors =
            build_factor_summaries(&sol.b, col_offset, n_levels, n_factors, &theta, sigma);
        Ok(FittedGlmm::new_multi(
            self.with_intercept,
            sol.beta,
            std_errors,
            sigma,
            factors,
            deviance,
            self.reml,
            true,
            1,
        ))
    }

    /// Fit a non-Gaussian GLMM with several crossed / nested random-intercept
    /// factors by profiling the Laplace deviance, PIRLS in the inner loop.
    #[allow(clippy::too_many_arguments)]
    fn fit_glmm_multi(
        &self,
        family: &dyn GlmFamily,
        design: &Mat<f64>,
        y: &Col<f64>,
        obs_cols: &[Vec<usize>],
        col_factor: &[usize],
        m_dim: usize,
        p: usize,
        n_factors: usize,
        n_levels: &[usize],
        col_offset: &[usize],
    ) -> Result<FittedGlmm, RegressionError> {
        let n = design.nrows();
        let y_slice: Vec<f64> = (0..n).map(|i| y[i]).collect();
        let beta0 = glm_warm_start(family, design, &y_slice, p, self.max_iterations)?;

        let objective = |theta: &[f64]| -> Option<f64> {
            pirls_multi(
                family,
                design,
                &y_slice,
                obs_cols,
                col_factor,
                m_dim,
                p,
                theta,
                &beta0,
                self.max_iterations,
                self.tolerance,
            )
            .map(|s| s.laplace_deviance)
        };

        let theta = optimize_theta_vec(n_factors, self.tolerance, &objective)?;
        let sol = pirls_multi(
            family,
            design,
            &y_slice,
            obs_cols,
            col_factor,
            m_dim,
            p,
            &theta,
            &beta0,
            self.max_iterations,
            self.tolerance,
        )
        .ok_or(RegressionError::ConvergenceFailed {
            iterations: self.max_iterations,
        })?;

        let factors = build_factor_summaries(&sol.b, col_offset, n_levels, n_factors, &theta, 1.0);
        Ok(FittedGlmm::new_multi(
            self.with_intercept,
            sol.beta,
            diag_sqrt(&sol.m_inv),
            1.0,
            factors,
            sol.laplace_deviance,
            false,
            sol.converged,
            sol.iterations,
        ))
    }
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
        deviance: f64,
        reml: bool,
        converged: bool,
        iterations: usize,
    ) -> Self {
        let f0 = &factors[0];
        let re_matrix: Vec<Vec<f64>> = f0.blups.iter().map(|&b| vec![b]).collect();
        let re_intercept = f0.blups.clone();
        let cov = vec![vec![f0.sd * f0.sd]];
        let theta0 = if sigma > 0.0 { f0.sd / sigma } else { 0.0 };
        Self {
            with_intercept,
            fixed_effects,
            std_errors,
            re_matrix,
            re_intercept,
            n_groups: f0.n_levels,
            q: 1,
            theta0,
            sigma,
            cov,
            deviance,
            log_likelihood: -0.5 * deviance,
            reml,
            converged,
            iterations,
            factors,
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

    /// Residual standard deviation σ (1.0 for Poisson/binomial families).
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

    /// Log-likelihood (−deviance / 2).
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
}

/// Builder for [`GlmmRegressor`].
pub struct GlmmRegressorBuilder {
    kind: ResponseKind,
    with_intercept: bool,
    random_intercept: bool,
    random_slopes: Vec<usize>,
    reml: bool,
    max_iterations: usize,
    tolerance: f64,
    theta_max: f64,
}

impl GlmmRegressorBuilder {
    fn new(kind: ResponseKind) -> Self {
        Self {
            kind,
            with_intercept: true,
            random_intercept: true,
            random_slopes: Vec::new(),
            reml: true,
            max_iterations: 100,
            tolerance: 1e-8,
            theta_max: 1000.0,
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
        let response = match self.kind {
            ResponseKind::Gaussian => Response::Gaussian,
            ResponseKind::Poisson => Response::Glm(Box::new(PoissonFamily::new(PoissonLink::Log))),
            ResponseKind::Binomial => {
                Response::Glm(Box::new(BinomialFamily::new(BinomialLink::Logit)))
            }
        };
        GlmmRegressor {
            response,
            with_intercept: self.with_intercept,
            random_intercept: self.random_intercept,
            random_slopes: self.random_slopes,
            reml: self.reml,
            max_iterations: self.max_iterations,
            tolerance: self.tolerance,
            theta_max: self.theta_max,
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
    /// Laplace deviance d(θ) = Σ dᵢ(yᵢ, μᵢ) + ‖u‖² + log|L_θ|².
    laplace_deviance: f64,
    m_inv: Vec<Vec<f64>>,
    converged: bool,
    iterations: usize,
}

/// Penalized IRLS: find the conditional modes `(β, u)` for a fixed `T`, then
/// form the Laplace-approximate deviance at the mode.
#[allow(clippy::too_many_arguments)]
fn pirls(
    family: &dyn GlmFamily,
    design: &Mat<f64>,
    z: &Mat<f64>,
    y: &[f64],
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

    for iter in 0..max_iter {
        iterations = iter + 1;

        // Working response ζ at the current (β, u): η = Xβ + Z T u.
        let mut zeta = vec![0.0; n];
        let mut weights = vec![0.0; n];
        for i in 0..n {
            let g = group_idx[i];
            let tu = matvec(t, &u[g]);
            let mut eta = 0.0;
            for k in 0..p {
                eta += design[(i, k)] * beta[k];
            }
            for a in 0..q {
                eta += z[(i, a)] * tu[a];
            }
            let mu = family.clamp_mu(family.link_inverse(eta));
            weights[i] = family.irls_weight(mu).max(1e-10);
            zeta[i] = family.working_response(y[i], mu, eta);
        }

        // Per-group cross-products at the current working response and weights.
        let stats = GroupStats::new(design, z, &zeta, group_idx, n_groups, p, q, Some(&weights));

        // Schur complement over the fixed effects.
        let mut m = stats.xtx.clone();
        let mut rhs = stats.xty.clone();
        let mut aib_all = Vec::with_capacity(n_groups);
        let mut aic_all = Vec::with_capacity(n_groups);
        let mut ok = true;
        for g in 0..n_groups {
            let a = add_identity(&mm(&tt, &mm(&stats.ztz[g], t)));
            let bmat = mm(&tt, &stats.ztx[g]);
            let c = matvec(&tt, &stats.zty[g]);
            let chol = match cholesky(&a) {
                Some(ch) => ch,
                None => {
                    ok = false;
                    break;
                }
            };
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
        if !ok {
            return None;
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

    // Laplace deviance at the mode: Σ unit_deviance + ‖u‖² + log|L|².
    let mut disc = 0.0;
    let mut u_sq = 0.0;
    let mut logdet_l2 = 0.0;
    let mut weights = vec![0.0; n];
    for i in 0..n {
        let g = group_idx[i];
        let tu = matvec(t, &u[g]);
        let mut eta = 0.0;
        for k in 0..p {
            eta += design[(i, k)] * beta[k];
        }
        for a in 0..q {
            eta += z[(i, a)] * tu[a];
        }
        let mu = family.clamp_mu(family.link_inverse(eta));
        weights[i] = family.irls_weight(mu).max(1e-10);
        disc += family.unit_deviance(y[i], mu);
    }
    for ug in &u {
        u_sq += ug.iter().map(|&v| v * v).sum::<f64>();
    }
    // log|L|² = Σⱼ log det(Aⱼ) at the converged weights (only ZᵀWZ is needed).
    let zeros = vec![0.0; n];
    let stats = GroupStats::new(design, z, &zeros, group_idx, n_groups, p, q, Some(&weights));
    for g in 0..n_groups {
        let a = add_identity(&mm(&tt, &mm(&stats.ztz[g], t)));
        let chol = cholesky(&a)?;
        logdet_l2 += 2.0 * (0..q).map(|i| chol[i][i].ln()).sum::<f64>();
    }

    let b: Vec<Vec<f64>> = u.iter().map(|ug| matvec(t, ug)).collect();

    Some(PirlsSolution {
        beta,
        b,
        laplace_deviance: disc + u_sq + logdet_l2,
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
            let zeta = family.working_response(y[i], m, eta);
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
            let mut eta = 0.0;
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

/// Raw (θ-independent) cross-products for the combined random design `Z` of a
/// crossed / nested fit. `Z` is an indicator with one 1 per factor per row.
struct MultiRawStats {
    ztz: Vec<Vec<f64>>, // M×M
    ztx: Vec<Vec<f64>>, // M×p
    ztr: Vec<f64>,      // M  (against the response / working response)
    xtx: Vec<Vec<f64>>, // p×p
    xtr: Vec<f64>,      // p
}

impl MultiRawStats {
    fn new(
        design: &Mat<f64>,
        resp: &[f64],
        obs_cols: &[Vec<usize>],
        m_dim: usize,
        p: usize,
        weights: Option<&[f64]>,
    ) -> Self {
        let n = design.nrows();
        let mut ztz = vec![vec![0.0; m_dim]; m_dim];
        let mut ztx = vec![vec![0.0; p]; m_dim];
        let mut ztr = vec![0.0; m_dim];
        let mut xtx = vec![vec![0.0; p]; p];
        let mut xtr = vec![0.0; p];
        for i in 0..n {
            let w = weights.map_or(1.0, |w| w[i]);
            let ri = resp.get(i).copied().unwrap_or(0.0);
            let cols = &obs_cols[i];
            for &c in cols {
                ztr[c] += w * ri;
                for k in 0..p {
                    ztx[c][k] += w * design[(i, k)];
                }
                for &d in cols {
                    ztz[c][d] += w;
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
        Self {
            ztz,
            ztx,
            ztr,
            xtx,
            xtr,
        }
    }
}

/// Solution of the multi-factor penalized least-squares problem at fixed θ.
struct MultiSolution {
    beta: Vec<f64>,
    /// Random BLUPs per combined column, `b[c] = θ_{factor(c)}·u[c]`.
    b: Vec<f64>,
    /// Σ u² for the penalty term of the pRSS.
    u_sq: f64,
    logdet_l2: f64,
    logdet_rx2: f64,
    m_inv: Vec<Vec<f64>>,
}

/// Converged PIRLS solution for the multi-factor GLMM at fixed θ.
struct PirlsMultiSolution {
    beta: Vec<f64>,
    /// Random BLUPs per combined column.
    b: Vec<f64>,
    laplace_deviance: f64,
    m_inv: Vec<Vec<f64>>,
    converged: bool,
    iterations: usize,
}

/// Assemble the combined random block `A = D_θ ZᵀWZ D_θ + I`, the cross term
/// `B = D_θ ZᵀWX`, and `c = D_θ ZᵀWr` from raw stats and the per-factor θ.
/// Returns `(column-scales, A, B, c)`.
#[allow(clippy::type_complexity)]
fn multi_blocks(
    raw: &MultiRawStats,
    theta: &[f64],
    col_factor: &[usize],
    m_dim: usize,
    p: usize,
) -> (Vec<f64>, Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<f64>) {
    let tc: Vec<f64> = (0..m_dim).map(|c| theta[col_factor[c]].abs()).collect();
    let mut a = vec![vec![0.0; m_dim]; m_dim];
    for c in 0..m_dim {
        for d in 0..m_dim {
            a[c][d] = tc[c] * tc[d] * raw.ztz[c][d];
        }
        a[c][c] += 1.0;
    }
    let bmat: Vec<Vec<f64>> = (0..m_dim)
        .map(|c| (0..p).map(|k| tc[c] * raw.ztx[c][k]).collect())
        .collect();
    let cvec: Vec<f64> = (0..m_dim).map(|c| tc[c] * raw.ztr[c]).collect();
    (tc, a, bmat, cvec)
}

/// Solve for `(β, u)` given raw stats and θ (crossed / nested LMM step, or one
/// PIRLS iteration). `None` if the combined block or Schur complement is
/// singular.
fn solve_multi(
    raw: &MultiRawStats,
    theta: &[f64],
    col_factor: &[usize],
    m_dim: usize,
    p: usize,
) -> Option<MultiSolution> {
    let (tc, a, bmat, cvec) = multi_blocks(raw, theta, col_factor, m_dim, p);
    let chol_a = cholesky(&a)?;
    let logdet_l2 = 2.0 * (0..m_dim).map(|i| chol_a[i][i].ln()).sum::<f64>();
    let aib = chol_solve_mat(&chol_a, &bmat); // M×p
    let aic = cholesky_solve(&chol_a, &cvec); // M

    let mut mfix = raw.xtx.clone();
    let mut rhs = raw.xtr.clone();
    let btaib = mm(&transpose(&bmat), &aib);
    let btaic = matvec(&transpose(&bmat), &aic);
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

    let mut b = vec![0.0; m_dim];
    let mut u_sq = 0.0;
    for c in 0..m_dim {
        let dot: f64 = (0..p).map(|k| aib[c][k] * beta[k]).sum();
        let u = aic[c] - dot;
        u_sq += u * u;
        b[c] = tc[c] * u;
    }

    Some(MultiSolution {
        beta,
        b,
        u_sq,
        logdet_l2,
        logdet_rx2,
        m_inv,
    })
}

/// Residual sum of squares ‖y − Xβ − Zb‖² for the multi-factor LMM.
fn multi_prss(
    design: &Mat<f64>,
    y: &[f64],
    obs_cols: &[Vec<usize>],
    beta: &[f64],
    b: &[f64],
    p: usize,
) -> f64 {
    let n = design.nrows();
    let mut rss = 0.0;
    for i in 0..n {
        let mut fit = 0.0;
        for k in 0..p {
            fit += design[(i, k)] * beta[k];
        }
        for &c in &obs_cols[i] {
            fit += b[c];
        }
        let e = y[i] - fit;
        rss += e * e;
    }
    rss
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

/// PIRLS for the multi-factor GLMM: conditional modes `(β, u)` at fixed θ.
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
fn pirls_multi(
    family: &dyn GlmFamily,
    design: &Mat<f64>,
    y: &[f64],
    obs_cols: &[Vec<usize>],
    col_factor: &[usize],
    m_dim: usize,
    p: usize,
    theta: &[f64],
    beta0: &[f64],
    max_iter: usize,
    tol: f64,
) -> Option<PirlsMultiSolution> {
    let n = design.nrows();
    let tc: Vec<f64> = (0..m_dim).map(|c| theta[col_factor[c]].abs()).collect();

    let mut beta = beta0.to_vec();
    let mut u = vec![0.0; m_dim];
    let mut converged = false;
    let mut iterations = 0;
    let mut last_m_inv = vec![vec![0.0; p]; p];

    let eta_of = |beta: &[f64], u: &[f64], i: usize| -> f64 {
        let mut eta = 0.0;
        for k in 0..p {
            eta += design[(i, k)] * beta[k];
        }
        for &c in &obs_cols[i] {
            eta += tc[c] * u[c];
        }
        eta
    };

    for iter in 0..max_iter {
        iterations = iter + 1;
        let mut zeta = vec![0.0; n];
        let mut weights = vec![0.0; n];
        for i in 0..n {
            let eta = eta_of(&beta, &u, i);
            let mu = family.clamp_mu(family.link_inverse(eta));
            weights[i] = family.irls_weight(mu).max(1e-10);
            zeta[i] = family.working_response(y[i], mu, eta);
        }
        let raw = MultiRawStats::new(design, &zeta, obs_cols, m_dim, p, Some(&weights));
        let sol = solve_multi(&raw, theta, col_factor, m_dim, p)?;
        last_m_inv = sol.m_inv;

        // Recover u from b (= tc·u); tc is 0 only if a factor's θ hit 0.
        let mut max_change = 0.0f64;
        let mut u_new = vec![0.0; m_dim];
        for c in 0..m_dim {
            u_new[c] = if tc[c] > 0.0 { sol.b[c] / tc[c] } else { 0.0 };
            max_change = max_change.max((u_new[c] - u[c]).abs());
        }
        for k in 0..p {
            max_change = max_change.max((sol.beta[k] - beta[k]).abs());
        }
        beta = sol.beta;
        u = u_new;
        if max_change < tol {
            converged = true;
            break;
        }
    }

    // Laplace deviance at the mode.
    let mut disc = 0.0;
    let mut weights = vec![0.0; n];
    for i in 0..n {
        let eta = eta_of(&beta, &u, i);
        let mu = family.clamp_mu(family.link_inverse(eta));
        weights[i] = family.irls_weight(mu).max(1e-10);
        disc += family.unit_deviance(y[i], mu);
    }
    let u_sq: f64 = u.iter().map(|&v| v * v).sum();
    let raw = MultiRawStats::new(design, &vec![0.0; n], obs_cols, m_dim, p, Some(&weights));
    let (_, a, _, _) = multi_blocks(&raw, theta, col_factor, m_dim, p);
    let chol_a = cholesky(&a)?;
    let logdet_l2 = 2.0 * (0..m_dim).map(|i| chol_a[i][i].ln()).sum::<f64>();
    let b: Vec<f64> = (0..m_dim).map(|c| tc[c] * u[c]).collect();

    Some(PirlsMultiSolution {
        beta,
        b,
        laplace_deviance: disc + u_sq + logdet_l2,
        m_inv: last_m_inv,
        converged,
        iterations,
    })
}

/// Build per-factor summaries (variance component + BLUPs) from the combined
/// random-effect vector.
fn build_factor_summaries(
    b: &[f64],
    col_offset: &[usize],
    n_levels: &[usize],
    n_factors: usize,
    theta: &[f64],
    sigma: f64,
) -> Vec<FactorSummary> {
    (0..n_factors)
        .map(|f| {
            let start = col_offset[f];
            let jf = n_levels[f];
            FactorSummary {
                n_levels: jf,
                sd: theta[f].abs() * sigma,
                blups: b[start..start + jf].to_vec(),
            }
        })
        .collect()
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

/// Optimise the profiled deviance over one variance-ratio `θ_f` per grouping
/// factor (crossed / nested case) by Nelder–Mead.
fn optimize_theta_vec(
    n_factors: usize,
    tol: f64,
    f: &dyn Fn(&[f64]) -> Option<f64>,
) -> Result<Vec<f64>, RegressionError> {
    let x0 = vec![1.0; n_factors];
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
