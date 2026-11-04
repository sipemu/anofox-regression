# The statistician's workflow (Harrell, *Regression Modeling Strategies*)

This is the methodology the skill follows. It distils Harrell's *Regression Modeling Strategies* (2nd ed., 2015, abbreviated RMS) into working rules. The software reference is in the other file. This file is about *how to think*.

If a Markdown copy of the book is on disk (for example `~/projects/datascience/statistical_modelling/docs_md/`, one file per chapter), read the relevant chapter when you need more depth than this summary gives.

## Contents
1. Start from the question
2. Effective sample size and the d.f. budget
3. Missing data
4. Specifying predictor complexity
5. Choosing the model family by response type
6. Fitting and checking
7. Inference: what to test, and how
8. Validation and overfitting
9. Interpreting and reporting
10. Anti-patterns to refuse (politely) and what to do instead
11. Computing things the software does not ship yet

---

## 1. Start from the question

Ask, or infer and state, which of the three goals applies. The goal changes the strategy (RMS §4.12):

| Goal | Parsimony | Validation | Shrinkage of the variable of interest |
|---|---|---|---|
| **Prediction** (risk scores, forecasts) | Matters little. Accuracy and calibration matter | Essential: bootstrap optimism, calibration | Yes, shrink everything |
| **Effect estimation** (how much does X change Y, adjusted?) | Keep adjustment variables even if "insignificant" | Optional | Do not shrink the effect of interest |
| **Hypothesis testing** (is there any effect of X?) | Full prespecified model | Optional | Do not shrink tested terms |

Also pin down the unit of analysis, the response and its type, any clustering or repeated measures, censoring, and what decision the analysis feeds.

## 2. Effective sample size and the d.f. budget

Overfitting is governed by the *limiting sample size* m, not by n (RMS Table 4.1):

| Response | m |
|---|---|
| Continuous | n |
| Binary | min(n₁, n₀) |
| Ordinal, k categories | n − (1/n²) Σ nᵢ³ |
| Survival time | number of events |

Rule of thumb: the number of **candidate** parameters p (every nonlinear term, interaction and dummy; also every variable you *looked at*) should be at most m/15, and m/10 at the most liberal. Below that you should:
- reduce dimensionality **without looking at Y**: drop redundant variables, cluster the variables and score each cluster with a principal component, combine predictors using subject-matter knowledge; or
- use penalized estimation (ridge / elastic net / Gaussian priors), which reduces the effective d.f.

Decide **before fitting** how many d.f. each predictor gets. Give more to predictors believed to be important and to those with large expected nonlinearity. Then spend that budget once (RMS §4.1).

Post-fit check: the heuristic shrinkage factor γ̂ = (LR χ² − p) / LR χ², where p is the number of non-intercept parameters. If γ̂ < 0.9 the model is overfitted and will need shrinkage or data reduction (RMS §4.5).

## 3. Missing data

- Quantify the missingness first: the fraction of rows with any missing value, and per variable. Look for patterns, such as variables that are missing together.
- Complete-case analysis (`NaAction::Omit`, or SQL row dropping) is acceptable only when **< ~3–5 %** of rows are incomplete *and* the data are plausibly MCAR. Otherwise it is biased and inefficient.
- The default is **multiple imputation** (m ≥ 5, more when much is missing). Use predictive mean matching, with **Y included in the imputation model**, and pool with Rubin's rules (RMS Ch 3).
- Never impute Y for the analysis model, and never use mean imputation followed by inference as if the values were observed.
- Report how many observations were dropped, and why.

## 4. Specifying predictor complexity

- **Do not dichotomize continuous predictors** (no "age > 65"). Categorizing loses information, mis-models the relationship and invites cutpoint hacking.
- Default nonlinear representation: **restricted cubic splines** with k knots at fixed quantiles (RMS Table 2.3). Use k = 3 for small samples, 4 as the usual default, and 5 for large n or important predictors:

  | k | knot quantiles |
  |---|---|
  | 3 | .10 .50 .90 |
  | 4 | .05 .35 .65 .95 |
  | 5 | .05 .275 .50 .725 .95 |
  | 6 | .05 .23 .41 .59 .77 .95 |
  | 7 | .025 .1833 .3417 .50 .6583 .8167 .975 |

  A k-knot spline costs k − 1 d.f. (`splines::Rcs`; Section 11 gives the basis formula).
- Prespecify interactions from subject-matter knowledge, for example treatment × a key covariate, or age × sex. Test them as a **pooled** chunk. Drop them only if the pooled test is decisive (P > 0.3).
- Linearity tests are for demonstrating nonlinearity, not for deciding to drop spline terms. Keeping the prespecified complexity keeps the inference honest.

## 5. Choosing the model family by response type

| Response | First choice | Alternatives / when |
|---|---|---|
| Continuous, roughly symmetric residuals | OLS | Robust (Huber, Theil–Sen) if there are outliers. WLS for known heteroscedasticity. Quantile regression for conditional quantiles |
| Continuous, skewed / heteroscedastic | Model log Y (smearing for the mean), Gamma / Tweedie GLM | Ordinal regression (semiparametric, RMS Ch 15) when no transformation is obviously right |
| Binary | Logistic (binomial, logit) | Probit / cloglog when there is a reason. Penalize when m/p is small. Check for (quasi-)separation |
| Counts | Poisson with offset for exposure | Negative binomial if overdispersed (Pearson χ²/df ≫ 1). Tweedie when there are many zeros plus positive mass |
| Ordinal | Proportional odds | Continuation ratio. Check the PO assumption |
| Time to event, right-censored | Cox PH | Parametric AFT (Weibull, log-normal, log-logistic) when you need extrapolation or quantiles, or PH fails |
| Repeated measures / clusters | GLS with a correlation structure, or a mixed model (GLMM) | Cluster-robust SE as a cheap fix for marginal inference |
| Proportions in (0,1), positive skewed | Beta / Gamma / log-normal (ALM families) | — |

Never analyse censored times as if they were complete, and never drop the censored observations.

## 6. Fitting and checking

After every fit, in this order:
1. **Did it converge?** Check the iteration count, the convergence flag, and whether any coefficients are aliased (NaN or dropped columns). Check for separation in logistic models.
2. **Collinearity** only matters for interpreting individual coefficients, not for prediction. Look at VIF and the condition number. Use pooled tests for groups of collinear predictors instead of reading individual p-values.
3. **Influence:** leverage h_ii > 2p/n, Cook's D, DFBETAS. Investigate influential points (data errors?). Do not delete them silently.
4. **Distributional assumptions:**
   - OLS: residuals vs fitted (variance), QQ-plot (only important for prediction intervals with small n), Breusch–Pagan / White tests.
   - GLM: deviance or Pearson residuals, dispersion φ̂ = Pearson χ² / (n − p) (≫ 1 means overdispersion), partial residual plots for functional form.
   - Survival: check PH (Schoenfeld) for Cox. For AFT, compare against Kaplan–Meier.
5. **Functional form:** partial residuals / added-variable plots, or a spline fit, for any predictor still modelled linearly.
6. **Heteroscedasticity or clustering:** use robust (HC3, or cluster) standard errors, or model the variance explicitly.

## 7. Inference: what to test, and how

- Prefer **likelihood-ratio** tests and profile CIs over Wald tests for GLMs with small m or large effects. Wald tests are poor near separation and for odds ratios far from 1.
- Test **predictors**, not columns: the overall effect of a spline variable is a multi-d.f. test of all its columns, including its interactions. "Nonlinearity" is the test of the spline columns beyond the linear one.
- Rank predictor importance by χ² − d.f. of the partial test, not by p-values or standardized coefficients.
- Use the dispersion conventions R uses: fixed φ = 1 for Poisson / binomial / NB, estimated φ (t / F tests) for Gaussian / Gamma / Tweedie.
- Multiple comparisons: say how many tests were run. Prefer one prespecified global test over many separate ones.
- Report effect sizes with CIs on an interpretable scale: inter-quartile-range odds, hazard or time ratios for continuous X, and contrasts against a reference level for categories. Leave raw coefficients of spline bases out.

## 8. Validation and overfitting

- Apparent performance is optimistic. For prediction goals, estimate optimism with the **bootstrap**: fit on a resample, evaluate on the resample and on the original data, and repeat B ≈ 200–300 times. Corrected index = apparent − mean optimism. Every data-driven step (selection, transformation choice) must be repeated in each resample.
- Hold-out splits waste data and are unstable when n < ~20 000. Use them only when the whole modelling process cannot be automated. Use time-series splits for temporal data.
- Indexes:
  - Discrimination: c / Somers' Dxy, R² (Nagelkerke for binary), g-index.
  - Calibration: calibration intercept and slope (slope < 1 means overfitting), Emax / Eavg from a smoothed calibration curve.
  - Overall: Brier score.
- If the calibration slope is < 0.9, shrink the coefficients (uniformly by γ̂, or by penalization).

## 9. Interpreting and reporting

The report should let a reader reproduce and judge the analysis:
1. Question, design, and the goal (prediction / estimation / testing).
2. Data: n, m, missingness and how it was handled, exclusions.
3. Model specification and its *a priori* justification: d.f. per predictor, knots, prespecified interactions.
4. Fit summary: effects with 95 % CI on an interpretable scale, pooled tests (ANOVA-style table), model LR χ², R² / c.
5. Diagnostics performed and what they showed.
6. Validation (optimism-corrected indexes, calibration) for prediction models.
7. Limitations: what could not be done with the available tools, and any assumptions that are doubtful.

Use plain language and keep p-values in context. Never write "no effect" for P > 0.05. Write "the data are compatible with effects from A to B".

## 10. Anti-patterns to refuse (politely) and what to do instead

| Request | Problem | Do instead |
|---|---|---|
| Stepwise / univariable screening then fit | Biased coefficients, invalid SEs and P, unstable selection, optimistic R² | Prespecify. Use data reduction blinded to Y, or penalization (elastic net / ridge). If selection is unavoidable, bootstrap the whole process and report how unstable it is |
| Dichotomize a continuous X or Y | Information loss, residual confounding | Splines for X. Ordinal / continuous models for Y |
| Drop "non-significant" terms | Invalidates CIs; does not improve prediction | Keep the prespecified model. Use pooled tests |
| Report training-set R² / accuracy as performance | Optimistic | Bootstrap optimism-corrected indexes |
| Classification accuracy at 0.5 for a risk model | An improper scoring rule that ignores calibration | Brier score, log-loss, c-index, calibration curve. Pick thresholds with a decision analysis |
| Complete-case analysis with 30 % missing | Bias and lost power | Multiple imputation |
| Treat censored times as events, or drop them | Biased survival estimates | KM / Cox / AFT |
| SMOTE / class re-balancing for a probability model | Destroys calibration | Model the true prevalence. Adjust thresholds afterwards |

Voice the concern, explain it in one or two sentences, offer the better approach, and still help if the user insists (with the caveat recorded in the report).

## 11. What the formulas behind the tools are

The Rust crate (0.11) ships all of the computations below; use its functions and keep these formulas as background for checking and explaining results: `splines::Rcs` (spline basis), `inference::{wald_test, wald_chunk, anova, lr_test}` (Wald / LR tests), `validation::{validate_lrm, validate_ols, predab_resample}` (bootstrap optimism), `diagnostics::accuracy::{somers2, concordance, lrm_stats, val_prob}` (c, Dxy, calibration), `inference::pool_rubin` (Rubin's rules) and `transform::smearing_estimate`. Other anofox tools (e.g. the DuckDB extension) may still lack some of them; their skills list the gaps.

**Restricted cubic spline basis** (Harrell parametrisation, k knots t₁ < … < t_k). This gives k − 2 nonlinear columns, plus x itself:

```
d(x)_+ = max(x, 0)
X_j = [ (x − t_j)_+³
        − (x − t_{k−1})_+³ · (t_k − t_j)/(t_k − t_{k−1})
        + (x − t_k)_+³     · (t_{k−1} − t_j)/(t_k − t_{k−1}) ] / (t_k − t_1)²
for j = 1 … k−2
```

Store the knots and reuse them for prediction on new data.

**Multi-d.f. Wald test** of H₀: Cβ = 0 (q rows), given β̂ and V̂ from the fit:
W = (Cβ̂)ᵀ (C V̂ Cᵀ)⁻¹ (Cβ̂) ~ χ²(q). With an estimated dispersion, use F = W/q on (q, n − p) d.f. Alternatively, fit the reduced model and use the LR test: 2(ℓ_full − ℓ_reduced) ~ χ²(Δp).

**Bootstrap optimism** (any model, any index S):
```
S_app = S(fit(D), D)
repeat b = 1..B:  D_b = resample(D); f_b = fit(D_b)          # include every data-driven step
                  O_b = S(f_b, D_b) − S(f_b, D)
S_corrected = S_app − mean(O_b)
```

**c-index / Somers' Dxy** for a binary y and a score s: c = P(s_i > s_j | y_i = 1, y_j = 0), with ties counting ½. This equals the Mann–Whitney U / (n₁n₀), and Dxy = 2(c − ½).

**Calibration slope:** fit a logistic model of y on the linear predictor (logit p̂). The slope should be ≈ 1.

**Rubin's rules** for m imputations: Q̄ = mean(Q_i), Ū = mean(U_i), B = var(Q_i), T = Ū + (1 + 1/m)B, ν = (m − 1)(1 + Ū/((1 + 1/m)B))².

**Smearing estimate** of E[Y] after fitting log Y: exp(x'β̂) · mean(exp(residuals)).
