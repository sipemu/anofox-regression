# anofox-regression API Reference

This comprehensive API documentation covers all regression methods, distributions, and return types for the anofox-regression library.

## Core Regression Methods

**Linear Regression** includes `OlsRegressor` (Ordinary Least Squares with full inference), `WlsRegressor` (Weighted Least Squares), `RidgeRegressor` (L2 regularization), `ElasticNetRegressor` (L1+L2 via L-BFGS optimization), `HuberRegressor` (robust regression with Huber loss, resistant to outliers), and `RlsRegressor` (Recursive Least Squares for online learning).

**Constrained Regression** provides `BlsRegressor` for bounded least squares with box constraints and non-negative least squares (NNLS).

**Generalized Linear Models** include `LogisticRegression` (binary classifier with sklearn-like API: `predict`, `predict_proba`, `decision_function`, `score`), `PoissonRegressor` (count data with log/identity/sqrt links), `BinomialRegressor` (logistic/probit/cloglog), `NegativeBinomialRegressor` (overdispersed counts with theta estimation), and `TweedieRegressor` (Gaussian, Poisson, Gamma, Inverse-Gaussian, Compound Poisson-Gamma).

**Generalized Linear Mixed Models** via `GlmmRegressor` fit a random intercept, optional random slopes, and crossed / nested factors, `g(μ_ij) = x_ij'β + z_ij'b_j` with `b_j ~ N(0, Σ)` and unstructured `Σ`. `GlmmRegressor::gaussian()` is a linear mixed model fit by profiled REML/ML (matches `lme4::lmer`); `GlmmRegressor::poisson()` / `GlmmRegressor::binomial()` use Laplace-approximate PIRLS (match `glmer(nAGQ = 0)`). `random_slopes(cols)` names the `x` columns carrying a random slope. `fit(x, y, group)` handles a single grouping factor; `fit_crossed(x, y, &[&a, &b, …])` handles crossed `(1|a) + (1|b)` and nested `(1|a/b)` (via the `a:b` interaction id) random intercepts. `FittedGlmm` exposes fixed effects and their standard errors, per-group random-effect BLUPs (`random_effects` / `random_effects_matrix`), the covariance (`random_cov` / `random_sd` / `random_corr`), per-factor summaries for crossed/nested fits (`factors` / `n_factors`), the scalar variance components (`theta`, `sigma`, `sd_random`, `var_random`), `deviance` / `log_likelihood`, and `converged`.

**Augmented Linear Models** via `AlmRegressor` support 24 distribution families: Normal, Laplace, Student-t, Logistic, Asymmetric Laplace, Generalised Normal, S, Log-Normal, Log-Laplace, Log-S, Log-Generalised Normal, Gamma, Inverse Gaussian, Exponential, Folded Normal, Rectified Normal, Beta, Logit-Normal, Poisson, Negative Binomial, Binomial, Geometric, Cumulative Logistic, Cumulative Normal, and Box-Cox Normal.

**Dynamic Models** include `LmDynamic` for time-varying coefficient regression using pointwise information criteria.

## Smoothing & Classification

**LOWESS** (`lowess_smooth`) provides locally weighted scatterplot smoothing with configurable bandwidth.

**AID** (`AidClassifier`) implements Automatic Identification of Demand for classifying demand patterns (regular vs intermittent, count vs fractional) with distribution recommendation and optimized closed-form fitting.

## Loss Functions

Built-in loss functions include `mae()` (Mean Absolute Error), `mse()` (Mean Squared Error), `rmse()` (Root Mean Squared Error), `mape()` (Mean Absolute Percentage Error), `smape()` (Symmetric MAPE), `mase()` (Mean Absolute Scaled Error), and `pinball_loss()` for quantile regression.

## Enums and Configuration

Key enums control regression behavior:
- `LinkFunction`: Identity, Log, Logit, Probit, Inverse, Sqrt, Cloglog
- `AlmDistribution`: 24 distribution families for ALM
- `IntervalType`: Prediction or Confidence intervals
- `PredictionType`: Response or Link scale predictions
- `LambdaScaling`: Raw or Glmnet (λ×n) scaling convention
- `NaAction`: Omit, Exclude, Fail, or Pass for missing value handling

## Result Structures

**RegressionResult** contains coefficients, intercept, standard errors, t-statistics, p-values, confidence intervals, R², adjusted R², MSE, RMSE, F-statistic, AIC, AICc, BIC, log-likelihood, residuals, and fitted values.

**GLM Results** (Poisson, Binomial, NegativeBinomial, Tweedie) add deviance, null deviance, dispersion, iteration count, and a `converged` flag. NegativeBinomial includes the estimated theta parameter. By default a non-converged IRLS fit returns `Err(ConvergenceFailed)`; call `error_on_non_convergence(false)` on the builder to instead return the last iterate with `converged == false`. `FittedGamma` exposes `converged()` (it wraps Tweedie).

**GLMM Results** (`FittedGlmm`) contain fixed effects and their standard errors, per-group random-intercept BLUPs, variance components (`theta = σ_b/σ`, `sigma`, `sd_random`, `var_random`), deviance, log-likelihood, group count, and a `converged` flag.

**ALM Results** include log-likelihood, scale parameter, and distribution-specific diagnostics.

## Diagnostics

Diagnostic functions include `compute_leverage()` (hat values), `cooks_distance()`, `studentized_residuals()`, `standardized_residuals()`, `dffits()`, and `variance_inflation_factor()` (VIF) for multicollinearity detection.

## Prediction

All fitted models implement `FittedRegressor` trait with:
- `predict(&x)`: Point predictions
- `predict_with_interval(&x, interval_type, level)`: Predictions with confidence/prediction intervals
- `result()`: Access full regression results

GLM models add `predict_with_se()` for predictions with standard errors on response or link scale.
