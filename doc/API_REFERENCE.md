# anofox-regression API Reference

This comprehensive API documentation covers all regression methods, distributions, and return types for the anofox-regression library.

## Core Regression Methods

**Linear Regression** includes `OlsRegressor` (Ordinary Least Squares with full inference), `WlsRegressor` (Weighted Least Squares), `RidgeRegressor` (L2 regularization), `ElasticNetRegressor` (L1+L2 via L-BFGS optimization), `HuberRegressor` (robust regression with Huber loss, resistant to outliers), and `RlsRegressor` (Recursive Least Squares for online learning).

**Constrained Regression** provides `BlsRegressor` for bounded least squares with box constraints and non-negative least squares (NNLS).

**Generalized Linear Models** include `LogisticRegression` (binary classifier with sklearn-like API: `predict`, `predict_proba`, `decision_function`, `score`), `PoissonRegressor` (count data with log/identity/sqrt links), `BinomialRegressor` (logistic/probit/cloglog), `NegativeBinomialRegressor` (overdispersed counts with theta estimation), and `TweedieRegressor` (Gaussian, Poisson, Gamma, Inverse-Gaussian, Compound Poisson-Gamma).

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

**GLM Results** (Poisson, Binomial, NegativeBinomial, Tweedie) add deviance, null deviance, dispersion, and iteration count. NegativeBinomial includes estimated theta parameter.

**ALM Results** include log-likelihood, scale parameter, and distribution-specific diagnostics.

## Diagnostics

Diagnostic functions include `compute_leverage()` (hat values), `cooks_distance()`, `studentized_residuals()`, `standardized_residuals()`, `dffits()`, and `variance_inflation_factor()` (VIF) for multicollinearity detection.

## Prediction

All fitted models implement `FittedRegressor` trait with:
- `predict(&x)`: Point predictions
- `predict_with_interval(&x, interval_type, level)`: Predictions with confidence/prediction intervals
- `result()`: Access full regression results

GLM models add `predict_with_se()` for predictions with standard errors on response or link scale.
