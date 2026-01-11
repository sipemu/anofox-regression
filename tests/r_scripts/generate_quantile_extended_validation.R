#!/usr/bin/env Rscript
# Extended Validation Data for Quantile Regression
# Based on R quantreg package examples: Engel, Stackloss datasets
# Plus edge cases for extreme quantiles and outlier robustness

# Install quantreg package if not available
if (!require("quantreg", quietly = TRUE)) {
  install.packages("quantreg", repos = "https://cloud.r-project.org")
  library(quantreg)
}

set.seed(42)

cat("// =============================================================================\n")
cat("// Extended Quantile Regression Validation Data Generated from R\n")
cat(sprintf("// Generated on: %s\n", format(Sys.time())))
cat(sprintf("// R version: %s\n", R.version.string))
cat(sprintf("// quantreg package version: %s\n", packageVersion("quantreg")))
cat("// =============================================================================\n\n")

# -----------------------------------------------------------------------------
# Test 1: Engel Food Expenditure Dataset (Classic Heteroscedastic Example)
# From Koenker & Bassett (1982) - 235 Belgian household observations
# -----------------------------------------------------------------------------
cat("// -----------------------------------------------------------------------------\n")
cat("// Test 1: Engel Food Expenditure Dataset\n")
cat("// Classic heteroscedastic example from Koenker & Bassett (1982)\n")
cat("// 235 Belgian working class households: food expenditure vs income\n")
cat("// -----------------------------------------------------------------------------\n")

data(engel)
n_engel <- nrow(engel)

# Fit multiple quantiles
taus_engel <- c(0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)

cat(sprintf("const N_ENGEL: usize = %d;\n", n_engel))
cat("#[rustfmt::skip]\n")
cat(sprintf("const X_ENGEL: [f64; %d] = [\n    ", n_engel))
cat(paste(sprintf("%.6f", engel$income), collapse = ", "))
cat("\n];\n")
cat("#[rustfmt::skip]\n")
cat(sprintf("const Y_ENGEL: [f64; %d] = [\n    ", n_engel))
cat(paste(sprintf("%.6f", engel$foodexp), collapse = ", "))
cat("\n];\n\n")

for (tau in taus_engel) {
  fit <- rq(foodexp ~ income, data = engel, tau = tau)
  coefs <- coef(fit)
  tau_str <- sprintf("%02d", as.integer(tau * 100))
  cat(sprintf("// tau = %.2f\n", tau))
  cat(sprintf("const EXPECTED_ENGEL_INTERCEPT_TAU%s: f64 = %.10f;\n", tau_str, coefs[1]))
  cat(sprintf("const EXPECTED_ENGEL_SLOPE_TAU%s: f64 = %.10f;\n", tau_str, coefs[2]))
}
cat("\n")

# Also compute OLS for comparison (to show heteroscedasticity)
fit_ols <- lm(foodexp ~ income, data = engel)
cat(sprintf("// OLS for comparison (to show heteroscedasticity effect)\n"))
cat(sprintf("const EXPECTED_ENGEL_OLS_INTERCEPT: f64 = %.10f;\n", coef(fit_ols)[1]))
cat(sprintf("const EXPECTED_ENGEL_OLS_SLOPE: f64 = %.10f;\n", coef(fit_ols)[2]))
cat("\n")

# -----------------------------------------------------------------------------
# Test 2: Stackloss Dataset (Multivariate with Exact Fit Edge Case)
# 21 observations, 3 predictors
# Known: 8 of 21 points lie exactly on the Q25 regression plane
# -----------------------------------------------------------------------------
cat("// -----------------------------------------------------------------------------\n")
cat("// Test 2: Stackloss Dataset (Multivariate)\n")
cat("// 21 observations from stack loss plant operations\n")
cat("// Known edge case: 8 of 21 points lie exactly on Q25 plane\n")
cat("// -----------------------------------------------------------------------------\n")

data(stackloss)
n_stack <- nrow(stackloss)
p_stack <- 3  # Air.Flow, Water.Temp, Acid.Conc.

cat(sprintf("const N_STACK: usize = %d;\n", n_stack))
cat(sprintf("const P_STACK: usize = %d;\n", p_stack))

# Output X matrix (Air.Flow, Water.Temp, Acid.Conc.) in column-major order
X_stack <- as.matrix(stackloss[, 1:3])
cat("#[rustfmt::skip]\n")
cat(sprintf("const X_STACK: [f64; %d] = [\n    ", n_stack * p_stack))
cat(paste(sprintf("%.6f", as.vector(X_stack)), collapse = ", "))
cat("\n];\n")

cat("#[rustfmt::skip]\n")
cat(sprintf("const Y_STACK: [f64; %d] = [\n    ", n_stack))
cat(paste(sprintf("%.6f", stackloss$stack.loss), collapse = ", "))
cat("\n];\n\n")

# Fit at tau = 0.25 (the famous edge case)
fit_stack_25 <- rq(stack.loss ~ Air.Flow + Water.Temp + Acid.Conc.,
                   data = stackloss, tau = 0.25)
coefs_25 <- coef(fit_stack_25)
cat("// tau = 0.25 (8 of 21 points lie exactly on this plane)\n")
cat(sprintf("const EXPECTED_STACK_INTERCEPT_TAU25: f64 = %.10f;\n", coefs_25[1]))
cat(sprintf("const EXPECTED_STACK_AIRFLOW_TAU25: f64 = %.10f;\n", coefs_25[2]))
cat(sprintf("const EXPECTED_STACK_WATERTEMP_TAU25: f64 = %.10f;\n", coefs_25[3]))
cat(sprintf("const EXPECTED_STACK_ACIDCONC_TAU25: f64 = %.10f;\n", coefs_25[4]))

# Fit at tau = 0.50 (median)
fit_stack_50 <- rq(stack.loss ~ Air.Flow + Water.Temp + Acid.Conc.,
                   data = stackloss, tau = 0.50)
coefs_50 <- coef(fit_stack_50)
cat("// tau = 0.50 (median)\n")
cat(sprintf("const EXPECTED_STACK_INTERCEPT_TAU50: f64 = %.10f;\n", coefs_50[1]))
cat(sprintf("const EXPECTED_STACK_AIRFLOW_TAU50: f64 = %.10f;\n", coefs_50[2]))
cat(sprintf("const EXPECTED_STACK_WATERTEMP_TAU50: f64 = %.10f;\n", coefs_50[3]))
cat(sprintf("const EXPECTED_STACK_ACIDCONC_TAU50: f64 = %.10f;\n", coefs_50[4]))

# Fit at tau = 0.75
fit_stack_75 <- rq(stack.loss ~ Air.Flow + Water.Temp + Acid.Conc.,
                   data = stackloss, tau = 0.75)
coefs_75 <- coef(fit_stack_75)
cat("// tau = 0.75\n")
cat(sprintf("const EXPECTED_STACK_INTERCEPT_TAU75: f64 = %.10f;\n", coefs_75[1]))
cat(sprintf("const EXPECTED_STACK_AIRFLOW_TAU75: f64 = %.10f;\n", coefs_75[2]))
cat(sprintf("const EXPECTED_STACK_WATERTEMP_TAU75: f64 = %.10f;\n", coefs_75[3]))
cat(sprintf("const EXPECTED_STACK_ACIDCONC_TAU75: f64 = %.10f;\n", coefs_75[4]))
cat("\n")

# -----------------------------------------------------------------------------
# Test 3: Extreme Quantiles (0.01, 0.05, 0.95, 0.99)
# Tests numerical stability at distribution tails
# -----------------------------------------------------------------------------
cat("// -----------------------------------------------------------------------------\n")
cat("// Test 3: Extreme Quantiles (0.01, 0.05, 0.95, 0.99)\n")
cat("// Tests numerical stability at distribution tails\n")
cat("// -----------------------------------------------------------------------------\n")

n_ext <- 500  # Large sample for extreme quantiles
x_ext <- runif(n_ext, 0, 100)
y_ext <- 10 + 0.5 * x_ext + rnorm(n_ext, sd = 10)

cat(sprintf("const N_EXTREME: usize = %d;\n", n_ext))
cat("#[rustfmt::skip]\n")
cat(sprintf("const X_EXTREME: [f64; %d] = [\n    ", n_ext))
cat(paste(sprintf("%.10f", x_ext), collapse = ", "))
cat("\n];\n")
cat("#[rustfmt::skip]\n")
cat(sprintf("const Y_EXTREME: [f64; %d] = [\n    ", n_ext))
cat(paste(sprintf("%.10f", y_ext), collapse = ", "))
cat("\n];\n\n")

for (tau in c(0.01, 0.05, 0.95, 0.99)) {
  fit <- rq(y_ext ~ x_ext, tau = tau)
  coefs <- coef(fit)
  tau_str <- sprintf("%02d", as.integer(tau * 100))
  cat(sprintf("// tau = %.2f\n", tau))
  cat(sprintf("const EXPECTED_EXTREME_INTERCEPT_TAU%s: f64 = %.10f;\n", tau_str, coefs[1]))
  cat(sprintf("const EXPECTED_EXTREME_SLOPE_TAU%s: f64 = %.10f;\n", tau_str, coefs[2]))
}
cat("\n")

# -----------------------------------------------------------------------------
# Test 4: Outlier Robustness Comparison (Median vs Mean)
# Demonstrates breakdown point advantage of quantile regression
# -----------------------------------------------------------------------------
cat("// -----------------------------------------------------------------------------\n")
cat("// Test 4: Outlier Robustness (Median vs OLS)\n")
cat("// Demonstrates robustness advantage of quantile regression\n")
cat("// -----------------------------------------------------------------------------\n")

n_out <- 50
x_out <- seq(1, 10, length.out = n_out)
y_clean <- 2 + 1.5 * x_out + rnorm(n_out, sd = 0.5)

# Add 3 extreme outliers (6% contamination)
outlier_idx <- c(10, 25, 40)
y_outlier <- y_clean
y_outlier[outlier_idx] <- y_outlier[outlier_idx] + c(50, -40, 60)  # Large outliers

cat(sprintf("const N_OUTLIER: usize = %d;\n", n_out))
cat("#[rustfmt::skip]\n")
cat(sprintf("const X_OUTLIER: [f64; %d] = [\n    ", n_out))
cat(paste(sprintf("%.10f", x_out), collapse = ", "))
cat("\n];\n")
cat("#[rustfmt::skip]\n")
cat(sprintf("const Y_OUTLIER: [f64; %d] = [\n    ", n_out))
cat(paste(sprintf("%.10f", y_outlier), collapse = ", "))
cat("\n];\n")
cat("#[rustfmt::skip]\n")
cat(sprintf("const Y_CLEAN: [f64; %d] = [\n    ", n_out))
cat(paste(sprintf("%.10f", y_clean), collapse = ", "))
cat("\n];\n\n")

# Median regression on contaminated data
fit_median <- rq(y_outlier ~ x_out, tau = 0.5)
cat("// Median regression on contaminated data\n")
cat(sprintf("const EXPECTED_OUTLIER_MEDIAN_INTERCEPT: f64 = %.10f;\n", coef(fit_median)[1]))
cat(sprintf("const EXPECTED_OUTLIER_MEDIAN_SLOPE: f64 = %.10f;\n", coef(fit_median)[2]))

# OLS on contaminated data (for comparison - will be more affected)
fit_ols_out <- lm(y_outlier ~ x_out)
cat("// OLS regression on contaminated data (for comparison)\n")
cat(sprintf("const EXPECTED_OUTLIER_OLS_INTERCEPT: f64 = %.10f;\n", coef(fit_ols_out)[1]))
cat(sprintf("const EXPECTED_OUTLIER_OLS_SLOPE: f64 = %.10f;\n", coef(fit_ols_out)[2]))

# True coefficients for reference
cat("// True coefficients: intercept=2.0, slope=1.5\n")
cat("const TRUE_OUTLIER_INTERCEPT: f64 = 2.0;\n")
cat("const TRUE_OUTLIER_SLOPE: f64 = 1.5;\n")
cat("\n")

# -----------------------------------------------------------------------------
# Test 5: Heavy-Tailed Distribution (t-distribution errors)
# Quantile regression is optimal for non-Gaussian errors
# -----------------------------------------------------------------------------
cat("// -----------------------------------------------------------------------------\n")
cat("// Test 5: Heavy-Tailed Distribution (t-distribution errors)\n")
cat("// Quantile regression is optimal for non-Gaussian errors\n")
cat("// -----------------------------------------------------------------------------\n")

n_heavy <- 100
x_heavy <- runif(n_heavy, 0, 10)
# t-distribution with df=3 (heavy tails)
y_heavy <- 5 + 2 * x_heavy + rt(n_heavy, df = 3) * 2

cat(sprintf("const N_HEAVY: usize = %d;\n", n_heavy))
cat("#[rustfmt::skip]\n")
cat(sprintf("const X_HEAVY: [f64; %d] = [\n    ", n_heavy))
cat(paste(sprintf("%.10f", x_heavy), collapse = ", "))
cat("\n];\n")
cat("#[rustfmt::skip]\n")
cat(sprintf("const Y_HEAVY: [f64; %d] = [\n    ", n_heavy))
cat(paste(sprintf("%.10f", y_heavy), collapse = ", "))
cat("\n];\n\n")

fit_heavy <- rq(y_heavy ~ x_heavy, tau = 0.5)
cat(sprintf("const EXPECTED_HEAVY_INTERCEPT: f64 = %.10f;\n", coef(fit_heavy)[1]))
cat(sprintf("const EXPECTED_HEAVY_SLOPE: f64 = %.10f;\n", coef(fit_heavy)[2]))
cat("\n")

# -----------------------------------------------------------------------------
# Test 6: Small Sample (n=10) - Edge case for minimum observations
# -----------------------------------------------------------------------------
cat("// -----------------------------------------------------------------------------\n")
cat("// Test 6: Small Sample (n=10)\n")
cat("// Edge case for minimum observations\n")
cat("// -----------------------------------------------------------------------------\n")

n_small <- 10
x_small <- 1:10
y_small <- 3 + 0.5 * x_small + rnorm(n_small, sd = 0.3)

cat(sprintf("const N_SMALL: usize = %d;\n", n_small))
cat("#[rustfmt::skip]\n")
cat(sprintf("const X_SMALL: [f64; %d] = [\n    ", n_small))
cat(paste(sprintf("%.10f", as.numeric(x_small)), collapse = ", "))
cat("\n];\n")
cat("#[rustfmt::skip]\n")
cat(sprintf("const Y_SMALL: [f64; %d] = [\n    ", n_small))
cat(paste(sprintf("%.10f", y_small), collapse = ", "))
cat("\n];\n\n")

fit_small <- rq(y_small ~ x_small, tau = 0.5)
cat(sprintf("const EXPECTED_SMALL_INTERCEPT: f64 = %.10f;\n", coef(fit_small)[1]))
cat(sprintf("const EXPECTED_SMALL_SLOPE: f64 = %.10f;\n", coef(fit_small)[2]))
cat("\n")

# -----------------------------------------------------------------------------
# Test 7: Crossing Quantiles Check (Monotonicity in tau)
# Fitted values at higher quantiles should be >= lower quantiles (mostly)
# -----------------------------------------------------------------------------
cat("// -----------------------------------------------------------------------------\n")
cat("// Test 7: Quantile Crossing Check Data\n")
cat("// Test that fitted values maintain proper ordering across quantiles\n")
cat("// -----------------------------------------------------------------------------\n")

n_cross <- 30
x_cross <- seq(0, 10, length.out = n_cross)
y_cross <- 1 + 2 * x_cross + rnorm(n_cross, sd = 1.5)

cat(sprintf("const N_CROSS: usize = %d;\n", n_cross))
cat("#[rustfmt::skip]\n")
cat(sprintf("const X_CROSS: [f64; %d] = [\n    ", n_cross))
cat(paste(sprintf("%.10f", x_cross), collapse = ", "))
cat("\n];\n")
cat("#[rustfmt::skip]\n")
cat(sprintf("const Y_CROSS: [f64; %d] = [\n    ", n_cross))
cat(paste(sprintf("%.10f", y_cross), collapse = ", "))
cat("\n];\n\n")

for (tau in c(0.10, 0.25, 0.50, 0.75, 0.90)) {
  fit <- rq(y_cross ~ x_cross, tau = tau)
  tau_str <- sprintf("%02d", as.integer(tau * 100))
  cat(sprintf("const EXPECTED_CROSS_INTERCEPT_TAU%s: f64 = %.10f;\n", tau_str, coef(fit)[1]))
  cat(sprintf("const EXPECTED_CROSS_SLOPE_TAU%s: f64 = %.10f;\n", tau_str, coef(fit)[2]))
}
cat("\n")

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
cat("// =============================================================================\n")
cat("// Summary of Extended Test Cases\n")
cat("// =============================================================================\n")
cat(sprintf("// Test 1: Engel food expenditure dataset, n=%d, 7 quantiles\n", n_engel))
cat(sprintf("// Test 2: Stackloss multivariate dataset, n=%d, p=%d, 3 quantiles\n", n_stack, p_stack))
cat(sprintf("// Test 3: Extreme quantiles (0.01, 0.05, 0.95, 0.99), n=%d\n", n_ext))
cat(sprintf("// Test 4: Outlier robustness comparison, n=%d with 3 outliers\n", n_out))
cat(sprintf("// Test 5: Heavy-tailed t-distribution errors, n=%d\n", n_heavy))
cat(sprintf("// Test 6: Small sample edge case, n=%d\n", n_small))
cat(sprintf("// Test 7: Quantile crossing check, n=%d\n", n_cross))
cat("//\n")
cat("// Tolerance recommendation: 0.1-1.0 for coefficients depending on algorithm\n")
cat("// R uses Barrodale-Roberts simplex; Rust uses IRLS approximation\n")
