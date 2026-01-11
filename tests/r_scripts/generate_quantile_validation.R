#!/usr/bin/env Rscript
# Generate validation data for Quantile Regression
# This script creates test cases with known outputs from R's quantreg package

# Install quantreg package if not available
if (!require("quantreg", quietly = TRUE)) {
  install.packages("quantreg", repos = "https://cloud.r-project.org")
  library(quantreg)
}

set.seed(42)

cat("// =============================================================================\n")
cat("// Quantile Regression Validation Data Generated from R\n")
cat(sprintf("// Generated on: %s\n", format(Sys.time())))
cat(sprintf("// R version: %s\n", R.version.string))
cat(sprintf("// quantreg package version: %s\n", packageVersion("quantreg")))
cat("// =============================================================================\n\n")

# -----------------------------------------------------------------------------
# Test 1: Simple Quantile Regression - Median (tau = 0.5)
# -----------------------------------------------------------------------------
cat("// -----------------------------------------------------------------------------\n")
cat("// Test 1: Simple Quantile Regression - Median (tau = 0.5)\n")
cat("// -----------------------------------------------------------------------------\n")

n <- 50
x1 <- seq(1, 10, length.out = n)
# Generate heteroscedastic data - variance increases with x
y1 <- 2.0 + 1.5 * x1 + rnorm(n, sd = 0.5 * x1)

# Fit quantile regression at median
fit1 <- rq(y1 ~ x1, tau = 0.5)
coefs1 <- coef(fit1)

# Get fitted values
fitted1 <- predict(fit1)

# Compute pseudo R-squared (based on sum of absolute deviations)
rho <- function(u, tau) u * (tau - (u < 0))
r1 <- sum(rho(y1 - fitted1, 0.5))
r0 <- sum(rho(y1 - quantile(y1, 0.5), 0.5))
pseudo_r2_1 <- 1 - r1/r0

cat(sprintf("// R Code: rq(y ~ x, tau = 0.5)\n"))
cat(sprintf("const N_QR1: usize = %d;\n", n))
cat("#[rustfmt::skip]\n")
cat(sprintf("const X_QR1: [f64; %d] = [\n    ", n))
cat(paste(sprintf("%.10f", x1), collapse = ", "))
cat("\n];\n")
cat("#[rustfmt::skip]\n")
cat(sprintf("const Y_QR1: [f64; %d] = [\n    ", n))
cat(paste(sprintf("%.10f", y1), collapse = ", "))
cat("\n];\n")
cat(sprintf("const TAU_QR1: f64 = 0.5;\n"))
cat(sprintf("const EXPECTED_INTERCEPT_QR1: f64 = %.10f;\n", coefs1[1]))
cat(sprintf("const EXPECTED_COEF_QR1: f64 = %.10f;\n", coefs1[2]))
cat(sprintf("const EXPECTED_PSEUDO_R2_QR1: f64 = %.10f;\n", pseudo_r2_1))
cat("\n")

# -----------------------------------------------------------------------------
# Test 2: Multiple Quantiles (0.1, 0.25, 0.5, 0.75, 0.9)
# -----------------------------------------------------------------------------
cat("// -----------------------------------------------------------------------------\n")
cat("// Test 2: Multiple Quantiles (0.1, 0.25, 0.5, 0.75, 0.9)\n")
cat("// -----------------------------------------------------------------------------\n")

n <- 100
x2 <- runif(n, 0, 10)
# Asymmetric error distribution
y2 <- 1.0 + 2.0 * x2 + rexp(n, rate = 0.5) - 2  # Shifted exponential errors

taus <- c(0.1, 0.25, 0.5, 0.75, 0.9)

cat(sprintf("// R Code: rq(y ~ x, tau = c(0.1, 0.25, 0.5, 0.75, 0.9))\n"))
cat(sprintf("const N_QR2: usize = %d;\n", n))
cat("#[rustfmt::skip]\n")
cat(sprintf("const X_QR2: [f64; %d] = [\n    ", n))
cat(paste(sprintf("%.10f", x2), collapse = ", "))
cat("\n];\n")
cat("#[rustfmt::skip]\n")
cat(sprintf("const Y_QR2: [f64; %d] = [\n    ", n))
cat(paste(sprintf("%.10f", y2), collapse = ", "))
cat("\n];\n")
cat("#[rustfmt::skip]\n")
cat(sprintf("const TAUS_QR2: [f64; %d] = [%s];\n", length(taus),
            paste(sprintf("%.2f", taus), collapse = ", ")))

for (tau in taus) {
  fit <- rq(y2 ~ x2, tau = tau)
  coefs <- coef(fit)
  cat(sprintf("const EXPECTED_INTERCEPT_QR2_TAU%02d: f64 = %.10f;\n",
              as.integer(tau * 100), coefs[1]))
  cat(sprintf("const EXPECTED_COEF_QR2_TAU%02d: f64 = %.10f;\n",
              as.integer(tau * 100), coefs[2]))
}
cat("\n")

# -----------------------------------------------------------------------------
# Test 3: Multiple Regression with Quantiles
# -----------------------------------------------------------------------------
cat("// -----------------------------------------------------------------------------\n")
cat("// Test 3: Multiple Regression with Quantiles\n")
cat("// -----------------------------------------------------------------------------\n")

n <- 80
p <- 3
X3 <- matrix(rnorm(n * p), n, p)
# True coefficients
beta_true <- c(1.5, -0.8, 2.0)
y3 <- 0.5 + X3 %*% beta_true + rnorm(n, sd = 1.5)

# Fit at tau = 0.5 (median)
fit3_50 <- rq(y3 ~ X3, tau = 0.5)
coefs3_50 <- coef(fit3_50)

# Fit at tau = 0.75
fit3_75 <- rq(y3 ~ X3, tau = 0.75)
coefs3_75 <- coef(fit3_75)

# Fit at tau = 0.25
fit3_25 <- rq(y3 ~ X3, tau = 0.25)
coefs3_25 <- coef(fit3_25)

cat(sprintf("// R Code: rq(y ~ X, tau = c(0.25, 0.5, 0.75))\n"))
cat(sprintf("const N_QR3: usize = %d;\n", n))
cat(sprintf("const P_QR3: usize = %d;\n", p))
cat("#[rustfmt::skip]\n")
cat(sprintf("const X_QR3: [f64; %d] = [\n    ", n * p))
cat(paste(sprintf("%.10f", as.vector(X3)), collapse = ", "))
cat("\n];\n")
cat("#[rustfmt::skip]\n")
cat(sprintf("const Y_QR3: [f64; %d] = [\n    ", n))
cat(paste(sprintf("%.10f", as.vector(y3)), collapse = ", "))
cat("\n];\n")

cat("// tau = 0.25\n")
cat(sprintf("const EXPECTED_INTERCEPT_QR3_TAU25: f64 = %.10f;\n", coefs3_25[1]))
cat("#[rustfmt::skip]\n")
cat(sprintf("const EXPECTED_COEFS_QR3_TAU25: [f64; %d] = [%s];\n", p,
            paste(sprintf("%.10f", coefs3_25[2:(p+1)]), collapse = ", ")))

cat("// tau = 0.50\n")
cat(sprintf("const EXPECTED_INTERCEPT_QR3_TAU50: f64 = %.10f;\n", coefs3_50[1]))
cat("#[rustfmt::skip]\n")
cat(sprintf("const EXPECTED_COEFS_QR3_TAU50: [f64; %d] = [%s];\n", p,
            paste(sprintf("%.10f", coefs3_50[2:(p+1)]), collapse = ", ")))

cat("// tau = 0.75\n")
cat(sprintf("const EXPECTED_INTERCEPT_QR3_TAU75: f64 = %.10f;\n", coefs3_75[1]))
cat("#[rustfmt::skip]\n")
cat(sprintf("const EXPECTED_COEFS_QR3_TAU75: [f64; %d] = [%s];\n", p,
            paste(sprintf("%.10f", coefs3_75[2:(p+1)]), collapse = ", ")))
cat("\n")

# -----------------------------------------------------------------------------
# Test 4: Weighted Quantile Regression
# -----------------------------------------------------------------------------
cat("// -----------------------------------------------------------------------------\n")
cat("// Test 4: Weighted Quantile Regression\n")
cat("// -----------------------------------------------------------------------------\n")

n <- 40
x4 <- seq(1, 8, length.out = n)
y4 <- 3.0 + 0.8 * x4 + rnorm(n, sd = 0.5)
# Weights - higher weight for middle observations
weights4 <- dnorm(x4, mean = 4.5, sd = 2) * 10
weights4 <- weights4 / sum(weights4) * n  # Normalize

fit4 <- rq(y4 ~ x4, tau = 0.5, weights = weights4)
coefs4 <- coef(fit4)

cat(sprintf("// R Code: rq(y ~ x, tau = 0.5, weights = w)\n"))
cat(sprintf("const N_QR4: usize = %d;\n", n))
cat("#[rustfmt::skip]\n")
cat(sprintf("const X_QR4: [f64; %d] = [\n    ", n))
cat(paste(sprintf("%.10f", x4), collapse = ", "))
cat("\n];\n")
cat("#[rustfmt::skip]\n")
cat(sprintf("const Y_QR4: [f64; %d] = [\n    ", n))
cat(paste(sprintf("%.10f", y4), collapse = ", "))
cat("\n];\n")
cat("#[rustfmt::skip]\n")
cat(sprintf("const WEIGHTS_QR4: [f64; %d] = [\n    ", n))
cat(paste(sprintf("%.10f", weights4), collapse = ", "))
cat("\n];\n")
cat(sprintf("const EXPECTED_INTERCEPT_QR4: f64 = %.10f;\n", coefs4[1]))
cat(sprintf("const EXPECTED_COEF_QR4: f64 = %.10f;\n", coefs4[2]))
cat("\n")

# -----------------------------------------------------------------------------
# Test 5: Extreme Quantiles (0.05, 0.95)
# -----------------------------------------------------------------------------
cat("// -----------------------------------------------------------------------------\n")
cat("// Test 5: Extreme Quantiles (0.05, 0.95)\n")
cat("// -----------------------------------------------------------------------------\n")

n <- 200  # Larger sample for extreme quantiles
x5 <- runif(n, 0, 10)
y5 <- 2.5 + 1.2 * x5 + rnorm(n, sd = 2.0)

fit5_05 <- rq(y5 ~ x5, tau = 0.05)
fit5_95 <- rq(y5 ~ x5, tau = 0.95)
coefs5_05 <- coef(fit5_05)
coefs5_95 <- coef(fit5_95)

cat(sprintf("// R Code: rq(y ~ x, tau = c(0.05, 0.95))\n"))
cat(sprintf("const N_QR5: usize = %d;\n", n))
cat("#[rustfmt::skip]\n")
cat(sprintf("const X_QR5: [f64; %d] = [\n    ", n))
cat(paste(sprintf("%.10f", x5), collapse = ", "))
cat("\n];\n")
cat("#[rustfmt::skip]\n")
cat(sprintf("const Y_QR5: [f64; %d] = [\n    ", n))
cat(paste(sprintf("%.10f", y5), collapse = ", "))
cat("\n];\n")
cat("// tau = 0.05\n")
cat(sprintf("const EXPECTED_INTERCEPT_QR5_TAU05: f64 = %.10f;\n", coefs5_05[1]))
cat(sprintf("const EXPECTED_COEF_QR5_TAU05: f64 = %.10f;\n", coefs5_05[2]))
cat("// tau = 0.95\n")
cat(sprintf("const EXPECTED_INTERCEPT_QR5_TAU95: f64 = %.10f;\n", coefs5_95[1]))
cat(sprintf("const EXPECTED_COEF_QR5_TAU95: f64 = %.10f;\n", coefs5_95[2]))
cat("\n")

# -----------------------------------------------------------------------------
# Test 6: No Intercept
# -----------------------------------------------------------------------------
cat("// -----------------------------------------------------------------------------\n")
cat("// Test 6: No Intercept\n")
cat("// -----------------------------------------------------------------------------\n")

n <- 60
x6 <- runif(n, 1, 10)
y6 <- 2.5 * x6 + rnorm(n, sd = 1.0)

fit6 <- rq(y6 ~ x6 - 1, tau = 0.5)  # No intercept
coef6 <- coef(fit6)

cat(sprintf("// R Code: rq(y ~ x - 1, tau = 0.5)\n"))
cat(sprintf("const N_QR6: usize = %d;\n", n))
cat("#[rustfmt::skip]\n")
cat(sprintf("const X_QR6: [f64; %d] = [\n    ", n))
cat(paste(sprintf("%.10f", x6), collapse = ", "))
cat("\n];\n")
cat("#[rustfmt::skip]\n")
cat(sprintf("const Y_QR6: [f64; %d] = [\n    ", n))
cat(paste(sprintf("%.10f", y6), collapse = ", "))
cat("\n];\n")
cat(sprintf("const EXPECTED_COEF_QR6: f64 = %.10f;\n", coef6[1]))
cat("\n")

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
cat("// =============================================================================\n")
cat("// Summary of test cases\n")
cat("// =============================================================================\n")
cat(sprintf("// Test 1: Simple median regression, n=%d\n", 50))
cat(sprintf("// Test 2: Multiple quantiles (0.1, 0.25, 0.5, 0.75, 0.9), n=%d\n", 100))
cat(sprintf("// Test 3: Multiple regression with 3 predictors, n=%d\n", 80))
cat(sprintf("// Test 4: Weighted quantile regression, n=%d\n", 40))
cat(sprintf("// Test 5: Extreme quantiles (0.05, 0.95), n=%d\n", 200))
cat(sprintf("// Test 6: No intercept, n=%d\n", 60))
cat("//\n")
cat("// Tolerance recommendation: 0.05-0.1 for coefficients\n")
cat("// Quantile regression uses linear programming which may produce\n")
cat("// slightly different solutions depending on algorithm choice.\n")
