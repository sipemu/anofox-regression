#!/usr/bin/env Rscript
# Generate validation data for GammaRegressor implementation.
#
# Oracle: base R glm(formula, family = Gamma(link = "log")).
# All random draws use set.seed(42) so the script is bit-stable across runs
# on the same R version.

set.seed(42)

cat("// =============================================================================\n")
cat("// Gamma Regression Validation Data Generated from R glm(family=Gamma(link='log'))\n")
cat("// Generated on:", format(Sys.time()), "\n")
cat("// R version:", R.version.string, "\n")
cat("// =============================================================================\n\n")

# -----------------------------------------------------------------------------
# Test Case 1: Univariate Gamma GLM with log link
# -----------------------------------------------------------------------------
cat("// -----------------------------------------------------------------------------\n")
cat("// Test 1: Univariate Gamma GLM with log link\n")
cat("// R: glm(y ~ x, family = Gamma(link = 'log'))\n")
cat("// -----------------------------------------------------------------------------\n")

n <- 60
x1 <- seq(0.5, 3.0, length.out = n)
mu1 <- exp(0.5 + 0.4 * x1)
y1 <- rgamma(n, shape = 2, rate = 2 / mu1)

fit1 <- glm(y1 ~ x1, family = Gamma(link = "log"))
s1 <- summary(fit1)

cat(sprintf("const N_GAMMA_UNI: usize = %d;\n", n))
cat(sprintf("const X_GAMMA_UNI: [f64; %d] = [%s];\n", n,
            paste(sprintf("%.10f", x1), collapse = ", ")))
cat(sprintf("const Y_GAMMA_UNI: [f64; %d] = [%s];\n", n,
            paste(sprintf("%.10f", y1), collapse = ", ")))
cat(sprintf("const EXPECTED_INTERCEPT_GAMMA_UNI: f64 = %.12f;\n", coef(fit1)[1]))
cat(sprintf("const EXPECTED_COEF_GAMMA_UNI: f64 = %.12f;\n", coef(fit1)[2]))
cat(sprintf("const EXPECTED_DEVIANCE_GAMMA_UNI: f64 = %.10f;\n", deviance(fit1)))
cat(sprintf("const EXPECTED_NULL_DEVIANCE_GAMMA_UNI: f64 = %.10f;\n", fit1$null.deviance))
cat(sprintf("const EXPECTED_DISPERSION_GAMMA_UNI: f64 = %.10f;\n", s1$dispersion))
cat("\n")

# -----------------------------------------------------------------------------
# Test Case 2: Multivariate Gamma GLM (two predictors)
# -----------------------------------------------------------------------------
cat("// -----------------------------------------------------------------------------\n")
cat("// Test 2: Multivariate Gamma GLM with log link\n")
cat("// R: glm(y ~ x1 + x2, family = Gamma(link = 'log'))\n")
cat("// -----------------------------------------------------------------------------\n")

set.seed(7)
n <- 80
x2a <- runif(n, 0.1, 2.0)
x2b <- runif(n, 0.5, 4.0)
mu2 <- exp(0.2 + 0.6 * x2a - 0.3 * x2b)
y2 <- rgamma(n, shape = 3, rate = 3 / mu2)

fit2 <- glm(y2 ~ x2a + x2b, family = Gamma(link = "log"))
s2 <- summary(fit2)

cat(sprintf("const N_GAMMA_MULTI: usize = %d;\n", n))
cat(sprintf("const X1_GAMMA_MULTI: [f64; %d] = [%s];\n", n,
            paste(sprintf("%.10f", x2a), collapse = ", ")))
cat(sprintf("const X2_GAMMA_MULTI: [f64; %d] = [%s];\n", n,
            paste(sprintf("%.10f", x2b), collapse = ", ")))
cat(sprintf("const Y_GAMMA_MULTI: [f64; %d] = [%s];\n", n,
            paste(sprintf("%.10f", y2), collapse = ", ")))
cat(sprintf("const EXPECTED_INTERCEPT_GAMMA_MULTI: f64 = %.12f;\n", coef(fit2)[1]))
cat(sprintf("const EXPECTED_COEF1_GAMMA_MULTI: f64 = %.12f;\n", coef(fit2)[2]))
cat(sprintf("const EXPECTED_COEF2_GAMMA_MULTI: f64 = %.12f;\n", coef(fit2)[3]))
cat(sprintf("const EXPECTED_DEVIANCE_GAMMA_MULTI: f64 = %.10f;\n", deviance(fit2)))
cat(sprintf("const EXPECTED_DISPERSION_GAMMA_MULTI: f64 = %.10f;\n", s2$dispersion))
cat("\n")
