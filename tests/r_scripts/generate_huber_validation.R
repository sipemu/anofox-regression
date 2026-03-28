#!/usr/bin/env Rscript
# Generate validation data for HuberRegressor implementation
# Uses MASS::rlm() with method="M" and psi=psi.huber as the oracle

library(MASS)

set.seed(42)

cat("// =============================================================================\n")
cat("// Huber Regression Validation Data Generated from R MASS::rlm()\n")
cat("// Generated on:", format(Sys.time()), "\n")
cat("// R version:", R.version.string, "\n")
cat("// =============================================================================\n\n")

# -----------------------------------------------------------------------------
# Test Case 1: Clean data (no outliers) - should match OLS closely
# -----------------------------------------------------------------------------
cat("// -----------------------------------------------------------------------------\n")
cat("// Test 1: Clean data (no outliers)\n")
cat("// R: rlm(y ~ x, method='M', psi=psi.huber, k=1.345)\n")
cat("// -----------------------------------------------------------------------------\n")

n <- 50
x1 <- seq(1, 10, length.out = n)
y1 <- 2.0 + 3.0 * x1 + rnorm(n, sd = 0.5)

fit1 <- rlm(y1 ~ x1, method = "M", psi = psi.huber, k = 1.345)
s1 <- summary(fit1)

cat(sprintf("const N_CLEAN: usize = %d;\n", n))
cat(sprintf("const X_CLEAN: [f64; %d] = [%s];\n", n, paste(sprintf("%.10f", x1), collapse = ", ")))
cat(sprintf("const Y_CLEAN: [f64; %d] = [%s];\n", n, paste(sprintf("%.10f", y1), collapse = ", ")))
cat(sprintf("const EXPECTED_INTERCEPT_CLEAN: f64 = %.10f;\n", coef(fit1)[1]))
cat(sprintf("const EXPECTED_COEF_CLEAN: f64 = %.10f;\n", coef(fit1)[2]))
cat(sprintf("const EXPECTED_SCALE_CLEAN: f64 = %.10f;\n", fit1$s))
cat("\n")

# -----------------------------------------------------------------------------
# Test Case 2: Data with outliers - Huber should be robust
# -----------------------------------------------------------------------------
cat("// -----------------------------------------------------------------------------\n")
cat("// Test 2: Data with outliers\n")
cat("// R: rlm(y ~ x, method='M', psi=psi.huber, k=1.345)\n")
cat("// -----------------------------------------------------------------------------\n")

n <- 50
x2 <- seq(1, 10, length.out = n)
y2 <- 2.0 + 3.0 * x2 + rnorm(n, sd = 0.5)
# Inject outliers
y2[5] <- 100.0
y2[15] <- -50.0
y2[25] <- 80.0
y2[35] <- -40.0
y2[45] <- 90.0

fit2 <- rlm(y2 ~ x2, method = "M", psi = psi.huber, k = 1.345)
s2 <- summary(fit2)

# Also fit OLS for comparison
fit2_ols <- lm(y2 ~ x2)

cat(sprintf("const N_OUTLIER: usize = %d;\n", n))
cat(sprintf("const X_OUTLIER: [f64; %d] = [%s];\n", n, paste(sprintf("%.10f", x2), collapse = ", ")))
cat(sprintf("const Y_OUTLIER: [f64; %d] = [%s];\n", n, paste(sprintf("%.10f", y2), collapse = ", ")))
cat(sprintf("const EXPECTED_INTERCEPT_OUTLIER: f64 = %.10f;\n", coef(fit2)[1]))
cat(sprintf("const EXPECTED_COEF_OUTLIER: f64 = %.10f;\n", coef(fit2)[2]))
cat(sprintf("const EXPECTED_SCALE_OUTLIER: f64 = %.10f;\n", fit2$s))
cat(sprintf("const OLS_INTERCEPT_OUTLIER: f64 = %.10f;\n", coef(fit2_ols)[1]))
cat(sprintf("const OLS_COEF_OUTLIER: f64 = %.10f;\n", coef(fit2_ols)[2]))
cat("\n")

# -----------------------------------------------------------------------------
# Test Case 3: Multiple predictors
# -----------------------------------------------------------------------------
cat("// -----------------------------------------------------------------------------\n")
cat("// Test 3: Multiple predictors\n")
cat("// R: rlm(y ~ x1 + x2, method='M', psi=psi.huber, k=1.345)\n")
cat("// -----------------------------------------------------------------------------\n")

n <- 60
x3a <- rnorm(n, mean = 5, sd = 2)
x3b <- rnorm(n, mean = 3, sd = 1)
y3 <- 1.0 + 2.0 * x3a - 1.5 * x3b + rnorm(n, sd = 1.0)
# Add some outliers
y3[10] <- 50.0
y3[30] <- -30.0

fit3 <- rlm(y3 ~ x3a + x3b, method = "M", psi = psi.huber, k = 1.345)

cat(sprintf("const N_MULTI: usize = %d;\n", n))
cat(sprintf("const X_MULTI_1: [f64; %d] = [%s];\n", n, paste(sprintf("%.10f", x3a), collapse = ", ")))
cat(sprintf("const X_MULTI_2: [f64; %d] = [%s];\n", n, paste(sprintf("%.10f", x3b), collapse = ", ")))
cat(sprintf("const Y_MULTI: [f64; %d] = [%s];\n", n, paste(sprintf("%.10f", y3), collapse = ", ")))
cat(sprintf("const EXPECTED_INTERCEPT_MULTI: f64 = %.10f;\n", coef(fit3)[1]))
cat(sprintf("const EXPECTED_COEF_MULTI_1: f64 = %.10f;\n", coef(fit3)[2]))
cat(sprintf("const EXPECTED_COEF_MULTI_2: f64 = %.10f;\n", coef(fit3)[3]))
cat(sprintf("const EXPECTED_SCALE_MULTI: f64 = %.10f;\n", fit3$s))
cat("\n")

# -----------------------------------------------------------------------------
# Test Case 4: Huber weights analysis
# -----------------------------------------------------------------------------
cat("// -----------------------------------------------------------------------------\n")
cat("// Test 4: Huber weight analysis (which observations get downweighted)\n")
cat("// -----------------------------------------------------------------------------\n")

# Get weights from fit2 (outlier case)
w2 <- fit2$w
cat(sprintf("const EXPECTED_WEIGHTS_OUTLIER: [f64; %d] = [%s];\n", n,
            paste(sprintf("%.10f", w2), collapse = ", ")))
# Indices where weight < 1 (0-indexed)
downweighted <- which(w2 < 1.0) - 1  # Convert to 0-indexed
cat(sprintf("const EXPECTED_DOWNWEIGHTED_INDICES: [usize; %d] = [%s];\n",
            length(downweighted), paste(downweighted, collapse = ", ")))
cat("\n")

# -----------------------------------------------------------------------------
# Test Case 5: No intercept
# -----------------------------------------------------------------------------
cat("// -----------------------------------------------------------------------------\n")
cat("// Test 5: No intercept\n")
cat("// R: rlm(y ~ x - 1, method='M', psi=psi.huber, k=1.345)\n")
cat("// -----------------------------------------------------------------------------\n")

n <- 40
x5 <- seq(0.5, 5, length.out = n)
y5 <- 3.0 * x5 + rnorm(n, sd = 0.3)
y5[5] <- 30.0  # outlier

fit5 <- rlm(y5 ~ x5 - 1, method = "M", psi = psi.huber, k = 1.345)

cat(sprintf("const N_NOINT: usize = %d;\n", n))
cat(sprintf("const X_NOINT: [f64; %d] = [%s];\n", n, paste(sprintf("%.10f", x5), collapse = ", ")))
cat(sprintf("const Y_NOINT: [f64; %d] = [%s];\n", n, paste(sprintf("%.10f", y5), collapse = ", ")))
cat(sprintf("const EXPECTED_COEF_NOINT: f64 = %.10f;\n", coef(fit5)[1]))
cat(sprintf("const EXPECTED_SCALE_NOINT: f64 = %.10f;\n", fit5$s))
cat("\n")
