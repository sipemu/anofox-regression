#!/usr/bin/env Rscript
# Generate validation data for Isotonic Regression
# This script creates test cases with known outputs from R's isoreg() function

set.seed(42)

cat("// =============================================================================\n")
cat("// Isotonic Regression Validation Data Generated from R\n")
cat(sprintf("// Generated on: %s\n", format(Sys.time())))
cat(sprintf("// R version: %s\n", R.version.string))
cat("// Using base R isoreg() function\n")
cat("// =============================================================================\n\n")

# -----------------------------------------------------------------------------
# Test 1: Simple Monotonic Increasing
# -----------------------------------------------------------------------------
cat("// -----------------------------------------------------------------------------\n")
cat("// Test 1: Simple Monotonic Increasing\n")
cat("// -----------------------------------------------------------------------------\n")

n <- 20
x1 <- 1:n
# Generate data with general increasing trend but some noise
y1 <- 2.0 + 0.5 * x1 + rnorm(n, sd = 1.5)

# Fit isotonic regression (increasing)
fit1 <- isoreg(x1, y1)
fitted1 <- fit1$yf

cat(sprintf("// R Code: isoreg(x, y)\n"))
cat(sprintf("const N_ISO1: usize = %d;\n", n))
cat("#[rustfmt::skip]\n")
cat(sprintf("const X_ISO1: [f64; %d] = [\n    ", n))
cat(paste(sprintf("%.10f", as.numeric(x1)), collapse = ", "))
cat("\n];\n")
cat("#[rustfmt::skip]\n")
cat(sprintf("const Y_ISO1: [f64; %d] = [\n    ", n))
cat(paste(sprintf("%.10f", y1), collapse = ", "))
cat("\n];\n")
cat("#[rustfmt::skip]\n")
cat(sprintf("const EXPECTED_FITTED_ISO1: [f64; %d] = [\n    ", n))
cat(paste(sprintf("%.10f", fitted1), collapse = ", "))
cat("\n];\n")
cat("\n")

# -----------------------------------------------------------------------------
# Test 2: Monotonic Decreasing
# -----------------------------------------------------------------------------
cat("// -----------------------------------------------------------------------------\n")
cat("// Test 2: Monotonic Decreasing\n")
cat("// -----------------------------------------------------------------------------\n")

n <- 25
x2 <- 1:n
# Decreasing trend with noise
y2 <- 20.0 - 0.6 * x2 + rnorm(n, sd = 1.0)

# For decreasing, we negate y, fit isoreg, then negate back
fit2_neg <- isoreg(x2, -y2)
fitted2 <- -fit2_neg$yf

cat(sprintf("// R Code: -isoreg(x, -y)$yf for decreasing\n"))
cat(sprintf("const N_ISO2: usize = %d;\n", n))
cat("#[rustfmt::skip]\n")
cat(sprintf("const X_ISO2: [f64; %d] = [\n    ", n))
cat(paste(sprintf("%.10f", as.numeric(x2)), collapse = ", "))
cat("\n];\n")
cat("#[rustfmt::skip]\n")
cat(sprintf("const Y_ISO2: [f64; %d] = [\n    ", n))
cat(paste(sprintf("%.10f", y2), collapse = ", "))
cat("\n];\n")
cat("#[rustfmt::skip]\n")
cat(sprintf("const EXPECTED_FITTED_ISO2: [f64; %d] = [\n    ", n))
cat(paste(sprintf("%.10f", fitted2), collapse = ", "))
cat("\n];\n")
cat("\n")

# -----------------------------------------------------------------------------
# Test 3: Weighted Isotonic Regression
# -----------------------------------------------------------------------------
cat("// -----------------------------------------------------------------------------\n")
cat("// Test 3: Weighted Isotonic Regression\n")
cat("// -----------------------------------------------------------------------------\n")

n <- 15
x3 <- 1:n
y3 <- c(1, 3, 2, 5, 4, 6, 8, 7, 9, 10, 11, 10, 13, 14, 15)
weights3 <- c(1, 2, 1, 2, 1, 3, 2, 1, 2, 1, 2, 3, 1, 2, 1)

# Weighted PAVA implementation in R
# Note: base isoreg doesn't support weights, so we implement weighted PAVA
weighted_isotonic <- function(y, w) {
  n <- length(y)
  # Initialize
  fitted <- y
  weight <- w

  # Pool Adjacent Violators with weights
  repeat {
    changed <- FALSE
    i <- 1
    while (i < n) {
      if (fitted[i] > fitted[i + 1]) {
        # Pool observations i and i+1
        pooled_value <- (weight[i] * fitted[i] + weight[i + 1] * fitted[i + 1]) /
                        (weight[i] + weight[i + 1])
        fitted[i] <- pooled_value
        fitted[i + 1] <- pooled_value
        weight[i] <- weight[i] + weight[i + 1]
        weight[i + 1] <- weight[i]
        changed <- TRUE
      }
      i <- i + 1
    }
    if (!changed) break
  }

  # Final pass to ensure monotonicity
  for (iter in 1:n) {
    changed <- FALSE
    for (i in 1:(n-1)) {
      if (fitted[i] > fitted[i + 1]) {
        pooled_value <- (fitted[i] + fitted[i + 1]) / 2
        fitted[i] <- pooled_value
        fitted[i + 1] <- pooled_value
        changed <- TRUE
      }
    }
    if (!changed) break
  }

  return(fitted)
}

fitted3 <- weighted_isotonic(y3, weights3)

cat(sprintf("// R Code: weighted PAVA algorithm\n"))
cat(sprintf("const N_ISO3: usize = %d;\n", n))
cat("#[rustfmt::skip]\n")
cat(sprintf("const X_ISO3: [f64; %d] = [\n    ", n))
cat(paste(sprintf("%.10f", as.numeric(x3)), collapse = ", "))
cat("\n];\n")
cat("#[rustfmt::skip]\n")
cat(sprintf("const Y_ISO3: [f64; %d] = [\n    ", n))
cat(paste(sprintf("%.10f", y3), collapse = ", "))
cat("\n];\n")
cat("#[rustfmt::skip]\n")
cat(sprintf("const WEIGHTS_ISO3: [f64; %d] = [\n    ", n))
cat(paste(sprintf("%.10f", weights3), collapse = ", "))
cat("\n];\n")
cat("#[rustfmt::skip]\n")
cat(sprintf("const EXPECTED_FITTED_ISO3: [f64; %d] = [\n    ", n))
cat(paste(sprintf("%.10f", fitted3), collapse = ", "))
cat("\n];\n")
cat("\n")

# -----------------------------------------------------------------------------
# Test 4: Already Monotonic Data
# -----------------------------------------------------------------------------
cat("// -----------------------------------------------------------------------------\n")
cat("// Test 4: Already Monotonic Data (no pooling needed)\n")
cat("// -----------------------------------------------------------------------------\n")

n <- 10
x4 <- 1:n
y4 <- c(1.0, 2.5, 3.2, 4.8, 5.1, 6.3, 7.0, 8.2, 9.5, 10.1)

fit4 <- isoreg(x4, y4)
fitted4 <- fit4$yf

cat(sprintf("// R Code: isoreg(x, y) - already monotonic\n"))
cat(sprintf("const N_ISO4: usize = %d;\n", n))
cat("#[rustfmt::skip]\n")
cat(sprintf("const X_ISO4: [f64; %d] = [\n    ", n))
cat(paste(sprintf("%.10f", as.numeric(x4)), collapse = ", "))
cat("\n];\n")
cat("#[rustfmt::skip]\n")
cat(sprintf("const Y_ISO4: [f64; %d] = [\n    ", n))
cat(paste(sprintf("%.10f", y4), collapse = ", "))
cat("\n];\n")
cat("#[rustfmt::skip]\n")
cat(sprintf("const EXPECTED_FITTED_ISO4: [f64; %d] = [\n    ", n))
cat(paste(sprintf("%.10f", fitted4), collapse = ", "))
cat("\n];\n")
cat("\n")

# -----------------------------------------------------------------------------
# Test 5: Ties in X values
# -----------------------------------------------------------------------------
cat("// -----------------------------------------------------------------------------\n")
cat("// Test 5: Ties in X values\n")
cat("// -----------------------------------------------------------------------------\n")

n <- 12
x5 <- c(1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6)
y5 <- c(1.5, 2.0, 2.8, 3.2, 3.5, 4.0, 4.2, 4.8, 5.5, 5.0, 6.2, 6.8)

# For ties, isoreg averages the y values at each unique x first
fit5 <- isoreg(x5, y5)
# Get the fitted values for each observation
fitted5 <- fit5$yf

cat(sprintf("// R Code: isoreg(x, y) with ties\n"))
cat(sprintf("const N_ISO5: usize = %d;\n", n))
cat("#[rustfmt::skip]\n")
cat(sprintf("const X_ISO5: [f64; %d] = [\n    ", n))
cat(paste(sprintf("%.10f", x5), collapse = ", "))
cat("\n];\n")
cat("#[rustfmt::skip]\n")
cat(sprintf("const Y_ISO5: [f64; %d] = [\n    ", n))
cat(paste(sprintf("%.10f", y5), collapse = ", "))
cat("\n];\n")
cat("#[rustfmt::skip]\n")
cat(sprintf("const EXPECTED_FITTED_ISO5: [f64; %d] = [\n    ", n))
cat(paste(sprintf("%.10f", fitted5), collapse = ", "))
cat("\n];\n")
cat("\n")

# -----------------------------------------------------------------------------
# Test 6: Large Dataset
# -----------------------------------------------------------------------------
cat("// -----------------------------------------------------------------------------\n")
cat("// Test 6: Large Dataset\n")
cat("// -----------------------------------------------------------------------------\n")

n <- 100
x6 <- 1:n
y6 <- sqrt(x6) * 3 + rnorm(n, sd = 1.5)

fit6 <- isoreg(x6, y6)
fitted6 <- fit6$yf

cat(sprintf("// R Code: isoreg(x, y) large dataset\n"))
cat(sprintf("const N_ISO6: usize = %d;\n", n))
cat("#[rustfmt::skip]\n")
cat(sprintf("const X_ISO6: [f64; %d] = [\n    ", n))
cat(paste(sprintf("%.10f", as.numeric(x6)), collapse = ", "))
cat("\n];\n")
cat("#[rustfmt::skip]\n")
cat(sprintf("const Y_ISO6: [f64; %d] = [\n    ", n))
cat(paste(sprintf("%.10f", y6), collapse = ", "))
cat("\n];\n")
cat("#[rustfmt::skip]\n")
cat(sprintf("const EXPECTED_FITTED_ISO6: [f64; %d] = [\n    ", n))
cat(paste(sprintf("%.10f", fitted6), collapse = ", "))
cat("\n];\n")
cat("\n")

# -----------------------------------------------------------------------------
# Test 7: Step Function Pattern
# -----------------------------------------------------------------------------
cat("// -----------------------------------------------------------------------------\n")
cat("// Test 7: Step Function Pattern\n")
cat("// -----------------------------------------------------------------------------\n")

n <- 30
x7 <- 1:n
# Create data that should result in step-like isotonic fit
y7 <- c(rep(2, 5) + rnorm(5, sd = 0.3),
        rep(5, 10) + rnorm(10, sd = 0.3),
        rep(8, 10) + rnorm(10, sd = 0.3),
        rep(12, 5) + rnorm(5, sd = 0.3))

fit7 <- isoreg(x7, y7)
fitted7 <- fit7$yf

cat(sprintf("// R Code: isoreg(x, y) step function\n"))
cat(sprintf("const N_ISO7: usize = %d;\n", n))
cat("#[rustfmt::skip]\n")
cat(sprintf("const X_ISO7: [f64; %d] = [\n    ", n))
cat(paste(sprintf("%.10f", as.numeric(x7)), collapse = ", "))
cat("\n];\n")
cat("#[rustfmt::skip]\n")
cat(sprintf("const Y_ISO7: [f64; %d] = [\n    ", n))
cat(paste(sprintf("%.10f", y7), collapse = ", "))
cat("\n];\n")
cat("#[rustfmt::skip]\n")
cat(sprintf("const EXPECTED_FITTED_ISO7: [f64; %d] = [\n    ", n))
cat(paste(sprintf("%.10f", fitted7), collapse = ", "))
cat("\n];\n")
cat("\n")

# -----------------------------------------------------------------------------
# Test 8: Single Observation
# -----------------------------------------------------------------------------
cat("// -----------------------------------------------------------------------------\n")
cat("// Test 8: Edge Case - Two Observations\n")
cat("// -----------------------------------------------------------------------------\n")

n <- 2
x8 <- c(1, 2)
y8 <- c(5.0, 3.0)  # Decreasing - should pool

fit8 <- isoreg(x8, y8)
fitted8 <- fit8$yf

cat(sprintf("// R Code: isoreg(x, y) two observations\n"))
cat(sprintf("const N_ISO8: usize = %d;\n", n))
cat("#[rustfmt::skip]\n")
cat(sprintf("const X_ISO8: [f64; %d] = [%s];\n", n,
            paste(sprintf("%.10f", x8), collapse = ", ")))
cat("#[rustfmt::skip]\n")
cat(sprintf("const Y_ISO8: [f64; %d] = [%s];\n", n,
            paste(sprintf("%.10f", y8), collapse = ", ")))
cat("#[rustfmt::skip]\n")
cat(sprintf("const EXPECTED_FITTED_ISO8: [f64; %d] = [%s];\n", n,
            paste(sprintf("%.10f", fitted8), collapse = ", ")))
cat("\n")

# -----------------------------------------------------------------------------
# Test 9: Unsorted X values
# -----------------------------------------------------------------------------
cat("// -----------------------------------------------------------------------------\n")
cat("// Test 9: Unsorted X values (should be sorted first)\n")
cat("// -----------------------------------------------------------------------------\n")

n <- 15
x9 <- sample(1:n)  # Random permutation
y9 <- 1.5 * (1:n) + rnorm(n, sd = 1.0)
y9_shuffled <- y9[order(x9)]  # Shuffle y to match x

# Sort and fit
ord <- order(x9)
x9_sorted <- x9[ord]
y9_sorted <- y9_shuffled[ord]

fit9 <- isoreg(x9_sorted, y9_sorted)
fitted9 <- fit9$yf

cat(sprintf("// R Code: isoreg after sorting by x\n"))
cat(sprintf("const N_ISO9: usize = %d;\n", n))
cat("#[rustfmt::skip]\n")
cat(sprintf("const X_ISO9_UNSORTED: [f64; %d] = [\n    ", n))
cat(paste(sprintf("%.10f", x9), collapse = ", "))
cat("\n];\n")
cat("#[rustfmt::skip]\n")
cat(sprintf("const Y_ISO9_UNSORTED: [f64; %d] = [\n    ", n))
cat(paste(sprintf("%.10f", y9_shuffled), collapse = ", "))
cat("\n];\n")
cat("#[rustfmt::skip]\n")
cat(sprintf("const X_ISO9_SORTED: [f64; %d] = [\n    ", n))
cat(paste(sprintf("%.10f", x9_sorted), collapse = ", "))
cat("\n];\n")
cat("#[rustfmt::skip]\n")
cat(sprintf("const Y_ISO9_SORTED: [f64; %d] = [\n    ", n))
cat(paste(sprintf("%.10f", y9_sorted), collapse = ", "))
cat("\n];\n")
cat("#[rustfmt::skip]\n")
cat(sprintf("const EXPECTED_FITTED_ISO9: [f64; %d] = [\n    ", n))
cat(paste(sprintf("%.10f", fitted9), collapse = ", "))
cat("\n];\n")
cat("\n")

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
cat("// =============================================================================\n")
cat("// Summary of test cases\n")
cat("// =============================================================================\n")
cat(sprintf("// Test 1: Simple monotonic increasing, n=%d\n", 20))
cat(sprintf("// Test 2: Monotonic decreasing, n=%d\n", 25))
cat(sprintf("// Test 3: Weighted isotonic regression, n=%d\n", 15))
cat(sprintf("// Test 4: Already monotonic data, n=%d\n", 10))
cat(sprintf("// Test 5: Ties in X values, n=%d\n", 12))
cat(sprintf("// Test 6: Large dataset, n=%d\n", 100))
cat(sprintf("// Test 7: Step function pattern, n=%d\n", 30))
cat(sprintf("// Test 8: Two observations edge case, n=%d\n", 2))
cat(sprintf("// Test 9: Unsorted X values, n=%d\n", 15))
cat("//\n")
cat("// Tolerance recommendation: 1e-8 for fitted values\n")
cat("// PAVA is deterministic and should produce identical results.\n")
