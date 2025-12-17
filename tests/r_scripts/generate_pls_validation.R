#!/usr/bin/env Rscript
# Generate validation data for PLS (Partial Least Squares) regression
# This script creates test cases with known outputs from R's pls package

# Install pls package if not available
if (!require("pls", quietly = TRUE)) {
  install.packages("pls", repos = "https://cloud.r-project.org")
  library(pls)
}

set.seed(42)

cat("// =============================================================================\n")
cat("// PLS Regression Validation Data Generated from R\n")
cat(sprintf("// Generated on: %s\n", format(Sys.time())))
cat(sprintf("// R version: %s\n", R.version.string))
cat(sprintf("// pls package version: %s\n", packageVersion("pls")))
cat("// =============================================================================\n\n")

# -----------------------------------------------------------------------------
# Test 1: Simple PLS with 2 components
# -----------------------------------------------------------------------------
cat("// -----------------------------------------------------------------------------\n")
cat("// Test 1: Simple PLS with 2 components (SIMPLS)\n")
cat("// -----------------------------------------------------------------------------\n")

n <- 30
p <- 5
ncomp <- 2

# Generate correlated predictors
X1 <- matrix(rnorm(n * p), n, p)
# Add correlation structure
X1[,2] <- X1[,1] * 0.7 + rnorm(n, sd = 0.3)
X1[,4] <- X1[,3] * 0.8 + rnorm(n, sd = 0.2)

# True relationship
beta_true <- c(1.5, 0.8, -1.2, 0.4, 0.6)
y1 <- X1 %*% beta_true + rnorm(n, sd = 0.5)

# Create data frame for pls
df1 <- data.frame(y = y1, X = I(X1))

# Fit PLS with SIMPLS algorithm
fit1 <- plsr(y ~ X, data = df1, ncomp = ncomp, method = "simpls",
             scale = FALSE, center = TRUE)

# Get coefficients for ncomp components
coefs1 <- as.vector(coef(fit1, ncomp = ncomp))
intercept1 <- mean(y1) - sum(colMeans(X1) * coefs1)

# Get predictions on training data
fitted1 <- as.vector(predict(fit1, ncomp = ncomp))

# Compute R² manually (same way as our implementation)
y_mean1 <- mean(y1)
tss1 <- sum((y1 - y_mean1)^2)
rss1 <- sum((y1 - fitted1)^2)
r2_1 <- 1 - rss1 / tss1

cat(sprintf("// R Code: plsr(y ~ X, ncomp = %d, method = \"simpls\", scale = FALSE)\n", ncomp))
cat(sprintf("const N_PLS1: usize = %d;\n", n))
cat(sprintf("const P_PLS1: usize = %d;\n", p))
cat(sprintf("const NCOMP_PLS1: usize = %d;\n", ncomp))
cat("#[rustfmt::skip]\n")
cat(sprintf("const X_PLS1: [f64; %d] = [\n    ", n * p))
cat(paste(sprintf("%.10f", as.vector(X1)), collapse = ", "))
cat("\n];\n")
cat("#[rustfmt::skip]\n")
cat(sprintf("const Y_PLS1: [f64; %d] = [\n    ", n))
cat(paste(sprintf("%.10f", as.vector(y1)), collapse = ", "))
cat("\n];\n")
cat(sprintf("const EXPECTED_INTERCEPT_PLS1: f64 = %.10f;\n", intercept1))
cat("#[rustfmt::skip]\n")
cat(sprintf("const EXPECTED_COEFS_PLS1: [f64; %d] = [\n    ", p))
cat(paste(sprintf("%.10f", coefs1), collapse = ", "))
cat("\n];\n")
cat(sprintf("const EXPECTED_R2_PLS1: f64 = %.10f;\n", r2_1))
cat("\n")

# -----------------------------------------------------------------------------
# Test 2: PLS with more components
# -----------------------------------------------------------------------------
cat("// -----------------------------------------------------------------------------\n")
cat("// Test 2: PLS with 4 components (SIMPLS)\n")
cat("// -----------------------------------------------------------------------------\n")

n <- 50
p <- 8
ncomp <- 4

# Generate predictors with more complex structure
X2 <- matrix(rnorm(n * p), n, p)
# Add some correlation
for (i in 2:p) {
  X2[,i] <- X2[,i] + 0.3 * X2[,1]
}

# True relationship (sparse)
beta_true2 <- c(2.0, -1.5, 0.0, 1.0, 0.0, -0.5, 0.3, 0.0)
y2 <- X2 %*% beta_true2 + rnorm(n, sd = 1.0)

# Create data frame for pls
df2 <- data.frame(y = y2, X = I(X2))

# Fit PLS with SIMPLS algorithm
fit2 <- plsr(y ~ X, data = df2, ncomp = ncomp, method = "simpls",
             scale = FALSE, center = TRUE)

# Get coefficients for ncomp components
coefs2 <- as.vector(coef(fit2, ncomp = ncomp))
intercept2 <- mean(y2) - sum(colMeans(X2) * coefs2)

# Get predictions on training data
fitted2 <- as.vector(predict(fit2, ncomp = ncomp))

# Compute R²
y_mean2 <- mean(y2)
tss2 <- sum((y2 - y_mean2)^2)
rss2 <- sum((y2 - fitted2)^2)
r2_2 <- 1 - rss2 / tss2

cat(sprintf("// R Code: plsr(y ~ X, ncomp = %d, method = \"simpls\", scale = FALSE)\n", ncomp))
cat(sprintf("const N_PLS2: usize = %d;\n", n))
cat(sprintf("const P_PLS2: usize = %d;\n", p))
cat(sprintf("const NCOMP_PLS2: usize = %d;\n", ncomp))
cat("#[rustfmt::skip]\n")
cat(sprintf("const X_PLS2: [f64; %d] = [\n    ", n * p))
cat(paste(sprintf("%.10f", as.vector(X2)), collapse = ", "))
cat("\n];\n")
cat("#[rustfmt::skip]\n")
cat(sprintf("const Y_PLS2: [f64; %d] = [\n    ", n))
cat(paste(sprintf("%.10f", as.vector(y2)), collapse = ", "))
cat("\n];\n")
cat(sprintf("const EXPECTED_INTERCEPT_PLS2: f64 = %.10f;\n", intercept2))
cat("#[rustfmt::skip]\n")
cat(sprintf("const EXPECTED_COEFS_PLS2: [f64; %d] = [\n    ", p))
cat(paste(sprintf("%.10f", coefs2), collapse = ", "))
cat("\n];\n")
cat(sprintf("const EXPECTED_R2_PLS2: f64 = %.10f;\n", r2_2))
cat("\n")

# -----------------------------------------------------------------------------
# Test 3: PLS with scaling
# -----------------------------------------------------------------------------
cat("// -----------------------------------------------------------------------------\n")
cat("// Test 3: PLS with scaling (SIMPLS)\n")
cat("// -----------------------------------------------------------------------------\n")

n <- 40
p <- 6
ncomp <- 3

# Generate predictors with different scales
X3 <- matrix(rnorm(n * p), n, p)
X3[,1] <- X3[,1] * 100    # Large scale
X3[,2] <- X3[,2] * 0.01   # Small scale
X3[,3] <- X3[,3] * 1000   # Very large scale

# True relationship
beta_true3 <- c(0.02, 50.0, 0.001, 1.0, -0.5, 0.3)
y3 <- X3 %*% beta_true3 + rnorm(n, sd = 2.0)

# Create data frame for pls
df3 <- data.frame(y = y3, X = I(X3))

# Fit PLS with SIMPLS algorithm WITH scaling
fit3 <- plsr(y ~ X, data = df3, ncomp = ncomp, method = "simpls",
             scale = TRUE, center = TRUE)

# Get coefficients for ncomp components (on original scale)
coefs3 <- as.vector(coef(fit3, ncomp = ncomp))
intercept3 <- mean(y3) - sum(colMeans(X3) * coefs3)

# Get predictions on training data
fitted3 <- as.vector(predict(fit3, ncomp = ncomp))

# Compute R²
y_mean3 <- mean(y3)
tss3 <- sum((y3 - y_mean3)^2)
rss3 <- sum((y3 - fitted3)^2)
r2_3 <- 1 - rss3 / tss3

cat(sprintf("// R Code: plsr(y ~ X, ncomp = %d, method = \"simpls\", scale = TRUE)\n", ncomp))
cat(sprintf("const N_PLS3: usize = %d;\n", n))
cat(sprintf("const P_PLS3: usize = %d;\n", p))
cat(sprintf("const NCOMP_PLS3: usize = %d;\n", ncomp))
cat("#[rustfmt::skip]\n")
cat(sprintf("const X_PLS3: [f64; %d] = [\n    ", n * p))
cat(paste(sprintf("%.10f", as.vector(X3)), collapse = ", "))
cat("\n];\n")
cat("#[rustfmt::skip]\n")
cat(sprintf("const Y_PLS3: [f64; %d] = [\n    ", n))
cat(paste(sprintf("%.10f", as.vector(y3)), collapse = ", "))
cat("\n];\n")
cat(sprintf("const EXPECTED_INTERCEPT_PLS3: f64 = %.10f;\n", intercept3))
cat("#[rustfmt::skip]\n")
cat(sprintf("const EXPECTED_COEFS_PLS3: [f64; %d] = [\n    ", p))
cat(paste(sprintf("%.10f", coefs3), collapse = ", "))
cat("\n];\n")
cat(sprintf("const EXPECTED_R2_PLS3: f64 = %.10f;\n", r2_3))
cat("\n")

# -----------------------------------------------------------------------------
# Test 4: Small dataset with single component
# -----------------------------------------------------------------------------
cat("// -----------------------------------------------------------------------------\n")
cat("// Test 4: Small dataset with 1 component (SIMPLS)\n")
cat("// -----------------------------------------------------------------------------\n")

n <- 15
p <- 3
ncomp <- 1

X4 <- matrix(rnorm(n * p), n, p)
beta_true4 <- c(2.0, 1.0, 0.5)
y4 <- X4 %*% beta_true4 + rnorm(n, sd = 0.3)

df4 <- data.frame(y = y4, X = I(X4))
fit4 <- plsr(y ~ X, data = df4, ncomp = ncomp, method = "simpls",
             scale = FALSE, center = TRUE)

coefs4 <- as.vector(coef(fit4, ncomp = ncomp))
intercept4 <- mean(y4) - sum(colMeans(X4) * coefs4)
fitted4 <- as.vector(predict(fit4, ncomp = ncomp))

y_mean4 <- mean(y4)
tss4 <- sum((y4 - y_mean4)^2)
rss4 <- sum((y4 - fitted4)^2)
r2_4 <- 1 - rss4 / tss4

cat(sprintf("// R Code: plsr(y ~ X, ncomp = %d, method = \"simpls\", scale = FALSE)\n", ncomp))
cat(sprintf("const N_PLS4: usize = %d;\n", n))
cat(sprintf("const P_PLS4: usize = %d;\n", p))
cat(sprintf("const NCOMP_PLS4: usize = %d;\n", ncomp))
cat("#[rustfmt::skip]\n")
cat(sprintf("const X_PLS4: [f64; %d] = [\n    ", n * p))
cat(paste(sprintf("%.10f", as.vector(X4)), collapse = ", "))
cat("\n];\n")
cat("#[rustfmt::skip]\n")
cat(sprintf("const Y_PLS4: [f64; %d] = [\n    ", n))
cat(paste(sprintf("%.10f", as.vector(y4)), collapse = ", "))
cat("\n];\n")
cat(sprintf("const EXPECTED_INTERCEPT_PLS4: f64 = %.10f;\n", intercept4))
cat("#[rustfmt::skip]\n")
cat(sprintf("const EXPECTED_COEFS_PLS4: [f64; %d] = [\n    ", p))
cat(paste(sprintf("%.10f", coefs4), collapse = ", "))
cat("\n];\n")
cat(sprintf("const EXPECTED_R2_PLS4: f64 = %.10f;\n", r2_4))
cat("\n")

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
cat("// =============================================================================\n")
cat("// Summary of expected values\n")
cat("// =============================================================================\n")
cat(sprintf("// Test 1: n=%d, p=%d, ncomp=%d, R²=%.6f\n", 30, 5, 2, r2_1))
cat(sprintf("// Test 2: n=%d, p=%d, ncomp=%d, R²=%.6f\n", 50, 8, 4, r2_2))
cat(sprintf("// Test 3: n=%d, p=%d, ncomp=%d, R²=%.6f (with scaling)\n", 40, 6, 3, r2_3))
cat(sprintf("// Test 4: n=%d, p=%d, ncomp=%d, R²=%.6f\n", 15, 3, 1, r2_4))
