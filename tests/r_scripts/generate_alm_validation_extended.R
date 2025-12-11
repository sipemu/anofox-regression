#!/usr/bin/env Rscript
# Extended ALM validation data for 17 additional distributions
# This script extends generate_alm_validation.R with missing distributions
#
# Distributions covered here:
# - Count: Binomial, (NegBinom and Geometric already in base, but we add more tests)
# - Positive Continuous: Exponential, FoldedNormal, RectifiedNormal
# - Unit Interval: Beta, LogitNormal
# - Robust/Shape: GeneralisedNormal, S
# - Log-domain: LogLaplace, LogGeneralisedNormal, BoxCoxNormal
# - Cumulative: CumulativeLogistic, CumulativeNormal

library(greybox)
library(statmod)  # for rinvgauss

# Set seed for reproducibility
set.seed(42)

# Helper function to print results in Rust-compatible format
print_results <- function(name, model, x, y) {
  cat(sprintf("\n// %s\n", name))
  cat(sprintf("// Distribution: %s\n", model$distribution))

  # Print data
  cat("// X data:\n")
  cat(sprintf("// %s\n", paste(round(x, 6), collapse=", ")))
  cat("// Y data:\n")
  cat(sprintf("// %s\n", paste(round(y, 6), collapse=", ")))

  # Coefficients
  coefs <- coef(model)
  cat(sprintf("// Intercept: %.10f\n", coefs[1]))
  if(length(coefs) > 1) {
    cat(sprintf("// Coefficients: %s\n", paste(sprintf("%.10f", coefs[-1]), collapse=", ")))
  }

  # Scale parameter
  if(!is.null(model$scale)) {
    cat(sprintf("// Scale (sigma): %.10f\n", model$scale))
  }

  # Log-likelihood
  cat(sprintf("// Log-likelihood: %.10f\n", logLik(model)))

  # AIC/BIC
  cat(sprintf("// AIC: %.10f\n", AIC(model)))
  cat(sprintf("// BIC: %.10f\n", BIC(model)))

  cat("\n")
}

cat("// =============================================================================\n")
cat("// Extended ALM Validation Data Generated from R greybox package\n")
cat("// Generated on:", format(Sys.time()), "\n")
cat("// greybox version:", as.character(packageVersion("greybox")), "\n")
cat("// =============================================================================\n\n")

# Common x data for most tests
n <- 50
x1 <- seq(1, 50, length.out = n)

# =============================================================================
# GROUP 1: COUNT DISTRIBUTIONS
# =============================================================================

cat("// =============================================================================\n")
cat("// GROUP 1: COUNT DISTRIBUTIONS\n")
cat("// =============================================================================\n\n")

# -----------------------------------------------------------------------------
# Test Case 1: Binomial distribution (counts, not proportions)
# -----------------------------------------------------------------------------
x_binom <- seq(0.1, 2.0, length.out = n)
p_true <- 1 / (1 + exp(-(0.5 + 0.8 * x_binom)))  # logistic
y_binom <- rbinom(n, size = 10, prob = p_true)  # integer counts 0-10

model_binom <- alm(y_binom ~ x_binom, distribution = "dbinom", size = 10)
print_results("Binomial Distribution (size=10)", model_binom, x_binom, y_binom)

cat("// Rust test for Binomial:\n")
cat(sprintf("const X_BINOMIAL: [f64; %d] = [%s];\n", n, paste(sprintf("%.6f", x_binom), collapse=", ")))
cat(sprintf("const Y_BINOMIAL: [f64; %d] = [%s];\n", n, paste(sprintf("%.6f", y_binom), collapse=", ")))
cat(sprintf("const EXPECTED_INTERCEPT_BINOMIAL: f64 = %.10f;\n", coef(model_binom)[1]))
cat(sprintf("const EXPECTED_COEF_BINOMIAL: f64 = %.10f;\n", coef(model_binom)[2]))
cat(sprintf("const EXPECTED_LL_BINOMIAL: f64 = %.10f;\n", logLik(model_binom)))
cat("const BINOMIAL_SIZE: usize = 10;\n")
cat("\n")

# -----------------------------------------------------------------------------
# Test Case 2: Geometric distribution
# -----------------------------------------------------------------------------
x_geom <- seq(0.1, 2.0, length.out = n)
# Geometric needs positive integers (number of failures before first success)
p_geom <- 1 / (1 + exp(-(0.5 + 0.5 * x_geom)))
y_geom <- rgeom(n, prob = p_geom)

model_geom <- alm(y_geom ~ x_geom, distribution = "dgeom")
print_results("Geometric Distribution", model_geom, x_geom, y_geom)

cat("// Rust test for Geometric:\n")
cat(sprintf("const X_GEOMETRIC: [f64; %d] = [%s];\n", n, paste(sprintf("%.6f", x_geom), collapse=", ")))
cat(sprintf("const Y_GEOMETRIC: [f64; %d] = [%s];\n", n, paste(sprintf("%.6f", y_geom), collapse=", ")))
cat(sprintf("const EXPECTED_INTERCEPT_GEOMETRIC: f64 = %.10f;\n", coef(model_geom)[1]))
cat(sprintf("const EXPECTED_COEF_GEOMETRIC: f64 = %.10f;\n", coef(model_geom)[2]))
cat(sprintf("const EXPECTED_LL_GEOMETRIC: f64 = %.10f;\n", logLik(model_geom)))
cat("\n")

# =============================================================================
# GROUP 2: POSITIVE CONTINUOUS DISTRIBUTIONS
# =============================================================================

cat("// =============================================================================\n")
cat("// GROUP 2: POSITIVE CONTINUOUS DISTRIBUTIONS\n")
cat("// =============================================================================\n\n")

# -----------------------------------------------------------------------------
# Test Case 3: Exponential distribution
# -----------------------------------------------------------------------------
lambda_exp <- exp(0.5 + 0.02 * x1)
y_exp <- rexp(n, rate = 1/lambda_exp)

model_exp <- alm(y_exp ~ x1, distribution = "dexp")
print_results("Exponential Distribution", model_exp, x1, y_exp)

cat("// Rust test for Exponential:\n")
cat(sprintf("const Y_EXPONENTIAL: [f64; %d] = [%s];\n", n, paste(sprintf("%.6f", y_exp), collapse=", ")))
cat(sprintf("const EXPECTED_INTERCEPT_EXPONENTIAL: f64 = %.10f;\n", coef(model_exp)[1]))
cat(sprintf("const EXPECTED_COEF_EXPONENTIAL: f64 = %.10f;\n", coef(model_exp)[2]))
cat(sprintf("const EXPECTED_LL_EXPONENTIAL: f64 = %.10f;\n", logLik(model_exp)))
cat("\n")

# -----------------------------------------------------------------------------
# Test Case 4: Folded Normal distribution
# -----------------------------------------------------------------------------
mu_fn <- 0.5 + 0.05 * x1
y_fn <- abs(rnorm(n, mean = mu_fn, sd = 2))

model_fn <- alm(y_fn ~ x1, distribution = "dfnorm")
print_results("Folded Normal Distribution", model_fn, x1, y_fn)

cat("// Rust test for Folded Normal:\n")
cat(sprintf("const Y_FOLDEDNORMAL: [f64; %d] = [%s];\n", n, paste(sprintf("%.6f", y_fn), collapse=", ")))
cat(sprintf("const EXPECTED_INTERCEPT_FOLDEDNORMAL: f64 = %.10f;\n", coef(model_fn)[1]))
cat(sprintf("const EXPECTED_COEF_FOLDEDNORMAL: f64 = %.10f;\n", coef(model_fn)[2]))
cat(sprintf("const EXPECTED_SCALE_FOLDEDNORMAL: f64 = %.10f;\n", model_fn$scale))
cat(sprintf("const EXPECTED_LL_FOLDEDNORMAL: f64 = %.10f;\n", logLik(model_fn)))
cat("\n")

# -----------------------------------------------------------------------------
# Test Case 5: Rectified Normal distribution
# -----------------------------------------------------------------------------
x_rect <- seq(-2, 3, length.out = n)
mu_rect <- 1.0 + 0.5 * x_rect
sigma_rect <- 1.5
latent <- mu_rect + rnorm(n, sd = sigma_rect)
y_rect <- pmax(latent, 0)  # rectify at zero

model_rect <- alm(y_rect ~ x_rect, distribution = "drectnorm")
print_results("Rectified Normal Distribution", model_rect, x_rect, y_rect)

cat("// Rust test for Rectified Normal:\n")
cat(sprintf("const X_RECTIFIEDNORMAL: [f64; %d] = [%s];\n", n, paste(sprintf("%.6f", x_rect), collapse=", ")))
cat(sprintf("const Y_RECTIFIEDNORMAL: [f64; %d] = [%s];\n", n, paste(sprintf("%.6f", y_rect), collapse=", ")))
cat(sprintf("const EXPECTED_INTERCEPT_RECTIFIEDNORMAL: f64 = %.10f;\n", coef(model_rect)[1]))
cat(sprintf("const EXPECTED_COEF_RECTIFIEDNORMAL: f64 = %.10f;\n", coef(model_rect)[2]))
cat(sprintf("const EXPECTED_SCALE_RECTIFIEDNORMAL: f64 = %.10f;\n", model_rect$scale))
cat(sprintf("const EXPECTED_LL_RECTIFIEDNORMAL: f64 = %.10f;\n", logLik(model_rect)))
cat("\n")

# =============================================================================
# GROUP 3: UNIT INTERVAL DISTRIBUTIONS (0, 1)
# =============================================================================

cat("// =============================================================================\n")
cat("// GROUP 3: UNIT INTERVAL DISTRIBUTIONS (0, 1)\n")
cat("// =============================================================================\n\n")

# -----------------------------------------------------------------------------
# Test Case 6: Beta distribution
# -----------------------------------------------------------------------------
x_unit <- seq(-2, 2, length.out = n)
mu_beta <- 1 / (1 + exp(-(0.3 + 0.5 * x_unit)))  # logistic mean in (0,1)
phi_beta <- 5  # precision
alpha_beta <- mu_beta * phi_beta
beta_beta <- (1 - mu_beta) * phi_beta
y_beta <- rbeta(n, alpha_beta, beta_beta)
# Clamp to avoid exact 0 or 1
y_beta <- pmin(pmax(y_beta, 0.001), 0.999)

model_beta <- alm(y_beta ~ x_unit, distribution = "dbeta")
print_results("Beta Distribution", model_beta, x_unit, y_beta)

cat("// Rust test for Beta:\n")
cat(sprintf("const X_BETA: [f64; %d] = [%s];\n", n, paste(sprintf("%.6f", x_unit), collapse=", ")))
cat(sprintf("const Y_BETA: [f64; %d] = [%s];\n", n, paste(sprintf("%.6f", y_beta), collapse=", ")))
cat(sprintf("const EXPECTED_INTERCEPT_BETA: f64 = %.10f;\n", coef(model_beta)[1]))
cat(sprintf("const EXPECTED_COEF_BETA: f64 = %.10f;\n", coef(model_beta)[2]))
cat(sprintf("const EXPECTED_LL_BETA: f64 = %.10f;\n", logLik(model_beta)))
cat("\n")

# -----------------------------------------------------------------------------
# Test Case 7: Logit-Normal distribution
# -----------------------------------------------------------------------------
logit_mu <- 0.2 + 0.4 * x_unit
y_logitnorm <- 1 / (1 + exp(-(logit_mu + rnorm(n, sd = 0.5))))
y_logitnorm <- pmin(pmax(y_logitnorm, 0.001), 0.999)

model_logitnorm <- alm(y_logitnorm ~ x_unit, distribution = "dlogitnorm")
print_results("Logit-Normal Distribution", model_logitnorm, x_unit, y_logitnorm)

cat("// Rust test for Logit-Normal:\n")
cat(sprintf("const Y_LOGITNORMAL: [f64; %d] = [%s];\n", n, paste(sprintf("%.6f", y_logitnorm), collapse=", ")))
cat(sprintf("const EXPECTED_INTERCEPT_LOGITNORMAL: f64 = %.10f;\n", coef(model_logitnorm)[1]))
cat(sprintf("const EXPECTED_COEF_LOGITNORMAL: f64 = %.10f;\n", coef(model_logitnorm)[2]))
cat(sprintf("const EXPECTED_SCALE_LOGITNORMAL: f64 = %.10f;\n", model_logitnorm$scale))
cat(sprintf("const EXPECTED_LL_LOGITNORMAL: f64 = %.10f;\n", logLik(model_logitnorm)))
cat("\n")

# =============================================================================
# GROUP 4: ROBUST/SHAPE DISTRIBUTIONS
# =============================================================================

cat("// =============================================================================\n")
cat("// GROUP 4: ROBUST/SHAPE DISTRIBUTIONS\n")
cat("// =============================================================================\n\n")

# -----------------------------------------------------------------------------
# Test Case 8: Generalised Normal (Subbotin) distribution
# -----------------------------------------------------------------------------
y_gn <- 2.5 + 1.5 * x1 + rnorm(n, sd = 2)

model_gnorm <- alm(y_gn ~ x1, distribution = "dgnorm", shape = 1.5)
print_results("Generalised Normal (shape=1.5)", model_gnorm, x1, y_gn)

cat("// Rust test for Generalised Normal:\n")
cat(sprintf("const Y_GENERALISEDNORMAL: [f64; %d] = [%s];\n", n, paste(sprintf("%.6f", y_gn), collapse=", ")))
cat(sprintf("const EXPECTED_INTERCEPT_GENERALISEDNORMAL: f64 = %.10f;\n", coef(model_gnorm)[1]))
cat(sprintf("const EXPECTED_COEF_GENERALISEDNORMAL: f64 = %.10f;\n", coef(model_gnorm)[2]))
cat(sprintf("const EXPECTED_SCALE_GENERALISEDNORMAL: f64 = %.10f;\n", model_gnorm$scale))
cat(sprintf("const EXPECTED_LL_GENERALISEDNORMAL: f64 = %.10f;\n", logLik(model_gnorm)))
cat("const GENERALISEDNORMAL_SHAPE: f64 = 1.5;\n")
cat("\n")

# -----------------------------------------------------------------------------
# Test Case 9: S distribution
# -----------------------------------------------------------------------------
y_s <- 2.5 + 1.5 * x1 + rnorm(n, sd = 3)
# Add outliers to test robustness
y_s[10] <- y_s[10] + 40
y_s[40] <- y_s[40] - 40

model_s <- alm(y_s ~ x1, distribution = "ds")
print_results("S Distribution", model_s, x1, y_s)

cat("// Rust test for S distribution:\n")
cat(sprintf("const Y_S: [f64; %d] = [%s];\n", n, paste(sprintf("%.6f", y_s), collapse=", ")))
cat(sprintf("const EXPECTED_INTERCEPT_S: f64 = %.10f;\n", coef(model_s)[1]))
cat(sprintf("const EXPECTED_COEF_S: f64 = %.10f;\n", coef(model_s)[2]))
cat(sprintf("const EXPECTED_SCALE_S: f64 = %.10f;\n", model_s$scale))
cat(sprintf("const EXPECTED_LL_S: f64 = %.10f;\n", logLik(model_s)))
cat("\n")

# =============================================================================
# GROUP 5: LOG-DOMAIN DISTRIBUTIONS
# =============================================================================

cat("// =============================================================================\n")
cat("// GROUP 5: LOG-DOMAIN DISTRIBUTIONS\n")
cat("// =============================================================================\n\n")

# -----------------------------------------------------------------------------
# Test Case 10: Log-Laplace distribution
# -----------------------------------------------------------------------------
y_llaplace <- exp(0.5 + 0.03 * x1 + rnorm(n, sd = 0.3))

model_llaplace <- alm(y_llaplace ~ x1, distribution = "dllaplace")
print_results("Log-Laplace Distribution", model_llaplace, x1, y_llaplace)

cat("// Rust test for Log-Laplace:\n")
cat(sprintf("const Y_LOGLAPLACE: [f64; %d] = [%s];\n", n, paste(sprintf("%.6f", y_llaplace), collapse=", ")))
cat(sprintf("const EXPECTED_INTERCEPT_LOGLAPLACE: f64 = %.10f;\n", coef(model_llaplace)[1]))
cat(sprintf("const EXPECTED_COEF_LOGLAPLACE: f64 = %.10f;\n", coef(model_llaplace)[2]))
cat(sprintf("const EXPECTED_SCALE_LOGLAPLACE: f64 = %.10f;\n", model_llaplace$scale))
cat(sprintf("const EXPECTED_LL_LOGLAPLACE: f64 = %.10f;\n", logLik(model_llaplace)))
cat("\n")

# -----------------------------------------------------------------------------
# Test Case 11: Log-Generalised Normal distribution
# -----------------------------------------------------------------------------
y_lgn <- exp(0.5 + 0.03 * x1 + rnorm(n, sd = 0.25))

model_lgn <- alm(y_lgn ~ x1, distribution = "dlgnorm", shape = 1.5)
print_results("Log-Generalised Normal (shape=1.5)", model_lgn, x1, y_lgn)

cat("// Rust test for Log-Generalised Normal:\n")
cat(sprintf("const Y_LOGGENERALISEDNORMAL: [f64; %d] = [%s];\n", n, paste(sprintf("%.6f", y_lgn), collapse=", ")))
cat(sprintf("const EXPECTED_INTERCEPT_LOGGENERALISEDNORMAL: f64 = %.10f;\n", coef(model_lgn)[1]))
cat(sprintf("const EXPECTED_COEF_LOGGENERALISEDNORMAL: f64 = %.10f;\n", coef(model_lgn)[2]))
cat(sprintf("const EXPECTED_SCALE_LOGGENERALISEDNORMAL: f64 = %.10f;\n", model_lgn$scale))
cat(sprintf("const EXPECTED_LL_LOGGENERALISEDNORMAL: f64 = %.10f;\n", logLik(model_lgn)))
cat("const LOGGENERALISEDNORMAL_SHAPE: f64 = 1.5;\n")
cat("\n")

# -----------------------------------------------------------------------------
# Test Case 12: Box-Cox Normal distribution (lambda = 0.5 for sqrt transform)
# -----------------------------------------------------------------------------
y_bcn <- (2 + 0.1 * x1 + rnorm(n, sd = 0.5))^2  # will be sqrt-transformed with lambda=0.5

model_bcn <- alm(y_bcn ~ x1, distribution = "dbcnorm", lambdaBC = 0.5)
print_results("Box-Cox Normal (lambda=0.5)", model_bcn, x1, y_bcn)

cat("// Rust test for Box-Cox Normal:\n")
cat(sprintf("const Y_BOXCOXNORMAL: [f64; %d] = [%s];\n", n, paste(sprintf("%.6f", y_bcn), collapse=", ")))
cat(sprintf("const EXPECTED_INTERCEPT_BOXCOXNORMAL: f64 = %.10f;\n", coef(model_bcn)[1]))
cat(sprintf("const EXPECTED_COEF_BOXCOXNORMAL: f64 = %.10f;\n", coef(model_bcn)[2]))
cat(sprintf("const EXPECTED_SCALE_BOXCOXNORMAL: f64 = %.10f;\n", model_bcn$scale))
cat(sprintf("const EXPECTED_LL_BOXCOXNORMAL: f64 = %.10f;\n", logLik(model_bcn)))
cat("const BOXCOXNORMAL_LAMBDA: f64 = 0.5;\n")
cat("\n")

# =============================================================================
# GROUP 6: CUMULATIVE/ORDINAL DISTRIBUTIONS
# =============================================================================

cat("// =============================================================================\n")
cat("// GROUP 6: CUMULATIVE/ORDINAL DISTRIBUTIONS\n")
cat("// =============================================================================\n\n")

# -----------------------------------------------------------------------------
# Test Case 13: Cumulative Logistic (plogis) - Binary outcome
# -----------------------------------------------------------------------------
x_ord <- seq(-3, 3, length.out = n)
eta <- 0.5 + 1.0 * x_ord
prob_clogis <- 1 / (1 + exp(-eta))
y_clogis <- rbinom(n, 1, prob_clogis)

# Try fitting - this may have issues
tryCatch({
  model_clogis <- alm(y_clogis ~ x_ord, distribution = "plogis")
  print_results("Cumulative Logistic Distribution", model_clogis, x_ord, y_clogis)

  cat("// Rust test for Cumulative Logistic:\n")
  cat(sprintf("const X_CUMULATIVELOGISTIC: [f64; %d] = [%s];\n", n, paste(sprintf("%.6f", x_ord), collapse=", ")))
  cat(sprintf("const Y_CUMULATIVELOGISTIC: [f64; %d] = [%s];\n", n, paste(sprintf("%.6f", y_clogis), collapse=", ")))
  cat(sprintf("const EXPECTED_INTERCEPT_CUMULATIVELOGISTIC: f64 = %.10f;\n", coef(model_clogis)[1]))
  cat(sprintf("const EXPECTED_COEF_CUMULATIVELOGISTIC: f64 = %.10f;\n", coef(model_clogis)[2]))
  cat(sprintf("const EXPECTED_LL_CUMULATIVELOGISTIC: f64 = %.10f;\n", logLik(model_clogis)))
}, error = function(e) {
  cat(sprintf("// ERROR fitting Cumulative Logistic: %s\n", e$message))
  cat("// Cumulative Logistic may need special handling\n")
})
cat("\n")

# -----------------------------------------------------------------------------
# Test Case 14: Cumulative Normal (pnorm) - Binary outcome
# -----------------------------------------------------------------------------
prob_cnorm <- pnorm(0.5 + 1.0 * x_ord)
y_cnorm <- rbinom(n, 1, prob_cnorm)

tryCatch({
  model_cnorm <- alm(y_cnorm ~ x_ord, distribution = "pnorm")
  print_results("Cumulative Normal Distribution", model_cnorm, x_ord, y_cnorm)

  cat("// Rust test for Cumulative Normal:\n")
  cat(sprintf("const X_CUMULATIVENORMAL: [f64; %d] = [%s];\n", n, paste(sprintf("%.6f", x_ord), collapse=", ")))
  cat(sprintf("const Y_CUMULATIVENORMAL: [f64; %d] = [%s];\n", n, paste(sprintf("%.6f", y_cnorm), collapse=", ")))
  cat(sprintf("const EXPECTED_INTERCEPT_CUMULATIVENORMAL: f64 = %.10f;\n", coef(model_cnorm)[1]))
  cat(sprintf("const EXPECTED_COEF_CUMULATIVENORMAL: f64 = %.10f;\n", coef(model_cnorm)[2]))
  cat(sprintf("const EXPECTED_LL_CUMULATIVENORMAL: f64 = %.10f;\n", logLik(model_cnorm)))
}, error = function(e) {
  cat(sprintf("// ERROR fitting Cumulative Normal: %s\n", e$message))
  cat("// Cumulative Normal may need special handling\n")
})
cat("\n")

# =============================================================================
# SUMMARY
# =============================================================================

cat("// =============================================================================\n")
cat("// Summary of expected values for extended validation tests\n")
cat("// =============================================================================\n")
cat(sprintf("// Binomial:           intercept=%.6f, coef=%.6f, LL=%.6f\n",
            coef(model_binom)[1], coef(model_binom)[2], logLik(model_binom)))
cat(sprintf("// Geometric:          intercept=%.6f, coef=%.6f, LL=%.6f\n",
            coef(model_geom)[1], coef(model_geom)[2], logLik(model_geom)))
cat(sprintf("// Exponential:        intercept=%.6f, coef=%.6f, LL=%.6f\n",
            coef(model_exp)[1], coef(model_exp)[2], logLik(model_exp)))
cat(sprintf("// FoldedNormal:       intercept=%.6f, coef=%.6f, scale=%.6f, LL=%.6f\n",
            coef(model_fn)[1], coef(model_fn)[2], model_fn$scale, logLik(model_fn)))
cat(sprintf("// RectifiedNormal:    intercept=%.6f, coef=%.6f, scale=%.6f, LL=%.6f\n",
            coef(model_rect)[1], coef(model_rect)[2], model_rect$scale, logLik(model_rect)))
cat(sprintf("// Beta:               intercept=%.6f, coef=%.6f, LL=%.6f\n",
            coef(model_beta)[1], coef(model_beta)[2], logLik(model_beta)))
cat(sprintf("// LogitNormal:        intercept=%.6f, coef=%.6f, scale=%.6f, LL=%.6f\n",
            coef(model_logitnorm)[1], coef(model_logitnorm)[2], model_logitnorm$scale, logLik(model_logitnorm)))
cat(sprintf("// GeneralisedNormal:  intercept=%.6f, coef=%.6f, scale=%.6f, LL=%.6f\n",
            coef(model_gnorm)[1], coef(model_gnorm)[2], model_gnorm$scale, logLik(model_gnorm)))
cat(sprintf("// S:                  intercept=%.6f, coef=%.6f, scale=%.6f, LL=%.6f\n",
            coef(model_s)[1], coef(model_s)[2], model_s$scale, logLik(model_s)))
cat(sprintf("// LogLaplace:         intercept=%.6f, coef=%.6f, scale=%.6f, LL=%.6f\n",
            coef(model_llaplace)[1], coef(model_llaplace)[2], model_llaplace$scale, logLik(model_llaplace)))
cat(sprintf("// LogGenNormal:       intercept=%.6f, coef=%.6f, scale=%.6f, LL=%.6f\n",
            coef(model_lgn)[1], coef(model_lgn)[2], model_lgn$scale, logLik(model_lgn)))
cat(sprintf("// BoxCoxNormal:       intercept=%.6f, coef=%.6f, scale=%.6f, LL=%.6f\n",
            coef(model_bcn)[1], coef(model_bcn)[2], model_bcn$scale, logLik(model_bcn)))
