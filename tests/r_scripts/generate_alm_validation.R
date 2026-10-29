#!/usr/bin/env Rscript
# Generate validation data for ALM implementation
# This script creates test cases with known outputs from R's greybox package

library(greybox)

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
  cat(sprintf("// Scale (sigma): %.10f\n", model$scale))

  # Log-likelihood
  cat(sprintf("// Log-likelihood: %.10f\n", logLik(model)))

  # AIC/BIC
  cat(sprintf("// AIC: %.10f\n", AIC(model)))
  cat(sprintf("// BIC: %.10f\n", BIC(model)))

  # Fitted values (first 5)
  fitted_vals <- fitted(model)
  cat(sprintf("// Fitted values (first 5): %s\n",
              paste(sprintf("%.10f", head(fitted_vals, 5)), collapse=", ")))

  # Residuals (first 5)
  resids <- residuals(model)
  cat(sprintf("// Residuals (first 5): %s\n",
              paste(sprintf("%.10f", head(resids, 5)), collapse=", ")))

  cat("\n")
}

cat("// =============================================================================\n")
cat("// ALM Validation Data Generated from R greybox package\n")
cat("// Generated on:", format(Sys.time()), "\n")
cat("// greybox version:", as.character(packageVersion("greybox")), "\n")
cat("// =============================================================================\n\n")

# -----------------------------------------------------------------------------
# Test Case 1: Normal distribution (simple linear regression)
# -----------------------------------------------------------------------------
n <- 50
x1 <- seq(1, 50, length.out = n)
y1 <- 2.5 + 1.5 * x1 + rnorm(n, sd = 3)

model_normal <- alm(y1 ~ x1, distribution = "dnorm")
print_results("Normal Distribution", model_normal, x1, y1)

# Print Rust test case
cat("// Rust test for Normal:\n")
cat(sprintf("const X_NORMAL: [f64; %d] = [%s];\n", n, paste(sprintf("%.6f", x1), collapse=", ")))
cat(sprintf("const Y_NORMAL: [f64; %d] = [%s];\n", n, paste(sprintf("%.6f", y1), collapse=", ")))
cat(sprintf("const EXPECTED_INTERCEPT_NORMAL: f64 = %.10f;\n", coef(model_normal)[1]))
cat(sprintf("const EXPECTED_COEF_NORMAL: f64 = %.10f;\n", coef(model_normal)[2]))
cat(sprintf("const EXPECTED_SCALE_NORMAL: f64 = %.10f;\n", model_normal$scale))
cat(sprintf("const EXPECTED_LL_NORMAL: f64 = %.10f;\n", logLik(model_normal)))
cat("\n")

# -----------------------------------------------------------------------------
# Test Case 2: Laplace distribution (robust regression)
# -----------------------------------------------------------------------------
y2 <- 2.5 + 1.5 * x1 + rnorm(n, sd = 3)
# Add outliers
y2[10] <- y2[10] + 50
y2[40] <- y2[40] - 50

model_laplace <- alm(y2 ~ x1, distribution = "dlaplace")
print_results("Laplace Distribution", model_laplace, x1, y2)

cat("// Rust test for Laplace:\n")
cat(sprintf("const Y_LAPLACE: [f64; %d] = [%s];\n", n, paste(sprintf("%.6f", y2), collapse=", ")))
cat(sprintf("const EXPECTED_INTERCEPT_LAPLACE: f64 = %.10f;\n", coef(model_laplace)[1]))
cat(sprintf("const EXPECTED_COEF_LAPLACE: f64 = %.10f;\n", coef(model_laplace)[2]))
cat(sprintf("const EXPECTED_SCALE_LAPLACE: f64 = %.10f;\n", model_laplace$scale))
cat(sprintf("const EXPECTED_LL_LAPLACE: f64 = %.10f;\n", logLik(model_laplace)))
cat("\n")

# -----------------------------------------------------------------------------
# Test Case 3: Student-t distribution
# -----------------------------------------------------------------------------
y3 <- 2.5 + 1.5 * x1 + rt(n, df = 5) * 3

model_t <- alm(y3 ~ x1, distribution = "dt")
print_results("Student-t Distribution", model_t, x1, y3)

cat("// Rust test for Student-t:\n")
cat(sprintf("const Y_STUDENT_T: [f64; %d] = [%s];\n", n, paste(sprintf("%.6f", y3), collapse=", ")))
cat(sprintf("const EXPECTED_INTERCEPT_T: f64 = %.10f;\n", coef(model_t)[1]))
cat(sprintf("const EXPECTED_COEF_T: f64 = %.10f;\n", coef(model_t)[2]))
cat(sprintf("const EXPECTED_SCALE_T: f64 = %.10f;\n", model_t$scale))
cat(sprintf("const EXPECTED_LL_T: f64 = %.10f;\n", logLik(model_t)))
# greybox's dt has no scale (and estimates df here); ALM's Student t is the
# location-scale t with df = 5. Its converged MLE for MLE_T (#145):
nll_t <- function(q) { s <- exp(q[3]); -sum(dt((y3 - q[1] - q[2] * x1) / s, df = 5, log = TRUE) - log(s)) }
o <- optim(c(coef(lm(y3 ~ x1)), log(sd(y3))), nll_t, method = "BFGS", control = list(reltol = 1e-16, maxit = 1000))
o <- optim(o$par, nll_t, method = "Nelder-Mead", control = list(reltol = 1e-16, maxit = 5000))
o <- optim(o$par, nll_t, method = "BFGS", control = list(reltol = 1e-16, maxit = 1000))
cat(sprintf("const MLE_T: [f64; 3] = [%.10f, %.10f, %.10f];\n", o$par[1], o$par[2], exp(o$par[3])))
cat(sprintf("const MLE_LL_T: f64 = %.10f;\n", -o$value))
cat("\n")

# -----------------------------------------------------------------------------
# Test Case 4: Log-Normal distribution
# -----------------------------------------------------------------------------
y4 <- exp(0.5 + 0.03 * x1 + rnorm(n, sd = 0.3))

model_lnorm <- alm(y4 ~ x1, distribution = "dlnorm")
print_results("Log-Normal Distribution", model_lnorm, x1, y4)

cat("// Rust test for Log-Normal:\n")
cat(sprintf("const Y_LOGNORMAL: [f64; %d] = [%s];\n", n, paste(sprintf("%.6f", y4), collapse=", ")))
cat(sprintf("const EXPECTED_INTERCEPT_LOGNORM: f64 = %.10f;\n", coef(model_lnorm)[1]))
cat(sprintf("const EXPECTED_COEF_LOGNORM: f64 = %.10f;\n", coef(model_lnorm)[2]))
cat(sprintf("const EXPECTED_SCALE_LOGNORM: f64 = %.10f;\n", model_lnorm$scale))
cat(sprintf("const EXPECTED_LL_LOGNORM: f64 = %.10f;\n", logLik(model_lnorm)))
cat("\n")

# -----------------------------------------------------------------------------
# Test Case 5: Poisson distribution (count data)
# -----------------------------------------------------------------------------
x5 <- seq(0, 2, length.out = n)
lambda <- exp(0.5 + 1.0 * x5)
y5 <- rpois(n, lambda)

model_pois <- alm(y5 ~ x5, distribution = "dpois")
print_results("Poisson Distribution", model_pois, x5, y5)

cat("// Rust test for Poisson:\n")
cat(sprintf("const X_POISSON: [f64; %d] = [%s];\n", n, paste(sprintf("%.6f", x5), collapse=", ")))
cat(sprintf("const Y_POISSON: [f64; %d] = [%s];\n", n, paste(sprintf("%.6f", y5), collapse=", ")))
cat(sprintf("const EXPECTED_INTERCEPT_POISSON: f64 = %.10f;\n", coef(model_pois)[1]))
cat(sprintf("const EXPECTED_COEF_POISSON: f64 = %.10f;\n", coef(model_pois)[2]))
cat(sprintf("const EXPECTED_LL_POISSON: f64 = %.10f;\n", logLik(model_pois)))
cat("\n")

# -----------------------------------------------------------------------------
# Test Case 6: Negative Binomial distribution
# -----------------------------------------------------------------------------
# Use same x5
mu_nb <- exp(0.5 + 0.8 * x5)
y6 <- rnbinom(n, size = 2, mu = mu_nb)

model_nbinom <- alm(y6 ~ x5, distribution = "dnbinom")
print_results("Negative Binomial Distribution", model_nbinom, x5, y6)

cat("// Rust test for Negative Binomial:\n")
cat(sprintf("const Y_NEGBINOM: [f64; %d] = [%s];\n", n, paste(sprintf("%.6f", y6), collapse=", ")))
cat(sprintf("const EXPECTED_INTERCEPT_NEGBINOM: f64 = %.10f;\n", coef(model_nbinom)[1]))
cat(sprintf("const EXPECTED_COEF_NEGBINOM: f64 = %.10f;\n", coef(model_nbinom)[2]))
cat(sprintf("const EXPECTED_LL_NEGBINOM: f64 = %.10f;\n", logLik(model_nbinom)))
cat("\n")

# -----------------------------------------------------------------------------
# Test Case 7: Gamma distribution
# -----------------------------------------------------------------------------
shape <- 2
rate <- shape / exp(0.5 + 0.05 * x1)
y7 <- rgamma(n, shape = shape, rate = rate)

model_gamma <- alm(y7 ~ x1, distribution = "dgamma")
print_results("Gamma Distribution", model_gamma, x1, y7)

cat("// Rust test for Gamma:\n")
cat(sprintf("const Y_GAMMA: [f64; %d] = [%s];\n", n, paste(sprintf("%.6f", y7), collapse=", ")))
cat(sprintf("const EXPECTED_INTERCEPT_GAMMA: f64 = %.10f;\n", coef(model_gamma)[1]))
cat(sprintf("const EXPECTED_COEF_GAMMA: f64 = %.10f;\n", coef(model_gamma)[2]))
cat(sprintf("const EXPECTED_SCALE_GAMMA: f64 = %.10f;\n", model_gamma$scale))
cat(sprintf("const EXPECTED_LL_GAMMA: f64 = %.10f;\n", logLik(model_gamma)))
cat("\n")

# -----------------------------------------------------------------------------
# Test Case 8: Inverse Gaussian distribution
# -----------------------------------------------------------------------------
mu_ig <- exp(0.5 + 0.03 * x1)
y8 <- statmod::rinvgauss(n, mean = mu_ig, shape = 2)

model_ig <- alm(y8 ~ x1, distribution = "dinvgauss")
print_results("Inverse Gaussian Distribution", model_ig, x1, y8)

cat("// Rust test for Inverse Gaussian:\n")
cat(sprintf("const Y_INVGAUSS: [f64; %d] = [%s];\n", n, paste(sprintf("%.6f", y8), collapse=", ")))
cat(sprintf("const EXPECTED_INTERCEPT_INVGAUSS: f64 = %.10f;\n", coef(model_ig)[1]))
cat(sprintf("const EXPECTED_COEF_INVGAUSS: f64 = %.10f;\n", coef(model_ig)[2]))
cat(sprintf("const EXPECTED_LL_INVGAUSS: f64 = %.10f;\n", logLik(model_ig)))
cat("\n")

# -----------------------------------------------------------------------------
# Test Case 9: Asymmetric Laplace (Quantile regression at 0.75)
# -----------------------------------------------------------------------------
y9 <- 2.5 + 1.5 * x1 + rnorm(n, sd = 3)

model_alaplace <- alm(y9 ~ x1, distribution = "dalaplace", alpha = 0.75)
print_results("Asymmetric Laplace (alpha=0.75)", model_alaplace, x1, y9)

cat("// Rust test for Asymmetric Laplace (quantile = 0.75):\n")
cat(sprintf("const Y_ALAPLACE: [f64; %d] = [%s];\n", n, paste(sprintf("%.6f", y9), collapse=", ")))
cat(sprintf("const EXPECTED_INTERCEPT_ALAPLACE: f64 = %.10f;\n", coef(model_alaplace)[1]))
cat(sprintf("const EXPECTED_COEF_ALAPLACE: f64 = %.10f;\n", coef(model_alaplace)[2]))
cat(sprintf("const EXPECTED_SCALE_ALAPLACE: f64 = %.10f;\n", model_alaplace$scale))
cat(sprintf("const EXPECTED_LL_ALAPLACE: f64 = %.10f;\n", logLik(model_alaplace)))
cat("\n")

# -----------------------------------------------------------------------------
# Test Case 10: Logistic distribution
# -----------------------------------------------------------------------------
y10 <- 2.5 + 1.5 * x1 + rlogis(n, scale = 2)

model_logis <- alm(y10 ~ x1, distribution = "dlogis")
print_results("Logistic Distribution", model_logis, x1, y10)

cat("// Rust test for Logistic:\n")
cat(sprintf("const Y_LOGISTIC: [f64; %d] = [%s];\n", n, paste(sprintf("%.6f", y10), collapse=", ")))
cat(sprintf("const EXPECTED_INTERCEPT_LOGISTIC: f64 = %.10f;\n", coef(model_logis)[1]))
cat(sprintf("const EXPECTED_COEF_LOGISTIC: f64 = %.10f;\n", coef(model_logis)[2]))
cat(sprintf("const EXPECTED_SCALE_LOGISTIC: f64 = %.10f;\n", model_logis$scale))
cat(sprintf("const EXPECTED_LL_LOGISTIC: f64 = %.10f;\n", logLik(model_logis)))
cat("\n")

# -----------------------------------------------------------------------------
# Summary statistics comparison
# -----------------------------------------------------------------------------
cat("// =============================================================================\n")
cat("// Summary of expected values for validation tests\n")
cat("// =============================================================================\n")
cat(sprintf("// Normal:       intercept=%.6f, coef=%.6f, scale=%.6f, LL=%.6f\n",
            coef(model_normal)[1], coef(model_normal)[2], model_normal$scale, logLik(model_normal)))
cat(sprintf("// Laplace:      intercept=%.6f, coef=%.6f, scale=%.6f, LL=%.6f\n",
            coef(model_laplace)[1], coef(model_laplace)[2], model_laplace$scale, logLik(model_laplace)))
cat(sprintf("// Student-t:    intercept=%.6f, coef=%.6f, scale=%.6f, LL=%.6f\n",
            coef(model_t)[1], coef(model_t)[2], model_t$scale, logLik(model_t)))
cat(sprintf("// Log-Normal:   intercept=%.6f, coef=%.6f, scale=%.6f, LL=%.6f\n",
            coef(model_lnorm)[1], coef(model_lnorm)[2], model_lnorm$scale, logLik(model_lnorm)))
cat(sprintf("// Poisson:      intercept=%.6f, coef=%.6f, LL=%.6f\n",
            coef(model_pois)[1], coef(model_pois)[2], logLik(model_pois)))
cat(sprintf("// NegBinom:     intercept=%.6f, coef=%.6f, LL=%.6f\n",
            coef(model_nbinom)[1], coef(model_nbinom)[2], logLik(model_nbinom)))
cat(sprintf("// Gamma:        intercept=%.6f, coef=%.6f, scale=%.6f, LL=%.6f\n",
            coef(model_gamma)[1], coef(model_gamma)[2], model_gamma$scale, logLik(model_gamma)))
cat(sprintf("// InvGauss:     intercept=%.6f, coef=%.6f, LL=%.6f\n",
            coef(model_ig)[1], coef(model_ig)[2], logLik(model_ig)))
cat(sprintf("// AsymLaplace:  intercept=%.6f, coef=%.6f, scale=%.6f, LL=%.6f\n",
            coef(model_alaplace)[1], coef(model_alaplace)[2], model_alaplace$scale, logLik(model_alaplace)))
cat(sprintf("// Logistic:     intercept=%.6f, coef=%.6f, scale=%.6f, LL=%.6f\n",
            coef(model_logis)[1], coef(model_logis)[2], model_logis$scale, logLik(model_logis)))
