#!/usr/bin/env Rscript
# Generate validation data for LmDynamicRegressor against greybox::lmDynamic
# (normal distribution, bruteforce = TRUE, explicit lowess span).
#
#   Rscript tests/r_scripts/generate_lm_dynamic_validation.R > tests/fixtures/lm_dynamic_validation.rs

suppressMessages(library(greybox))

emit <- function(name, v) {
  cat("#[rustfmt::skip]\n")
  cat(sprintf("const %s: [f64; %d] = [\n    ", name, length(v)))
  cat(paste(sprintf("%.17e", as.numeric(v)), collapse = ", "))
  cat("\n];\n")
}

# x is emitted row-major (n x p); coefficientsDynamic (n x (p + 1), intercept
# first) and weights (n x 2^p) row-major as well.
emit_case <- function(id, title, data, ic, lowess, f = NULL) {
  cat("// -----------------------------------------------------------------------------\n")
  cat(sprintf("// %s\n", title))
  cat("// -----------------------------------------------------------------------------\n")
  fit <- suppressWarnings(lmDynamic(data, ic = ic, bruteforce = TRUE, lowess = lowess, f = f))
  x <- as.matrix(data[, -1, drop = FALSE])
  cat(sprintf("const N_%s: usize = %d;\n", id, nrow(x)))
  cat(sprintf("const P_%s: usize = %d;\n", id, ncol(x)))
  emit(sprintf("X_%s", id), t(x))
  emit(sprintf("Y_%s", id), data[[1]])
  cat(sprintf("const IC_%s: &str = \"%s\";\n", id, ic))
  cat(sprintf("const SPAN_%s: Option<f64> = %s;\n", id,
              if (lowess) sprintf("Some(%.17e)", f) else "None"))
  emit(sprintf("EXPECTED_DYNAMIC_%s", id), t(fit$coefficientsDynamic))
  emit(sprintf("EXPECTED_WEIGHTS_%s", id), t(fit$weights))
  emit(sprintf("EXPECTED_COEF_%s", id), fit$coefficients)
  cat(sprintf("const EXPECTED_LOGLIK_%s: f64 = %.17e;\n", id, as.numeric(logLik(fit))))
  cat("\n")
}

cat("// =============================================================================\n")
cat("// lmDynamic Validation Data Generated from R\n")
cat(sprintf("// R version: %s\n", R.version.string))
cat(sprintf("// greybox: %s\n", as.character(packageVersion("greybox"))))
cat("// Using greybox::lmDynamic(data, ic, bruteforce = TRUE, lowess, f)\n")
cat("// =============================================================================\n\n")

set.seed(1)
n <- 30
x1 <- rnorm(n)
x2 <- rnorm(n)
y <- 1 + ifelse(1:n > 15, 3, 0.5) * x1 + 0.2 * x2 + rnorm(n, sd = 0.5)
d <- data.frame(y = y, x1 = x1, x2 = x2)
emit_case("STEP_AICC", "Step change in the x1 effect, AICc, no smoothing", d, "AICc", FALSE)
# greybox smooths the logits of the weights, so a weight of exactly 0 or 1
# makes its lowess NaN; noisier data keeps every weight inside (0, 1).
d$y <- 1 + ifelse(1:n > 15, 1, 0.3) * x1 + 0.2 * x2 + rnorm(n, sd = 1.5)
emit_case("STEP_AIC_F05", "Step change, noisier, AIC, lowess f = 0.5", d, "AIC", TRUE, 0.5)

set.seed(7)
n <- 50
x1 <- rnorm(n)
x2 <- runif(n)
x3 <- rnorm(n)
y <- 2 + 1.5 * x1 * (1:n > 25) + 0.8 * x3 + rnorm(n, sd = 0.7)
d <- data.frame(y = y, x1 = x1, x2 = x2, x3 = x3)
emit_case("THREE_BIC_F03", "Three regressors, BIC, lowess f = 0.3", d, "BIC", TRUE, 0.3)
emit_case("THREE_AICC_F09", "Three regressors, AICc, lowess f = 0.9", d, "AICc", TRUE, 0.9)
