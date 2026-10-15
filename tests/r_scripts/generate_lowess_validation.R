#!/usr/bin/env Rscript
# Generate validation data for solvers::lowess::lowess against R's stats::lowess.
#
#   Rscript tests/r_scripts/generate_lowess_validation.R > tests/fixtures/lowess_validation.rs

emit <- function(name, v) {
  cat("#[rustfmt::skip]\n")
  cat(sprintf("const %s: [f64; %d] = [\n    ", name, length(v)))
  cat(paste(sprintf("%.17e", as.numeric(v)), collapse = ", "))
  cat("\n];\n")
}

# One case: input x, y (in input order), the lowess call's arguments and the
# sorted x / fitted y it returns.
emit_case <- function(id, title, x, y, f, iter, delta = NULL) {
  cat("// -----------------------------------------------------------------------------\n")
  cat(sprintf("// %s\n", title))
  cat("// -----------------------------------------------------------------------------\n")
  fit <- if (is.null(delta)) lowess(x, y, f = f, iter = iter) else lowess(x, y, f = f, iter = iter, delta = delta)
  emit(sprintf("X_%s", id), x)
  emit(sprintf("Y_%s", id), y)
  cat(sprintf("const F_%s: f64 = %.17e;\n", id, f))
  cat(sprintf("const ITER_%s: usize = %d;\n", id, iter))
  cat(sprintf("const DELTA_%s: Option<f64> = %s;\n", id,
              if (is.null(delta)) "None" else sprintf("Some(%.17e)", delta)))
  emit(sprintf("EXPECTED_X_%s", id), fit$x)
  emit(sprintf("EXPECTED_Y_%s", id), fit$y)
  cat("\n")
}

cat("// =============================================================================\n")
cat("// LOWESS Validation Data Generated from R\n")
cat(sprintf("// R version: %s\n", R.version.string))
cat("// Using stats::lowess(x, y, f, iter, delta)\n")
cat("// =============================================================================\n\n")

set.seed(42)

# Noisy sine on unsorted x, R defaults (f = 2/3, iter = 3, delta = 1% of range).
x <- runif(60, 0, 10)
y <- sin(x) + rnorm(60, sd = 0.3)
emit_case("SINE", "Noisy sine, unsorted x, R defaults", x, y, 2 / 3, 3L)

# Same data, small span, no robustness iterations.
emit_case("SINE_F02", "Noisy sine, f = 0.2, iter = 0", x, y, 0.2, 0L)

# Outliers: robustness iterations pull the curve back.
x <- 1:40
y <- 0.5 * x + rnorm(40)
y[c(7, 21, 33)] <- y[c(7, 21, 33)] + c(25, -30, 20)
emit_case("OUTLIER", "Linear trend with outliers, f = 0.5, iter = 4", x, y, 0.5, 4L)

# Ties in x and delta = 0 (every point fitted).
x <- rep(c(1, 2, 2.5, 4, 4, 4, 7, 8, 8, 10), 2)
y <- log(x) + rnorm(length(x), sd = 0.2)
emit_case("TIES", "Tied x values, delta = 0", x, y, 0.6, 3L, delta = 0)

# Large delta: points within delta are interpolated.
x <- sort(runif(80, 0, 100))
y <- sqrt(x) + rnorm(80, sd = 0.5)
emit_case("DELTA", "Interpolation with delta = 5", x, y, 0.3, 2L, delta = 5)

# Tiny input: three points, f large.
emit_case("TINY", "Three points", c(3, 1, 2), c(2, 1, 4), 1, 3L)
