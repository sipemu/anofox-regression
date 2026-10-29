#!/usr/bin/env Rscript
# Validation data for `transform::areg` (#112, RMS Ch 16) against
# Hmisc::areg: additive regression with optimal (restricted cubic spline)
# transformations of y and the predictors by canonical correlation, the
# inverse y transformation, and bootstrap optimism of R^2 and |error|.
#
#   Rscript tests/r_scripts/generate_areg_validation.R > tests/fixtures/areg_validation.rs

suppressMessages(library(Hmisc))
emit <- function(name, v) {
  v <- as.numeric(v)
  cat("#[rustfmt::skip]\n")
  cat(sprintf("pub const %s: [f64; %d] = [\n    ", name, length(v)))
  cat(paste(sprintf("%.17e", v), collapse = ", "))
  cat("\n];\n")
}

set.seed(20261101)
n <- 150
x1 <- rnorm(n); x2 <- runif(n); g <- sample(c("a", "b", "c"), n, TRUE)
y <- exp(0.5 + 0.5 * x1 + sin(2 * x2) + c(a = 0, b = 0.4, c = -0.3)[g] + rnorm(n, sd = 0.3))
X <- cbind(x1, x2, g = as.integer(factor(g)))
emit("X", t(X)); emit("Y", y)

out <- function(tag, f) {
  emit(paste0(tag, "_XCOEF"), f$xcoefficients)
  emit(paste0(tag, "_YCOEF"), f$ycoefficients)
  emit(paste0(tag, "_STATS"), c(f$rsquared, f$mad, f$med))
  emit(paste0(tag, "_TY"), f$ty)
  emit(paste0(tag, "_LP"), f$linear.predictors)
  emit(paste0(tag, "_TX"), t(f$tx))
}

# smooth y, smooth x1 and x2, categorical g
f <- areg(X, y, xtype = c("s", "s", "c"), nk = 4)
out("S", f)
emit("S_YKNOTS", f$yparms)
yy <- c(min(y), median(y), 5, max(y))
emit("S_YTRANS_AT", yy); emit("S_YTRANS", f$ytrans(yy, coef = f$ycoefficients))
emit("S_YINV", f$yinv(f$linear.predictors[1:8], what = "sample"))

# linear y: ordinary least squares on the x bases
fl <- areg(X, y, xtype = c("s", "l", "c"), ytype = "l", nk = 5)
out("L", fl)

# bootstrap optimism (smooth y): replay areg's own resamples
set.seed(7)
fb <- areg(X, y, xtype = c("s", "s", "c"), nk = 4, B = 20)
set.seed(7)
idx <- replicate(20, sample(1:n, replace = TRUE))
emit("BOOT_IDX", idx - 1)
emit("BOOT_STATS", c(fb$r2opt, fb$r2boot, fb$madopt, fb$madboot, fb$medopt, fb$medboot))
