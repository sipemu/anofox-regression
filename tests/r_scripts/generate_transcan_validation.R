#!/usr/bin/env Rscript
# Validation data for `reduction::transcan` (#107, RMS §4.7.3, §8.4) against
# Hmisc::transcan(method = "canonical"): transformations, R^2, adjusted R^2
# with and without missing values. Transformation signs are
# arbitrary (LAPACK's); the Rust side aligns them.
#
#   Rscript tests/r_scripts/generate_transcan_validation.R > tests/fixtures/transcan_validation.rs

suppressMessages(library(Hmisc))
emit <- function(name, v) {
  v <- as.numeric(v)
  txt <- ifelse(is.na(v), "f64::NAN", sprintf("%.17e", v))
  cat("#[rustfmt::skip]\n")
  cat(sprintf("pub const %s: [f64; %d] = [\n    ", name, length(v)))
  cat(paste(txt, collapse = ", "))
  cat("\n];\n")
}

set.seed(20261113)
n <- 150
a <- rnorm(n); b <- a^2 + rnorm(n, sd = 0.5); c <- exp(0.7 * a) + rnorm(n, sd = 0.3)
g <- cut(a + rnorm(n), c(-Inf, -0.5, 0.5, Inf), labels = FALSE)
e <- runif(n)
X <- cbind(a, b, c, g, e)
emit("X", t(X))
out <- function(tag, x) {
  t <- transcan(x, categorical = "g", transformed = TRUE, pr = FALSE, pl = FALSE)
  emit(paste0(tag, "_TRANS"), t$transformed)
  emit(paste0(tag, "_RSQ"), t$rsq); emit(paste0(tag, "_RSQADJ"), t$rsq.adj)
}
out("FULL", X)
Xm <- X; Xm[sample(n, 12), "b"] <- NA; Xm[sample(n, 8), "g"] <- NA
emit("XM", t(Xm))
out("MISS", Xm)
