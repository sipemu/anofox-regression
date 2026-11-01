#!/usr/bin/env Rscript
# Validation data for `imputation::areg_impute` (#106, RMS §3.7) against
# Hmisc::aregImpute. The Rust side uses R's random number generator, so with
# the same seed the imputations must agree exactly.
#
#   Rscript tests/r_scripts/generate_aregimpute_validation.R > tests/fixtures/aregimpute_validation.rs

suppressMessages(library(Hmisc))
emit <- function(name, v) {
  v <- as.numeric(v)
  txt <- ifelse(is.na(v), "f64::NAN", sprintf("%.17e", v))
  cat("#[rustfmt::skip]\n")
  cat(sprintf("pub const %s: [f64; %d] = [\n    ", name, length(v)))
  cat(paste(txt, collapse = ", "))
  cat("\n];\n")
}

set.seed(20261112)
n <- 120
x1 <- rnorm(n); b <- rbinom(n, 1, 0.4); g <- factor(sample(c("a", "b", "c"), n, TRUE))
x2 <- exp(0.5 * x1) + 0.5 * b + rnorm(n, sd = 0.3)
y <- 1 + x1 + 0.5 * x2 - 0.7 * b + c(0, 0.5, -0.5)[g] + rnorm(n)
x1[sample(n, 15)] <- NA; x2[sample(n, 12)] <- NA; g[sample(n, 10)] <- NA
d <- data.frame(y, x1, x2, b, g)
emit("X", t(sapply(d, as.numeric)))

out <- function(tag, a) {
  for (v in c("x1", "x2", "g")) emit(sprintf("%s_%s", tag, toupper(v)), t(a$imputed[[v]]))
  emit(paste0(tag, "_RSQ"), a$rsq[c("x1", "x2", "g")])
}
set.seed(7); out("W", aregImpute(~ y + x1 + x2 + b + g, data = d, n.impute = 5, pr = FALSE))
set.seed(8); out("C", aregImpute(~ y + x1 + x2 + b + g, data = d, n.impute = 3, match = "closest", pr = FALSE))
# #210: a linear-only target (I(x1)), tlinear = FALSE and burnin = 1
set.seed(9); out("L", aregImpute(~ y + I(x1) + x2 + b + g, data = d, n.impute = 3, pr = FALSE))
set.seed(10); out("T", aregImpute(~ y + x1 + x2 + b + g, data = d, n.impute = 3, tlinear = FALSE, pr = FALSE))
set.seed(11); out("B", aregImpute(~ y + x1 + x2 + b + g, data = d, n.impute = 2, burnin = 1, pr = FALSE))
