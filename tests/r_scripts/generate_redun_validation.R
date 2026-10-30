#!/usr/bin/env Rscript
# Validation data for `reduction::redun` (#107, RMS §4.7.1) against
# Hmisc::redun: redundancy analysis, dropping in turn the variable best
# predicted (R^2 above a cutoff) from flexible (spline / dummy / rank)
# expansions of the others.
#
#   Rscript tests/r_scripts/generate_redun_validation.R > tests/fixtures/redun_validation.rs

suppressMessages(library(Hmisc))
emit <- function(name, v) {
  v <- as.numeric(v)
  cat("#[rustfmt::skip]\n")
  cat(sprintf("pub const %s: [f64; %d] = [\n    ", name, length(v)))
  cat(paste(sprintf("%.17e", v), collapse = ", "))
  cat("\n];\n")
}

set.seed(20261103)
n <- 200
a <- rnorm(n); b <- rnorm(n); cc <- runif(n)
d <- data.frame(
  a = a,
  b = b,
  ab = a + b + rnorm(n, sd = 0.2),               # nearly linear in a and b
  a2 = exp(a) + rnorm(n, sd = 0.1),              # nonlinear in a
  g = factor(ifelse(cc < 0.3, "x", ifelse(cc < 0.7, "y", "z"))),
  cc = cc + rnorm(n, sd = 0.05),                 # determines g
  e = rnorm(n)                                   # noise
)
num <- d; num$g <- as.integer(d$g)
emit("X", t(as.matrix(num)))
idx <- function(v) match(sub("^I\\((.*)\\)$", "\\1", v), names(d)) - 1   # 0-based column indices
dump <- function(tag, r) {
  emit(paste0(tag, "_OUT"), idx(r$Out))
  emit(paste0(tag, "_IN"), idx(r$In))
  emit(paste0(tag, "_RSQ"), r$rsquared)
  emit(paste0(tag, "_RSQ1"), r$rsq1)
  later <- if (length(r$r2later)) unlist(lapply(r$r2later, function(z) if (is.null(z)) numeric(0) else z)) else numeric(0)
  emit(paste0(tag, "_LATER"), later)
}
dump("DEF", redun(~ ., data = d, r2 = 0.8))
dump("ADJ", redun(~ ., data = d, r2 = 0.8, type = "adjusted", nk = 4))
dump("NOTL", redun(~ ., data = d, r2 = 0.7, tlinear = FALSE))
dump("ALLCAT", redun(~ ., data = d, r2 = 0.5, allcat = TRUE))
dump("RANK", redun(~ ., data = d, r2 = 0.7, rank = TRUE))
dump("QRANK", redun(~ ., data = d, r2 = 0.7, qrank = TRUE))
dump("LIN", redun(~ a + b + ab + I(a2) + g + cc + e, data = d, r2 = 0.8))
