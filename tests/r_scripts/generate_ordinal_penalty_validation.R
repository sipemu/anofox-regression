#!/usr/bin/env Rscript
# Validation data for penalized `solvers::ordinal` fits (#110) against
# rms::orm(penalty = , penalty.matrix = ).
#
#   Rscript tests/r_scripts/generate_ordinal_penalty_validation.R > tests/fixtures/ordinal_penalty_validation.rs

suppressMessages(library(rms))
emit <- function(name, v) {
  v <- as.numeric(v)
  cat("#[rustfmt::skip]\n")
  cat(sprintf("pub const %s: [f64; %d] = [\n    ", name, length(v)))
  cat(paste(sprintf("%.17e", v), collapse = ", "))
  cat("\n];\n")
}

# The data of generate_ordinal_validation.R
set.seed(20261030)
n <- 150
x1 <- rnorm(n); x2 <- rbinom(n, 1, 0.4); x3 <- runif(n)
lin <- 0.8 * x1 - 0.6 * x2 + 1.2 * x3
y_cont <- round(lin + rlogis(n), 6)
y_cat <- cut(lin + rlogis(n), c(-Inf, -1, 0, 0.8, 1.8, Inf), labels = FALSE)
d <- data.frame(x1, x2, x3, y_cat, y_cont)
emit("X", t(cbind(x1, x2, x3))); emit("Y_CAT", y_cat); emit("Y_CONT", y_cont)
# orm uses penalty.matrix as given only with penalty > 0 (penalty = 0 ignores it)
pm <- matrix(c(1, 0.3, 0, 0.3, 2, 0, 0, 0, 0.5), 3)
emit("CUSTOM_PENALTY", pm)

out <- function(id, f, full_vcov = TRUE) {
  ns <- f$non.slopes
  emit(sprintf("%s_COEF", id), coef(f))
  emit(sprintf("%s_DEVIANCE", id), f$deviance)                 # null, penalized model
  emit(sprintf("%s_LR_DF", id), f$stats[c("Model L.R.", "d.f.")])
  emit(sprintf("%s_EFFECTIVE_DF", id), f$effective.df.diagonal[-(1:ns)])
  emit(sprintf("%s_VCOV_SLOPES", id), as.matrix(vcov(f, intercepts = "none")))
  if (full_vcov) emit(sprintf("%s_VCOV", id), as.matrix(vcov(f, intercepts = "all")))
}
ctl <- list(eps = 1e-10, gradtol = 1e-9, maxit = 100)
out("PEN2", do.call(orm, c(list(y_cat ~ x1 + x2 + x3, data = d, penalty = 2), ctl)))
out("PROBIT_MATRIX", do.call(orm, c(list(y_cat ~ x1 + x2 + x3, data = d, family = "probit", penalty = 1, penalty.matrix = pm), ctl)))
out("CONT_PEN1", do.call(orm, c(list(y_cont ~ x1 + x2 + x3, data = d, penalty = 1), ctl)), full_vcov = FALSE)
