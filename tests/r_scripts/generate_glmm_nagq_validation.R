#!/usr/bin/env Rscript
# Validation data for GlmmRegressorBuilder::n_agq (#67) against lme4::glmer
# with the default Laplace approximation (nAGQ = 1: fixed effects optimised
# with the variance parameters) and adaptive Gauss-Hermite quadrature
# (nAGQ = 7, a single scalar random intercept).
#
#   Rscript tests/r_scripts/generate_glmm_nagq_validation.R > tests/fixtures/glmm_nagq_validation.rs

suppressMessages(library(lme4))
emit <- function(name, v) {
  v <- as.numeric(v)
  cat("#[rustfmt::skip]\n")
  cat(sprintf("pub const %s: [f64; %d] = [\n    ", name, length(v)))
  cat(paste(sprintf("%.17e", v), collapse = ", "))
  cat("\n];\n")
}

set.seed(20261105)
J <- 25; nj <- 8
g <- rep(1:J, each = nj); n <- J * nj
x1 <- rnorm(n); x2 <- runif(n)
b0 <- rnorm(J, sd = 0.7); b1 <- rnorm(J, sd = 0.3)
yp <- rpois(n, exp(0.3 + 0.4 * x1 - 0.5 * x2 + b0[g] + b1[g] * x1))
yb <- rbinom(n, 1, plogis(-0.2 + 0.8 * x1 + 0.6 * x2 + 1.2 * b0[g]))
emit("G", g); emit("X", t(cbind(x1, x2))); emit("YP", yp); emit("YB", yb)

ctl <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5))
out <- function(tag, f) {
  emit(paste0(tag, "_FIXEF"), fixef(f))
  emit(paste0(tag, "_VCOV"), as.matrix(vcov(f)))
  emit(paste0(tag, "_THETA"), getME(f, "theta"))
  emit(paste0(tag, "_LOGLIK"), c(logLik(f), deviance(f)))
  emit(paste0(tag, "_RANEF"), t(as.matrix(ranef(f)[[1]])))
}
out("P1", glmer(yp ~ x1 + x2 + (1 | g), family = poisson, nAGQ = 1, control = ctl))
out("P7", glmer(yp ~ x1 + x2 + (1 | g), family = poisson, nAGQ = 7, control = ctl))
out("B1", glmer(yb ~ x1 + x2 + (1 | g), family = binomial, nAGQ = 1, control = ctl))
out("B7", glmer(yb ~ x1 + x2 + (1 | g), family = binomial, nAGQ = 7, control = ctl))
out("PS1", glmer(yp ~ x1 + x2 + (1 + x1 | g), family = poisson, nAGQ = 1, control = ctl))
