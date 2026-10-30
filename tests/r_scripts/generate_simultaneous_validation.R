#!/usr/bin/env Rscript
# Validation data for `effects::simultaneous_critical` /
# `ModelSpec::predict_grid_simultaneous` (#116 follow-up in #135) against
# rms::Predict(conf.type = "simultaneous") (multcomp single-step). R's
# critical values come from mvtnorm's randomised Genz-Bretz algorithm; a
# tighter run (abseps = 1e-6, mean of four seeds) is emitted alongside.
#
#   Rscript tests/r_scripts/generate_simultaneous_validation.R > tests/fixtures/simultaneous_validation.rs

suppressMessages({library(rms); library(multcomp)})
emit <- function(name, v) {
  v <- as.numeric(v)
  cat("#[rustfmt::skip]\n")
  cat(sprintf("pub const %s: [f64; %d] = [\n    ", name, length(v)))
  cat(paste(sprintf("%.17e", v), collapse = ", "))
  cat("\n];\n")
}

set.seed(20261111)
n <- 220
age <- round(rnorm(n, 50, 10), 1); sex <- rbinom(n, 1, 0.5); w <- runif(n, 40, 100)
yb <- rbinom(n, 1, plogis(-3 + 0.05 * age + 0.5 * sex))
yc <- 2 + 0.03 * age - 0.4 * sex + 0.02 * w + rnorm(n)
emit("AGE", age); emit("SEX", sex); emit("W", w)
dd <- datadist(age, sex, w); options(datadist = "dd")

tight <- function(R, df = 0) {
  q <- sapply(1:4, function(s) { set.seed(s)
    if (df == 0) qmvnorm(0.95, corr = R, tail = "both.tails", algorithm = GenzBretz(maxpts = 2e6, abseps = 1e-6))$quantile
    else qmvt(0.95, df = df, corr = R, tail = "both.tails", algorithm = GenzBretz(maxpts = 2e6, abseps = 1e-6))$quantile })
  mean(q)
}
case <- function(tag, f, grid, df = 0) {
  emit(paste0(tag, "_BETA"), coef(f)); emit(paste0(tag, "_VCOV"), vcov(f))
  emit(paste0(tag, "_GRID"), grid[[1]])
  set.seed(1); p <- do.call(Predict, c(list(f), grid, list(conf.type = "simultaneous")))
  p1 <- do.call(Predict, c(list(f), grid))
  z <- if (df == 0) qnorm(0.975) else qt(0.975, df)
  se <- (p1$upper - p1$yhat) / z
  emit(paste0(tag, "_PRED"), c(p$yhat, p$lower, p$upper))
  X <- cbind(1, predict(f, as.data.frame(c(grid, lapply(dd$limits["Adjust to", setdiff(names(dd$limits), names(grid))], function(v) v))), type = "x"))
  R <- cov2cor(X %*% vcov(f) %*% t(X))
  emit(paste0(tag, "_CRIT"), c(mean((p$upper - p$yhat) / se), tight(R, df)))
}
case("LRM", lrm(yb ~ rcs(age, 4) + sex), list(age = seq(30, 70, length = 9)))
case("OLS", ols(yc ~ age + sex + w), list(w = seq(45, 95, length = 6)), df = n - 4)
case("CAT", lrm(yb ~ rcs(age, 4) + sex), list(sex = c(0, 1)))
