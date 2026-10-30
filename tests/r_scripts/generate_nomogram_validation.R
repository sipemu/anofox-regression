#!/usr/bin/env Rscript
# Validation data for `effects::ModelSpec::nomogram` (#116 follow-up in
# #135) against rms::nomogram: per-predictor ticks, terms and points, the
# scale, the total-points, linear-predictor and risk axes.
#
#   Rscript tests/r_scripts/generate_nomogram_validation.R > tests/fixtures/nomogram_validation.rs

suppressMessages(library(rms))
emit <- function(name, v) {
  v <- as.numeric(v)
  cat("#[rustfmt::skip]\n")
  cat(sprintf("pub const %s: [f64; %d] = [\n    ", name, length(v)))
  cat(paste(sprintf("%.17e", v), collapse = ", "))
  cat("\n];\n")
}

set.seed(20261109)
n <- 250
age <- round(rnorm(n, 50, 10), 1); sexn <- rbinom(n, 1, 0.45); chol <- round(rnorm(n, 200, 30))
grade <- sample(1:4, n, TRUE)
y <- rbinom(n, 1, plogis(-6 + 0.06 * age + 0.002 * (age - 50)^2 + 0.6 * sexn + 0.01 * chol + 0.3 * grade))
sex <- factor(sexn, labels = c("0", "1"))
emit("AGE", age); emit("SEX", sexn); emit("CHOL", chol); emit("GRADE", grade)
dd <- datadist(age, sex, chol, grade); options(datadist = "dd")
f <- lrm(y ~ rcs(age, 4) + sex + chol + grade)
emit("BETA", coef(f)); emit("LP", f$linear.predictors)
nom <- nomogram(f, fun = plogis, funlabel = "Risk")
for (v in c("age", "sex", "chol", "grade")) {
  a <- nom[[v]]
  emit(sprintf("%s_VALUES", toupper(v)), as.numeric(as.character(a[[v]])))
  emit(sprintf("%s_XBETA", toupper(v)), a$Xbeta); emit(sprintf("%s_POINTS", toupper(v)), a$points)
}
info <- attr(nom, "info")
emit("SCALE_INTERCEPT", c(info$sc, info$Intercept))
emit("TOTAL_POINTS", nom$total.points$x)
emit("LP_AT", nom$lp$x.real); emit("LP_POINTS", nom$lp$x)
emit("RISK_AT", nom$Risk$x.real); emit("RISK_POINTS", nom$Risk$x)
