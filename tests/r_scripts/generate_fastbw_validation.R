#!/usr/bin/env Rscript
# Validation data for `inference::fastbw` (#108) against rms::fastbw.
#
#   Rscript tests/r_scripts/generate_fastbw_validation.R > tests/fixtures/fastbw_validation.rs

suppressMessages(library(rms))
emit <- function(name, v) {
  v <- as.numeric(v)
  cat("#[rustfmt::skip]\n")
  cat(sprintf("const %s: [f64; %d] = [\n    ", name, length(v)))
  cat(paste(ifelse(is.na(v), "f64::NAN", sprintf("%.17e", v)), collapse = ", "))
  cat("\n];\n")
}
emit_usize <- function(name, v) cat(sprintf("const %s: [usize; %d] = [%s];\n", name, length(v), paste(v, collapse = ", ")))

set.seed(20261028)
n <- 300
x1 <- rnorm(n); x2 <- rnorm(n); x3 <- rnorm(n); x4 <- rnorm(n); x5 <- rnorm(n)
g <- sample(1:3, n, replace = TRUE)
y <- rbinom(n, 1, plogis(-0.3 + 0.8 * x1 + 0.3 * x1^2 - 0.6 * x2 + 0.15 * x3 + c(0, 0.4, 0.1)[g]))
d <- data.frame(x1, x2, x3, x4, x5, g = factor(g), y)
f <- lrm(y ~ rcs(x1, 4) + x2 + x3 + x4 + x5 + g, data = d)
emit("COEF", coef(f)); emit("VCOV", t(vcov(f)))
for (rule in c("aic", "p")) {
  for (type in c("residual", "individual")) {
    b <- fastbw(f, rule = rule, type = type)
    id <- toupper(paste(rule, type, sep = "_"))
    emit_usize(sprintf("%s_DELETED", id), b$factors.deleted - 1)
    emit(sprintf("%s_RESULT", id), if (is.null(b$result)) numeric(0) else t(b$result))
    emit(sprintf("%s_COEF", id), b$Coefficients[length(b$factors.deleted), ])
  }
}
# #210: non-default sls / aics / k.aic, and an ols fit
emit_case <- function(id, b) {
  emit_usize(sprintf("%s_DELETED", id), b$factors.deleted - 1)
  emit(sprintf("%s_RESULT", id), if (is.null(b$result)) numeric(0) else t(b$result))
  emit(sprintf("%s_COEF", id), b$Coefficients[length(b$factors.deleted), ])
}
emit_case("P20_RESIDUAL", fastbw(f, rule = "p", sls = 0.2))
emit_case("AICS2_LOGN_INDIVIDUAL", fastbw(f, rule = "aic", type = "individual", aics = 2, k.aic = log(n)))
yc <- -0.3 + 0.8 * x1 + 0.3 * x1^2 - 0.6 * x2 + 0.15 * x3 + c(0, 0.4, 0.1)[g] + rnorm(n)
d$yc <- yc
fo <- ols(yc ~ rcs(x1, 4) + x2 + x3 + x4 + x5 + g, data = d)
emit("OLS_COEF", coef(fo)); emit("OLS_VCOV", t(vcov(fo)))
emit_case("OLS_AIC_RESIDUAL", fastbw(fo))
emit_case("OLS_P_INDIVIDUAL", fastbw(fo, rule = "p", type = "individual", sls = 0.1))
