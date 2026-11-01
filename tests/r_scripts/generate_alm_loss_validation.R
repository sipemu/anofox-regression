#!/usr/bin/env Rscript
# Reference values for tests/r_validation_alm_loss.rs: greybox::alm with the
# likelihood, MSE, MAE, HAM and ROLE losses on clean data and on data with
# two outliers, at tight nloptr tolerances (greybox's defaults stop short).
#
#   Rscript tests/r_scripts/generate_alm_loss_validation.R
suppressMessages(library(greybox))
fmt <- function(v) { s <- sprintf("%.17g", v); paste(ifelse(grepl("[.e]|Inf|NaN", s), s, paste0(s, ".0")), collapse = ", ") }
vec <- function(tag, v) cat(sprintf("const %s: [f64; %d] = [%s];\n", tag, length(v), fmt(unname(as.numeric(v)))))
tight <- function(...) alm(..., distribution = "dnorm", xtol_rel = 1e-14, ftol_rel = 1e-14, maxeval = 1e5)
set.seed(42)
n <- 50
x <- seq(1, 50, length.out = n)
yc <- 2.5 + 1.5 * x + rnorm(n, sd = 3)
yo <- yc; yo[10] <- yo[10] + 80; yo[40] <- yo[40] - 80
vec("X", x); vec("YC", yc); vec("YO", yo)
# intercept, slope, scale
out <- function(tag, f) vec(tag, c(coef(f), f$scale))
out("LIKELIHOOD", tight(yc ~ x))
out("MSE", tight(yc ~ x, loss = "MSE"))
out("MSE_OUTLIERS", tight(yo ~ x, loss = "MSE"))
out("MAE", tight(yc ~ x, loss = "MAE"))
out("MAE_OUTLIERS", tight(yo ~ x, loss = "MAE"))
out("HAM", tight(yc ~ x, loss = "HAM"))
out("HAM_OUTLIERS", tight(yo ~ x, loss = "HAM"))
out("ROLE05_OUTLIERS", tight(yo ~ x, loss = "ROLE"))
out("ROLE10_OUTLIERS", tight(yo ~ x, loss = "ROLE", trim = 0.10))
