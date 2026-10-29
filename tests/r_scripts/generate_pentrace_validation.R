#!/usr/bin/env Rscript
# Validation data for `solvers::pentrace` (#109, RMS §9.10) against rms:
# Penalty.matrix / Penalty.setup, lrm(penalty = list(...)) with both
# var.penalty choices, effective.df.diagonal and pentrace.
#
#   Rscript tests/r_scripts/generate_pentrace_validation.R > tests/fixtures/pentrace_validation.rs

suppressMessages(library(rms))
emit <- function(name, v) {
  v <- as.numeric(v)
  cat("#[rustfmt::skip]\n")
  cat(sprintf("pub const %s: [f64; %d] = [\n    ", name, length(v)))
  cat(paste(sprintf("%.17e", v), collapse = ", "))
  cat("\n];\n")
}

set.seed(20261031)
n <- 300
x1 <- rnorm(n); x2 <- rnorm(n); g <- factor(sample(c("a", "b", "c"), n, replace = TRUE))
y <- rbinom(n, 1, plogis(-0.2 + 0.7 * x1 - 0.3 * x1^2 + 0.5 * x2 + 0.3 * x1 * x2 + c(0, 0.4, -0.3)[g]))
d <- data.frame(x1, x2, g, y)
# columns: x1 x1' x1'' x2 g=b g=c x1*x2 x1'*x2 x1''*x2
f <- lrm(y ~ rcs(x1, 4) * x2 + g, data = d, x = TRUE, y = TRUE)
emit("X", t(f$x)); emit("Y", y)
emit("BASE_PENALTY", rms:::Penalty.matrix(f$Design, f$x))

pen <- list(simple = 1, nonlinear = 4, interaction = 2, nonlinear.interaction = 8)
emit("MULTIPLIERS", rms:::Penalty.setup(f$Design, pen)$multiplier)
fp <- lrm(y ~ rcs(x1, 4) * x2 + g, data = d, penalty = pen)
emit("PEN_COEF", coef(fp))
emit("PEN_VAR", as.matrix(vcov(fp)))          # penalized inverse information
emit("PEN_EFFECTIVE_DF", fp$effective.df.diagonal[-1])
emit("PEN_LR_DF", fp$stats[c("Model L.R.", "d.f.")])
# lrm's var.penalty is deprecated; pentrace's var.adj is the sandwich
# V I_unpenalized V of its best penalty (here the only one)
emit("PEN_SANDWICH", as.matrix(pentrace(f, pen)$var.adj))

pt <- pentrace(f, list(simple = c(0, 2), nonlinear = c(0, 4, 16)))
r <- pt$results.all
emit("TRACE_PENALTY", t(as.matrix(r[, c("simple", "nonlinear")])))
emit("TRACE_STATS", t(as.matrix(r[, c("df", "aic", "bic", "aic.c")])))
emit("TRACE_BEST", unlist(pt$penalty))
pt2 <- pentrace(f, c(0, 0.5, 1, 2, 4, 8))
emit("UNIFORM_STATS", t(as.matrix(pt2$results.all[, c("penalty", "df", "aic", "bic", "aic.c")])))
emit("UNIFORM_BEST", pt2$penalty)

# Least squares: ols(penalty = ) (lm.pfit) and pentrace on an ols fit
set.seed(20261101)
y_c <- -0.2 + 0.7 * x1 - 0.3 * x1^2 + 0.5 * x2 + 0.3 * x1 * x2 + c(0, 0.4, -0.3)[g] + rnorm(n)
d$y_c <- y_c
emit("Y_C", y_c)
fo <- ols(y_c ~ rcs(x1, 4) * x2 + g, data = d, penalty = pen, x = TRUE)
emit("OLS_COEF", coef(fo)); emit("OLS_VAR", as.matrix(vcov(fo)))
emit("OLS_STATS", fo$stats[c("Model L.R.", "d.f.", "R2", "Sigma")])
emit("OLS_EFFECTIVE_DF", fo$effective.df.diagonal)
emit("OLS_SANDWICH", lm.pfit(fo$x, y_c, penalty.matrix = fo$penalty.matrix, var.penalty = "sandwich")$var)
fo0 <- ols(y_c ~ rcs(x1, 4) * x2 + g, data = d, x = TRUE, y = TRUE)
pto <- pentrace(fo0, list(simple = c(0, 2), nonlinear = c(0, 4, 16)))
emit("OLS_TRACE_STATS", t(as.matrix(pto$results.all[, c("df", "aic", "bic", "aic.c")])))
emit("OLS_TRACE_BEST", unlist(pto$penalty))
