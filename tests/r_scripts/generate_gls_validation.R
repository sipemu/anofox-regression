#!/usr/bin/env Rscript
# Validation data for `solvers::gls` (#111, RMS Ch 7) against nlme::gls:
# correlation structures (corCompSymm, corAR1, corCAR1, corExp / corGaus /
# corLin / corSpher with and without nugget, corSymm), variance functions
# (varIdent, varPower, varExp), REML and ML, approximate intervals of the
# variance-covariance parameters, normalised residuals and the variogram.
#
#   Rscript tests/r_scripts/generate_gls_validation.R > tests/fixtures/gls_validation.rs

suppressMessages(library(nlme))
emit <- function(name, v) {
  v <- as.numeric(v)
  cat("#[rustfmt::skip]\n")
  cat(sprintf("pub const %s: [f64; %d] = [\n    ", name, length(v)))
  cat(paste(sprintf("%.17e", v), collapse = ", "))
  cat("\n];\n")
}

set.seed(20261030)
n_id <- 40
d <- do.call(rbind, lapply(seq_len(n_id), function(i) {
  visits <- sort(sample(1:5, sample(3:5, 1)))
  if (!(1 %in% visits)) visits <- c(1, visits[-1])
  data.frame(id = i, visit = visits, time = visits + runif(length(visits), -0.3, 0.3),
             trt = i %% 2)
}))
n <- nrow(d)
d$x <- rnorm(n)
e <- unlist(lapply(split(seq_len(n), d$id), function(ix) {
  t <- d$time[ix]
  R <- 0.85^abs(outer(t, t, "-"))
  drop(t(chol(R)) %*% rnorm(length(ix)))
}))
d$y <- 1 + 0.5 * d$time + 0.8 * d$trt - 0.3 * d$time * d$trt + 0.4 * d$x +
  e * (1 + 0.15 * d$visit) + 0.6 * rnorm(n)
d$ax <- abs(d$x) + 0.5
f0 <- y ~ time * trt + x

emit("ID", d$id); emit("VISIT", d$visit); emit("TIME", d$time)
emit("X", t(model.matrix(f0, d)[, -1])); emit("Y", d$y); emit("AX", d$ax)

out <- function(tag, fit) {
  emit(paste0(tag, "_COEF"), coef(fit))
  emit(paste0(tag, "_VCOV"), vcov(fit))
  emit(paste0(tag, "_SIGMA"), fit$sigma)
  emit(paste0(tag, "_LOGLIK"), c(logLik(fit), AIC(fit), BIC(fit)))
  cs <- fit$modelStruct$corStruct
  if (!is.null(cs)) emit(paste0(tag, "_COR"), coef(cs, unconstrained = FALSE))
  vs <- fit$modelStruct$varStruct
  if (!is.null(vs)) emit(paste0(tag, "_VAR"), coef(vs, unconstrained = FALSE))
}
iv <- function(tag, fit) {
  ci <- intervals(fit, which = "var-cov")
  parts <- list()
  if (!is.null(ci$corStruct)) parts <- c(parts, list(t(ci$corStruct)))
  if (!is.null(ci$varStruct)) parts <- c(parts, list(t(ci$varStruct)))
  parts <- c(parts, list(ci$sigma))
  emit(paste0(tag, "_INTERVALS"), unlist(parts))  # (lower, est, upper) per parameter
}

g1 <- gls(f0, d, correlation = corCAR1(form = ~ time | id))
out("CAR1", g1); iv("CAR1", g1)
emit("CAR1_NORMALIZED", resid(g1, type = "normalized"))
emit("CAR1_PEARSON", resid(g1, type = "pearson"))
v1 <- Variogram(g1, form = ~ time | id, resType = "normalized")
emit("CAR1_VARIOGRAM", t(as.matrix(v1[, c("variog", "dist", "n.pairs")])))
v2 <- Variogram(g1, form = ~ visit | id, resType = "pearson")
emit("CAR1_VARIOGRAM_VISIT", t(as.matrix(v2[, c("variog", "dist", "n.pairs")])))
nd <- data.frame(time = c(1, 3, 5), trt = c(0, 1, 1), x = c(0, 0.5, -1))
emit("CAR1_PREDICT", predict(g1, nd))

g2 <- gls(f0, d, correlation = corAR1(form = ~ visit | id), method = "ML")
out("AR1_ML", g2); iv("AR1_ML", g2)

g3 <- gls(f0, d, correlation = corCompSymm(form = ~ 1 | id),
          weights = varIdent(form = ~ 1 | visit))
out("CS_IDENT", g3); iv("CS_IDENT", g3)

g4 <- gls(f0, d, correlation = corExp(form = ~ time | id, nugget = TRUE))
out("EXP_NUG", g4); iv("EXP_NUG", g4)
g5 <- gls(f0, d, correlation = corGaus(form = ~ time | id))
out("GAUS", g5)
# from its default start (1.1 minD) nlme stops at a local optimum
# (logLik -271.58); from range 4 it reaches the maximum (-268.53)
g6 <- gls(f0, d, correlation = corLin(4, form = ~ time | id))
out("LIN", g6)
g7 <- gls(f0, d, correlation = corSpher(form = ~ time | id))
out("SPHER", g7)

g8 <- gls(f0, d, correlation = corSymm(form = ~ visit | id),
          weights = varIdent(form = ~ 1 | visit))
out("SYMM", g8); iv("SYMM", g8)

g9 <- gls(f0, d, weights = varPower(form = ~ ax))
out("POWER", g9); iv("POWER", g9)
g10 <- gls(f0, d, weights = varExp(form = ~ x), method = "ML")
out("VEXP_ML", g10)
g11 <- gls(f0, d)
out("INDEP", g11); iv("INDEP", g11)
