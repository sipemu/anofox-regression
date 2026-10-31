#!/usr/bin/env Rscript
# #210: nlme::gls fits not covered by generate_gls_validation.R, on the same
# data (the data block below is copied from that script): AR(1) REML,
# continuous AR(1) ML, corExp without nugget, corGaus / corLin / corSpher with
# nugget, varIdent alone, no intercept, and standard errors of predictions.
#
#   Rscript tests/r_scripts/generate_gls_validation_210.R > tests/fixtures/gls_validation_210.rs

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
ctl <- glsControl(tolerance = 1e-12, msTol = 1e-14, maxIter = 1000, msMaxIter = 1000)

out("AR1_REML", gls(f0, d, correlation = corAR1(form = ~ visit | id), control = ctl))
out("CAR1_ML", gls(f0, d, correlation = corCAR1(form = ~ time | id), method = "ML", control = ctl))
out("EXP", gls(f0, d, correlation = corExp(form = ~ time | id), control = ctl))
out("GAUS_NUG", gls(f0, d, correlation = corGaus(form = ~ time | id, nugget = TRUE), control = ctl))
out("LIN_NUG", gls(f0, d, correlation = corLin(c(4, 0.1), form = ~ time | id, nugget = TRUE), control = ctl))
out("SPHER_NUG", gls(f0, d, correlation = corSpher(form = ~ time | id, nugget = TRUE), control = ctl))
out("IDENT", gls(f0, d, weights = varIdent(form = ~ 1 | visit), control = ctl))
g0 <- gls(y ~ time * trt + x - 1, d, correlation = corCAR1(form = ~ time | id), control = ctl)
out("NOINT", g0)

# predictions and their standard errors sqrt(diag(X0 vcov X0')) at new rows
g1 <- gls(f0, d, correlation = corCAR1(form = ~ time | id), control = ctl)
nd <- data.frame(time = c(1, 3, 5), trt = c(0, 1, 1), x = c(0, 0.5, -1))
X0 <- model.matrix(~ time * trt + x, nd)
emit("NEW_X", t(X0[, -1]))
emit("PREDICT_FIT", predict(g1, nd))
emit("PREDICT_SE", sqrt(diag(X0 %*% vcov(g1) %*% t(X0))))
X0n <- model.matrix(~ time * trt + x - 1, nd)
emit("NOINT_PREDICT_FIT", predict(g0, nd))
emit("NOINT_PREDICT_SE", sqrt(diag(X0n %*% vcov(g0) %*% t(X0n))))
