#!/usr/bin/env Rscript
# Validation data for `transform` (#112, RMS Ch 16) against MASS::boxcox
# (profile log-likelihood of the Box-Cox lambda) and Hmisc::smearingEst
# (Duan's smearing estimator of the mean / quantiles on the original scale).
#
#   Rscript tests/r_scripts/generate_transform_validation.R > tests/fixtures/transform_validation.rs

suppressMessages({library(MASS); library(Hmisc)})
emit <- function(name, v) {
  v <- as.numeric(v)
  cat("#[rustfmt::skip]\n")
  cat(sprintf("pub const %s: [f64; %d] = [\n    ", name, length(v)))
  cat(paste(sprintf("%.17e", v), collapse = ", "))
  cat("\n];\n")
}

set.seed(20261031)
n <- 120
x1 <- rnorm(n); x2 <- runif(n)
y <- (2 + 0.6 * x1 + 1.2 * x2 + rnorm(n, sd = 0.5))^2   # sqrt(y) is linear
emit("X", t(cbind(x1, x2))); emit("Y", y)

f <- lm(y ~ x1 + x2)
grid <- seq(-2, 2, 1/10)
bc <- boxcox(f, lambda = grid, plotit = FALSE)
emit("BC_GRID_LOGLIK", bc$y)
# the same profile as a function, maximised and cut at the 95% chi-square limit
prof <- function(la) boxcox(f, lambda = la, plotit = FALSE)$y
opt <- optimize(prof, c(-2, 2), maximum = TRUE, tol = 1e-10)
lim <- opt$objective - qchisq(0.95, 1) / 2
lo <- uniroot(function(l) prof(l) - lim, c(-2, opt$maximum), tol = 1e-12)$root
hi <- uniroot(function(l) prof(l) - lim, c(opt$maximum, 2), tol = 1e-12)$root
emit("BC_MLE", c(opt$maximum, opt$objective, lo, hi))
# small |lambda| uses the series for log(y)
emit("BC_SMALL", prof(c(-0.015, 0, 0.01)))
# a model without an intercept
f0 <- lm(y ~ 0 + x1 + x2 + I(rep(1, n)))
emit("BC_NOINT_CHECK", boxcox(f0, lambda = c(0.3, 0.5), plotit = FALSE)$y)

# smearing on the sqrt scale: transform y^0.5, back-transform z^2
g <- lm(sqrt(y) ~ x1 + x2)
lp <- fitted(g)[1:6]; r <- resid(g)
inv <- function(z) z^2
emit("SM_LP", lp)
emit("SM_MEAN", smearingEst(lp, inv, r, "mean"))
emit("SM_MEDIAN", smearingEst(lp, inv, r, "median"))
emit("SM_Q90", smearingEst(lp, inv, r, "quantile", q = 0.9))
emit("SM_FITTED", smearingEst(lp, inv, r, "fitted"))
# log scale: exp back-transform
h <- lm(log(y) ~ x1 + x2)
emit("SM_LOG_MEAN", smearingEst(fitted(h)[1:6], exp, resid(h), "mean"))
# Box-Cox lambda = 0.3 with its inverse
lam <- 0.3
k <- lm(((y^lam - 1) / lam) ~ x1 + x2)
emit("SM_BC_MEAN", smearingEst(fitted(k)[1:6], function(z) (lam * z + 1)^(1 / lam), resid(k), "mean"))
