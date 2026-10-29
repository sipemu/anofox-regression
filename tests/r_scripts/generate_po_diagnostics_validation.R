#!/usr/bin/env Rscript
# Validation data for the proportional-odds diagnostics of `solvers::ordinal`
# (#110, RMS §13.3): the score test of proportional odds against the partial
# proportional-odds alternative (score and information by numDeriv), and the
# numbers of rms::plot.xmean.ordinaly.
#
#   Rscript tests/r_scripts/generate_po_diagnostics_validation.R > tests/fixtures/po_diagnostics_validation.rs

suppressMessages({ library(rms); library(numDeriv) })
emit <- function(name, v) {
  v <- as.numeric(v)
  cat("#[rustfmt::skip]\n")
  cat(sprintf("pub const %s: [f64; %d] = [\n    ", name, length(v)))
  cat(paste(sprintf("%.17e", v), collapse = ", "))
  cat("\n];\n")
}

# The 5-category data of generate_ordinal_validation.R
set.seed(20261030)
n <- 150
x1 <- rnorm(n); x2 <- rbinom(n, 1, 0.4); x3 <- runif(n)
lin <- 0.8 * x1 - 0.6 * x2 + 1.2 * x3
y_cont <- round(lin + rlogis(n), 6)
y_cat <- cut(lin + rlogis(n), c(-Inf, -1, 0, 0.8, 1.8, Inf), labels = FALSE)
X <- cbind(x1, x2, x3)
d <- data.frame(x1, x2, x3, y_cat)
emit("X", t(X)); emit("Y_CAT", y_cat)
k <- 5; m <- k - 1

# Partial proportional odds: tested columns T have a slope per cut-point.
# par = (alpha_1..m, beta_T for cut-point 1, ..., for cut-point m, beta_U)
score_test <- function(family, T) {
  f <- orm(y_cat ~ x1 + x2 + x3, data = d, family = family, eps = 1e-10, gradtol = 1e-9, maxit = 100)
  cdf <- switch(family, logistic = plogis, probit = pnorm)
  a <- coef(f)[1:m]; b <- coef(f)[-(1:m)]
  U <- setdiff(1:3, T); s <- length(T)
  ll <- function(par) {
    al <- par[1:m]
    bt <- matrix(par[m + seq_len(m * s)], nrow = s)          # column j = cut-point j
    bu <- par[m + m * s + seq_along(U)]
    eta_u <- if (length(U)) X[, U, drop = FALSE] %*% bu else 0
    cum <- sapply(1:m, function(j) cdf(al[j] + X[, T, drop = FALSE] %*% bt[, j] + eta_u))
    cum <- cbind(1, cum, 0)
    sum(log(cum[cbind(1:n, y_cat)] - cum[cbind(1:n, y_cat + 1)]))
  }
  par0 <- c(a, rep(b[T], m), b[U])
  g <- grad(ll, par0); H <- hessian(ll, par0)
  as.numeric(t(g) %*% solve(-H, g))
}
emit("SCORE_ALL", score_test("logistic", 1:3))
emit("SCORE_X1", score_test("logistic", 1))
emit("SCORE_X3", score_test("logistic", 3))
emit("SCORE_PROBIT_X1_X2", score_test("probit", c(1, 2)))

# plot.xmean.ordinaly(y ~ x, cr = TRUE)'s dopl(), converged tightly
xmean <- function(x, y) {
  f <- lrm.fit(x, y, gradtol = 1e-12, maxit = 100)
  fy <- f$freq / n; ns <- length(fy) - 1; kk <- ns + 1
  intcept <- f$coef[1:ns]
  xb <- f$linear.predictors - intcept[1]
  P <- matrix(plogis(sapply(intcept, "+", xb)), ncol = ns)
  P <- cbind(1, P) - cbind(P, 0)
  po <- apply(x * P / n, 2, sum) / fy
  u <- cr.setup(y); xc <- x[u$subs]
  xcohort <- matrix(0, nrow = length(xc), ncol = length(levels(u$cohort)) - 1)
  xcohort[col(xcohort) == unclass(u$cohort) - 1] <- 1
  cof <- lrm.fit(cbind(xcohort, xc), u$y, gradtol = 1e-12, maxit = 100)$coefficients
  cumprob <- rep(1, n)
  for (j in 1:kk) {
    P[, j] <- cumprob * (if (j == kk) 1 else plogis(cof[1] + (if (j > 1) cof[j] else 0) + cof[kk] * x))
    cumprob <- cumprob - P[, j]
  }
  cr <- apply(x * P / n, 2, sum) / fy
  c(tapply(x, y, mean), po, cr)
}
emit("XMEAN_X1", xmean(x1, y_cat))
emit("XMEAN_X3", xmean(x3, y_cat))
