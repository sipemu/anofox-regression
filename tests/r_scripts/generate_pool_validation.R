#!/usr/bin/env Rscript
# Validation data for `inference::pool` (#106, RMS §3.8) against
# Hmisc::fit.mult.impute (Rubin's rules: pooled coefficients and
# covariance, variance inflation, fraction of missing information, d.f.).
# The Barnard-Rubin d.f. and the D1 pooled Wald test (Li, Raghunathan and
# Rubin 1991) follow mice::pool / mice::D1, written out below because mice
# is not a dependency of this validation.
#
#   Rscript tests/r_scripts/generate_pool_validation.R > tests/fixtures/pool_validation.rs

suppressMessages({library(Hmisc); library(rms)})
emit <- function(name, v) {
  v <- as.numeric(v)
  cat("#[rustfmt::skip]\n")
  cat(sprintf("pub const %s: [f64; %d] = [\n    ", name, length(v)))
  cat(paste(sprintf("%.17e", v), collapse = ", "))
  cat("\n];\n")
}

set.seed(20261104)
n <- 150
x1 <- rnorm(n); x2 <- 0.5 * x1 + rnorm(n); x3 <- rnorm(n)
y <- 1 + 0.8 * x1 - 0.5 * x2 + 0.3 * x3 + rnorm(n)
x1[sample(n, 30)] <- NA; x2[sample(n, 20)] <- NA
d <- data.frame(y, x1, x2, x3)
m <- 6
imp <- aregImpute(~ y + x1 + x2 + x3, data = d, n.impute = m, pr = FALSE)
f <- fit.mult.impute(y ~ x1 + x2 + x3, ols, imp, data = d, fit.reps = TRUE, pr = FALSE)

emit("M", m)
emit("COEFS", sapply(f$fits, coef))                       # p x m, column per imputation
emit("VCOVS", unlist(lapply(f$fits, function(g) as.vector(vcov(g)))))
emit("POOL_COEF", coef(f)); emit("POOL_VCOV", vcov(f))
emit("POOL_R", f$variance.inflation.impute)
emit("POOL_FMI", f$missingInfo); emit("POOL_DF", f$dfmi)

# mice::pool: Barnard-Rubin d.f. with complete-data d.f. n - p, and fmi
Q <- sapply(f$fits, coef); U <- lapply(f$fits, vcov)
qbar <- rowMeans(Q); ubar <- Reduce(`+`, U) / m
B <- cov(t(Q)); t_ <- diag(ubar) + (1 + 1/m) * diag(B)
riv <- (1 + 1/m) * diag(B) / diag(ubar); lambda <- (1 + 1/m) * diag(B) / t_
dfcom <- n - 4
dfold <- (m - 1) / lambda^2
dfobs <- (dfcom + 1) / (dfcom + 3) * dfcom * (1 - lambda)
df <- dfold * dfobs / (dfold + dfobs)
fmi <- (riv + 2 / (df + 3)) / (1 + riv)
emit("MICE_DF", df); emit("MICE_FMI", fmi); emit("MICE_LAMBDA", lambda); emit("MICE_RIV", riv)

# D1 for the slopes of x1 and x2 (and for all three slopes)
d1 <- function(idx) {
  k <- length(idx)
  th <- qbar[idx]; W <- ubar[idx, idx]; Bk <- B[idx, idx]
  r1 <- (1 + 1/m) * sum(diag(Bk %*% solve(W))) / k
  Tt <- (1 + r1) * W
  F <- drop(t(th) %*% solve(Tt) %*% th) / k
  nu1 <- k * (m - 1)
  df2 <- if (nu1 > 4) 4 + (nu1 - 4) * (1 + (1 - 2 / nu1) / r1)^2 else nu1 * (1 + 1/k) * (1 + 1/r1)^2 / 2
  c(F, k, df2, pf(F, k, df2, lower.tail = FALSE), r1)
}
emit("D1_X1X2", d1(2:3)); emit("D1_ALL", d1(2:4))
# #210: mice::D1 passes the model's complete-data d.f. (get.dfcom) to
# mitml::testModels, whose .D1 then uses Reiter's (2007) d.f. (code of
# mitml:::.D1, written out as mitml is not required)
d1_reiter <- function(idx, df.com) {
  k <- length(idx)
  W <- ubar[idx, idx]; Bk <- B[idx, idx]
  r <- (1 + 1/m) * sum(diag(Bk %*% solve(W))) / k
  F <- drop(t(qbar[idx]) %*% solve((1 + r) * W) %*% qbar[idx]) / k
  t <- k * (m - 1)
  a <- r * t / (t - 2)
  vstar <- ((df.com + 1) / (df.com + 3)) * df.com
  c0 <- 1 / (t - 4); c1 <- vstar - 2 * (1 + a); c2 <- vstar - 4 * (1 + a)
  z <- 1 / c2 + c0 * (a^2 * c1 / ((1 + a)^2 * c2)) +
    c0 * (8 * a^2 * c1 / ((1 + a) * c2^2) + 4 * a^2 / ((1 + a) * c2)) +
    c0 * (4 * a^2 / (c2 * c1) + 16 * a^2 * c1 / c2^3) + c0 * (8 * a^2 / c2^2)
  v <- 4 + 1 / z
  c(F, k, v, pf(F, k, v, lower.tail = FALSE), r)
}
emit("D1_REITER_X1X2", d1_reiter(2:3, dfcom)); emit("D1_REITER_ALL", d1_reiter(2:4, dfcom))
