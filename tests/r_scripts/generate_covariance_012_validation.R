#!/usr/bin/env Rscript
# Validation data for the 0.12 covariance options (tests/covariance_012.rs):
# HC covariance matrices of OLS / WLS (sandwich::vcovHC), the cluster-robust
# covariance of the GLM and AFT entry points with and without priors
# (rms::robcov on a penalized lrm, and the penalized sandwich of
# a Gamma GLM and a Weibull AFT computed directly).
#
# The data are those of generate_cluster_validation.R (same seed, same draws);
# the Rust test includes both fixtures.
#
#   Rscript tests/r_scripts/generate_covariance_012_validation.R > tests/fixtures/covariance_012_validation.rs

suppressMessages({ library(sandwich); library(rms); library(survival); library(numDeriv) })

emit <- function(name, v) {
  cat("#[rustfmt::skip]\n")
  cat(sprintf("const %s: [f64; %d] = [\n    ", name, length(v)))
  cat(paste(sprintf("%.17e", as.numeric(v)), collapse = ", "))
  cat("\n];\n")
}
vc <- function(name, m) emit(name, t(m))

# --- the data of generate_cluster_validation.R ------------------------------
set.seed(20261028)
n <- 200
g1 <- sample(1:25, n, replace = TRUE)
g2 <- sample(1:8, n, replace = TRUE)
u1 <- rnorm(25)[g1]; u2 <- rnorm(8)[g2]
x1 <- rnorm(n) + 0.5 * u1; x2 <- runif(n)
y <- 1 + 0.5 * x1 - x2 + u1 + 0.5 * u2 + rnorm(n)
w <- runif(n, 0.5, 2)
yc <- rpois(n, exp(0.3 + 0.4 * x1 - 0.5 * x2 + 0.3 * u1))
yb <- rbinom(n, 1, plogis(-0.2 + x1 - x2 + 0.5 * u1))
yg <- rgamma(n, shape = 3, rate = 3 / exp(0.5 + 0.3 * x1 + 0.2 * u1))
tt <- rweibull(n, shape = 1.5, scale = exp(1 + 0.3 * x1 - 0.4 * x2 + 0.3 * u1))
cens <- runif(n, 0, 8)
time <- pmin(tt, cens); ev <- as.integer(tt <= cens)
X <- cbind(1, x1, x2)

# --- HC covariance matrices (vcovHC) ----------------------------------------
f <- lm(y ~ x1 + x2)
fw <- lm(y ~ x1 + x2, weights = w)
for (t in c("HC0", "HC1", "HC2", "HC3")) {
  vc(sprintf("HC_OLS_%s", t), vcovHC(f, type = t))
  vc(sprintf("HC_WLS_%s", t), vcovHC(fw, type = t))
}

# --- penalized logistic: lrm(penalty.matrix) + robcov --------------------------
# Normal(0, s) priors on the slopes are the penalty ½ βᵀ diag(1/s²) β (lrm
# uses penalty.matrix only with penalty > 0).
s_logit <- c(0.5, 0.3)
fl <- lrm(yb ~ x1 + x2, penalty = 1, penalty.matrix = diag(1 / s_logit^2), x = TRUE, y = TRUE, tol = 1e-14)
emit("PEN_LOGIT_COEF", coef(fl))
vc("PEN_LOGIT_VCOV", vcov(fl))
vc("PEN_LOGIT_ROBCOV", vcov(robcov(fl, cluster = g1)))

# --- penalized Gamma (log link): the IRLS objective deviance/2 + ½ βᵀPβ -------
# Fisher weights are 1 for the Gamma log link, the unit scores (y − μ)/μ x.
s_gamma <- c(0.2, 0.1)
P <- diag(c(0, 1 / s_gamma^2))
obj <- function(b) { mu <- exp(drop(X %*% b)); sum((yg - mu) / mu - log(yg / mu)) + 0.5 * sum(b * (P %*% b)) }
grad <- function(b) { mu <- exp(drop(X %*% b)); -drop(t(X) %*% ((yg - mu) / mu)) + drop(P %*% b) }
b <- coef(glm(yg ~ x1 + x2, family = Gamma(link = "log")))
for (it in 1:50) b <- b - solve(crossprod(X) + P, grad(b))
mu <- exp(drop(X %*% b))
bread <- solve(crossprod(X) + P)
U <- X * ((yg - mu) / mu)
Ug <- rowsum(U, g1)
G <- nrow(Ug)
emit("PEN_GAMMA_COEF", b)
vc("PEN_GAMMA_HC0", bread %*% (G / (G - 1) * crossprod(Ug)) %*% bread)

# --- AFT: a penalized Weibull -------------------------------------------------
# (The unpenalized reference is survreg(cluster =) in cluster_validation.)
fp <- survreg(Surv(time, ev) ~ x1 + x2, dist = "weibull")

# Weibull log-likelihood in survreg's parametrisation (β, log σ) with
# Normal(0, s) priors on the slopes.
s_aft <- c(0.1, 0.2)
Pa <- diag(c(0, 1 / s_aft^2, 0))
scores <- function(th) {
  sig <- exp(th[4]); z <- (log(time) - drop(X %*% th[1:3])) / sig; e <- exp(z)
  d_eta <- ifelse(ev == 1, (e - 1) / sig, e / sig)
  d_ls <- ifelse(ev == 1, -1 - z * (1 - e), z * e)
  cbind(X * d_eta, d_ls)
}
pen_grad <- function(th) colSums(scores(th)) - drop(Pa %*% th)
th <- c(coef(fp), log(fp$scale))
for (it in 1:50) {
  H <- jacobian(pen_grad, th)
  th <- th - solve(H, pen_grad(th))
}
info <- -jacobian(pen_grad, th)
info <- (info + t(info)) / 2
A <- solve(info)
Sg <- rowsum(scores(th), g1)
emit("PEN_AFT_COEF", th)
vc("PEN_AFT_ROBUST", A %*% crossprod(Sg) %*% A)
