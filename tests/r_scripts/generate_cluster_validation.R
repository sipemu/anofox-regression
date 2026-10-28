#!/usr/bin/env Rscript
# Validation data for `inference::cluster` (#105) against sandwich::vcovCL,
# rms::robcov and survival::survreg(cluster =).
#
#   Rscript tests/r_scripts/generate_cluster_validation.R > tests/fixtures/cluster_validation.rs

suppressMessages({ library(sandwich); library(rms); library(survival) })

emit <- function(name, v) {
  cat("#[rustfmt::skip]\n")
  cat(sprintf("const %s: [f64; %d] = [\n    ", name, length(v)))
  cat(paste(sprintf("%.17e", as.numeric(v)), collapse = ", "))
  cat("\n];\n")
}
emit_usize <- function(name, v) cat(sprintf("const %s: [usize; %d] = [%s];\n", name, length(v), paste(v, collapse = ", ")))
vc <- function(name, m) emit(name, t(m))

set.seed(20261028)
n <- 200
g1 <- sample(1:25, n, replace = TRUE)          # firm
g2 <- sample(1:8, n, replace = TRUE)           # year
u1 <- rnorm(25)[g1]; u2 <- rnorm(8)[g2]
x1 <- rnorm(n) + 0.5 * u1; x2 <- runif(n)
y <- 1 + 0.5 * x1 - x2 + u1 + 0.5 * u2 + rnorm(n)
w <- runif(n, 0.5, 2)
emit("X1", x1); emit("X2", x2); emit("Y", y); emit("W", w)
emit_usize("G1", g1 - 1); emit_usize("G2", g2 - 1)

f <- lm(y ~ x1 + x2)
for (t in c("HC0", "HC1", "HC2", "HC3")) {
  vc(sprintf("LM_%s", t), vcovCL(f, cluster = ~g1, type = t))
  vc(sprintf("LM_%s_NOADJ", t), vcovCL(f, cluster = ~g1, type = t, cadjust = FALSE))
}
vc("LM_TWOWAY_HC1", vcovCL(f, cluster = ~g1 + g2, type = "HC1"))
vc("LM_TWOWAY_HC0", vcovCL(f, cluster = ~g1 + g2, type = "HC0"))
fw <- lm(y ~ x1 + x2, weights = w)
for (t in c("HC0", "HC1", "HC2", "HC3")) vc(sprintf("WLS_%s", t), vcovCL(fw, cluster = ~g1, type = t))
# No intercept.
f0 <- lm(y ~ 0 + x1 + x2)
vc("LM0_HC1", vcovCL(f0, cluster = ~g1, type = "HC1"))

# rms::robcov on ols = HC0 without cluster adjustment.
fo <- ols(y ~ x1 + x2, x = TRUE, y = TRUE)
vc("ROBCOV", vcov(robcov(fo, cluster = g1)))

# GLMs: Poisson (log) and binomial (logit, probit), Gamma (log).
yc <- rpois(n, exp(0.3 + 0.4 * x1 - 0.5 * x2 + 0.3 * u1))
yb <- rbinom(n, 1, plogis(-0.2 + x1 - x2 + 0.5 * u1))
yg <- rgamma(n, shape = 3, rate = 3 / exp(0.5 + 0.3 * x1 + 0.2 * u1))
emit("YC", yc); emit("YB", yb); emit("YG", yg)
tight <- glm.control(epsilon = 1e-14, maxit = 100)
glms <- list(
  POIS = glm(yc ~ x1 + x2, family = poisson, control = tight),
  LOGIT = glm(yb ~ x1 + x2, family = binomial, control = tight),
  PROBIT = glm(yb ~ x1 + x2, family = binomial(link = "probit"), control = tight),
  GAMMA = glm(yg ~ x1 + x2, family = Gamma(link = "log"), control = tight)
)
for (id in names(glms)) {
  g <- glms[[id]]
  emit(sprintf("%s_MU", id), fitted(g))
  vc(sprintf("%s_HC0", id), vcovCL(g, cluster = ~g1, type = "HC0"))
  vc(sprintf("%s_HC1", id), vcovCL(g, cluster = ~g1, type = "HC1"))
}
vc("POIS_TWOWAY", vcovCL(glms$POIS, cluster = ~g1 + g2, type = "HC0"))
fpw <- glm(yc ~ x1 + x2, family = poisson, weights = w, control = tight)
emit("POISW_MU", fitted(fpw))
vc("POISW_HC0", vcovCL(fpw, cluster = ~g1, type = "HC0"))

# AFT (Weibull) with clusters: survreg's robust (infinitesimal jackknife) variance.
tt <- rweibull(n, shape = 1.5, scale = exp(1 + 0.3 * x1 - 0.4 * x2 + 0.3 * u1))
cens <- runif(n, 0, 8)
time <- pmin(tt, cens); ev <- as.integer(tt <= cens)
emit("TIME", time); emit("EV", ev)
fs <- survreg(Surv(time, ev) ~ x1 + x2 + cluster(g1), dist = "weibull")
emit("AFT_COEF", c(coef(fs), log(fs$scale)))
vc("AFT_ROBUST", fs$var)
vc("AFT_NAIVE", fs$naive.var)
