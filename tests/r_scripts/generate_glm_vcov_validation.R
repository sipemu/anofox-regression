#!/usr/bin/env Rscript
# Validation data for the GLM `vcov()` methods (#135) against R's vcov(glm)
# and MASS::glm.nb.
#
#   Rscript tests/r_scripts/generate_glm_vcov_validation.R > tests/fixtures/glm_vcov_validation.rs

suppressMessages({ library(MASS); library(statmod) })
emit <- function(name, v) {
  v <- as.numeric(v)
  cat("#[rustfmt::skip]\n")
  cat(sprintf("const %s: [f64; %d] = [\n    ", name, length(v)))
  cat(paste(sprintf("%.17e", v), collapse = ", "))
  cat("\n];\n")
}

set.seed(20261030)
n <- 80
x1 <- rnorm(n); x2 <- runif(n)
eta <- 0.3 + 0.5 * x1 - 0.8 * x2
y_pois <- rpois(n, exp(eta))
y_bin <- rbinom(n, 1, plogis(eta))
y_gamma <- rgamma(n, shape = 3, rate = 3 / exp(eta))
y_nb <- rnbinom(n, size = 2, mu = exp(eta))
y_tw <- y_gamma * rbinom(n, 1, 0.8)
emit("X", t(cbind(x1, x2)))
emit("Y_POIS", y_pois); emit("Y_BIN", y_bin); emit("Y_GAMMA", y_gamma); emit("Y_NB", y_nb); emit("Y_TW", y_tw)
# converged tightly, so the comparison is not limited by glm's default epsilon
# (glm.nb keeps its default: it alternates theta and beta and does not reach 1e-14)
ctl <- glm.control(epsilon = 1e-14, maxit = 200)
emit("VCOV_POIS", vcov(glm(y_pois ~ x1 + x2, family = poisson, control = ctl)))
emit("VCOV_BIN", vcov(glm(y_bin ~ x1 + x2, family = binomial, control = ctl)))
emit("VCOV_GAMMA", vcov(glm(y_gamma ~ x1 + x2, family = Gamma(link = "log"), control = ctl)))
emit("VCOV_NB", vcov(glm.nb(y_nb ~ x1 + x2)))
emit("VCOV_TW", vcov(glm(y_tw ~ x1 + x2, family = tweedie(var.power = 1.5, link.power = 0), control = ctl)))
