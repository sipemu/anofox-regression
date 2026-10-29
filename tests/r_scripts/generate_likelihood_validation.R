#!/usr/bin/env Rscript
# Validation data for `inference::likelihood` (#101) against glm,
# anova(fit0, fit1, test = "LRT" / "Rao"), statmod::glm.scoretest and
# confint() (profile likelihood, R >= 4.4).
#
#   Rscript tests/r_scripts/generate_likelihood_validation.R > tests/fixtures/likelihood_validation.rs

suppressMessages(library(statmod))

emit <- function(name, v) {
  v <- as.numeric(v)
  cat("#[rustfmt::skip]\n")
  cat(sprintf("const %s: [f64; %d] = [\n    ", name, length(v)))
  cat(paste(sprintf("%.17e", v), collapse = ", "))
  cat("\n];\n")
}
scalar <- function(name, v) cat(sprintf("const %s: f64 = %.17e;\n", name, as.numeric(v)))

set.seed(20261028)
n <- 200
x1 <- rnorm(n); x2 <- rnorm(n); x3 <- runif(n)
expo <- runif(n, 0.5, 3); w <- round(runif(n, 0.5, 2), 2)
yb <- rbinom(n, 1, plogis(-0.3 + 0.8 * x1 - 0.5 * x2 + 0.6 * x3))
yp <- rpois(n, expo * exp(0.2 + 0.4 * x1 - 0.3 * x2 + 0.2 * x3))
yg <- rgamma(n, shape = 4, rate = 4 / exp(0.5 + 0.3 * x1 - 0.2 * x2 + 0.4 * x3))
emit("X1", x1); emit("X2", x2); emit("X3", x3); emit("EXPO", expo); emit("W", w)
emit("YB", yb); emit("YP", yp); emit("YG", yg)

tight <- glm.control(epsilon = 1e-14, maxit = 100)
case <- function(id, full, red, x_add) {
  emit(sprintf("%s_COEF", id), coef(full))
  emit(sprintf("%s_VCOV", id), t(vcov(full)))
  scalar(sprintf("%s_DEV", id), deviance(full))
  scalar(sprintf("%s_NULLDEV", id), full$null.deviance)
  scalar(sprintf("%s_DISP", id), summary(full)$dispersion)
  a <- anova(red, full, test = "LRT")
  disp <- summary(full)$dispersion
  scalar(sprintf("%s_LR", id), a$Deviance[2] / disp)
  scalar(sprintf("%s_LR_P", id), a[["Pr(>Chi)"]][2])
  r <- anova(red, full, test = "Rao")
  scalar(sprintf("%s_RAO", id), r$Rao[2])
  scalar(sprintf("%s_RAO_P", id), r[["Pr(>Chi)"]][2])
  emit(sprintf("%s_SCORE_Z", id), glm.scoretest(red, x_add))
  ci <- suppressMessages(confint(full))
  emit(sprintf("%s_CI95", id), t(ci))
  ci90 <- suppressMessages(confint(full, level = 0.9))
  emit(sprintf("%s_CI90", id), t(ci90))
}
X23 <- cbind(x2, x3)
case("BIN", glm(yb ~ x1 + x2 + x3, family = binomial, control = tight),
            glm(yb ~ x1, family = binomial, control = tight), X23)
case("POIS", glm(yp ~ x1 + x2 + x3 + offset(log(expo)), family = poisson, weights = w, control = tight),
             glm(yp ~ x1 + offset(log(expo)), family = poisson, weights = w, control = tight), X23)
case("GAMMA", glm(yg ~ x1 + x2 + x3, family = Gamma(link = "log"), control = tight),
              glm(yg ~ x1, family = Gamma(link = "log"), control = tight), X23)
