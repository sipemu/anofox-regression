# Reference values for tests/issue192_glm_aliasing.rs (#192): IRLS GLMs alias
# columns that are linear combinations of the intercept and earlier columns.
suppressMessages(library(MASS)); suppressMessages(library(statmod))
set.seed(192)
n <- 40
a <- round(runif(n, -1, 1), 3); b <- round(runif(n, 0, 2), 3)
x3 <- a + b                 # collinear with a, b
x4 <- 2 - 0.5 * a           # collinear with the intercept and a
fmt <- function(x) paste(ifelse(is.na(x), "f64::NAN", ifelse(x == round(x), sprintf("%.1f", x), sprintf("%.17g", x))), collapse = ", ")
v <- function(name, x) cat(sprintf("const %s: [f64; %d] = [%s];\n", name, length(x), fmt(x)))
s1 <- function(name, x) cat(sprintf("const %s: f64 = %.17g;\n", name, x))
yc <- rpois(n, exp(0.3 + 0.5 * a + 0.2 * b))
yb <- rbinom(n, 1, plogis(-0.2 + 1.2 * a))
yg <- round(rgamma(n, shape = 3, rate = 3 / exp(0.1 + 0.4 * a - 0.2 * b)), 4)
v("A", a); v("B", b); v("YC", yc); v("YB", yb); v("YG", yg)
# glm.fit uses QR tolerance min(1e-7, epsilon / 1000): keep it >= 1e-13 so
# aliasing is detected, while converging tightly.
ctl <- glm.control(epsilon = 1e-10, maxit = 200)
# Aliasing pattern from `f`; values from a tight refit on the kept columns
# (the 1e-10 fit is only converged to ~1e-7).
out <- function(tag, f, dispersion = NULL) {
  cf <- coef(f); keep <- !is.na(cf)
  X <- model.matrix(f)[, keep, drop = FALSE]
  g <- glm.fit(X, f$y, family = f$family, control = glm.control(epsilon = 1e-15, maxit = 200))
  class(g) <- c("glm", "lm")
  cs <- coef(summary.glm(g, dispersion = dispersion))
  se <- rep(NA, length(cf)); cf[keep] <- cs[, 1]; se[keep] <- cs[, 2]
  v(paste0(tag, "_COEF"), cf); v(paste0(tag, "_SE"), se); s1(paste0(tag, "_DEV"), g$deviance)
}
# columns: a, b, x3, x4 (x3 = a + b, x4 = 2 - a/2)
out("POIS", glm(yc ~ a + b + x3 + x4, family = poisson, control = ctl))
out("BIN", glm(yb ~ a + b + x3 + x4, family = binomial, control = ctl))
out("GAMMA", glm(yg ~ a + b + x3 + x4, family = Gamma("log"), control = ctl))
out("NB", glm(yc ~ a + b + x3 + x4, family = negative.binomial(2), control = ctl), dispersion = 1)
out("TW", glm(yg ~ a + b + x3 + x4, family = tweedie(var.power = 1.5, link.power = 0), control = ctl))
# without intercept: duplicate column (x4 not aliased here)
out("POIS_NOINT", glm(yc ~ a + b + x3 + x4 - 1, family = poisson, control = ctl))
