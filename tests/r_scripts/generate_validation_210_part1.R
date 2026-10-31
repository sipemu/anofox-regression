# Reference values for tests/r_validation_210_part1.rs (#210): Tweedie link
# powers / Gamma links (statmod, glm), RLS with forgetting factor (closed-form
# exponentially weighted least squares) and weighted isotonic regression (Iso).
suppressMessages({library(statmod); library(Iso)})
fmt <- function(x) paste(ifelse(x == round(x), sprintf("%.1f", x), sprintf("%.17g", x)), collapse = ", ")
v <- function(name, x) cat(sprintf("const %s: [f64; %d] = [%s];\n", name, length(x), fmt(unname(as.numeric(x)))))
set.seed(210)
n <- 50
x <- round(runif(n, 0.5, 3), 3)
yg <- round(rgamma(n, shape = 4, rate = 4 / (1 + 0.8 * x)), 4)
v("X", x); v("YG", yg)
ctl <- glm.control(epsilon = 1e-15, maxit = 200)
cf <- function(f) c(coef(f), deviance(f))
v("GAMMA_INVERSE", cf(glm(yg ~ x, family = Gamma("inverse"), control = ctl)))
v("GAMMA_IDENTITY", cf(glm(yg ~ x, family = Gamma("identity"), control = ctl, start = c(1, 0.8))))
v("TW15_IDENTITY", cf(glm(yg ~ x, family = tweedie(var.power = 1.5, link.power = 1), control = ctl, start = c(1, 0.8))))
v("TW15_SQRT", cf(glm(yg ~ x, family = tweedie(var.power = 1.5, link.power = 0.5), control = ctl)))
v("TW15_INVERSE", cf(glm(yg ~ x, family = tweedie(var.power = 1.5, link.power = -1), control = ctl)))
# RLS: exponentially weighted LS with the P0 = delta I prior
x2 <- round(rnorm(n), 3); yr <- round(1 + 0.5 * x + 2 * x2 + rnorm(n, sd = 0.3), 4); v("X2", x2); v("YR", yr)
rls <- function(lam, delta = 1e6) {
  X <- cbind(1, x, x2); w <- lam^(n - seq_len(n))
  solve(crossprod(X, w * X) + lam^n / delta * diag(3), crossprod(X, w * yr))
}
v("RLS_095", rls(0.95)); v("RLS_099", rls(0.99))
# Weighted isotonic (increasing) regression
yi <- round(x + rnorm(n, sd = 0.6), 4); wi <- round(runif(n, 0.2, 3), 3); o <- order(x)
v("YI", yi); v("WI", wi)
fit <- numeric(n); fit[o] <- pava(yi[o], wi[o]); v("ISO_W", fit)
