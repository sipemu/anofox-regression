# Reference values for tests/r_validation_210_part2.rs (#210): survreg AIC /
# BIC / z / p / CI and orm category / exceedance probabilities.
suppressMessages({library(survival); library(rms)})
fmt <- function(x) paste(ifelse(x == round(x), sprintf("%.1f", x), sprintf("%.17g", x)), collapse = ", ")
v <- function(name, x) cat(sprintf("const %s: [f64; %d] = [%s];\n", name, length(x), fmt(unname(as.numeric(x)))))
set.seed(2102)
n <- 60
x1 <- round(rnorm(n), 3); x2 <- round(runif(n), 3)
t0 <- rweibull(n, shape = 1.5, scale = exp(1 + 0.4 * x1 - 0.6 * x2)); cens <- runif(n, 0.5, 6)
time <- round(pmin(t0, cens), 4); ev <- as.numeric(t0 <= cens)
v("X1", x1); v("X2", x2); v("TIME", time); v("EVENT", ev)
for (d in c("weibull", "lognormal", "loglogistic", "exponential")) {
  f <- survreg(Surv(time, ev) ~ x1 + x2, dist = d, control = survreg.control(rel.tolerance = 1e-14, maxiter = 200))
  s <- summary(f)$table[1:3, ]; ci <- confint(f)
  v(paste0(toupper(d), "_IC"), c(AIC(f), BIC(f)))
  v(paste0(toupper(d), "_Z"), s[, 3]); v(paste0(toupper(d), "_P"), s[, 4])
  v(paste0(toupper(d), "_CI_LO"), ci[, 1]); v(paste0(toupper(d), "_CI_HI"), ci[, 2])
}
# Ordinal: 4 levels, orm with three links; probabilities at three new points
yo <- as.integer(cut(0.8 * x1 - x2 + rlogis(n), c(-Inf, -1, 0, 1, Inf)))
v("YO", yo); v("NEW_X1", c(-1, 0, 1.2)); v("NEW_X2", c(0.2, 0.5, 0.9))
nd <- data.frame(x1 = c(-1, 0, 1.2), x2 = c(0.2, 0.5, 0.9))
for (l in c("logistic", "probit", "cloglog")) {
  f <- orm(yo ~ x1 + x2, family = l, eps = 1e-12)
  v(paste0("ORM_", toupper(l), "_IND"), t(predict(f, nd, type = "fitted.ind")))
  v(paste0("ORM_", toupper(l), "_EXC"), t(predict(f, nd, type = "fitted")))
}
