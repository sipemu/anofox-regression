# Reference values for tests/r_validation_tight.rs (#209): full-precision R
# goldens for functionality whose existing tests only used loose tolerances.
suppressMessages({library(sandwich); library(lmtest); library(MASS); library(quantreg)})
fmt <- function(x) paste(ifelse(x == round(x), sprintf("%.1f", x), sprintf("%.17g", x)), collapse = ", ")
v <- function(name, x) cat(sprintf("const %s: [f64; %d] = [%s];\n", name, length(x), fmt(unname(as.numeric(x)))))
s1 <- function(name, x) cat(sprintf("const %s: f64 = %.17g;\n", name, x))
set.seed(209)
n <- 30
x1 <- round(runif(n, 0, 10), 3); x2 <- round(rnorm(n), 3)
y <- round(1 + 0.5 * x1 + 2 * x2 + rnorm(n) * (0.2 + 0.1 * x1), 4)
v("X1", x1); v("X2", x2); v("Y", y)
m <- lm(y ~ x1 + x2)
# HC0-HC3: SEs, coeftest t / p, coefci (95%)
for (t in c("HC0", "HC1", "HC2", "HC3")) {
  V <- vcovHC(m, type = t); ct <- coeftest(m, vcov. = V); ci <- coefci(m, vcov. = V)
  v(paste0(t, "_SE"), ct[, 2]); v(paste0(t, "_T"), ct[, 3]); v(paste0(t, "_P"), ct[, 4])
  v(paste0(t, "_CI_LO"), ci[, 1]); v(paste0(t, "_CI_HI"), ci[, 2])
}
m0 <- lm(y ~ x1 + x2 - 1)
for (t in c("HC0", "HC3")) v(paste0(t, "_NOINT_SE"), sqrt(diag(vcovHC(m0, type = t))))
# OLS: confint at 90%, F p-value, intervals at 90%
ci90 <- confint(m, level = 0.9); v("CI90_LO", ci90[, 1]); v("CI90_HI", ci90[, 2])
s1("F_P", with(summary(m), pf(fstatistic[1], fstatistic[2], fstatistic[3], lower.tail = FALSE)))
nd <- data.frame(x1 = c(2, 5, 9.5), x2 = c(-1, 0, 1.5)); v("NEW_X1", nd$x1); v("NEW_X2", nd$x2)
pc <- predict(m, nd, interval = "confidence", level = 0.9); v("CONF90_LO", pc[, 2]); v("CONF90_HI", pc[, 3])
pp <- predict(m, nd, interval = "prediction", level = 0.9); v("PRED90_LO", pp[, 2]); v("PRED90_HI", pp[, 3])
# Influence diagnostics
v("HAT", hatvalues(m)); v("COOKS", cooks.distance(m)); v("RSTANDARD", rstandard(m)); v("RSTUDENT", rstudent(m)); v("DFFITS", dffits(m))
# Ridge (RSS + lambda ||b||^2, intercept unpenalised): closed form on centred data
lam <- 2.5; Xc <- scale(cbind(x1, x2), scale = FALSE); yc <- y - mean(y)
b <- solve(crossprod(Xc) + lam * diag(2), crossprod(Xc, yc)); v("RIDGE_COEF", c(mean(y) - colMeans(cbind(x1, x2)) %*% b, b))
# Quantile regression (exact LP)
for (tau in c(0.25, 0.5, 0.8)) v(sprintf("RQ%02d", round(100 * tau)), coef(rq(y ~ x1 + x2, tau = tau)))
# Logistic and Huber with tight control
yb <- as.numeric(y > median(y)); v("YB", yb)
g <- glm(yb ~ x1 + x2, family = binomial, control = glm.control(epsilon = 1e-15, maxit = 100))
v("LOGIT_COEF", coef(g)); v("LOGIT_SE", sqrt(diag(vcov(g))))
h <- rlm(y ~ x1 + x2, acc = 1e-14, maxit = 200); v("HUBER_COEF", coef(h)); s1("HUBER_SCALE", h$s)
