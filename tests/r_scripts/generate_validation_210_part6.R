# Reference values for tests/r_validation_210_part6.rs (#210): Huber IRLS
# weights and Wald table, isotonic prediction between knots, AFT without
# intercept and its prediction functions, RLS intervals, Sobel ADE / total.
suppressMessages({ library(MASS); library(survival) })
fmt <- function(v) { s <- sprintf("%.17g", v); paste(ifelse(grepl("[.e]|Inf|NaN", s), s, paste0(s, ".0")), collapse = ", ") }
vec <- function(tag, v) cat(sprintf("const %s: [f64; %d] = [%s];\n", tag, length(v), fmt(unname(as.numeric(v)))))
set.seed(2106)
n <- 50
x1 <- rnorm(n); x2 <- runif(n, -1, 1)
vec("X1", x1); vec("X2", x2)
# Huber: rlm(k = 1.345, MAD scale) to convergence; summary(method = "XtX")
yh <- 1 + 2 * x1 - x2 + rnorm(n, sd = 0.5); yh[c(4, 17, 33)] <- yh[c(4, 17, 33)] + c(6, -5, 8)
vec("Y_HUBER", yh)
h <- rlm(yh ~ x1 + x2, k = 1.345, acc = 1e-15, maxit = 500)
tab <- summary(h, method = "XtX")$coefficients; q <- qt(0.975, n - 3)
vec("HUBER_TABLE", c(tab[, 1], tab[, 2], tab[, 3], 2 * pt(-abs(tab[, 3]), n - 3),
                     tab[, 1] - q * tab[, 2], tab[, 1] + q * tab[, 2], h$s))
vec("HUBER_W", h$w)
# Isotonic: the step between the training x (value of the nearest x to the
# left) is approx(method = "constant", f = 0, rule = 2) on the fitted values
xi <- sort(runif(25, 0, 10)); yi <- sqrt(xi) + rnorm(25, sd = 0.4)
vec("X_ISO", xi); vec("Y_ISO", yi)
iso <- isoreg(xi, yi)
xout <- c(-1, xi[1], (xi[3] + xi[4]) / 2, xi[10], (xi[12] + xi[13]) / 2, 5, (xi[20] + xi[21]) / 2, xi[25], 12)
vec("ISO_XOUT", xout)
vec("ISO_PRED", approx(xi, iso$yf, xout, method = "constant", f = 0, rule = 2)$y)
# AFT: Weibull without intercept; with intercept: linear predictor,
# quantile and CDF at new points (Weibull and log-normal)
t0 <- rweibull(n, shape = 1.4, scale = exp(0.8 + 0.5 * x1 - 0.7 * x2)); cens <- runif(n, 0.5, 8)
time <- pmin(t0, cens); ev <- as.numeric(t0 <= cens)
vec("TIME", time); vec("EVENT", ev)
sctl <- survreg.control(rel.tolerance = 1e-14, maxiter = 200)
f0 <- survreg(Surv(time, ev) ~ x1 + x2 - 1, dist = "weibull", control = sctl)
vec("AFT_NOINT", c(coef(f0), sqrt(diag(vcov(f0)))[1:2], f0$scale, logLik(f0), AIC(f0)))
nd <- data.frame(x1 = c(-0.5, 0.3, 1.2), x2 = c(0.4, -0.2, 0.7))
vec("NEW_X1", nd$x1); vec("NEW_X2", nd$x2)
for (d in c("weibull", "lognormal")) {
  f <- survreg(Surv(time, ev) ~ x1 + x2, dist = d, control = sctl)
  lp <- predict(f, nd, type = "lp")
  vec(sprintf("AFT_%s_PRED", toupper(d)), c(lp, predict(f, nd, type = "quantile", p = 0.3),
      psurvreg(2, lp, f$scale, distribution = d)))
}
# RLS without forgetting = least squares: predict.lm intervals (90 %)
yr <- 0.5 + 1.5 * x1 + 0.8 * x2 + rnorm(n, sd = 0.7)
vec("Y_RLS", yr)
l <- lm(yr ~ x1 + x2)
pc <- predict(l, nd, interval = "confidence", level = 0.9); pp <- predict(l, nd, interval = "prediction", level = 0.9)
vec("RLS_CONF", c(pc[, 2], pc[, 3])); vec("RLS_PRED", c(pp[, 2], pp[, 3]))
# Sobel: ADE and total effect use the t inference of the outcome model and
# of y ~ x + c (confint.lm, summary.lm p)
xm <- rbinom(n, 1, 0.5); cm <- rnorm(n)
mm <- 0.3 + 0.6 * xm + 0.4 * cm + rnorm(n)
ym <- 0.2 + 0.3 * xm + 0.5 * mm - 0.3 * cm + rnorm(n)
vec("X_MED", xm); vec("C_MED", cm); vec("M_MED", mm); vec("Y_MED", ym)
fy <- lm(ym ~ xm + mm + cm); ft <- lm(ym ~ xm + cm)
row <- function(f) { s <- coef(summary(f))["xm", ]; ci <- confint(f, "xm", level = 0.9); c(s[1], s[2], s[3], s[4], ci) }
vec("SOBEL_ADE", row(fy)); vec("SOBEL_TOTAL", row(ft))
