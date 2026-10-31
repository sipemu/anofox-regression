# Reference values for tests/issue193_binomial_loglik.rs (#193): logLik / AIC /
# BIC of binomial GLMs with proportion responses and trial weights.
set.seed(193)
n <- 30
x <- round(runif(n, -1.5, 1.5), 3)
m <- sample(3:20, n, replace = TRUE)
s <- rbinom(n, m, plogis(-0.4 + 0.9 * x))
yb <- rbinom(n, 1, plogis(0.3 - 0.7 * x)); wb <- sample(1:4, n, replace = TRUE)
fmt <- function(x) paste(ifelse(x == round(x), sprintf("%.1f", x), sprintf("%.17g", x)), collapse = ", ")
v <- function(name, x) cat(sprintf("const %s: [f64; %d] = [%s];\n", name, length(x), fmt(x)))
s1 <- function(name, x) cat(sprintf("const %s: f64 = %.17g;\n", name, x))
v("X", x); v("M", m); v("S", s); v("YB", yb); v("WB", wb)
ctl <- glm.control(epsilon = 1e-15, maxit = 100)
out <- function(tag, f) {
  s1(paste0(tag, "_LOGLIK"), as.numeric(logLik(f))); s1(paste0(tag, "_AIC"), AIC(f)); s1(paste0(tag, "_BIC"), BIC(f))
}
out("PROP_LOGIT", glm(s / m ~ x, family = binomial, weights = m, control = ctl))
out("PROP_PROBIT", glm(cbind(s, m - s) ~ x, family = binomial("probit"), control = ctl))
wz <- m; wz[c(4, 11)] <- 0
out("ZEROW", glm(s / m ~ x, family = binomial, weights = wz, control = ctl))
out("CASEW", glm(yb ~ x, family = binomial, weights = wb, control = ctl))
out("PLAIN", glm(yb ~ x, family = binomial, control = ctl))
