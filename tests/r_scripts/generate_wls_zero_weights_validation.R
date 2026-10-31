# Reference values for tests/issue195_wls_zero_weights.rs (#195): WLS with
# zero weights counts only positive-weight rows for nobs and df.residual.
set.seed(195)
n <- 12
x1 <- round(runif(n, 0, 5), 3); x2 <- round(rnorm(n), 3)
w <- round(runif(n, 0.5, 2), 3); w[c(3, 9)] <- 0
y <- round(1 + 0.8 * x1 - 0.5 * x2 + rnorm(n, sd = 0.4), 3)
fmt <- function(x) paste(ifelse(x == round(x), sprintf("%.1f", x), sprintf("%.17g", x)), collapse = ", ")
v <- function(name, x) cat(sprintf("const %s: [f64; %d] = [%s];\n", name, length(x), fmt(x)))
s1 <- function(name, x) cat(sprintf("const %s: f64 = %.17g;\n", name, x))
v("X1", x1); v("X2", x2); v("W", w); v("Y", y)
f <- lm(y ~ x1 + x2, weights = w)
cs <- coef(summary(f))
v("COEF", cs[, 1]); v("SE", cs[, 2]); v("T", cs[, 3]); v("P", cs[, 4])
ci <- confint(f); v("CI_LO", ci[, 1]); v("CI_HI", ci[, 2])
s1("NOBS", nobs(f)); s1("DF_RESID", df.residual(f)); s1("SIGMA", summary(f)$sigma)
s1("R2", summary(f)$r.squared); s1("F", summary(f)$fstatistic[1])
nd <- data.frame(x1 = c(1, 2.5, 4), x2 = c(-0.5, 0, 1))
v("NEW_X1", nd$x1); v("NEW_X2", nd$x2)
pc <- predict(f, nd, interval = "confidence"); v("CONF_LO", pc[, 2]); v("CONF_HI", pc[, 3])
pp <- predict(f, nd, interval = "prediction"); v("PRED_LO", pp[, 2]); v("PRED_HI", pp[, 3])
