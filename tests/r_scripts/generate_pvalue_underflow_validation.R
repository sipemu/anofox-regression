# Reference values for tests/issue200_pvalue_underflow.rs (#200): p-values
# far below 1e-16 (R computes them in the upper tail).
suppressMessages(library(nlme)); suppressMessages(library(sandwich))
set.seed(200)
n <- 40
x1 <- round(seq(1, 10, length.out = n), 3); x2 <- round(runif(n, -1, 1), 3)
y <- round(3 + 2 * x1 + 0.5 * x2 + rnorm(n, sd = 0.05), 4)
w <- round(runif(n, 0.5, 2), 3)
fmt <- function(x) paste(ifelse(x == round(x), sprintf("%.1f", x), sprintf("%.17g", x)), collapse = ", ")
v <- function(name, x) cat(sprintf("const %s: [f64; %d] = [%s];\n", name, length(x), fmt(unname(x))))
s1 <- function(name, x) cat(sprintf("const %s: f64 = %.17g;\n", name, x))
v("X1", x1); v("X2", x2); v("Y", y); v("W", w)
f <- lm(y ~ x1 + x2)
v("OLS_P", coef(summary(f))[, 4]); s1("OLS_F_P", with(summary(f), pf(fstatistic[1], fstatistic[2], fstatistic[3], lower.tail = FALSE)))
fw <- lm(y ~ x1 + x2, weights = w)
v("WLS_P", coef(summary(fw))[, 4])
g <- gls(y ~ x1 + x2)
v("GLS_P", summary(g)$tTable[, 4])
hc <- sqrt(diag(vcovHC(f, type = "HC1"))); tt <- coef(f) / hc
v("HC1_P", 2 * pt(abs(tt), f$df.residual, lower.tail = FALSE))
