# Reference values for tests/r_validation_hetero.rs (#210): White's test with
# three regressors and Breusch-Pagan against custom auxiliary regressors
# (lmtest::bptest).
suppressMessages(library(lmtest))
fmt <- function(x) paste(ifelse(x == round(x), sprintf("%.1f", x), sprintf("%.17g", x)), collapse = ", ")
v <- function(name, x) cat(sprintf("const %s: [f64; %d] = [%s];\n", name, length(x), fmt(unname(as.numeric(x)))))
set.seed(2104)
n <- 60
x1 <- round(runif(n, 0, 3), 3); x2 <- round(rnorm(n), 3); x3 <- round(runif(n, -1, 1), 3)
z1 <- round(rexp(n), 3); z2 <- round(runif(n), 3)
y <- round(1 + x1 - 0.5 * x2 + 0.8 * x3 + rnorm(n, sd = 0.3 + 0.4 * x1), 4)
v("X1", x1); v("X2", x2); v("X3", x3); v("Z1", z1); v("Z2", z2); v("Y", y)
m <- lm(y ~ x1 + x2 + x3)
out <- function(tag, t) v(tag, c(t$statistic, t$parameter, t$p.value))
out("WHITE3", bptest(m, ~ x1 + x2 + x3 + I(x1^2) + I(x2^2) + I(x3^2) + x1:x2 + x1:x3 + x2:x3))
out("BPZ_KOENKER", bptest(m, ~ z1 + z2))
out("BPZ_CLASSIC", bptest(m, ~ z1 + z2, studentize = FALSE))
