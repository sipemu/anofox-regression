# Reference values for tests/issue205_na_exclude.rs (#205): na.exclude pads
# residuals and fitted values with NA at the incomplete rows.
set.seed(205)
n <- 15
x1 <- round(runif(n, 0, 5), 3); x2 <- round(rnorm(n), 3)
y <- round(1 + 0.7 * x1 - 0.4 * x2 + rnorm(n, sd = 0.3), 3)
yc <- rpois(n, exp(0.2 + 0.3 * x1))
x1[4] <- NA; x2[9] <- NA; y[12] <- NA; yc[12] <- NA
fmt <- function(x) paste(ifelse(is.na(x), "f64::NAN", ifelse(x == round(x), sprintf("%.1f", x), sprintf("%.17g", x))), collapse = ", ")
v <- function(name, x) cat(sprintf("const %s: [f64; %d] = [%s];\n", name, length(x), fmt(unname(x))))
v("X1", x1); v("X2", x2); v("Y", y); v("YC", yc)
f <- lm(y ~ x1 + x2, na.action = na.exclude)
v("OLS_RESID", residuals(f)); v("OLS_FITTED", fitted(f))
g <- glm(yc ~ x1 + x2, family = poisson, na.action = na.exclude, control = glm.control(epsilon = 1e-15))
v("POIS_RESID", residuals(g, type = "response")); v("POIS_FITTED", fitted(g))
