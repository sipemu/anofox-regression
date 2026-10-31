# Reference values for tests/issue206_glm_default_tolerance.rs (#206): fits at
# the crate's *default* tolerance must agree with fully converged R glm
# (epsilon = 1e-15) to well below what users can see.
set.seed(206)
n <- 50
x <- round(runif(n, -1, 1), 3)
yg <- round(rgamma(n, shape = 2, rate = 2 / exp(0.6 + 0.8 * x)), 4)
yb <- rbinom(n, 1, 1 - exp(-exp(-0.3 + 1.2 * x)))
yp <- rpois(n, 2 + 1.5 * (x + 1))
fmt <- function(x) paste(ifelse(x == round(x), sprintf("%.1f", x), sprintf("%.17g", x)), collapse = ", ")
v <- function(name, x) cat(sprintf("const %s: [f64; %d] = [%s];\n", name, length(x), fmt(unname(x))))
v("X", x); v("YG", yg); v("YB", yb); v("YP", yp)
ctl <- glm.control(epsilon = 1e-15, maxit = 200)
v("GAMMA_LOG", coef(glm(yg ~ x, family = Gamma("log"), control = ctl)))
v("BIN_CLOGLOG", coef(glm(yb ~ x, family = binomial("cloglog"), control = ctl)))
v("BIN_PROBIT", coef(glm(yb ~ x, family = binomial("probit"), control = ctl)))
v("POIS_IDENTITY", coef(glm(yp ~ x, family = poisson("identity"), control = ctl, start = c(3, 1))))
v("POIS_LOG", coef(glm(yp ~ x, family = poisson, control = ctl)))
