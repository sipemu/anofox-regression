# Reference values for tests/r_validation_210_part14.rs (#210): greybox::alm
# predictions with intervals, the log-S distribution, scale(), fits without
# intercept, and the MAE / HAM / ROLE losses, all at tight nloptr tolerances.
suppressMessages(library(greybox))
fmt <- function(v) { s <- sprintf("%.17g", v); paste(ifelse(grepl("[.e]|Inf|NaN", s), s, paste0(s, ".0")), collapse = ", ") }
vec <- function(tag, v) cat(sprintf("const %s: [f64; %d] = [%s];\n", tag, length(v), fmt(unname(as.numeric(v)))))
tight <- function(...) alm(..., xtol_rel = 1e-14, ftol_rel = 1e-14, maxeval = 1e5)
set.seed(2114)
n <- 60
x <- runif(n, 0, 4)
yn <- 1 + 0.8 * x + rnorm(n, sd = 0.7)
ypos <- exp(0.2 + 0.3 * x + rnorm(n, sd = 0.4))
yp <- rpois(n, exp(0.3 + 0.4 * x))
yo <- yn; yo[c(7, 30, 51)] <- yo[c(7, 30, 51)] + c(8, -7, 9)
vec("X", x); vec("YN", yn); vec("YPOS", ypos); vec("YP", yp); vec("YO", yo)
nd <- data.frame(x = c(0.5, 2, 3.5)); vec("X_NEW", nd$x)
# confidence intervals (90 %): mean, lower, upper
ci <- function(tag, f) {
  p <- predict(f, nd, interval = "confidence", level = 0.9)
  vec(tag, c(p$mean, p$lower, p$upper))
}
ci("CI_NORMAL", tight(yn ~ x, distribution = "dnorm"))
ci("CI_LAPLACE", tight(yn ~ x, distribution = "dlaplace"))
ci("CI_LOGISTIC", tight(yn ~ x, distribution = "dlogis"))
ci("CI_S", tight(yn ~ x, distribution = "ds"))
ci("CI_LOGNORMAL", tight(ypos ~ x, distribution = "dlnorm"))
ci("CI_LOGLAPLACE", tight(ypos ~ x, distribution = "dllaplace"))
p <- predict(tight(yn ~ x, distribution = "dnorm"), nd, interval = "prediction", level = 0.9)
vec("PI_NORMAL", c(p$mean, p$lower, p$upper))
# log-S, and scale() of several distributions
f <- tight(ypos ~ x, distribution = "dls"); vec("LOGS", c(coef(f), f$scale, logLik(f)))
sc <- function(d, y) tight(y ~ x, distribution = d)$scale
vec("SCALES", c(sc("dnorm", yn), sc("dlaplace", yn), sc("dlogis", yn), sc("ds", yn),
                sc("dlnorm", ypos), sc("dgamma", ypos), sc("dinvgauss", ypos)))
# without intercept: coefficient, scale (not for Poisson), logLik
ni <- function(d, y) { f <- suppressWarnings(tight(y ~ x - 1, distribution = d)); c(coef(f), if (d != "dpois") f$scale, logLik(f)) }
vec("NOINT_NORMAL", ni("dnorm", yn)); vec("NOINT_POISSON", ni("dpois", yp)); vec("NOINT_GAMMA", ni("dgamma", ypos))
# losses on data with outliers
lo <- function(loss, ...) coef(tight(yo ~ x, distribution = "dnorm", loss = loss, ...))
vec("LOSS_MAE", lo("MAE")); vec("LOSS_HAM", lo("HAM"))
vec("LOSS_ROLE05", lo("ROLE")); vec("LOSS_ROLE20", lo("ROLE", trim = 0.2))
