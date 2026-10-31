# Reference values for tests/issue202_alm_ic.rs (#202): greybox::alm's
# nparam / logLik / AIC / AICc / BIC for distributions with and without a
# scale parameter.
suppressMessages(library(greybox))
set.seed(202)
n <- 80
x <- round(rnorm(n), 3)
yn <- round(1 + 0.5 * x + rnorm(n, sd = 0.7), 3)
yp <- rpois(n, exp(1 + 0.3 * x))
ye <- round(rexp(n, 1 / exp(0.5 + 0.2 * x)), 4)
yb <- rbinom(n, 1, plogis(0.2 + x))
fmt <- function(x) paste(ifelse(x == round(x), sprintf("%.1f", x), sprintf("%.17g", x)), collapse = ", ")
v <- function(name, x) cat(sprintf("const %s: [f64; %d] = [%s];\n", name, length(x), fmt(unname(as.numeric(x)))))
v("X", x); v("YN", yn); v("YP", yp); v("YE", ye); v("YB", yb)
out <- function(tag, f) v(tag, c(nparam(f), logLik(f), AIC(f), AICc(f), BIC(f)))
out("NORMAL", alm(yn ~ x, distribution = "dnorm", xtol_rel = 1e-14, ftol_rel = 1e-14, maxeval = 1e5))
out("POISSON", alm(yp ~ x, distribution = "dpois", xtol_rel = 1e-14, ftol_rel = 1e-14, maxeval = 1e5))
out("GEOMETRIC", suppressWarnings(alm(yp ~ x, distribution = "dgeom", xtol_rel = 1e-14, ftol_rel = 1e-14, maxeval = 1e5)))
out("EXPONENTIAL", alm(ye ~ x, distribution = "dexp", xtol_rel = 1e-14, ftol_rel = 1e-14, maxeval = 1e5))
out("BINOMIAL", alm(yb ~ x, distribution = "dbinom", xtol_rel = 1e-14, ftol_rel = 1e-14, maxeval = 1e5))
