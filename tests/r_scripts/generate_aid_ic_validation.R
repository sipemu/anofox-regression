# Reference values for tests/issue203_aid_ic.rs (#203): the information
# criterion (AICc) of each AID candidate equals AICc(greybox::alm(y ~ 1, d))
# at its maximum-likelihood fit (Poisson directly at lambda = mean(y), since
# greybox's intercept-only dpois optimiser does not converge).
suppressMessages(library(greybox))
set.seed(203)
fmt <- function(x) paste(ifelse(x == round(x), sprintf("%.1f", x), sprintf("%.17g", x)), collapse = ", ")
v <- function(name, x) cat(sprintf("const %s: [f64; %d] = [%s];\n", name, length(x), fmt(unname(as.numeric(x)))))
s1 <- function(name, x) cat(sprintf("const %s: f64 = %.17g;\n", name, x))
aicc <- function(f) AICc(f)
fit <- function(y, d) alm(y ~ 1, distribution = d, xtol_rel = 1e-14, ftol_rel = 1e-14, maxeval = 1e5)
pois_aicc <- function(y) { ll <- sum(dpois(y, mean(y), log = TRUE)); n <- length(y); -2 * ll + 2 + 4 / (n - 2) }
yc <- rnbinom(60, mu = 5, size = 3)                       # regular count
yi <- rnbinom(60, mu = 0.6, size = 0.8)                   # intermittent count
yf <- round(rgamma(60, 4, 4 / 6), 3)                      # regular fractional
v("YC", yc); v("YI", yi); v("YF", yf)
s1("YC_POIS", pois_aicc(yc)); s1("YC_NB", aicc(fit(yc, "dnbinom"))); s1("YC_NORM", aicc(fit(yc, "dnorm")))
s1("YI_POIS", pois_aicc(yi)); s1("YI_NB", aicc(fit(yi, "dnbinom"))); s1("YI_GEOM", aicc(suppressWarnings(fit(yi, "dgeom"))))
s1("YF_NORM", aicc(fit(yf, "dnorm"))); s1("YF_GAMMA", aicc(fit(yf, "dgamma"))); s1("YF_LNORM", aicc(fit(yf, "dlnorm")))
s1("YC_NB_SIZE", fit(yc, "dnbinom")$other$size)
cat(sprintf("// zero share: yc %.3f, yi %.3f\n", mean(yc == 0), mean(yi == 0)))
