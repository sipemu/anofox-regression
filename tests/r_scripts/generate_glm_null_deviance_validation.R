# Reference values for tests/issue189_190_glm_dispersion_null_deviance.rs
# (#189 Poisson dispersion, #190 null deviance with offset / no intercept / weights).
suppressMessages(library(MASS))
set.seed(189)
n <- 40
x1 <- round(runif(n, -1, 1), 3)
x2 <- round(runif(n, 0, 2), 3)
expo <- round(runif(n, 0.5, 3), 3)
# Overdispersed counts (negative binomial, theta = 1.5)
y <- rnbinom(n, mu = expo * exp(0.4 + 0.6 * x1 - 0.3 * x2), size = 1.5)
yg <- round(rgamma(n, shape = 2, rate = 2 / (expo * exp(0.2 + 0.5 * x1))), 4)
m <- sample(5:15, n, replace = TRUE)
s <- rbinom(n, m, plogis(-0.3 + 0.8 * x1))
p <- s / m
off <- log(expo)
ctl <- glm.control(epsilon = 1e-15, maxit = 200)
v <- function(name, x) cat(sprintf("const %s: [f64; %d] = [%s];\n", name, length(x),
                                   paste(ifelse(x == round(x), sprintf("%.1f", x), sprintf("%.17g", x)), collapse = ", ")))
s1 <- function(name, x) cat(sprintf("const %s: f64 = %.17g;\n", name, x))
v("X1", x1); v("X2", x2); v("OFF", off); v("Y", y); v("YG", yg); v("M", m); v("P", p)

# 189: Poisson on overdispersed counts, dispersion 1 and quasipoisson
f <- glm(y ~ x1 + x2, family = poisson, control = ctl)
cs <- coef(summary(f))
v("POIS_COEF", cs[, 1]); v("POIS_SE", cs[, 2]); v("POIS_Z", cs[, 3]); v("POIS_P", cs[, 4])
q <- glm(y ~ x1 + x2, family = quasipoisson, control = ctl)
cq <- coef(summary(q))
s1("QPOIS_DISP", summary(q)$dispersion)
v("QPOIS_SE", cq[, 2]); v("QPOIS_P", cq[, 4])
s1("PEARSON_OVER_DF", sum(residuals(f, "pearson")^2) / f$df.residual)

nd <- function(f) f$null.deviance
# 190: null deviance
s1("ND_POIS", nd(f))
s1("ND_POIS_OFF", nd(glm(y ~ x1 + x2 + offset(off), family = poisson, control = ctl)))
s1("ND_POIS_NOINT", nd(glm(y ~ x1 + x2 - 1, family = poisson, control = ctl)))
s1("ND_POIS_OFF_NOINT", nd(glm(y ~ x1 + x2 - 1 + offset(off), family = poisson, control = ctl)))
s1("ND_POIS_SQRT_OFF", nd(glm(y ~ x1 + x2 + offset(sqrt(expo)), family = poisson("sqrt"), control = ctl,
                              start = c(1, 0, 0))))
th <- 1.5
s1("ND_NB_OFF", nd(glm(y ~ x1 + x2 + offset(off), family = negative.binomial(th), control = ctl)))
s1("ND_NB_NOINT", nd(glm(y ~ x1 + x2 - 1, family = negative.binomial(th), control = ctl)))
s1("ND_GAMMA_OFF", nd(glm(yg ~ x1 + offset(off), family = Gamma("log"), control = ctl)))
s1("ND_GAMMA_NOINT", nd(glm(yg ~ x1 + x2 - 1, family = Gamma("log"), control = ctl)))
s1("ND_BIN_W", nd(glm(p ~ x1, family = binomial, weights = m, control = ctl)))
s1("ND_BIN_W_OFF", nd(glm(p ~ x1 + offset(0.1 * x2), family = binomial, weights = m, control = ctl)))
s1("ND_BIN_W_NOINT", nd(glm(p ~ x1 - 1, family = binomial, weights = m, control = ctl)))
