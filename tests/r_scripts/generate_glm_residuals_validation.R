# Reference values for tests/issue197_glm_residuals.rs (#197): residuals.glm
# (pearson, deviance, working) incl. prior weights.
suppressMessages(library(MASS)); suppressMessages(library(statmod))
set.seed(197)
n <- 20
x <- round(runif(n, -1, 1), 3)
m <- sample(2:12, n, replace = TRUE); s <- rbinom(n, m, plogis(0.2 + 1.1 * x))
yc <- rpois(n, exp(0.5 + 0.7 * x))
yg <- round(rgamma(n, shape = 2, rate = 2 / exp(0.3 + 0.5 * x)), 4)
fmt <- function(x) paste(ifelse(x == round(x), sprintf("%.1f", x), sprintf("%.17g", x)), collapse = ", ")
v <- function(name, x) cat(sprintf("const %s: [f64; %d] = [%s];\n", name, length(x), fmt(unname(x))))
v("X", x); v("M", m); v("S", s); v("YC", yc); v("YG", yg)
ctl <- glm.control(epsilon = 1e-15, maxit = 200)
out <- function(tag, f) for (t in c("pearson", "deviance", "working")) v(paste0(tag, "_", toupper(t)), residuals(f, type = t))
out("BINW", glm(s / m ~ x, family = binomial, weights = m, control = ctl))
out("POIS", glm(yc ~ x, family = poisson, control = ctl))
out("GAMMA", glm(yg ~ x, family = Gamma("log"), control = ctl))
out("NB", glm(yc ~ x, family = negative.binomial(3), control = ctl))
out("TW", glm(yg ~ x, family = tweedie(var.power = 1.5, link.power = 0), control = ctl))
