# Reference values for tests/r_validation_210_part5.rs (#210): GLM options
# without an R comparison so far.
suppressMessages({ library(statmod); library(glmnet); library(MASS) })
fmt <- function(v) { s <- sprintf("%.17g", v); paste(ifelse(grepl("[.e]|Inf|NaN", s), s, paste0(s, ".0")), collapse = ", ") }
vec <- function(tag, v) cat(sprintf("const %s: [f64; %d] = [%s];\n", tag, length(v), fmt(v)))
ctl <- glm.control(epsilon = 1e-15, maxit = 200)
set.seed(2105)
n <- 40
x1 <- runif(n, 0.2, 2); x2 <- rnorm(n)
expo <- runif(n, 0.5, 3)
vec("X1", x1); vec("X2", x2); vec("EXPO", expo)
# Gamma(log) without intercept and with an offset
yg <- rgamma(n, shape = 3, rate = 3 / exp(0.3 + 0.6 * x1 - 0.4 * x2))
yo <- rgamma(n, shape = 3, rate = 3 / (expo * exp(0.2 + 0.5 * x1)))
vec("Y_GAMMA", yg); vec("Y_GAMMA_OFFSET", yo)
# coefficients, SEs, t-based Wald limits (summary.glm's t on n - p df), deviance, logLik
glm_row <- function(tag, f) {
  cs <- coef(summary(f)); q <- qt(0.975, df.residual(f))
  vec(tag, c(cs[, 1], cs[, 2], cs[, 1] - q * cs[, 2], cs[, 1] + q * cs[, 2], deviance(f), logLik(f)))
}
glm_row("GAMMA_NOINT", glm(yg ~ x1 + x2 - 1, family = Gamma("log"), control = ctl))
glm_row("GAMMA_OFFSET", glm(yo ~ x1 + offset(log(expo)), family = Gamma("log"), control = ctl))
# Tweedie defaults: var.power 1.5 with tweedie's default link.power = 1 - var.power
ycp <- pmax(0, rgamma(n, shape = 1.2, rate = 1.2 / exp(0.4 + 0.5 * x1)) - 0.4)
vec("Y_CPG", ycp)
f <- glm(ycp ~ x1 + x2, family = tweedie(var.power = 1.5), control = ctl)
cs <- coef(summary(f)); vec("TWEEDIE_DEFAULT_LINK", c(cs[, 1], cs[, 2], deviance(f)))
# var.power 0 = Gaussian identity: lm
yn <- 1 + 2 * x1 - x2 + rnorm(n)
vec("Y_NORMAL", yn)
l <- lm(yn ~ x1 + x2); cs <- coef(summary(l))
vec("TWEEDIE_NORMAL", c(cs[, 1], cs[, 2], deviance(l), logLik(l)))
# inverse Gaussian(log) log-likelihood
yig <- statmod::rinvgauss(n, mean = exp(0.2 + 0.3 * x1), shape = 4)
vec("Y_INVGAUSS", yig)
f <- glm(yig ~ x1 + x2, family = inverse.gaussian("log"), control = ctl)
vec("INVGAUSS_LOGLIK", c(coef(f), deviance(f), logLik(f), AIC(f), BIC(f)))
# ridge-penalised IRLS = glmnet(alpha = 0) with lambda_glmnet = lambda / n
X <- cbind(x1, x2); LAMBDA <- 3
cat(sprintf("const LAMBDA: f64 = %s;\n", fmt(LAMBDA)))
gctl <- list(thresh = 1e-22, maxit = 1e7, epsnr = 1e-15, mxitnr = 1000)
ridge <- function(tag, y, family) {
  g <- glmnet(X, y, family = family, alpha = 0, lambda = LAMBDA / n, standardize = FALSE, control = gctl)
  vec(tag, c(as.numeric(g$a0), as.numeric(g$beta)))
}
yp <- rpois(n, exp(0.5 + 0.7 * x1 - 0.3 * x2))
yb <- rbinom(n, 1, plogis(-0.5 + 0.9 * x1 + 0.8 * x2))
ynb <- rnbinom(n, mu = exp(0.6 + 0.5 * x1), size = 2)
vec("Y_POISSON", yp); vec("Y_BINOMIAL", yb); vec("Y_NEGBIN", ynb)
ridge("RIDGE_POISSON", yp, "poisson")
ridge("RIDGE_BINOMIAL", yb, "binomial")
ridge("RIDGE_NEGBIN", ynb, negative.binomial(2))
# Gamma: glmnet's family-object IRLS stops ~1e-9 short; Fisher scoring for
# the penalised score X'((y - mu) / mu) - lambda * beta = 0 to convergence
b <- c(0, 0, 0); Xd <- cbind(1, X); P <- diag(c(0, LAMBDA, LAMBDA))
for (i in 1:200) { mu <- exp(drop(Xd %*% b)); s <- crossprod(Xd, (yg - mu) / mu) - P %*% b; b <- b + drop(solve(crossprod(Xd) + P, s)) }
vec("RIDGE_GAMMA", b)
# binomial predict_with_se for probit / cloglog: link fit, link SE,
# response fit, response SE (delta method), response limits from the link
nd <- data.frame(x1 = c(0.4, 1.1, 1.8), x2 = c(0.5, -0.2, 1.0))
for (lk in c("probit", "cloglog")) {
  f <- glm(yb ~ x1 + x2, family = binomial(lk), control = ctl)
  pl <- predict(f, nd, type = "link", se.fit = TRUE)
  pr <- predict(f, nd, type = "response", se.fit = TRUE)
  z <- qnorm(0.95); inv <- binomial(lk)$linkinv
  vec(sprintf("PREDICT_%s", toupper(lk)), c(pl$fit, pl$se.fit, pr$fit, pr$se.fit,
      inv(pl$fit - z * pl$se.fit), inv(pl$fit + z * pl$se.fit)))
}
vec("NEW_X1", nd$x1); vec("NEW_X2", nd$x2)
