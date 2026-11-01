# Reference values for tests/r_validation_210_part13.rs (#210): Laplace
# priors of the penalized GLM engine vs glmnet(penalty.factor), and the
# engine's Tweedie 1 < p < 2 vs statmod::tweedie.
suppressMessages({ library(glmnet); library(statmod) })
fmt <- function(v) { s <- sprintf("%.17g", v); paste(ifelse(grepl("[.e]|Inf|NaN", s), s, paste0(s, ".0")), collapse = ", ") }
vec <- function(tag, v) cat(sprintf("const %s: [f64; %d] = [%s];\n", tag, length(v), fmt(unname(as.numeric(v)))))
set.seed(2113)
n <- 80; p <- 3
X <- matrix(rnorm(n * p), n, p)
vec("X_FLAT", t(X))
yp <- rpois(n, exp(0.4 + X %*% c(0.6, -0.3, 0.05)))
yb <- rbinom(n, 1, plogis(-0.2 + X %*% c(1.0, -0.6, 0.1)))
vec("Y_POIS", yp); vec("Y_BIN", yb)
# Laplace(loc, s) prior on each slope: -loglik + sum |b - loc| / s, which is
# glmnet's lasso with penalty.factor = 1 / s (rescaled to sum to p) and
# lambda = sum(1 / s) / (n p); a location enters as offset X loc.
s <- c(0.5, 0.2, 1.0); vec("SCALES", s)
lam <- sum(1 / s) / (n * p)
ctl <- list(thresh = 1e-22, maxit = 1e7, epsnr = 1e-15, mxitnr = 1000)
lasso <- function(y, fam, loc = c(0, 0, 0)) {
  g <- glmnet(X, y, family = fam, alpha = 1, lambda = lam, penalty.factor = 1 / s,
              offset = drop(X %*% loc), standardize = FALSE, control = ctl)
  c(as.numeric(g$a0), as.numeric(g$beta) + loc)
}
vec("LAPLACE_POIS", lasso(yp, "poisson"))
vec("LAPLACE_BIN", lasso(yb, "binomial"))
loc <- c(0.5, 0, -0.2); vec("LOC", loc)
vec("LAPLACE_POIS_LOC", lasso(yp, "poisson", loc))
# Tweedie p = 1.5, log link, no penalty
yt <- pmax(0, rgamma(n, shape = 1.5, rate = 1.5 / exp(0.3 + X %*% c(0.5, -0.4, 0.2))) - 0.3)
vec("Y_TWEEDIE", yt)
f <- glm(yt ~ X, family = tweedie(var.power = 1.5, link.power = 0), control = glm.control(epsilon = 1e-15, maxit = 200))
vec("TWEEDIE_COEF", coef(f)); vec("TWEEDIE_SE", sqrt(diag(vcov(f))))
