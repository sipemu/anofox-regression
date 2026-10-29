#!/usr/bin/env Rscript
# Validation data for `inference::approximation` (#108): heuristic shrinkage
# (RMS §4.5) and model approximation (RMS §5.5.2, Ch 19).
#
#   Rscript tests/r_scripts/generate_approximation_validation.R > tests/fixtures/approximation_validation.rs

suppressMessages(library(rms))
emit <- function(name, v) {
  v <- as.numeric(v)
  cat("#[rustfmt::skip]\n")
  cat(sprintf("const %s: [f64; %d] = [\n    ", name, length(v)))
  cat(paste(sprintf("%.17e", v), collapse = ", "))
  cat("\n];\n")
}
emit_usize <- function(name, v) cat(sprintf("const %s: [usize; %d] = [%s];\n", name, length(v), paste(v, collapse = ", ")))

# The data and model of generate_fastbw_validation.R
set.seed(20261028)
n <- 300
x1 <- rnorm(n); x2 <- rnorm(n); x3 <- rnorm(n); x4 <- rnorm(n); x5 <- rnorm(n)
g <- sample(1:3, n, replace = TRUE)
y <- rbinom(n, 1, plogis(-0.3 + 0.8 * x1 + 0.3 * x1^2 - 0.6 * x2 + 0.15 * x3 + c(0, 0.4, 0.1)[g]))
d <- data.frame(x1, x2, x3, x4, x5, g = factor(g), y)
f <- lrm(y ~ rcs(x1, 4) + x2 + x3 + x4 + x5 + g, data = d, x = TRUE)
X <- f$x                       # slope columns, in coef order
lp <- as.vector(cbind(1, X) %*% coef(f))
emit("COEF", coef(f)); emit("VCOV", t(vcov(f))); emit("X", t(X)); emit("LP", lp)

# Heuristic shrinkage (van Houwelingen & le Cessie): (LR - p) / LR
lr <- f$stats[["Model L.R."]]; p <- f$stats[["d.f."]]
emit("LR_DF_GAMMA", c(lr, p, (lr - p) / lr))

# Ch 19: order the terms by fastbw on the approximation ols(lp ~ ., sigma = 1)
# with every term deleted, and the R^2 of the approximation after each deletion.
d$lp <- lp
a <- ols(lp ~ rcs(x1, 4) + x2 + x3 + x4 + x5 + g, data = d, sigma = 1)
fb <- fastbw(a, aics = 10000)
emit_usize("PATH_DELETED", fb$factors.deleted - 1)
labels <- c("rcs(x1, 4)", "x2", "x3", "x4", "x5", "g")
r2 <- sapply(seq_along(fb$factors.deleted), function(k) {
  keep <- labels[-fb$factors.deleted[1:k]]
  if (length(keep) == 0) return(0)
  summary(lm(as.formula(paste("lp ~", paste(keep, collapse = " + "))), data = d))$r.squared
})
emit("PATH_R2", r2)

# RMS §5.5.2: approximate the full model by x1 and x2; W = (Z'Z)^-1 Z'X,
# var(gamma) = W V W'
Xf <- cbind(1, X)
Z <- Xf[, c(1, 2:4, 5)]        # intercept, rcs(x1) (3 columns), x2
gam <- solve(crossprod(Z), crossprod(Z, lp))
W <- solve(crossprod(Z), crossprod(Z, Xf))
emit("APPROX_COEF", gam)
emit("APPROX_VCOV", t(W %*% vcov(f) %*% t(W)))
emit("APPROX_R2", 1 - sum((lp - Z %*% gam)^2) / sum((lp - mean(lp))^2))
