# Reference values for tests/issue191_bls.rs (#191): bounded least squares
# with non-zero bounds, unbounded coefficients and an intercept.
# nnls 1.6, bvls 1.4, quadprog 1.5.8.
suppressMessages({library(nnls); library(bvls); library(quadprog)})
set.seed(191)
n <- 30
X <- cbind(round(runif(n, 0, 2), 3), round(rnorm(n), 3), round(runif(n, -1, 1), 3))
y <- round(1.5 - 0.8 * X[, 1] + 0.6 * X[, 2] + 2 * X[, 3] + rnorm(n, sd = 0.3), 3)
fmt <- function(x) paste(ifelse(x == round(x), sprintf("%.1f", x), sprintf("%.17g", x)), collapse = ", ")
v <- function(name, x) cat(sprintf("const %s: [f64; %d] = [%s];\n", name, length(x), fmt(x)))
v("X0", X[, 1]); v("X1", X[, 2]); v("X2", X[, 3]); v("Y", y)
rss <- function(b, A) sum((y - A %*% b)^2)

# 1. NNLS without intercept
b <- coef(nnls(X, y)); v("NNLS", b)
# 2. Box bounds without intercept, all binding at non-zero values: lower [0.2,-1,0], upper [0.5,0.25,1.5]
b <- bvls(X, y, bl = c(0.2, -1, 0), bu = c(0.5, 0.25, 1.5))$x; v("BOX", b)
v("BOX_QP", qp0 <- solve.QP(crossprod(X), crossprod(X, y), cbind(diag(3), -diag(3)), c(0.2, -1, 0, -0.5, -0.25, -1.5))$solution)
# 3. NNLS with an unbounded intercept: quadprog
A <- cbind(1, X)
qp <- function(lo, up) {
  D <- crossprod(A); d <- crossprod(A, y)
  cons <- NULL; b0 <- NULL
  for (j in seq_len(ncol(A))) {
    if (is.finite(lo[j])) { e <- numeric(ncol(A)); e[j] <- 1; cons <- cbind(cons, e); b0 <- c(b0, lo[j]) }
    if (is.finite(up[j])) { e <- numeric(ncol(A)); e[j] <- -1; cons <- cbind(cons, e); b0 <- c(b0, -up[j]) }
  }
  solve.QP(D, d, cons, b0)$solution
}
v("NNLS_INT", qp(c(-Inf, 0, 0, 0), c(Inf, Inf, Inf, Inf)))
# 4. Mixed: intercept, x0 unbounded, x1 in [-Inf, 0.3], x2 in [2.5, 4] (lower bound active)
v("MIXED_INT", qp(c(-Inf, -Inf, -Inf, 2.5), c(Inf, Inf, 0.3, 4)))
# 5. Intercept, nnls on y = -5 + 0.5 x (the issue's case)
xs <- seq(0, 10, length.out = 21); ys <- -5 + 0.5 * xs + 0.01 * sin(1:21)
v("XS", xs); v("YS", ys)
v("LINE", coef(lm(ys ~ xs)))
