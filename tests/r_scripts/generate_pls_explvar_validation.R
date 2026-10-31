# Reference values for tests/issue198_pls.rs (#198): pls::plsr (SIMPLS)
# explained X variance and fits without centring.
suppressMessages(library(pls))
set.seed(198)
n <- 25
X <- cbind(round(rnorm(n, 10, 2), 3), round(rnorm(n, 5, 1), 3), round(runif(n, 0, 4), 3), round(rnorm(n, -3, 1.5), 3))
X[, 2] <- round(X[, 2] + 0.4 * X[, 1], 3)
y <- round(100 + 0.8 * X[, 1] - 1.2 * X[, 2] + 0.5 * X[, 3] + rnorm(n, sd = 0.5), 3)
fmt <- function(x) paste(ifelse(x == round(x), sprintf("%.1f", x), sprintf("%.17g", x)), collapse = ", ")
v <- function(name, x) cat(sprintf("const %s: [f64; %d] = [%s];\n", name, length(x), fmt(unname(as.numeric(x)))))
for (j in 1:4) v(paste0("X", j - 1), X[, j]); v("Y", y)
ncomp <- 2
out <- function(tag, f) {
  v(paste0(tag, "_EXPLVAR"), explvar(f) / 100)
  # coef() is per scaled predictor when scale = TRUE: convert to the original scale
  sc <- if (is.numeric(f$scale)) f$scale else 1
  v(paste0(tag, "_COEF"), coef(f, ncomp = ncomp)[, 1, 1] / sc)
  v(paste0(tag, "_FITTED"), fitted(f)[, 1, ncomp])
}
out("CENTER", plsr(y ~ X, ncomp = ncomp, method = "simpls"))
out("SCALE", plsr(y ~ X, ncomp = ncomp, method = "simpls", scale = TRUE))
out("NOCENTER", plsr(y ~ X, ncomp = ncomp, method = "simpls", center = FALSE))
