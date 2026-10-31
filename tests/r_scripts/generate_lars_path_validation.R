# Reference values for tests/r_validation_lars_path.rs (#210): LAR / lasso
# paths of lars::lars (normalize = FALSE, intercept = TRUE).
suppressMessages(library(lars))
fmt <- function(x) paste(ifelse(x == round(x), sprintf("%.1f", x), sprintf("%.17g", x)), collapse = ", ")
v <- function(name, x) cat(sprintf("const %s: [f64; %d] = [%s];\n", name, length(x), fmt(unname(as.numeric(x)))))
set.seed(2103)
n <- 40; p <- 4
X <- matrix(round(rnorm(n * p), 3), n, p); X[, 3] <- round(X[, 3] + 0.6 * X[, 1], 3)
y <- round(1 + 2 * X[, 1] - 1.5 * X[, 2] + 0.5 * X[, 4] + rnorm(n, sd = 0.5), 4)
v("X", t(X)); v("Y", y)
for (ty in c("lar", "lasso")) {
  f <- lars(X, y, type = ty, normalize = FALSE, intercept = TRUE)
  v(paste0(toupper(ty), "_LAMBDA"), f$lambda)   # max |correlation| at each step
  v(paste0(toupper(ty), "_BETA"), t(f$beta))    # one row per step, step 0 = zeros
}
