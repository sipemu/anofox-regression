# Reference values for tests/r_validation_cv_mae.rs (#210):
# cv.glmnet(type.measure = "mae") on fixed folds.
suppressMessages(library(glmnet))
fmt <- function(v) { s <- sprintf("%.17g", v); paste(ifelse(grepl("[.e]|Inf|NaN", s), s, paste0(s, ".0")), collapse = ", ") }
vec <- function(tag, v) cat(sprintf("const %s: [f64; %d] = [%s];\n", tag, length(v), fmt(unname(as.numeric(v)))))
set.seed(2108)
n <- 60; p <- 4
X <- matrix(rnorm(n * p), n, p)
y <- drop(X %*% c(1.2, 0, -0.8, 0.3)) + rt(n, df = 3)
foldid <- sample(rep(1:5, length.out = n))
vec("X_FLAT", t(X)); vec("Y", y)
cat(sprintf("const FOLDID: [usize; %d] = [%s];\n", n, paste(foldid - 1, collapse = ", ")))
ctl <- list(thresh = 1e-20, maxit = 1e7)
for (a in c(1, 0.5)) {
  lam <- glmnet(X, y, alpha = a, standardize = FALSE, nlambda = 20)$lambda
  cv <- cv.glmnet(X, y, alpha = a, lambda = lam, foldid = foldid, type.measure = "mae",
                  standardize = FALSE, control = ctl)
  tag <- if (a == 1) "LASSO" else "ENET"
  vec(paste0(tag, "_LAMBDA"), lam); vec(paste0(tag, "_CVM"), cv$cvm); vec(paste0(tag, "_CVSD"), cv$cvsd)
  vec(paste0(tag, "_LMIN_L1SE"), c(cv$lambda.min, cv$lambda.1se))
}
