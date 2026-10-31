# Reference values for tests/issue194_l2_logistic_se.rs (#194): L2-penalised
# logistic regression, objective loglik - 0.5 * lambda * ||beta||^2 (intercept
# unpenalised), covariance = inverse penalised information (rms default
# var.penalty = "simple").
suppressMessages(library(rms))
set.seed(194)
n <- 60
x1 <- round(rnorm(n), 3); x2 <- round(runif(n, -2, 2), 3)
y <- rbinom(n, 1, plogis(-0.3 + 1.1 * x1 - 0.6 * x2))
fmt <- function(x) paste(ifelse(x == round(x), sprintf("%.1f", x), sprintf("%.17g", x)), collapse = ", ")
v <- function(name, x) cat(sprintf("const %s: [f64; %d] = [%s];\n", name, length(x), fmt(unname(as.numeric(x)))))
v("X1", x1); v("X2", x2); v("Y", y)
for (lam in c(0.5, 5)) {
  tag <- sprintf("L%s", gsub("\\.", "", format(lam)))
  f <- lrm(y ~ x1 + x2, penalty = 1, penalty.matrix = diag(lam, 2), tol = 1e-14, eps = 1e-12)
  v(paste0(tag, "_COEF"), coef(f)); v(paste0(tag, "_SE"), sqrt(diag(vcov(f))))
}
