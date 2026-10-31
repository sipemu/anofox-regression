# Reference values for tests/issue199_quantile_loss.rs (#199): weighted check
# loss and Koenker-Machado R1 = 1 - rho(rq) / rho(rq(y ~ 1)).
suppressMessages(library(quantreg))
data(engel)
x <- engel$income; y <- engel$foodexp; n <- length(y)
set.seed(199)
w <- round(runif(n, 0.2, 3), 3)
fmt <- function(x) paste(ifelse(x == round(x), sprintf("%.1f", x), sprintf("%.17g", x)), collapse = ", ")
v <- function(name, x) cat(sprintf("const %s: [f64; %d] = [%s];\n", name, length(x), fmt(unname(x))))
s1 <- function(name, x) cat(sprintf("const %s: f64 = %.17g;\n", name, x))
v("X", x); v("Y", y); v("W", w)
rho <- function(r, tau, w = 1) sum(w * r * (tau - (r < 0)))
for (tau in c(0.05, 0.3, 0.5, 0.9)) {
  tag <- sprintf("T%02d", round(100 * tau))
  f <- rq(y ~ x, tau = tau); f0 <- rq(y ~ 1, tau = tau)
  s1(paste0(tag, "_LOSS"), rho(resid(f), tau)); s1(paste0(tag, "_R1"), 1 - rho(resid(f), tau) / rho(resid(f0), tau))
  fw <- rq(y ~ x, tau = tau, weights = w); f0w <- rq(y ~ 1, tau = tau, weights = w)
  v(paste0(tag, "_W_COEF"), coef(fw))
  s1(paste0(tag, "_W_LOSS"), rho(resid(fw), tau, w)); s1(paste0(tag, "_W_R1"), 1 - rho(resid(fw), tau, w) / rho(resid(f0w), tau, w))
}
