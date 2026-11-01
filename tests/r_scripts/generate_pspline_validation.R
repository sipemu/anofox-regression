# Reference values for tests/r_validation_pspline.rs (#210):
# mgcv::gam(y ~ s(x, bs = "ps", k, m = c(2, m2)), method = "GCV.Cp").
suppressMessages(library(mgcv))
fmt <- function(v) { s <- sprintf("%.17g", v); paste(ifelse(grepl("[.e]|Inf|NaN", s), s, paste0(s, ".0")), collapse = ", ") }
vec <- function(tag, v) cat(sprintf("const %s: [f64; %d] = [%s];\n", tag, length(v), fmt(unname(as.numeric(v)))))
set.seed(2107)
n <- 80
x <- sort(runif(n, 0, 6))
y <- sin(x) + 0.3 * x + rnorm(n, sd = 0.25)
vec("X", x); vec("Y", y)
xn <- c(-0.5, 0, x[1], 1.7, 3.3, 5.2, x[n], 6.4)
vec("X_NEW", xn)
ctl <- gam.control(epsilon = 1e-12, newton = list(conv.tol = 1e-12))
# edf, sig2, GCV score, lambda on the scale of the raw difference penalty,
# fitted values, then fit and SE (Bayesian Vp) at the new points
emit <- function(tag, b) {
  lam <- b$full.sp / b$smooth[[1]]$S.scale
  vec(paste0(tag, "_STATS"), c(sum(b$edf), b$sig2, b$gcv.ubre, lam))
  vec(paste0(tag, "_FITTED"), fitted(b))
  p <- predict(b, data.frame(x = xn), se.fit = TRUE)
  vec(paste0(tag, "_PRED"), c(p$fit, p$se.fit))
}
emit("GCV_K12", gam(y ~ s(x, bs = "ps", k = 12, m = c(2, 2)), method = "GCV.Cp", control = ctl))
emit("GCV_K20_M3", gam(y ~ s(x, bs = "ps", k = 20, m = c(2, 3)), method = "GCV.Cp", control = ctl))
# fixed smoothing parameter: lambda_raw = 2 on the raw D'D penalty
b0 <- gam(y ~ s(x, bs = "ps", k = 12, m = c(2, 2)), method = "GCV.Cp")
emit("FIXED_K12", gam(y ~ s(x, bs = "ps", k = 12, m = c(2, 2)), sp = 2 * b0$smooth[[1]]$S.scale))
