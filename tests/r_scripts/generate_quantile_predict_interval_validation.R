# Reference values for tests/r_validation_quantile_predict_interval.rs (#210):
# predict.rq(interval = "confidence") with se = "iid" / "nid".
suppressMessages(library(quantreg))
set.seed(210)
n <- 60
x1 <- runif(n, 0, 4); x2 <- rnorm(n)
y <- 1 + 0.8 * x1 - 0.5 * x2 + (0.5 + 0.3 * x1) * rnorm(n)
nd <- data.frame(x1 = c(0.5, 2, 3.7), x2 = c(-1, 0.2, 1.4))
fmt <- function(v) { s <- sprintf("%.17g", v); paste(ifelse(grepl("[.e]", s), s, paste0(s, ".0")), collapse = ", ") }
vec <- function(tag, v) cat(sprintf("const %s: [f64; %d] = [%s];\n", tag, length(v), fmt(v)))
vec("X1", x1); vec("X2", x2); vec("Y", y)
vec("NEW_X1", nd$x1); vec("NEW_X2", nd$x2)
for (tau in c(0.5, 0.25)) for (se in c("iid", "nid")) for (level in c(0.95, 0.8)) {
  f <- rq(y ~ x1 + x2, tau = tau)
  p <- predict(f, nd, interval = "confidence", level = level, se = se)
  tag <- sprintf("%s_TAU%02d_L%02d", toupper(se), round(100 * tau), round(100 * level))
  vec(tag, c(p[, "fit"], p[, "lower"], p[, "higher"]))
}
f0 <- rq(y ~ x1 + x2 - 1, tau = 0.5)
p0 <- predict(f0, nd, interval = "confidence", level = 0.9, se = "nid")
vec("NID_NOINT_TAU50_L90", c(p0[, "fit"], p0[, "lower"], p0[, "higher"]))
