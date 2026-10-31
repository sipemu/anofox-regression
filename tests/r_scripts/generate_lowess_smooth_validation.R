# Reference values for tests/issue204_lowess_smooth.rs (#204): lowess_smooth
# is lowess(seq_along(y), y, f = span); lowess_smooth_weights smooths each
# column that way, clips at 0 and renormalises the rows.
set.seed(204)
n <- 30
y <- round(sin(seq_len(n) / 4) + rnorm(n, sd = 0.3), 4)
W <- matrix(round(runif(n * 3), 4), n, 3); W <- W / rowSums(W)
fmt <- function(x) paste(ifelse(x == round(x), sprintf("%.1f", x), sprintf("%.17g", x)), collapse = ", ")
v <- function(name, x) cat(sprintf("const %s: [f64; %d] = [%s];\n", name, length(x), fmt(unname(as.numeric(x)))))
v("Y", y); v("W", t(W))   # W row-major
for (f in c(0.3, 2/3)) {
  tag <- if (f < 0.5) "F03" else "F067"
  v(paste0(tag, "_SMOOTH"), lowess(seq_len(n), y, f = f)$y)
  S <- sapply(1:3, function(j) lowess(seq_len(n), W[, j], f = f)$y)
  S <- pmax(S, 0); S <- S / rowSums(S)
  v(paste0(tag, "_WEIGHTS"), t(S))
}
