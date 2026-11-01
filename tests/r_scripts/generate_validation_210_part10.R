# Reference values for tests/r_validation_210_part10.rs (#210): robcov(orm)
# and the LR test for a multi-category ordinal model, and pentrace on an
# ordinal (4-level) response.
suppressMessages(library(rms))
fmt <- function(v) { s <- sprintf("%.17g", v); paste(ifelse(grepl("[.e]|Inf|NaN", s), s, paste0(s, ".0")), collapse = ", ") }
vec <- function(tag, v) cat(sprintf("const %s: [f64; %d] = [%s];\n", tag, length(v), fmt(unname(as.numeric(v)))))
set.seed(2109); n <- 90
x1 <- rnorm(n); x2 <- runif(n); cl <- sample(1:18, n, TRUE)
y <- as.integer(cut(0.8 * x1 - x2 + rlogis(n), c(-Inf, -1, 0, 1, Inf)))
vec("X1", x1); vec("X2", x2); vec("Y", y)
cat(sprintf("const CLUSTER: [usize; %d] = [%s];\n", n, paste(cl, collapse = ", ")))
for (fam in c("logistic", "probit")) {
  f <- orm(y ~ x1 + x2, family = fam, x = TRUE, y = TRUE, eps = 1e-12)
  tag <- toupper(fam)
  vec(paste0(tag, "_ROBCOV"), vcov(robcov(f, cluster = cl), intercepts = "all"))
  vec(paste0(tag, "_LR"), f$stats[c("Model L.R.", "d.f.", "P")])
}
# pentrace on a 4-level response with the design of generate_pentrace_validation.R
set.seed(20261031)
n2 <- 300
z1 <- rnorm(n2); z2 <- rnorm(n2); g <- factor(sample(c("a", "b", "c"), n2, replace = TRUE))
lp <- -0.2 + 0.7 * z1 - 0.3 * z1^2 + 0.5 * z2 + 0.3 * z1 * z2 + c(0, 0.4, -0.3)[g]
yo <- as.integer(cut(lp + rlogis(n2), c(-Inf, -1, 0, 1, Inf)))
d <- data.frame(z1, z2, g, yo)
fo <- lrm(yo ~ rcs(z1, 4) * z2 + g, data = d, x = TRUE, y = TRUE)
vec("PT_X", t(fo$x)); vec("PT_Y", yo)
pt <- pentrace(fo, list(simple = c(0, 2), nonlinear = c(0, 4, 16)))
vec("PT_STATS", t(as.matrix(pt$results.all[, c("simple", "nonlinear", "df", "aic", "bic", "aic.c")])))
vec("PT_BEST", unlist(pt$penalty))
