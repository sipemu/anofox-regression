# Reference values for tests/issue207_bkw.rs (#207): Belsley-Kuh-Welsch
# condition indices / variance-decomposition proportions (olsrr 0.7.0) and
# kappa(exact = TRUE).
suppressMessages(library(olsrr))
fmt <- function(x) paste(ifelse(x == round(x), sprintf("%.1f", x), sprintf("%.17g", x)), collapse = ", ")
v <- function(name, x) cat(sprintf("const %s: [f64; %d] = [%s];\n", name, length(x), fmt(unname(as.numeric(x)))))
s1 <- function(name, x) cat(sprintf("const %s: f64 = %.17g;\n", name, x))
d <- mtcars
v("WT", d$wt); v("HP", d$hp); v("DISP", d$disp); v("DRAT", d$drat)
f <- lm(mpg ~ wt + hp + disp + drat, data = d)
e <- ols_eigen_cindex(f)
v("EIGEN", e[, 1]); v("CINDEX", e[, 2])
# proportions: rows = components, columns = intercept, wt, hp, disp, drat (row-major)
v("PROP", t(as.matrix(e[, 3:7])))
s1("KAPPA", kappa(model.matrix(f), exact = TRUE))
s1("KAPPA_NOINT", kappa(model.matrix(f)[, -1], exact = TRUE))
