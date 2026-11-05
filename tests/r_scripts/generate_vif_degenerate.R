# Reference values for tests/issue_vif_degenerate.rs: car::vif on the designs
# without their aliased columns (car::vif stops on aliased coefficients).
# The mtcars constants (WT, ..., VIF_PLAIN, GVIF, GVIF_ADJ) are those of
# generate_vif_validation.R.
suppressMessages(library(car))
fmt <- function(x) paste(sprintf("%.17g", x), collapse = ", ")
v <- function(name, x) cat(sprintf("const %s: [f64; %d] = [%s];\n", name, length(x), fmt(x)))
x <- data.frame(a = c(1, 2, 4, 3, 7), b = c(2, 1, 5, 2, 3), c = c(0, 3, 1, 4, 2))
x$y <- c(1, 3, 2, 5, 4)
v("VIF_SMALL", vif(lm(y ~ a + b + c, data = x)))
# With n <= p every auxiliary regression is exact: R^2 = 1, VIF = Inf.
z <- data.frame(a = c(1, 2, 4, 3), b = c(2, 1, 5, 2), c = c(0, 3, 1, 4), d = c(5, 1, 2, 2), e = c(1, 1, 3, 7))
cat("R^2(a ~ b + c + d + e), n = 4:", summary(lm(a ~ b + c + d + e, data = z))$r.squared, "\n")
