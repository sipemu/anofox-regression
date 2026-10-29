#!/usr/bin/env Rscript
# Validation data for `solvers::continuation_ratio` (#110, RMS §13.4)
# against rms::cr.setup + lrm.
#
#   Rscript tests/r_scripts/generate_continuation_ratio_validation.R > tests/fixtures/continuation_ratio_validation.rs

suppressMessages(library(rms))
emit <- function(name, v) {
  v <- as.numeric(v)
  cat("#[rustfmt::skip]\n")
  cat(sprintf("pub const %s: [f64; %d] = [\n    ", name, length(v)))
  cat(paste(sprintf("%.17e", v), collapse = ", "))
  cat("\n];\n")
}

# cr.setup on a small response
small <- c(1, 3, 2, 4, 1, 4)
s <- cr.setup(small)
emit("SMALL_Y", small); emit("SMALL_CR_Y", s$y); emit("SMALL_COHORT", as.integer(s$cohort) - 1)
emit("SMALL_SUBS", s$subs - 1); emit("SMALL_REPS", s$reps)

# The 5-category data of generate_ordinal_validation.R
set.seed(20261030)
n <- 150
x1 <- rnorm(n); x2 <- rbinom(n, 1, 0.4); x3 <- runif(n)
lin <- 0.8 * x1 - 0.6 * x2 + 1.2 * x3
y_cont <- round(lin + rlogis(n), 6)
y_cat <- cut(lin + rlogis(n), c(-Inf, -1, 0, 0.8, 1.8, Inf), labels = FALSE)
X <- cbind(x1, x2, x3)
emit("X", t(X)); emit("Y_CAT", y_cat)
xnew <- rbind(c(0, 0, 0.5), c(1, 1, 0.2), c(-1.5, 0, 0.9))
emit("XNEW", t(xnew))

s <- cr.setup(y_cat)
d <- data.frame(y = s$y, cohort = s$cohort, x1 = x1[s$subs], x2 = x2[s$subs], x3 = x3[s$subs])
fits <- list(EQUAL = lrm(y ~ cohort + x1 + x2 + x3, data = d, tol = 1e-14, maxit = 100),
             X1 = lrm(y ~ cohort + x1 + x2 + x3 + cohort:x1, data = d, tol = 1e-14, maxit = 100))
for (id in names(fits)) {
  f <- fits[[id]]
  emit(sprintf("%s_COEF", id), coef(f)); emit(sprintf("%s_VCOV", id), vcov(f))
  emit(sprintf("%s_LL", id), -f$deviance[2] / 2)
  # P(Y = y_j | Y >= y_j, xnew), cohort fastest within each row
  nd <- data.frame(cohort = factor(rep(levels(d$cohort), 3), levels = levels(d$cohort)),
                   x1 = rep(xnew[, 1], each = 4), x2 = rep(xnew[, 2], each = 4), x3 = rep(xnew[, 3], each = 4))
  emit(sprintf("%s_COND", id), plogis(predict(f, nd)))
}
