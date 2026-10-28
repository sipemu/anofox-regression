#!/usr/bin/env Rscript
# Validation data for `inference::anova` / `wald_test` (#100) against
# rms::anova.rms and nested-model F tests.
#
#   Rscript tests/r_scripts/generate_anova_validation.R > tests/fixtures/anova_validation.rs

suppressMessages(library(rms))

emit <- function(name, v) {
  cat("#[rustfmt::skip]\n")
  cat(sprintf("const %s: [f64; %d] = [\n    ", name, length(v)))
  cat(paste(sprintf("%.17e", as.numeric(v)), collapse = ", "))
  cat("\n];\n")
}
emit_usize <- function(name, v) {
  cat(sprintf("const %s: [usize; %d] = [%s];\n", name, length(v), paste(v, collapse = ", ")))
}
emit_str <- function(name, v) {
  cat(sprintf("const %s: [&str; %d] = [\n", name, length(v)))
  cat(paste(sprintf("    %s,", encodeString(v, quote = "\"")), collapse = "\n"))
  cat("\n];\n")
}

set.seed(20261028)
n <- 300
x1 <- rnorm(n); x2 <- runif(n); x3 <- rnorm(n); x4 <- rnorm(n)
y <- x1 + 0.3 * x1^2 + sin(3 * x2) * x1 + x3 * x2 + rnorm(n)
yb <- as.integer(y > median(y))
emit("X1", x1); emit("X2", x2); emit("X3", x3); emit("X4", x4)
emit("Y", y); emit("YB", yb)
emit("KNOTS_X1", rcspline.eval(x1, nk = 4, knots.only = TRUE))
emit("KNOTS_X2", rcspline.eval(x2, nk = 3, knots.only = TRUE))

model <- function(id, f, stats_ss = NULL) {
  # Slopes only (rms drops the intercept(s) in anova).
  b <- coef(f)
  ni <- num.intercepts(f)
  b <- b[-(1:ni)]
  v <- vcov(f, regcoef.only = TRUE, intercepts = "none")
  emit(sprintf("BETA_%s", id), b)
  emit(sprintf("VCOV_%s", id), t(v))
  a <- anova(f, test = "Chisq", ss = FALSE)
  emit_str(sprintf("LABELS_%s", id), rownames(a))
  emit(sprintf("CHISQ_%s", id), a[, "Chi-Square"])
  emit(sprintf("DF_%s", id), a[, "d.f."])
  emit(sprintf("P_%s", id), a[, "P"])
  w <- attr(a, "which")
  emit_usize(sprintf("WHICH_LEN_%s", id), sapply(w, length))
  emit_usize(sprintf("WHICH_%s", id), unlist(w) - 1)
}

# M1: OLS, rcs x linear interaction. Also F test with partial SS.
f1 <- ols(y ~ rcs(x1, 4) * x3 + rcs(x2, 3) + x4)
model("M1", f1)
a <- anova(f1)   # F, SS, ERROR row
emit("F_M1", a[, "F"][-nrow(a)])
emit("FP_M1", a[, "P"][-nrow(a)])
emit("SS_M1", a[, "Partial SS"])
cat(sprintf("const SIGMA2_M1: f64 = %.17e;\n", f1$stats["Sigma"]^2))
cat(sprintf("const DFE_M1: f64 = %.17e;\n", f1$df.residual))

# M2: the same design, binary logistic.
f2 <- lrm(yb ~ rcs(x1, 4) * x3 + rcs(x2, 3) + x4)
model("M2", f2)

# M3: full tensor rcs x rcs, a second interaction of x1, and a main effect.
f3 <- ols(y ~ rcs(x1, 4) * rcs(x2, 3) + rcs(x1, 4) * x3 + x4)
model("M3", f3)

# M4: restricted interaction %ia%.
f4 <- ols(y ~ rcs(x1, 4) + rcs(x2, 3) + rcs(x1, 4) %ia% rcs(x2, 3))
model("M4", f4)

# General linear hypotheses on M1 (lm with the same design).
X <- f1$x
if (is.null(X)) X <- model.matrix(f1)[, -1]
fl <- lm(y ~ rcs(x1, 4) * x3 + rcs(x2, 3) + x4)
bl <- coef(fl); vl <- vcov(fl)
# H0: the three x1 columns are zero -> nested F test.
red <- lm(y ~ x3 + rcs(x2, 3) + x4 + rcs(x1, 4):x3)
nest <- anova(red, fl)
cat(sprintf("const NESTED_F: f64 = %.17e;\n", nest$F[2]))
cat(sprintf("const NESTED_P: f64 = %.17e;\n", nest[["Pr(>F)"]][2]))
# H0: C beta = r with two non-trivial rows (intercept included).
C <- matrix(0, 2, length(bl)); C[1, 2] <- 1; C[1, 5] <- -1; C[2, 1] <- 1; C[2, 7] <- 2
r <- c(0.5, 1)
d <- C %*% bl - r
W <- as.numeric(t(d) %*% solve(C %*% vl %*% t(C), d))
emit("GLH_C", t(C)); emit("GLH_R", r)
emit("LM_BETA", bl); emit("LM_VCOV", t(vl))
cat(sprintf("const GLH_CHISQ: f64 = %.17e;\n", W))
cat(sprintf("const GLH_F_P: f64 = %.17e;\n", pf(W / 2, 2, fl$df.residual, lower.tail = FALSE)))
cat(sprintf("const GLH_CHISQ_P: f64 = %.17e;\n", pchisq(W, 2, lower.tail = FALSE)))
cat(sprintf("const LM_DFE: f64 = %.17e;\n", fl$df.residual))
