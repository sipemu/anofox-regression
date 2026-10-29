#!/usr/bin/env Rscript
# Validation data for `solvers::ordinal` (#110) against rms::orm / lrm and
# MASS::polr.
#
#   Rscript tests/r_scripts/generate_ordinal_validation.R > tests/fixtures/ordinal_validation.rs

suppressMessages({ library(rms); library(MASS) })
emit <- function(name, v) {
  v <- as.numeric(v)
  cat("#[rustfmt::skip]\n")
  cat(sprintf("pub const %s: [f64; %d] = [\n    ", name, length(v)))
  cat(paste(sprintf("%.17e", v), collapse = ", "))
  cat("\n];\n")
}
tight <- list(eps = 1e-10, gradtol = 1e-9, maxit = 100)

set.seed(20261030)
n <- 150
x1 <- rnorm(n); x2 <- rbinom(n, 1, 0.4); x3 <- runif(n)
lin <- 0.8 * x1 - 0.6 * x2 + 1.2 * x3
y_cont <- round(lin + rlogis(n), 6)                       # continuous: one intercept per value
y_cat <- cut(lin + rlogis(n), c(-Inf, -1, 0, 0.8, 1.8, Inf), labels = FALSE)  # 5 categories
w <- sample(1:3, n, replace = TRUE)
off <- 0.3 * sin(seq_len(n))
X <- cbind(x1, x2, x3)
emit("X", t(X)); emit("Y_CONT", y_cont); emit("Y_CAT", y_cat); emit("W", w); emit("OFFSET", off)
xnew <- rbind(c(0, 0, 0.5), c(1, 1, 0.2), c(-1.5, 0, 0.9))
emit("XNEW", t(xnew))

fit_out <- function(id, f, full_vcov = TRUE) {
  emit(sprintf("%s_COEF", id), coef(f))
  # orm deviance: null, [null with the offset,] model
  emit(sprintf("%s_LL", id), -f$deviance / 2)
  if (full_vcov) emit(sprintf("%s_VCOV", id), as.matrix(vcov(f, intercepts = "all")))
  emit(sprintf("%s_VCOV_SLOPES", id), as.matrix(vcov(f, intercepts = "none")))
}

# orm on the continuous response, every family
for (fam in c("logistic", "probit", "loglog", "cloglog", "cauchit")) {
  f <- do.call(orm, c(list(y_cont ~ x1 + x2 + x3, family = fam, x = TRUE), tight))
  id <- toupper(paste0("ORM_", fam))
  fit_out(id, f, full_vcov = FALSE)
  M <- Mean(f); Q <- Quantile(f)
  # lp from X %*% slopes (an explicit lp would be taken relative to the
  # reference intercept)
  mm <- M(X = xnew, conf.int = 0.95)
  emit(sprintf("%s_MEAN", id), mm)
  emit(sprintf("%s_MEAN_SE", id), (attr(mm, "limits")$upper - mm) / qnorm(0.975))
  emit(sprintf("%s_Q_INTERP", id), c(Q(0.25, X = xnew), Q(0.5, X = xnew), Q(0.9, X = xnew)))
  emit(sprintf("%s_Q_DISCRETE", id), c(Q(0.25, X = xnew, method = "discrete"), Q(0.5, X = xnew, method = "discrete")))
}

# lrm on 5 categories (full vcov), and polr for the sign convention
f <- do.call(orm, c(list(y_cat ~ x1 + x2 + x3), tight))
fit_out("CAT", f)
pl <- polr(factor(y_cat) ~ x1 + x2 + x3, method = "logistic", control = list(reltol = 1e-14))
emit("POLR_ZETA", pl$zeta); emit("POLR_BETA", coef(pl))
f <- do.call(orm, c(list(y_cat ~ x1 + x2 + x3, family = "probit"), tight))
fit_out("CAT_PROBIT", f)

# weights and offset
f <- do.call(orm, c(list(y_cat ~ x1 + x2 + x3, weights = w), tight))
fit_out("CAT_W", f)
f <- do.call(orm, c(list(y_cat ~ x1 + x2 + x3 + offset(off)), tight))
fit_out("CAT_OFF", f)
