#!/usr/bin/env Rscript
# Validation data for FittedOrdinal::predict_quantile_ci (#110 follow-up in
# #135) against rms::Quantile.orm(conf.int = ): pointwise confidence limits
# of a predicted quantile, interpolated and discrete.
#
#   Rscript tests/r_scripts/generate_quantile_orm_validation.R > tests/fixtures/quantile_orm_validation.rs

suppressMessages(library(rms))
emit <- function(name, v) {
  v <- as.numeric(v)
  cat("#[rustfmt::skip]\n")
  cat(sprintf("pub const %s: [f64; %d] = [\n    ", name, length(v)))
  cat(paste(sprintf("%.17e", v), collapse = ", "))
  cat("\n];\n")
}

set.seed(20261108)
n <- 200
x1 <- rnorm(n); x2 <- runif(n)
y <- round(3 + 1.2 * x1 - 0.8 * x2 + rnorm(n), 0)    # 10-ish distinct values
emit("X", t(cbind(x1, x2))); emit("Y", y)
nd <- cbind(x1 = c(-1, 0, 0.5, 1.5), x2 = c(0.2, 0.5, 0.9, 0.1))
emit("NEW", t(nd))
for (fam in c("logistic", "probit")) {
  f <- orm(y ~ x1 + x2, family = fam, x = TRUE, y = TRUE)
  qu <- Quantile(f)
  for (q in c(0.5, 0.9)) for (m in c("interpolated", "discrete")) {
    z <- qu(q, X = nd, conf.int = 0.95, method = m)
    tag <- sprintf("%s_Q%d_%s", toupper(fam), round(100 * q), toupper(substr(m, 1, 4)))
    emit(tag, c(as.numeric(z), attr(z, "limits")$lower, attr(z, "limits")$upper))
  }
}
