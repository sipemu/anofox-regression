#!/usr/bin/env Rscript
# Validation data for `validation` (#103) against rms::validate and
# rms::calibrate (lrm and ols, boot / crossvalidation / .632). The resample
# indices are regenerated with the same seeds and the same sample() calls
# predab.resample makes, so the Rust side replays R's resamples exactly.
#
#   Rscript tests/r_scripts/generate_validate_validation.R > tests/fixtures/validate_validation.rs

suppressMessages(library(rms))

emit <- function(name, v) {
  v <- as.numeric(v)
  txt <- ifelse(is.na(v), "f64::NAN", sprintf("%.17e", v))
  cat("#[rustfmt::skip]\n")
  cat(sprintf("const %s: [f64; %d] = [\n    ", name, length(v)))
  cat(paste(txt, collapse = ", "))
  cat("\n];\n")
}
emit_usize <- function(name, v) cat(sprintf("const %s: [usize; %d] = [%s];\n", name, length(v), paste(v, collapse = ", ")))

set.seed(20261028)
n <- 150
x1 <- rnorm(n); x2 <- rnorm(n); x3 <- runif(n)
yb <- rbinom(n, 1, plogis(-0.4 + 0.9 * x1 - 0.6 * x2 + 0.5 * x3))
yc <- 1 + 0.8 * x1 - 0.5 * x2 + 0.3 * x3 + rnorm(n)
emit("X1", x1); emit("X2", x2); emit("X3", x3); emit("YB", yb); emit("YC", yc)

B <- 30
fl <- lrm(yb ~ x1 + x2 + x3, x = TRUE, y = TRUE)
fo <- ols(yc ~ x1 + x2 + x3, x = TRUE, y = TRUE)

tab <- function(id, v) {
  emit(sprintf("%s", id), t(unclass(v)[, c("index.orig", "training", "test", "optimism", "index.corrected", "n")]))
}

# Bootstrap.
set.seed(1); boot_idx <- replicate(B, sample(n, replace = TRUE))
emit_usize("BOOT_IDX", c(boot_idx) - 1)
set.seed(1); tab("LRM_BOOT", validate(fl, B = B))
set.seed(1); tab("OLS_BOOT", validate(fo, B = B))
# Crossvalidation (10-fold).
set.seed(2); cv_perm <- sample(n, replace = FALSE)
emit_usize("CV_PERM", cv_perm - 1)
set.seed(2); tab("LRM_CV", validate(fl, method = "crossvalidation", B = 10))
set.seed(2); tab("OLS_CV", validate(fo, method = "crossvalidation", B = 10))
# .632.
set.seed(3); s632 <- replicate(B, sample(n, replace = TRUE))
emit_usize("P632_IDX", c(s632) - 1)
set.seed(3); tab("LRM_632", validate(fl, method = ".632", B = B))

# calibrate (bootstrap) for lrm and ols.
cal_out <- function(id, cal, fit) {
  emit(sprintf("%s_PREDY", id), cal[, "predy"])
  emit(sprintf("%s_ORIG", id), cal[, "calibrated.orig"])
  emit(sprintf("%s_CORR", id), cal[, "calibrated.corrected"])
  emit(sprintf("%s_OPT", id), cal[, "optimism"])
  emit(sprintf("%s_N", id), cal[, "n"])
  predicted <- attr(cal, "predicted")
  s <- !is.na(cal[, "predy"] + cal[, "calibrated.corrected"])
  err <- predicted - approx(cal[s, "predy"], cal[s, "calibrated.corrected"], xout = predicted, ties = mean)$y
  emit(sprintf("%s_ERR", id), c(mean(abs(err), na.rm = TRUE), mean(err^2, na.rm = TRUE), quantile(abs(err), 0.9, na.rm = TRUE)))
}
set.seed(1); cal_out("CAL_LRM", calibrate(fl, B = B), fl)
set.seed(1); cal_out("CAL_OLS", calibrate(fo, B = B), fo)
