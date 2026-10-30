#!/usr/bin/env Rscript
# Validation data for `validation::calibrate_cph` (#115, RMS §20.10) against
# rms::calibrate.cph with cmethod = "KM": predicted survival at u grouped by
# quantiles, Kaplan-Meier at u per group, bootstrap-corrected. The resamples
# are regenerated with predab.resample's own sample() calls.
#
#   Rscript tests/r_scripts/generate_calibrate_cph_validation.R > tests/fixtures/calibrate_cph_validation.rs

suppressMessages(library(rms))
emit <- function(name, v) {
  v <- as.numeric(v)
  txt <- ifelse(is.na(v), "f64::NAN", sprintf("%.17e", v))
  cat("#[rustfmt::skip]\n")
  cat(sprintf("pub const %s: [f64; %d] = [\n    ", name, length(v)))
  cat(paste(txt, collapse = ", "))
  cat("\n];\n")
}

set.seed(20261106)
n <- 300
x1 <- rnorm(n); x2 <- rbinom(n, 1, 0.4); x3 <- runif(n)
h <- 0.1 * exp(0.7 * x1 - 0.5 * x2 + 0.8 * x3)
t <- rexp(n, h); cens <- runif(n, 2, 15)
time <- round(pmin(t, cens), 2); event <- as.numeric(t <= cens)
emit("X", t(cbind(x1, x2, x3))); emit("TIME", time); emit("EVENT", event)

u <- 5; B <- 25
f <- cph(Surv(time, event) ~ x1 + x2 + x3, x = TRUE, y = TRUE, surv = TRUE, time.inc = u)
set.seed(1); idx <- replicate(B, sample(n, replace = TRUE)); emit("BOOT_IDX", c(idx) - 1)
set.seed(1); invisible(capture.output(cal <- calibrate(f, cmethod = "KM", u = u, m = 60, B = B)))
emit("PREDICTED", attr(cal, "predicted"))
emit("CAL", t(unclass(cal)[, c("index.orig", "training", "test", "mean.optimism", "mean.corrected",
                              "n", "mean.predicted", "KM", "KM.corrected", "std.err")]))
emit("U_M", c(u, 60))
