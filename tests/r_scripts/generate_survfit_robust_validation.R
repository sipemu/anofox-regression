#!/usr/bin/env Rscript
# Validation data for the robust (infinitesimal-jackknife) variance of
# `survival::survfit` (#113): the default for non-integer case weights,
# with clusters (id), the Fleming-Harrington hazard and left truncation.
#
#   Rscript tests/r_scripts/generate_survfit_robust_validation.R > tests/fixtures/survfit_robust_validation.rs

suppressMessages(library(survival))
emit <- function(name, v) {
  v <- as.numeric(v)
  txt <- ifelse(is.na(v), "f64::NAN", ifelse(is.infinite(v), ifelse(v > 0, "f64::INFINITY", "f64::NEG_INFINITY"), sprintf("%.17e", v)))
  cat("#[rustfmt::skip]\n")
  cat(sprintf("pub const %s: [f64; %d] = [\n    ", name, length(v)))
  cat(paste(txt, collapse = ", "))
  cat("\n];\n")
}

set.seed(20261107)
n <- 60
time <- round(rexp(n, 0.3), 1) + 0.1; event <- rbinom(n, 1, 0.7)
w <- round(runif(n, 0.5, 2), 3); id <- sample(1:20, n, TRUE)
entry <- pmax(0, round(time - rexp(n, 0.5) - 0.2, 1))
emit("TIME", time); emit("EVENT", event); emit("W", w); emit("ID", id); emit("ENTRY", entry)

out <- function(tag, f) {
  emit(paste0(tag, "_SURV"), f$surv); emit(paste0(tag, "_SE"), f$std.err)
  emit(paste0(tag, "_CHAZ"), f$cumhaz); emit(paste0(tag, "_SECHAZ"), f$std.chaz)
  emit(paste0(tag, "_LOWER"), f$lower); emit(paste0(tag, "_UPPER"), f$upper)
}
out("W", survfit(Surv(time, event) ~ 1, weights = w))                       # robust by default
out("ID", survfit(Surv(time, event) ~ 1, weights = w, id = id))              # clustered
out("FH", survfit(Surv(time, event) ~ 1, weights = w, ctype = 2))
out("LT", survfit(Surv(entry, time, event) ~ 1, weights = w))               # left truncated
out("UNW", survfit(Surv(time, event) ~ 1, robust = TRUE))
out("GW", survfit(Surv(time, event) ~ 1, weights = w, robust = FALSE))      # Greenwood
