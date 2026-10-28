#!/usr/bin/env Rscript
# Validation data for `survival` (#113) against survival::survfit,
# summary(survfit)$table and survival::survdiff.
#
#   Rscript tests/r_scripts/generate_survival_validation.R > tests/fixtures/survival_validation.rs

suppressMessages(library(survival))

emit <- function(name, v) {
  cat("#[rustfmt::skip]\n")
  cat(sprintf("const %s: [f64; %d] = [\n    ", name, length(v)))
  v <- as.numeric(v)
  txt <- ifelse(is.na(v), "f64::NAN", ifelse(is.infinite(v), ifelse(v > 0, "f64::INFINITY", "f64::NEG_INFINITY"), sprintf("%.17e", v)))
  cat(paste(txt, collapse = ", "))
  cat("\n];\n")
}
emit_usize <- function(name, v) cat(sprintf("const %s: [usize; %d] = [%s];\n", name, length(v), paste(v, collapse = ", ")))

d <- lung[!is.na(lung$ph.ecog), ]
d$status <- d$status - 1
emit("TIME", d$time); emit("STATUS", d$status)
emit_usize("SEX", d$sex - 1); emit_usize("ECOG", d$ph.ecog)
set.seed(20261028)
w <- round(runif(nrow(d), 0.5, 2), 2)
emit("W", w)

curve <- function(id, f) {
  emit(sprintf("%s_TIME", id), f$time)
  emit(sprintf("%s_NRISK", id), f$n.risk)
  emit(sprintf("%s_NEVENT", id), f$n.event)
  emit(sprintf("%s_NCENSOR", id), f$n.censor)
  emit(sprintf("%s_SURV", id), f$surv)
  emit(sprintf("%s_SE", id), f$std.err)
  emit(sprintf("%s_LOWER", id), f$lower)
  emit(sprintf("%s_UPPER", id), f$upper)
  emit(sprintf("%s_CUMHAZ", id), f$cumhaz)
  emit(sprintf("%s_SECHAZ", id), f$std.chaz)
  if (!is.null(f$strata)) emit_usize(sprintf("%s_STRATA", id), f$strata)
}

f <- survfit(Surv(time, status) ~ 1, data = d)
curve("KM", f)
emit("KM_TABLE", summary(f)$table)
curve("KM_LOGLOG", survfit(Surv(time, status) ~ 1, data = d, conf.type = "log-log", conf.int = 0.9))
curve("KM_PLAIN", survfit(Surv(time, status) ~ 1, data = d, conf.type = "plain"))
curve("FH", survfit(Surv(time, status) ~ 1, data = d, ctype = 2))
fs <- survfit(Surv(time, status) ~ sex, data = d)
curve("KMS", fs)
emit("KMS_TABLE", t(summary(fs)$table))
emit("KMS_RMEAN_END", max(fs$time))
# robust = FALSE: the Greenwood variance (R defaults to a robust one for
# non-integer weights).
fw <- survfit(Surv(time, status) ~ 1, data = d, weights = w, robust = FALSE)
curve("KMW", fw)

# Left truncation (counting-process data).
set.seed(7)
n <- 150
entry <- round(runif(n, 0, 2), 2)
stop <- entry + round(rexp(n, 0.4), 2) + 0.01
ev <- rbinom(n, 1, 0.75)
emit("LT_ENTRY", entry); emit("LT_STOP", stop); emit("LT_EVENT", ev)
fl <- survfit(Surv(entry, stop, ev) ~ 1)
curve("LT", fl)

# Median exactly at 0.5 over an interval (midpoint rule).
tm <- c(1, 2, 3, 4, 5, 6); st <- c(1, 1, 1, 1, 1, 1)
emit("MID_TABLE", summary(survfit(Surv(tm, st) ~ 1))$table)

# Log-rank and G-rho tests.
sd_out <- function(id, s) {
  emit(sprintf("%s_OBS", id), if (is.matrix(s$obs)) rowSums(s$obs) else s$obs)
  emit(sprintf("%s_EXP", id), if (is.matrix(s$exp)) rowSums(s$exp) else s$exp)
  emit(sprintf("%s_VAR", id), s$var)
  cat(sprintf("const %s_CHISQ: f64 = %.17e;\n", id, s$chisq))
  cat(sprintf("const %s_P: f64 = %.17e;\n", id, pchisq(s$chisq, length(s$n) - 1, lower.tail = FALSE)))
}
sd_out("LR2", survdiff(Surv(time, status) ~ sex, data = d))
sd_out("LRK", survdiff(Surv(time, status) ~ ph.ecog, data = d))
sd_out("PETO", survdiff(Surv(time, status) ~ ph.ecog, data = d, rho = 1))
sd_out("STRAT", survdiff(Surv(time, status) ~ ph.ecog + strata(sex), data = d))
