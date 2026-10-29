#!/usr/bin/env Rscript
# Validation data for `survival::cox` (#114) against survival::coxph,
# survfit.coxph and rms::cph.
#
#   Rscript tests/r_scripts/generate_cox_validation.R > tests/fixtures/cox_validation.rs

suppressMessages({ library(survival); library(rms) })

emit <- function(name, v) {
  v <- as.numeric(v)
  txt <- ifelse(is.na(v), "f64::NAN", ifelse(is.infinite(v), ifelse(v > 0, "f64::INFINITY", "f64::NEG_INFINITY"), sprintf("%.17e", v)))
  cat("#[rustfmt::skip]\n")
  cat(sprintf("const %s: [f64; %d] = [\n    ", name, length(v)))
  cat(paste(txt, collapse = ", "))
  cat("\n];\n")
}
emit_usize <- function(name, v) cat(sprintf("const %s: [usize; %d] = [%s];\n", name, length(v), paste(v, collapse = ", ")))
scalar <- function(name, v) cat(sprintf("const %s: f64 = %.17e;\n", name, as.numeric(v)))

d <- na.omit(lung[, c("time", "status", "age", "sex", "ph.ecog", "wt.loss")])
d$status <- d$status - 1
emit("TIME", d$time); emit("STATUS", d$status)
emit("AGE", d$age); emit("SEX", d$sex); emit("ECOG", d$ph.ecog); emit("WTLOSS", d$wt.loss)
set.seed(20261028)
w <- round(runif(nrow(d), 0.5, 2), 2); off <- round(rnorm(nrow(d), 0, 0.2), 3)
id <- sample(1:40, nrow(d), replace = TRUE)
emit("W", w); emit("OFF", off); emit_usize("ID", id - 1)

ctl <- coxph.control(eps = 1e-12, iter.max = 50)
fit_out <- function(id, f) {
  emit(sprintf("%s_COEF", id), coef(f))
  emit(sprintf("%s_VAR", id), t(f$var))
  emit(sprintf("%s_LOGLIK", id), f$loglik)
  scalar(sprintf("%s_SCORE", id), f$score)
  scalar(sprintf("%s_WALD", id), f$wald.test)
}
f1 <- coxph(Surv(time, status) ~ age + sex + ph.ecog + wt.loss, data = d, control = ctl)
fit_out("EFRON", f1)
scalar("EFRON_CONCORDANCE", f1$concordance["concordance"])
emit("EFRON_CONC_COUNTS", f1$concordance[c("concordant", "discordant", "tied.x", "tied.y", "tied.xy")])
cp <- cph(Surv(time, status) ~ age + sex + ph.ecog + wt.loss, data = d, x = TRUE, y = TRUE, surv = TRUE)
emit("CPH_STATS", cp$stats[c("Model L.R.", "R2", "Dxy")])
fit_out("BRESLOW", coxph(Surv(time, status) ~ age + sex + ph.ecog + wt.loss, data = d, ties = "breslow", control = ctl))
fit_out("WEIGHTED", coxph(Surv(time, status) ~ age + sex + ph.ecog + wt.loss + offset(off), data = d, weights = w, control = ctl, robust = FALSE))
fit_out("STRATA", coxph(Surv(time, status) ~ age + ph.ecog + wt.loss + strata(sex), data = d, control = ctl))
fr <- coxph(Surv(time, status) ~ age + sex + ph.ecog + wt.loss, data = d, cluster = id, control = ctl)
emit("ROBUST_VAR", t(fr$var))
emit("SCORE_RESID", t(residuals(f1, type = "score")))
fb <- coxph(Surv(time, status) ~ age + sex + ph.ecog + wt.loss, data = d, ties = "breslow", cluster = id, control = ctl)
emit("ROBUST_BRESLOW_VAR", t(fb$var))
fwr <- coxph(Surv(time, status) ~ age + sex + ph.ecog + wt.loss, data = d, weights = w, cluster = id, control = ctl)
emit("ROBUST_WEIGHTED_VAR", t(fwr$var))

# Ridge.
fp <- coxph(Surv(time, status) ~ ridge(age, sex, ph.ecog, wt.loss, theta = 2, scale = FALSE), data = d, control = ctl)
emit("RIDGE_COEF", coef(fp)); emit("RIDGE_VAR", t(fp$var))

# Predicted survival for new covariates (Efron fit), and the baseline at 0.
nd <- data.frame(age = c(60, 70), sex = c(1, 2), ph.ecog = c(1, 2), wt.loss = c(5, 10))
sf <- survfit(f1, newdata = nd)
emit("PRED_TIME", sf$time)
emit("PRED_SURV", sf$surv)      # n_time x 2, column-major
emit("PRED_SE", sf$std.err)
emit("PRED_LOWER", sf$lower)
emit("PRED_UPPER", sf$upper)
emit("PRED_CUMHAZ", sf$cumhaz)
emit("PRED_NRISK", sf$n.risk)
sfb <- survfit(coxph(Surv(time, status) ~ age + sex + ph.ecog + wt.loss, data = d, ties = "breslow", control = ctl), newdata = nd[1, ])
emit("PREDB_SURV", sfb$surv); emit("PREDB_SE", sfb$std.err)

# Counting-process data with a time-dependent covariate (Stanford heart).
h <- heart
emit("H_START", h$start); emit("H_STOP", h$stop); emit("H_EVENT", h$event)
emit("H_AGE", h$age); emit("H_TRANSPLANT", as.numeric(h$transplant) - 1); emit("H_SURGERY", h$surgery)
fh <- coxph(Surv(start, stop, event) ~ age + transplant + surgery, data = h, control = ctl)
fit_out("HEART", fh)
fhb <- coxph(Surv(start, stop, event) ~ age + transplant + surgery, data = h, ties = "breslow", control = ctl)
fit_out("HEARTB", fhb)
emit_usize("H_ID", h$id - 1)
fhr <- coxph(Surv(start, stop, event) ~ age + transplant + surgery, data = h, cluster = id, control = ctl)
emit("HEART_ROBUST_VAR", t(fhr$var))
